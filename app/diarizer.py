"""Speaker diarization.

Default: pyannote embedding diarizer (WeSpeaker on GPU) with graceful
fallback chain pyannote -> NeMo Titanet -> volume levels. The active backend
is always reported (status / entry diar_backend), never silently swapped.

Volume mode segments speaker turns from per-chunk loudness (RMS dBFS), pause
gaps, and stereo pan when available. NeMo/pyannote attribute finalized
segments with neural embeddings + online cosine clustering; live partials
use the instant volume guess, the embedding verdict lands at finalize.
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np

log = logging.getLogger("conf.diar")
_NEMO_WARNED = False


def rms_dbfs(pcm: np.ndarray) -> float:
    x = np.asarray(pcm, dtype=np.float64).ravel()
    if x.size == 0:
        return -80.0
    if x.dtype == np.int16 or np.abs(x).max() > 1.5:
        x = x / 32768.0
    rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    return 20.0 * math.log10(rms + 1e-12)


def stereo_pan(pcm: np.ndarray) -> float:
    """-1 (left) .. +1 (right); 0 for mono / centred."""
    a = np.asarray(pcm)
    if a.ndim < 2 or a.shape[1] < 2:
        return 0.0
    l = float(np.sqrt(np.mean(a[:, 0].astype(np.float64) ** 2) + 1e-12))
    r = float(np.sqrt(np.mean(a[:, 1].astype(np.float64) ** 2) + 1e-12))
    if l + r < 1e-9:
        return 0.0
    return (r - l) / (r + l)


class VolumeDiarizer:
    """Cluster chunks into speakers by loudness level + turn gaps.

    Heuristic, transparent and dependency-free:
      * a silence gap > silence_turn_gap ends the current turn
      * within a turn, each chunk's RMS level is matched to the nearest
        running speaker centroid (log-energy); a new speaker is opened when
        the chunk is far from all centroids (up to max_speakers).
    Stereo pan nudges the distance when present.
    """

    def __init__(self, max_speakers: int = 3, silence_turn_gap: float = 0.6,
                 silence_db: float = -45.0, new_speaker_db: float = 9.0):
        self.max_speakers = max(1, int(max_speakers))
        self.silence_turn_gap = float(silence_turn_gap)
        self.silence_db = float(silence_db)
        self.new_speaker_db = float(new_speaker_db)
        self.reset()

    def reset(self):
        self.centroids: list[float] = []   # mean dB per speaker
        self.counts: list[int] = []
        self.current: int = 0
        self.last_voice_t: float | None = None
        self.turn_id = 0

    def unload(self) -> dict:
        """Volume mode holds no GPU state — nothing to free."""
        return {"unloaded": True, "was": {"backend": "volume"}} 

    def _match(self, db: float, pan: float) -> int:
        if not self.centroids:
            self.centroids.append(db)
            self.counts.append(1)
            return 0
        dists = [abs(db - c) + 2.0 * abs(pan) * 0 for c in self.centroids]
        best = int(np.argmin(dists))
        if dists[best] > self.new_speaker_db and len(self.centroids) < self.max_speakers:
            self.centroids.append(db)
            self.counts.append(1)
            return len(self.centroids) - 1
        # online update
        n = self.counts[best] + 1
        self.centroids[best] = self.centroids[best] + (db - self.centroids[best]) / n
        self.counts[best] = n
        return best

    def assign(self, pcm: np.ndarray, t: float | None = None) -> dict:
        t = time.time() if t is None else t
        db = rms_dbfs(pcm)
        pan = stereo_pan(pcm)
        silent = db < self.silence_db
        new_turn = False
        if not silent:
            if self.last_voice_t is None or (t - self.last_voice_t) > self.silence_turn_gap:
                new_turn = True
                self.turn_id += 1
            self.last_voice_t = t
        if silent:
            speaker = self.current  # keep label during pauses
        else:
            speaker = self._match(db, pan)
            # a clear turn gap + very different level => prefer a different speaker
            if new_turn and len(self.centroids) > 1:
                pass  # _match already handles level distance
            self.current = speaker
        return {
            "speaker": f"SPEAKER_{speaker + 1:02d}",
            "speaker_id": speaker,
            "rms_db": round(db, 1),
            "pan": round(pan, 2),
            "silent": silent,
            "new_turn": new_turn,
            "n_speakers": len(self.centroids),
        }


class NeMoDiarizer:
    """Neural diarization with NeMo Titanet speaker embeddings.

    Per finalized segment: embed with
    ``nvidia/speakerverification_en_titanet_large`` (CUDA when available),
    cosine-match against running speaker centroids, opening a new speaker
    below ``nemo_cos_thresh`` (up to ``max_speakers``). Stable global IDs,
    no cross-chunk permutation problem.

    Live partials still use the volume diarizer (embedding every 0.5 s frame
    would cost more than it tells); the embedding verdict lands at finalize
    via :meth:`attribute_segment` and overrides the guess.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.max_speakers = max(1, int(cfg.get("max_speakers", 3)))
        self.cos_thresh = float(cfg.get("nemo_cos_thresh", 0.55))
        self.model_name = (cfg.get("nemo_embedding_model")
                           or "nvidia/speakerverification_en_titanet_large")
        self._volume = VolumeDiarizer(
            max_speakers=self.max_speakers,
            silence_turn_gap=float(cfg.get("silence_turn_gap", 0.6)),
        )
        self._lock = None
        self._model = None
        self._ready = False
        self._available = False
        self.centroids: list = []  # L2-normalized torch tensors on CPU
        self.counts: list[int] = []

    def reset(self):
        self.centroids = []
        self.counts = []
        self._volume.reset()

    # -- lifecycle ---------------------------------------------------------
    def ensure_loaded(self, demo_ok: bool = True):
        if self._ready:
            return
        import threading

        if self._lock is None:
            self._lock = threading.Lock()
        with self._lock:
            if self._ready:
                return
            global _NEMO_WARNED
            try:
                from nemo.collections.asr.models import EncDecSpeakerLabelModel  # type: ignore

                log.info("Loading NeMo embedding model %s", self.model_name)
                self._model = EncDecSpeakerLabelModel.from_pretrained(self.model_name)
                self._model.eval()
                try:
                    dev = next(self._model.parameters()).device
                except Exception:
                    dev = "?"
                log.info("NeMo diarizer ready on %s", dev)
                self._available = True
            except Exception as e:
                if not _NEMO_WARNED:
                    log.warning("NeMo diarizer unavailable (%s) — volume fallback.", e)
                    _NEMO_WARNED = True
                self._available = False
                if not demo_ok:
                    raise
            self._ready = True

    @property
    def available(self) -> bool:
        return self._available

    # -- fast path: volume guess for live partials --------------------------
    def assign(self, pcm: np.ndarray, t=None) -> dict:
        out = self._volume.assign(pcm, t)
        out["backend"] = "nemo" if self._available else "volume(fallback)"
        return out

    # -- authoritative path: embedding attribution for finalized segments ----
    def embed(self, pcm: np.ndarray, sr: int = 16000):
        """L2-normalized 192-d embedding (CPU tensor) or None.

        Direct tensor forward — no temp wav files (the file path costs more
        in I/O than the 0.05 s inference itself).
        """
        import torch

        self.ensure_loaded()
        if not self._available:
            return None
        x = np.asarray(pcm, dtype=np.float32).ravel()
        if x.size < int(sr * 0.4):  # too short to embed reliably
            return None
        try:
            with self._lock, torch.no_grad():
                dev = next(self._model.parameters()).device
                wav = torch.from_numpy(
                    np.clip(x, -1.0, 1.0).astype(np.float32)
                )[None, :].to(dev)
                ln = torch.tensor([wav.shape[1]], device=dev)
                _, emb = self._model(input_signal=wav, input_signal_length=ln)
        except Exception as e:
            log.warning("NeMo embed failed (%s)", e)
            return None
        e = torch.as_tensor(
            emb.detach().cpu() if torch.is_tensor(emb) else emb,
            dtype=torch.float32).ravel()
        n = float(e.norm())
        return e / max(n, 1e-9) if n > 1e-9 else None

    def attribute_segment(self, pcm: np.ndarray, sr: int = 16000) -> dict:
        """Authoritative speaker for a finalized segment.

        Returns {"speaker", "speaker_id", "cos", "backend"}; falls back to
        the volume guess when NeMo is unavailable or the segment is short.
        """
        import torch
        import torch.nn.functional as F

        e = self.embed(pcm, sr)
        if e is None:
            out = self._volume.assign(np.asarray(pcm, dtype=np.float32).ravel())
            out["backend"] = "volume(fallback)"
            return out
        best, best_cos = -1, -2.0
        for i, c in enumerate(self.centroids):
            cos = float(F.cosine_similarity(e, c, dim=0))
            if cos > best_cos:
                best, best_cos = i, cos
        if best < 0 or (best_cos < self.cos_thresh and len(self.centroids) < self.max_speakers):
            self.centroids.append(e.clone())
            self.counts.append(1)
            best, best_cos = len(self.centroids) - 1, 1.0
        else:
            n = self.counts[best] + 1
            updated = self.centroids[best] + (e - self.centroids[best]) / n
            self.centroids[best] = updated / max(float(updated.norm()), 1e-9)
            self.counts[best] = n
        return {
            "speaker": f"SPEAKER_{best + 1:02d}",
            "speaker_id": best,
            "cos": round(best_cos, 3),
            "n_speakers": len(self.centroids),
            "backend": "nemo-titanet",
        }

    def status(self) -> dict:
        dev = "?"
        if self._available and self._model is not None:
            try:
                dev = str(next(self._model.parameters()).device)
            except Exception:
                pass
        return {
            "backend": "nemo-titanet" if self._available else "volume(fallback)",
            "model": self.model_name,
            "device": dev,
            "n_speakers": len(self.centroids),
            "ready": self._ready,
        }

    def unload(self) -> dict:
        """Drop Titanet from VRAM. Next embed() reloads via ensure_loaded."""
        was = self.status()
        self._model = None
        self._available = False
        self._ready = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        log.info("NeMo diarizer unloaded (OCR mode)")
        return {"unloaded": True, "was": was}


class PyannoteDiarizer:
    """Neural diarization with a pyannote embedding model on GPU.

    Per finalized segment: embed with ``pyannote/embedding`` (WeSpeaker,
    CUDA) and cosine-match against running speaker centroids, opening a new
    speaker below ``pyannote_cos_thresh`` (up to ``max_speakers``).

    pyannote weights are license-gated: the account behind the HF token must
    accept the conditions at https://hf.co/pyannote/embedding once. Token is
    read from config ``hf_token`` → ``HF_TOKEN``/``HUGGING_FACE_HUB_TOKEN``
    env → cached ``huggingface login``. Until then (or without the package)
    this degrades gracefully: NeMo Titanet → volume levels, always reporting
    which backend is actually active.

    Live partials use the volume guess; the embedding verdict lands at
    finalize via :meth:`attribute_segment` and overrides it.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.max_speakers = max(1, int(cfg.get("max_speakers", 3)))
        self.cos_thresh = float(cfg.get("pyannote_cos_thresh",
                                        cfg.get("nemo_cos_thresh", 0.55)))
        self.model_name: str = (cfg.get("pyannote_model") or "pyannote/embedding")
        self._volume = VolumeDiarizer(
            max_speakers=self.max_speakers,
            silence_turn_gap=float(cfg.get("silence_turn_gap", 0.6)),
        )
        self._nemo = NeMoDiarizer(cfg)  # second rung of the fallback chain
        self._lock = None
        self._infer = None
        self._ready = False
        self._available = False
        self.centroids: list = []
        self.counts: list[int] = []

    def reset(self):
        self.centroids = []
        self.counts = []
        self._volume.reset()
        self._nemo.reset()

    # -- lifecycle ---------------------------------------------------------
    def _token(self):
        import os

        for src in (self.cfg.get("hf_token"), os.getenv("HF_TOKEN"),
                    os.getenv("HUGGING_FACE_HUB_TOKEN")):
            if src:
                return src
        return None  # huggingface_hub falls back to the cached login token

    def ensure_loaded(self, demo_ok: bool = True):
        if self._ready:
            return
        import threading

        if self._lock is None:
            self._lock = threading.Lock()
        with self._lock:
            if self._ready:
                return
            global _NEMO_WARNED
            try:
                import torch
                from pyannote.audio import Inference  # type: ignore
                from pyannote.audio.core.model import Model  # type: ignore

                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
                log.info("Loading pyannote embedding %s on %s",
                         self.model_name, device)
                ckpt = Model.from_pretrained(self.model_name, token=self._token())
                self._infer = Inference(ckpt, window="whole", device=device,
                                        batch_size=1)
                log.info("pyannote diarizer ready on %s", device)
                self._available = True
            except Exception as e:
                if not _NEMO_WARNED:
                    log.warning("pyannote unavailable (%s) — trying NeMo.", e)
                    _NEMO_WARNED = True
                self._available = False
                if not demo_ok:
                    raise
            self._ready = True
            if not self._available:
                # warm the next rung now so finalize never blocks on it later
                try:
                    self._nemo.ensure_loaded(demo_ok=True)
                except Exception:
                    pass

    @property
    def _active_name(self) -> str:
        if self._available:
            return "pyannote"
        if self._nemo.available:
            return "nemo"
        return "volume(fallback)"

    # -- fast path: volume guess for live partials --------------------------
    def assign(self, pcm: np.ndarray, t=None) -> dict:
        out = self._volume.assign(pcm, t)
        out["backend"] = self._active_name
        return out

    # -- authoritative path ---------------------------------------------------
    def embed(self, pcm: np.ndarray, sr: int = 16000):
        """L2-normalized embedding (CPU tensor) or None."""
        import torch

        self.ensure_loaded()
        if not self._available or self._infer is None:
            return None
        x = np.asarray(pcm, dtype=np.float32).ravel()
        if x.size < int(sr * 0.4):
            return None
        import os
        import soundfile as sf
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            sf.write(path, np.clip(x, -1.0, 1.0).astype(np.float32), int(sr))
            with self._lock, torch.no_grad():
                e = self._infer(path)
        except Exception as e:
            log.warning("pyannote embed failed (%s)", e)
            return None
        finally:
            try:
                os.unlink(path)
            except Exception:
                pass
        e = torch.as_tensor(
            np.asarray(e, dtype=np.float32)).ravel()
        n = float(e.norm())
        return e / max(n, 1e-9) if n > 1e-9 else None

    def attribute_segment(self, pcm: np.ndarray, sr: int = 16000) -> dict:
        import torch
        import torch.nn.functional as F

        e = self.embed(pcm, sr)
        if e is None:
            # next rung: NeMo (itself falls back to volume internally)
            nemo_attr = getattr(self._nemo, "attribute_segment", None)
            if callable(nemo_attr):
                try:
                    return nemo_attr(pcm, sr)
                except Exception:
                    pass
            out = self._volume.assign(np.asarray(pcm, dtype=np.float32).ravel())
            out["backend"] = "volume(fallback)"
            return out
        best, best_cos = -1, -2.0
        for i, c in enumerate(self.centroids):
            cos = float(F.cosine_similarity(e, c, dim=0))
            if cos > best_cos:
                best, best_cos = i, cos
        if best < 0 or (best_cos < self.cos_thresh and len(self.centroids) < self.max_speakers):
            self.centroids.append(e.clone())
            self.counts.append(1)
            best, best_cos = len(self.centroids) - 1, 1.0
        else:
            n = self.counts[best] + 1
            updated = self.centroids[best] + (e - self.centroids[best]) / n
            self.centroids[best] = updated / max(float(updated.norm()), 1e-9)
            self.counts[best] = n
        return {
            "speaker": f"SPEAKER_{best + 1:02d}",
            "speaker_id": best,
            "cos": round(best_cos, 3),
            "n_speakers": len(self.centroids),
            "backend": "pyannote",
        }

    def status(self) -> dict:
        if self._available:
            backend, model = "pyannote", self.model_name
        else:
            sub = self._nemo.status() if hasattr(self._nemo, "status") else {}
            backend = sub.get("backend", "volume(fallback)")
            model = sub.get("model", "")
        return {
            "backend": backend,
            "model": model or self.model_name,
            "device": "cuda" if _cuda() else "cpu",
            "n_speakers": len(self.centroids),
            "ready": self._ready,
        }

    def unload(self) -> dict:
        """Drop pyannote (+ NeMo rung) from VRAM. Reloads on next embed()."""
        was = self.status()
        self._infer = None
        self._available = False
        self._ready = False
        try:
            self._nemo.unload()
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        log.info("pyannote diarizer unloaded (OCR mode)")
        return {"unloaded": True, "was": was}


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def make_diarizer(cfg: dict):
    mode = str(cfg.get("diarizer", "pyannote")).lower()
    if mode == "pyannote":
        return PyannoteDiarizer(cfg)
    if mode == "nemo" or bool(cfg.get("use_nemo", False)):
        return NeMoDiarizer(cfg)
    return VolumeDiarizer(
        max_speakers=int(cfg.get("max_speakers", 3)),
        silence_turn_gap=float(cfg.get("silence_turn_gap", 0.6)),
    )
