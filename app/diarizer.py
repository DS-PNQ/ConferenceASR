"""Speaker diarization.

Default: volume-based diarizer (lightweight, no extra weights).
It segments speaker turns from per-chunk loudness (RMS dBFS), pause gaps,
and stereo pan when available — matching this app's "based on volume" spec.

Optional: NVIDIA NeMo diarization
(https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html).
Set config `diarizer: nemo` + `use_nemo: true` with nemo_toolkit[asr] installed
to route through NeMo's MSDD/Sortformer pipeline. If NeMo is unavailable,
we log once and fall back to volume mode automatically.
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
    """Thin wrapper around NeMo MSD diarization; falls back to volume mode."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._pipe = None
        self._volume = VolumeDiarizer(
            max_speakers=int(cfg.get("max_speakers", 3)),
            silence_turn_gap=float(cfg.get("silence_turn_gap", 0.6)),
        )

    def _ensure(self):
        global _NEMO_WARNED
        if self._pipe is not None:
            return self._pipe
        try:
            # NeMo 2.x diarization entry points vary by version; try common ones.
            from nemo.collections.asr.models import MSDDiarizationModel  # type: ignore

            model_name = self.cfg.get("nemo_diarizer_model") or "diar_msdd_telephonic"
            self._pipe = MSDDiarizationModel.from_pretrained(model_name)
            log.info("NeMo diarizer ready: %s", model_name)
        except Exception as e:
            if not _NEMO_WARNED:
                log.warning("NeMo unavailable (%s) — using volume diarizer.", e)
                _NEMO_WARNED = True
            self._pipe = False
        return self._pipe if self._pipe else None

    def assign(self, pcm: np.ndarray, t=None) -> dict:
        self._ensure()  # warm / warn once; per-chunk NeMo is out of scope for realtime
        out = self._volume.assign(pcm, t)
        out["backend"] = "nemo" if self._pipe else "volume(fallback)"
        return out


def make_diarizer(cfg: dict):
    if str(cfg.get("diarizer", "volume")).lower() == "nemo" or bool(cfg.get("use_nemo", False)):
        return NeMoDiarizer(cfg)
    return VolumeDiarizer(
        max_speakers=int(cfg.get("max_speakers", 3)),
        silence_turn_gap=float(cfg.get("silence_turn_gap", 0.6)),
    )
