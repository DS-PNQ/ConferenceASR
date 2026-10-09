"""Neural voice activity detection (Silero VAD, https://github.com/snakers4/silero-vad).

Replaces the energy-threshold-only speech/silence decision with a learned
gate: keyboard bursts and HVAC hum above -45 dB no longer open segments, and
quiet speech below the energy floor no longer splits them.

Runs statefully (RNN state carries across feed() frames within a stream;
reset on configure/stop) with ONE batched forward per 0.5 s frame — 31
windows as a [31, 512] batch, not 31 launches. CUDA when available (the model
is ~2 MB; CPU fallback is just as fast on small batches).
"""
from __future__ import annotations

import logging
import os
import threading

import numpy as np

log = logging.getLogger("conf.vad")

WIN = 512  # Silero window @16 kHz
_WARNED = False


class SileroVAD:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.threshold = float(cfg.get("vad_threshold", 0.5))
        self.enabled = bool(cfg.get("vad_enable", True))
        pref = str(os.getenv("VAD_DEVICE", cfg.get("vad_device", "auto"))).lower()
        self.want_cuda = pref in ("auto", "cuda", "gpu") and pref != "cpu"
        self.device = "cpu"
        self._lock = threading.Lock()
        self._model = None
        self._ready = False
        self._failed = False

    def ensure_loaded(self, demo_ok: bool = True):
        """Load the JIT model. On failure: fail-open (all frames = speech)."""
        if self._ready or not self.enabled:
            return
        with self._lock:
            if self._ready or not self.enabled:
                return
            global _WARNED
            try:
                import torch

                # silero_vad's import does torch.set_num_threads(1): that
                # invalidates the compiled MT graph (full ~16 s recompile on
                # the first live translation) and single-threads torch CPU.
                n_threads = torch.get_num_threads()
                from silero_vad import load_silero_vad
                torch.set_num_threads(n_threads)

                if self.want_cuda and torch.cuda.is_available():
                    self.device = "cuda"
                log.info("Loading Silero VAD on %s", self.device)
                self._model = load_silero_vad().to(self.device)
                self._model.eval()
                # warmup: first forward pays autotune, not a live frame
                with torch.no_grad():
                    z = torch.zeros((4, WIN), dtype=torch.float32,
                                    device=self.device)
                    self._model.reset_states()
                    self._model(z, 16000)
                    self._model.reset_states()
                log.info("Silero VAD ready on %s", self.device)
            except Exception as e:
                if not _WARNED:
                    log.warning("Silero VAD unavailable (%s) — fail-open.", e)
                    _WARNED = True
                self._failed = True
                if not demo_ok:
                    raise
            self._ready = True

    @property
    def active(self) -> bool:
        return bool(self.enabled and self._ready and not self._failed
                    and self._model is not None)

    def reset(self):
        """Clear RNN state (new stream / new file)."""
        try:
            if self._model is not None:
                with self._lock:
                    self._model.reset_states()
        except Exception:
            pass

    def frame_speech(self, pcm: np.ndarray, sr: int = 16000) -> tuple[bool, float]:
        """One batched forward per frame -> (is_speech, mean_prob).

        Fail-open: model missing/disabled/error => (True, 1.0) so recording
        and endpointing behave exactly as before VAD existed.
        """
        import torch

        self.ensure_loaded()
        if not self.active:
            return True, 1.0
        x = np.asarray(pcm, dtype=np.float32).ravel()
        if x.size == 0:
            return False, 0.0
        if int(sr) != 16000:  # feed() already resamples; guard anyway
            ratio = 16000 / float(sr)
            idx = (np.arange(int(len(x) * ratio)) / ratio).astype(int)
            x = x[np.clip(idx, 0, len(x) - 1)]
        n = (len(x) + WIN - 1) // WIN
        try:
            with self._lock, torch.no_grad():
                t = torch.from_numpy(
                    np.pad(x, (0, n * WIN - len(x)))).reshape(n, WIN).to(
                    self.device)
                probs = self._model(t, 16000).reshape(-1)
                mean_p = float(probs.mean().item())
        except Exception as e:
            log.warning("VAD forward failed (%s) — fail-open frame.", e)
            return True, 1.0
        return mean_p >= self.threshold, mean_p

    def unload(self) -> dict:
        with self._lock:
            was = self.status()
            self._model = None
            self._ready = False
            self._failed = False
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        return {"unloaded": True, "was": was}

    def status(self) -> dict:
        return {
            "backend": "silero" if self.active else (
                "off" if not self.enabled else "fail-open"),
            "model": "snakers4/silero-vad",
            "device": self.device if self.active else "-",
            "threshold": self.threshold,
            "ready": self._ready,
        }
