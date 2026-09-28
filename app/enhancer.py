"""DeepFilterNet voice suppression (https://github.com/Rikorose/DeepFilterNet).

Runs as a pre-ASR stage in the conference pipeline::

    16 kHz mic/file audio -> 48 kHz -> DeepFilterNet enhance -> 16 kHz -> diarizer/ASR

Device handling: DeepFilterNet's own ``init_df`` moves the model to GPU when
one is available (``df.modules.get_device``), otherwise CPU — matching this
app's CUDA + CPU fall-back story with zero extra code.

If ``deepfilternet`` is not installed (or weights can't load), the enhancer
degrades to a transparent pass-through so recording / diarization / ASR keep
working. :meth:`status` always reports which mode is active.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger("conf.df")

DF_SR = 48000  # DeepFilterNet is a full-band 48 kHz model
_WARNED = False


def _shim_torchaudio_backend():
    """torchaudio>=2.9 removed ``torchaudio.backend``; df 0.5.x still does
    ``from torchaudio.backend.common import AudioMetaData`` at import time.
    Inject a minimal stand-in so ``import df.enhance`` keeps working with
    modern torchaudio. ``enhance()`` itself never touches file I/O
    (torch + libdf only), so the shim is never exercised at runtime.
    """
    try:
        import torchaudio.backend.common  # noqa: F401  (old torchaudio: nothing to do)
        return
    except Exception:
        pass
    try:
        import torchaudio  # noqa: F401
    except Exception:
        return  # no torchaudio at all: df import will fail -> pass-through
    import sys
    import types
    from collections import namedtuple

    if "torchaudio.backend.common" in sys.modules:
        return
    AudioMetaData = namedtuple(
        "AudioMetaData", ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"]
    )
    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = AudioMetaData
    backend = types.ModuleType("torchaudio.backend")
    backend.common = common
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common
    try:
        torchaudio.backend = backend  # type: ignore[attr-defined]
    except Exception:
        pass


def _resample(a: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Resample mono float32 between sample rates.

    Prefers DeepFilterNet's own ``df.io.resample`` when importable
    (best quality); otherwise torch linear interpolation with a moving-average
    anti-alias stage for downsampling. Dependency-free fallback otherwise.
    """
    a = np.asarray(a, dtype=np.float32).ravel()
    if src == dst or a.size == 0:
        return a
    # 1) df.io.resample (ships with deepfilternet)
    try:
        import torch
        from df.io import resample as df_resample  # type: ignore

        t = torch.from_numpy(a)
        out = df_resample(t, src, dst)
        return np.asarray(out.numpy() if hasattr(out, "numpy") else out, dtype=np.float32).ravel()
    except Exception:
        pass
    # 2) torch interpolate (+ moving-average anti-alias on downsample)
    try:
        import torch
        import torch.nn.functional as F

        x = a
        if dst < src:  # crude anti-alias: average over each decimation window
            k = max(2, round(src / dst))
            kernel = np.ones(k, dtype=np.float32) / k
            x = np.convolve(x, kernel, mode="same").astype(np.float32)
        t = torch.from_numpy(x)[None, None, :]
        n_out = max(1, int(round(len(x) * dst / src)))
        y = F.interpolate(t, size=n_out, mode="linear", align_corners=False)
        return y[0, 0].numpy().astype(np.float32)
    except Exception:
        pass
    # 3) last-resort index resample (no new deps at all)
    ratio = dst / float(src)
    idx = (np.arange(int(len(a) * ratio)) / ratio).astype(int)
    return a[np.clip(idx, 0, len(a) - 1)]


class DeepFilterNetEnhancer:
    """Lazy, thread-safe DeepFilterNet wrapper with pass-through fallback."""

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.enabled: bool = bool(self.cfg.get("noise_suppress", True))
        self.model_name: str | None = self.cfg.get("df_model") or None  # None -> pkg default (DeepFilterNet3)
        self.post_filter: bool = bool(self.cfg.get("df_post_filter", False))
        self.atten_lim: float | None = self.cfg.get("df_atten_lim_db")
        self._lock = threading.Lock()
        self._model = None
        self._df_state = None
        self._ready = False
        self._available = False  # package importable AND weights loaded

    # -- lifecycle ---------------------------------------------------------
    def ensure_loaded(self, demo_ok: bool = True):
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            global _WARNED
            try:
                _shim_torchaudio_backend()
                from df.enhance import enhance as _enhance  # noqa: F401  (validates install)
                from df.enhance import init_df  # type: ignore

                kw: dict = dict(post_filter=self.post_filter, log_level="ERROR", log_file=None)
                if self.model_name:
                    kw["model_base_dir"] = self.model_name
                out = init_df(**kw)
                # installed versions return 3- or 4-tuples; only first two matter
                self._model, self._df_state = out[0], out[1]
                self._ready = True
                self._available = True
                log.info("DeepFilterNet ready (model=%s)", getattr(self, "model_name", None) or "default")
            except Exception as e:
                if not _WARNED:
                    log.warning("DeepFilterNet unavailable (%s) — pass-through mode.", e)
                    _WARNED = True
                self._ready = True  # don't retry every chunk; pass-through from here
                self._available = False
                if not demo_ok:
                    raise

    @property
    def active(self) -> bool:
        """True only when suppression will actually run."""
        return self.enabled and self._available

    # -- inference ---------------------------------------------------------
    def enhance_array(self, pcm: np.ndarray, sr: int) -> np.ndarray:
        """Denoise mono audio, returning float32 at the *input* sample rate.

        Returns the input unchanged when disabled or unavailable.
        """
        x = np.asarray(pcm, dtype=np.float32).ravel()
        if not self.enabled:
            return x
        self.ensure_loaded()
        if not self._available:
            return x
        with self._lock:
            try:
                import torch

                from df.enhance import enhance  # type: ignore

                up = _resample(x, int(sr), DF_SR)
                audio = torch.from_numpy(up)[None, :]  # [C=1, T] @ 48 kHz
                with torch.no_grad():
                    enh = enhance(self._model, self._df_state, audio,
                                  pad=True, atten_lim_db=self.atten_lim)
                y = np.asarray(enh.numpy() if hasattr(enh, "numpy") else enh,
                               dtype=np.float32).ravel()
                return _resample(y, DF_SR, int(sr))
            except Exception as e:
                global _WARNED
                if not _WARNED:
                    log.warning("DeepFilterNet enhance failed (%s) — passing through.", e)
                    _WARNED = True
                self._available = False
                return x

    # -- introspection ------------------------------------------------------
    def status(self) -> dict:
        device = "cpu"
        if self._available:
            try:
                from df.modules import get_device  # type: ignore

                device = str(get_device())
            except Exception:
                pass
        return {
            "backend": "deepfilternet" if self._available else "passthrough",
            "model": self.model_name or "DeepFilterNet3 (default)",
            "device": device,
            "target_sr": DF_SR,
            "enabled": self.enabled,
            "ready": self._ready,
        }
