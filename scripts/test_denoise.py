"""Denoise self-check: the configured chain (DF in-process / DF sidecar) and
the spectral-gate fallback must both cut noise, and report their latency."""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.enhancer import DeepFilterNetEnhancer, spectral_gate  # noqa: E402

sr, secs = 16000, 10
t = np.arange(sr * secs) / sr
voice = (0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)).astype(np.float32)
noise = (np.random.default_rng(0).standard_normal(t.size) * 0.05).astype(np.float32)
x = voice + noise


def noise_left(y):  # energy where the "voice" is silent, relative to input
    off = np.sin(2 * np.pi * 0.5 * t) <= 0
    return 10 * np.log10(np.mean(y[off] ** 2) / np.mean(x[off] ** 2))


enh = DeepFilterNetEnhancer({"noise_suppress": True})
enh.ensure_loaded()
enh.enhance_array(x[:sr], sr)  # warm
spectral_gate(x[:sr], sr)  # warm scipy import
for name, fn in ((enh.status()["backend"] + "/" + enh.mode, lambda a: enh.enhance_array(a, sr)),
                 ("spectral-gate", lambda a: spectral_gate(a, sr))):
    t0 = time.time()
    y = fn(x)
    ms = (time.time() - t0) * 1000
    print(f"{name:28s} {secs}s audio: {ms:7.1f} ms, noise {noise_left(y):6.1f} dB")
    assert y.shape == x.shape and noise_left(y) < -6, name
print("ok")
