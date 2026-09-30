"""Audio helpers: decode uploads, normalise to 16 kHz mono float32."""
from __future__ import annotations

import io
import wave

import numpy as np


def decode_bytes(raw: bytes, mime: str = "") -> tuple[np.ndarray, int]:
    """Decode webm/wav/ogg/mp3 bytes -> (mono float32, sr). Prefers soundfile."""
    # 1) soundfile (wav/flac/ogg)
    try:
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        a = np.asarray(data, dtype=np.float32)
        if a.ndim > 1:
            a = a.mean(axis=-1).astype(np.float32)
        return _to_16k(a, sr)
    except Exception:
        pass
    # 2) stdlib wav
    try:
        with wave.open(io.BytesIO(raw), "rb") as w:
            n, ch, sr, sw = w.getnframes(), w.getnchannels(), w.getframerate(), w.getsampwidth()
            frames = w.readframes(n)
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
        a = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        if sw == 1:
            a = (a - 128) / 128.0
        elif sw == 2:
            a /= 32768.0
        else:
            a /= 2147483648.0
        a = a.reshape(-1, ch).mean(axis=1)
        return _to_16k(a, sr)
    except Exception as e:
        raise ValueError(
            f"Could not decode audio ({mime or 'unknown type'}). "
            "Upload WAV (16-bit PCM) or install ffmpeg/imageio-ffmpeg for webm/mp3."
        ) from e


def decode_b64_pcm16(b64: str, sr_in: int = 16000) -> tuple[np.ndarray, int]:
    import base64

    raw = base64.b64decode(b64)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return _to_16k(a, int(sr_in or 16000))


def _to_16k(a: np.ndarray, sr: int, target: int = 16000) -> tuple[np.ndarray, int]:
    a = np.asarray(a, dtype=np.float32).ravel()
    if sr != target and len(a) > 0:
        ratio = target / float(sr)
        idx = (np.arange(int(len(a) * ratio)) / ratio).astype(int)
        a = a[np.clip(idx, 0, len(a) - 1)]
    # normalise softly to use ASR dynamic range
    peak = float(np.abs(a).max()) if len(a) else 0.0
    if peak > 1e-6:
        a = (a / max(peak, 0.05) * 0.9).astype(np.float32)
    return a, target


def chunk_stream(a: np.ndarray, sr: int, seconds: float):
    n = int(sr * seconds)
    for i in range(0, len(a), n):
        yield a[i:i + n]


def split_at_pauses(a: np.ndarray, sr: int, min_pause: float = 0.4, max_seconds: float = 20.0,
                    silence_db: float = -45.0, frame: float = 0.02):
    """Cut a whole file into utterances at pauses (fixed windows cut words in half).

    A pause is >= min_pause s of frames below the silence level (silence_db, raised to
    10 dB above the file's noise floor for noisy audio). Cuts go in the middle of each
    pause; a stretch longer than max_seconds is cut at its quietest frame.
    """
    a = np.asarray(a, dtype=np.float32).ravel()
    hop = max(1, int(sr * frame))
    n = len(a) // hop
    if n < 2:
        if len(a):
            yield a
        return
    db = 10 * np.log10(np.mean(a[:n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    thr = max(silence_db, float(np.percentile(db, 5)) + 10)
    quiet = db < thr
    cuts, i, need = [], 0, max(1, int(min_pause / frame))
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if j - i >= need and i > 0 and j < n:
                cuts.append((i + j) // 2)
            i = j
        else:
            i += 1
    bounds, start, max_f = [], 0, int(max_seconds / frame)
    for c in cuts + [n]:
        while c - start > max_f:  # run-on speech: cut at the quietest frame in its second half
            lo = start + max_f // 2
            k = lo + int(np.argmin(db[lo:start + max_f]))
            bounds.append((start, k))
            start = k
        bounds.append((start, c))
        start = c
    for s, e in bounds:
        if (~quiet[s:e]).any():
            yield a[s * hop: len(a) if e == n else e * hop]
