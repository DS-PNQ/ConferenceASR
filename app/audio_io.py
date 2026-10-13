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
    """Chunks of at most `seconds`, each cut at the quietest 50 ms in the window's
    second half (a pause / sentence end) instead of a fixed mid-word boundary."""
    n, hop = int(sr * seconds), int(sr * 0.05)
    i = 0
    while len(a) - i > n:
        # ponytail: energy minimum, not a VAD; a 5 s window with no pause still cuts at the softest frame
        win = a[i + n // 2:i + n]
        e = (win[:len(win) // hop * hop].reshape(-1, hop) ** 2).mean(axis=1)
        cut = i + n // 2 + (int(np.argmin(e)) + 1) * hop  # end of the quietest frame
        yield a[i:cut]
        i = cut
    if i < len(a):
        yield a[i:]


if __name__ == "__main__":
    sr = 16000
    tone = np.ones(sr, np.float32)
    # 3.2 s speech, 0.3 s pause, 3 s speech: the cut must land inside the pause, not at 5.0 s
    x = np.concatenate([tone] * 3 + [tone[:3200], np.zeros(4800, np.float32)] + [tone] * 3)
    parts = list(chunk_stream(x, sr, 5.0))
    assert 3.2 < len(parts[0]) / sr <= 3.5, len(parts[0]) / sr
    assert sum(map(len, parts)) == len(x) and all(len(p) <= 5 * sr for p in parts)
    print("chunk_stream OK", [round(len(p) / sr, 2) for p in parts])
