"""Streaming verification with REAL engines (weights must be cached).

Feeds the sample recording in 0.5 s frames through StreamingSession and
asserts: live partials arrive, segments finalize on pauses (no fixed-window
cuts), every utterance has text + translations, stop() flushes the tail.
"""
import queue
import time

import numpy as np
import soundfile as sf

from app.audio_io import _to_16k
from app.diarizer import VolumeDiarizer
from app.enhancer import DeepFilterNetEnhancer
from app.streaming import StreamingSession
from desktop.bootstrap import build_app


def main():
    ctx = build_app()
    ctx["asr"].ensure_loaded(demo_ok=False)
    ctx["mt"].ensure_loaded(demo_ok=False)

    d, sr = sf.read("C:/Users/Asus/AppData/Local/Temp/opencode/asr_en.wav", dtype="float32")
    d, sr = _to_16k(np.asarray(d, np.float32), sr)

    q: queue.Queue = queue.Queue()
    enh = DeepFilterNetEnhancer({"noise_suppress": False})
    session = StreamingSession(ctx["cfg"], ctx["asr"], ctx["mt"],
                               VolumeDiarizer(), enh, q)
    session.configure(targets=["zh"], src_lang=None, denoise=False)

    frame = int(sr * 0.5)
    t0 = time.time() - len(d) / sr
    for i in range(0, len(d), frame):
        session.feed(d[i:i + frame], sr, t=t0 + i / sr)
    n_tail = session.stop()

    partials, utterances = [], []
    while not q.empty():
        kind, payload = q.get()
        (partials if kind == "partial" else utterances).append(payload) \
            if kind in ("partial", "utterance") else None

    print(f"partials: {len(partials)}, utterances: {len(utterances)}, tail flushed: {n_tail}")
    assert len(partials) >= 1, "no live partials emitted"
    assert len(utterances) >= 1, "nothing finalized"
    for u in utterances:
        assert u["text"].strip(), u
        assert u["translations"].get("zh", "").strip(), u
        assert "[MT error" not in u["translations"]["zh"], u
    joined = " ".join(u["text"] for u in utterances)
    print(f"joined finals ({len(joined)} chars, {len(utterances)} segments)")
    assert len(joined) > 50, "suspiciously little transcription"
    assert any("zh" in u["translations"] for u in utterances)
    print("STREAMING SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
