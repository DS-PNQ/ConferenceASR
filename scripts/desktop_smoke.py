"""Desktop headless smoke test: worker + pipeline + recorder listing + UI import.

No window is opened and no mic is captured; models are faked.
"""
import queue

import numpy as np

from app.diarizer import VolumeDiarizer
from app.enhancer import DeepFilterNetEnhancer
from app.pipeline import ConferencePipeline
from desktop.recorder import MicRecorder, list_input_devices
from desktop.worker import InferenceWorker


class FakeASR:
    _rec = None  # no recognizer -> session.configure must raise RuntimeError

    def ensure_loaded(self, demo_ok=True):
        pass

    def transcribe_array(self, pcm, sr, language=None):
        return ("English", "desktop smoke test utterance")

    def status(self):
        return {"model": "fake-asr", "backend": "fake", "ready": True}


class FakeMT:
    def ensure_loaded(self, demo_ok=True):
        pass

    def translate(self, text, tgt="en", src="auto"):
        return f"[{tgt}] {text}"

    def translate_multi(self, text, targets, src="auto"):
        return {t: self.translate(text, tgt=t, src=src) for t in targets}

    def status(self):
        return {"model": "fake-mt", "ready": True}


def drain(q, timeout=15.0, want=("utterance", "ready", "file_done")):
    import time

    out, t0 = [], time.time()
    while time.time() - t0 < timeout:
        try:
            out.append(q.get(timeout=0.5))
        except queue.Empty:
            pass
        if any(m[0] in want for m in out):
            break
    return out


def main():
    cfg = {"max_speakers": 3, "silence_turn_gap": 0.6, "default_targets": ["en", "zh"],
           "sample_rate": 16000, "segment_seconds": 4.0, "frame_seconds": 0.5,
           "endpoint_silence": 1.0, "endpoint_min_speech": 0.5, "max_segment": 20.0,
           "partial_translate_interval": 2.5, "partial_min_delta": 6, "partial_min_chars": 10}
    enh = DeepFilterNetEnhancer({"noise_suppress": False})
    dia = VolumeDiarizer()
    pipe = ConferencePipeline(cfg, FakeASR(), FakeMT(), dia, enh)
    worker = InferenceWorker(cfg, pipe, pipe.asr, pipe.mt, dia, enh)
    worker.start()

    # 1. warmup posts ready
    worker.warmup()
    msgs = drain(worker.results)
    assert any(m[0] == "ready" for m in msgs), msgs
    print("warmup ready OK")

    # 2. stream_start with unavailable ASR -> clean error, worker stays alive
    # (FakeASR has no recognizer; configure() must fail loudly, not hang)
    worker.submit(("stream_start", {"targets": ["en"], "src_lang": None, "denoise": False}))
    msgs = drain(worker.results, want=("error",))
    assert any(m[0] == "error" for m in msgs), msgs
    print("stream_start failure contained OK")
    worker.submit(("stream_stop",))
    msgs = drain(worker.results, want=("status",))
    assert any(m[0] == "status" for m in msgs), msgs
    # worker still alive after the failed session?
    worker.warmup()
    assert any(m[0] == "ready" for m in drain(worker.results)), "worker died"
    print("worker resilience OK")

    # 3. file task routes through a session too
    sr = 16000
    t = np.arange(sr * 2) / sr
    chunk = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    worker.submit(("file", chunk, sr, ["en"], None, False))
    msgs = drain(worker.results, timeout=10.0, want=("file_done", "error"))
    assert any(m[0] in ("file_done", "error") for m in msgs), msgs
    print("worker file task OK:", [m[0] for m in msgs])

    # 4. mic device listing (no capture)
    devs = list_input_devices()
    assert isinstance(devs, list)
    print(f"input devices: {len(devs)} found")

    # 5. recorder settings plumbing (no stream opened)
    captured = {}

    def fake_settings():
        return {"targets": ["en"], "src_lang": None, "denoise": False, "device": None}

    rec = MicRecorder(queue.Queue(), fake_settings)
    assert rec.level == 0.0 and not rec.running
    print("recorder plumbing OK")

    # 6. UI module imports without opening a window
    import desktop.ui as ui

    assert hasattr(ui, "ConferenceApp")
    print("desktop.ui import OK")

    worker.submit(("stop",))
    print("DESKTOP SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
