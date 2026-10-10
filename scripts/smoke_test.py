"""Smoke test with NO model weights: exercises device, denoiser, diarizer, pipeline (mock), API routes."""
import numpy as np

from app.diarizer import NemotronDiarizer, VolumeDiarizer, rms_dbfs
from app.device import device_report, resolve_device, resolve_dtype
from app.enhancer import DeepFilterNetEnhancer, _resample
from app.pipeline import ConferencePipeline


class FakeASR:
    def transcribe_array(self, pcm, sr, language=None):
        return ("English", "hello conference, this is a smoke test")

    def transcribe_file(self, p, language=None):
        return ("English", "hello file")


class FakeMT:
    def translate(self, text, tgt="en", src="auto", context=None, terms=None, **kw):
        return f"[{tgt}] {text}"


def main():
    cfg = {"max_speakers": 3, "silence_turn_gap": 0.6, "default_targets": ["en", "zh"],
           "device": "auto", "asr_dtype": "auto"}
    dev = resolve_device("auto")
    dt = resolve_dtype(dev, "auto")
    print("device:", dev, dt)
    print("report:", device_report(cfg))

    sr = 16000
    t = np.arange(sr * 2) / sr
    loud = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    quiet = (0.05 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    print("loud dB:", rms_dbfs(loud), "quiet dB:", rms_dbfs(quiet))

    dia = VolumeDiarizer(max_speakers=2)
    print("diar loud:", dia.assign(loud, t=0.0))
    print("diar quiet:", dia.assign(quiet, t=2.0))

    # Nemotron hand-over logic on fake 10 ms scores (no sidecar, no weights):
    # A 1 s -> B alone 1 s splits at B's start; B talking over A never splits.
    nd = NemotronDiarizer({"diar_split_min": 0.4})
    a, b, both = np.array([.9, .1]), np.array([.1, .9]), np.array([.9, .8])
    nd._preds.append((0, np.stack([a] * 100 + [b] * 100).astype(np.float32)))
    v = nd.turn(0)
    assert (v["speaker"], v["change_at"], v["next_speaker"]) == ("SPEAKER_01", 100 * 160, "SPEAKER_02"), v
    nd.reset()
    nd._preds.append((0, np.stack([a] * 100 + [both] * 100).astype(np.float32)))
    assert nd.turn(0)["change_at"] is None
    nd.reset()  # 0.2 s of A (< diar_split_min) then B: relabel, no split
    nd._preds.append((0, np.stack([a] * 20 + [b] * 100).astype(np.float32)))
    v = nd.turn(0)
    assert v["change_at"] is None and v["speaker"] == "SPEAKER_02", v
    print("nemotron turn(): split / overlap / short-head OK")

    # DeepFilterNet enhancer: resample round-trip is shape-preserving;
    # without the package/weights it must pass audio through untouched.
    up = _resample(loud, 16000, 48000)
    back = _resample(up, 48000, 16000)
    assert abs(len(back) - len(loud)) <= 2, (len(back), len(loud))
    print(f"resample 16k->48k->16k: {len(loud)} -> {len(up)} -> {len(back)}")
    enh = DeepFilterNetEnhancer({"noise_suppress": True})
    out = enh.enhance_array(loud, sr)
    assert out.shape == loud.shape and out.dtype == np.float32, out.shape
    print("enhancer status:", enh.status())
    # disabled enhancer is a strict identity
    enh_off = DeepFilterNetEnhancer({"noise_suppress": False})
    assert np.array_equal(enh_off.enhance_array(loud, sr), loud)
    print("enhancer disabled -> identity OK")

    pipe = ConferencePipeline(cfg, FakeASR(), FakeMT(), dia, enh)
    entry = pipe.process_chunk(loud, sr, targets=["en", "zh", "vi"])
    assert entry and entry["type"] == "utterance", entry
    print("pipeline entry:", entry["speaker"], entry["src_lang"], entry["translations"].keys(),
          "denoised:", entry["denoised"])
    file_entries = pipe.process_file(np.concatenate([loud, quiet]), sr, targets=["en"])
    assert len(file_entries) >= 1, file_entries
    print(f"pipeline file: {len(file_entries)} utterances, denoised={file_entries[0]['denoised']}")

    # live MT: SSBD off the last draft, plain without one; no history on short text
    import queue
    import threading

    from app.streaming import StreamingSession

    class SpyMT(FakeMT):
        _lock = threading.RLock()

        def translate_ssbd(self, text, draft, tgt="en", src="auto", context=None, terms=None, **kw):
            return {"text": f"ssbd({draft}) {text}"}

    s = StreamingSession(cfg, FakeASR(), SpyMT(), dia, enh, queue.Queue())
    assert s._translate("ONE TWO", "vi", "en", ["hist"], None) == "[vi] one two"
    assert s._translate("ONE TWO", "vi", "en", None, None, draft="d") == "ssbd(d) one two"
    s._drafts[7] = {"vi": "d1"}
    s._run_retr_job({"retr": "vi", "seg_id": 7, "stop": threading.Event(), "text": "ONE TWO THREE", "context": None, "terms": None})
    assert s._drafts[7]["vi"] == "ssbd(d1) one two three", s._drafts
    print("live SSBD routing OK")

    # kept-FP8 Linear == fp16 bake, and a failed compile re-bakes it to nn.Linear
    import torch

    from app.mt_engine import _dequantize_fp8, _FP8Linear

    w8 = (torch.randn(8, 16) * 40).to(torch.float8_e4m3fn)
    scale = torch.tensor([0.01])
    x = torch.randn(3, 16)
    lin = _FP8Linear(w8, scale)
    ref = x @ (w8.to(torch.float32) * scale).T
    assert torch.allclose(lin(x), ref, atol=1e-5), "FP8Linear mismatch"
    holder = torch.nn.Sequential(lin)
    holder.hf_quantizer = None
    assert _dequantize_fp8(holder) == 1 and isinstance(holder[0], torch.nn.Linear)
    assert torch.allclose(holder[0](x.half()).float(), ref, atol=0.05)
    print("FP8Linear + re-bake OK")

    # FastAPI routes (no server needed)
    from fastapi.testclient import TestClient

    from app.main import app

    c = TestClient(app)
    h = c.get("/api/health")
    assert h.status_code == 200, "health failed"
    assert "denoise" in h.json(), h.json().keys()
    print("health denoise:", h.json()["denoise"])
    assert c.get("/").status_code == 200, "landing failed"
    assert "tts" in h.json() and "mem" in h.json(), h.json().keys()
    assert c.post("/api/tts", json={"text": "", "lang": "en"}).status_code == 400
    assert c.post("/api/tts", json={"text": "hi", "lang": "xx"}).status_code == 404
    print("API health + landing + tts errors OK (voices:", h.json()["tts"]["langs"], ")")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
