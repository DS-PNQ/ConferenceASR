"""Smoke test with NO model weights: exercises device, denoiser, diarizer, pipeline (mock), API routes."""
import numpy as np

from app.diarizer import VolumeDiarizer, rms_dbfs
from app.device import device_report, resolve_device, resolve_dtype
from app.enhancer import DeepFilterNetEnhancer, _resample
from app.pipeline import ConferencePipeline


class FakeASR:
    def transcribe_array(self, pcm, sr, language=None):
        return ("English", "hello conference, this is a smoke test")

    def transcribe_file(self, p, language=None):
        return ("English", "hello file")


class FakeMT:
    def translate(self, text, tgt="en", src="auto"):
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

    # FastAPI routes (no server needed)
    from fastapi.testclient import TestClient

    from app.main import app

    c = TestClient(app)
    h = c.get("/api/health")
    assert h.status_code == 200, "health failed"
    assert "denoise" in h.json(), h.json().keys()
    print("health denoise:", h.json()["denoise"])
    assert c.get("/").status_code == 200, "landing failed"
    print("API health + landing OK")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
