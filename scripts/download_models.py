"""Pre-download weights so first run has no network stall.

ASR -> models/zipformer-2e (the engine falls back there when config's
asr_model dir doesn't exist on this machine). MT prefetch below.
"""
import os
from pathlib import Path

MT = os.getenv("MT_MODEL", "tencent/Hy-MT2-1.8B-FP8")
ASR_REPO = "lmcu000/sherpa-onnx-streaming-zipformer-zh-en-vi-2e"
ASR_DIR = Path(__file__).resolve().parent.parent / "models" / "zipformer-2e"

print(f"Downloading ASR: {ASR_REPO} -> {ASR_DIR}")
try:
    from huggingface_hub import snapshot_download

    snapshot_download(ASR_REPO, local_dir=ASR_DIR, allow_patterns=["*.onnx", "tokens.txt"])
except Exception as e:
    print("ASR download FAILED (app falls back to demo ASR):", e)

print(f"Downloading MT: {MT}")
try:
    from huggingface_hub import snapshot_download

    snapshot_download(MT)
except Exception as e:
    print("MT download note:", e)

print("Testing sherpa-onnx import…")
try:
    import sherpa_onnx  # noqa

    print("sherpa-onnx OK")
except Exception as e:
    print("sherpa-onnx missing — pip install sherpa-onnx. Err:", e)

print("Testing DeepFilterNet import + prefetching weights…")
try:
    from df.enhance import init_df

    model, df_state, *_ = init_df(log_level="ERROR", log_file=None)
    print(f"DeepFilterNet OK (sr={df_state.sr()})")
except Exception as e:
    print("DeepFilterNet note (app will run in pass-through mode):", e)

# The app loads it offline from this folder (diarizer falls back here when
# config's nemotron_model dir doesn't exist on this machine, like the ASR).
NEMOTRON_DIR = Path(__file__).resolve().parent.parent / "models" / "nemotron-3-diarization"
print(f"Downloading Nemotron-3-Diarization -> {NEMOTRON_DIR}")
try:
    from huggingface_hub import snapshot_download

    snapshot_download("nvidia/Nemotron-3-Diarization", local_dir=NEMOTRON_DIR,
                      allow_patterns=["config.json", "processor_config.json", "model.safetensors"])
    print("Nemotron OK")
except Exception as e:
    print("Nemotron note (app will use NeMo Titanet):", e)

print("Prefetching NeMo Titanet embedding model (Nemotron fallback)…")
try:
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    m = EncDecSpeakerLabelModel.from_pretrained(
        "nvidia/speakerverification_en_titanet_large")
    print("NeMo Titanet OK")
    del m
except Exception as e:
    print("NeMo note (app will use volume diarizer):", e)

print("Done.")
