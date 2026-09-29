"""Pre-download weights so first run has no network stall.

ASR needs nothing (local ONNX folder). MT prefetch below.
"""
import os

MT = os.getenv("MT_MODEL", "tencent/Hy-MT2-1.8B-FP8")

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

print("Prefetching NeMo Titanet embedding model…")
try:
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    m = EncDecSpeakerLabelModel.from_pretrained(
        "nvidia/speakerverification_en_titanet_large")
    print("NeMo Titanet OK")
    del m
except Exception as e:
    print("NeMo note (app will use volume diarizer):", e)

print("Done.")
