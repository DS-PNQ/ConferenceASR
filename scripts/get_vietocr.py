"""Optional Vietnamese OCR: export VietOCR vgg_seq2seq to fp16 ONNX.

  python scripts/get_vietocr.py      # -> models/vietocr-s2s (config.yaml › vi_ocr_model)

vietocr and its export deps go into a throwaway --target dir, never into
the backend env. Weights download from vocr.vn on first run (~90 MB).
The exporter is ocr_test/export_vietocr_onnx.py (fp32 parity-checked, then fp16).
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "models" / "vietocr-s2s"

with tempfile.TemporaryDirectory() as tmp:
    deps = Path(tmp) / "deps"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                           "--target", str(deps), "vietocr==0.3.13", "einops", "gdown",
                           "beautifulsoup4", "soupsieve", "prefetch_generator", "lmdb",
                           "onnxconverter-common"])
    sys.path[:0] = [str(deps), str(ROOT / "ocr_test")]
    import export_vietocr_onnx as ex

    fp32 = Path(tmp) / "fp32"
    ex.export_fp32(ex.load_torch(), fp32)
    ex.export_fp16(fp32, DST)
print(f"VietOCR fp16 ready: {DST} "
      f"({sum(f.stat().st_size for f in DST.glob('*.onnx')) / 1e6:.1f} MB)")
