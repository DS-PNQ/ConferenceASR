"""Speaker-button voices: one small int8 Piper voice per language (sherpa-onnx format).

  python scripts/get_tts.py          # -> models/tts/{vi,en,zh}  (config.yaml › tts_dir)

Picked as the lightest that still read back cleanly through our own Zipformer ASR
(checkpoints/WORKLOG.md has the comparison). ~58 MB download in total, CPU only.
"""
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "models" / "tts"
URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{}.tar.bz2"
VOICES = {
    "vi": "vits-piper-vi_VN-vais1000-medium-int8",  # 21.6 MB, CC BY 4.0
    "en": "vits-piper-en_US-ljspeech-medium-int8",  # 21.1 MB, public domain (lessac reads as
                                                    # well but its licence is research-only)
    "zh": "vits-piper-zh_CN-chaowen-medium-int8",   # 14.0 MB, CC0
}

for lang, name in VOICES.items():
    out = DST / lang
    if any(out.glob("*.onnx")):
        print(f"{lang}: already installed ({out})")
        continue
    with tempfile.TemporaryDirectory() as tmp:
        arc = Path(tmp) / f"{name}.tar.bz2"
        print(f"{lang}: downloading {name} ...")
        urllib.request.urlretrieve(URL.format(name), arc)
        with tarfile.open(arc) as t:
            t.extractall(tmp, filter="data")
        shutil.rmtree(out, ignore_errors=True)
        shutil.move(str(Path(tmp) / name), out)
    print(f"{lang}: ready ({sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 1e6:.1f} MB)")
