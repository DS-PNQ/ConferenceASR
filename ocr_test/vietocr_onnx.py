"""VietOCR seq2seq ONNX runtime — the Python twin of the planned OcrModule.java.

Keep this file's preprocessing and decode loop line-for-line portable: the Java
port should match it, and eval below is the parity gate for that port.

  python vietocr_onnx.py            # eval all variants vs torch on out/results.json lines
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models" / "vietocr_s2s"
SOS, EOS, MAX_STEPS = 1, 2, 128


def preprocess(crop_bgr: np.ndarray, resample: str = "linear") -> np.ndarray:
    """Height 32, width = 32*w/h rounded UP to a multiple of 10, clamped 32..512.

    VietOCR trains with PIL LANCZOS; Android's Bitmap.createScaledBitmap(filter=true)
    is bilinear, so 'linear' is what the phone will actually feed.
    """
    h, w = crop_bgr.shape[:2]
    nw = int(32 * w / max(h, 1))
    nw = int(np.ceil(nw / 10) * 10)
    nw = min(max(nw, 32), 512)
    rgb = crop_bgr[:, :, ::-1]
    if resample == "lanczos":
        out = np.asarray(Image.fromarray(rgb).resize((nw, 32), Image.LANCZOS))
    else:
        interp = {"linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA}[resample]
        out = cv2.resize(np.ascontiguousarray(rgb), (nw, 32), interpolation=interp)
    return (out.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


class VietOcrOnnx:
    def __init__(self, variant: str = "fp32", threads: int = 4):
        d = MODELS / variant
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        prov = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(d / "encoder.onnx"), so, providers=prov)
        self.dec = ort.InferenceSession(str(d / "decoder.onnx"), so, providers=prov)
        self.vocab = (d / "vocab.txt").read_text(encoding="utf-8").split("\n")

    def read(self, crop_bgr: np.ndarray, resample: str = "linear") -> tuple[str, float]:
        enc_out, hidden = self.enc.run(None, {"image": preprocess(crop_bgr, resample)})
        tok, chars, probs = SOS, [], []
        for _ in range(MAX_STEPS):
            logits, hidden = self.dec.run(None, {"token": np.array([tok], np.int64),
                                                 "hidden": hidden, "enc_out": enc_out})
            row = logits[0]
            tok = int(row.argmax())
            if tok == EOS:
                break
            e = np.exp(row - row.max())
            p = float(e[tok] / e.sum())
            if tok > 3:
                chars.append(self.vocab[tok - 4])
                probs.append(p)
        return "".join(chars), (float(np.mean(probs)) if probs else 0.0)


# ---------------- shared crops ----------------

def _load_lines():
    from run_ocr_test import crop_quad
    d = json.load(open(HERE / "out" / "results.json", encoding="utf-8"))
    for r in d["results"]:
        pil = ImageOps.exif_transpose(Image.open(HERE / "photos" / r["photo"])).convert("RGB")
        img = np.ascontiguousarray(np.asarray(pil)[:, :, ::-1])
        for ln in r["lines"]:
            yield r["photo"], crop_quad(img, ln["box"]), ln


def calibration_crops(n: int) -> list[np.ndarray]:
    photos = sorted({p for p, _, _ in _load_lines()})
    calib_photos = set(photos[: len(photos) // 2])
    crops = [preprocess(c) for p, c, _ in _load_lines() if p in calib_photos]
    return crops[:n]


# ---------------- eval ----------------

def main():
    from run_ocr_test import VietOCR, cer
    lines = list(_load_lines())
    photos = sorted({p for p, _, _ in lines})
    calib_photos = set(photos[: len(photos) // 2])
    print(f"{len(lines)} lines from {len(photos)} photos "
          f"(calibration photos: {len(calib_photos)}, held-out: {len(photos) - len(calib_photos)})")

    torch_s2s = VietOCR("vgg_seq2seq")
    ref = []
    t0 = time.perf_counter()
    for _, c, _ in lines:
        ref.append(torch_s2s.read(c)[0])
    torch_ms = (time.perf_counter() - t0) * 1000 / len(lines)
    transformer = [ln["vietocr"]["text"] for _, _, ln in lines]

    rows = [("torch fp32 (lanczos)", torch_ms, ref, None)]
    # int8_dyn is left out: ConvInteger made it ~10x slower than fp32 on CPU
    for variant in ("fp32", "fp16", "int8_qdq"):
        if not (MODELS / variant / "encoder.onnx").exists():
            continue
        eng = VietOcrOnnx(variant)
        for resample in ("lanczos", "area", "linear"):
            eng.read(lines[0][1], resample)                       # warm-up
            t0 = time.perf_counter()
            hyp = [eng.read(c, resample)[0] for _, c, _ in lines]
            ms = (time.perf_counter() - t0) * 1000 / len(lines)
            rows.append((f"onnx {variant} ({resample})", ms, hyp, variant))

    print(f"\n{'engine':28} {'ms/line':>8} {'size MB':>8} {'=torch':>7} "
          f"{'CER vs torch':>13} {'held-out':>9} {'CER vs transformer':>19}")
    for name, ms, hyp, variant in rows:
        size = (sum(f.stat().st_size for f in (MODELS / variant).glob("*.onnx")) / 1e6
                if variant else 89.6)
        same = sum(a == b for a, b in zip(hyp, ref))
        c_all = np.mean([cer(a, b) for a, b in zip(ref, hyp)])
        c_held = np.mean([cer(a, b) for (p, _, _), a, b in zip(lines, ref, hyp) if p not in calib_photos])
        c_tr = np.mean([cer(a, b) for a, b in zip(transformer, hyp)])
        print(f"{name:28} {ms:8.0f} {size:8.1f} {same:4}/{len(lines)} {c_all:13.4f} "
              f"{c_held:9.4f} {c_tr:19.4f}")


if __name__ == "__main__":
    main()
