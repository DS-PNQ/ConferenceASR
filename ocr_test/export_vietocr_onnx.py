"""Export VietOCR vgg_seq2seq to ONNX for the phone (OnSpeak47 OcrModule).

Two graphs, so the Java side only runs a plain greedy loop:

  encoder.onnx  image (1,3,32,W) float 0..1 RGB
                  -> enc_out (T,1,512), hidden (1,256)          T = W/4
  decoder.onnx  token (1,) int64, hidden (1,256), enc_out (T,1,512)
                  -> logits (1,V), hidden_out (1,256)

Tokens: 0 pad, 1 <sos>, 2 <eos>, 3 mask, 4.. = vocab.txt chars in order.
Decode: start with 1, argmax each step, stop on 2 (max 128 steps).

Variants written to models/vietocr_s2s/:
  fp32/    reference
  fp16/    weights + compute in fp16 (I/O kept fp32)
  int8_dyn/  dynamic int8 (weights int8, activations quantised at runtime)
  int8_qdq/  static int8 QDQ, calibrated on crops from photos/ (encoder only;
             decoder stays dynamic int8 — it is 2.2M params and RNN-shaped)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
OUT = HERE / "models" / "vietocr_s2s"


class Encoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.vgg, self.enc = m.cnn.model, m.transformer.encoder

    def forward(self, img):
        # vietocr Vgg.forward, but with explicit permute axes (its
        # permute(-1, 0, 1) exports as an invalid negative Transpose perm)
        conv = self.vgg.last_conv_1x1(self.vgg.features(img))   # (B, 256, H', W')
        src = conv.transpose(2, 3).flatten(2).permute(2, 0, 1)  # (T, B, 256)
        enc_out, h = self.enc.rnn(src)            # (T, B, 512), (2, B, 256)
        hidden = torch.tanh(self.enc.fc(torch.cat((h[-2], h[-1]), dim=1)))
        return enc_out, hidden


class DecoderStep(nn.Module):
    """vietocr Decoder.forward minus its tensor assert (breaks tracing)."""

    def __init__(self, m):
        super().__init__()
        self.d = m.transformer.decoder

    def forward(self, token, hidden, enc_out):
        d = self.d
        emb = d.embedding(token.unsqueeze(0))     # (1, B, E)
        a = d.attention(hidden, enc_out).unsqueeze(1)          # (B, 1, T)
        weighted = torch.bmm(a, enc_out.permute(1, 0, 2)).permute(1, 0, 2)  # (1, B, 512)
        out, h = d.rnn(torch.cat((emb, weighted), dim=2), hidden.unsqueeze(0))
        logits = d.fc_out(torch.cat((out.squeeze(0), weighted.squeeze(0), emb.squeeze(0)), dim=1))
        return logits, h.squeeze(0)


def load_torch():
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor
    cfg = Cfg.load_config_from_name("vgg_seq2seq")
    cfg["device"] = "cpu"
    cfg["cnn"]["pretrained"] = False
    p = Predictor(cfg)
    p.model.eval()
    return p


def export_fp32(p, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    enc, dec = Encoder(p.model).eval(), DecoderStep(p.model).eval()
    img = torch.rand(1, 3, 32, 160)
    with torch.no_grad():
        enc_out, hidden = enc(img)
        torch.onnx.export(enc, (img,), dst / "encoder.onnx", dynamo=False, opset_version=17,
                          input_names=["image"], output_names=["enc_out", "hidden"],
                          dynamic_axes={"image": {3: "W"}, "enc_out": {0: "T"}})
        tok = torch.tensor([1], dtype=torch.long)
        torch.onnx.export(dec, (tok, hidden, enc_out), dst / "decoder.onnx", dynamo=False,
                          opset_version=17,
                          input_names=["token", "hidden", "enc_out"],
                          output_names=["logits", "hidden_out"],
                          dynamic_axes={"enc_out": {0: "T"}})
    (dst / "vocab.txt").write_text("\n".join(p.vocab.chars), encoding="utf-8")
    # parity on the export inputs
    import onnxruntime as ort
    se = ort.InferenceSession(str(dst / "encoder.onnx"), providers=["CPUExecutionProvider"])
    sd = ort.InferenceSession(str(dst / "decoder.onnx"), providers=["CPUExecutionProvider"])
    for w in (32, 160, 512):
        x = torch.rand(1, 3, 32, w)
        with torch.no_grad():
            te, th = enc(x)
            tl, _ = dec(torch.tensor([1]), th, te)
        oe, oh = se.run(None, {"image": x.numpy()})
        ol, _ = sd.run(None, {"token": np.array([1], np.int64), "hidden": oh, "enc_out": oe})
        print(f"  W={w}: enc max|d|={np.abs(oe - te.numpy()).max():.2e} "
              f"logits max|d|={np.abs(ol - tl.numpy()).max():.2e}")


def export_fp16(src: Path, dst: Path):
    import onnx
    from onnxconverter_common import float16
    dst.mkdir(parents=True, exist_ok=True)
    for n in ("encoder", "decoder"):
        m = float16.convert_float_to_float16(onnx.load(src / f"{n}.onnx"), keep_io_types=True)
        onnx.save(m, dst / f"{n}.onnx")
    shutil.copy(src / "vocab.txt", dst / "vocab.txt")


def export_int8_dyn(src: Path, dst: Path):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    dst.mkdir(parents=True, exist_ok=True)
    for n in ("encoder", "decoder"):
        quantize_dynamic(src / f"{n}.onnx", dst / f"{n}.onnx", weight_type=QuantType.QInt8)
    shutil.copy(src / "vocab.txt", dst / "vocab.txt")


def export_int8_qdq(src: Path, dst: Path, calib: list[np.ndarray]):
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_dynamic, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"image": x} for x in calib])

        def get_next(self):
            return next(self.it, None)

    dst.mkdir(parents=True, exist_ok=True)
    pre = dst / "encoder.pre.onnx"
    quant_pre_process(str(src / "encoder.onnx"), str(pre))
    quantize_static(pre, dst / "encoder.onnx", Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True, op_types_to_quantize=["Conv", "MatMul", "Gemm"])
    pre.unlink()
    quantize_dynamic(src / "decoder.onnx", dst / "decoder.onnx", weight_type=QuantType.QInt8)
    shutil.copy(src / "vocab.txt", dst / "vocab.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=int, default=120, help="crops for static int8 calibration")
    args = ap.parse_args()

    p = load_torch()
    print("fp32 export + parity:")
    export_fp32(p, OUT / "fp32")
    print("fp16 ...")
    export_fp16(OUT / "fp32", OUT / "fp16")
    print("int8 dynamic ...")
    export_int8_dyn(OUT / "fp32", OUT / "int8_dyn")

    # calibration crops: the first half of photos/ (eval reports both halves)
    from vietocr_onnx import calibration_crops
    calib = calibration_crops(args.calib)
    print(f"int8 static QDQ (calibrated on {len(calib)} crops) ...")
    export_int8_qdq(OUT / "fp32", OUT / "int8_qdq", calib)

    for d in sorted(OUT.iterdir()):
        size = sum(f.stat().st_size for f in d.glob("*.onnx"))
        print(f"  {d.name:9} {size / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
