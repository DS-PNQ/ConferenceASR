"""Export Titanet embedding path to ONNX (run once; output is gitignored)."""
import torch
import torch.nn as nn
from nemo.collections.asr.models import EncDecSpeakerLabelModel


class TitaWrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, audio, length):
        _, emb = self.m.forward(input_signal=audio, input_signal_length=length)
        return emb


m = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
m.eval().cuda()
wrap = TitaWrap(m).cuda().eval()

sr = 16000
dummy = torch.randn(1, sr * 6, device="cuda")  # 6 s exemplar
length = torch.tensor([dummy.shape[1]], device="cuda")
with torch.no_grad():
    ref = wrap(dummy, length)
print("ref shape:", tuple(ref.shape), flush=True)

import os
os.makedirs("models", exist_ok=True)
out = "models/titanet-embed.onnx"
torch.onnx.export(
    wrap, (dummy, length), out,
    input_names=["audio", "length"], output_names=["emb"],
    dynamic_axes={"audio": {0: "batch", 1: "time"}},
    opset_version=17, do_constant_folding=True, dynamo=False,
)
print("exported", out, flush=True)

# parity: ORT CUDA vs torch
import onnxruntime as ort
sess = ort.InferenceSession(out, providers=["CUDAExecutionProvider"])
print("providers:", sess.get_providers(), flush=True)
import numpy as np
got = sess.run(None, {"audio": dummy.cpu().numpy().astype(np.float32),
                      "length": np.array([dummy.shape[1]], dtype=np.int64)})[0]
cos = float((got.ravel() / np.linalg.norm(got.ravel()) @
             (ref.cpu().numpy().ravel() / np.linalg.norm(ref.cpu().numpy().ravel()))))
print(f"cos(ort, torch) = {cos:.5f}", flush=True)
assert cos > 0.999, cos
print("EXPORT PARITY OK", flush=True)
