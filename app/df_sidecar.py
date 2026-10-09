"""DeepFilterNet worker for a Python that has `deepfilternet` (3.8-3.11).

The backend runs on 3.12 (triton/CUDA stack), where deepfilterlib has no
wheel, so enhancer.py spawns this script under e.g. ../dfenv (Python 3.11).
Stand-alone on purpose: imports nothing from `app`.

Protocol (binary, stdin/stdout): uint32 n + n float32 samples @16 kHz in,
the same shape back. Prints b"READY\n" once the model is loaded.
"""
import struct
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchaudio.functional as AF  # noqa: E402
from df.enhance import enhance, init_df  # noqa: E402


def main():
    atten = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "none" else None
    model, state, *_ = init_df(log_level="ERROR", log_file=None)
    sr = state.sr()
    torch.set_num_threads(4)  # leave cores for ASR + MT
    out, inp = sys.stdout.buffer, sys.stdin.buffer
    out.write(b"READY\n")
    out.flush()
    while True:
        head = inp.read(4)
        if len(head) < 4:
            return  # parent closed the pipe
        (n,) = struct.unpack("<I", head)
        x = torch.from_numpy(np.frombuffer(inp.read(n * 4), dtype=np.float32).copy())
        with torch.no_grad():
            y = enhance(model, state, AF.resample(x, 16000, sr)[None], pad=True, atten_lim_db=atten)
            y = AF.resample(y[0], sr, 16000)[:n]
        y = np.pad(y.numpy().astype(np.float32), (0, n - y.shape[0]))
        out.write(struct.pack("<I", n) + y.tobytes())
        out.flush()


if __name__ == "__main__":
    main()
