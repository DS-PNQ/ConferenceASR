"""GPU ASR: drive the streaming Zipformer via ONNX Runtime CUDA directly.

sherpa-onnx wheels are CPU-only builds, so the encoder/decoder/joiner are run
here with onnxruntime(-gpu) + CUDAExecutionProvider, mirroring sherpa's
streaming protocol exactly (validated byte-identical vs sherpa CPU):

  16 kHz mono -> 80-dim fbank (25 ms / 10 ms, dither 0) -> non-overlapping
  39-frame chunks (+ zero-padded tail) through the stateful encoder ->
  stateless-transducer greedy decode per encoder frame -> BPE merge.

Role in the app: live partials keep coming from the fast CPU streaming
recognizer; finalized segments are re-decoded here on GPU (like the denoise
re-decode pattern). One-shot per call => no cross-call state to manage.

Requires onnxruntime-gpu with a CUDA EP compatible with local CUDA libs
(v1.22 ~= CUDA 12, works with torch's vendored DLLs; v1.30 wants CUDA 13).
Raises CudaUnavailable when CUDA EP can't be constructed.
"""
from __future__ import annotations

import logging
import os
import threading

import numpy as np

log = logging.getLogger("conf.cudazf")

CHUNK = 39
FEAT = 80


class CudaUnavailable(Exception):
    pass


def _torch_dll_dirs() -> list[str]:
    try:
        import torch

        d = os.path.join(os.path.dirname(torch.__file__), "lib")
        return [d] if os.path.isdir(d) else []
    except Exception:
        return []


class CudaZipformer:
    def __init__(self, model_dir: str, quant: str = "fp32", prefer_cuda: bool = False,
                 threads: int = 2):
        self.model_dir = str(model_dir)
        self.threads = threads
        self.quant = quant  # fp32 only on CUDA (int8 lacks CUDA EP kernels)
        self.prefer_cuda = prefer_cuda
        self._lock = threading.Lock()
        self._sessions = None
        self._tokens: list[str] | None = None
        self.provider = "?"

    def _paths(self) -> dict:
        import pathlib

        d = pathlib.Path(self.model_dir)
        return {
            "encoder": str(d / "encoder-phaseB_s2a.onnx"),
            "decoder": str(d / "decoder-phaseB_s2a.onnx"),
            "joiner": str(d / "joiner-phaseB_s2a.onnx"),
            "tokens": str(d / "tokens.txt"),
        }

    def ensure_loaded(self):
        if self._sessions is not None:
            return
        with self._lock:
            if self._sessions is not None:
                return
            # Order matters on Windows: import torch FIRST so its bundled
            # CUDA/cuDNN DLLs are preloaded into the process, then expose
            # torch/lib to the loader, and only then bring in onnxruntime.
            # (ORT alone often fails EP init even with the dir on PATH.)
            import torch  # noqa: F401

            for d in _torch_dll_dirs():
                try:
                    os.add_dll_directory(d)
                except Exception:
                    pass
            import onnxruntime as ort

            p = self._paths()
            missing = [v for v in p.values() if not os.path.isfile(v)]
            if missing:
                raise CudaUnavailable(f"model files missing: {missing}")
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # MEASURED i7-13620H, 12 s segment while MT generates: ORT's default
            # pool (10 spinning threads, P+E cores) took 13.4 s; 2 threads 1.2 s
            # (4 swung 1.0-5.4 s). Leave cores for MT, streaming ASR and the OS.
            opts.intra_op_num_threads = max(1, int(self.threads))
            # MEASURED: CUDA EP on these builds either crashes mid-inference
            # (garbage Expand dims) or falls back per-op with memcpy shuttling
            # that turns a segment into 30-40 s. CPU int8/fp32 does 15 s audio
            # in ~1 s. So CPU unless explicitly asked (and probe-validated).
            cands = ([["CUDAExecutionProvider", "CPUExecutionProvider"],
                      ["CPUExecutionProvider"]] if self.prefer_cuda
                     else [["CPUExecutionProvider"]])
            last_err: Exception | None = None
            for providers in cands:
                try:
                    sessions = {
                        "enc": ort.InferenceSession(
                            p["encoder"], sess_options=opts, providers=providers),
                        "dec": ort.InferenceSession(
                            p["decoder"], sess_options=opts, providers=providers),
                        "joi": ort.InferenceSession(
                            p["joiner"], sess_options=opts, providers=providers),
                    }
                    self._probe(sessions)
                    self._sessions = sessions
                    self.provider = sessions["enc"].get_providers()[0]
                    break
                except Exception as e:
                    last_err = e
                    log.warning("ORT providers %s rejected (%s)", providers, e)
            if self._sessions is None:
                raise CudaUnavailable(f"no working provider: {last_err}")
            self._tokens = self._load_tokens(p["tokens"])
            log.info("ORT Zipformer ready on %s (ORT %s)",
                     self.provider, ort.__version__)

    @staticmethod
    def _probe(sessions: dict) -> None:
        """Throw unless encoder + one decode step actually execute."""
        import numpy as np

        enc = sessions["enc"]
        feed = {"x": np.zeros((1, 39, 80), dtype=np.float32)}
        for i in enc.get_inputs():
            if i.name not in feed:
                shp = [1 if isinstance(x, str) else x for x in i.shape]
                feed[i.name] = np.zeros(
                    shp, dtype=np.int64 if "int64" in i.type else np.float32)
        outs = enc.run(None, feed)
        out = dict(zip([o.name for o in enc.get_outputs()], outs))
        enc_frame = out["encoder_out"][0, 0]
        dec = sessions["dec"]
        d_out = dec.run(None, {"y": np.zeros((1, 2), dtype=np.int64)})[0]
        sessions["joi"].run(None, {
            "encoder_out": enc_frame[None, :].astype(np.float32),
            "decoder_out": d_out.astype(np.float32)})

    @staticmethod
    def _load_tokens(path: str) -> list[str]:
        table: dict[int, str] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    sym, idx = line.rsplit(" ", 1)
                    table[int(idx)] = sym
        return [table[i] for i in range(len(table))]

    @staticmethod
    def _features(pcm: np.ndarray) -> np.ndarray:
        import torch
        from torchaudio.compliance.kaldi import fbank

        wav = torch.from_numpy(np.asarray(pcm, dtype=np.float32))
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        feats = fbank(wav, num_mel_bins=FEAT, frame_length=25.0,
                      frame_shift=10.0, dither=0.0, energy_floor=0.0,
                      window_type="povey", preemphasis_coefficient=0.97,
                      use_energy=False)
        return feats.numpy().astype(np.float32)

    def transcribe_array(self, pcm: np.ndarray, sr: int = 16000) -> str:
        """One-shot GPU decode of a mono segment. Returns text (may be '')."""
        self.ensure_loaded()
        x = np.asarray(pcm, dtype=np.float32).ravel()
        if x.size == 0:
            return ""
        if int(sr) != 16000:
            ratio = 16000 / float(sr)
            idx = (np.arange(int(len(x) * ratio)) / ratio).astype(int)
            x = x[np.clip(idx, 0, len(x) - 1)]
        feats = self._features(np.clip(x, -1.0, 1.0))
        with self._lock:
            enc_frames = self._encode(feats)
            hyp = self._greedy(enc_frames)
        return "".join(self._tokens[i] for i in hyp).replace("▁", " ").strip()

    def _encode(self, feats: np.ndarray) -> np.ndarray:
        enc = self._sessions["enc"]
        state: dict[str, np.ndarray] = {}
        for i in enc.get_inputs():
            if i.name == "x":
                continue
            shape = [(1 if isinstance(s, str) else s) for s in i.shape]
            dt = np.int64 if "int64" in i.type else np.float32
            state[i.name] = np.zeros(shape, dtype=dt)
        outs_all = []
        n = (len(feats) + CHUNK - 1) // CHUNK
        for c in range(n):
            blk = feats[c * CHUNK:(c + 1) * CHUNK]
            if len(blk) < CHUNK:
                blk = np.concatenate([blk, np.zeros((CHUNK - len(blk), FEAT), np.float32)])
            feed = {"x": blk[None, :, :].astype(np.float32), **state}
            names = [o.name for o in enc.get_outputs()]
            out = dict(zip(names, enc.run(None, feed)))
            outs_all.append(out["encoder_out"][0])
            for k in list(state):
                state[k] = out["new_" + k]
        return np.concatenate(outs_all, axis=0)

    def _greedy(self, enc_all: np.ndarray) -> list[int]:
        dec, joi = self._sessions["dec"], self._sessions["joi"]
        ctx = [0, 0]
        hyp: list[int] = []
        for f in enc_all:
            for _ in range(5):
                d_out = dec.run(None, {"y": np.array([ctx], dtype=np.int64)})[0]
                j_out = joi.run(
                    None, {"encoder_out": f[None, :].astype(np.float32),
                           "decoder_out": d_out.astype(np.float32)})[0]
                y = int(np.argmax(j_out[0]))
                if y == 0:
                    break
                hyp.append(y)
                ctx = [ctx[1], y]
        return hyp
