"""Local HY-MT INT8 ONNX engine (subclass of HyMT2Engine).

Drives D:/DENSEV2 - reading/HY-MT/model_int8_final.onnx directly with
onnxruntime: standard past_tate KV-cache IO (input_ids, attention_mask,
past_{i}.key/value -> logits, present_*). Greedy decode loop in Python means
TRUE per-token streaming (each step yields) instead of waiting on a full
transformers generate().

Provider: CUDA EP first (torch imported first so its bundled CUDA DLLs
preload, same lesson as app/cuda_zipformer.py), CPU EP fallback. INT8
kernels are CPU-friendly; the pick is reported, never silent.

Only model loading + generation are overridden — prompts, terms, context,
batching orchestration, streaming fan-out and status plumbing all reuse the
parent implementation.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

from .mt_engine import HyMT2Engine

log = logging.getLogger("conf.onnxmt")

_FALLBACK_TOK = "tencent/Hy-MT2-1.8B-FP8"
_EOS_FALLBACK = 120020
_PAD_FALLBACK = 120002


class OnnxMTEngine(HyMT2Engine):
    backend = "onnx"
    # This export is batch-1 (padded batches break an internal reshape);
    # sequential per-target streaming is the path (still true per-token
    # streaming via the native loop).
    _supports_batch = False

    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        self.model_dir = Path(os.getenv("MT_MODEL", self.cfg.get("mt_model", "")))
        self.model_id = str(self.model_dir)
        self.device = "cpu"  # refined to cuda when the EP constructs
        self.dtype = "onnx-int8"
        self._lock = threading.RLock()
        self._tok = None
        self._model = None  # the ORT session (truthy => loaded)
        self._sess = None
        self._past_names: list[str] = []
        self._eos = _EOS_FALLBACK
        self._pad = _PAD_FALLBACK
        self._mock = False

    # -- loading ------------------------------------------------------------
    def _find_onnx(self) -> str:
        files = sorted(self.model_dir.glob("*.onnx"))
        if not files:
            raise FileNotFoundError(f"no .onnx in {self.model_dir}")
        # prefer an explicitly final/quantized artifact
        for f in files:
            if "final" in f.name.lower() or "int8" in f.name.lower():
                return str(f)
        return str(files[0])

    def ensure_loaded(self, demo_ok: bool = True):
        if self._model is not None or self._mock:
            return
        with self._lock:
            if self._model is not None or self._mock:
                return
            try:
                import torch  # first: preload bundled CUDA DLLs (see module doc)

                for d in (os.path.join(os.path.dirname(torch.__file__), "lib"),):
                    try:
                        os.add_dll_directory(d)
                    except Exception:
                        pass
                from transformers import AutoTokenizer

                try:
                    self._tok = AutoTokenizer.from_pretrained(
                        str(self.model_dir), trust_remote_code=True)
                except Exception as e:
                    log.warning("local tokenizer failed (%s), trying %s", e, _FALLBACK_TOK)
                    self._tok = None
                if self._tok is None or not getattr(self._tok, "chat_template", None):
                    ref = AutoTokenizer.from_pretrained(
                        _FALLBACK_TOK, trust_remote_code=True)
                    if self._tok is None:
                        self._tok = ref
                    else:
                        self._tok.chat_template = ref.chat_template
                self._eos = self._tok.eos_token_id or _EOS_FALLBACK
                self._pad = (self._tok.pad_token_id
                             if self._tok.pad_token_id is not None else _PAD_FALLBACK)

                import onnxruntime as ort

                onnx_path = self._find_onnx()
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = None
                try:
                    sess = ort.InferenceSession(
                        onnx_path, sess_options=opts,
                        providers=["CUDAExecutionProvider"])
                    if sess.get_providers()[0] != "CUDAExecutionProvider":
                        raise RuntimeError("CUDA EP fell back")
                    self.device = "cuda"
                except Exception as e:
                    log.warning("ONNX CUDA EP unavailable (%s) — CPU EP.", e)
                    sess = ort.InferenceSession(
                        onnx_path, sess_options=opts,
                        providers=["CPUExecutionProvider"])
                    self.device = "cpu"
                self._past_names = [i.name for i in sess.get_inputs()
                                    if i.name not in ("input_ids", "attention_mask")]
                self._sess = sess
                self._model = sess
                log.info("MT ready: %s (%s, past states: %d)",
                         self.model_dir.name, self.device, len(self._past_names) // 2)
            except Exception as e:
                log.warning("ONNX MT load failed (%s): %s", self.model_dir, e)
                if demo_ok:
                    from .mt_engine import MockMT

                    log.warning("Using MockMT demo mode.")
                    self._mock = True
                    self._model = MockMT()
                else:
                    raise

    # -- generation (native greedy loop; streamer-compatible) ------------------
    def _prepare(self, text: str, tgt: str, context=None, terms=None):
        """Tokenize prompt -> (input_ids [1, S] int64, mask ones). Holds _lock."""
        from .mt_engine import to_full as _to_full

        messages = [{"role": "user",
                     "content": self._prompt(text, _to_full(tgt), context, terms)}]
        enc = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt")
        import torch

        ids = enc["input_ids"] if not torch.is_tensor(enc) else enc
        ids_np = np.asarray(ids.cpu().numpy() if torch.is_tensor(ids) else ids,
                            dtype=np.int64).reshape(1, -1)
        return ids_np, np.ones_like(ids_np, dtype=np.int64)

    def _prepare_batched(self, text: str, tgts: list[str],
                         context=None, terms=None):
        """One prompt per target, left-padded numpy batch. Holds _lock."""
        ids_list, pad = [], self._pad
        for tgt in tgts:
            ids, _ = self._prepare(text, tgt, context, terms)
            ids_list.append(ids[0].tolist())
        max_len = max(len(x) for x in ids_list)
        batch = np.array([[pad] * (max_len - len(x)) + x for x in ids_list],
                         dtype=np.int64)
        mask = np.array([[0] * (max_len - len(x)) + [1] * len(x) for x in ids_list],
                        dtype=np.int64)
        return batch, mask

    def _generate(self, input_ids, attn_mask, gen_kwargs: dict):
        """Batched greedy loop with past cache. Feeds gen_kwargs['streamer']
        per step (single id for B==1, [B] rows otherwise) and end()s it, which
        transformers used to do for us. Returns [B, S+G] like generate().
        Rows stop individually at EOS (unlike the transformers batch path)."""
        import numpy as np

        max_new = int(gen_kwargs.get("max_new_tokens",
                                     int(self.cfg.get("mt_max_new_tokens", 256))))
        rep = float(gen_kwargs.get("repetition_penalty", 1.05))
        streamer = gen_kwargs.get("streamer")
        if bool(gen_kwargs.get("do_sample", False)):
            log.warning("onnx engine is greedy-only; ignoring do_sample")
        sess = self._sess
        ids = np.asarray(input_ids, dtype=np.int64).reshape(
            *np.asarray(input_ids).shape[:1], -1)
        if ids.ndim == 1:
            ids = ids.reshape(1, -1)
        mask = np.asarray(attn_mask, dtype=np.int64).reshape(ids.shape)
        B = ids.shape[0]
        past = {name: np.zeros((B, 4, 0, 128), dtype=np.float32)
                for name in self._past_names}
        out_ids = [row.tolist() for row in ids]
        seen = [set(row) for row in out_ids]
        done = [False] * B
        if streamer is not None:
            # mimic transformers: first put carries the (padded) prompt batch,
            # which collectors skip via prompt length before streaming tokens
            try:
                streamer.put(ids.tolist())
            except Exception:
                pass
        try:
            steps = 0
            while not all(done) and steps < max_new:
                feed = {"input_ids": ids, "attention_mask": mask, **past}
                outs = sess.run(None, feed)
                logits = np.asarray(outs[0]).astype(np.float64)
                if logits.ndim == 3:
                    logits = logits[:, -1, :]
                step_ids = []
                for b in range(B):
                    if done[b]:
                        step_ids.append(self._eos)
                        continue
                    lv = logits[b]
                    if rep != 1.0:
                        for tok_id in seen[b]:
                            if 0 <= tok_id < lv.shape[0]:
                                s = lv[tok_id]
                                lv[tok_id] = s / rep if s > 0 else s * rep
                    nxt = int(np.argmax(lv))
                    if nxt == self._eos:
                        done[b] = True
                    else:
                        out_ids[b].append(nxt)
                        seen[b].add(nxt)
                    step_ids.append(nxt)
                steps += 1
                if streamer is not None:
                    try:
                        streamer.put(step_ids if B > 1 else step_ids[0])
                    except Exception:
                        pass
                present = outs[1:]
                past = {name: np.asarray(arr) for name, arr in
                        zip(self._past_names, present)}
                ids = np.array(step_ids, dtype=np.int64).reshape(B, 1)
                mask = np.concatenate(
                    [mask, np.ones((B, 1), dtype=np.int64)], axis=1)
        finally:
            if streamer is not None:
                try:
                    streamer.end()
                except Exception:
                    pass
        max_len = max(len(r) for r in out_ids)
        return np.array([r + [self._pad] * (max_len - len(r)) for r in out_ids],
                        dtype=np.int64)

    def status(self) -> dict:
        return {
            "model": self.model_id if not self._mock else "mock-mt-demo",
            "backend": f"onnx-{self.device}" if not self._mock else "mock",
            "device": self.device,
            "dtype": str(self.dtype),
            "ready": self.loaded,
        }
