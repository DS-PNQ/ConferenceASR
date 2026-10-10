"""Document OCR with PaddleOCR-VL (https://huggingface.co/PaddlePaddle/PaddleOCR-VL).

The 0.9B ultra-compact VLM is the lightest PaddleOCR-VL available (~1B params
BF16, ~2 GB VRAM) and runs through transformers with trust_remote_code — no
paddlepaddle-gpu install needed. Element tasks: ocr | table | formula | chart.

Lazy like every other engine: loads on first /api/ocr call, never in warmup.
Calling .unload() drops it from VRAM. OCR mode deloads the speaker diarizer
first (see main._enter_ocr_mode), so the 8 GB card holds OCR + Hy-MT only.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path

log = logging.getLogger("conf.ocr")

TASKS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}


class MockOCR:
    name = "mock-ocr-demo"

    def read(self, image, task: str = "ocr") -> str:
        return f"[ocr demo] task={task}"


_CJK = re.compile(r"[　-〿぀-ヿ㐀-鿿가-힯＀-￯]")


def _crop_quad(img, box):
    """Warp a detected quad flat (PaddleOCR get_rotate_crop_image)."""
    import cv2
    import numpy as np

    pts = np.array(box, dtype=np.float32)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    m = cv2.getPerspectiveTransform(
        pts, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    out = cv2.warpPerspective(img, m, (max(w, 1), max(h, 1)),
                              borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    return np.rot90(out) if h and w and h / w >= 1.5 else out


class VietRec:
    """VietOCR vgg_seq2seq ONNX line reader (ocr_test/vietocr_onnx.py twin).

    PP-OCR rec dicts lack most Vietnamese letters; VietOCR reads the full
    alphabet + Latin/digits. Model dir = encoder.onnx, decoder.onnx,
    vocab.txt from scripts/get_vietocr.py. Tokens: 1 sos, 2 eos, 4.. vocab.
    """

    def __init__(self, model_dir: str, threads: int = 4):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        prov = ["CPUExecutionProvider"]
        d = Path(model_dir)
        self.enc = ort.InferenceSession(str(d / "encoder.onnx"), so, providers=prov)
        self.dec = ort.InferenceSession(str(d / "decoder.onnx"), so, providers=prov)
        self.vocab = (d / "vocab.txt").read_text(encoding="utf-8").split("\n")

    def read(self, crop_rgb) -> tuple[str, float]:
        import numpy as np
        from PIL import Image

        h, w = crop_rgb.shape[:2]
        nw = min(max(int(np.ceil(int(32 * w / max(h, 1)) / 10) * 10), 32), 512)
        # MEASURED ocr_test/README: LANCZOS is what it was trained on;
        # bilinear/area cost 6-7% CER
        x = np.asarray(Image.fromarray(np.ascontiguousarray(crop_rgb))
                       .resize((nw, 32), Image.LANCZOS), dtype=np.float32)
        enc_out, hidden = self.enc.run(None, {"image": (x / 255.0).transpose(2, 0, 1)[None]})
        tok, chars, probs = 1, [], []
        for _ in range(128):
            logits, hidden = self.dec.run(None, {"token": np.array([tok], np.int64),
                                                 "hidden": hidden, "enc_out": enc_out})
            row = logits[0]
            tok = int(row.argmax())
            if tok == 2:
                break
            if tok > 3:
                e = np.exp(row - row.max())
                chars.append(self.vocab[tok - 4])
                probs.append(float(e[tok] / e.sum()))
        return "".join(chars), (float(np.mean(probs)) if probs else 0.0)


class BlockOCR:
    """PP-OCR via rapidocr_onnxruntime: real boxes + text + scores on CPU.

    No paddle dependency (paddle 3.3 CPU is broken on this machine's
    executor); ONNX models run through the already-installed ORT.
    With `vi_ocr_model` present, non-CJK lines are re-read by VietRec so
    Vietnamese keeps its diacritics (and its boxes).
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._engine = None
        self._failed = False
        self._viet = None

    @property
    def has_viet(self) -> bool:
        d = self.cfg.get("vi_ocr_model") or ""
        return bool(d) and (Path(d) / "encoder.onnx").exists()

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    @property
    def available(self) -> bool:
        return not self._failed

    def ensure_loaded(self):
        if self._engine is not None or self._failed:
            return
        with self._lock:
            if self._engine is not None or self._failed:
                return
            try:
                from rapidocr_onnxruntime import RapidOCR

                log.info("Loading RapidOCR (PP-OCR ONNX, CPU)")
                self._engine = RapidOCR()
                log.info("RapidOCR ready")
            except Exception as e:
                log.warning("RapidOCR unavailable (%s)", e)
                self._failed = True
                return
            if self.has_viet:
                try:
                    self._viet = VietRec(self.cfg["vi_ocr_model"])
                    log.info("VietOCR line reader ready (%s)", self.cfg["vi_ocr_model"])
                except Exception as e:
                    log.warning("VietOCR unavailable (%s) — PP-OCR text only", e)

    def unload(self) -> dict:
        """Drop PP-OCR + VietOCR sessions (CPU RAM). Next read reloads lazily."""
        with self._lock:
            was = self.loaded
            self._engine = self._viet = None
        if was:
            import gc

            gc.collect()
            log.info("Block OCR unloaded")
        return {"unloaded": was}

    def read_blocks(self, image) -> dict:
        """-> {text, blocks: [{id, text, conf, box:[x0,y0,x1,y1] rel}], overall_conf}."""
        import numpy as np

        self.ensure_loaded()
        if self._engine is None:
            raise RuntimeError("block OCR unavailable")
        img = image.convert("RGB")
        w, h = img.size
        arr = np.asarray(img)
        blocks = []
        with self._lock:
            out, _ = self._engine(arr)
            for i, (box, text, conf) in enumerate(out or []):
                text, conf, rec = str(text or ""), float(conf or 0.0), "pp-ocr"
                if self._viet is not None and not _CJK.search(text):
                    try:
                        text, conf = self._viet.read(_crop_quad(arr, box))
                        rec = "vietocr"
                    except Exception as e:
                        log.warning("VietOCR line failed (%s), keeping PP-OCR", e)
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                blocks.append({
                    "id": i + 1,
                    "text": text,
                    "conf": round(conf, 4),
                    "rec": rec,
                    "box": [min(xs) / w, min(ys) / h,
                            max(xs) / w, max(ys) / h],
                })
        confs = [b["conf"] for b in blocks]
        overall = round(sum(confs) / len(confs), 4) if confs else 0.0
        return {
            "text": "\n".join(b["text"] for b in blocks),
            "blocks": blocks,
            "overall_conf": overall,
        }


def _patch_causal_mask_alias():
    """Shim for PaddleOCR-VL's custom modeling vs installed transformers.

    The model's code calls create_causal_mask(..., inputs_embeds=...) while
    installed transformers renamed the kwarg to input_embeds (singular).
    Patch both the transformers namespace and the already-imported dynamic
    modeling module (which bound the function by name at import).
    """
    try:
        import sys

        import transformers.masking_utils as _mu

        orig = _mu.create_causal_mask

        def _shim(*args, **kwargs):
            if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
                kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
            return orig(*args, **kwargs)

        _mu.create_causal_mask = _shim
        for name, mod in list(sys.modules.items()):
            if "paddleocr" in name.lower() and hasattr(mod, "create_causal_mask"):
                try:
                    setattr(mod, "create_causal_mask", _shim)
                except Exception:
                    pass
    except Exception as e:
        log.warning("causal-mask shim failed (%s) — OCR may still work", e)


class PaddleOCRVLEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_id: str = os.getenv(
            "OCR_MODEL", cfg.get("ocr_model", "PaddlePaddle/PaddleOCR-VL"))
        self.device = self._resolve_device()
        self._lock = threading.Lock()
        self._model = None
        self._proc = None
        self._mock = False

    def _resolve_device(self) -> str:
        pref = str(os.getenv("OCR_DEVICE", self.cfg.get("ocr_device", "auto"))).lower()
        if pref in ("cuda", "gpu"):
            return "cuda"
        if pref == "cpu":
            return "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    @property
    def loaded(self) -> bool:
        return self._model is not None or self._mock

    def ensure_loaded(self, demo_ok: bool = True):
        if self.loaded:
            return
        with self._lock:
            if self.loaded:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoProcessor

                _patch_causal_mask_alias()
                dtype = (torch.bfloat16 if self.device == "cuda"
                         else torch.float32)
                log.info("Loading OCR %s on %s (%s)", self.model_id,
                         self.device, dtype)
                self._proc = AutoProcessor.from_pretrained(
                    self.model_id, trust_remote_code=True)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, trust_remote_code=True,
                    dtype=dtype,
                    device_map="auto" if self.device == "cuda" else "cpu",
                )
                self._model.eval()
                _patch_causal_mask_alias()  # again: dynamic module loaded above
                log.info("OCR ready: %s", self.model_id)
            except Exception as e:
                log.warning("OCR load failed (%s): %s", self.model_id, e)
                if self.device == "cuda":
                    try:
                        import torch
                        from transformers import (AutoModelForCausalLM,
                                                  AutoProcessor)

                        torch.cuda.empty_cache()
                        log.info("Retrying OCR on CPU (float32)")
                        self.device = "cpu"
                        self._proc = AutoProcessor.from_pretrained(
                            self.model_id, trust_remote_code=True)
                        self._model = AutoModelForCausalLM.from_pretrained(
                            self.model_id, trust_remote_code=True,
                            dtype=torch.float32, device_map="cpu")
                        self._model.eval()
                        log.info("OCR ready on CPU fallback")
                        return
                    except Exception as e2:
                        log.warning("OCR CPU fallback failed: %s", e2)
                if demo_ok:
                    log.warning("Using MockOCR demo mode.")
                    self._mock = True
                    self._model = MockOCR()
                else:
                    raise

    def unload(self) -> dict:
        """Drop the VLM from VRAM. Next read() reloads lazily."""
        with self._lock:
            was = self.status()
            self._model = None
            self._proc = None
            self._mock = False
            try:
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        log.info("OCR unloaded (was %s)", was.get("device"))
        return {"unloaded": True, "was": was}

    def read(self, image, task: str = "ocr") -> str:
        """Recognize one PIL image. Returns the decoded element text."""
        task = (task or "ocr").strip().lower()
        if task not in TASKS:
            task = "ocr"
        self.ensure_loaded()
        with self._lock:
            if self._mock:
                return self._model.read(image, task)
            import torch

            img = image.convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": TASKS[task]},
            ]}]
            inputs = self._proc.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt")
            dev = getattr(self._model, "device", None)
            if dev is None:
                try:
                    dev = next(self._model.parameters()).device
                except Exception:
                    dev = torch.device(self.device)
            try:
                inputs = {k: v.to(dev) if torch.is_tensor(v) else v
                          for k, v in inputs.items()}
            except Exception:
                pass
            prompt_len = int(inputs["input_ids"].shape[-1])
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=int(self.cfg.get("ocr_max_new_tokens", 1024)),
                    do_sample=False,
                    use_cache=True,
                    # MEASURED ocr_test/samples: vi_sign looped "00 00 …" to the
                    # 1024-token cap (22 s); 8 stops it (1 s), en/zh unchanged
                    no_repeat_ngram_size=8,
                )
            gen = out[0][prompt_len:]
            return self._proc.decode(gen, skip_special_tokens=True).strip()

    def status(self) -> dict:
        dev = self.device
        if self._model is not None and not self._mock:
            try:
                dev = str(next(self._model.parameters()).device)
            except Exception:
                pass
        return {
            "model": self.model_id if not self._mock else "mock-ocr-demo",
            "device": dev,
            "ready": self.loaded,
        }
