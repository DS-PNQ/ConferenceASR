"""Tencent Hy-MT2 translation engine (https://huggingface.co/collections/tencent/hy-mt2).

Default: tencent/Hy-MT2-1.8B — fits RTX 4060 8 GB in bf16.
CUDA + CPU fall-back with the same pattern as ASR: try requested device,
retry CPU float32 on OOM, else clearly-labelled mock mode.

Hy-MT2 is a chat-style causal LM: we wrap the source in its documented
instruction template and generate only the translation.
"""
from __future__ import annotations

import logging
import os
import threading

log = logging.getLogger("conf.mt")

# Full language names Hy-MT2 expects in prompts
FULL = {
    "vi": "Vietnamese", "en": "English", "zh": "Chinese",
    "fr": "French", "de": "German", "ja": "Japanese", "ko": "Korean",
    "es": "Spanish", "pt": "Portuguese", "ru": "Russian", "th": "Thai",
    "ar": "Arabic", "it": "Italian",
}
CODE_FROM_NAME = {v.lower(): k for k, v in FULL.items()}


def to_full(code_or_name: str) -> str:
    s = (code_or_name or "en").strip()
    if len(s) <= 4 and s.lower() in FULL:
        return FULL[s.lower()]
    cap = s.capitalize()
    return cap if cap in FULL.values() else s


class MockMT:
    name = "mock-mt-demo"

    def translate(self, text, src="auto", tgt="en"):
        return f"[{tgt} demo] {text}"


class HyMT2Engine:
    def __init__(self, cfg: dict, device: str, dtype):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.model_id: str = os.getenv("MT_MODEL", cfg.get("mt_model", "tencent/Hy-MT2-1.8B"))
        self._lock = threading.Lock()
        self._tok = None
        self._model = None
        self._mock = False

    @property
    def loaded(self):
        return self._model is not None or self._mock

    def ensure_loaded(self, demo_ok: bool = True):
        if self._model is not None or self._mock:
            return
        with self._lock:
            if self._model is not None or self._mock:
                return
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                quant = "fp8" in self.model_id.lower()
                log.info("Loading MT %s on %s (%s%s)", self.model_id, self.device,
                         self.dtype, ", FP8-quantized" if quant else "")
                self._tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
                load_kw: dict = dict(
                    device_map="auto" if self.device == "cuda" else "cpu",
                    trust_remote_code=True,
                )
                if not quant:
                    # compressed-tensors FP8 repos carry their own quantization
                    # config — passing dtype would fight it, so only set it
                    # for regular (bf16/fp32) checkpoints.
                    load_kw["dtype"] = self.dtype
                self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kw)
                self._model.eval()
                log.info("MT ready: %s", self.model_id)
            except Exception as e:
                log.warning("MT load failed (%s): %s", self.model_id, e)
                if self.device == "cuda" and self.cfg.get("cuda_fallback_on_oom", True):
                    try:
                        import torch

                        torch.cuda.empty_cache()
                        from transformers import AutoModelForCausalLM, AutoTokenizer

                        log.info("Retrying MT on CPU (float32)")
                        self.device = "cpu"
                        self.dtype = torch.float32
                        self._tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
                        retry_kw: dict = dict(device_map="cpu", trust_remote_code=True)
                        if "fp8" not in self.model_id.lower():
                            retry_kw["dtype"] = self.dtype
                        self._model = AutoModelForCausalLM.from_pretrained(
                            self.model_id, **retry_kw
                        )
                        self._model.eval()
                        log.info("MT ready on CPU fallback")
                        return
                    except Exception as e2:
                        log.warning("MT CPU fallback failed: %s", e2)
                if demo_ok:
                    log.warning("Using MockMT demo mode.")
                    self._mock = True
                    self._model = MockMT()
                else:
                    raise

    def _prompt(self, text: str, tgt_full: str) -> str:
        return (
            f"Translate the following text into {tgt_full}. "
            f"Note that you should only output the translated result without any additional explanation:\n\n{text}"
        )

    def translate(self, text: str, tgt: str = "en", src: str = "auto") -> str:
        text = (text or "").strip()
        if not text:
            return ""
        self.ensure_loaded()
        tgt_full = to_full(tgt)
        with self._lock:
            if self._mock:
                return self._model.translate(text, src=src, tgt=tgt)
            import torch

            messages = [{"role": "user", "content": self._prompt(text, tgt_full)}]
            enc = self._tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            # apply_chat_template with return_tensors="pt" yields a BatchEncoding,
            # not a tensor — extract input_ids before .to() / generate().
            input_ids = enc["input_ids"] if not torch.is_tensor(enc) else enc
            input_ids = input_ids.to(self._model.device)
            attn_mask = torch.ones_like(input_ids)
            greedy = not bool(self.cfg.get("mt_do_sample", False))
            gen_kwargs = dict(
                max_new_tokens=int(self.cfg.get("mt_max_new_tokens", 256)),
                repetition_penalty=1.05,
                do_sample=not greedy,
                pad_token_id=self._tok.eos_token_id,
            )
            # The repo config ships use_cache=false (slow recompute decode).
            # Override to true unless explicitly disabled — standard
            # transformers generate honors it when the modeling supports cache.
            if bool(self.cfg.get("mt_use_cache", True)):
                gen_kwargs["use_cache"] = True
            if not greedy:  # sampling hyperparams only matter off-greedy
                gen_kwargs.update(temperature=float(self.cfg.get("mt_temperature", 0.3)),
                                  top_p=0.6, top_k=20)
            with torch.no_grad():
                out = self._model.generate(
                    input_ids,
                    attention_mask=attn_mask,
                    **gen_kwargs,
                )
            gen = out[0][input_ids.shape[-1]:]
            return self._tok.decode(gen, skip_special_tokens=True).strip()

    def translate_multi(self, text: str, targets: list[str], src: str = "auto") -> dict:
        return {t: self.translate(text, tgt=t, src=src) for t in targets}

    def status(self) -> dict:
        return {
            "model": self.model_id if not self._mock else "mock-mt-demo",
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "ready": self.loaded,
        }
