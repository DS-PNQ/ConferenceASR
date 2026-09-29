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

import torch

log = logging.getLogger("conf.mt")

# Full language names Hy-MT2 expects in prompts
FULL = {
    "vi": "Vietnamese", "en": "English", "zh": "Chinese",
    "fr": "French", "de": "German", "ja": "Japanese", "ko": "Korean",
    "es": "Spanish", "pt": "Portuguese", "ru": "Russian", "th": "Thai",
    "ar": "Arabic", "it": "Italian",
}
CODE_FROM_NAME = {v.lower(): k for k, v in FULL.items()}


class _IdCollector:
    """Minimal generate() streamer: collects raw token ids (prompt included).

    The consumer skips the prompt length and decodes the rest incrementally.
    Thread-safe for one producer (generate) + one consumer.
    """

    def __init__(self):
        import threading

        self._ids: list[int] = []
        self._done = False
        self._ev = threading.Event()
        self.saw_any = False

    def put(self, value):
        try:
            import torch

            if torch.is_tensor(value):
                value = value[0].tolist()
            self._ids.extend(int(x) for x in value)
            self.saw_any = True
        except Exception:
            pass
        finally:
            self._ev.set()

    def end(self):
        self._done = True
        self._ev.set()

    def take(self, timeout: float = 60.0) -> list[int] | None:
        """Return all ids collected so far, or None when done and drained."""
        while True:
            if self._ids:
                out = self._ids
                self._ids = []
                if not self._done:
                    self._ev.clear()
                return out
            if self._done:
                return None
            if not self._ev.wait(timeout):
                return None


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

    def _prompt(self, text: str, tgt_full: str, context: list[str] | None = None,
                terms: dict[str, str] | None = None) -> str:
        # History is context-ONLY: fenced first with an explicit do-not-translate
        # line, then the documented instruction + the single segment to translate.
        # (Appending history after the instruction made the model translate the
        # history too and ramble.)
        parts: list[str] = []
        hist = "\n".join(f"- {c}" for c in (context or []) if (c or "").strip())
        term_lines = "\n".join(f"`{s}` translates to `{t}`"
                               for s, t in (terms or {}).items() if s and t)
        if term_lines:
            parts.append(
                "Reference the following translations:\n" + term_lines
            )
        if hist:
            parts.append(
                "Conversation context for terminology consistency only. "
                f"DO NOT translate this part:\n{hist}"
            )
        parts.append(
            f"Translate ONLY the following text into {tgt_full}. "
            f"Note that you should only output the translated result without any additional explanation:\n\n{text}"
        )
        return "\n\n".join(parts)

    def _prepare(self, text: str, tgt: str, context: list[str] | None = None,
                 terms: dict[str, str] | None = None):
        """Tokenize prompt -> (input_ids, attn_mask). Caller must hold _lock."""
        import torch

        tgt_full = to_full(tgt)
        messages = [{"role": "user", "content": self._prompt(text, tgt_full, context, terms)}]
        enc = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        # apply_chat_template with return_tensors="pt" yields a BatchEncoding,
        # not a tensor — extract input_ids before .to() / generate().
        input_ids = enc["input_ids"] if not torch.is_tensor(enc) else enc
        input_ids = input_ids.to(self._model.device)
        return input_ids, torch.ones_like(input_ids)

    def _gen_kwargs(self, streamer=None, speculate: bool = False) -> dict:
        greedy = not bool(self.cfg.get("mt_do_sample", False))
        kw: dict = dict(
            max_new_tokens=int(self.cfg.get("mt_max_new_tokens", 256)),
            repetition_penalty=1.05,
            do_sample=not greedy,
            pad_token_id=self._tok.eos_token_id,
        )
        # The repo config ships use_cache=false (slow recompute decode).
        # Override to true unless explicitly disabled — standard
        # transformers generate honors it when the modeling supports cache.
        if bool(self.cfg.get("mt_use_cache", True)):
            kw["use_cache"] = True
        if not greedy:  # sampling hyperparams only matter off-greedy
            kw.update(temperature=float(self.cfg.get("mt_temperature", 0.3)),
                      top_p=0.6, top_k=20)
        # Context-aware speculative decoding: prompt-lookup drafting needs no
        # drafter model — candidate n-grams come from the prompt itself.
        # Measured: ~1.6x on same-language output (en->en), ~0.6x (slower!)
        # cross-lingual (drafts are English, output is Chinese). So: on when
        # configured, or auto when src and tgt are the same language.
        if greedy and speculate:
            kw["prompt_lookup_num_tokens"] = int(self.cfg.get("mt_lookup_tokens", 10))
        if streamer is not None:
            kw["streamer"] = streamer
        return kw

    def _should_speculate(self, src: str, tgt: str, text: str = "") -> bool:
        if bool(self.cfg.get("mt_speculative", False)):
            return True
        try:
            if bool(src) and src.strip().lower() != "auto" and to_full(src) == to_full(tgt):
                return True
            # No LID on the Zipformer side (src is usually "auto"): fall back to
            # a script heuristic — ASCII text rendered into English is almost
            # always English already, where prompt n-grams draft well (2.8x).
            if (tgt or "").strip().lower() in ("en", "english") and text.isascii():
                return True
        except Exception:
            pass
        return False

    def _generate(self, input_ids, attn_mask, gen_kwargs: dict):
        """generate() with graceful fallback if the modeling rejects a kwarg
        (e.g. prompt_lookup on custom modeling). Retries stripped-down."""
        try:
            with torch.no_grad():
                return self._model.generate(
                    input_ids, attention_mask=attn_mask, **gen_kwargs)
        except TypeError as e:
            for key in ("prompt_lookup_num_tokens", "streamer"):
                gen_kwargs.pop(key, None)
            log.warning("generate kwarg unsupported (%s) — retrying plain", e)
            with torch.no_grad():
                return self._model.generate(
                    input_ids, attention_mask=attn_mask, **gen_kwargs)

    def translate(self, text: str, tgt: str = "en", src: str = "auto",
                  context: list[str] | None = None,
                  terms: dict[str, str] | None = None) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        self.ensure_loaded()
        with self._lock:
            if self._mock:
                return self._model.translate(text, src=src, tgt=tgt)
            return self._translate_locked(text, tgt, context, terms,
                                          self._should_speculate(src, tgt, text))

    def _translate_locked(self, text: str, tgt: str, context: list[str] | None,
                          terms: dict[str, str] | None, speculate: bool) -> str:
        """Plain one-shot generation. Caller MUST hold _lock."""
        input_ids, attn_mask = self._prepare(text, tgt, context, terms)
        out = self._generate(input_ids, attn_mask,
                             self._gen_kwargs(speculate=speculate))
        gen = out[0][input_ids.shape[-1]:]
        return self._tok.decode(gen, skip_special_tokens=True).strip()

    def translate_stream(self, text: str, tgt: str = "en", src: str = "auto",
                         context: list[str] | None = None,
                         terms: dict[str, str] | None = None):
        """Yield translation text deltas as tokens generate.

        Uses an id-collecting streamer (not TextIteratorStreamer: this custom
        modeling echoes prompt tokens through the text streamer, so we collect
        raw ids, skip the prompt length, and decode incrementally ourselves).
        Holds _lock for the whole stream. Falls back to one-shot translate if
        the modeling never calls the streamer.
        """
        text = (text or "").strip()
        if not text:
            return
            yield  # make this a generator in all paths
        self.ensure_loaded()
        if self._mock:
            yield self._model.translate(text, src=src, tgt=tgt)
            return
        import threading

        with self._lock:
            input_ids, attn_mask = self._prepare(text, tgt, context, terms)
            prompt_len = int(input_ids.shape[-1])
            collector = _IdCollector()
            speculate = self._should_speculate(src, tgt, text)
            gen_kwargs = self._gen_kwargs(streamer=collector, speculate=speculate)
            thread = threading.Thread(
                target=self._generate, args=(input_ids, attn_mask, gen_kwargs),
                daemon=True)
            thread.start()
            try:
                shown = ""
                gen_all: list[int] = []  # decoded cumulatively: BPE merges
                consumed = 0  # total ids seen (prompt_len skipped once overall)
                while True:
                    ids = collector.take(timeout=2.0) or []
                    if ids:
                        gen_all += ids[max(0, prompt_len - consumed):]
                        consumed += len(ids)
                        new_text = self._tok.decode(gen_all, skip_special_tokens=True)
                        if len(new_text) > len(shown):
                            yield new_text[len(shown):]
                            shown = new_text
                    if not thread.is_alive():
                        break
            finally:
                thread.join(timeout=10)
            if not shown.strip():
                # Modeling never streamed (or streamed nothing usable):
                # one-shot fallback WITHOUT re-locking (lock already held).
                yield self._translate_locked(text, tgt, context, terms, speculate)

    def translate_multi(self, text: str, targets: list[str], src: str = "auto",
                        context: list[str] | None = None,
                        terms: dict[str, str] | None = None) -> dict:
        return {t: self.translate(text, tgt=t, src=src, context=context, terms=terms)
                for t in targets}

    def status(self) -> dict:
        return {
            "model": self.model_id if not self._mock else "mock-mt-demo",
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "ready": self.loaded,
        }
