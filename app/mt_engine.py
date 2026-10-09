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
import time

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


def _dequantize_fp8(model) -> int:
    """Bake compressed-tensors FP8 Linears into plain fp16 nn.Linear, once.

    Left as-is, every forward re-dequantizes each weight and fake-quantizes
    its input (~3000 kernel launches per token): decode was CPU launch-bound
    at ~3 tok/s with the GPU mostly idle. Costs ~2x weight VRAM (3.6 GB).
    """
    import torch.nn as nn

    swaps = [(name, mod) for name, mod in model.named_modules()
             if isinstance(mod, nn.Linear) and hasattr(mod, "weight_scale")
             and mod.weight.dtype == torch.float8_e4m3fn]  # still-compressed only
    for name, mod in swaps:
        w = mod.weight.to(torch.float16) * mod.weight_scale.to(torch.float16)
        lin = nn.Linear(mod.in_features, mod.out_features, bias=mod.bias is not None,
                        device=w.device, dtype=torch.float16)
        lin.weight.data.copy_(w)
        if mod.bias is not None:
            lin.bias.data.copy_(mod.bias.to(torch.float16))
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, child, lin)
    if swaps:
        log.info("FP8 -> fp16: %d Linear layers baked", len(swaps))
        # compressed-tensors also wraps EVERY module's forward to re-send its
        # args to the device (~1100 .to() per token, and a graph break per
        # module under torch.compile). Weights are plain + on-device now.
        try:
            from compressed_tensors.offload.dispatch import remove_dispatch

            remove_dispatch(model, onload_tensors=True)
        except ImportError:
            pass
        # ...and its first-forward ct_decompress_hook would re-wrap all of them
        for k, h in list(model._forward_pre_hooks.items()):
            if getattr(getattr(h, "func", h), "__name__", "") == "ct_decompress_hook":
                del model._forward_pre_hooks[k]
        model.hf_quantizer = None  # no longer quantized: lets generate() compile
    return len(swaps)


def _ban_repeat_trigrams(logits_row, history: list[int], n: int = 3):
    """In-place ban of tokens that would repeat an n-gram already seen.

    Argmax decoding without generate()'s warpers can sit in a degenerate
    loop; this is the standard no_repeat_ngram_size guard, done manually.
    """
    if len(history) + 1 < n:
        return
    prefix = tuple(history[-(n - 1):])
    seen: dict[tuple, set[int]] = {}
    for i in range(len(history) - n + 1):
        key = tuple(history[i:i + n - 1])
        seen.setdefault(key, set()).add(history[i + n - 1])
    for tok in seen.get(prefix, ()):
        try:
            logits_row[tok] = float("-inf")
        except Exception:
            pass


class _EventStop:
    """generate() stopping criterion: ends the decode once `ev` is set."""

    def __init__(self, ev: threading.Event):
        self.ev = ev

    def __call__(self, input_ids, scores, **kw):
        return torch.full((input_ids.shape[0],), self.ev.is_set(),
                          dtype=torch.bool, device=input_ids.device)


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
        # generate() calls put() with 1-D tensors ([prompt_len] once, then
        # [batch] per step). Flatten defensively: an earlier version indexed
        # value[0], which silently dropped EVERYTHING on 1-D input (and the
        # bare except hid it — streaming never fired). Never swallow blindly.
        try:
            import torch

            if torch.is_tensor(value):
                value = value.detach().cpu().tolist()
            if isinstance(value, int):
                value = [value]
            flat: list[int] = []

            def _walk(v):
                if isinstance(v, (list, tuple)):
                    for x in v:
                        _walk(x)
                else:
                    flat.append(int(v))

            _walk(value)
            self._ids.extend(flat)
            self.saw_any = True
        except Exception as e:
            log.warning("streamer put failed (%s) — deltas for this call lost", e)
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


class _RowDemux:
    """Demuxes batched generate() put() calls into per-row id streams.

    First put() is the (padded) prompt batch and is skipped; afterwards each
    put() is [B] (or [B, 1]) generated ids. A row is closed at its first EOS
    so padding/repeat ids after finish never leak into its text.
    """

    def __init__(self, n_rows: int, eos_id: int | None):
        import threading

        self.n = n_rows
        self.eos = eos_id
        self.rows: list[list[int]] = [[] for _ in range(n_rows)]
        self.pending: list[list[int]] = [[] for _ in range(n_rows)]
        self.done_rows: list[bool] = [False] * n_rows
        self._primed = False
        self._done = False
        self._ev = threading.Event()
        self.saw_any = False

    def put(self, value):
        try:
            import torch

            if torch.is_tensor(value):
                value = value.detach().cpu().tolist()
            if isinstance(value, int):
                value = [value]
            if (isinstance(value, list) and value
                    and all(isinstance(x, list) for x in value)):
                rows_in = [x[0] if x else None for x in value]
            else:
                rows_in = list(value) if isinstance(value, list) else [value]
            if not self._primed:
                # full prompt batch (padded): skip, start streaming after it
                self._primed = True
            else:
                for i in range(min(self.n, len(rows_in))):
                    if self.done_rows[i]:
                        continue
                    v = rows_in[i]
                    if v is None:
                        continue
                    v = int(v)
                    if self.eos is not None and v == self.eos:
                        self.done_rows[i] = True
                        continue
                    self.pending[i].append(v)
                    self.rows[i].append(v)
                    self.saw_any = True
        except Exception as e:
            log.warning("demux put failed (%s)", e)
        finally:
            self._ev.set()

    def end(self):
        self._done = True
        self._ev.set()

    def take_any(self, timeout: float = 60.0) -> list[tuple[int, list[int]]] | None:
        """Drain newly arrived ids per row; None when done and drained."""
        while True:
            out = [(i, self.pending[i]) for i in range(self.n) if self.pending[i]]
            if out:
                self.pending = [[] for _ in range(self.n)]
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
    _supports_batch = True  # subclass may disable (e.g. batch-1 ONNX export)

    def __init__(self, cfg: dict, device: str, dtype):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.model_id: str = os.getenv("MT_MODEL", cfg.get("mt_model", "tencent/Hy-MT2-1.8B"))
        # RLock: translate_targets_stream holds it across a fallback into
        # translate_stream (same thread) — a plain Lock would deadlock there.
        self._lock = threading.RLock()
        self._tok = None
        self._model = None
        self._mock = False
        self._last_ssbd = {"path": "-", "draft_len": 0, "accepted": 0}
        self._static = None  # StaticCache when the decode step is compiled

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
                if quant:
                    _dequantize_fp8(self._model)
                self._model.eval()
                self._enable_compile()
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
        # Glossary: emit original + UPPER + lower variants — our ASR outputs
        # UPPERCASE English, users type lowercase, and the model otherwise
        # treats them as different words and ignores the term.
        seen: set[str] = set()
        term_lines = []
        for s, t in (terms or {}).items():
            if not s or not t:
                continue
            for variant in (s, s.upper(), s.lower()):
                if variant and variant not in seen:
                    seen.add(variant)
                    term_lines.append(f"`{variant}` translates to `{t}`")
        if term_lines:
            parts.append(
                "You MUST use exactly the following translations for the "
                "matching words (match regardless of letter case):\n"
                + "\n".join(term_lines)
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

    def _gen_kwargs(self, streamer=None, speculate: bool = False,
                    stop: threading.Event | None = None) -> dict:
        greedy = not bool(self.cfg.get("mt_do_sample", False))
        # NOTE: no repetition_penalty — it is a full-vocab (120k) op paid on
        # EVERY decode step and buys nothing for translation output.
        kw: dict = dict(
            max_new_tokens=int(self.cfg.get("mt_max_new_tokens", 128)),
            do_sample=not greedy,
            pad_token_id=self._tok.eos_token_id,
        )
        # The repo config ships use_cache=false (slow recompute decode).
        # Override to true unless explicitly disabled — standard
        # transformers generate honors it when the modeling supports cache.
        if bool(self.cfg.get("mt_use_cache", True)):
            kw["use_cache"] = True
        # Quantized KV cache (KIVI-style, quanto backend): shrinks cache
        # memory, not per-step fixed costs — measured below; default off.
        if bool(self.cfg.get("mt_kvquant", False)):
            kw["cache_implementation"] = "quantized"
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
        if stop is not None:
            from transformers import StoppingCriteriaList

            kw["stopping_criteria"] = StoppingCriteriaList([_EventStop(stop)])
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

    def _enable_compile(self):
        """CUDA-graph the decode step: generate() auto-compiles when handed a
        StaticCache. One fixed-size cache = one graph. Measured RTX 4060,
        same process: 9.6 -> 31.2 tok/s. Needs triton (triton-windows on
        Windows, see run.py); compiles here so warmup pays the ~100 s, not
        the first live segment."""
        if not bool(self.cfg.get("mt_compile", True)) or self.device != "cuda":
            return
        try:
            import triton  # noqa: F401
            from transformers import StaticCache

            self._static = StaticCache(
                config=self._model.config, max_batch_size=1,
                max_cache_len=int(self.cfg.get("mt_cache_len", 1024)),
                device=self._model.device, dtype=self._model.dtype)
            t0 = time.time()
            self._translate_locked("Good morning, everyone.", "zh", None, None, False)
            log.info("MT decode compiled in %.0f s (CUDA graphs)", time.time() - t0)
        except Exception as e:
            log.warning("MT compile unavailable (%s) — eager decode.", e)
            self._static = None

    def _generate(self, input_ids, attn_mask, gen_kwargs: dict):
        """generate() with graceful fallback if the modeling rejects a kwarg
        (e.g. prompt_lookup on custom modeling). Retries stripped-down."""
        static = getattr(self, "_static", None)
        if (static is not None and input_ids.shape[0] == 1
                and input_ids.shape[-1] + gen_kwargs.get("max_new_tokens", 128)
                <= static.max_cache_len):
            static.reset()  # callers hold _lock: one generate at a time
            gen_kwargs = {k: v for k, v in gen_kwargs.items()
                          if k not in ("prompt_lookup_num_tokens", "cache_implementation")}
            gen_kwargs["past_key_values"] = static
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
                  terms: dict[str, str] | None = None,
                  stop: threading.Event | None = None) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        self.ensure_loaded()
        with self._lock:
            if self._mock:
                return self._model.translate(text, src=src, tgt=tgt)
            return self._translate_locked(text, tgt, context, terms,
                                          self._should_speculate(src, tgt, text), stop)

    def _translate_locked(self, text: str, tgt: str, context: list[str] | None,
                           terms: dict[str, str] | None, speculate: bool,
                           stop: threading.Event | None = None) -> str:
        """Plain one-shot generation. Caller MUST hold _lock."""
        input_ids, attn_mask = self._prepare(text, tgt, context, terms)
        out = self._generate(input_ids, attn_mask,
                             self._gen_kwargs(speculate=speculate, stop=stop))
        gen = out[0][input_ids.shape[-1]:]
        return self._tok.decode(gen, skip_special_tokens=True).strip()

    # -- Self-Speculative Biased Decoding (SSBD) --------------------------
    # COLM 2026 (Zoom): reuse the previous output as a draft for the updated
    # input, verify all draft tokens in ONE parallel forward with a bias
    # toward draft tokens, resume autoregressive decode at first divergence.
    # Fits our pipeline exactly: partial_{t+1} drafts off partial_t, and a
    # final drafts off its last partial translation.
    def translate_ssbd(self, text: str, draft_text: str, tgt: str = "en",
                       src: str = "auto", context: list[str] | None = None,
                       terms: dict[str, str] | None = None,
                       stop: threading.Event | None = None) -> dict:
        """SSBD one-shot. Caller MUST hold _lock (or call via translate()).

        Returns {text, draft_len, accepted, beta, path} where path is
        'crop' (KV cropped, no re-prefill), 'reprefill' (prefill accepted
        prefix again), or 'scratch' (fallback = plain generate).
        """
        import torch

        beta = float(self.cfg.get("ssbd_beta", 0.2))
        input_ids, attn_mask = self._prepare(text, tgt, context, terms)
        prompt_len = int(input_ids.shape[-1])
        max_new = int(self.cfg.get("mt_max_new_tokens", 128))
        draft_ids = self._tok(draft_text or "", return_tensors="pt",
                              add_special_tokens=False)["input_ids"].to(
            self._model.device)
        m = int(draft_ids.shape[-1])
        if m == 0:
            out = self._generate(input_ids, attn_mask,
                                 self._gen_kwargs(speculate=False, stop=stop))
            gen = out[0][prompt_len:]
            return {"text": self._tok.decode(gen, skip_special_tokens=True).strip(),
                    "draft_len": 0, "accepted": 0, "beta": beta, "path": "scratch"}

        def _scratch():
            out = self._generate(input_ids, attn_mask,
                                 self._gen_kwargs(speculate=False, stop=stop))
            gen = out[0][prompt_len:]
            return {"text": self._tok.decode(gen, skip_special_tokens=True).strip(),
                    "draft_len": m, "accepted": 0, "beta": beta, "path": "scratch"}

        try:
            with torch.no_grad():
                full = torch.cat([input_ids, draft_ids], dim=-1)
                full_mask = torch.ones_like(full)
                fwd = self._model(input_ids=full, attention_mask=full_mask,
                                  use_cache=self._static is None)
                logits = fwd.logits[0, prompt_len - 1:prompt_len - 1 + m]
                probs = torch.softmax(logits, dim=-1)
                draft = draft_ids[0]
                mixed = probs * (1.0 - beta)
                mixed[torch.arange(m, device=mixed.device), draft] += beta
                pred = mixed.argmax(dim=-1)
                match = (pred == draft)
                d = int(torch.nonzero(~match)[0, 0]) if not bool(match.all()) else m
                past = getattr(fwd, "past_key_values", None)
        except Exception as e:
            log.warning("SSBD verify failed (%s) — scratch fallback", e)
            return _scratch()

        # Resume from divergence: crop the verified KV and decode manually
        # token-by-token (this custom modeling's generate() cannot resume
        # from an externally supplied past — its input preparation indexes
        # an empty cache_position). No second prefill.
        try:
            if self._static is not None:  # re-prefill below runs CUDA-graphed: 3x the eager loop
                raise LookupError("compiled decode")
            if past is None or not hasattr(past, "crop"):
                raise RuntimeError("no croppable KV")
            past.crop(prompt_len + d)
            cur = full[:, prompt_len + d - 1:prompt_len + d]
            dev = self._model.device
            suffix: list[int] = []
            eos = self._tok.eos_token_id
            hist = draft[:d].tolist()
            with torch.no_grad():
                for _step in range(max_new):
                    L = prompt_len + d + _step
                    o = self._model(
                        input_ids=cur,
                        attention_mask=torch.ones((1, L + 1), dtype=torch.long,
                                                  device=dev),
                        past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    row = o.logits[0, -1]
                    _ban_repeat_trigrams(row, hist + suffix)
                    nxt = int(row.argmax())
                    if nxt == eos:
                        break
                    suffix.append(nxt)
                    cur = torch.tensor([[nxt]], device=dev)
            path = "crop"
        except Exception as e:
            if not isinstance(e, LookupError):
                log.warning("SSBD crop-resume failed (%s) — re-prefill", e)
            try:
                pre = torch.cat([input_ids, draft_ids[:, :d]], dim=-1) \
                    if d else input_ids
                out = self._generate(pre, torch.ones_like(pre),
                                     self._gen_kwargs(speculate=False, stop=stop))
                suffix = out[0][pre.shape[-1]:].tolist()
                path = "reprefill"
            except Exception as e2:
                log.warning("SSBD re-prefill failed (%s) — scratch", e2)
                return _scratch()

        gen_ids = draft[:d].tolist() + suffix
        # Cut at first EOS like generate() would (draft carries no EOS).
        try:
            eos = self._tok.eos_token_id
            if eos in gen_ids:
                gen_ids = gen_ids[:gen_ids.index(eos)]
        except Exception:
            pass
        return {"text": self._tok.decode(gen_ids, skip_special_tokens=True).strip(),
                "draft_len": m, "accepted": d, "beta": beta, "path": path}

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
                    ids = collector.take(timeout=2.0)
                    if ids:
                        gen_all += ids[max(0, prompt_len - consumed):]
                        consumed += len(ids)
                        # held back: a split UTF-8 char decodes as U+FFFD until its
                        # next byte-token lands; emitting it would skip the real char
                        new_text = self._tok.decode(gen_all, skip_special_tokens=True).rstrip("\ufffd")
                        if len(new_text) > len(shown):
                            yield new_text[len(shown):]
                            shown = new_text
                    # break only on an EMPTY take: checking is_alive() after a
                    # non-empty one dropped ids put between take and the check
                    elif collector._done or not thread.is_alive():
                        break
            finally:
                thread.join(timeout=10)
            if not shown.strip():
                # Modeling never streamed (or streamed nothing usable):
                # one-shot fallback WITHOUT re-locking (lock already held).
                yield self._translate_locked(text, tgt, context, terms, speculate)

    def _prepare_batched(self, text: str, tgts: list[str],
                         context: list[str] | None = None,
                         terms: dict[str, str] | None = None):
        """Tokenize one prompt per target, left-pad to a batch. Caller holds _lock."""
        import torch

        encs = []
        for tgt in tgts:
            messages = [{"role": "user",
                         "content": self._prompt(text, to_full(tgt), context, terms)}]
            enc = self._tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt")
            ids = enc["input_ids"] if not torch.is_tensor(enc) else enc
            encs.append(ids[0])
        pad = self._tok.eos_token_id
        max_len = max(e.shape[0] for e in encs)
        batch = torch.stack([
            torch.cat([torch.full((max_len - e.shape[0],), pad, dtype=e.dtype), e])
            for e in encs])
        mask = torch.stack([
            torch.cat([torch.zeros(max_len - e.shape[0], dtype=torch.long),
                       torch.ones(e.shape[0], dtype=torch.long)])
            for e in encs])
        return batch.to(self._model.device), mask.to(self._model.device)

    def translate_targets_stream(self, text: str, targets: list[str],
                                 src: str = "auto",
                                 context: list[str] | None = None,
                                 terms: dict[str, str] | None = None):
        """Translate to every target in ONE generate call, yielding (tgt, delta).

        Shared prefill + parallel decode beats sequential per-target calls
        (~2-3x on 3 targets); per-row token streams demux into the same (tgt,
        delta) events the UI already renders. Falls back to sequential
        per-target streaming when batching is unsupported. Holds _lock.
        """
        text = (text or "").strip()
        tgts = [t for t in (targets or []) if t]
        if not text or not tgts:
            return
            yield  # generator in all paths
        self.ensure_loaded()
        if self._mock:
            for t in tgts:
                yield t, self._model.translate(text, src=src, tgt=t)
            return
        if not self._supports_batch:
            yield from self._sequential_stream(text, tgts, src, context, terms)
            return
        import threading

        with self._lock:
            try:
                input_ids, attn_mask = self._prepare_batched(text, tgts, context, terms)
            except Exception as e:
                log.warning("batched prepare failed (%s) — sequential fallback", e)
                yield from self._sequential_stream(text, tgts, src, context, terms)
                return
            demux = _RowDemux(len(tgts), self._tok.eos_token_id)
            speculate = all(self._should_speculate(src, t, text) for t in tgts)
            gen_kwargs = self._gen_kwargs(streamer=demux, speculate=speculate)
            thread = threading.Thread(
                target=self._generate, args=(input_ids, attn_mask, gen_kwargs),
                daemon=True)
            thread.start()
            shown = ["" for _ in tgts]
            gen_all: list[list[int]] = [[] for _ in tgts]
            try:
                while True:
                    got = demux.take_any(timeout=2.0) or []
                    for row, ids in got:
                        gen_all[row] += ids
                        new_text = self._tok.decode(gen_all[row], skip_special_tokens=True).rstrip("\ufffd")
                        if len(new_text) > len(shown[row]):
                            yield tgts[row], new_text[len(shown[row]):]
                            shown[row] = new_text
                    if not thread.is_alive():
                        break
            finally:
                thread.join(timeout=10)
            if not any(s.strip() for s in shown):
                yield from self._sequential_stream(text, tgts, src, context, terms)

    def _sequential_stream(self, text, tgts, src, context, terms):
        """Old path, kept as the batched fallback. Yields (tgt, delta)."""
        for tgt in tgts:
            try:
                for delta in self.translate_stream(text, tgt=tgt, src=src,
                                                   context=context, terms=terms):
                    if delta:
                        yield tgt, delta
            except Exception as e:
                log.warning("streamed translate (%s) failed: %s", tgt, e)

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
