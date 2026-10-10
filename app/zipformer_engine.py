"""Local Zipformer zh-en-vi ASR via sherpa-onnx (replaces Qwen3-ASR).

Model folder (encoder/decoder/joiner .onnx + tokens.txt) is the streaming
variant — the encoder takes fixed [N, 39, 80] feature chunks — so we use
sherpa-onnx's *online* (streaming) recognizer, one stream per segment.

Providers: sherpa-onnx PyPI wheels are CPU-only builds (they fall back to CPU
even with onnxruntime-gpu installed), so live partials always run on CPU.
Finalized segments are re-decoded on GPU via app/cuda_zipformer.py (direct
ORT CUDA, byte-identical protocol) when available; see transcribe_final().
The int8 encoder decodes ~15x realtime on a laptop CPU, which is plenty for
5 s conference segments; MT stays on CUDA. If a GPU-enabled sherpa-onnx
build is ever installed, set `asr_provider: cuda` to use it.

Same interface as the old Qwen engine: ensure_loaded / transcribe_array /
transcribe_file / status. No language-ID head on this model, so detected
language is always reported as "auto" (pipeline then fills every target).
"""
from __future__ import annotations

import difflib
import functools
import logging
import os
import tempfile
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger("conf.asr")

DEFAULT_DIR = "D:/CONFERENCE ASR/zipformer zh-en-vi onnx phaseB 2e"
TAIL_SECONDS = 0.66  # zero-pad flushes the last chunk (model card: 32 frames + 7 look-ahead)


def model_file(d, part: str, int8: bool = False) -> str:
    """The one `<part>-<tag>[.int8].onnx` in d, whatever the tag (s2a, 2e, ...)."""
    hits = [p for p in Path(d).glob(f"{part}-*.onnx") if p.name.endswith(".int8.onnx") == int8]
    if len(hits) != 1:
        raise FileNotFoundError(f"need one {part}-*{'.int8' if int8 else ''}.onnx in {d}, "
                                f"found {[p.name for p in hits]}")
    return str(hits[0])


class ZipformerEngine:
    backend = "sherpa-onnx"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_dir = Path(os.getenv("ASR_MODEL", cfg.get("asr_model", DEFAULT_DIR)))
        if not self.model_dir.is_dir():  # other machines: scripts/download_models.py puts it here
            self.model_dir = Path(__file__).resolve().parent.parent / "models" / "zipformer-2e"
        self.provider: str = str(cfg.get("asr_provider", "cpu")).lower()
        quant = str(cfg.get("asr_quant", "auto")).lower()
        # int8 encoder is ~2x smaller/faster on CPU; fp32 for CUDA builds
        self.quant = "int8" if (quant == "int8" or (quant == "auto" and self.provider == "cpu")) else "fp32"
        self.num_threads = int(cfg.get("asr_threads", 4))
        self.hotwords_score = float(cfg.get("asr_hotwords_score", 0))
        self.lex_cutoff = float(cfg.get("asr_lexicon_cutoff", 0.85))
        self._lock = threading.Lock()
        self._rec = None
        self.model_id = f"zipformer zh-en-vi (onnx {self.quant}, local)"

    @property
    def loaded(self):
        return self._rec is not None

    def _paths(self) -> dict:
        q = self.quant == "int8"
        tokens = self.model_dir / "tokens.txt"
        if not tokens.is_file():
            raise FileNotFoundError(f"Zipformer tokens missing: {tokens}")
        # int8 = int8 encoder + joiner with the fp32 decoder (model card setup)
        return {"encoder": model_file(self.model_dir, "encoder", q),
                "decoder": model_file(self.model_dir, "decoder"),
                "joiner": model_file(self.model_dir, "joiner", q),
                "tokens": str(tokens)}

    def _hotword_kw(self, tokens: str) -> dict:
        """Beam search + bpe vocab so new_stream() can bias toward glossary terms.

        The model ships no bpe.model, so tokens.txt becomes the bpe vocab (equal scores =
        fewest pieces). Off (greedy) when asr_hotwords_score is 0.
        """
        if self.hotwords_score <= 0:
            return {}
        vocab = Path(tempfile.gettempdir()) / "conflive-asr-bpe.vocab"
        with open(tokens, encoding="utf-8") as f, open(vocab, "w", encoding="utf-8") as v:
            v.writelines(f"{t}\t-1\n" for t in (l.rsplit(" ", 1)[0] for l in f) if not t.startswith("<"))
        return dict(decoding_method="modified_beam_search", modeling_unit="cjkchar+bpe",
                    bpe_vocab=str(vocab), hotwords_score=self.hotwords_score)

    def new_stream(self, terms: dict | None = None):
        """Recognizer stream biased toward both sides of the glossary {source: target}."""
        if not terms or self.hotwords_score <= 0:
            return self._rec.create_stream()
        # Pieces are case-split: English CAPS, Vietnamese lowercase.
        # ponytail: ASCII word = English, so a diacritic-free Vietnamese word gets English pieces
        hw = {" ".join(w.upper() if w.isascii() else w.lower() for w in t.replace("/", " ").split())
              for kv in terms.items() for t in kv}
        return self._rec.create_stream(hotwords="/".join(h for h in hw if h))

    @functools.cached_property
    def lexicon(self) -> dict:
        """{first letter: {word: count}} from asr_lexicon (scripts/get_lexicon.py); {} = off."""
        path = Path(self.cfg.get("asr_lexicon") or "")
        if not path.is_file():
            return {}
        lex: dict[str, dict[str, int]] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                w, n = line.split()
                lex.setdefault(w[0], {})[w] = int(n)
        log.info("ASR lexicon: %d words from %s", sum(map(len, lex.values())), path)
        return lex

    def correct(self, text: str, terms: dict | None = None) -> str:
        """Snap misheard (out-of-lexicon) words to the closest frequent lexicon word.

        Glossary words are never touched; CJK, short and non-alphabetic words pass through.
        ponytail: per-word spelling only, no context (a real-word mishearing like
        "there"/"their" stays); an n-gram LM (sherpa-onnx lm=) is the upgrade.
        """
        if not self.lexicon or not text:
            return text
        keep = {w.lower() for kv in (terms or {}).items() for t in kv for w in t.split()}
        out = []
        for word in text.split():
            k = word.lower()
            bucket = self.lexicon.get(k[0], {})
            if (len(k) >= 4 and k.isalpha() and k not in bucket and k not in keep
                    and not any("\u3400" <= c <= "\u9fff" for c in k)):
                hits = difflib.get_close_matches(k, bucket, 3, self.lex_cutoff)
                if hits:
                    best = max(hits, key=bucket.get)
                    word = best.upper() if word.isupper() else best
            out.append(word)
        return " ".join(out)

    def ensure_loaded(self, demo_ok: bool = True):
        if self._rec is not None:
            return
        with self._lock:
            if self._rec is not None:
                return
            try:
                import sherpa_onnx

                p = self._paths()
                log.info("Loading Zipformer %s (%s) provider=%s",
                         self.model_dir.name, self.quant, self.provider)
                self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(
                    encoder=p["encoder"], decoder=p["decoder"], joiner=p["joiner"],
                    tokens=p["tokens"], provider=self.provider,
                    num_threads=self.num_threads, **self._hotword_kw(p["tokens"]))
                log.info("ASR ready: %s", self.model_id)
            except Exception as e:
                log.warning("Zipformer load failed: %s", e)
                if not demo_ok:
                    raise
                log.warning("ASR unavailable and demo_ok=True — transcriptions will be empty.")
                self._rec = False

    def transcribe_array(self, pcm: np.ndarray, sr: int, language=None, terms=None) -> tuple[str, str]:
        """Transcribe mono float32 audio @16 kHz. Returns ("auto", text)."""
        self.ensure_loaded()
        x = np.asarray(pcm, dtype=np.float32).ravel()
        if x.size == 0 or self._rec is False:
            return "auto", ""
        if int(sr) != 16000 and x.size:
            ratio = 16000 / float(sr)
            idx = (np.arange(int(len(x) * ratio)) / ratio).astype(int)
            x = x[np.clip(idx, 0, len(x) - 1)]
        with self._lock:
            import sherpa_onnx  # noqa: F401 (already imported in ensure_loaded)

            s = self.new_stream(terms)
            s.accept_waveform(16000, np.clip(x, -1.0, 1.0))
            s.accept_waveform(16000, np.zeros(int(16000 * TAIL_SECONDS), dtype=np.float32))
            s.input_finished()
            while self._rec.is_ready(s):
                self._rec.decode_stream(s)
            return "auto", self._rec.get_result(s).strip()

    def transcribe_file(self, path: str, language=None) -> tuple[str, str]:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=False)
        a = np.asarray(data, dtype=np.float32)
        if a.ndim > 1:
            a = a.mean(axis=-1).astype(np.float32)
        return self.transcribe_array(a, sr, language=language)

    def _ort(self):
        """Shared direct-ORT driver (CPU default; CUDA opt-in + probed)."""
        from .cuda_zipformer import CudaZipformer

        if not hasattr(self, "_cuda") or self._cuda is None:
            prefer = str(self.cfg.get("asr_ort_provider", "cpu")).lower() == "cuda"
            self._cuda = CudaZipformer(str(self.model_dir), prefer_cuda=prefer,
                                       threads=int(self.cfg.get("asr_ort_threads", 2)))
        return self._cuda

    def transcribe_final(self, pcm: np.ndarray, sr: int = 16000) -> tuple[str, str]:
        """Final-segment decode via direct ORT (legacy CPU path preserved).

        Tries the ORT driver first (see app/cuda_zipformer.py); falls back
        to the CPU streaming recognizer. Returns (text, backend).
        """
        x = np.asarray(pcm, dtype=np.float32).ravel()
        if x.size == 0:
            return "", "none"
        try:
            with self._lock:
                text = self._ort().transcribe_array(x, sr)
            prov = self._ort().provider
            return text.strip(), ("ort-cuda" if "CUDA" in prov else "ort-cpu")
        except Exception as e:
            log.warning("ORT final decode unavailable (%s) — CPU fallback.", e)
            self._cuda = False
        _, text = self.transcribe_array(x, sr)
        return text, "sherpa-cpu"

    def redecode_cuda(self, pcm: np.ndarray, sr: int = 16000,
                      ) -> tuple[str | None, str | None]:
        """One-shot ORT re-decode; (None, None) when unavailable.

        Returns (text, backend) with backend like "ort-cpu"/"ort-cuda".
        Streaming finalize prefers this; on None the caller keeps the
        live-stream result. Never raises for missing providers.
        """
        from .cuda_zipformer import CudaUnavailable

        x = np.asarray(pcm, dtype=np.float32).ravel()
        if x.size == 0:
            return None, None
        try:
            with self._lock:
                driver = self._ort()
                text = driver.transcribe_array(x, sr).strip() or None
                prov = driver.provider
                backend = ("ort-cuda" if "CUDA" in prov else "ort-cpu"
                           if text else None)
                return text, backend
        except CudaUnavailable as e:
            log.warning("ORT redecode unavailable (%s) — live result kept.", e)
            self._cuda = False
            return None, None
        except Exception as e:
            log.warning("ORT redecode failed (%s) — live result kept.", e)
            return None, None

    def status(self) -> dict:
        return {
            "model": self.model_id,
            "backend": self.backend,
            "device": self.provider,
            "dir": str(self.model_dir),
            "ready": self.loaded and self._rec is not False,
        }
