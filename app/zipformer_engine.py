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

import logging
import os
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger("conf.asr")

DEFAULT_DIR = "D:/DENSEV2 - reading/zipformer zh-en-vi onnx phaseB s2a"
TAIL_SECONDS = 0.4  # zero-pad flushes trailing words out of the streaming model


class ZipformerEngine:
    backend = "sherpa-onnx"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_dir = Path(os.getenv("ASR_MODEL", cfg.get("asr_model", DEFAULT_DIR)))
        self.provider: str = str(cfg.get("asr_provider", "cpu")).lower()
        quant = str(cfg.get("asr_quant", "auto")).lower()
        # int8 encoder is ~2x smaller/faster on CPU; fp32 for CUDA builds
        self.quant = "int8" if (quant == "int8" or (quant == "auto" and self.provider == "cpu")) else "fp32"
        self.num_threads = int(cfg.get("asr_threads", 4))
        self._lock = threading.Lock()
        self._rec = None
        self.model_id = f"zipformer zh-en-vi (onnx {self.quant}, local)"

    @property
    def loaded(self):
        return self._rec is not None

    def _paths(self) -> dict:
        q = "" if self.quant == "fp32" else ".int8"
        files = {
            "encoder": self.model_dir / f"encoder-phaseB_s2a{q}.onnx",
            "decoder": self.model_dir / f"decoder-phaseB_s2a{q}.onnx",
            "joiner": self.model_dir / f"joiner-phaseB_s2a{q}.onnx",
            "tokens": self.model_dir / "tokens.txt",
        }
        missing = [str(p) for p in files.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Zipformer model files missing: {missing}")
        return {k: str(v) for k, v in files.items()}

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
                    num_threads=self.num_threads)
                log.info("ASR ready: %s", self.model_id)
            except Exception as e:
                log.warning("Zipformer load failed: %s", e)
                if not demo_ok:
                    raise
                log.warning("ASR unavailable and demo_ok=True — transcriptions will be empty.")
                self._rec = False

    def transcribe_array(self, pcm: np.ndarray, sr: int, language=None) -> tuple[str, str]:
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

            s = self._rec.create_stream()
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

    def transcribe_final(self, pcm: np.ndarray, sr: int = 16000) -> tuple[str, str]:
        """Final-segment decode, GPU preferred (legacy CPU path preserved).

        Tries the direct ORT-CUDA driver first (see app/cuda_zipformer.py);
        falls back to the CPU streaming recognizer. Returns (text, backend)
        where backend is "ort-cuda" or "sherpa-cpu".
        """
        x = np.asarray(pcm, dtype=np.float32).ravel()
        if x.size == 0:
            return "", "none"
        if bool(self.cfg.get("asr_cuda_final", True)):
            try:
                from .cuda_zipformer import CudaZipformer

                if not hasattr(self, "_cuda") or self._cuda is None:
                    self._cuda = CudaZipformer(str(self.model_dir))
                with self._lock:
                    text = self._cuda.transcribe_array(x, sr)
                return text.strip(), "ort-cuda"
            except Exception as e:
                log.warning("CUDA final decode unavailable (%s) — CPU fallback.", e)
                self._cuda = False
        _, text = self.transcribe_array(x, sr)
        return text, "sherpa-cpu"

    def redecode_cuda(self, pcm: np.ndarray, sr: int = 16000) -> str | None:
        """One-shot GPU decode; None when CUDA is unavailable.

        Streaming finalize prefers this (GPU text, zero CPU cost); on None
        the caller keeps the live-stream result. Never raises for missing
        CUDA — only for real errors after a successful load.
        """
        x = np.asarray(pcm, dtype=np.float32).ravel()
        if x.size == 0 or not bool(self.cfg.get("asr_cuda_final", True)):
            return None
        try:
            from .cuda_zipformer import CudaZipformer, CudaUnavailable
        except Exception as e:
            log.warning("CUDA redecode unavailable (%s).", e)
            return None
        try:
            if not hasattr(self, "_cuda") or self._cuda is None:
                self._cuda = CudaZipformer(str(self.model_dir))
            with self._lock:
                return self._cuda.transcribe_array(x, sr).strip() or None
        except CudaUnavailable as e:
            log.warning("CUDA redecode unavailable (%s) — live result kept.", e)
            self._cuda = False
            return None

    def status(self) -> dict:
        return {
            "model": self.model_id,
            "backend": self.backend,
            "device": self.provider,
            "dir": str(self.model_dir),
            "ready": self.loaded and self._rec is not False,
        }
