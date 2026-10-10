"""Conference pipeline: denoise (DeepFilterNet) -> diarize -> ASR -> translate (vi/en/zh)."""
from __future__ import annotations

import itertools
import logging
import time
from collections import deque

import numpy as np

from .audio_io import chunk_stream
from .diarizer import rms_dbfs
from .langid import detect as detect_lang

log = logging.getLogger("conf.pipe")

LANG_TO_CODE = {
    "vietnamese": "vi", "tiếng việt": "vi", "viet": "vi",
    "english": "en", "chinese": "zh", "mandarin": "zh", "中文": "zh",
    "cantonese": "yue",
}


def norm_lang_code(name: str | None) -> str:
    if not name:
        return "auto"
    s = str(name).strip().lower()
    if s in ("vi", "en", "zh", "auto"):
        return s
    return LANG_TO_CODE.get(s, s[:2])


class ConferencePipeline:
    def __init__(self, cfg, asr, mt, diarizer, enhancer=None):
        self.cfg = cfg
        self.asr = asr
        self.mt = mt
        self.diarizer = diarizer
        self.enhancer = enhancer  # DeepFilterNetEnhancer | None
        self._ids = itertools.count(1)
        self.history: list[dict] = []
        self.mt_history: deque[str] = deque(
            maxlen=max(1, int(cfg.get("mt_history_turns", 3))))
        self.mt_history_chars = int(cfg.get("mt_history_chars", 600))

    def _mt_context(self) -> list[str]:
        out: list[str] = []
        total = 0
        for text in reversed(self.mt_history):
            text = (text or "").strip()
            if not text:
                continue
            if total + len(text) > self.mt_history_chars and out:
                break
            out.append(text)
            total += len(text)
        out.reverse()
        return out

    def _want_denoise(self, denoise: bool | None) -> bool:
        if denoise is not None:
            return bool(denoise)
        if self.enhancer is not None:
            return bool(self.enhancer.enabled)
        return bool(self.cfg.get("noise_suppress", True))

    def process_chunk(self, pcm: np.ndarray, sr: int, targets: list[str] | None = None,
                      src_lang: str | None = None, t: float | None = None,
                      denoise: bool | None = None,
                      terms: dict[str, str] | None = None) -> dict | None:
        pcm = np.asarray(pcm, dtype=np.float32).ravel()
        if pcm.size < sr * 0.3:  # ignore blips
            return None
        t = time.time() if t is None else t
        denoised = False
        if self.enhancer is not None and self._want_denoise(denoise):
            pcm = self.enhancer.enhance_array(pcm, sr)
            denoised = self.enhancer.active
        dia = self.diarizer.assign(pcm, t)
        if dia.get("silent"):
            return {"type": "silence", "rms_db": dia["rms_db"], "speaker": dia["speaker"]}
        # The one-shot ORT re-decode gives the same text as the streaming
        # recognizer and holds a second model copy (+1 GB RAM): only with
        # asr_cuda_final, like the live path.
        lang_name, text, asr_backend = None, "", "sherpa-cpu"
        try:
            rd = getattr(self.asr, "redecode_cuda", None)
            if callable(rd) and self.cfg.get("asr_cuda_final", False):
                r_text, r_back = rd(pcm, sr)
                if r_text:
                    text, asr_backend = r_text, r_back or "ort"
            if not text:
                lang_name, text = self.asr.transcribe_array(pcm, sr, language=src_lang, terms=terms)
        except Exception as e:
            log.warning("ASR chunk failed: %s", e)
            return {"type": "error", "error": str(e), "speaker": dia["speaker"]}
        if not (text or "").strip():
            return {"type": "empty", "speaker": dia["speaker"], "rms_db": dia["rms_db"]}
        text = self.asr.correct(text, terms)
        detected = norm_lang_code(lang_name)
        targets = targets or list(self.cfg.get("default_targets", ["en", "zh"]))
        # Skip MT where the target IS the detected source (a copy, not a
        # ~3 s generate). None = unsure = translate everything.
        lid = detect_lang(text)
        src = lid or detected
        # Neural speaker verdict overrides the volume guess (no-op otherwise).
        speaker, diar_backend = dia["speaker"], dia.get("backend")
        try:
            attr = getattr(self.diarizer, "attribute_segment", None)
            if callable(attr):
                verdict = attr(pcm, sr)
                if verdict and verdict.get("speaker"):
                    speaker = verdict["speaker"]
                    diar_backend = verdict.get("backend")
        except Exception as e:
            log.warning("segment attribution failed: %s", e)
        translations = {}
        context = self._mt_context()
        for tgt in targets:
            if tgt == detected or (lid and tgt.lower() == lid):
                translations[tgt] = text
                continue
            try:
                translations[tgt] = self.mt.translate(text, tgt=tgt, src=src,
                                                      context=context, terms=terms)
            except Exception as e:
                translations[tgt] = f"[MT error: {e}]"
        entry = {
            "id": next(self._ids),
            "type": "utterance",
            "speaker": speaker,
            "rms_db": dia["rms_db"],
            "denoised": denoised,
            "src_lang": src,
            "text": text,
            "translations": translations,
            "start": round(t, 2),
            "asr_backend": asr_backend,
        }
        if diar_backend:
            entry["diar_backend"] = diar_backend
        self.history.append(entry)
        self.mt_history.append(text)
        if len(self.history) > 500:
            self.history = self.history[-500:]
        return entry

    def process_file(self, pcm: np.ndarray, sr: int, targets: list[str] | None = None,
                     src_lang: str | None = None, denoise: bool | None = None,
                     terms: dict[str, str] | None = None) -> list[dict]:
        out = []
        # Whole-file denoise in one pass (cleaner than per-chunk: no boundaries),
        # then split the *enhanced* audio for diarization + ASR.
        file_denoised = False
        if self.enhancer is not None and self._want_denoise(denoise):
            pcm = self.enhancer.enhance_array(np.asarray(pcm, dtype=np.float32).ravel(), sr)
            file_denoised = self.enhancer.active
        seg = float(self.cfg.get("segment_seconds", 5.0))
        t0 = time.time() - len(pcm) / sr
        for ch in chunk_stream(pcm, sr, seg):
            r = self.process_chunk(ch, sr, targets=targets, src_lang=src_lang, t=t0,
                                   denoise=False, terms=terms)  # already denoised above
            t0 += len(ch) / sr
            if r and r.get("type") == "utterance":
                r["denoised"] = file_denoised
                out.append(r)
        return out

    def export_txt(self) -> str:
        lines = []
        for e in self.history:
            tr = " | ".join(f"[{k}] {v}" for k, v in e.get("translations", {}).items())
            lines.append(f"{e['speaker']} ({e.get('src_lang','')}): {e['text']}\n  -> {tr}")
        return "\n".join(lines)

    def export_srt(self) -> str:
        def ts(s):
            h, rem = divmod(int(s), 3600)
            m, sec = divmod(rem, 60)
            return f"{h:02d}:{m:02d}:{sec:02d},000"

        blocks = []
        for i, e in enumerate(self.history, 1):
            tr = " ".join(e.get("translations", {}).values())
            blocks.append(f"{i}\n{ts(i*5)} --> {ts(i*5+4)}\n{e['speaker']}: {e['text']}\n{tr}\n")
        return "\n".join(blocks)
