"""Streaming conference session: continuous ASR + live re-translation.

Fixed 5 s windows slice sentences mid-word. Instead, one sherpa-onnx
streaming recognizer stream stays open while audio flows in, and segments
are finalized on *silence* (endpointing) — utterances break at pauses::

    mic/file frames -> recognizer stream -> partial text (live on screen)
                     -> silence >= endpoint_silence -> finalize:
                        diarize segment -> stream-translate finals
                        (token events) -> utterance event

Translation is context-aware: the last few finalized segments travel in the
prompt as terminology/style history. Finals are generated token-streamed
("tok" events) with prompt-lookup speculative decoding (drafter-free:
candidate n-grams come from the prompt/history itself). Partials keep the
cheaper debounced full-string re-translation.
"""
from __future__ import annotations

import itertools
import logging
import time
from collections import deque

import numpy as np

from .diarizer import rms_dbfs

log = logging.getLogger("conf.stream")


class StreamingSession:
    def __init__(self, cfg, asr_engine, mt_engine, diarizer, enhancer,
                 results_queue, id_start: int = 1_000_000):
        self.cfg = cfg or {}
        self.asr = asr_engine
        self.mt = mt_engine
        self.diarizer = diarizer
        self.enhancer = enhancer
        self.results = results_queue
        self._ids = itertools.count(id_start)

        self.endpoint_silence = float(cfg.get("endpoint_silence", 1.0))
        self.min_speech = float(cfg.get("endpoint_min_speech", 0.5))
        self.max_segment = float(cfg.get("max_segment", 20.0))
        self.tr_interval = float(cfg.get("partial_translate_interval", 2.5))
        self.tr_min_delta = int(cfg.get("partial_min_delta", 6))
        self.tr_min_chars = int(cfg.get("partial_min_chars", 10))

        self.targets: list[str] = list(cfg.get("default_targets", ["en", "zh"]))
        self.src_lang = None
        self.denoise: bool | None = None
        self.terms: dict[str, str] = {}
        self.display_lang: str | None = None  # streamed + translated first

        self.history: deque[str] = deque(maxlen=max(1, int(cfg.get("mt_history_turns", 3))))
        self.history_chars = int(cfg.get("mt_history_chars", 600))

        self._stream = None
        self._rec = None
        self._reset_segment()
        self._n_final = 0

    # -- setup --------------------------------------------------------------
    def configure(self, targets=None, src_lang=None, denoise=None, terms=None,
                  display_lang=None):
        if targets:
            self.targets = list(targets)
        self.src_lang = src_lang
        self.denoise = denoise
        self.terms = dict(terms or {})
        self.display_lang = (display_lang or None)
        self.asr.ensure_loaded()
        self._rec = self.asr._rec
        if self._rec is False or self._rec is None:
            raise RuntimeError("ASR unavailable for streaming")
        self._stream = self._rec.create_stream()

    def live_update(self, targets=None, denoise=None, terms=None,
                    display_lang=None):
        """Hot-update mid-stream settings (no recognizer reset)."""
        if targets:
            self.targets = list(targets)
        if denoise is not None:
            self.denoise = bool(denoise)
        if terms is not None:
            self.terms = dict(terms)
        if display_lang is not None:
            self.display_lang = display_lang or None

    def _ordered_targets(self) -> list[str]:
        """Display language first so its tokens stream earliest."""
        if self.display_lang and self.display_lang in self.targets:
            return [self.display_lang] + [t for t in self.targets
                                          if t != self.display_lang]
        return list(self.targets)

    def _reset_segment(self):
        self._seg_audio: list[np.ndarray] = []
        self._seg_dur = 0.0
        self._voice_dur = 0.0
        self._trailing_sil = 0.0
        self._seg_start: float | None = None
        self._seg_speaker = "SPEAKER_01"
        self._seg_id: int | None = None
        self._last_partial = ""
        self._last_tr_text = ""
        self._last_tr_t = 0.0

    def _want_denoise(self) -> bool:
        if self.denoise is not None:
            return bool(self.denoise)
        if self.enhancer is not None:
            return bool(self.enhancer.enabled)
        return bool(self.cfg.get("noise_suppress", True))

    def _context(self) -> list[str]:
        """Recent finalized source texts, oldest-first, capped by chars."""
        out: list[str] = []
        total = 0
        for text in reversed(self.history):
            text = (text or "").strip()
            if not text:
                continue
            if total + len(text) > self.history_chars and out:
                break
            out.append(text)
            total += len(text)
        out.reverse()
        return out

    # -- streaming input ------------------------------------------------------
    def feed(self, pcm: np.ndarray, sr: int, t: float | None = None):
        x = np.asarray(pcm, dtype=np.float32).ravel()
        if x.size == 0:
            return
        if int(sr) != 16000:
            ratio = 16000 / float(sr)
            idx = (np.arange(int(len(x) * ratio)) / ratio).astype(int)
            x = x[np.clip(idx, 0, len(x) - 1)]
            sr = 16000
        t = time.time() if t is None else t
        dur = len(x) / 16000.0

        dia = self.diarizer.assign(x, t)
        silent = bool(dia.get("silent"))

        if self._seg_start is None and not silent:
            self._seg_start = t
            self._seg_speaker = dia.get("speaker", "SPEAKER_01")
            self._seg_id = next(self._ids)
        if self._seg_start is not None:
            self._seg_audio.append(x.copy())
            self._seg_dur += dur
            if silent:
                self._trailing_sil += dur
            else:
                self._voice_dur += dur
                self._trailing_sil = 0.0

        # recognizer hears everything (silence gives it endpoint context)
        try:
            self._stream.accept_waveform(16000, np.clip(x, -1.0, 1.0))
            while self._rec.is_ready(self._stream):
                self._rec.decode_stream(self._stream)
        except Exception as e:
            log.warning("stream decode failed: %s", e)
            return

        if self._seg_start is not None:
            self._maybe_emit_partial(t)
            if (silent and self._voice_dur >= self.min_speech
                    and self._trailing_sil >= self.endpoint_silence):
                self._finalize(t)
            elif self._seg_dur >= self.max_segment and self._voice_dur >= self.min_speech:
                self._finalize(t)

    # -- partials ---------------------------------------------------------------
    def _maybe_emit_partial(self, t: float):
        try:
            text = self._rec.get_result(self._stream).strip()
        except Exception:
            return
        if not text or text == self._last_partial:
            return
        self._last_partial = text
        translations = None
        if (len(text) >= self.tr_min_chars
                and len(text) - len(self._last_tr_text) >= self.tr_min_delta
                and (t - self._last_tr_t) >= self.tr_interval):
            try:
                # Live re-translation covers the display language only: one
                # MT call keeps the feed loop realtime; full targets land
                # at finalize. Falls back to all targets when unset.
                dl = self.display_lang
                live_tgts = [dl] if dl and dl in self.targets else self.targets
                translations = self.mt.translate_multi(
                    text, live_tgts, src="auto", context=self._context(),
                    terms=self.terms or None)
                self._last_tr_text = text
                self._last_tr_t = t
            except Exception as e:
                log.warning("partial translate failed: %s", e)
        self.results.put(("partial", {
            "id": self._seg_id, "speaker": self._seg_speaker,
            "rms_db": round(rms_dbfs(np.concatenate(self._seg_audio[-4:])), 1)
            if self._seg_audio else -80.0,
            "text": text, "translations": translations, "final": False,
        }))

    # -- finalize ---------------------------------------------------------------
    def _finalize(self, t: float):
        seg = np.concatenate(self._seg_audio) if self._seg_audio else np.zeros(0, np.float32)
        denoised = False
        if self._want_denoise() and self.enhancer is not None and seg.size:
            seg = self.enhancer.enhance_array(seg, 16000)
            denoised = self.enhancer.active
        # Final text: GPU one-shot re-decode when available (also lets the
        # denoised segment be heard fresh), else the live-stream result.
        # Either way seg is what the diarizer/translator see below.
        asr_backend = "stream"
        text = ""
        rd = getattr(self.asr, "redecode_cuda", None)
        if callable(rd) and seg.size:
            try:
                cuda_text = rd(seg, 16000)
            except Exception as e:
                log.warning("CUDA redecode failed (%s).", e)
                cuda_text = None
            if cuda_text:
                text, asr_backend = cuda_text, "ort-cuda"
        if not text:
            if denoised:
                # streaming ASR ran on raw audio — re-decode the enhanced
                # segment one-shot so the final text matches what was heard
                try:
                    s2 = self._rec.create_stream()
                    s2.accept_waveform(16000, np.clip(seg, -1.0, 1.0))
                    s2.accept_waveform(16000, np.zeros(int(16000 * 0.4), dtype=np.float32))
                    s2.input_finished()
                    while self._rec.is_ready(s2):
                        self._rec.decode_stream(s2)
                    text = self._rec.get_result(s2).strip()
                except Exception as e:
                    log.warning("enhanced re-decode failed: %s", e)
                    text = self._last_partial
            else:
                try:
                    text = self._rec.get_result(self._stream).strip()
                except Exception:
                    text = self._last_partial
        if not text:
            self._fresh_stream()
            self._reset_segment()
            return
        context = self._context()
        terms = self.terms or None
        # Neural speaker verdict: NeMo embedding attribution overrides the
        # volume guess used for live partials (no-op on volume diarizers).
        speaker = self._seg_speaker
        diar_backend = None
        try:
            attr = getattr(self.diarizer, "attribute_segment", None)
            if callable(attr) and seg.size:
                verdict = attr(seg, 16000)
                if verdict and verdict.get("speaker"):
                    speaker = verdict["speaker"]
                    diar_backend = verdict.get("backend")
        except Exception as e:
            log.warning("segment attribution failed: %s", e)
        translations: dict[str, str] = {}
        try:
            # Token-stream each target so the UI fills in live ("tok" events),
            # then confirm with the assembled strings in the utterance event.
            # Display language streams first so visible progress starts ASAP.
            for tgt in self._ordered_targets():
                parts: list[str] = []
                seq = 0
                try:
                    for delta in self.mt.translate_stream(text, tgt=tgt, src="auto",
                                                          context=context, terms=terms):
                        if delta:
                            parts.append(delta)
                            self.results.put(("tok", {"id": self._seg_id, "tgt": tgt,
                                                      "seq": seq, "delta": delta}))
                            seq += 1
                except Exception as e:
                    log.warning("streamed translate (%s) failed: %s", tgt, e)
                chunk = "".join(parts).strip()
                translations[tgt] = chunk or self.mt.translate(
                    text, tgt=tgt, src="auto", context=context, terms=terms)
        except Exception as e:
            translations = {tgt: f"[MT error: {e}]" for tgt in self.targets}
        self.history.append(text)
        entry = {
            "id": self._seg_id, "type": "utterance", "speaker": speaker,
            "rms_db": round(rms_dbfs(seg), 1) if seg.size else -80.0,
            "denoised": denoised, "src_lang": "auto", "text": text,
            "translations": translations, "start": round(self._seg_start or t, 2),
            "asr_backend": asr_backend,
        }
        if diar_backend:
            entry["diar_backend"] = diar_backend
        self.results.put(("utterance", entry))
        self._n_final += 1
        self._fresh_stream()
        self._reset_segment()

    def _fresh_stream(self):
        try:
            self._stream = self._rec.create_stream()
        except Exception as e:
            log.warning("stream reset failed: %s", e)

    def stop(self) -> int:
        """Finalize any pending speech. Returns number of finalized segments."""
        if self._seg_start is not None and self._voice_dur >= 0.2:
            try:
                text = self._rec.get_result(self._stream).strip() or self._last_partial
            except Exception:
                text = self._last_partial
            if text:
                self._last_partial = text
                self._finalize(time.time())
            else:
                self._reset_segment()
        n = self._n_final
        self._n_final = 0
        return n
