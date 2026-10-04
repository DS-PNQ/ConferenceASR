"""Streaming conference session: continuous ASR + live re-translation.

Fixed 5 s windows slice sentences mid-word. Instead, one sherpa-onnx
streaming recognizer stream stays open while audio flows in, and segments
are finalized on *silence* (endpointing) — utterances break at pauses::

    mic/file frames -> recognizer stream -> partial text (live on screen)
                     -> silence >= endpoint_silence -> finalize:
                        diarize segment -> utterance event INSTANTLY (ASR text,
                        LID copies filled) -> MT runs in a background FIFO
                        worker (tok events stream in) -> utterance event again
                        with full translations. ASR never waits for MT.

Translation is context-aware: the last few finalized segments travel in the
prompt as terminology/style history. Finals generate token-streamed
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
from .langid import detect as detect_lang

log = logging.getLogger("conf.stream")


class StreamingSession:
    def __init__(self, cfg, asr_engine, mt_engine, diarizer, enhancer,
                 results_queue, id_start: int = 1_000_000, vad=None):
        self.cfg = cfg or {}
        self.asr = asr_engine
        self.mt = mt_engine
        self.diarizer = diarizer
        self.enhancer = enhancer
        self.vad = vad  # SileroVAD or None (None = energy-only, old behavior)
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
        self.vad_min_frac = float(cfg.get("vad_min_speech_frac", 0.3))

        self.history: deque[str] = deque(maxlen=max(1, int(cfg.get("mt_history_turns", 3))))
        self.history_chars = int(cfg.get("mt_history_chars", 600))

        self._stream = None
        self._rec = None
        self._reset_segment()
        self._n_final = 0
        self._mt_queue = None  # lazy background FIFO worker, see _submit_mt
        self._mt_thread = None
        # SSBD drafts: (seg_id, tgt) -> last translation text. Partial_{t+1}
        # drafts off partial_t; the final drafts off its last partial.
        self._ssbd_drafts: dict = {}

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
        if self.vad is not None:
            try:
                self.vad.ensure_loaded()
                self.vad.reset()  # fresh RNN state for the new stream
            except Exception as e:
                log.warning("VAD init failed (%s) — energy-only.", e)

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

    @property
    def mt_backlog(self) -> int:
        """Unfinished final-translation jobs. Partials yield to these."""
        q = self._mt_queue
        try:
            return int(q.unfinished_tasks) if q is not None else 0
        except Exception:
            return 0

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
        self._seg_vad_sum = 0.0  # neural speech probs (VAD-OR + junk gate)
        self._seg_vad_n = 0
        self._last_vad = True  # fail-open until the first VAD frame scores
        self._seg_start: float | None = None
        self._seg_speaker = "SPEAKER_01"
        self._seg_id: int | None = None
        self._last_partial = ""
        self._last_tr_text = ""
        self._last_tr_t = 0.0
        self._tr_busy = False
        self._tr_pending: tuple[str, list[str]] | None = None

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
        energy_silent = bool(dia.get("silent"))
        # Neural gate: one batched Silero forward per frame. silent when
        # EITHER detector says silence — loud non-speech opens no segment,
        # quiet speech holds one open. Fail-open (no VAD) = old behavior.
        vad_speech = True
        if self.vad is not None:
            try:
                vad_speech, vad_p = self.vad.frame_speech(x)
                self._seg_vad_sum += vad_p
                self._seg_vad_n += 1
            except Exception:
                vad_speech = True
        self._last_vad = vad_speech
        silent = energy_silent or not vad_speech

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
        # Text goes out IMMEDIATELY (realtime ASR); translations follow async
        # via _kick_partial_tr so slow MT never blocks frame ingestion.
        self.results.put(("partial", {
            "id": self._seg_id, "speaker": self._seg_speaker,
            "rms_db": round(rms_dbfs(np.concatenate(self._seg_audio[-4:])), 1)
            if self._seg_audio else -80.0,
            "text": text, "translations": None, "final": False,
        }))
        if (len(text) >= self.tr_min_chars
                and len(text) - len(self._last_tr_text) >= self.tr_min_delta
                and (t - self._last_tr_t) >= self.tr_interval):
            if self.mt_backlog > 0:
                return  # finals own the MT lock — don't queue partials behind them
            if not self._last_vad:
                return  # current frame is non-speech — nothing worth translating
            self._last_tr_text = text  # claim now: dedupes while worker runs
            self._last_tr_t = t
            # Display language only (1 generate, not N): the full set lands
            # with the finalize seconds later. Keeps partials ~3x cheaper so
            # finals — the rows users keep — start sooner.
            focus = ([self.display_lang] if self.display_lang
                      and self.display_lang in list(self.targets)
                      else list(self.targets)[:1])
            self._kick_partial_tr(text, focus, self._seg_id)

    def _kick_partial_tr(self, text: str, targets: list[str], seg_id: int | None):
        """Single-flight background re-translation of the live partial."""
        if self._tr_busy:
            self._tr_pending = (text, targets)
            return
        self._tr_busy = True
        self._tr_pending = (text, targets)
        import threading

        thread = threading.Thread(target=self._partial_tr_worker,
                                  args=(seg_id,), daemon=True)
        thread.start()

    def _partial_tr_worker(self, seg_id: int | None):
        """Translate latest pending partial; loop if text moved meanwhile."""
        try:
            while True:
                pending = self._tr_pending
                self._tr_pending = None
                if pending is None:
                    return
                text, targets = pending
                lid = detect_lang(text)
                translations: dict[str, str] = {}
                for tgt in targets:
                    if lid and tgt.lower() == lid:
                        translations[tgt] = text
                        continue
                    try:
                        draft = self._ssbd_drafts.get((seg_id, tgt))
                        ssbd = getattr(self.mt, "translate_ssbd", None)
                        if (draft and callable(ssbd)
                                and bool(self.cfg.get("mt_ssbd_enable", False))):
                            with self.mt._lock:
                                r = self.mt.translate_ssbd(
                                    text, draft, tgt=tgt, src=lid or "auto",
                                    context=self._context(),
                                    terms=self.terms or None)
                            translations[tgt] = r["text"]
                            log.info("SSBD partial %s A/D=%d/%d path=%s",
                                     tgt, r["accepted"], r["draft_len"], r["path"])
                        else:
                            translations[tgt] = self.mt.translate(
                                text, tgt=tgt, src=lid or "auto",
                                context=self._context(), terms=self.terms or None)
                        self._ssbd_drafts[(seg_id, tgt)] = translations[tgt]
                    except Exception as e:
                        translations[tgt] = f"[MT error: {e}]"
                # stale (segment finalized meanwhile)? drop, don't resurrect.
                if seg_id is not None and seg_id != self._seg_id:
                    return
                self._last_tr_text = text
                self._last_tr_t = time.time()
                if text == self._last_partial or self._tr_pending is not None:
                    self.results.put(("partial", {
                        "id": seg_id, "speaker": self._seg_speaker,
                        "rms_db": -80.0, "text": text,
                        "translations": translations, "final": False,
                    }))
        finally:
            self._tr_busy = False

    # -- finalize ---------------------------------------------------------------
    def _finalize(self, t: float):
        seg = np.concatenate(self._seg_audio) if self._seg_audio else np.zeros(0, np.float32)
        vad_frac = (self._seg_vad_sum / self._seg_vad_n
                    if self._seg_vad_n else 1.0)
        t_start = time.time()
        denoised = False
        skip_denoise = (bool(self.cfg.get("vad_skip_denoise", False))
                        and vad_frac >= 0.9)
        if self._want_denoise() and self.enhancer is not None and seg.size \
                and not skip_denoise:
            seg = self.enhancer.enhance_array(seg, 16000)
            denoised = self.enhancer.active
        # Final text: one-shot ORT re-decode when available (also lets the
        # denoised segment be heard fresh), else the live-stream result.
        # Either way seg is what the diarizer/translator see below.
        asr_backend = "stream"
        text = ""
        rd = getattr(self.asr, "redecode_cuda", None)
        if callable(rd) and seg.size:
            try:
                cuda_text, cuda_backend = rd(seg, 16000)
            except Exception as e:
                log.warning("ORT redecode failed (%s).", e)
                cuda_text, cuda_backend = None, None
            if cuda_text:
                text, asr_backend = cuda_text, cuda_backend or "ort"
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
        t_asr = time.time()
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
        t_diar = time.time()
        # Skip MT where the target IS the detected source (a copy, not a
        # ~3 s generate). detect() is ms-level; None = unsure = translate all.
        lid = detect_lang(text)
        src = lid or "auto"
        translations: dict[str, str] = {}
        for tgt in self.targets:
            if lid and tgt.lower() == lid:
                translations[tgt] = text
        to_translate = [t for t in self._ordered_targets()
                        if not (lid and t.lower() == lid)]
        if self.vad is not None and vad_frac < self.vad_min_frac:
            # Junk segment (cough, keyboard, HVAC): keep the transcript row,
            # skip every GPU generate. VAD-OR already keeps most of these
            # from opening segments at all; this catches the rest.
            log.info("VAD junk skip (frac=%.2f): MT skipped for %d chars",
                     vad_frac, len(text))
            to_translate = []
        self.history.append(text)
        pending = {
            "id": self._seg_id, "type": "utterance", "speaker": speaker,
            "rms_db": round(rms_dbfs(seg), 1) if seg.size else -80.0,
            "denoised": denoised, "src_lang": lid or "auto", "text": text,
            "translations": dict(translations), "start": round(self._seg_start or t, 2),
            "asr_backend": asr_backend, "pending": True,
            "timings": {
                "asr_ms": int((t_asr - t_start) * 1000),
                "diar_ms": int((t_diar - t_asr) * 1000),
                "mt_ms": 0,
                "total_ms": int((t_diar - t_start) * 1000),
            },
        }
        if diar_backend:
            pending["diar_backend"] = diar_backend
        # ASR text leaves NOW; MT catches up in the background worker below.
        self.results.put(("utterance", pending))
        self._submit_mt(
            seg_id=self._seg_id, text=text, targets=to_translate, src=src,
            context=context, terms=terms, t_mt_start=t_diar,
            base=dict(pending),
        )
        self._n_final += 1
        self._fresh_stream()
        self._reset_segment()

    # -- background MT --------------------------------------------------------
    def _submit_mt(self, seg_id, text, targets, src, context, terms,
                   t_mt_start, base: dict):
        """Queue a translation job; one FIFO daemon thread runs them in order.

        Emits tok events live, then re-emits the utterance with full
        translations (same id → UI upserts). Feed/ASR never block on MT.
        """
        import queue as _queue
        import threading

        if self._mt_queue is None:
            self._mt_queue = _queue.Queue()
            self._mt_thread = threading.Thread(target=self._mt_loop, daemon=True)
            self._mt_thread.start()
        if not targets:
            base["pending"] = False
            base["translations"] = dict(base.get("translations", {}))
            self.results.put(("utterance", base))
            return
        self._mt_queue.put({
            "seg_id": seg_id, "text": text, "targets": list(targets),
            "src": src, "context": list(context or []),
            "terms": dict(terms) if terms else None,
            "t_mt_start": t_mt_start, "base": base,
        })

    def _mt_loop(self):
        while True:
            job = self._mt_queue.get()
            try:
                if job is None:  # poison pill (see stop())
                    return
                self._run_mt_job(job)
            except Exception as e:
                log.warning("MT job failed: %s", e)
                try:
                    base = job.get("base", {}) if isinstance(job, dict) else {}
                    base["pending"] = False
                    self.results.put(("utterance", base))
                except Exception:
                    pass
            finally:
                try:
                    self._mt_queue.task_done()
                except Exception:
                    pass

    def _emit_mt_update(self, job: dict, translations: dict[str, str],
                          pending: bool):
        """Re-emit the utterance with translations merged so far.

        Builds a fresh snapshot each time — the queue may hold several
        emits per job and they must not alias each other.
        """
        t_mt = time.time()
        base = job["base"]
        if "base_total" not in job:
            job["base_total"] = base.get("timings", {}).get("total_ms", 0)
        merged = dict(base.get("translations", {}))  # LID copies, fixed
        merged.update(translations)
        timings = dict(base.get("timings", {}))
        timings["mt_ms"] = int((t_mt - job["t_mt_start"]) * 1000)
        timings["total_ms"] = job["base_total"] + timings["mt_ms"]
        ev = dict(base)
        ev["translations"] = merged
        ev["pending"] = pending
        ev["timings"] = timings
        self.results.put(("utterance", ev))

    def _stream_one_target(self, seg_id: int, text: str, tgt: str, src: str,
                           context, terms) -> str:
        """Token-stream a single target (tok events); return assembled text.

        When SSBD is enabled and a draft exists (the segment's last partial
        translation), the accepted prefix lands as one instant delta and only
        the divergent suffix decodes step-by-step.
        """
        parts: list[str] = []
        seq = 0
        draft = self._ssbd_drafts.get((seg_id, tgt))
        ssbd = getattr(self.mt, "translate_ssbd_stream", None)
        stream = None
        if (draft and callable(ssbd)
                and bool(self.cfg.get("mt_ssbd_enable", False))):
            stream = self.mt.translate_ssbd_stream(
                text, draft, tgt=tgt, src=src, context=context, terms=terms)
        try:
            it = stream if stream is not None else self.mt.translate_stream(
                text, tgt=tgt, src=src, context=context, terms=terms)
            for delta in it:
                if delta:
                    parts.append(delta)
                    self.results.put(("tok", {"id": seg_id,
                                              "tgt": tgt, "seq": seq,
                                              "delta": delta}))
                    seq += 1
        except Exception as e:
            log.warning("streamed translate (%s) failed: %s", tgt, e)
        if stream is not None:
            info = getattr(self.mt, "_last_ssbd", {}) or {}
            log.info("SSBD final %s A/D=%s/%s path=%s", tgt,
                     info.get("accepted"), info.get("draft_len"),
                     info.get("path"))
        chunk = "".join(parts).strip()
        out = chunk or self.mt.translate(
            text, tgt=tgt, src=src, context=context, terms=terms)
        self._ssbd_drafts.pop((seg_id, tgt), None)  # draft consumed
        if len(self._ssbd_drafts) > 200:  # stale segments (never finalized)
            self._ssbd_drafts.clear()
        return out

    def _run_mt_job(self, job: dict):
        seg_id = job["seg_id"]
        text, src = job["text"], job["src"]
        context, terms = job["context"], job["terms"]
        translations: dict[str, str] = {}
        try:
            # Batched (one generate for all targets) is ~10x SLOWER on this
            # modeling (no early stop: runs all 256 steps) — sequential with
            # display-language-first stays default. mt_batch=true re-enables
            # the batched path if the modeling ever gets fixed.
            if bool(self.cfg.get("mt_batch", False)):
                parts: dict[str, list[str]] = {}
                seqs: dict[str, int] = {}
                for tgt, delta in self.mt.translate_targets_stream(
                        text, job["targets"], src=src,
                        context=context, terms=terms):
                    if delta:
                        parts.setdefault(tgt, []).append(delta)
                        seq = seqs.get(tgt, 0)
                        self.results.put(("tok", {"id": seg_id, "tgt": tgt,
                                                  "seq": seq, "delta": delta}))
                        seqs[tgt] = seq + 1
                for tgt in job["targets"]:
                    chunk = "".join(parts.get(tgt, [])).strip()
                    translations[tgt] = chunk or self.mt.translate(
                        text, tgt=tgt, src=src, context=context, terms=terms)
                self._emit_mt_update(job, translations, pending=False)
                return
            # Two-phase: display language streams + lands FIRST (the row users
            # read), remaining targets follow in the same job and re-emit the
            # utterance when done. Live latency = 1 generate, not N.
            ordered = list(job["targets"])  # already display-first
            phase1, phase2 = ordered[:1], ordered[1:]
            for tgt in phase1:
                translations[tgt] = self._stream_one_target(
                    seg_id, text, tgt, src, context, terms)
            self._emit_mt_update(job, translations, pending=bool(phase2))
            for tgt in phase2:
                translations[tgt] = self._stream_one_target(
                    seg_id, text, tgt, src, context, terms)
                # progressive update per deferred target (cheap put, same id)
                self._emit_mt_update(job, translations, pending=True)
            if phase2:
                self._emit_mt_update(job, translations, pending=False)
        except Exception as e:
            for tgt in job["targets"]:
                translations[tgt] = f"[MT error: {e}]"
            self._emit_mt_update(job, translations, pending=False)

    def _fresh_stream(self):
        try:
            self._stream = self._rec.create_stream()
        except Exception as e:
            log.warning("stream reset failed: %s", e)

    def stop(self) -> int:
        """Finalize any pending speech, then wait for background MT to catch
        up so file_done implies complete translations. Returns number of
        finalized segments."""
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
        try:
            if self.vad is not None:
                self.vad.reset()
        except Exception:
            pass
        self._ssbd_drafts.clear()
        if self._mt_queue is not None:
            deadline = time.time() + 300
            while time.time() < deadline:
                if self._mt_queue.unfinished_tasks == 0:
                    break
                time.sleep(0.2)
            else:
                log.warning("MT backlog not drained in 300 s — translations may land late")
        return n
