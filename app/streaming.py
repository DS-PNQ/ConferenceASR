"""Streaming conference session: continuous ASR + live re-translation.

Fixed 5 s windows slice sentences mid-word. Instead, one sherpa-onnx
streaming recognizer stream stays open while audio flows in, and segments
are finalized on *silence* (endpointing) — utterances break at pauses::

    mic/file frames -> recognizer stream -> partial text (live on screen)
                     -> silence >= endpoint_silence -> finalize:
                        diarize segment -> utterance event INSTANTLY (ASR text,
                        LID copies filled) -> MT runs in a background FIFO
                        worker -> utterance event again with full
                        translations. ASR never waits for MT.

With a streaming diarizer (Nemotron) every frame is also scored per 10 ms:
the open row's speaker label follows it live, and a hand-over to another
voice finalizes the words before the change as their own row (_split) instead
of waiting for a pause.

The growing sentence is re-translated while it is spoken, each run drafting
off the previous one with Self-Speculative Biased Decoding (SSBD, COLM 2026):
the final only verifies the last draft and decodes from the first divergence.
"""
from __future__ import annotations

import itertools
import logging
import re
import threading
import time
from collections import deque

import numpy as np

from .diarizer import rms_dbfs
from .langid import detect as detect_lang

log = logging.getLogger("conf.stream")

# Process-wide segment ids: clients upsert rows by id, so a second session
# restarting at the same number would overwrite the first session's rows.
_SEG_IDS = itertools.count(1_000_000)

# Words for re-translation pacing: one per CJK char, one per whitespace word otherwise.
_UNIT = re.compile(r"[\u3400-\u9fff]|[^\s\u3400-\u9fff]+")


class StreamingSession:
    def __init__(self, cfg, asr_engine, mt_engine, diarizer, enhancer,
                 results_queue, vad=None):
        self.cfg = cfg or {}
        self.asr = asr_engine
        self.mt = mt_engine
        self.diarizer = diarizer
        self.enhancer = enhancer
        self.vad = vad  # SileroVAD or None (None = energy-only, old behavior)
        self.results = results_queue
        self._ids = _SEG_IDS

        self.endpoint_silence = float(cfg.get("endpoint_silence", 1.0))
        self.min_speech = float(cfg.get("endpoint_min_speech", 0.5))
        self.max_segment = float(cfg.get("max_segment", 20.0))
        # Re-translate the live sentence every N new words (SSBD drafts): a 15 s
        # utterance from scratch is ~2 s per target on the RTX 4060, verifying
        # a fresh draft ~0.1-0.6 s.
        self.retr_words = int(cfg.get("mt_retranslate_words", 3))
        # MT slower than speech = unbounded lag over a long meeting. Past this
        # many queued finals: display language only; past 2x: skip MT.
        self.max_backlog = int(cfg.get("mt_max_backlog", 4))
        self._idle = 0.0  # silence (s) with no segment open

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
        # Streaming diarizer (Nemotron): sample clock shared with it, so frames
        # and recognizer token times can be addressed by diarizer sample.
        self._live_diar = False
        self.split_slack = float(cfg.get("split_token_slack", 0.12))
        self._pos_end = 0         # diarizer sample after the last fed frame
        self._stream_s0 = None    # diarizer sample where the recognizer stream began
        self._reset_segment()
        self._n_final = 0
        self._mt_queue = None  # lazy background FIFO worker, see _submit_mt
        self._mt_thread = None
        # seg_id -> {tgt: latest live translation}: the next run's SSBD draft
        self._drafts: dict = {}
        self._n_retr = 0  # queued/running live re-translation jobs

    # -- setup --------------------------------------------------------------
    def configure(self, targets=None, src_lang=None, denoise=None, terms=None,
                  display_lang=None, resume=False):
        if not resume:
            # The diarizer is process-wide: without this, a new meeting's
            # voices get matched to (and capped by) the last meeting's
            # speaker centroids. resume = client reconnect, same meeting.
            self.diarizer.reset()
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
        self._stream = self.asr.new_stream(self.terms)
        if callable(getattr(self.diarizer, "push", None)):
            self.diarizer.ensure_loaded()  # streaming diarizer: up before the first frame
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
    def _mt_unfinished(self) -> int:
        q = self._mt_queue
        return int(q.unfinished_tasks) if q is not None else 0

    @property
    def mt_backlog(self) -> int:
        """Unfinished final-translation jobs (live re-translations excluded:
        a segment's end skips its queued ones)."""
        return max(0, self._mt_unfinished - self._n_retr)

    def _ordered_targets(self) -> list[str]:
        """Display language first so its tokens stream earliest."""
        if self.display_lang and self.display_lang in self.targets:
            return [self.display_lang] + [t for t in self.targets
                                          if t != self.display_lang]
        return list(self.targets)

    def _reset_segment(self):
        # Set when the segment ends: cuts its in-flight live re-translation
        # and skips queued ones, so the final job gets the GPU at once.
        self._seg_stop = threading.Event()
        self._seg_audio: list[np.ndarray] = []
        self._seg_dur = 0.0
        self._voice_dur = 0.0
        self._trailing_sil = 0.0
        self._seg_vad_sum = 0.0  # neural speech probs (VAD-OR + junk gate)
        self._seg_vad_n = 0
        self._seg_start: float | None = None
        self._seg_s0: int | None = None  # diarizer sample of the segment's first frame
        self._seg_speaker = "SPEAKER_01"
        self._seg_id: int | None = None
        self._last_partial = ""
        self._tr_text = ""  # partial text last sent for re-translation
        self._tr_words = 0
        self._tr_rounds = 0

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

        t0 = time.perf_counter()
        push = getattr(self.diarizer, "push", None)
        off = push(x) if callable(push) else None
        self._live_diar = off is not None
        if off is not None:
            if self._stream_s0 is None:
                self._stream_s0 = off
            self._pos_end = off + x.size
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
        t_vad = time.perf_counter()
        silent = energy_silent or not vad_speech

        if self._seg_start is None and not silent:
            self._seg_start = t
            self._seg_speaker = dia.get("speaker", "SPEAKER_01")
            self._seg_s0 = off
            self._seg_id = next(self._ids)
        if self._seg_start is None:
            self._idle = self._idle + dur if silent else 0.0
        else:
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
        t_dec = time.perf_counter()

        if self._seg_start is not None:
            if self._live_diar and self._seg_s0 is not None:
                self._track_speaker(t)
            self._maybe_emit_partial(t, silent)
            if (silent and self._voice_dur >= self.min_speech
                    and self._trailing_sil >= self.endpoint_silence):
                self._finalize(t)
            elif self._seg_dur >= self.max_segment and self._voice_dur >= self.min_speech:
                self._finalize(t)
            elif silent and self._trailing_sil >= self.endpoint_silence:
                # blip shorter than min_speech, then quiet: drop it. Kept open,
                # _seg_audio grew for as long as the room stayed silent and the
                # next utterance re-decoded all of it.
                self._discard_segment()
        elif self._idle >= self.endpoint_silence:
            # idle stream: reset so noise decoded during silence never prefixes
            # the next segment and the recognizer state stays short.
            self._fresh_stream()
            self._idle = 0.0
        t_end = time.perf_counter()
        if t_end - t0 > 2 * dur:  # slower than realtime: say which stage
            log.warning("slow frame %.2fs: diar+vad %.2f decode %.2f finalize/partial %.2f",
                        t_end - t0, t_vad - t0, t_dec - t_vad, t_end - t_dec)

    # -- streaming diarizer: live label + split at speaker change -------------
    def _track_speaker(self, t: float):
        info = self.diarizer.turn(self._seg_s0)
        spk = info.get("speaker")
        if info.get("change_at") is not None:
            if self._split(int(info["change_at"]), info["next_speaker"], t):
                return
            spk = info["next_speaker"]  # nothing said before the change: relabel
        if spk and spk != self._seg_speaker:
            self._seg_speaker = spk
            if self._last_partial:  # caption shows the new label now
                self.results.put(("partial", {
                    "id": self._seg_id, "speaker": spk, "rms_db": -80.0,
                    "text": self._last_partial,
                    "translations": self._drafts.get(self._seg_id), "final": False}))

    def _split(self, tc: int, new_speaker: str, t: float) -> bool:
        """Finalize the words before diarizer sample `tc` as their own row and
        carry the audio after it into a new segment for `new_speaker`.
        Recognizer token times decide which words fall before the cut."""
        seg = np.concatenate(self._seg_audio) if self._seg_audio else np.zeros(0, np.float32)
        k = tc - self._seg_s0
        if k < 4800 or k >= seg.size or self._stream_s0 is None:  # < 0.3 s head: relabel only
            return False
        try:
            r = self._rec.get_result_all(self._stream)
            # tokens are stamped when emitted, after the audio: see split_token_slack
            cut = (tc - self._stream_s0) / 16000.0 + self.split_slack
            head ="".join(tok for tok, ts in zip(r.tokens, r.timestamps) if ts < cut).strip()
        except Exception as e:
            log.warning("speaker split: token times unavailable (%s)", e)
            return False
        if not head:
            return False
        tail = seg[k:].copy()
        tail_start = (self._seg_start or t) + k / 16000.0
        self._seg_audio = [seg[:k]]
        self._trailing_sil = 0.0
        self._finalize(t, text_override=head)  # resets the segment + recognizer stream
        self._stream_s0 = tc
        self._seg_start, self._seg_s0 = tail_start, tc
        self._seg_id = next(self._ids)
        self._seg_speaker = new_speaker
        self._seg_audio = [tail]
        self._seg_dur = self._voice_dur = tail.size / 16000.0
        try:
            self._stream.accept_waveform(16000, np.clip(tail, -1.0, 1.0))
            while self._rec.is_ready(self._stream):
                self._rec.decode_stream(self._stream)
        except Exception as e:
            log.warning("speaker split: tail decode failed (%s)", e)
        log.info("speaker change -> %s: split at %.2fs into the segment", new_speaker, k / 16000.0)
        return True

    def _discard_segment(self):
        self._seg_stop.set()
        if self._last_partial:  # clear the live caption in the UI
            self.results.put(("partial", {
                "id": self._seg_id, "speaker": self._seg_speaker, "rms_db": -80.0,
                "text": "", "translations": None, "final": False}))
        self._drafts.pop(self._seg_id, None)
        self._fresh_stream()
        self._reset_segment()

    # -- partials ---------------------------------------------------------------
    def _maybe_emit_partial(self, t: float, silent: bool = False):
        try:
            text = self._rec.get_result(self._stream).strip()
        except Exception:
            return
        if text and text != self._last_partial:
            self._last_partial = text
            # Text goes out IMMEDIATELY (realtime ASR) with the latest live
            # translation; it never blocks on MT.
            self.results.put(("partial", {
                "id": self._seg_id, "speaker": self._seg_speaker,
                "rms_db": round(rms_dbfs(np.concatenate(self._seg_audio[-4:])), 1)
                if self._seg_audio else -80.0,
                "text": text, "translations": self._drafts.get(self._seg_id),
                "final": False,
            }))
        n = len(_UNIT.findall(self._last_partial))
        # Every few new words mid-speech (only when MT is idle), and on every
        # silent frame: the endpoint wait then works on the final's draft (if
        # finalize cuts it short, the partial output is still a valid draft).
        if self._last_partial == self._tr_text or not (
                silent or (n - self._tr_words >= self.retr_words and self._mt_unfinished == 0)):
            return
        self._tr_text, self._tr_words = self._last_partial, n
        context = self._context() if n >= 5 else None
        # Mid-speech the display language (fresh draft for the row users read),
        # every 3rd round and every pause all targets. One job per target: a
        # final waits on one, not all.
        self._tr_rounds += 1
        for tgt in self._ordered_targets()[:None if silent or self._tr_rounds % 3 == 0 else 1]:
            self._submit_job({"retr": tgt, "seg_id": self._seg_id, "stop": self._seg_stop,
                              "text": self._last_partial,
                              "context": context, "terms": dict(self.terms) if self.terms else None})

    def _translate(self, text, tgt, src, context, terms, draft=None, stop=None) -> str:
        """SSBD off `draft` when there is one, plain otherwise. lower(): the ASR
        emits ALL-CAPS English, which Hy-MT translates worse. MEASURED: history
        attached to 1-4 word sources makes Hy-MT translate the history instead."""
        if len(_UNIT.findall(text)) < 5:
            context = None
        kw = {"stop": stop} if stop else {}
        ssbd = getattr(self.mt, "translate_ssbd", None)
        if draft and callable(ssbd):
            with self.mt._lock:
                return ssbd(text.lower(), draft, tgt=tgt, src=src,
                            context=context, terms=terms, **kw)["text"]
        return self.mt.translate(text.lower(), tgt=tgt, src=src,
                                 context=context, terms=terms, **kw)

    def _run_retr_job(self, job: dict):
        seg_id, tgt, text = job["seg_id"], job["retr"], job["text"]
        if job["stop"].is_set():  # segment ended meanwhile: its final job redoes this
            return
        lid = detect_lang(text)
        if len(self._drafts) > 50:  # discarded segments never pop theirs
            self._drafts = {k: v for k, v in self._drafts.items() if k == self._seg_id}
        drafts = self._drafts.setdefault(seg_id, {})
        drafts[tgt] = text if (lid and tgt.lower() == lid) else self._translate(
            text, tgt, lid or "auto", job["context"], job["terms"], drafts.get(tgt),
            stop=job["stop"])  # cut short = shorter draft, still a valid one
        if seg_id == self._seg_id and self._last_partial:  # still live: refresh caption
            self.results.put(("partial", {
                "id": seg_id, "speaker": self._seg_speaker, "rms_db": -80.0,
                "text": self._last_partial, "translations": dict(drafts),
                "final": False}))

    # -- finalize ---------------------------------------------------------------
    def _finalize(self, t: float, text_override: str | None = None):
        """text_override: the head's words when _split cuts mid-stream (the
        recognizer result also holds the next speaker's words)."""
        self._seg_stop.set()
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
        # Final text: the live-stream result (already decoded, 0 ms). The
        # one-shot ORT re-decode (asr_cuda_final) costs 2-7 s per segment
        # under live MT load (measured) and blocks feed() meanwhile.
        asr_backend = "stream"
        text = ""
        rd = getattr(self.asr, "redecode_cuda", None)
        if callable(rd) and seg.size and bool(self.cfg.get("asr_cuda_final", False)):
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
                    s2 = self.asr.new_stream(self.terms)
                    s2.accept_waveform(16000, np.clip(seg, -1.0, 1.0))
                    s2.accept_waveform(16000, np.zeros(int(16000 * 0.4), dtype=np.float32))
                    s2.input_finished()
                    while self._rec.is_ready(s2):
                        self._rec.decode_stream(s2)
                    text = self._rec.get_result(s2).strip()
                except Exception as e:
                    log.warning("enhanced re-decode failed: %s", e)
                    text = self._last_partial
            elif text_override is not None:
                text = text_override
            else:
                try:
                    text = self._rec.get_result(self._stream).strip()
                except Exception:
                    text = self._last_partial
        if not text:
            self._discard_segment()
            return
        text = self.asr.correct(text, self.terms)
        t_asr = time.time()
        if t_asr - t_start > 3:
            log.warning("slow finalize ASR: %.1fs for %.1fs audio", t_asr - t_start, seg.size / 16000)
        context = self._context()
        terms = self.terms or None
        # Neural speaker verdict: NeMo embedding attribution overrides the
        # volume guess used for live partials (no-op on volume diarizers).
        speaker = self._seg_speaker
        diar_backend = None
        try:
            attr = getattr(self.diarizer, "attribute_segment", None)
            if self._live_diar and self._seg_s0 is not None:
                # streaming diarizer already scored this audio frame by frame
                voiced = max(1, seg.size - int(self._trailing_sil * 16000))
                spk = self.diarizer.turn(self._seg_s0, self._seg_s0 + voiced).get("speaker")
                if spk:
                    speaker, diar_backend = spk, "nemotron"
            elif callable(attr) and seg.size:
                # trailing endpoint silence only dilutes the voice embedding
                voiced = seg[:max(1, seg.size - int(self._trailing_sil * 16000))]
                verdict = attr(voiced, 16000)
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
    def _submit_job(self, job: dict):
        import queue as _queue
        import threading

        if self._mt_queue is None:
            self._mt_queue = _queue.Queue()
            self._mt_thread = threading.Thread(target=self._mt_loop, daemon=True)
            self._mt_thread.start()
        if "retr" in job:
            self._n_retr += 1  # ponytail: unlocked counter, off-by-one only skews shedding
        self._mt_queue.put(job)

    def _submit_mt(self, seg_id, text, targets, src, context, terms,
                   t_mt_start, base: dict):
        """Queue a translation job; one FIFO daemon thread runs them in order.

        Re-emits the utterance with full translations (same id → UI
        upserts). Feed/ASR never block on MT.
        """
        backlog = self.mt_backlog
        if targets and backlog >= 2 * self.max_backlog:
            log.warning("MT backlog %d: skipping MT for segment %s", backlog, seg_id)
            targets = []
        elif len(targets) > 1 and backlog >= self.max_backlog:
            log.warning("MT backlog %d: display language only for %s", backlog, seg_id)
            targets = targets[:1]  # display-first order (_ordered_targets)
        if not targets:
            base["pending"] = False
            base["translations"] = dict(base.get("translations", {}))
            self.results.put(("utterance", base))
            self._drafts.pop(seg_id, None)
            return
        self._submit_job({
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
                if "retr" in job:
                    self._run_retr_job(job)
                else:
                    self._run_mt_job(job)
            except Exception as e:
                log.warning("MT job failed: %s", e)
                if isinstance(job, dict) and "base" in job:
                    job["base"]["pending"] = False
                    self.results.put(("utterance", job["base"]))
            finally:
                try:
                    self._mt_queue.task_done()
                except Exception:
                    pass
                if isinstance(job, dict) and "retr" in job:
                    self._n_retr -= 1

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
            # Display language lands FIRST (the row users read); each other
            # target re-emits the utterance (same id) as it completes.
            ordered = list(job["targets"])  # already display-first
            for i, tgt in enumerate(ordered):
                translations[tgt] = self._translate(
                    text, tgt, src, context, terms,
                    (self._drafts.get(seg_id) or {}).get(tgt))
                self._emit_mt_update(job, translations, pending=i < len(ordered) - 1)
        except Exception as e:
            for tgt in job["targets"]:
                translations[tgt] = f"[MT error: {e}]"
            self._emit_mt_update(job, translations, pending=False)
        finally:
            self._drafts.pop(seg_id, None)

    def _fresh_stream(self):
        self._stream_s0 = self._pos_end if self._live_diar else None
        try:
            self._stream = self.asr.new_stream(self.terms)
        except Exception as e:
            log.warning("stream reset failed: %s", e)

    def flush(self):
        """Finalize the open segment now (mic paused / stopping)."""
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

    def stop(self) -> int:
        """Finalize any pending speech, then wait for background MT to catch
        up so file_done implies complete translations. Returns number of
        finalized segments."""
        self.flush()
        n = self._n_final
        self._n_final = 0
        try:
            if self.vad is not None:
                self.vad.reset()
        except Exception:
            pass
        if self._mt_queue is not None:
            deadline = time.time() + 300
            while time.time() < deadline:
                if self._mt_queue.unfinished_tasks == 0:
                    break
                time.sleep(0.2)
            else:
                log.warning("MT backlog not drained in 300 s — translations may land late")
            self._mt_queue.put(None)  # end the worker thread (one per session)
        return n

    def close(self):
        """Client gone: drop queued MT jobs and end the worker thread, so a
        dead session never holds the MT lock against the next one."""
        import queue as _queue

        q = self._mt_queue
        if q is None:
            return
        try:
            while True:
                q.get_nowait()
                q.task_done()
        except _queue.Empty:
            pass
        q.put(None)
