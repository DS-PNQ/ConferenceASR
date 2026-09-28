"""Background inference worker: owns StreamingSessions off the UI thread.

Tasks (all tuples):
    ("warmup",)                                   load all models now
    ("stream_start", settings)                    open a live session
    ("stream_audio", pcm, sr)                     feed frames to the session
    ("stream_stop",)                              endpoint + close session
    ("file", pcm, sr, targets, src_lang, denoise) run a recording through a session
    ("stop",)                                     end the thread

Results (consumed by the UI via ``result_queue``):
    ("status", str)            human-readable state for the status line
    ("ready", dict)            model statuses after warmup
    ("partial", {...})         live transcript (+ debounced translations)
    ("utterance", entry)       finalized segment (same shape as pipeline entries)
    ("file_done", n)           upload finished with n utterances
    ("error", str)             something failed (kept running)
"""
from __future__ import annotations

import logging
import queue
import threading
import time

import numpy as np

from app.streaming import StreamingSession

log = logging.getLogger("conf.worker")


class InferenceWorker(threading.Thread):
    def __init__(self, cfg, pipeline, asr, mt, diarizer, enhancer, demo_ok: bool = True):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.pipe = pipeline
        self.asr = asr
        self.mt = mt
        self.diarizer = diarizer
        self.enhancer = enhancer
        self.demo_ok = demo_ok
        self.tasks: "queue.Queue" = queue.Queue()
        self.results: "queue.Queue" = queue.Queue()
        self.session: StreamingSession | None = None

    # -- task helpers -------------------------------------------------------
    def submit(self, task: tuple):
        self.tasks.put(task)

    def warmup(self):
        self.submit(("warmup",))

    # -- main loop ----------------------------------------------------------
    def run(self):
        while True:
            task = self.tasks.get()
            kind = task[0]
            try:
                if kind == "stop":
                    self._close_session()
                    return
                if kind == "warmup":
                    self._do_warmup()
                elif kind == "stream_start":
                    self._close_session()
                    self.session = StreamingSession(
                        self.cfg, self.asr, self.mt, self.diarizer,
                        self.enhancer, self.results)
                    _, settings = task
                    self.session.configure(
                        targets=settings.get("targets"),
                        src_lang=settings.get("src_lang"),
                        denoise=settings.get("denoise"))
                    self.results.put(("status", "● Listening — streaming ASR + translation."))
                elif kind == "stream_audio":
                    if self.session is not None:
                        _, pcm, sr = task
                        self.session.feed(pcm, sr)
                elif kind == "stream_stop":
                    self._close_session()
                    self.results.put(("status", "Stopped."))
                elif kind == "file":
                    _, pcm, sr, targets, src_lang, denoise = task
                    self._run_file(pcm, sr, targets, src_lang, denoise)
            except Exception as e:
                log.warning("worker task %s failed: %s", kind, e)
                self.results.put(("error", str(e)))

    def _close_session(self):
        if self.session is not None:
            try:
                self.session.stop()
            except Exception as e:
                log.warning("session stop failed: %s", e)
            self.session = None

    def _run_file(self, pcm, sr, targets, src_lang, denoise):
        """Stream a whole recording through a session (endpointing included)."""
        self._close_session()
        self.results.put(("status", "Transcribing file…"))
        session = StreamingSession(
            self.cfg, self.asr, self.mt, self.diarizer, self.enhancer, self.results)
        try:
            session.configure(targets=targets, src_lang=src_lang, denoise=denoise)
        except Exception as e:
            self.results.put(("error", f"ASR unavailable: {e}"))
            return
        a = np.asarray(pcm, dtype=np.float32).ravel()
        frame = int(sr * 1.0)
        t0 = time.time() - len(a) / sr
        n = 0
        for i in range(0, max(len(a), 1), max(frame, 1)):
            session.feed(a[i:i + frame], sr, t=t0 + i / sr)
            n = session._n_final
        done = session.stop()
        self.results.put(("file_done", max(n, done)))

    def _do_warmup(self):
        self.results.put(("status", "Loading models (first run downloads weights)…"))
        t0 = time.time()
        try:
            self.asr.ensure_loaded(demo_ok=self.demo_ok)
            self.mt.ensure_loaded(demo_ok=self.demo_ok)
            if self.enhancer is not None:
                self.enhancer.ensure_loaded(demo_ok=self.demo_ok)
            info = {"asr": self.asr.status(), "mt": self.mt.status(),
                    "denoise": self.enhancer.status() if self.enhancer else {}}
            self.results.put(("ready", info))
            self.results.put(("status",
                              f"Ready in {time.time()-t0:.0f}s — ASR {info['asr']['model']} · "
                              f"MT {info['mt']['model']} · denoise {info['denoise'].get('backend','?')}. "
                              "Press Record."))
        except Exception as e:
            self.results.put(("error", f"Model load failed: {e}"))
