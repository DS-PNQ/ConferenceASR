"""Microphone capture for the desktop app (sounddevice, 16 kHz mono).

A sounddevice callback thread pushes raw blocks into a queue; a feeder thread
slices them into short frames and submits them as ("stream_audio", ...) tasks
for the worker's streaming session. The UI polls :attr:`level` (0..1) for the
volume meter. No inference happens here.
"""
from __future__ import annotations

import logging
import queue
import threading

import numpy as np

log = logging.getLogger("conf.mic")


def list_input_devices() -> list[tuple[int, str]]:
    """[(index, name)] for devices with at least one input channel."""
    import sounddevice as sd

    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append((i, d.get("name", f"device {i}")))
    return out


class MicRecorder:
    def __init__(self, task_queue: "queue.Queue", settings_fn,
                 samplerate: int = 16000, frame_seconds: float = 0.5,
                 chunk_seconds: float | None = None):
        """
        Args:
            task_queue: frames go here as ("stream_audio", pcm float32, sr).
                Open the worker session first with ("stream_start", settings).
            settings_fn: zero-arg callable returning
                dict(targets=[...], src_lang=...|None, denoise=bool|None,
                     device=int|None).
        """
        self.task_queue = task_queue
        self.settings_fn = settings_fn
        self.samplerate = int(samplerate)
        # small frames keep the streaming recognizer live; endpointing (not
        # the clock) decides segment boundaries. chunk_seconds kept as a
        # deprecated alias.
        self.frame_seconds = float(frame_seconds if chunk_seconds is None else chunk_seconds)
        self._blocks: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=256)
        self._stop = threading.Event()
        self._stream = None
        self._feeder: threading.Thread | None = None
        self._level = 0.0
        self._lock = threading.Lock()
        self.running = False

    # -- sounddevice callback (never block here) ---------------------------
    def _callback(self, indata, frames, time_info, status):
        try:
            mono = np.asarray(indata, dtype=np.float32).ravel().copy()
            peak = float(np.abs(mono).max()) if mono.size else 0.0
            with self._lock:
                self._level = min(1.0, peak * 2.2)
            try:
                self._blocks.put_nowait(mono)
            except queue.Full:
                pass
        except Exception:
            pass

    # -- feeder: slice blocks into short frames -> task_queue ----------------
    def _feed(self):
        need = max(1, int(self.samplerate * self.frame_seconds))
        buf = np.zeros(0, dtype=np.float32)
        while not self._stop.is_set():
            try:
                block = self._blocks.get(timeout=0.2)
            except queue.Empty:
                continue
            buf = np.concatenate([buf, block]) if buf.size else block
            while buf.size >= need and not self._stop.is_set():
                frame, buf = buf[:need].copy(), buf[need:].copy()
                self.task_queue.put(("stream_audio", frame, self.samplerate))

    def start(self):
        if self.running:
            return
        import sounddevice as sd

        try:
            s = self.settings_fn()
        except Exception:
            s = {}
        self._stop.clear()
        self._blocks = queue.Queue(maxsize=256)
        self._stream = sd.InputStream(
            samplerate=self.samplerate, channels=1, dtype="float32",
            blocksize=4096, callback=self._callback,
            device=s.get("device"),
        )
        self._stream.start()
        self._feeder = threading.Thread(target=self._feed, daemon=True)
        self._feeder.start()
        self.running = True
        log.info("mic recording started")

    def stop(self):
        self._stop.set()
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None
        self.running = False
        with self._lock:
            self._level = 0.0
        log.info("mic recording stopped")

    @property
    def level(self) -> float:
        with self._lock:
            return self._level
