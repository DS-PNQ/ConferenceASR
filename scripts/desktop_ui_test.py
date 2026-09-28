"""Desktop UI construction test: builds the real window hidden, feeds a fake
utterance through the worker result queue, and asserts bubbles render.

Uses fake engines (no model downloads). The window is withdrawn immediately
so nothing visible pops up; run when a display session exists.
"""
import queue
import time


class FakeEngine:
    model_id = "fake"

    def status(self):
        return {"model": "fake", "backend": "fake", "ready": True}


class FakePipe:
    history = []

    def export_txt(self):
        return "txt"

    def export_srt(self):
        return "srt"


class FakeWorker:
    def __init__(self):
        self.tasks = queue.Queue()
        self.results = queue.Queue()

    def start(self):
        pass

    def warmup(self):
        self.results.put(("ready", {"asr": FakeEngine().status(),
                                    "mt": FakeEngine().status(),
                                    "denoise": {"backend": "fake"}}))

    def submit(self, task):
        self.tasks.put(task)


def main():
    from desktop.ui import ConferenceApp

    ctx = {"cfg": {"sample_rate": 16000, "segment_seconds": 4.0,
                   "default_targets": ["en", "zh"]},
           "device": {"device": "cpu"},
           "asr": FakeEngine(), "mt": FakeEngine(),
           "diarizer": None, "enhancer": None,
           "pipe": FakePipe(), "worker": FakeWorker()}
    app = ConferenceApp(ctx)
    app.withdraw()
    app.update()

    # sidebar + feed built?
    assert app.btn_record is not None and app.feed is not None
    assert app.mic_var.get() == "System default"

    # fake utterance through the real result pump
    app.ctx["worker"].results.put(("utterance", {
        "id": 1, "type": "utterance", "speaker": "SPEAKER_01", "rms_db": -12.0,
        "denoised": True, "src_lang": "en", "text": "hello world",
        "translations": {"zh": "你好世界"}, "start": 0.0}))
    for _ in range(40):
        app.update()
        time.sleep(0.02)
    assert app.n_utterances == 1, app.n_utterances
    assert "1 utterances" in app.count_lbl.cget("text")
    bubbles = [w for w in app.feed.winfo_children() if "frame" in w.winfo_class().lower()]
    assert len(bubbles) >= 1, "no bubble rendered"
    print(f"UI render OK ({len(bubbles)} bubble widgets)")

    # streaming partial appears live, then is replaced by its finalized bubble
    app.ctx["worker"].results.put(("partial", {
        "id": 2, "speaker": "SPEAKER_02", "rms_db": -14.0,
        "text": "NO OTHER THE MACON IS", "translations": None, "final": False}))
    for _ in range(10):
        app.update()
        time.sleep(0.02)
    assert getattr(app, "_partial", None) is not None, "partial bubble missing"
    assert getattr(app, "_partial")["id"] == 2
    # live translation streams into the same bubble
    app.ctx["worker"].results.put(("partial", {
        "id": 2, "speaker": "SPEAKER_02", "rms_db": -14.0,
        "text": "NO OTHER THE MACON IS DEMONSTRATED",
        "translations": {"en": "No other Macon is demonstrated"}, "final": False}))
    for _ in range(10):
        app.update()
        time.sleep(0.02)
    assert len(app._partial["trs"]) == 1, "streaming translation missing"
    app.ctx["worker"].results.put(("utterance", {
        "id": 2, "type": "utterance", "speaker": "SPEAKER_02", "rms_db": -14.0,
        "denoised": False, "src_lang": "auto",
        "text": "NO OTHER THE MACON IS DEMONSTRATED EXCELLENCE",
        "translations": {"en": "No other Macon demonstrates excellence."},
        "start": 5.0}))
    for _ in range(40):
        app.update()
        time.sleep(0.02)
    assert app.n_utterances == 2, app.n_utterances
    assert getattr(app, "_partial", None) is None, "partial not replaced by final"
    print("partial -> final replacement OK")

    # settings plumbing
    s = app._settings()
    assert set(s) == {"targets", "src_lang", "denoise", "device"}, s
    print("settings OK:", s)

    app.destroy()
    print("DESKTOP UI TEST PASSED")


if __name__ == "__main__":
    main()
