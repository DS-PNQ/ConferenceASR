"""Long-meeting soak: real ASR + MT + diarizer + VAD through StreamingSession.

    py -3 scripts/soak_long.py [minutes=65] [speed=1] [wav]

speed=1 paces frames in real time (the honest test: MT competes with live
speech exactly as in a meeting). speed>1 overloads MT on purpose and so
exercises backlog shedding. Asserts the session stays bounded over the run:
memory, threads, segment buffer, MT backlog, translation lag, no lost finals.
"""
import os
import queue
import sys
import threading
import time

import numpy as np
import psutil
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.audio_io import _to_16k  # noqa: E402
from app.engines import build_app  # noqa: E402
from app.streaming import StreamingSession  # noqa: E402
from app.vad import SileroVAD  # noqa: E402

MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 65.0
SPEED = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
WAV = sys.argv[3] if len(sys.argv) > 3 else "C:/Users/Asus/AppData/Local/Temp/opencode/asr_en.wav"
SR, FRAME = 16000, 8000  # 0.5 s frames, like the Electron mic


def meeting_audio(clip: np.ndarray):
    """Endless conference shape: clip, 2 s pause, and every 8th loop a
    60 s silence (break) with low room noise."""
    rng = np.random.default_rng(0)
    i = 0
    while True:
        yield clip
        yield (rng.standard_normal(SR * 2) * 1e-4).astype(np.float32)
        if i % 8 == 7:
            yield (rng.standard_normal(SR * 60) * 1e-4).astype(np.float32)
        i += 1


def main():
    ctx = build_app()
    cfg = ctx["cfg"]
    ctx["asr"].ensure_loaded(demo_ok=False)
    ctx["mt"].ensure_loaded(demo_ok=False)
    vad = SileroVAD(cfg)
    clip, sr = sf.read(WAV, dtype="float32")
    clip = clip.mean(axis=1) if clip.ndim > 1 else clip
    clip, _ = _to_16k(np.asarray(clip, np.float32), sr)

    q: queue.Queue = queue.Queue()
    s = StreamingSession(cfg, ctx["asr"], ctx["mt"], ctx["diarizer"],
                         ctx["enhancer"], q, vad=vad)
    s.configure(targets=["vi", "en", "zh"], display_lang="en")

    first_seen: dict[int, float] = {}
    lag: dict[int, tuple[float, float]] = {}  # id -> (t_first, lag_s)
    counts = {"partial": 0, "tok": 0, "utterance": 0, "mt_error": 0, "untranslated": 0}
    stop_ev = threading.Event()

    def consume():  # stands in for main._pump_events
        while not stop_ev.is_set() or not q.empty():
            try:
                kind, p = q.get(timeout=0.2)
            except queue.Empty:
                continue
            counts[kind] = counts.get(kind, 0) + 1
            if kind != "utterance":
                continue
            now = time.time()
            first_seen.setdefault(p["id"], now)
            if not p.get("pending"):
                lag[p["id"]] = (first_seen[p["id"]], now - first_seen[p["id"]])
                tr = p.get("translations") or {}
                tm = p.get("timings") or {}
                if tm.get("mt_ms", 0) > 15000 or tm.get("diar_ms", 0) > 2000 or tm.get("asr_ms", 0) > 3000:
                    print(f"  slow seg {p['id']}: {tm} src={len(p['text'])}ch "
                          f"tr={ {k: len(v) for k, v in tr.items()} }", flush=True)
                if any("[MT error" in v for v in tr.values()):
                    counts["mt_error"] += 1
                if not any(k != p.get("src_lang") for k in tr):
                    counts["untranslated"] += 1

    consumer = threading.Thread(target=consume, daemon=True)
    consumer.start()

    proc = psutil.Process()
    samples = []  # (audio_min, rss_mb, threads, backlog, seg_s)
    total = int(MINUTES * 60 * SR)
    fed, buf, t_wall0, max_backlog = 0, np.zeros(0, np.float32), time.time(), 0
    t_audio0 = time.time()
    next_sample = 0
    print(f"soak: {MINUTES:.0f} min audio at {SPEED}x, targets vi/en/zh, "
          f"mt={ctx['mt'].status().get('backend', ctx['mt'].status().get('device'))}", flush=True)
    for chunk in meeting_audio(clip):
        buf = np.concatenate([buf, chunk])
        while len(buf) >= FRAME and fed < total:
            s.feed(buf[:FRAME], SR, t=t_audio0 + fed / SR)
            buf, fed = buf[FRAME:], fed + FRAME
            max_backlog = max(max_backlog, s.mt_backlog)
            ahead = fed / SR / SPEED - (time.time() - t_wall0)
            if ahead > 0:
                time.sleep(ahead)
            if fed >= next_sample:
                samples.append((fed / SR / 60, proc.memory_info().rss / 2**20,
                                threading.active_count(), s.mt_backlog,
                                sum(len(a) for a in s._seg_audio) / SR))
                print("t=%5.1f min  rss=%6.0f MB  threads=%3d  backlog=%2d  seg=%4.1fs  "
                      "finals=%d  late=%.1fs" % (*samples[-1], len(lag),
                                                 time.time() - t_wall0 - fed / SR / SPEED),
                      flush=True)
                next_sample += 60 * SR
        if fed >= total:
            break
    n_tail = s.stop()
    time.sleep(1)
    stop_ev.set()
    consumer.join(10)

    lost = set(first_seen) - set(lag)
    by_time = sorted(lag.values())
    lags = [l for _, l in by_time]

    def p95(xs):
        return float(np.percentile(xs, 95)) if xs else 0.0

    k = max(1, len(lags) // 6)
    early, late = p95(lags[:k]), p95(lags[-k:])
    warm = [x for x in samples if x[0] >= 5] or samples
    rss_growth = warm[-1][1] - warm[0][1]
    thread_growth = max(x[2] for x in warm) - warm[0][2]
    print(f"\nsegments={len(first_seen)} finals={len(lag)} lost={len(lost)} tail={n_tail}")
    print(f"events: {counts}")
    print(f"MT lag p95: first sixth {early:.1f}s, last sixth {late:.1f}s; max backlog {max_backlog}")
    print(f"RSS growth after warmup: {rss_growth:+.0f} MB; thread growth {thread_growth:+d}; "
          f"max open segment {max(x[4] for x in samples):.1f}s")

    limit = 2 * int(cfg.get("mt_max_backlog", 4))
    assert len(first_seen) > MINUTES * 1.5, "too few segments finalized"
    assert not lost, f"{len(lost)} segments never got a final event"
    assert counts["mt_error"] == 0, "MT errors in finals"
    assert max_backlog <= limit + 1, f"MT backlog {max_backlog} exceeded shedding limit {limit}"
    assert late <= max(45.0, 2 * early), f"translation lag grows over time ({early:.1f}s -> {late:.1f}s)"
    assert rss_growth < 400, f"memory grows: +{rss_growth:.0f} MB"
    assert thread_growth <= 4, f"thread leak: +{thread_growth}"
    assert max(x[4] for x in samples) <= float(cfg.get("max_segment", 20)) + 1, "segment buffer unbounded"
    print("SOAK PASSED")


if __name__ == "__main__":
    main()
