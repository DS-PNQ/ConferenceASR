"""Backend up (py -3 run.py): py -3 scripts/bench_latency.py [denoise 0|1] [snr_db|none]
E2E latency over /ws/live, like the UI: TTS sentences (en/vi/zh) with pauses,
0.5 s frames in real time. Per sentence: speech end -> final text (`final`),
all translations (`done`); `tok` only if MT tokens stream for finals."""
import asyncio, base64, io, json, os, sys, time
import numpy as np, soundfile as sf, websockets, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from app.engines import load_cfg
from app.tts_engine import PiperTTS
from app.enhancer import _resample

DENOISE = sys.argv[1] == "1" if len(sys.argv) > 1 else True
SNR = None if len(sys.argv) < 3 or sys.argv[2] == "none" else float(sys.argv[2])
S = [("en", "Today we will review the quarterly budget and discuss the new deployment schedule."),
     ("vi", "Hôm nay chúng ta sẽ xem xét ngân sách quý này và thảo luận về lịch triển khai mới."),
     ("zh", "今天我们将审查本季度的预算，并讨论服务器集群的新部署时间表。"),
     ("en", "The second release fixes the memory leak in the streaming decoder."),
     ("vi", "Tuần sau đội kỹ thuật sẽ báo cáo kết quả kiểm thử hiệu năng."),
     ("zh", "我们需要在下个月之前完成所有的测试工作。")] * 2
tts = PiperTTS(load_cfg())
GAP = 16000 * 2
parts, spans, pos = [np.zeros(16000, np.float32)], [], 16000
for lang, text in S:
    y, sr = sf.read(io.BytesIO(tts.synth(text, lang)), dtype="float32")
    y = _resample(y, sr, 16000) * 0.5
    y = y[: np.nonzero(np.abs(y) > 0.01)[0][-1] + 1]  # trim Piper's trailing silence: span end = speech end
    spans.append((pos / 16000, (pos + y.size) / 16000)); pos += y.size + GAP
    parts += [y, np.zeros(GAP, np.float32)]
pcm = np.concatenate(parts)
if SNR is not None:
    nz = np.random.default_rng(0).standard_normal(pcm.size).astype(np.float32)
    pcm = pcm + nz * np.sqrt(np.mean(np.concatenate(parts[1::2]) ** 2) / 10 ** (SNR / 10))

async def main():
    urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/api/warmup", method="POST"), timeout=600).read()
    ev = []  # (wall, msg)
    async with websockets.connect("ws://127.0.0.1:8000/ws/live", max_size=None) as ws:
        await ws.send(json.dumps({"type": "stream_start", "targets": ["en", "vi", "zh"], "src_lang": None,
                                  "denoise": DENOISE, "terms": {}, "display_lang": "en"}))
        while json.loads(await ws.recv()).get("type") != "stream_started": pass
        async def rx():
            async for m in ws:
                d = json.loads(m); ev.append((time.time(), d))
                if d.get("type") == "stream_stopped": return
        r = asyncio.create_task(rx())
        T0 = time.time()
        for i in range(0, pcm.size, 8000):
            f = (np.clip(pcm[i:i + 8000], -1, 1) * 32767).astype(np.int16)
            await asyncio.sleep(max(0, T0 + i / 16000 - time.time()))
            await ws.send(json.dumps({"type": "stream_audio", "audio_b64": base64.b64encode(f.tobytes()).decode(), "sr": 16000}))
        await asyncio.sleep(1); await ws.send(json.dumps({"type": "stream_stop"})); await r
    # frame containing sample k is sent at T0 + floor(k/8000)*0.5, so a sentence's last
    # sample reaches the server at the send time of its frame (+0.5 s of frame buffering).
    rows = {}
    for w, d in ev:
        if d.get("id") is None: continue
        r = rows.setdefault(d["id"], {})
        t = d.get("type")
        if t == "partial": r.setdefault("partial", w)
        if t == "tok": r.setdefault("tok", w)
        if t == "utterance":
            r.setdefault("final", w); r["text"] = d["text"]; r["start"] = d["start"]
            r["timings"] = d.get("timings")
            if not d.get("pending"): r["done"] = w
    out = []
    for r in rows.values():
        if "final" not in r: continue
        st = r["start"] - T0
        i = min(range(len(spans)), key=lambda k: abs(spans[k][0] - st))
        end_sent = T0 + spans[i][1]  # wall time the last speech sample was spoken (real-time clock)
        out.append((i, {k: r[k] - end_sent for k in ("final", "tok", "done") if k in r}, r["text"], r["timings"]))
    out.sort()
    for i, l, txt, tm in out:
        print(f"s{i:02d} {S[i][0]}  final {l['final']:5.2f}s  tok {l.get('tok', float('nan')):5.2f}s  done {l.get('done', float('nan')):5.2f}s  {tm}  {txt}")
    import statistics as st
    for k in ("final", "tok", "done"):
        v = [l[k] for _, l, _, _ in out if k in l]
        if not v: continue
        print(f"{k:5s}: median {st.median(v):.2f}s  max {max(v):.2f}s  (n={len(v)})")
    print(f"rows {len(out)} for {len(S)} sentences")
    import difflib, re
    norm = lambda t: re.sub(r"[\W_]+", "", t.lower())
    cer = [1 - difflib.SequenceMatcher(None, norm(txt), norm(S[i][1])).ratio() for i, _, txt, _ in out]
    print(f"ASR CER vs script: mean {sum(cer) / len(cer):.3f}  " + " ".join(f"{c:.2f}" for c in cer))
asyncio.run(main())
