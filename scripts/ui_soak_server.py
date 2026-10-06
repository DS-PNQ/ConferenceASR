"""UI soak: fake backend that replays an hour-long meeting into the real UI.

    py -3 scripts/ui_soak_server.py [segments=900] [ms_per_segment=40]
    then open http://127.0.0.1:8000/ and press Start recording.

Serves electron/ui/dist/index.html plus /api/health, /api/warmup and the
/ws/live streaming protocol (partials, tok deltas, pending -> final utterance
upserts, 3 targets) — no models, so it isolates renderer cost. 900 segments
~= 1 h of meeting. Mic frames from the page are accepted and ignored.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

N = int(sys.argv[1]) if len(sys.argv) > 1 else 900
MS = float(sys.argv[2]) if len(sys.argv) > 2 else 40
UI = Path(__file__).resolve().parent.parent / "electron" / "ui" / "dist" / "index.html"
app = FastAPI()
SENT = "the quarterly results show strong growth in the asia pacific region and we expect "


@app.get("/")
def index():
    return FileResponse(UI)


@app.get("/api/health")
def health():
    st = {"model": "mock", "ready": True}
    return {"ok": True, "device": {"device": "cpu"}, "asr": st, "mt": st, "langs": ["vi", "en", "zh"],
            "diarizer": "mock", "diarizer_status": {"backend": "mock", "ready": True},
            "denoise": {"backend": "off", "ready": True}}


@app.post("/api/warmup")
def warmup():
    return {"ok": True}


async def replay(ws: WebSocket):
    t0 = time.time() - N * 4.0  # server clock: 4 s of meeting per segment
    for i in range(N):
        sid, spk = 2_000_000 + i, f"SPEAKER_{i % 3 + 1:02d}"
        text = (SENT * 2)[: 60 + (i * 37) % 100] + f" #{i}"
        for k in (20, 40, 60):
            await ws.send_json({"ok": True, "type": "partial", "id": sid, "speaker": spk, "rms_db": -20,
                                "text": text[:k], "translations": None, "final": False})
        base = {"ok": True, "type": "utterance", "id": sid, "speaker": spk, "rms_db": -20.0,
                "denoised": False, "src_lang": "en", "text": text, "translations": {"en": text},
                "start": t0 + i * 4.0, "asr_backend": "mock",
                "timings": {"asr_ms": 300, "diar_ms": 50, "mt_ms": 0, "total_ms": 350}}
        await ws.send_json({**base, "pending": True})
        vi = f"bản dịch tiếng Việt số {i} " * 3
        for j in range(0, len(vi), 4):  # ~20 tok deltas
            await ws.send_json({"ok": True, "type": "tok", "id": sid, "tgt": "vi", "seq": j, "delta": vi[j:j + 4]})
        await ws.send_json({**base, "translations": {"en": text, "vi": vi}, "pending": True})
        await ws.send_json({**base, "translations": {"en": text, "vi": vi, "zh": f"中文翻译 {i} " * 4},
                            "pending": False})
        await asyncio.sleep(MS / 1000)
    print(f"replayed {N} segments", flush=True)


@app.websocket("/ws/live")
async def live(ws: WebSocket):
    await ws.accept()
    task = None
    try:
        while True:
            m = json.loads(await ws.receive_text())
            if m.get("type") == "stream_start":
                await ws.send_json({"ok": True, "type": "stream_started"})
                task = asyncio.create_task(replay(ws))
            elif m.get("type") == "stream_stop":
                await ws.send_json({"ok": True, "type": "stream_stopped", "finalized": N})
    except WebSocketDisconnect:
        pass
    finally:
        if task:
            task.cancel()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
