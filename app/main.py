"""FastAPI server: landing + live conference studio + REST/WS inference API."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from pathlib import Path

import yaml
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from .audio_io import chunk_stream, decode_b64_pcm16, decode_bytes
from .device import device_report, resolve_device, resolve_dtype
from .diarizer import make_diarizer
from .enhancer import DeepFilterNetEnhancer
from .mt_engine import HyMT2Engine
from .pipeline import ConferencePipeline
from .streaming import StreamingSession
from .zipformer_engine import ZipformerEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
log = logging.getLogger("conf.server")

ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "config.yaml"


def load_cfg() -> dict:
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


CFG = load_cfg()
DEVICE = resolve_device(os.getenv("DEVICE", CFG.get("device", "auto")))
MT_DTYPE = resolve_dtype(DEVICE, os.getenv("MT_DTYPE", CFG.get("mt_dtype", "auto")))

asr = ZipformerEngine(CFG)
mt = HyMT2Engine(CFG, DEVICE, MT_DTYPE)
diarizer = make_diarizer(CFG)
enhancer = DeepFilterNetEnhancer(CFG)
pipe = ConferencePipeline(CFG, asr, mt, diarizer, enhancer)

DEMO = os.getenv("DEMO_MODE", str(CFG.get("demo_mode", "auto"))).lower()
if DEMO == "true":
    asr.ensure_loaded(demo_ok=True)
    mt.ensure_loaded(demo_ok=True)
    enhancer.ensure_loaded(demo_ok=True)

app = FastAPI(title="Trilingual Conference ASR + Translator", version="1.0.0")
# Local desktop clients (Electron renderer, Tauri, browser tabs) call the API
# from non-http origins. The server binds 127.0.0.1 only, so this is safe.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(ROOT / "web")), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(ROOT / "web" / "index.html"))


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "device": device_report(CFG),
        "asr": asr.status(),
        "mt": mt.status(),
        "langs": CFG.get("supported_langs", ["vi", "en", "zh"]),
        "diarizer": type(diarizer).__name__,
        "denoise": enhancer.status(),
    }


def _parse_denoise(v) -> bool | None:
    """Per-request denoise override: True/False, or None to use server default."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return None


@app.post("/api/warmup")
def warmup():
    """Load models now (else they lazy-load on first request)."""
    demo_ok = DEMO != "false"
    asr.ensure_loaded(demo_ok=demo_ok)
    mt.ensure_loaded(demo_ok=demo_ok)
    enhancer.ensure_loaded(demo_ok=demo_ok)
    return {"asr": asr.status(), "mt": mt.status(), "denoise": enhancer.status()}


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    targets: str = Form("en,zh"),
    src_lang: str = Form("auto"),
    denoise: str = Form("auto"),
):
    raw = await file.read()
    try:
        pcm, sr = decode_bytes(raw, file.content_type or "")
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    tgt_list = [t.strip() for t in targets.split(",") if t.strip()]
    entries = pipe.process_file(pcm, sr, targets=tgt_list,
                                src_lang=None if src_lang == "auto" else src_lang,
                                denoise=_parse_denoise(denoise))
    return {"ok": True, "utterances": entries, "asr": asr.status(),
            "denoise": enhancer.status()}


@app.post("/api/translate")
async def translate(payload: dict):
    text = payload.get("text", "")
    targets = payload.get("targets", ["en"])
    src = payload.get("src", "auto")
    return {"ok": True, "translations": mt.translate_multi(text, targets, src=src)}


@app.get("/api/export.txt")
def export_txt():
    return PlainTextResponse(pipe.export_txt(), media_type="text/plain")


@app.get("/api/export.srt")
def export_srt():
    return PlainTextResponse(pipe.export_srt(), media_type="text/plain")


async def _ws_stream_msg(ws: WebSocket, data: dict, holder: dict):
    """Streaming protocol for the C# client. holder['session'] persists."""
    import queue as _queue

    kind = data.get("type")
    if kind == "stream_start":
        if holder["session"] is not None:
            try:
                holder["session"].stop()
            except Exception:
                pass
        q: _queue.Queue = _queue.Queue()
        session = StreamingSession(CFG, asr, mt, diarizer, enhancer, q)
        try:
            # configure() may download/load the ASR model: keep it off the
            # event loop so WS pings and other clients stay responsive.
            await asyncio.to_thread(
                session.configure,
                targets=data.get("targets") or list(CFG.get("default_targets", ["en", "zh"])),
                src_lang=data.get("src_lang") or None,
                denoise=_parse_denoise(data.get("denoise", "auto")),
            )
        except Exception as e:
            await ws.send_json({"ok": False, "error": f"stream start failed: {e}"})
            return
        holder["session"] = session
        await ws.send_json({"ok": True, "type": "stream_started",
                            "asr": asr.status(), "mt": mt.status()})
        return
    session = holder.get("session")
    if session is None:
        await ws.send_json({"ok": False, "error": "no stream open (send stream_start first)"})
        return
    if kind == "stream_audio":
        try:
            pcm, sr = decode_b64_pcm16(data.get("audio_b64", ""), data.get("sr", 16000))
        except Exception as e:
            await ws.send_json({"ok": False, "error": f"bad audio: {e}"})
            return
        try:
            # feed() runs ASR decode + MT generate (seconds on GPU): off the
            # event loop, otherwise WS ping/pong stalls and clients drop us.
            await asyncio.to_thread(session.feed, pcm, sr)
        except Exception as e:
            await ws.send_json({"ok": False, "error": f"decode failed: {e}"})
            return
    elif kind == "stream_stop":
        try:
            n = await asyncio.to_thread(session.stop)
        except Exception as e:
            await ws.send_json({"ok": False, "error": f"stop failed: {e}"})
            return
        holder["session"] = None
    # drain session events (partial / utterance) without blocking
    try:
        while True:
            ev_kind, payload = session.results.get_nowait()
            if ev_kind == "partial":
                await ws.send_json({"ok": True, "type": "partial", **payload})
            elif ev_kind == "utterance":
                await ws.send_json({"ok": True, "type": "utterance", **payload})
    except _queue.Empty:
        pass
    if kind == "stream_stop":
        await ws.send_json({"ok": True, "type": "stream_stopped", "finalized": n})


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    """Two protocols, chosen per message:

    Legacy (browser demo): {audio_b64, sr, targets, src_lang, denoise} with
    ~segment_seconds of PCM16 per message -> fixed-window utterances.

    Streaming (C# desktop client): {"type": "stream_start", targets, src_lang,
    denoise} opens a StreamingSession, then {"type": "stream_audio", audio_b64,
    sr} feeds short frames, then {"type": "stream_stop"}. Server emits
    ("partial", ...) live events and ("utterance", ...) finals from the
    session's result queue.
    """
    await ws.accept()
    holder: dict = {"session": None}
    try:
        while True:
            msg = await ws.receive_text()
            try:
                data = json.loads(msg)
            except Exception:
                await ws.send_json({"ok": False, "error": "send JSON, not text"})
                continue
            if data.get("type") == "ping":
                await ws.send_json({"ok": True, "type": "pong", "asr": asr.status()})
                continue
            if data.get("type") in ("stream_start", "stream_audio", "stream_stop"):
                await _ws_stream_msg(ws, data, holder)
                continue
            try:
                pcm, sr = decode_b64_pcm16(data.get("audio_b64", ""), data.get("sr", 16000))
            except Exception as e:
                await ws.send_json({"ok": False, "error": f"bad audio: {e}"})
                continue
            # Split long buffers into pipeline segments
            seg = float(CFG.get("segment_seconds", 5.0))
            import time

            denoise = _parse_denoise(data.get("denoise", "auto"))
            t0 = time.time()
            for ch in chunk_stream(pcm, sr, seg):
                entry = pipe.process_chunk(
                    ch, sr,
                    targets=data.get("targets") or list(CFG.get("default_targets", ["en", "zh"])),
                    src_lang=data.get("src_lang") or None,
                    t=t0,
                    denoise=denoise,
                )
                t0 += len(ch) / sr
                if entry and entry.get("type") == "utterance":
                    await ws.send_json({"ok": True, **entry})
                elif entry:
                    await ws.send_json({"ok": True, **entry})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.warning("ws error: %s", e)
        try:
            await ws.close()
        except Exception:
            pass
