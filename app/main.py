"""FastAPI inference server for the Electron/C# desktop clients.

API-only (no bundled UI): REST (/api/*) + streaming WebSocket (/ws/live).
Interactive docs at /docs when running.
"""
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
from fastapi.responses import JSONResponse, PlainTextResponse

from .audio_io import chunk_stream, decode_b64_pcm16, decode_bytes
from .device import device_report, resolve_device, resolve_dtype
from .diarizer import make_diarizer
from .enhancer import DeepFilterNetEnhancer
from .engines import make_mt_engine
from .ocr_engine import BlockOCR, PaddleOCRVLEngine
from .pipeline import ConferencePipeline
from .streaming import StreamingSession
from .vad import SileroVAD
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
mt = make_mt_engine(CFG, DEVICE, MT_DTYPE)
diarizer = make_diarizer(CFG)
enhancer = DeepFilterNetEnhancer(CFG)
ocr = PaddleOCRVLEngine(CFG)  # lazy: loads on first /api/ocr, never in warmup
blockocr = BlockOCR(CFG)  # PP-OCR boxes+scores (CPU); lazy like ocr
vad = SileroVAD(CFG)  # neural speech gate; fail-open when unavailable
pipe = ConferencePipeline(CFG, asr, mt, diarizer, enhancer)
# Runtime-swappable diarizer (POST /api/settings). New streaming sessions and
# the legacy pipeline resolve through here; in-flight sessions keep theirs.
RUNTIME: dict = {"diarizer": diarizer}


def _diarizer_status() -> dict:
    d = RUNTIME.get("diarizer", diarizer)
    status = getattr(d, "status", None)
    if callable(status):
        try:
            return status()
        except Exception:
            pass
    return {"backend": type(d).__name__, "ready": False}

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
@app.get("/")
def index():
    return {
        "service": "ConfLive inference API",
        "frontends": ["electron/", "csharp/ConfLive"],
        "docs": "/docs",
        "health": "/api/health",
        "websocket": "/ws/live",
    }


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "device": device_report(CFG),
        "asr": asr.status(),
        "mt": mt.status(),
        "langs": CFG.get("supported_langs", ["vi", "en", "zh"]),
        "diarizer": type(RUNTIME.get("diarizer", diarizer)).__name__,
        "diarizer_status": _diarizer_status(),
        "denoise": enhancer.status(),
        "ocr": ocr.status(),
        "vad": vad.status(),
    }


def _parse_terms(v) -> dict[str, str]:
    """Glossary pairs {source: target}. Accepts dict, JSON string, or null."""
    if not v:
        return {}
    if isinstance(v, dict):
        return {str(k): str(w) for k, w in v.items() if k and w}
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return {str(k): str(w) for k, w in parsed.items() if k and w}
    return {}


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
    try:
        asr._ort().ensure_loaded()  # final re-decoder: else ~10 s on the first segment
    except Exception as e:
        log.warning("ORT re-decoder warmup failed: %s", e)
    mt.ensure_loaded(demo_ok=demo_ok)
    enhancer.ensure_loaded(demo_ok=demo_ok)
    try:
        vad.ensure_loaded(demo_ok=demo_ok)  # preloaded: no first-frame stall
    except Exception as e:
        log.warning("vad warmup failed: %s", e)
    d = RUNTIME.get("diarizer", diarizer)
    ensure = getattr(d, "ensure_loaded", None)
    if callable(ensure):
        try:
            ensure(demo_ok=demo_ok)
        except Exception as e:
            log.warning("diarizer warmup failed: %s", e)
    return {"asr": asr.status(), "mt": mt.status(), "denoise": enhancer.status(),
            "diarizer": _diarizer_status()}


@app.post("/api/settings")
def settings(payload: dict):
    """Runtime settings. Currently: {"diarizer": "pyannote"|"nemo"|"volume"}.

    Applies to new streaming sessions and the legacy pipeline immediately;
    in-flight sessions keep theirs. Affects /api/health output.
    """
    out: dict = {"ok": True}
    which = (payload or {}).get("diarizer")
    if which is not None:
        mode = str(which).strip().lower()
        if mode not in ("pyannote", "volume", "nemo"):
            return JSONResponse({"ok": False,
                                 "error": "diarizer must be 'pyannote', 'nemo' or 'volume'"},
                                status_code=400)
        cfg = dict(CFG)
        cfg["diarizer"] = mode
        cfg["use_nemo"] = (mode == "nemo")
        try:
            new_d = make_diarizer(cfg)
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"diarizer init failed: {e}"},
                                status_code=500)
        RUNTIME["diarizer"] = new_d
        pipe.diarizer = new_d
        out["diarizer"] = _diarizer_status()
    return out


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    targets: str = Form("en,zh"),
    src_lang: str = Form("auto"),
    denoise: str = Form("auto"),
    terms: str = Form("{}"),
):
    raw = await file.read()
    try:
        pcm, sr = decode_bytes(raw, file.content_type or "")
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    tgt_list = [t.strip() for t in targets.split(",") if t.strip()]
    entries = pipe.process_file(pcm, sr, targets=tgt_list,
                                src_lang=None if src_lang == "auto" else src_lang,
                                denoise=_parse_denoise(denoise),
                                terms=_parse_terms(terms))
    return {"ok": True, "utterances": entries, "asr": asr.status(),
            "denoise": enhancer.status()}


@app.post("/api/translate")
async def translate(payload: dict):
    text = payload.get("text", "")
    targets = payload.get("targets", ["en"])
    src = payload.get("src", "auto")
    return {"ok": True, "translations": mt.translate_multi(
        text, targets, src=src, terms=_parse_terms(payload.get("terms")))}


def _enter_ocr_mode() -> dict:
    """OCR owns the GPU: speaker diarization off + deloaded from VRAM.

    The diarizer reloads lazily on the next stream (embed -> ensure_loaded),
    so live sessions keep working after OCR without a restart.
    """
    d = RUNTIME.get("diarizer", diarizer)
    unload = getattr(d, "unload", None)
    if callable(unload):
        try:
            return unload()
        except Exception as e:
            log.warning("diarizer unload failed: %s", e)
    return {"unloaded": False}


def _ocr_pages(raw: bytes, filename: str, dpi: int) -> list:
    """Decode an upload to PIL pages: images as-is, PDFs rasterized."""
    import io

    from PIL import Image

    name = (filename or "").lower()
    if name.endswith(".pdf"):
        import fitz  # PyMuPDF

        pages = []
        with fitz.open(stream=raw, filetype="pdf") as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi)
                pages.append(Image.open(io.BytesIO(pix.tobytes("png"))))
        return pages
    return [Image.open(io.BytesIO(raw))]


def _preview_data_url(img, max_w: int = 900) -> str:
    """Downscaled JPEG data URL so box overlays align with the transform."""
    import base64
    import io

    im = img.convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return ("data:image/jpeg;base64,"
            + base64.b64encode(buf.getvalue()).decode("ascii"))


def _apply_transform(img, rotate: int, crop: list | None):
    """Bake client toolbar state: rotate (deg CW) then relative crop."""
    if crop:
        try:
            x0, y0, x1, y1 = (max(0.0, min(1.0, float(v))) for v in crop[:4])
            if x1 > x0 and y1 > y0:
                w, h = img.size
                img = img.crop((int(x0 * w), int(y0 * h),
                                int(x1 * w), int(y1 * h)))
        except Exception as e:
            log.warning("crop ignored (%s)", e)
    rotate = int(rotate or 0) % 360
    if rotate:
        # PIL rotates CCW; toolbar degrees are CW
        img = img.rotate(-rotate, expand=True)
    return img


def _translate_long(text: str, target: str, src: str, terms: dict) -> str:
    """Hy-MT over long OCR text: chunk (live MT caps output length), join."""
    import re

    width = max(200, int(CFG.get("ocr_translate_chunk_chars", 800)))
    parts = re.split(r"(?<=[.!?。！？\n])\s+", text)
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) + 1 > width and cur.strip():
            chunks.append(cur.strip())
            cur = p
        else:
            cur = (cur + " " + p).strip()
    if cur.strip():
        chunks.append(cur.strip())
    if not chunks:
        return ""
    mt.ensure_loaded(demo_ok=True)
    return "\n".join(
        mt.translate(c, tgt=target, src=src, terms=terms or None) for c in chunks
    ).strip()


@app.post("/api/ocr")
async def ocr_docs(
    files: list[UploadFile] = File(...),
    task: str = Form("ocr"),
    translate_to: str = Form(""),
    src: str = Form("auto"),
    terms: str = Form("{}"),
    rotate: int = Form(0),
    crop: str = Form(""),
):
    """OCR images/PDFs, optionally Hy-MT-translated.

    Form: files (1+, image/* or .pdf), task (ocr|table|formula|chart),
    translate_to ("" = none, else e.g. "en"), src, terms (glossary JSON),
    rotate (deg CW baked before OCR), crop ("[x0,y0,x1,y1]" relative).
    task=ocr uses PP-OCR blocks (boxes+scores); table/formula/chart use the
    PaddleOCR-VL reader (text only). Entering OCR mode deloads the diarizer.
    """
    import time

    task = (task or "ocr").strip().lower()
    target = (translate_to or "").strip().lower()
    terms_d = _parse_terms(terms)
    src_lang = None if (src or "auto") == "auto" else src
    try:
        crop_rect = (json.loads(crop) if crop else None) or None
    except Exception:
        crop_rect = None
    dpi = max(72, min(400, int(CFG.get("ocr_dpi", 200))))
    deloaded = _enter_ocr_mode()
    use_blocks = (task == "ocr")
    if use_blocks:
        try:
            await asyncio.to_thread(blockocr.ensure_loaded)
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"block ocr init failed: {e}"},
                                status_code=500)
        if not blockocr.available:
            use_blocks = False  # fall through to VLM text-only
    if not use_blocks:
        try:
            await asyncio.to_thread(ocr.ensure_loaded, True)
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"ocr load failed: {e}"},
                                status_code=500)
    mt.ensure_loaded(demo_ok=True)
    pages_out: list[dict] = []
    try:
        for f in files:
            raw = await f.read()
            try:
                pages = await asyncio.to_thread(
                    _ocr_pages, raw, f.filename or "upload", dpi)
            except Exception as e:
                pages_out.append({"file": f.filename, "page": 0,
                                  "error": f"decode failed: {e}"})
                continue
            for i, img in enumerate(pages):
                t0 = time.time()
                try:
                    work = await asyncio.to_thread(
                        _apply_transform, img, rotate, crop_rect)
                    preview = _preview_data_url(work)
                except Exception as e:
                    pages_out.append({"file": f.filename, "page": i + 1,
                                      "error": f"transform failed: {e}"})
                    continue
                entry: dict = {"file": f.filename, "page": i + 1,
                               "preview": preview}
                try:
                    if use_blocks:
                        res = await asyncio.to_thread(
                            blockocr.read_blocks, work)
                        entry.update(res)
                        entry["backend"] = "pp-ocr"
                    else:
                        entry["text"] = await asyncio.to_thread(
                            ocr.read, work, task)
                        entry["blocks"] = []
                        entry["overall_conf"] = 0.0
                        entry["backend"] = "paddleocr-vl"
                except Exception as e:
                    entry["error"] = f"ocr failed: {e}"
                    pages_out.append(entry)
                    continue
                entry["ocr_ms"] = int((time.time() - t0) * 1000)
                if target and entry.get("text"):
                    try:
                        entry["translation"] = await asyncio.to_thread(
                            _translate_long, entry["text"], target,
                            src_lang or "auto", terms_d)
                        for b in entry.get("blocks", []):
                            if b.get("text"):
                                try:
                                    b["translation"] = mt.translate(
                                        b["text"], tgt=target,
                                        src=src_lang or "auto",
                                        terms=terms_d or None)
                                except Exception as e:
                                    b["translation_error"] = str(e)
                    except Exception as e:
                        entry["translation_error"] = str(e)
                pages_out.append(entry)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True, "task": task, "pages": pages_out, "ocr": ocr.status(),
            "block_ocr": {"ready": blockocr.loaded,
                          "available": blockocr.available},
            "diarizer": _diarizer_status(), "diarizer_deloaded": deloaded}


@app.get("/api/export.txt")
def export_txt():
    return PlainTextResponse(pipe.export_txt(), media_type="text/plain")


@app.get("/api/export.srt")
def export_srt():
    return PlainTextResponse(pipe.export_srt(), media_type="text/plain")


async def _ws_stream_msg(ws: WebSocket, data: dict, holder: dict):
    """Streaming protocol. holder['session'] persists; its events go to
    holder['q'], which _pump_events sends on its own (not per message)."""
    kind = data.get("type")
    if kind == "stream_start":
        holder["starting"] = True
        if holder["session"] is not None:
            try:
                await asyncio.to_thread(holder["session"].stop)  # drains MT: off the loop
            except Exception:
                pass
            holder["session"] = None
        session = StreamingSession(CFG, asr, mt, RUNTIME.get("diarizer", diarizer),
                                   enhancer, holder["q"], vad=vad)
        try:
            # configure() may download/load the ASR model: keep it off the
            # event loop so WS pings and other clients stay responsive.
            await asyncio.to_thread(
                session.configure,
                targets=data.get("targets") or list(CFG.get("default_targets", ["en", "zh"])),
                src_lang=data.get("src_lang") or None,
                denoise=_parse_denoise(data.get("denoise", "auto")),
                terms=_parse_terms(data.get("terms")),
                display_lang=data.get("display_lang") or None,
            )
        except Exception as e:
            holder["starting"] = False
            await ws.send_json({"ok": False, "error": f"stream start failed: {e}"})
            return
        holder["session"] = session
        holder["starting"] = False
        await ws.send_json({"ok": True, "type": "stream_started",
                            "asr": asr.status(), "mt": mt.status(),
                            "diarizer": _diarizer_status()})
        return
    session = holder.get("session")
    if session is None:
        if holder.get("starting"):
            # stream_start is still loading models (configure runs off the
            # event loop and can take seconds on first run). Tell the client
            # to hold audio instead of surfacing a false "no stream open".
            await ws.send_json({"ok": True, "type": "starting"})
            return
        await ws.send_json({"ok": False, "error": "no stream open (send stream_start first)"})
        return
    if kind == "stream_config":
        # Hot-update mid-stream settings without resetting the recognizer.
        try:
            session.live_update(
                targets=data.get("targets") or None,
                denoise=_parse_denoise(data.get("denoise", "auto"))
                if "denoise" in data else None,
                terms=_parse_terms(data.get("terms")) if "terms" in data else None,
                display_lang=data.get("display_lang", None) if "display_lang" in data else None,
            )
            await ws.send_json({"ok": True, "type": "stream_configured"})
        except Exception as e:
            await ws.send_json({"ok": False, "error": f"config failed: {e}"})
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
    elif kind == "stream_flush":
        await asyncio.to_thread(session.flush)
    elif kind == "stream_stop":
        try:
            n = await asyncio.to_thread(session.stop)
        except Exception as e:
            await ws.send_json({"ok": False, "error": f"stop failed: {e}"})
            return
        holder["session"] = None
        # queued behind the session's last events, so it arrives after them
        holder["q"].put(("stream_stopped", {"finalized": n}))


async def _stream_worker(ws: WebSocket, inbox: asyncio.Queue, holder: dict):
    """Run stream_* messages in arrival order, off the socket read loop."""
    while True:
        data = await inbox.get()
        try:
            await _ws_stream_msg(ws, data, holder)
        except Exception as e:
            log.warning("stream message %s failed: %s", data.get("type"), e)


async def _pump_events(ws: WebSocket, q):
    """Send session events as they happen. Draining only when a client
    message arrived stalled every result while the mic was paused, and
    held all stop-time translations until stop() returned."""
    import queue as _queue

    while True:
        try:
            kind, payload = q.get_nowait()
        except _queue.Empty:
            await asyncio.sleep(0.05)
            continue
        await ws.send_json({"ok": True, "type": kind, **payload})


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    """Two protocols, chosen per message:

    Legacy (browser demo): {audio_b64, sr, targets, src_lang, denoise} with
    ~segment_seconds of PCM16 per message -> fixed-window utterances.

    Streaming (C# desktop client): {"type": "stream_start", targets, src_lang,
    denoise} opens a StreamingSession, then {"type": "stream_audio", audio_b64,
    sr} feeds short frames, then {"type": "stream_stop"}. Server emits
    ("partial", ...) live events, ("tok", ...) per-target token deltas during
    final translation, and ("utterance", ...) finals from the session queue.
    """
    await ws.accept()
    import queue as _queue

    holder: dict = {"session": None, "starting": False, "q": _queue.Queue()}
    pump = asyncio.create_task(_pump_events(ws, holder["q"]))
    # The read loop must never wait on processing: a finalize blocks feed()
    # for seconds, uvicorn's 32-message inbox fills, the client's pong sits
    # unread and the socket dies with 1011 keepalive ping timeout (~1 min in).
    inbox: asyncio.Queue = asyncio.Queue()
    worker = asyncio.create_task(_stream_worker(ws, inbox, holder))
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
            if data.get("type") in ("stream_start", "stream_audio", "stream_stop",
                                      "stream_config", "stream_flush"):
                inbox.put_nowait(data)  # ponytail: unbounded; lag shows as late rows, not a dead socket
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
    finally:
        pump.cancel()
        worker.cancel()
        if holder["session"] is not None:
            holder["session"].close()
