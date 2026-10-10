# AGENTS.md — ConfLive (trilingual conference ASR + translator)

Operating manual for AI agents (Claude, Codex, Cursor…) in this repo. `CLAUDE.md` imports
this file. **Read it before editing anything, and keep it true after every change** (see
"Rules for every change" at the bottom — those rules are mandatory, not suggestions).

## What the app is

Records a meeting in **Vietnamese · English · Chinese**, shows it as a live chat feed:
speaker-labelled rows, live captions, token-streamed translations into the other languages.
One Python backend (FastAPI, `run.py` → `app/main.py`, `127.0.0.1:8000`) serves two
frontends that speak the same protocol:

- `electron/` — **primary** desktop app (Electron main process + React/Vite/Tailwind UI).
- `csharp/ConfLive/` — WPF alternative; also what the CI installer ships.

Windows-only, developed on an RTX 4060 Laptop (8 GB VRAM). Not a git repo.

## Workspace layout (`D:\CONFERENCE ASR\`)

| Path | What it is |
| --- | --- |
| `conference-asr-translator/` | **This repo** (the app). |
| `zipformer zh-en-vi onnx phaseB 2e/` | ASR model in use (`config.yaml › asr_model`). Fallback: `models/zipformer-2e`. |
| `zipformer zh-en-vi onnx phaseB s2a/` | Alternate ASR checkpoint (same file layout, other tag). |
| `HY-MT/` | Local Hy-MT INT8 ONNX export (used only if `mt_model` points at an `.onnx` dir). |
| `Nemotron-3-Diarization/` | Local diarizer model (`config.yaml › nemotron_model`, loaded offline). Fallback: `models/nemotron-3-diarization`. |
| `diardeps/` | transformers 5.19 + huggingface_hub 1.x + tokenizers 0.23 (`--no-deps --target`) for `app/nemotron_sidecar.py` only. |
| `dfenv/` | Python 3.11 venv for the DeepFilterNet sidecar (`app/df_sidecar.py`). |
| `pydeps/` | `triton-windows` for compiled MT decode (`mt_compile`); found by `app/__init__.py`. |
| `.triton-cache/` | Triton/Inductor compile cache. Never edit, safe to delete (recompiles ~100 s). |
| `redesign-conferenceasr-desktop-ui/` | Original React design mock; the live copy is `electron/ui/`. |
| `AGENTS.md` (workspace root) | **Unrelated** Apple-design-review skill manual. Ignore it for this app. |

## Pipeline

### Live mic (`/ws/live`, the main path) — `app/streaming.py › StreamingSession`

```
UI mic or system audio (16 kHz PCM16, 0.5 s frames, base64) ── stream_audio ──► main._ws_stream_msg
  └► StreamingSession.feed(frame)
       ├ diarizer.push()     Nemotron sidecar scores it per 10 ms, async (app/nemotron_sidecar.py)
       ├ diarizer.assign()   volume level + energy silence; label = latest Nemotron voice (app/diarizer.py)
       ├ vad.frame_speech()  Silero gate, silent = energy OR VAD (app/vad.py)
       ├ sherpa-onnx streaming recognizer (CPU int8)           (app/zipformer_engine.py)
       ├ _track_speaker: diarizer.turn(seg) ─► relabel the open row, or on a hand-over
       │   (new voice alone ≥ diar_split_min) _split: words before the change (ASR token
       │   times + split_token_slack) finalize as their own row, the rest opens a new one
       ├ text changed ─► "partial" event (never waits on MT)
       │   every mt_retranslate_words words / on pause ─► live re-translation job
       │   (SSBD draft: mt.translate_ssbd off the previous draft)
       └ endpoint: trailing silence ≥ endpoint_silence, or ≥ max_segment ─► _finalize
  _finalize:
       denoise segment (app/enhancer.py: denoise_mode gate (default) = spectral gate only;
         deepfilternet = DF in-proc → DF sidecar → spectral gate)
       → final text (stream result; re-decode if denoised; ORT re-decode if asr_cuda_final)
       → speaker: diarizer.turn() over the segment (Nemotron), else
         diarizer.attribute_segment()  neural speaker verdict (NeMo/pyannote)
       → langid.detect()  same-language targets = copies, no MT
       → VAD junk gate (vad_min_speech_frac) → skip MT
       → "utterance" event, pending=true   (ASR text out NOW)
       → MT FIFO worker thread (_mt_loop): display_lang first, one target at a time,
         re-emit "utterance" (same id) per target; pending=false on the last.
         Backlog shedding: ≥ mt_max_backlog → display lang only; ≥ 2× → no MT.
```

Every event goes into a `queue.Queue`; `main._pump_events` sends them independently of
incoming messages. The WS read loop never blocks: `stream_*` messages go to an inbox
processed by `_stream_worker`, and heavy calls run in `asyncio.to_thread` (otherwise the
socket dies with 1011 keepalive timeout).

### Other paths

- **Upload** `POST /api/transcribe` → `app/pipeline.py › ConferencePipeline.process_file`:
  whole-file denoise → fixed `segment_seconds` chunks → ASR (streaming recognizer; the ORT
  re-decoder only with `asr_cuda_final`) → diarize → MT (synchronous).
- **Legacy WS** (message without `type`): fixed-window `pipe.process_chunk`. Browser-demo era.
- **Translate tab** `POST /api/translate` → `main._translate_long` (chunked by
  `translate_chunk_chars`, CJK costs ×3).
- **OCR** `POST /api/ocr` → PP-OCR blocks (`task=ocr`; non-CJK lines re-read by VietOCR when
  `vi_ocr_model` exists) or PaddleOCR-VL (table/formula/chart, and `task=ocr` with `src=vi`
  only when VietOCR is missing), optional MT: blocks translated in one pass off the event loop,
  page translation = their join. Entering OCR mode unloads the diarizer to free VRAM (`_enter_ocr_mode`).
- **Exports** `GET /api/export.txt|.srt` read `pipe.history` (upload/legacy path only).
- **Listen buttons** `POST /api/tts {text, lang}` → `app/tts_engine.py › PiperTTS.synth` (sherpa-onnx
  Piper VITS int8, CPU, off the event loop) → `audio/wav`; `404` = no voice for `lang`, the UI then
  uses a matching OS voice (`api.ts › speak`). Voices load on first use, all dropped after
  `tts_idle_unload` s idle.
- **Live mode** `stream_start` → `main._enter_live_mode`: unloads PaddleOCR-VL (VRAM) and
  `BlockOCR` (PP-OCR + VietOCR RAM); the next OCR reloads them. Mirror of `_enter_ocr_mode`.

## File map (what each file does → what else to check when you change it)

| File | Role | When you change it, also check |
| --- | --- | --- |
| `run.py` | chdirs to its own folder (so `config.yaml` and relative model paths resolve from any cwd), starts uvicorn on `config.yaml` host/port | `electron/src/main.ts` (spawns it), `csharp/.../BackendManager.cs` |
| `config.yaml` | All tunables, with MEASURED comments | Code reading the key (grep it), README §3 |
| `app/__init__.py` | `../pydeps` triton path, cache dirs, Windows EcoQoS opt-out | `mt_engine._enable_compile` |
| `app/main.py` | FastAPI app: builds engines, REST routes (incl. `/api/tts`, health `tts`/`mem`), `/ws/live` protocol, `_enter_live_mode` / `_enter_ocr_mode` | `electron/ui/src/api.ts`, `csharp/ConfLive/ApiClient.cs` + `Models.cs`, `scripts/live_ws_e2e.mjs`, `scripts/ui_soak_server.py`, README §8 |
| `app/engines.py` | `build_app()` / `make_mt_engine` for scripts | **`main.py` wires its own engines** — keep both in sync |
| `app/streaming.py` | Live session: endpointing, partials, SSBD re-translation, MT worker, backlog | Event shapes in `api.ts`/`Models.cs`; `scripts/streaming_smoke.py`, `soak_long.py` |
| `app/pipeline.py` | Upload/legacy fixed-window pipeline + exports | `/api/transcribe`, `smoke_test.py` |
| `app/zipformer_engine.py` | sherpa-onnx streaming recognizer (`_rec`), `redecode_cuda` | `streaming.py` (uses `asr._rec` directly), `cuda_zipformer.py` |
| `app/cuda_zipformer.py` | Same ONNX via ORT CUDA (final re-decode). Import torch **before** onnxruntime | `requirements.txt` pin `onnxruntime-gpu~=1.22` |
| `app/mt_engine.py` | `HyMT2Engine` (HF transformers, FP8 kept as `_FP8Linear` under the compiled decode / fp16 bake without it, compile, SSBD, streaming, batched) + `MockMT` | `onnx_mt.py` (subclass), `streaming._translate`, `main._translate_long` |
| `app/onnx_mt.py` | `OnnxMTEngine(HyMT2Engine)` for local INT8 ONNX; overrides `_prepare`/`_generate` | Any `HyMT2Engine` signature change |
| `app/diarizer.py` | `VolumeDiarizer`, `NeMoDiarizer`, `PyannoteDiarizer`, `NemotronDiarizer` (streaming: `push`/`turn`, sidecar client, falls back to Titanet), `make_diarizer` | `/api/settings`, `_enter_ocr_mode`, UI diarizer switch, `streaming._track_speaker/_split`, `smoke_test.py` turn() check |
| `app/nemotron_sidecar.py` | Stand-alone Nemotron-3-Diarization worker (transformers 5.19 from `../diardeps`, offline, local model dir); binary stdin/stdout protocol in its docstring | `NemotronDiarizer._start/_send/_read_loop` |
| `app/enhancer.py` | `denoise_mode: gate` (default, spectral gate) or DeepFilterNet chain → sidecar → spectral gate | `df_sidecar.py` (binary stdin/stdout protocol), `scripts/test_denoise.py` |
| `app/df_sidecar.py` | Stand-alone DF worker under `../dfenv`; imports nothing from `app` | `enhancer._sidecar` |
| `app/vad.py` | Silero VAD, stateful per stream, fail-open | `streaming.feed`, `/api/warmup` |
| `app/langid.py` | lingua vi/en/zh detect, `None` = unsure | `streaming`, `pipeline` |
| `app/ocr_engine.py` | `PaddleOCRVLEngine`, `BlockOCR` (+ `VietRec` VietOCR ONNX line reader, `_crop_quad`, `unload`), `MockOCR` | `/api/ocr`, `/api/health` `ocr.vietocr`, UI OCR view, `ocr_test/vietocr_onnx.py` (reference twin) |
| `app/tts_engine.py` | `PiperTTS`: per-language sherpa-onnx Piper voices from `tts_dir/<lang>/`, lazy load, idle unload, WAV out; `__main__` self-check | `/api/tts`, health `tts`, `scripts/get_tts.py` (folder layout), `api.ts › speak` |
| `app/audio_io.py`, `app/device.py` | Decode/resample to 16 kHz; CUDA/CPU + dtype resolution | All engines |
| `electron/src/main.ts` | Find `run.py`, probe Python + CUDA, spawn backend, open window (hidden title bar + `titleBarOverlay`: caption buttons over the 52 px toolbar, recoloured on the renderer's `theme` IPC), grant `getDisplayMedia` loopback audio (System audio input), whole-UI zoom (Ctrl+=/-/0, Ctrl+wheel) | `ConfLive.bat`, package.json `build.files`, `index.css` `.topbar` (height 52, `env(titlebar-area-*)` padding, drag region) |
| `electron/src/preload.ts` | Exposes `window.conflive` (version, port, `setTheme`) | `App.tsx` theme effect |
| `electron/ui/src/api.ts` | **Client contract**: REST calls, `LiveSocket`, mic / system-audio capture (`SYSTEM_AUDIO` entry in `listMics`), `speak`/`stopSpeaking` (one player app-wide, OS-voice fallback), TS types | `app/main.py`, `app/streaming.py` event fields |
| `electron/ui/src/App.tsx` | Whole UI (Live / Library / Glossary / Translate / OCR / Settings); `TrActions` = Listen + Copy under each finished translation | `api.ts` types, `index.css` |
| `electron/ui/src/index.css` | macOS-style design: semantic tokens (light + `[data-theme="dark"]`), `.glass` only on navigation (sidebar, toolbar, floating controls), components, OCR workbench | `App.tsx` class names |
| `csharp/ConfLive/*` | WPF client: `BackendManager` spawns backend, `ApiClient` WS, `MicCapture`, `Models` DTOs | Same protocol as `api.ts` |
| `installer/`, `.github/workflows/build-installer.yml` | Inno Setup + CI (builds UI + C# exe on `v*` tags) | `csharp/` paths |
| `ConfLive.bat` | Tester one-click: venv, torch, requirements, models, Listen voices (`get_tts.py`), build, launch | `requirements.txt`, `scripts/download_models.py`, `scripts/get_tts.py`, `electron/package.json` |
| `scripts/` | Checks and tools (see below) | — |
| `ocr_test/` | OCR recogniser comparison + `export_vietocr_onnx.py` (used by `scripts/get_vietocr.py`) | `scripts/get_vietocr.py` |
| `scripts/get_vietocr.py` | Optional Vietnamese OCR: vietocr into a temp `--target` dir, export fp16 ONNX → `models/vietocr-s2s` | `config.yaml › vi_ocr_model`, `BlockOCR` |
| `scripts/get_tts.py` | Listen voices: downloads the vi/en/zh Piper int8 archives (sherpa-onnx `tts-models` release) → `models/tts/<lang>/` | `config.yaml › tts_dir`, `PiperTTS`, `ConfLive.bat`, README "Listen voices" |
| `checkpoints/` | Source snapshots (no git): `snap.sh <name>` (app, scripts, config, docs, `electron/src`, `electron/ui/src`), history + restore steps in `WORKLOG.md` | — |

### WS contract (must match in `main.py`, `streaming.py`, `api.ts`, `ApiClient.cs`)

Client → server: `ping`, `stream_start {targets, src_lang, denoise, terms, display_lang, resume}`,
`stream_config {targets?, denoise?, terms?, display_lang?}`, `stream_audio {audio_b64, sr}`,
`stream_flush`, `stream_stop`.
Server → client (all `{"ok": true, "type": ...}`): `pong`, `starting`, `stream_started`,
`stream_configured`, `partial {id, speaker, rms_db, text, translations|null, final:false}`,
`tok {id, tgt, seq, delta}`, `utterance {id, speaker, rms_db, denoised, src_lang, text,
translations, start, asr_backend, pending, timings, diar_backend?}`, `stream_stopped {finalized}`.
Errors: `{"ok": false, "error": ...}`. Rows are **upserted by `id`** (process-wide counter from
1 000 000) — never reuse ids.

## Run and verify

```powershell
.venv\Scripts\python.exe run.py                    # backend only, docs at /docs
cd electron; npm run build; npx electron .         # app (spawns backend if not running)
python -m scripts.smoke_test                       # no weights needed: device, denoise, diarizer (+ Nemotron turn()), pipeline, routes
python scripts/test_translate_chunks.py            # no weights: Translate-tab chunker
python scripts/get_tts.py; python -m app.tts_engine # Listen voices: install, then each synthesizes
python scripts/streaming_smoke.py                  # REAL engines: partials, endpointed finals, MT
python scripts/test_denoise.py                     # denoise chain + spectral gate
node scripts/live_ws_e2e.mjs [file.wav]            # backend up: drives /ws/live like the UI
python scripts/soak_long.py [min] [speed]          # long-meeting boundedness (memory, backlog, lag)
python scripts/ui_soak_server.py                   # fake backend replaying 1 h into the real UI
```

UI dev server: `.claude/launch.json` config `ui` (Vite on :5179, needs the backend on :8000).
No weights/GPU → backend runs in labelled demo mode (mock ASR/MT); `DEMO_MODE=true|false` forces it.
Env overrides: `DEVICE`, `MT_DTYPE`, `ASR_MODEL`, `MT_MODEL`, `DEMO_MODE`, `CONF_LIVE_BACKEND`.

## Known traps (measured — don't "fix" them back)

- Import **torch before onnxruntime** or ORT CUDA silently falls back to CPU.
- `onnxruntime-gpu~=1.22` (1.30 needs CUDA 13; torch vendors CUDA 12).
- sherpa-onnx wheels are CPU-only: live partials are always CPU.
- HY-MT INT8 ONNX crashes on CUDA EP mid-inference → `mt_onnx_provider: cpu`.
- `mt_batch`, `mt_speculative`, `mt_kvquant`, `asr_cuda_final` are OFF because they measured
  slower; the reason is in the `config.yaml` comment next to each.
- ASR emits UPPERCASE English → `streaming._translate` lowercases; glossary terms are injected
  as original + UPPER + lower.
- Dialogue history must come **before** the instruction in the MT prompt, and is dropped for
  sources under 5 words (otherwise Hy-MT translates the history).
- `deepfilternet` has no Python 3.12 wheel → backend on 3.12 uses the `../dfenv` sidecar.
- `rapidocr_onnxruntime` is installed `--no-deps` (`ConfLive.bat`): its `onnxruntime` dep
  overwrites `onnxruntime-gpu`. Its other deps are in `requirements.txt`.
- Nemotron-3-Diarization needs **transformers ≥ 5.19**; NeMo 3.0.0 (latest PyPI) can't build
  it (`self_attention_model='rope' is not supported`), and upgrading the backend's
  transformers 4.57 risks Hy-MT. So it lives in a sidecar with `../diardeps` first on
  `PYTHONPATH`. Never `pip install` transformers 5 into the backend env for it.
- The diarizer is **offline by design**: local model dir only, sidecar env sets
  `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`, `local_files_only=True`. Don't pass a hub id.
- Nemotron `[1,0]`/`[2,x]` chunks look lower-latency but cost 64–86 ms per step (fixed
  speaker-cache context): 53–77% GPU, falls behind under MT. `[3,1]` (0.32 s) is the floor.
- sherpa token timestamps are *emission* times: up to +0.04 s late for the old speaker's
  last word, ≥ +0.22 s for the new speaker's first. `_split` adds `split_token_slack`
  (0.12 s); without it the last word before a hand-over vanished from both rows.
- Sidecar startup is ~20–30 s of transformers 5.x imports (not network): `/api/warmup`
  starts the diarizer first, in a background thread.
- PP-OCR rec dictionaries lack most Vietnamese letters (`CẤM ĐỖ XE` → `CAMDOXE`); PP-OCRv6
  (2026) still lacks 88/146. So non-CJK lines go to VietOCR (`vi_ocr_model`); without it
  `/api/ocr` sends `src=vi` to PaddleOCR-VL, which garbles Vietnamese signs (`Giờ` → `Giంద 00 00`).
  VietOCR input must be PIL LANCZOS-resized (bilinear costs 6–7% CER); its vocab has no `–`.
  SenOCR-Vi (PaddleOCR-VL-1.6 vi fine-tune) needs transformers ≥ 5 — not in the backend env.
  The VLM decodes with `no_repeat_ngram_size=8`
  (without it a Vietnamese sign looped `00 00 …` to the 1024-token cap, 22 s → 1 s).

- `_FP8Linear` is only fast under the compiled decode with
  `torch._inductor.config.coordinate_descent_tuning = True` (set in `_enable_compile`): that
  turns the batch-1 matmul into a reduction that fuses the FP8→fp16 cast. Without the compile
  every token would materialize fp16 weights, so `_dequantize_fp8` bakes fp16 instead (and
  re-bakes if the compile fails). It scales the *weight*, not the output: x @ raw-FP8 can
  overflow fp16 before the scale shrinks it.
- The ORT final re-decoder (`asr._ort()`) holds a second Zipformer copy: +1043 MB RAM. Never
  warm it unless `asr_cuda_final` is on; uploads get identical text from the streaming recognizer.
- Piper voices: lessac (en) and xiao_ya (zh) licences are research / non-commercial — keep
  ljspeech / chaowen. Windows' own voices are English-only, so OS speech can't replace them.

## Rules for every change (mandatory)

1. **Before editing a file**, read its row in the file map and open every file in its
   "also check" column. Grep for the symbol/key/event you are changing across `app/`,
   `scripts/`, `electron/src`, `electron/ui/src`, `csharp/` before touching it.
2. **Contract changes go everywhere at once.** A changed WS message, event field, REST
   route or form field is not done until `main.py`/`streaming.py`, `electron/ui/src/api.ts`,
   `csharp/ConfLive/ApiClient.cs` + `Models.cs`, and the WS section above all agree.
   Same for engine method signatures (`HyMT2Engine` ↔ `OnnxMTEngine`, diarizer classes share
   `assign / attribute_segment / reset / status / unload / ensure_loaded`).
3. **Config keys:** adding/renaming/removing one means updating `config.yaml` (with a
   comment, MEASURED numbers if it's a perf knob), every `cfg.get("key")`, and README §3.
4. **Engine wiring** lives twice: `app/main.py` (server) and `app/engines.py › build_app`
   (scripts). Change both.
5. **Update the docs in the same change, every time:**
   - this `AGENTS.md` — file map row, pipeline diagram, WS contract, traps, workspace layout;
   - `README.md` — user-facing behaviour, config, API, layout.
   A change that makes any sentence here false is unfinished until the sentence is fixed.
   New file → add a row. Deleted file → remove its row and every mention.
6. **Verify before saying done:** `python -m scripts.smoke_test` for any backend change;
   `npm run build` in `electron/` for any TS/UI change; `python scripts/streaming_smoke.py`
   (or `live_ws_e2e.mjs`) when `streaming.py`, `main.py` WS code, or an engine changed and
   weights are available. Report what ran and what was skipped.
7. Don't edit `.venv/`, `node_modules/`, `dist/`, `models/`, `../.triton-cache`, `../pydeps`,
   `../dfenv` or the model folders. Don't hard-code new absolute `D:/...` paths in code;
   put them in `config.yaml`.
8. Perf claims need a measurement. Keep the "MEASURED …" comment style in `config.yaml`.
