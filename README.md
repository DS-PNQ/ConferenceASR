# ConfLive — Trilingual Conference ASR + Translator

Record a conference in **Vietnamese · English · Chinese** and watch it appear on screen
like a chat app: speaker-labelled bubbles, live transcription, side-by-side translations.

Two frontends, one Python inference backend (`run.py`: Zipformer ASR, Hy-MT2 FP8 MT,
streaming Nemotron diarization, DeepFilterNet hook):

- **Electron + TypeScript app** (`electron/`, primary): `npm run build && npx electron .`.
  Finds `run.py`, picks CUDA/CPU itself, streams mic over `/ws/live` with live
  partials and token-streamed finals. `npm run dist` builds a Windows installer.
- **C# desktop app** (`csharp/ConfLive`): native WPF window, same protocol.

## GPU usage notes (measured RTX 4060 Laptop)

- The local `HY-MT` INT8 ONNX prose is driven directly (`app/onnx_mt.py`):
  prefill + greedy loop with past cache, repetition penalty like the HF path,
  TRUE per-token streaming (no collector thread needed). Verified quality
  matches the FP8 reference (same 大家早上好… output).
- **CUDA does NOT run it**: the int8 graph builds CUDA sessions that pass tiny
  probes, then crash mid-inference (garbage Expand dims); CPU EP verified
  clean (all prefill lengths + full decode loops). Engine probes providers and
  picks CPU automatically (`mt_onnx_provider: cuda` retries after toolchain
  upgrades; needs ORT~1.22 + CUDA 12 — 1.30 wants CUDA 13).
- A/B, same audio/targets: **local ONNX CPU 135 s vs HF FP8 CUDA 54 s**
  (~1 s/token vs ~0.3 s/token). Local = no HF dependency; FP8 = 2.5x faster.
  Switch back anytime: `MT_MODEL=tencent/Hy-MT2-1.8B-FP8` (weights still cached).

- Weights load to `cuda:0` as FP8, then `_dequantize_fp8` bakes them into plain fp16
  Linears once (compressed-tensors otherwise re-dequantizes every forward: 3000 vs
  1900 kernel launches/token, 72 vs 43 ms GPU/token; +1.4 GB VRAM). Generate
  runs there too — verified: params on `cuda:0`, SM utilization 21–43% during
  decode, ~4 tok/s batch-1 greedy. If `nvidia-smi` shows ~0% between utterances,
  that is normal: single-stream decode of a 1.8B model is latency-bound with
  tiny kernels, and the GPU idles between segments.
- ASR runs split-brain by design: live partials from sherpa-onnx CPU int8
  (~15x realtime — the wheels are CPU-only builds), finalized segments
  re-decoded on GPU by `app/cuda_zipformer.py`, which drives the same ONNX
  files through ORT CUDA directly (byte-identical protocol, verified
  character-for-character vs sherpa CPU; ~9x realtime steady, fp32).
  Entries carry `asr_backend: ort-cuda|stream`. Two load-bearing details:
  `onnxruntime-gpu~=1.22` (1.30 wants CUDA 13; torch vendors CUDA 12) and
  **import torch before onnxruntime** so its bundled CUDA DLLs preload —
  without that ordering even a good install silently falls back to CPU.
- Compiled MT decode (`mt_compile`): install triton-windows next to the repo, not on C:
  `py -m pip install --target "..\pydeps" --no-deps triton-windows==3.8.0.post29`.
  `app/__init__.py` finds `../pydeps`, points `CC` at its bundled TinyCC (an MSYS
  gcc on PATH breaks it) and keeps compile caches in `../.triton-cache`. MEASURED:
  9.6 -> 31.2 tok/s, first token ~65 ms; warmup compiles ~50 s cold, ~30 s cached.
  Without triton the engine logs it and decodes eager.
- `mt_use_cache: true` is set, though this custom modeling largely ignores it;
  per-token latency is a property of the checkpoint, not a device bug.

| Module | Model (default) | Source |
|---|---|---|
| Denoise | `DeepFilterNet3` (auto-download, Python 3.11 runtime) | https://github.com/Rikorose/DeepFilterNet |
| ASR | Zipformer zh-en-vi ONNX, stage 2e (CPU int8 partials) | https://huggingface.co/lmcu000/sherpa-onnx-streaming-zipformer-zh-en-vi-2e |
| Translation | `tencent/Hy-MT2-1.8B-FP8` (compressed-tensors, CUDA) | https://huggingface.co/collections/tencent/hy-mt2 |
| Diarization | `nvidia/Nemotron-3-Diarization` streaming, 0.32 s, local + offline (→ NeMo Titanet → volume fallback chain) | https://huggingface.co/nvidia/Nemotron-3-Diarization |

UI is a native desktop window (neutral tones + AI-purple `#7C3AED`):
sidebar controls + chat-style transcript feed with live partials, token-streamed
translations, speaker colours, volume meter, and TXT/SRT export. No browser needed.

## 0. Quick start for testers

Install **Python 3.11** (needed for DeepFilterNet denoise) and **Node.js LTS**, then
double-click **`ConfLive.bat`**. The first run creates `.venv`, installs torch (CUDA if
`nvidia-smi` exists, else CPU) + requirements + deepfilternet, downloads the models
(ASR → `models/zipformer-2e`, Hy-MT2, DeepFilterNet3, Nemotron diarizer →
`models/nemotron-3-diarization`, Titanet), installs the diarizer's `..\diardeps`, builds the UI and opens
the app. Later runs just launch. A failed step is retried on the next run.

## 1. Setup (Python 3.12, RTX 4060 / CPU)

```powershell
cd "D:\DENSEV2 - reading\conference-asr-translator"
C:\Users\Asus\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126   # or .../cpu for CPU-only
pip install -r requirements.txt
pip install --no-deps rapidocr_onnxruntime   # OCR blocks; its `onnxruntime` dep would clobber onnxruntime-gpu
python scripts/get_vietocr.py                # optional: Vietnamese OCR reader (VietOCR fp16 ONNX, 45 MB) -> models/vietocr-s2s
```

> Python version note: `deepfilternet` bundles prebuilt wheels for Python
> 3.8–3.11. **DeepFilterNet is fully working under Python 3.11** (verified:
> 41.6 dB noise reduction, full denoise→ASR→MT chain, enhancer even lands on
> `cuda:0`). On 3.12 its build needs a Rust toolchain — without it the app runs
> in pass-through mode and everything else still works.
>
> To run with denoising, use the 3.11 venv (torch cu126 + all requirements
> install cleanly there) and launch the app from it. The model weights
> ([DeepFilterNet/models](https://github.com/Rikorose/DeepFilterNet/tree/main/models))
> are fetched automatically by `init_df()` on first warmup (see
> `scripts/download_models.py`) — nothing to download by hand.

Speaker diarization (default `diarizer: nemotron`, see §5): transformers 5.19 in its own
folder for the sidecar, then the model once into a local folder:
```powershell
py -3 -m pip install --no-deps --target ..\diardeps "transformers==5.19.0" "huggingface_hub>=1.31,<2" "tokenizers>=0.23.1,<0.24"
python scripts/download_models.py   # -> models/nemotron-3-diarization (or set nemotron_model to a local dir)
```

Optional (extra):
```powershell
pip install "nemo_toolkit[asr]"                                   # Titanet: diarizer: nemo, and Nemotron's fallback
```

## 2. Run (desktop app)

### Electron + TypeScript app (current primary GUI)

```powershell
cd electron
npm install        # once: electron + typescript + electron-builder
npm run build      # builds React UI (ui/) + tsc -> dist/
npx electron .     # launches; finds run.py, picks CUDA/CPU, opens the window
npm run dist       # Windows NSIS installer (needs electron-builder downloads)
```

The window is the `redesign-conferenceasr-desktop-ui` React design, wired live:
dark sidebar (Live session / Library / Glossary + Settings), recording stage
with real mic level, transcript rows with timestamps + speaker avatars, live
caption partials, translation toggle + search, session stats panel, localStorage
session archive, and a working glossary (terms steer every translation).
Every row shows **all** requested target translations (display language first,
with lang chips); live partials stream theirs the same way. The display
language streams first at finalize too (`display_lang` in `stream_start`,
hot-updatable mid-session via `stream_config`), and live re-translation
covers the display language only — one MT call keeps the feed loop realtime.
Layout: fixed-height shell, sidebar scrolls independently of the transcript.
Zoom the whole UI with Ctrl+= / Ctrl+- / Ctrl+0 or Ctrl+wheel. The OCR view has its own
page viewer: Ctrl+wheel / touchpad pinch zooms at the cursor, `+` `-` `0` (fit width) when
the viewer is focused, a zoom slider + Fit button, drag to pan, double-click toggles 200%.
The picked image shows (with live rotate/crop) before Run OCR; rotate/crop clear old results.

### C# app (WPF alternative, same protocol)

```powershell
# needs: .NET 8 SDK + Python 3.11/3.12 with requirements installed
dotnet build csharp/ConfLive/ConfLive.csproj -c Release
.\csharp\ConfLive\bin\Release\net8.0-windows\ConfLive.exe
```

The exe finds `run.py` next to it, picks CUDA (`nvidia-smi` present) or CPU
(`DEVICE=cpu`), launches the backend itself, and shows the effective device in
the badge — `(CPU fallback active)` if CUDA was requested but unavailable.

```powershell
# one-file exe + Setup.exe installer:
powershell -ExecutionPolicy Bypass -File installer/build.ps1
# needs Inno Setup (iscc) for the final Setup.exe; without it, ship installer/stage.
```

### Python backend only (no GUI)

```powershell
python scripts/download_models.py   # optional pre-download (first run auto-downloads)
python run.py                        # API on http://127.0.0.1:8000, docs at /docs
```

In the window (React redesign: Live session / Library / Glossary / Translate + Settings):
- Models load in the background (status line reports ASR / MT / denoise backends).
- **● Record**: the streaming Zipformer recognizer stays open while you speak —
  a live partial bubble grows in realtime, and the ASR text lands the instant
  a segment endpoints — **MT never blocks it**. Same-language rows fill
  instantly (LID copies); other translations **stream in token by token**
  (blinking caret + `translating…`) into the same bubble, confirmed by a
  second event with full timings. Segments finalize on **pauses, not the
  clock**, so sentences are never cut mid-word.
- **⇪ Upload**: any recording streams through the same endpointing pipeline.
- **Translate view**: Google-style dual cards with source/target language tabs,
  swap button, Enter-to-translate and copy — same Hy-MT2 engine, glossary
  applied. Glossary terms are injected original + UPPER + lower with a MUST
  instruction, because our ASR emits UPPERCASE English while users type
  lowercase (verified: `standup → 站会` lands on `STANDUP` input).
- Settings: mic picker (hot-swappable mid-session; its last entry, **System audio**,
  transcribes what the computer plays — online meetings, videos — via Electron's
  Windows loopback capture), source language, **target
  chooser (vi/en/zh)** for what gets translated, DeepFilterNet switch, diarizer
  switch (nemotron/pyannote/nemo/volume), dev latency readout (on by default), model
  status + reload.
- Mic hardening: autoplay-policy resume, track-ended auto-stop, resume-pop
  guard, stall watchdog (one auto-recapture), WS auto-reconnect with fresh
  server session, and a "Catching up…" indicator when the server lags.

No weights / no GPU? The backend boots in clearly-labelled **demo mode**
(mock ASR/MT) so the UI, diarizer and exports all still work.

## 3. Config (`config.yaml`)

```yaml
device: auto            # auto | cuda | cpu   (env DEVICE overrides, affects MT)
asr_model: D:/CONFERENCE ASR/zipformer zh-en-vi onnx phaseB 2e  # local ONNX dir (any phaseB tag)
asr_provider: cpu       # sherpa-onnx wheels are CPU-only
asr_quant: auto         # auto = int8 on cpu
asr_threads: 4
mt_model: tencent/Hy-MT2-1.8B-FP8
mt_max_new_tokens: 256
mt_do_sample: false     # greedy decode = fastest
mt_use_cache: true      # passed through; this custom modeling largely ignores it
mt_speculative: false   # prompt-lookup speculative decoding (drafter-free, exact).
                        # MEASURED RTX 4060, interleaved n=3-4: 0.76-1.0x cross-lingual,
                        # ~1.0x same-language. No reliable win on this task: OFF.
                        # Auto-fires for src==tgt or ASCII->English regardless.
mt_lookup_tokens: 10
mt_history_turns: 3     # past segments as translation context (terminology/style)
mt_history_chars: 600
mt_do_sample: false     # greedy decode = fastest
noise_suppress: true    # pre-ASR denoising (UI toggle overrides per request)
denoise_mode: gate      # gate = lightest (spectral gate, 24 ms / -13.4 dB per 10 s) | deepfilternet
df_model: null          # null = DeepFilterNet3; alt: DeepFilterNet2, DeepFilterNet
df_post_filter: false
df_atten_lim_db: null   # e.g. 12 caps suppression, keeps ambience
diarizer: nemotron               # nemotron (streaming) | nemo (Titanet) | pyannote | volume (levels)
use_nemo: false
nemotron_model: D:/CONFERENCE ASR/Nemotron-3-Diarization  # LOCAL dir; else models/nemotron-3-diarization
nemotron_chunk: [3, 1]           # [chunk, right context] x 80 ms; [3,1] = 0.32 s, lowest that keeps up
nemotron_dtype: bfloat16
nemotron_threshold: 0.5
nemotron_python: null            # null = backend's Python
nemotron_deps: null              # null = ../diardeps
diar_split_min: 0.4              # new voice alone this long = split the live row
split_token_slack: 0.12          # ASR token-time lag allowance at a split
nemo_embedding_model: nvidia/speakerverification_en_titanet_large
nemo_cos_thresh: 0.55            # below this cosine -> new speaker (to max_speakers)
frame_seconds: 0.5       # mic frame size for the streaming recognizer
endpoint_silence: 1.0    # pause (s) that finalizes a segment
endpoint_min_speech: 0.5
max_segment: 20.0        # force-finalize run-on speech
mt_retranslate_words: 5  # live re-translation pace (new words); finals verify the last SSBD draft
segment_seconds: 5.0     # fixed-window size for the /api/transcribe upload path
vi_ocr_model: models/vietocr-s2s  # optional Vietnamese OCR reader (scripts/get_vietocr.py); missing = PP-OCR/VLM only
max_speakers: 3
```

Env overrides: `DEVICE=cuda|cpu`, `DTYPE`, `ASR_MODEL`, `MT_MODEL`, `DEMO_MODE=true|false`.

## 4. Voice suppression — DeepFilterNet pre-ASR stage

Backend chain (`/api/health → denoise.mode`): **inproc** (deepfilternet importable,
i.e. Python 3.11) → **sidecar** (`app/df_sidecar.py` under `../dfenv`, a Python 3.11
venv with CPU torch 2.5.1 + deepfilternet; used when the backend runs on 3.12 for
triton) → **gate** (numpy/scipy spectral gate, the lightest option). `denoise_mode: gate`
(the default) skips straight to the gate; `denoise_mode: deepfilternet` runs the chain. Measured on 10 s
of noisy audio: DF sidecar 445 ms / −26.5 dB, spectral gate 24 ms / −13.4 dB
(`py -3 scripts/test_denoise.py`). Recreate the sidecar env:

```powershell
uv venv ..\dfenv --python 3.11
uv pip install --python ..\dfenv\Scripts\python.exe torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python ..\dfenv\Scripts\python.exe deepfilternet "numpy<2"
```

Every segment is denoised *before* diarization/ASR (`app/enhancer.py`):
16 kHz audio → 48 kHz → `DeepFilterNet3` enhance → back to 16 kHz.
Uploads are denoised whole-file in one pass (no chunk boundaries); live-mic
chunks per segment. DeepFilterNet puts its own model on GPU when available,
CPU otherwise; if the package/weights are missing it becomes a transparent
pass-through (check `/api/health → denoise.backend`). Toggle per session with
the **🔇 Denoise** checkbox, `denoise=true|false` on `/api/transcribe`, or
`{"denoise": false}` on `/ws/live`.

## 5. Speaker diarization — Nemotron streaming (default)

`NemotronDiarizer` (`app/diarizer.py`, `diarizer: nemotron`) runs
`nvidia/Nemotron-3-Diarization` (99M params, arrival-order speaker ids, up to 8,
overlap-aware) **fully offline** from a local folder (`nemotron_model`, else
`models/nemotron-3-diarization`). Every live frame is pushed to it and scored per
10 ms, so a speaker change no longer waits for a pause:

- the open row's speaker label follows the voice live;
- when a new voice speaks alone for `diar_split_min` (0.4 s) after the previous one,
  the words before the change are finalized as their own row and the rest continues
  as a new row (`StreamingSession._split`, cut placed by ASR token times + `split_token_slack`);
- talking over each other or a short "yeah" never splits.

It runs in a sidecar process (`app/nemotron_sidecar.py`): the model needs
transformers 5.19 while the backend keeps 4.57 for Hy-MT, and NeMo 3.0.0 can't build
it (RoPE encoder). The sidecar uses the backend's Python with `..\diardeps` first on
`PYTHONPATH` and `HF_HUB_OFFLINE=1` — it never touches the network.

MEASURED (RTX 4060, bf16, `[3,1]` = 0.32 s buffer): 19–47 ms per 0.24 s chunk
(8–25% of the GPU), 205 MB VRAM. Smaller chunks don't pay: `[1,0]` (0.08 s) needs 64 ms
per 80 ms chunk (77% GPU) and falls behind under MT load. Two-voice meeting with
0–0.2 s hand-overs: 8/8 turns split and labelled correctly, every word on the right
side, boundaries within ±60 ms. bf16 is no faster than fp32 (launch-bound) but halves
VRAM. Startup ~20–30 s (transformers 5.x imports), overlapped with ASR/MT at warmup.
OCR mode stops the sidecar (all its VRAM back); the next stream restarts it. If the
sidecar can't start (no `..\diardeps`, no local model) the chain falls to Titanet →
volume, reported in `/api/health`.

### NeMo Titanet direct-forward (`diarizer: nemo`, Nemotron's fallback)

`NeMoDiarizer` (`app/diarizer.py`, `diarizer: nemo`): each finalized segment is
embedded with `nvidia/speakerverification_en_titanet_large` on CUDA and
cosine-matched against running speaker centroids — new speaker below
`nemo_cos_thresh` (0.55), capped at `max_speakers`. Stable global IDs, no
cross-chunk permutation problem. Verified: same voice → one ID (cos 0.77
across segments), noise → its own ID, every entry stamped
`diar_backend: nemo-titanet`. Live partials use the instant volume guess; the
neural verdict lands at finalize and overrides it.

Why this is the lightest NeMo setup that works: no smaller checkpoint exists
(nvidia ships only titanet_large); fp16 autocast measured *slower* (1.16 s vs
0.27 s first call); ONNX export is blocked (tracer chokes on the STFT); so the
win is calling `forward()` directly on tensors — no temp wav files per segment
(steady 41 ms vs 50 ms with file I/O). Chain on failure: pyannote (gated) →
volume, always reported, never silent. Switch at runtime without restart:
`POST /api/settings {"diarizer": "nemotron"|"pyannote"|"nemo"|"volume"}` (Settings view
does this; the replaced diarizer is unloaded — a Nemotron sidecar is stopped for good —
so an in-flight session finishes on the Titanet/volume fallback).

## 6. MT upgrades: token streaming, dialogue context, speculative decoding

- **Token streaming that actually streams**: finals generate with per-target
  `tok` events into the UI live. (Was silently dead for weeks: the id-collector
  indexed `value[0]`, which drops everything on 1-D streamer puts, and the bare
  except hid it — every call took the one-shot fallback. Fixed + verified:
  13 deltas in 2.2 s.)
- **Batched multi-target generate**: implemented (`translate_targets_stream`,
  demuxing streamer, `mt_batch` flag) — then MEASURED 0.09x on this modeling
  (no early stop, runs all 256 steps), so it stays OFF with sequential
  display-first as default. Kept for future modeling fixes.
- **Quantized KV cache** (`mt_kvquant`, optimum-quanto): works, 9.2 → 8.9 tok/s
  (neutral — per-step fixed costs dominate, not KV bandwidth). Kept ON: ~4x
  smaller cache means much longer conversations fit in 8 GB VRAM.
- **Longer conversations**: history 3 turns/600 chars → 5 turns/1000 chars
  (prefill is ~104 ms/100 tokens, so this is nearly free).
- **Dialogue context** (`mt_history_turns/chars`): recent finalized segments travel
  in the prompt as terminology/style reference, fenced with DO-NOT-translate.
  (Appending history after the instruction made the model translate the history
  and ramble 3x — prompt order matters.)
- **Prompt-lookup speculative decoding** (`mt_speculative`, drafter-free, exact):
  measured on RTX 4060, interleaved n=3–4 — **0.76–1.0x cross-lingual, ~1.0x
  same-language: no reliable win**, so OFF by default. It auto-fires for
  `src==tgt` or ASCII→English where drafts can only help. A dedicated drafter
  model would be needed for real speculative gains; none exists in Hy-MT2.
- **LID skip** (`app/langid.py`, lingua EN/VI/ZH, 8/8 test, ms-level): targets
  matching the detected source are copies, not generates — Vietnamese speech
  with vi/en/zh targets went 3 MT calls → 2. Entry `src_lang` shows the real
  code instead of `auto`.
- **Background partial translation**: debounced re-translation runs on a
  single-flight thread emitting follow-up partial events; the feed loop never
  blocks (worst `feed()` 0.6 s on 0.5 s frames), so endpointing stays realtime.

## 7. Verify (engines need cached weights; fakes don't)

```powershell
python -m scripts.smoke_test      # device, denoiser, diarizer, pipeline, API routes
python scripts/streaming_smoke.py # REAL engines: live partials + endpointed finals + MT
```

## 8. Backend API (serves Electron + C#)

- `POST /api/transcribe` (multipart `file`, `targets=en,zh`, `src_lang=auto`, `denoise=auto|true|false`, `terms={json}`)
- `POST /api/translate` (`{text, targets[], src, terms?}`) — text translation
- `WS /ws/live` — `stream_start {targets, src_lang, denoise, terms?}` /
  `stream_audio` / `stream_stop` → `partial`, per-target `tok` token deltas
  during final translation, `utterance` finals
- `POST /api/ocr` (multipart `files`, `task=ocr|table|formula|chart`, `translate_to`, `src`,
  `terms`, `rotate`, `crop`) — `task=ocr` = PP-OCR blocks (boxes + scores). PP-OCR's
  dictionaries lack most Vietnamese letters (`CẤM ĐỖ XE` → `CAMDOXE`), so with the optional
  VietOCR reader installed (`vi_ocr_model`) every non-CJK line is re-read by it (`rec` per
  block, page `backend: "pp-ocr + vietocr"`); without it `src=vi` falls back to the
  PaddleOCR-VL reader (diacritics, no boxes). `block_ocr.vietocr` / health `ocr.vietocr` say
  whether it is installed. Blocks are translated one MT pass, off the event loop.
- `POST /api/settings` (`{"diarizer": "nemotron"|"pyannote"|"nemo"|"volume"}`) — runtime switch
- `GET /api/health` — device, ASR/MT/denoise/diarizer status
- Glossary `terms` (`{source: target}`) ride Hy-MT2's documented terminology
  block; verified steering output (e.g. forced `光子引擎`).

## Layout

```
config.yaml  run.py  requirements.txt
app/engines.py  app/device.py  app/zipformer_engine.py  app/mt_engine.py
app/diarizer.py  app/nemotron_sidecar.py  app/enhancer.py  app/audio_io.py  app/pipeline.py
app/streaming.py  app/main.py
electron/package.json  electron/tsconfig.json
electron/src/main.ts  electron/src/preload.ts
electron/ui/ (React+Vite+Tailwind redesign, wired live: src/App.tsx, src/api.ts)
csharp/ConfLive/*.csproj,*.xaml,*.cs  installer/ConfLive.iss + build.ps1
scripts/download_models.py  scripts/get_vietocr.py  scripts/smoke_test.py  scripts/streaming_smoke.py
app/ocr_engine.py (PP-OCR blocks + VietRec + PaddleOCR-VL)  models/vietocr-s2s/ (optional)
checkpoints/ (source snapshots, see checkpoints/WORKLOG.md)
```

> Agents: this repo's `AGENTS.md` (imported by `CLAUDE.md`) maps every file, the
> pipeline, the WS contract and the update rules. The workspace-root `AGENTS.md`
> is an unrelated Apple design-review skill manual.
