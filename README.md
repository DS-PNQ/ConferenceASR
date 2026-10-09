# ConfLive — Trilingual Conference ASR + Translator

Record a conference in **Vietnamese · English · Chinese** and watch it appear on screen
like a chat app: speaker-labelled bubbles, live transcription, side-by-side translations.

Two frontends, one Python inference backend (`run.py`: Zipformer ASR, Hy-MT2 FP8 MT,
volume diarization, DeepFilterNet hook):

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
| ASR | Zipformer zh-en-vi ONNX, `phaseB_s2a` (CPU int8 partials + ORT CPU finals) | `D:\DENSEV2 - reading\zipformer zh-en-vi onnx phaseB s2a` |
| Translation | `tencent/Hy-MT2-1.8B-FP8` (compressed-tensors, CUDA) | https://huggingface.co/collections/tencent/hy-mt2 |
| Diarization | NeMo Titanet direct-forward on CUDA (pyannote → volume fallback chain) | https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html |

UI is a native desktop window (neutral tones + AI-purple `#7C3AED`):
sidebar controls + chat-style transcript feed with live partials, token-streamed
translations, speaker colours, volume meter, and TXT/SRT export. No browser needed.

## 1. Setup (Python 3.12, RTX 4060 / CPU)

```powershell
cd "D:\DENSEV2 - reading\conference-asr-translator"
C:\Users\Asus\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126   # or .../cpu for CPU-only
pip install -r requirements.txt
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

Optional (extra):
```powershell
pip install "nemo_toolkit[asr]"                                   # only if diarizer: nemo
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
- Settings: mic picker (hot-swappable mid-session), source language, **target
  chooser (vi/en/zh)** for what gets translated, DeepFilterNet switch, diarizer
  switch (pyannote/nemo/volume), dev latency readout (on by default), model
  status + reload.
- Mic hardening: autoplay-policy resume, track-ended auto-stop, resume-pop
  guard, stall watchdog (one auto-recapture), WS auto-reconnect with fresh
  server session, and a "Catching up…" indicator when the server lags.

No weights / no GPU? The backend boots in clearly-labelled **demo mode**
(mock ASR/MT) so the UI, diarizer and exports all still work.

## 3. Config (`config.yaml`)

```yaml
device: auto            # auto | cuda | cpu   (env DEVICE overrides, affects MT)
asr_model: D:/DENSEV2 - reading/zipformer zh-en-vi onnx phaseB s2a  # local ONNX dir
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
noise_suppress: true    # DeepFilterNet pre-ASR denoising (UI toggle overrides per request)
df_model: null          # null = DeepFilterNet3; alt: DeepFilterNet2, DeepFilterNet
df_post_filter: false
df_atten_lim_db: null   # e.g. 12 caps suppression, keeps ambience
diarizer: nemo                   # nemo (Titanet embeddings) | volume (levels)
use_nemo: true
nemo_embedding_model: nvidia/speakerverification_en_titanet_large
nemo_cos_thresh: 0.55            # below this cosine -> new speaker (to max_speakers)
frame_seconds: 0.5       # mic frame size for the streaming recognizer
endpoint_silence: 1.0    # pause (s) that finalizes a segment
endpoint_min_speech: 0.5
max_segment: 20.0        # force-finalize run-on speech
mt_retranslate_words: 5  # live re-translation pace (new words); finals verify the last SSBD draft
segment_seconds: 5.0     # fixed-window size for the /api/transcribe upload path
max_speakers: 3
```

Env overrides: `DEVICE=cuda|cpu`, `DTYPE`, `ASR_MODEL`, `MT_MODEL`, `DEMO_MODE=true|false`.

## 4. Voice suppression — DeepFilterNet pre-ASR stage

Every segment is denoised *before* diarization/ASR (`app/enhancer.py`):
16 kHz audio → 48 kHz → `DeepFilterNet3` enhance → back to 16 kHz.
Uploads are denoised whole-file in one pass (no chunk boundaries); live-mic
chunks per segment. DeepFilterNet puts its own model on GPU when available,
CPU otherwise; if the package/weights are missing it becomes a transparent
pass-through (check `/api/health → denoise.backend`). Toggle per session with
the **🔇 Denoise** checkbox, `denoise=true|false` on `/api/transcribe`, or
`{"denoise": false}` on `/ws/live`.

## 5. Speaker diarization — NeMo Titanet direct-forward (default)

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
`POST /api/settings {"diarizer": "pyannote"|"nemo"|"volume"}` (Settings view
does this; in-flight sessions keep theirs).

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
python scripts/smoke_test.py      # device, denoiser, diarizer, pipeline, API routes
python scripts/streaming_smoke.py # REAL engines: live partials + endpointed finals + MT
```

## 8. Backend API (serves Electron + C#)

- `POST /api/transcribe` (multipart `file`, `targets=en,zh`, `src_lang=auto`, `denoise=auto|true|false`, `terms={json}`)
- `POST /api/translate` (`{text, targets[], src, terms?}`) — text translation
- `WS /ws/live` — `stream_start {targets, src_lang, denoise, terms?}` /
  `stream_audio` / `stream_stop` → `partial`, per-target `tok` token deltas
  during final translation, `utterance` finals
- `POST /api/settings` (`{"diarizer": "pyannote"|"nemo"|"volume"}`) — runtime switch
- `GET /api/health` — device, ASR/MT/denoise/diarizer status
- Glossary `terms` (`{source: target}`) ride Hy-MT2's documented terminology
  block; verified steering output (e.g. forced `光子引擎`).

## Layout

```
config.yaml  run.py  requirements.txt
app/engines.py  app/device.py  app/zipformer_engine.py  app/mt_engine.py
app/diarizer.py  app/enhancer.py  app/audio_io.py  app/pipeline.py
app/streaming.py  app/main.py
electron/package.json  electron/tsconfig.json
electron/src/main.ts  electron/src/preload.ts
electron/ui/ (React+Vite+Tailwind redesign, wired live: src/App.tsx, src/api.ts)
csharp/ConfLive/*.csproj,*.xaml,*.cs  installer/ConfLive.iss + build.ps1
scripts/download_models.py  scripts/smoke_test.py  scripts/streaming_smoke.py
```

> Note on `AGENTS.md`: the workspace-root `AGENTS.md` is the Apple design-review
> skill manual — it contains no UI spec for this app, so the UI keeps its
> current layout, relabelled for Zipformer + Hy-MT2 FP8.
