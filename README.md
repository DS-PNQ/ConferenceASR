# ConfLive — Trilingual Conference ASR + Translator

Record a conference in **Vietnamese · English · Chinese** and watch it appear on screen
like a chat app: speaker-labelled bubbles, live transcription, side-by-side translations.

Two frontends, one Python inference backend (`run.py`: Zipformer ASR, Hy-MT2 FP8 MT,
volume diarization, DeepFilterNet hook):

- **Electron + TypeScript app** (`electron/`, primary): `npm run build && npx electron .`.
  Finds `run.py`, picks CUDA/CPU itself, streams mic over `/ws/live` with live
  partials and token-streamed finals. `npm run dist` builds a Windows installer.
- **C# desktop app** (`csharp/ConfLive`): native WPF window, same protocol.

## GPU usage notes (measured RTX 4060 Laptop, Hy-MT2-1.8B-FP8)

- Weights load to `cuda:0` (fp16 after compressed-tensors decompress) and generate
  runs there too — verified: params on `cuda:0`, SM utilization 21–43% during
  decode, ~4 tok/s batch-1 greedy. If `nvidia-smi` shows ~0% between utterances,
  that is normal: single-stream decode of a 1.8B model is latency-bound with
  tiny kernels, and the GPU idles between segments.
- ASR (sherpa-onnx wheels) is CPU-only by build; int8 decodes ~15x realtime,
  so it never bottlenecks. MT stays on CUDA.
- `mt_use_cache: true` is set, though this custom modeling largely ignores it;
  per-token latency is a property of the checkpoint, not a device bug.

| Module | Model (default) | Source |
|---|---|---|
| Denoise | `DeepFilterNet3` (auto-download) | https://github.com/Rikorose/DeepFilterNet |
| ASR | Zipformer zh-en-vi ONNX, `phaseB_s2a` (local folder, sherpa-onnx CPU int8) | `D:\DENSEV2 - reading\zipformer zh-en-vi onnx phaseB s2a` |
| Translation | `tencent/Hy-MT2-1.8B-FP8` (compressed-tensors, CUDA) | https://huggingface.co/collections/tencent/hy-mt2 |
| Diarization | volume-based (default) · NeMo optional | https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html |

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
npm run build      # tsc -> dist/
npx electron .     # launches; finds run.py, picks CUDA/CPU, opens the window
npm run dist       # Windows NSIS installer (needs electron-builder downloads)
```

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

In the window:
- Models load in the background (status line reports ASR / MT / denoise backends).
- **● Record**: the streaming Zipformer recognizer stays open while you speak —
  a live partial bubble grows in realtime; final translations **stream in token
  by token** (`tok` events) behind it. Segments finalize on **pauses, not the
  clock**, so sentences are never cut mid-word; the partial is then replaced by
  the final translated bubble.
- **⇪ Upload**: any recording streams through the same endpointing pipeline.
- Sidebar: mic picker, source language (or auto-detect), EN / 中文 / VI targets,
  DeepFilterNet switch, volume meter, **.txt / .srt export**, Clear.

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
diarizer: volume        # volume | nemo
use_nemo: false
frame_seconds: 0.5       # mic frame size for the streaming recognizer
endpoint_silence: 1.0    # pause (s) that finalizes a segment
endpoint_min_speech: 0.5
max_segment: 20.0        # force-finalize run-on speech
partial_translate_interval: 2.5
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

## 5. Speaker diarization — “based on volume” + NeMo path

Default `VolumeDiarizer` (`app/diarizer.py`): per-chunk RMS dBFS + pause-gap turn
detection + running loudness centroids per speaker (+ stereo pan when present).
Transparent, dependency-free, good for conference mics / per-seat level differences.

NeMo path: set `diarizer: nemo`, `use_nemo: true`, install `nemo_toolkit[asr]`;
`NeMoDiarizer` wraps `MSDDiarizationModel` per the NeMo diarization docs and falls
back to volume mode with a warning if NeMo is unavailable.

## 6. MT upgrades: token streaming, dialogue context, speculative decoding

- **Token streaming** (`translate_stream`): finals generate into `tok` events
  (`{id, tgt, seq, delta}`) over `/ws/live`, rendered live with a blinking caret
  in Electron. Implemented with an id-collecting streamer because this custom
  modeling echoes prompt tokens through `TextIteratorStreamer`; verified
  `joined == one-shot` output exactly.
- **Dialogue context** (`mt_history_turns/chars`): recent finalized segments travel
  in the prompt as terminology/style reference, fenced with DO-NOT-translate.
  (Appending history after the instruction made the model translate the history
  and ramble 3x — prompt order matters.)
- **Prompt-lookup speculative decoding** (`mt_speculative`, drafter-free, exact):
  measured on RTX 4060, interleaved n=3–4 — **0.76–1.0x cross-lingual, ~1.0x
  same-language: no reliable win**, so OFF by default. It auto-fires for
  `src==tgt` or ASCII→English where drafts can only help. A dedicated drafter
  model would be needed for real speculative gains; none exists in Hy-MT2.

## 7. Verify (engines need cached weights; fakes don't)

```powershell
python scripts/smoke_test.py      # device, denoiser, diarizer, pipeline, API routes
python scripts/streaming_smoke.py # REAL engines: live partials + endpointed finals + MT
```

## 8. Backend API (serves Electron + C#)

- `POST /api/transcribe` (multipart `file`, `targets=en,zh`, `src_lang=auto`, `denoise=auto|true|false`)
- `POST /api/translate` (`{text, targets[], src}`) — text translation
- `WS /ws/live` — `stream_start` / `stream_audio` / `stream_stop` → `partial`,
  per-target `tok` token deltas during final translation, `utterance` finals
- `GET /api/health` — device, ASR/MT/denoise status

## Layout

```
config.yaml  run.py  requirements.txt
app/engines.py  app/device.py  app/zipformer_engine.py  app/mt_engine.py
app/diarizer.py  app/enhancer.py  app/audio_io.py  app/pipeline.py
app/streaming.py  app/main.py
electron/package.json  electron/tsconfig.json  electron/index.html
electron/styles.css  electron/src/main.ts  electron/src/preload.ts  electron/src/renderer.ts
csharp/ConfLive/*.csproj,*.xaml,*.cs  installer/ConfLive.iss + build.ps1
scripts/download_models.py  scripts/smoke_test.py  scripts/streaming_smoke.py
```

> Note on `AGENTS.md`: the workspace-root `AGENTS.md` is the Apple design-review
> skill manual — it contains no UI spec for this app, so the UI keeps its
> current layout, relabelled for Zipformer + Hy-MT2 FP8.
