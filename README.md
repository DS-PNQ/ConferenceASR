# Conference Translator — on-device ASR + translation (vi / en / zh)

Android app that **records a conference, diarizes speakers, transcribes live,
and translates** — fully on-device after a one-time model download.

| Piece | Implementation |
|---|---|
| ASR | [`qwen_asr` 0.4.0](https://pub.dev/packages/qwen_asr) — Qwen3-ASR-0.6B, offline Rust engine, streaming + one-shot |
| Translation | [`tencent/Hy-MT2-1.8B-1.25Bit-GGUF`](https://huggingface.co/tencent/Hy-MT2-1.8B-1.25Bit-GGUF) (`Hy-MT2-1.8B-1.25Bit.gguf`, ~462 MB) via [`llama_flutter_android`](https://pub.dev/packages/llama_flutter_android) (llama.cpp + Vulkan) |
| Languages | Vietnamese · English · Chinese (auto-detect + code-switching per utterance) |
| Diarization | On-device VAD + spectral-embedding clustering (no server, no extra model) |
| Mic | `record` package, 16 kHz mono PCM16 → float |

## Screens

1. **Setup** — download/verify models, toggles (translation / diarization /
   low-latency streaming), engine warm-up.
2. **Conference (live session)** — record/stop, mic meter, speaker count,
   live partial strip (stable + *provisional* tail), transcript bubbles with
   per-utterance translation, long-press speaker chip to rename, copy / share /
   clear, language switch with re-translate-all.

## Project layout

```
lib/
  main.dart                    app root + wakelock
  config/app_config.dart       languages, model URLs, tuning constants
  models/transcript_segment.dart
  providers/session_provider.dart   orchestrates mic→diar→ASR→MT
  services/
    audio_capture_service.dart mic → 20 ms float frames
    vad.dart                   energy + ZCR voice activity detection
    speaker_embedding.dart     spectral fingerprint embedding
    diarization_service.dart   turn splitting + cosine clustering
    asr_service.dart           qwen_asr wrapper (utterance + streaming)
    translation_service.dart   Hy-MT2 GGUF via llama.cpp
    model_manager.dart         HF downloads with progress
  ui/screens/  setup_screen.dart, live_session_screen.dart
  ui/widgets/  transcript_widgets.dart (bubbles, language picker, live strip)
```

## Setup

Requirements: Flutter ≥ 3.24, Android SDK 35, NDK 28, JDK 17.

```bash
flutter pub get
# connect an arm64 Android device (Snapdragon 8-class recommended)
flutter run
```

In the app: **Download models** (Wi-Fi! ASR ~1.8 GB + MT ~462 MB), then
**Start conference session** and grant mic permission.

### Models

* **ASR** — `Qwen/Qwen3-ASR-0.6B` files `model.safetensors`, `vocab.json`,
  `merges.txt` into `<appdocs>/models/qwen3-asr-0.6b/`. The in-app downloader
  fetches them from `https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/main/`,
  or pre-seed with:
  ```bash
  huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir <dir>
  ```
* **MT** — `Hy-MT2-1.8B-1.25Bit.gguf` into `<appdocs>/models/`. In-app
  downloader uses the default URL in `AppConfig.mtHfUrl`, or:
  ```bash
  curl -L -o Hy-MT2-1.8B-1.25Bit.gguf \
    https://huggingface.co/tencent/Hy-MT2-1.8B-1.25Bit-GGUF/resolve/main/Hy-MT2-1.8B-1.25Bit.gguf
  ```

> **1.25-bit compatibility:** this GGUF needs Tencent's STQ kernel
> (ggml-org/llama.cpp PR #22836). If `loadModel` throws, the app surfaces a
> fallback hint — use `Hy-MT2-1.8B-GGUF` (Q4, most compatible) or
> `Hy-MT2-1.8B-2bit-GGUF`; same prompt format, just pass that file path to
> `TranslationService.load` / place it at the expected MT path. URLs are in
> `AppConfig.mtFallbacks`.

## How it works

### ASR (`AsrService`, qwen_asr)

* Engine loaded once: `QAsrEngine.load(modelDir)` + streaming tuning
  (`setStreamChunkSec(2.0)`, `setStreamUnfixedChunks(2)`, `setPastTextConditioning(true)`).
* **Utterance mode (default):** diarizer closes an utterance →
  `transcribePcm(float32 16 kHz mono)` (the most WER-validated path).
* **Low-latency mode:** each mic frame → `streamPush(chunk)`; UI renders
  `text` (stable) + `provisional` (grey italic tail that may be revised);
  `streamPush(empty, finalize: true)` on Stop.
* Languages: `setLanguage('Vietnamese'|'English'|'Chinese'|'' )`;
  Auto source also sets `setMultilingual(true)` so each utterance is
  re-detected (handles vi/en/zh code-switching inside one meeting).

### Translation (`TranslationService`, Hy-MT2)

* Prompt follows the model card exactly:
  `Translate the following {src} text into {tgt}. Note that you should only
  output the translated result without any additional explanation:\n\n{text}`
* Sampling from the card (1.8B): temperature 0.7, top_p 0.6, top_k 20,
  repeat penalty 1.05, max 256 tokens.
* Calls are serialized through an internal queue (single llama context);
  empty input short-circuits; same-language short-circuits; outputs are
  cleaned (prompt-echo / quote stripping).
* GPU layers auto-detected (`detectGpu().recommendedGpuLayers`), CPU fallback.

### Speaker diarization (`DiarizationService`, no extra model)

* `Vad`: 20 ms frames, energy + zero-crossing-rate with adaptive noise floor
  and hangover bridging.
* Utterance closes after ~700 ms silence (tunable) or 20 s safety cap.
* `SpeakerEmbedding`: per-utterance log-spectral mean+variance fingerprint,
  standardized + L2-normalized; cosine-distance threshold clustering
  (default 0.32, max 6 speakers) with online centroid updates.
* Honest limits: separates 2–4 speakers in a quiet room; merges similar
  voices in noise; no overlap handling. For production-grade diarization,
  swap `SpeakerEmbedding.embed` with an ONNX x-vector/ECAPA model — the
  clustering/turn logic is unchanged. Long-press a speaker chip to rename
  (rewrites past + future turns).

## Permissions (Android)

`RECORD_AUDIO`, `INTERNET` (one-time download only), foreground-service mic
bits. `minSdk 26`, `targetSdk 35`, shrinking disabled to protect JNI symbols.

## Tests

```bash
flutter test
```

Covers language-name mappings (Qwen + Hy-MT2), the Hy-MT2 prompt template,
VAD speech/silence classification, and transcript export.

## License

App code: MIT. Models keep their upstream licenses (Qwen ASR + Tencent
Hy-MT2 Apache-2.0); accept them when downloading. `qwen_asr` itself is MIT.
