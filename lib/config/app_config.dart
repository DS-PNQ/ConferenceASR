// App-wide constants: languages, model URLs, tuning defaults.
//
// ASR model: sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25 (ONNX int8,
//   encoder + decoder + conv frontend + tokenizer dir, ~0.94 GB).
// MT model:  tencent/Hy-MT2-1.8B-1.25Bit-GGUF, file Hy-MT2-1.8B-1.25Bit.gguf
//            (~462 MB, 1.25-bit, needs a llama.cpp build with Tencent's STQ
//            kernel — see TranslationService docs for the fallback models).

/// Supported UI / speech languages. Keep codes BCP-47-ish.
enum AppLang { auto, vi, en, zh }

extension AppLangX on AppLang {
  String get code {
    switch (this) {
      case AppLang.auto:
        return 'auto';
      case AppLang.vi:
        return 'vi';
      case AppLang.en:
        return 'en';
      case AppLang.zh:
        return 'zh';
    }
  }

  String get label {
    switch (this) {
      case AppLang.auto:
        return 'Auto';
      case AppLang.vi:
        return 'Tiếng Việt';
      case AppLang.en:
        return 'English';
      case AppLang.zh:
        return '中文';
    }
  }

  /// Full language name expected inside Hy-MT2 translation prompts.
  String get hyName {
    switch (this) {
      case AppLang.auto:
        return 'Auto';
      case AppLang.vi:
        return 'Vietnamese';
      case AppLang.en:
        return 'English';
      case AppLang.zh:
        return 'Chinese';
    }
  }

  static AppLang fromCode(String code) {
    switch (code) {
      case 'vi':
        return AppLang.vi;
      case 'en':
        return AppLang.en;
      case 'zh':
        return AppLang.zh;
      default:
        return AppLang.auto;
    }
  }
}

abstract final class AppConfig {
  // ---- ASR (sherpa-onnx Qwen3-ASR 0.6B int8) -------------------------------
  // Files expected inside the ASR model dir (paths relative to it).
  static const asrRequiredFiles = [
    'encoder.int8.onnx',
    'decoder.int8.onnx',
    'conv_frontend.onnx',
    'tokenizer/vocab.json',
    'tokenizer/merges.txt',
    'tokenizer/tokenizer_config.json',
  ];
  // Upstream mirror of
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2
  // (individual files so the in-app downloader supports resume + progress).
  static const asrHfRepo =
      'csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25';
  static const asrHfBase =
      'https://huggingface.co/csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/resolve/main';
  static const asrDirName = 'sherpa-onnx-qwen3-asr-0.6b-int8';
  // ~721 + ~174 + ~42 + ~4 MB.
  static const asrApproxBytes = 941000000;

  /// Local dir name of the previous (qwen_asr / safetensors) implementation.
  /// Deleted automatically on the next download run to reclaim ~1.9 GB.
  static const legacyAsrDirName = 'qwen3-asr-0.6b';

  // sherpa-onnx OfflineQwen3AsrModelConfig tuning (see model card examples).
  static const asrMaxNewTokens = 512;
  static const asrNumThreads = 2;

  // Live-preview cadence for low-latency mode (offline engine → provisional
  // text is a re-decode of the in-progress utterance, so keep it sparse).
  static const livePreviewMinIntervalMs = 6000;
  static const livePreviewMinAudioMs = 3000;

  // ---- MT (Hy-MT2) ------------------------------------------------------
  static const mtHfRepo = 'tencent/Hy-MT2-1.8B-1.25Bit-GGUF';
  static const mtFileName = 'Hy-MT2-1.8B-1.25Bit.gguf';
  static const mtHfUrl =
      'https://huggingface.co/tencent/Hy-MT2-1.8B-1.25Bit-GGUF/resolve/main/Hy-MT2-1.8B-1.25Bit.gguf';
  static const mtApproxBytes = 461860800;

  /// Fallbacks if the 1.25-bit file cannot be loaded by the bundled
  /// llama.cpp (it needs Tencent's STQ kernel, PR ggml-org/llama.cpp#22836).
  static const mtFallbacks = [
    // (label, url)
    (
      'Hy-MT2-1.8B-GGUF (Q4, most compatible)',
      'https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/resolve/main/hy-mt2-1.8b-q4_k_m.gguf',
    ),
    (
      'Hy-MT2-1.8B-2bit-GGUF',
      'https://huggingface.co/tencent/Hy-MT2-1.8B-2bit-GGUF/resolve/main/hy-mt2-1.8b-2bit.gguf',
    ),
  ];

  // Recommended Hy-MT2 sampling params for 1.8B/7B (model card).
  static const mtTemperature = 0.7;
  static const mtTopP = 0.6;
  static const mtTopK = 20;
  static const mtRepeatPenalty = 1.05;
  static const mtMaxTokens = 256;

  // ---- Device requirements ----------------------------------------------
  // Resident RAM: sherpa-onnx Qwen3-ASR-0.6B int8 ~1.5 GB (conservative),
  // Hy-MT2-1.8B ~0.9 GB. Below these free-RAM levels the OS low-memory
  // killer terminates the app (silent death, no exception). The provider
  // checks before loading.
  static const asrNeedFreeBytes = 1500000000;
  static const mtNeedFreeBytes = 900000000;

  // ---- Audio -------------------------------------------------------------
  static const sampleRate = 16000;
  static const channels = 1;

  // ---- Diarization defaults ----------------------------------------------
  static const diarSilenceMs = 700;
  static const diarThreshold = 0.32;
  static const diarMaxSpeakers = 6;
}
