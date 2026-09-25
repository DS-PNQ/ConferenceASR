// App-wide constants: languages, model URLs, tuning defaults.
//
// ASR model: Qwen/Qwen3-ASR-0.6B (safetensors + vocab.json + merges.txt).
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

  /// Name expected by Qwen3-ASR `setLanguage()`.
  /// Empty string = auto-detect.
  String get qwenName {
    switch (this) {
      case AppLang.auto:
        return '';
      case AppLang.vi:
        return 'Vietnamese';
      case AppLang.en:
        return 'English';
      case AppLang.zh:
        return 'Chinese';
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
  // ---- ASR (Qwen3-ASR-0.6B) -------------------------------------------
  // Files expected inside the ASR model dir (qwen_asr requirement).
  static const asrRequiredFiles = ['model.safetensors', 'vocab.json', 'merges.txt'];
  // Default upstream repo (user must accept its license). Download via
  // huggingface-cli or the in-app ModelManager (HF resolve URLs below).
  static const asrHfRepo = 'Qwen/Qwen3-ASR-0.6B';
  static const asrHfBase =
      'https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/main';

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

  // ---- Audio -------------------------------------------------------------
  static const sampleRate = 16000;
  static const channels = 1;

  // ---- Streaming ASR tuning (qwen_asr) -----------------------------------
  static const streamChunkSec = 2.0;
  static const streamUnfixedChunks = 2;
  static const streamMaxNewTokens = 32;
  static const streamRollback = 5;

  // ---- Diarization defaults ----------------------------------------------
  static const diarSilenceMs = 700;
  static const diarThreshold = 0.32;
  static const diarMaxSpeakers = 6;
}
