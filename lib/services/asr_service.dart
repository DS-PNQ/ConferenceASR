import 'dart:typed_data';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa;

import '../config/app_config.dart';

/// One offline transcription: text plus the model's own language id
/// (e.g. 'en', 'zh', 'vi', '' when unknown).
class Transcription {
  Transcription(this.text, this.lang);
  final String text;
  final String lang;
}

/// Thin wrapper over sherpa-onnx's offline Qwen3-ASR recognizer
/// (model `sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25`).
///
/// Notes on engine semantics (differences vs the previous implementation):
/// * **Offline-only.** sherpa-onnx exposes Qwen3-ASR as a non-streaming
///   model, so every call decodes a complete audio buffer. The app's
///   diarizer supplies utterance boundaries; low-latency "live" text is a
///   throttled re-decode of the in-progress utterance (see SessionProvider).
/// * **Language is auto-detected** by the model per utterance — there is no
///   `setLanguage` equivalent in `OfflineQwen3AsrModelConfig`. The source
///   selector in the UI drives translation + labels; [appLangForResult]
///   maps the model's own `lang` id back to [AppLang] for segment badges.
/// * `decode()` is **synchronous FFI** and blocks the calling isolate for
///   the decode duration (matches the official sherpa-onnx Flutter
///   examples). Calls are short (RTF ~0.1–0.2) and run one at a time from
///   the provider's serial pipeline.
class AsrService {
  AsrService();

  sherpa.OfflineRecognizer? _recognizer;
  static bool _bindingsInit = false;

  /// Milliseconds the last [transcribeUtterance] decode took (0 = none yet).
  int lastDecodeMs = 0;

  bool get isReady => _recognizer != null;

  /// Load the model once. [modelDir] is [ModelManager.asrDir]: it must
  /// contain encoder/decoder/conv-frontend ONNX files plus `tokenizer/`.
  Future<void> load(String modelDir) async {
    await dispose();
    if (!_bindingsInit) {
      sherpa.initBindings();
      _bindingsInit = true;
    }
    final cfg = sherpa.OfflineQwen3AsrModelConfig(
      convFrontend: '$modelDir/conv_frontend.onnx',
      encoder: '$modelDir/encoder.int8.onnx',
      decoder: '$modelDir/decoder.int8.onnx',
      tokenizer: '$modelDir/tokenizer',
      maxNewTokens: AppConfig.asrMaxNewTokens,
    );
    try {
      _recognizer = sherpa.OfflineRecognizer(
        sherpa.OfflineRecognizerConfig(
          model: sherpa.OfflineModelConfig(
            qwen3Asr: cfg,
            tokens: '',
            numThreads: AppConfig.asrNumThreads,
            debug: false,
          ),
        ),
      );
    } catch (e) {
      _recognizer = null;
      throw StateError(
        'Could not create sherpa-onnx Qwen3-ASR recognizer in $modelDir: $e. '
        'Re-download the ASR model (Setup → Download models); partial or '
        'truncated ONNX files fail here.',
      );
    }
  }

  /// Transcribe one 16 kHz mono float PCM buffer (e.g. a diarized utterance
  /// or the in-progress live buffer). Buffers shorter than 0.4 s are
  /// skipped without touching the engine.
  Future<Transcription> transcribeUtterance(Float32List pcm16kMono) {
    final r = _recognizer;
    if (r == null) throw StateError('ASR engine not loaded.');
    if (pcm16kMono.length < AppConfig.sampleRate * 4 ~/ 10) {
      return Future.value(Transcription('', ''));
    }
    final stream = r.createStream();
    try {
      stream.acceptWaveform(
        samples: pcm16kMono,
        sampleRate: AppConfig.sampleRate,
      );
      final sw = Stopwatch()..start();
      r.decode(stream);
      lastDecodeMs = sw.elapsedMilliseconds;
      final result = r.getResult(stream);
      return Future.value(
        Transcription(result.text.trim(), result.lang.trim()),
      );
    } finally {
      stream.free();
    }
  }

  /// Map a sherpa result `lang` id ('en', 'zh', 'yue', 'vi', …) to [AppLang].
  /// Unknown/empty ids fall back to [fallback] (the session source setting).
  static AppLang appLangForResult(String code, AppLang fallback) {
    switch (code.toLowerCase()) {
      case 'vi':
        return AppLang.vi;
      case 'en':
        return AppLang.en;
      case 'zh':
      case 'zh-cn':
      case 'zh-hant':
      case 'yue':
        return AppLang.zh;
      default:
        return fallback;
    }
  }

  Future<void> dispose() async {
    try {
      _recognizer?.free();
    } catch (_) {
      // Best-effort native cleanup.
    }
    _recognizer = null;
  }
}
