import 'dart:async';

import 'package:llama_flutter_android/llama_flutter_android.dart';

import '../config/app_config.dart';

/// On-device translation with Tencent Hy-MT2-1.8B Q4_K_M via llama.cpp.
///
/// Model: https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF
/// File: `Hy-MT2-1.8B-Q4_K_M.gguf` (~1.13 GB). Q4_K_M was chosen as the
/// default because it loads with any stock llama.cpp build.
///
/// HISTORY: the previous default (`Hy-MT2-1.8B-1.25Bit.gguf`, AngelSlim
/// 1.25-bit quant) depends on Tencent's STQ kernel (ggml-org/llama.cpp PR
/// #22836), which the prebuilt llama.cpp inside `llama_flutter_android`
/// lacks — `load` threw "Failed to load model" for it on device. Behaviour
/// on load failure:
/// * [load] throws with a message listing the alternative files
///   (2-bit, same prompt format).
/// * Everything else (prompt builder, queueing) is identical across the
///   Hy-MT2 GGUF family.
///
/// Prompt format follows the model card ("only output the translated result
/// without any additional explanation"), with sampling params from the card:
/// temperature 0.7, top_p 0.6, top_k 20, repeat penalty 1.05.
class TranslationService {
  TranslationService() : _controller = LlamaController();

  final LlamaController _controller;
  bool _loaded = false;
  String? _modelPath;
  Future<void> _queue = Future.value();

  bool get isLoaded => _loaded;
  String? get modelPath => _modelPath;

  bool isFallbackHint(Object e) =>
      e.toString().toLowerCase().contains('fallback') ||
      e.toString().toLowerCase().contains('stq') ||
      e.toString().toLowerCase().contains('load');

  /// Free device RAM in bytes, or -1 if it cannot be determined.
  /// Used for the pre-load memory check (the OS kills the app silently
  /// once both models are resident on a RAM-starved device).
  Future<int> freeRamBytes() async {
    try {
      return (await _controller.detectGpu()).freeRamBytes;
    } catch (_) {
      return -1;
    }
  }

  /// Load a GGUF file. Uses GPU auto-detection (Vulkan) when available.
  Future<void> load(String ggufPath) async {
    if (_loaded && _modelPath == ggufPath) return;
    await unload();
    try {
      int gpuLayers = 0;
      try {
        final gpu = await _controller.detectGpu();
        gpuLayers = gpu.recommendedGpuLayers;
      } catch (_) {
        gpuLayers = 0; // detection is best-effort; CPU always works
      }
      await _controller.loadModel(
        modelPath: ggufPath,
        threads: 4,
        contextSize: 2048,
        gpuLayers: gpuLayers,
      );
      _loaded = true;
      _modelPath = ggufPath;
    } catch (e) {
      _loaded = false;
      throw StateError(
        'Could not load $ggufPath with the bundled llama.cpp: $e. '
        'If this is the 1.25-bit file, your llama.cpp build likely lacks '
        'the STQ kernel (PR #22836). FALLBACK: use Hy-MT2-1.8B-GGUF (Q4) or '
        'Hy-MT2-1.8B-2bit-GGUF instead — same API, same prompts.',
      );
    }
  }

  /// Build the Hy-MT2 translation prompt (English instruction template).
  static String buildPrompt({
    required String text,
    required AppLang source,
    required AppLang target,
    String? terminology,
  }) {
    final targetName = target.hyName;
    final termBlock = (terminology != null && terminology.trim().isNotEmpty)
        ? 'Reference the following translations:\n$terminology\n\n'
        : '';
    final srcBlock = source == AppLang.auto
        ? 'Translate the following text into $targetName.'
        : 'Translate the following ${source.hyName} text into $targetName.';
    return '$termBlock$srcBlock Note that you should only output the '
        'translated result without any additional explanation:\n\n$text';
  }

  /// Translate [text] into [target]. Serializes concurrent calls through an
  /// internal queue because the underlying llama context is single-shot.
  /// Empty input short-circuits to '' without hitting the model.
  Future<String> translate({
    required String text,
    required AppLang source,
    required AppLang target,
  }) {
    final completer = Completer<String>();
    _queue = _queue.then((_) async {
      try {
        final out = await _run(text: text, source: source, target: target);
        completer.complete(out);
      } catch (e, st) {
        completer.completeError(e, st);
      }
    });
    return completer.future;
  }

  Future<String> _run({
    required String text,
    required AppLang source,
    required AppLang target,
  }) async {
    if (!_loaded) throw StateError('Translation model not loaded.');
    final input = text.trim();
    if (input.isEmpty) return '';
    // Skip translation when source == target (still normalize whitespace).
    if (source != AppLang.auto && source == target) return input;

    final prompt = buildPrompt(text: input, source: source, target: target);
    final buf = StringBuffer();
    await for (final token in _controller.generate(
      prompt: prompt,
      maxTokens: AppConfig.mtMaxTokens,
      temperature: AppConfig.mtTemperature,
      topP: AppConfig.mtTopP,
      topK: AppConfig.mtTopK,
      repeatPenalty: AppConfig.mtRepeatPenalty,
    )) {
      buf.write(token);
    }
    return _clean(buf.toString());
  }

  /// Strip echo / quotes / explanation wrappers models sometimes add.
  static String _clean(String raw) {
    var s = raw.trim();
    // If the model echoed the prompt, keep text after the last blank line.
    if (s.contains('\n\n')) {
      final parts = s.split('\n\n');
      s = parts.last.trim();
    }
    if ((s.startsWith('"') && s.endsWith('"')) ||
        (s.startsWith('“') && s.endsWith('”')) ||
        (s.startsWith("'") && s.endsWith("'"))) {
      s = s.substring(1, s.length - 1).trim();
    }
    return s;
  }

  Future<void> stop() async {
    try {
      await _controller.stop();
    } catch (_) {}
  }

  Future<void> unload() async {
    _loaded = false;
    _modelPath = null;
    try {
      await _controller.dispose();
    } catch (_) {}
  }
}
