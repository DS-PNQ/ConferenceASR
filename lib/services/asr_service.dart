import 'dart:async';
import 'dart:typed_data';

import 'package:qwen_asr/qwen_asr.dart';

import '../config/app_config.dart';

/// Result of one streaming push: committed stable text + revisable tail.
class AsrPartial {
  AsrPartial(this.stable, this.provisional);
  final String stable;
  final String provisional;
}

/// Thin wrapper over `qwen_asr`'s [QAsrEngine] with app-level conventions.
///
/// Two decode modes:
/// * **Utterance mode (default here):** diarization closes an utterance,
///   we call [transcribeUtterance] (one-shot PCM path — the most
///   WER-validated path in QwenASR). Best for conferences: clean speaker
///   turns, no cross-talk smearing.
/// * **Live streaming mode:** mic frames go straight to [pushStreamChunk]
///   for lowest latency (stable + provisional text). Speaker labels still
///   come from the diarizer running in parallel; turns may split mid-chunk.
///
/// Language handling (qwen_asr API):
/// * [setSourceLang] → `setLanguage(qwenName)` before the session, plus
///   `setMultilingual(true)` when source is Auto so each utterance is
///   re-detected (vi/en/zh code-switching inside one meeting).
class AsrService {
  AsrService();

  QAsrEngine? _engine;
  bool _ready = false;

  bool get isReady => _ready;

  /// Load the model once (heavy: ~2–3 s, ~1.8 GB dir for the 0.6B model).
  /// [modelDir] must contain model.safetensors, vocab.json, merges.txt.
  Future<void> load(String modelDir, {AppLang lang = AppLang.auto}) async {
    await dispose();
    _engine = await QAsrEngine.load(modelDir);
    _engine!.setStreamChunkSec(AppConfig.streamChunkSec);
    _engine!.setStreamUnfixedChunks(AppConfig.streamUnfixedChunks);
    _engine!.setStreamMaxNewTokens(AppConfig.streamMaxNewTokens);
    _engine!.setStreamRollback(AppConfig.streamRollback);
    _engine!.setPastTextConditioning(true);
    await setSourceLang(lang);
    _ready = true;
  }

  Future<void> setSourceLang(AppLang lang) async {
    final e = _engine;
    if (e == null) return;
    if (lang == AppLang.auto) {
      e.setLanguage('');
      e.setMultilingual(true);
    } else {
      e.setLanguage(lang.qwenName);
      e.setMultilingual(false);
    }
    try {
      await e.streamReset();
    } catch (_) {
      // streamReset before first session may no-op on some builds.
    }
  }

  Future<void> resetStream() async {
    await _engine?.streamReset();
  }

  /// One-shot transcription of a diarized utterance (preferred path).
  Future<String> transcribeUtterance(Float32List pcm16kMono) {
    final e = _engine;
    if (e == null) throw StateError('ASR engine not loaded.');
    return e.transcribePcm(pcm16kMono);
  }

  /// Streaming push of a mic chunk. [finalize] flushes the tail on Stop.
  Future<AsrPartial> pushStreamChunk(
    Float32List chunk, {
    bool finalize = false,
  }) async {
    final e = _engine;
    if (e == null) throw StateError('ASR engine not loaded.');
    final StreamPartial p = await e.streamPush(chunk, finalize: finalize);
    return AsrPartial(p.text, p.provisional);
  }

  Future<String> transcribeFile(String wavPath) {
    final e = _engine;
    if (e == null) throw StateError('ASR engine not loaded.');
    return e.transcribeFile(wavPath);
  }

  String perfStats() {
    try {
      return _engine?.perfStats() ?? '';
    } catch (_) {
      return '';
    }
  }

  Future<void> dispose() async {
    _ready = false;
    try {
      _engine?.dispose();
    } catch (_) {
      // Dispose is best-effort across plugin versions.
    }
    _engine = null;
  }
}
