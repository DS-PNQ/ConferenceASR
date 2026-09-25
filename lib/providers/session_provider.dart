import 'dart:async';

import 'package:flutter/foundation.dart';

import '../config/app_config.dart';
import '../models/transcript_segment.dart';
import '../services/asr_service.dart';
import '../services/audio_capture_service.dart';
import '../services/diarization_service.dart';
import '../services/model_manager.dart';
import '../services/translation_service.dart';

/// Owns the whole live-conference pipeline and exposes UI state.
///
/// Pipeline (both modes — sherpa-onnx Qwen3-ASR is an offline engine):
///   mic 16 kHz PCM → [DiarizationService] (VAD + turn split + speaker id)
///   → closed utterance → [AsrService.transcribeUtterance] (offline decode)
///   → [TranscriptSegment] appended → [TranslationService.translate] (async)
///   → segment updated in place.
///
/// Low-latency mode additionally re-decodes the in-progress utterance every
/// few seconds and renders it as provisional (grey italic) text; the final
/// committed segment always comes from the utterance-close decode, exactly
/// like utterance mode.
class SessionProvider extends ChangeNotifier {
  SessionProvider({
    ModelManager? models,
    AsrService? asr,
    TranslationService? mt,
    AudioCaptureService? audio,
    DiarizationService? diar,
  })  : models = models ?? ModelManager(),
        asr = asr ?? AsrService(),
        mt = mt ?? TranslationService(),
        audio = audio ?? AudioCaptureService(),
        diar = diar ?? DiarizationService() {
    this.diar.onUtteranceClosed = _onUtterance;
    this.diar.onLevel = (rms) {
      micLevel = rms.clamp(0.0, 1.0);
      notifyListeners();
    };
  }

  final ModelManager models;
  final AsrService asr;
  final TranslationService mt;
  final AudioCaptureService audio;
  final DiarizationService diar;

  // ---- Settings ---------------------------------------------------------
  AppLang sourceLang = AppLang.auto;
  AppLang targetLang = AppLang.vi;
  bool translationEnabled = true;
  bool diarizationEnabled = true;
  bool lowLatency = false; // streaming ASR vs utterance ASR

  // ---- Model state ------------------------------------------------------
  bool asrReady = false;
  bool mtReady = false;
  bool modelsLoading = false;
  String modelStatus = 'Checking models…';
  String downloadDetail = '';
  double asrProgress = 0;
  double mtProgress = 0;
  DateTime _lastProgressPush = DateTime.fromMillisecondsSinceEpoch(0);

  // ---- Session state ----------------------------------------------------
  bool sessionActive = false;
  bool busy = false; // transcribing / loading
  String status = 'Idle';
  double micLevel = 0;
  String stableText = '';
  String provisionalText = '';
  final List<TranscriptSegment> segments = [];
  int _nextId = 1;
  StreamSubscription<Float32List>? _audioSub;
  // Live-preview buffer (low-latency mode): frames of the in-progress
  // utterance, periodically re-decoded as provisional text.
  final List<Float32List> _liveBuffer = [];
  int _liveSamples = 0;
  DateTime _lastPreview = DateTime.fromMillisecondsSinceEpoch(0);
  bool _previewRunning = false;

  List<String> get speakers => diar.speakers;

  Future<void> refreshModelStatus() async {
    asrReady = await models.isAsrReady();
    mtReady = await models.isMtReady();
    if (!asrReady && !mtReady) {
      modelStatus = 'ASR + translation models missing.';
    } else if (!asrReady) {
      modelStatus = 'ASR model missing.';
    } else if (!mtReady && translationEnabled) {
      modelStatus = 'ASR ready. Translation model missing (optional).';
    } else {
      modelStatus = 'Models ready.';
    }
    notifyListeners();
  }

  Future<void> downloadModels() async {
    modelsLoading = true;
    downloadDetail = '';
    notifyListeners();
    try {
      modelStatus = 'Downloading ASR model…';
      notifyListeners();
      await models.downloadAsr(
        onProgress: (done, total, file, r, t) {
          asrProgress =
              ModelManager.combineFileProgress(done, total, r, t);
          modelStatus = 'ASR $done/$total: $file';
          downloadDetail = _mbLine(r, t);
          _pushProgressThrottled();
        },
      );
      if (translationEnabled) {
        modelStatus = 'Downloading Hy-MT2 GGUF (~462 MB)…';
        notifyListeners();
        await models.downloadMt(
          onProgress: (r, t) {
            mtProgress = t <= 0 ? 0 : (r / t).clamp(0.0, 1.0);
            modelStatus = 'MT ${AppConfig.mtFileName}';
            downloadDetail = _mbLine(r, t);
            _pushProgressThrottled();
          },
        );
      }
    } on DownloadCancelled {
      modelStatus = 'Download cancelled — partial files kept, resume anytime.';
    } catch (e) {
      modelStatus = 'Download failed: $e';
    } finally {
      modelsLoading = false;
      downloadDetail = '';
      await refreshModelStatus();
    }
  }

  /// Stop an in-progress model download. Partial `.part` files are kept so
  /// the next run resumes instead of restarting.
  void cancelDownload() => models.cancel();

  static String _mbLine(int received, int total) {
    String mb(int b) => (b / 1048576).toStringAsFixed(0);
    if (total <= 0) return '${mb(received)} MB downloaded';
    final pct = (received / total * 100).toStringAsFixed(0);
    return '${mb(received)} / ${mb(total)} MB ($pct%)';
  }

  /// Progress chunks arrive per network packet (~KBs) — rebuilding the UI
  /// on every one would jank. Throttle to ~4 Hz.
  void _pushProgressThrottled() {
    final now = DateTime.now();
    if (now.difference(_lastProgressPush).inMilliseconds < 250) return;
    _lastProgressPush = now;
    notifyListeners();
  }

  /// Load native engines. Call once from Setup before starting a session.
  /// sherpa-onnx Qwen3-ASR auto-detects the language per utterance — there
  /// is no set-language call; [sourceLang] drives translation + labels.
  Future<void> ensureEngines() async {
    final needMt = translationEnabled && !mt.isLoaded;
    if (!asr.isReady || needMt) {
      // Preflight: both models resident need ~2.4 GB. Without this check the
      // OS low-memory killer just terminates the app with no error.
      final free = await mt.freeRamBytes();
      final need = AppConfig.asrNeedFreeBytes +
          (needMt ? AppConfig.mtNeedFreeBytes : 0);
      if (free > 0 && free < need) {
        throw StateError(
          'Only ${(free / 1048576).round()} MB RAM free, but ~${(need / 1073741824).toStringAsFixed(1)} GB is needed '
          '(ASR ~1.5 GB${needMt ? ' + translation ~0.9 GB' : ''}). '
          'Fix: turn off translation, close other apps, or use a device '
          'with more RAM (8 GB+ recommended).',
        );
      }
    }
    if (!asr.isReady) {
      status = 'Loading ASR engine…';
      notifyListeners();
      await asr.load(await models.asrDir);
    }
    if (translationEnabled && !mt.isLoaded) {
      final p = await models.mtPath;
      status = 'Loading Hy-MT2…';
      notifyListeners();
      await mt.load(p);
    }
    status = 'Engines ready.';
    notifyListeners();
  }

  Future<void> start() async {
    if (sessionActive) return;
    busy = true;
    status = 'Starting…';
    notifyListeners();
    try {
      await ensureEngines();
      diar
        ..silenceMs = AppConfig.diarSilenceMs
        ..threshold = AppConfig.diarThreshold
        ..maxSpeakers = AppConfig.diarMaxSpeakers
        ..reset();
      segments.clear();
      stableText = '';
      provisionalText = '';
      _liveBuffer.clear();
      _liveSamples = 0;
      _previewRunning = false;
      _lastPreview = DateTime.fromMillisecondsSinceEpoch(0);
      await audio.start();
      _audioSub = audio.frames.listen(
        _onFrame,
        onError: (Object e) {
          status = 'Mic error: $e';
          notifyListeners();
        },
      );
      sessionActive = true;
      status = lowLatency ? 'Recording (live)…' : 'Recording…';
    } catch (e) {
      status = 'Start failed: $e';
    } finally {
      busy = false;
      notifyListeners();
    }
  }

  Future<void> stop() async {
    if (!sessionActive) return;
    await _audioSub?.cancel();
    _audioSub = null;
    await audio.stop();
    // Flush the in-progress utterance so its final decode still commits.
    diar.flush();
    // Let the flush-triggered transcription continuation run first, so the
    // segment lands and the count below is accurate (microtasks drain
    // before this zero-delay timer fires).
    await Future<void>.delayed(Duration.zero);
    sessionActive = false;
    stableText = '';
    provisionalText = '';
    _liveBuffer.clear();
    _liveSamples = 0;
    micLevel = 0;
    // An in-flight utterance left busy=true; the session is over, so clear
    // it here (late continuations keep it false — see _onUtterance).
    busy = false;
    status = 'Stopped. ${segments.length} utterance(s). '
        'Last decode ${asr.lastDecodeMs} ms.';
    notifyListeners();
  }

  int _pending = 0;

  Future<void> _onFrame(Float32List frame) async {
    if (!sessionActive) return;
    diar.pushFrame(frame);
    if (!lowLatency) return;
    _liveBuffer.add(frame);
    _liveSamples += frame.length;
    _maybePreview();
  }

  /// Throttled provisional re-decode of the in-progress utterance
  /// (low-latency mode only). Skipped while a final decode is running so
  /// previews never stack UI-blocking decode work.
  void _maybePreview() {
    if (_previewRunning || _pending > 0 || !sessionActive) return;
    final now = DateTime.now();
    if (now.difference(_lastPreview).inMilliseconds <
        AppConfig.livePreviewMinIntervalMs) {
      return;
    }
    if (_liveSamples <
        AppConfig.sampleRate * AppConfig.livePreviewMinAudioMs ~/ 1000) {
      return;
    }
    _previewRunning = true;
    _lastPreview = now;
    final pcm = Float32List(_liveSamples);
    var o = 0;
    for (final f in _liveBuffer) {
      pcm.setRange(o, o + f.length, f);
      o += f.length;
    }
    // transcribeUtterance usually throws asynchronously (handled by
    // catchError below), but guard the synchronous path too so a throw
    // can never escape into the audio stream listener.
    try {
      asr.transcribeUtterance(pcm).then((t) {
        if (!sessionActive) return;
        provisionalText = t.text;
        notifyListeners();
      }).catchError((Object e) {
        status = 'ASR preview error: $e';
        notifyListeners();
      }).whenComplete(() => _previewRunning = false);
    } catch (e) {
      _previewRunning = false;
      status = 'ASR preview error: $e';
      notifyListeners();
    }
  }

  /// Called by the diarizer each time an utterance closes: final decode,
  /// commit as a segment, translate. Same path in both modes.
  Future<void> _onUtterance(String speaker, Float32List pcm) async {
    _pending++;
    busy = true;
    notifyListeners();
    try {
      final t = await asr.transcribeUtterance(pcm);
      final text = t.text;
      // Utterance finished: drop the live preview state it was built from.
      _liveBuffer.clear();
      _liveSamples = 0;
      provisionalText = '';
      if (text.isEmpty) return;
      final seg = TranscriptSegment(
        id: _nextId++,
        speaker: diarizationEnabled ? speaker : 'Speaker 1',
        text: text,
        detectedLang: AsrService.appLangForResult(t.lang, sourceLang),
        startedAt: DateTime.now(),
        isTranslating: translationEnabled && mt.isLoaded,
      );
      segments.add(seg);
      // In low-latency mode the strip shows committed text as stable.
      if (lowLatency) {
        stableText = segments.length <= 3
            ? segments.map((s) => s.text).join(' ')
            : segments.sublist(segments.length - 3).map((s) => s.text).join(' ');
      }
      notifyListeners();
      if (translationEnabled && mt.isLoaded) {
        try {
          final tr = await mt.translate(
            text: text,
            source: sourceLang,
            target: targetLang,
          );
          seg.translation = tr;
          seg.translationTarget = targetLang;
        } catch (e) {
          seg.translation = '⚠ $e';
        } finally {
          seg.isTranslating = false;
          notifyListeners();
        }
      }
    } catch (e) {
      status = 'Transcription error: $e';
      notifyListeners();
    } finally {
      _pending--;
      if (_pending <= 0) {
        _pending = 0;
        busy = sessionActive ? false : busy;
        notifyListeners();
      }
    }
  }

  /// Retry translation for all segments (e.g. after target-lang change).
  Future<void> retranslateAll() async {
    if (!mt.isLoaded) return;
    for (final s in segments) {
      s.isTranslating = true;
      s.translationTarget = targetLang;
    }
    notifyListeners();
    for (final s in segments) {
      try {
        s.translation = await mt.translate(
          text: s.text,
          source: sourceLang,
          target: targetLang,
        );
      } catch (e) {
        s.translation = '⚠ $e';
      } finally {
        s.isTranslating = false;
        notifyListeners();
      }
    }
  }

  void renameSpeaker(String oldLabel, String newLabel) {
    diar.rename(oldLabel, newLabel);
    for (final s in segments) {
      if (s.speaker == oldLabel) s.speaker = newLabel;
    }
    notifyListeners();
  }

  void clear() {
    segments.clear();
    stableText = '';
    provisionalText = '';
    notifyListeners();
  }

  String exportText() => segments.map((s) => s.toExportLine()).join('\n\n');

  void setSource(AppLang v) {
    sourceLang = v;
    notifyListeners();
  }

  void setTarget(AppLang v) {
    targetLang = v;
    notifyListeners();
  }

  void setTranslationEnabled(bool v) {
    translationEnabled = v;
    notifyListeners();
  }

  void setDiarizationEnabled(bool v) {
    diarizationEnabled = v;
    notifyListeners();
  }

  void setLowLatency(bool v) {
    lowLatency = v;
    notifyListeners();
  }

  @override
  void dispose() {
    _audioSub?.cancel();
    audio.dispose();
    asr.dispose();
    mt.unload();
    models.dispose();
    super.dispose();
  }
}
