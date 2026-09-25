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
/// Pipeline (utterance mode, default — best for conferences):
///   mic 16 kHz PCM → [DiarizationService] (VAD + turn split + speaker id)
///   → closed utterance → [AsrService.transcribeUtterance]
///   → [TranscriptSegment] appended → [TranslationService.translate] (async)
///   → segment updated in place.
///
/// Pipeline (low-latency streaming mode):
///   mic frames → [AsrService.pushStreamChunk] → stable + provisional text
///   rendered live; on Stop (finalize) the stable text is committed as one
///   segment with the diarizer's current majority speaker.
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
  String _streamStable = '';
  String _lastSpeaker = 'Speaker 1';

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
  Future<void> ensureEngines() async {
    final needMt = translationEnabled && !mt.isLoaded;
    if (!asr.isReady || needMt) {
      // Preflight: both models resident need ~3 GB. Without this check the
      // OS low-memory killer just terminates the app with no error.
      final free = await mt.freeRamBytes();
      final need = AppConfig.asrNeedFreeBytes +
          (needMt ? AppConfig.mtNeedFreeBytes : 0);
      if (free > 0 && free < need) {
        throw StateError(
          'Only ${(free / 1048576).round()} MB RAM free, but ~${(need / 1073741824).toStringAsFixed(1)} GB is needed '
          '(ASR ~2.2 GB${needMt ? ' + translation ~0.9 GB' : ''}). '
          'Fix: turn off translation, close other apps, or use a device '
          'with more RAM (8 GB+ recommended).',
        );
      }
    }
    if (!asr.isReady) {
      status = 'Loading ASR engine…';
      notifyListeners();
      await asr.load(await models.asrDir, lang: sourceLang);
    } else {
      await asr.setSourceLang(sourceLang);
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
      _streamStable = '';
      await asr.resetStream();
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
    if (lowLatency) {
      // Flush streaming tail and commit (drain in-flight chunks first so
      // the finalize push sees the full history, in order).
      try {
        await _pushQueue;
        final tail = await asr.pushStreamChunk(
          Float32List(0),
          finalize: true,
        );
        _streamStable = tail.stable;
        _commitStreamTail();
      } catch (_) {}
    } else {
      diar.flush();
      // Give in-flight utterance transcriptions a moment is handled by
      // _pending counter below — here we just mark stopping.
    }
    sessionActive = false;
    stableText = '';
    provisionalText = '';
    micLevel = 0;
    status = 'Stopped. ${segments.length} utterance(s). ${asr.perfStats()}';
    notifyListeners();
  }

  int _pending = 0;
  Future<void> _pushQueue = Future.value();

  Future<void> _onFrame(Float32List frame) async {
    if (!sessionActive) return;
    if (lowLatency) {
      // Chain pushes so chunks reach the engine in order and never overlap;
      // listen() itself doesn't await us, so without this, concurrent
      // streamPush calls would interleave out of order.
      _pushQueue = _pushQueue.then((_) => _pushOne(frame));
      await _pushQueue;
    } else {
      diar.pushFrame(frame);
    }
  }

  Future<void> _pushOne(Float32List frame) async {
    if (!sessionActive) return;
    // Streaming ASR path + parallel diarization for speaker turns.
    diar.pushFrame(frame);
    try {
      final p = await asr.pushStreamChunk(frame);
      _streamStable = p.stable;
      stableText = p.stable;
      provisionalText = p.provisional;
      notifyListeners();
    } catch (e) {
      status = 'ASR error: $e';
      notifyListeners();
    }
  }

  /// Called by the diarizer each time an utterance closes.
  Future<void> _onUtterance(String speaker, Float32List pcm) async {
    if (lowLatency) {
      _lastSpeaker = speaker; // used at commit time
      return; // streaming path commits once at Stop / turn flush
    }
    _pending++;
    busy = true;
    notifyListeners();
    try {
      final text = (await asr.transcribeUtterance(pcm)).trim();
      if (text.isEmpty) return;
      final seg = TranscriptSegment(
        id: _nextId++,
        speaker: diarizationEnabled ? speaker : 'Speaker 1',
        text: text,
        detectedLang: sourceLang,
        startedAt: DateTime.now(),
        isTranslating: translationEnabled && mt.isLoaded,
      );
      segments.add(seg);
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

  void _commitStreamTail() {
    final text = _streamStable.trim();
    if (text.isEmpty) return;
    final seg = TranscriptSegment(
      id: _nextId++,
      speaker: diarizationEnabled ? _lastSpeaker : 'Speaker 1',
      text: text,
      detectedLang: sourceLang,
      startedAt: DateTime.now(),
      isTranslating: translationEnabled && mt.isLoaded,
    );
    segments.add(seg);
    if (translationEnabled && mt.isLoaded) {
      mt
          .translate(text: text, source: sourceLang, target: targetLang)
          .then((tr) {
        seg.translation = tr;
        seg.translationTarget = targetLang;
        seg.isTranslating = false;
        notifyListeners();
      }).catchError((Object e) {
        seg.translation = '⚠ $e';
        seg.isTranslating = false;
        notifyListeners();
      });
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
