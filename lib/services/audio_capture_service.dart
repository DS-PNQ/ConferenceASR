import 'dart:async';
import 'dart:typed_data';

import 'package:record/record.dart';

import '../config/app_config.dart';

/// Microphone capture as 16 kHz mono float PCM frames.
///
/// Uses the `record` package's `startStream(pcm16bits)` and converts
/// Int16 LE bytes → Float32List in [-1, 1] (the format qwen_asr expects).
/// Emits fixed ~20 ms frames (320 samples) so VAD/diarization get a steady
/// cadence regardless of the platform's native callback chunking.
class AudioCaptureService {
  AudioCaptureService({AudioRecorder? recorder})
      : _recorder = recorder ?? AudioRecorder();

  final AudioRecorder _recorder;
  StreamSubscription<Uint8List>? _sub;
  final StreamController<Float32List> _frames = StreamController.broadcast();

  /// 20 ms frames of 16 kHz mono float PCM.
  Stream<Float32List> get frames => _frames.stream;

  bool get isRecording => _sub != null;

  final List<double> _carry = [];

  Future<bool> hasPermission() => _recorder.hasPermission();

  Future<void> start() async {
    if (isRecording) return;
    final ok = await _recorder.hasPermission();
    if (!ok) {
      throw StateError(
          'Microphone permission denied. Grant it in system settings.',);
    }
    const config = RecordConfig(
      encoder: AudioEncoder.pcm16bits,
      sampleRate: AppConfig.sampleRate,
      numChannels: AppConfig.channels,
      autoGain: true,
      echoCancel: true,
      noiseSuppress: true,
    );
    final stream = await _recorder.startStream(config);
    _sub = stream.listen(
      _onBytes,
      onError: (Object e) => _frames.addError(e),
      cancelOnError: false,
    );
  }

  void _onBytes(Uint8List bytes) {
    final n = bytes.length ~/ 2;
    final bd = ByteData.sublistView(bytes);
    for (var i = 0; i < n; i++) {
      _carry.add(bd.getInt16(i * 2, Endian.little) / 32768.0);
    }
    // Slice into 320-sample (20 ms) frames.
    const frameLen = 320;
    while (_carry.length >= frameLen) {
      final f = Float32List(frameLen);
      for (var i = 0; i < frameLen; i++) {
        f[i] = _carry[i];
      }
      _carry.removeRange(0, frameLen);
      if (!_frames.isClosed) _frames.add(f);
    }
  }

  Future<void> stop() async {
    await _sub?.cancel();
    _sub = null;
    _carry.clear();
    try {
      await _recorder.stop();
    } catch (_) {
      // Not recording to file — stop() may throw; safe to ignore.
    }
  }

  Future<void> dispose() async {
    await stop();
    await _frames.close();
    _recorder.dispose();
  }
}
