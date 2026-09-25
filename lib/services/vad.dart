import 'dart:math' as math;
import 'dart:typed_data';

/// Lightweight energy + zero-crossing VAD over 16 kHz mono float PCM.
///
/// No model, no dependency — good enough to find utterance boundaries and
/// silence gaps that drive segmentation + speaker-turn detection. For noisy
/// conference rooms, raise [energyThreshold] or enable [adaptive] mode.
class Vad {
  Vad({
    this.sampleRate = 16000,
    this.frameMs = 20,
    this.energyThreshold = 0.008,
    this.zcrThreshold = 0.35,
    this.adaptive = true,
    this.hangoverFrames = 6,
  });

  final int sampleRate;
  final int frameMs;
  double energyThreshold;
  final double zcrThreshold;
  final bool adaptive;
  final int hangoverFrames;

  double _noiseFloor = 0.002;
  int _hangover = 0;
  bool _inSpeech = false;

  int get frameSamples => (sampleRate * frameMs) ~/ 1000;

  /// Classify one frame. Returns true if speech.
  bool isSpeech(Float32List frame) {
    if (frame.isEmpty) return _inSpeech;

    double energy = 0;
    int zc = 0;
    double prev = frame[0];
    for (var i = 0; i < frame.length; i++) {
      final s = frame[i];
      energy += s * s;
      if ((prev >= 0) != (s >= 0)) zc++;
      prev = s;
    }
    energy /= frame.length;
    final zcr = zc / frame.length;

    if (adaptive) {
      // Slowly track the noise floor during non-speech.
      if (energy < energyThreshold) {
        _noiseFloor = 0.95 * _noiseFloor + 0.05 * energy;
      }
    }
    final dynThreshold =
        adaptive ? math.max(energyThreshold, _noiseFloor * 4.0) : energyThreshold;

    final speechLike = energy > dynThreshold && zcr < zcrThreshold;

    if (speechLike) {
      _hangover = hangoverFrames;
      _inSpeech = true;
    } else if (_hangover > 0) {
      _hangover--;
      _inSpeech = true; // hangover: bridge short intra-word gaps
    } else {
      _inSpeech = false;
    }
    return _inSpeech;
  }

  /// Frame-level RMS, useful for the mic level meter.
  static double rms(Float32List frame) {
    if (frame.isEmpty) return 0;
    double e = 0;
    for (final s in frame) {
      e += s * s;
    }
    return math.sqrt(e / frame.length);
  }

  void reset() {
    _hangover = 0;
    _inSpeech = false;
  }
}
