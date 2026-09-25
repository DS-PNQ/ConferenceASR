import 'dart:math' as math;
import 'dart:typed_data';

/// Tiny speaker embedding: normalized log-mel-ish spectral fingerprint.
///
/// This is NOT a neural speaker embedding (no ECAPA / x-vector — those would
/// need an extra onnx model + onnxruntime dep). Instead we compute a compact
/// spectral profile per utterance (FFT magnitudes pooled into bands, mean +
/// variance, L2-normalized) and cluster with cosine distance.
///
/// In practice this separates 2–4 speakers in a quiet room reasonably well
/// and degrades gracefully (merges similar voices) in noise. If you need
/// production-grade diarization, swap [embed] with an ONNX x-vector model —
/// the rest of [DiarizationService] (threshold clustering, turn logic) stays
/// the same.
class SpeakerEmbedding {
  SpeakerEmbedding({this.bands = 24, this.fftSize = 512});

  final int bands;
  final int fftSize;

  /// Compute a unit-norm embedding from 16 kHz mono PCM floats.
  Float32List embed(Float32List pcm) {
    if (pcm.isEmpty) return Float32List(bands * 2);
    // Pre-emphasis + framing, magnitude spectrum averaged over frames.
    final frameLen = fftSize;
    final hop = fftSize ~/ 2;
    final bandSum = Float64List(bands);
    final bandSumSq = Float64List(bands);
    int frames = 0;

    final window = Float64List(frameLen);
    for (var i = 0; i < frameLen; i++) {
      window[i] = 0.54 - 0.46 * math.cos(2 * math.pi * i / (frameLen - 1));
    }

    double prev = 0;
    for (var start = 0; start + frameLen <= pcm.length; start += hop) {
      // DFT magnitude (naive O(n^2) on 512 pts, ~a few ms — fine here).
      // Only first half spectrum is used.
      final re = Float64List(fftSize ~/ 2);
      for (var k = 0; k < fftSize ~/ 2; k++) {
        double r = 0, ii = 0;
        for (var n = 0; n < frameLen; n++) {
          double s = pcm[start + n];
          s = s - 0.97 * prev; // pre-emphasis (approx per-sample chain)
          prev = pcm[start + n];
          final w = s * window[n];
          final ang = 2 * math.pi * k * n / frameLen;
          r += w * math.cos(ang);
          ii -= w * math.sin(ang);
        }
        re[k] = math.sqrt(r * r + ii * ii) / frameLen;
      }
      // Pool linear spectrum into log-spaced bands.
      for (var b = 0; b < bands; b++) {
        final lo = ((math.pow(2, b / 4.0) - 1) / 16.0 * (fftSize ~/ 2)).toInt();
        final hi = ((math.pow(2, (b + 1) / 4.0) - 1) / 16.0 * (fftSize ~/ 2)).toInt();
        final l = lo.clamp(0, fftSize ~/ 2 - 1);
        final h = hi.clamp(l + 1, fftSize ~/ 2);
        double m = 0;
        for (var k = l; k < h; k++) {
          m += re[k];
        }
        m /= (h - l);
        final logM = math.log(m + 1e-8);
        bandSum[b] += logM;
        bandSumSq[b] += logM * logM;
      }
      frames++;
    }
    if (frames == 0) return Float32List(bands * 2);

    final vec = Float32List(bands * 2);
    for (var b = 0; b < bands; b++) {
      final mean = bandSum[b] / frames;
      final variance = (bandSumSq[b] / frames - mean * mean).clamp(0.0, 1e9);
      vec[b] = mean;
      vec[bands + b] = math.sqrt(variance);
    }
    // Standardize + L2 normalize.
    double m = 0;
    for (final v in vec) {
      m += v;
    }
    m /= vec.length;
    double sd = 0;
    for (final v in vec) {
      sd += (v - m) * (v - m);
    }
    sd = math.sqrt(sd / vec.length + 1e-9);
    double norm = 0;
    for (var i = 0; i < vec.length; i++) {
      vec[i] = (vec[i] - m) / sd;
      norm += vec[i] * vec[i];
    }
    norm = math.sqrt(norm + 1e-9);
    for (var i = 0; i < vec.length; i++) {
      vec[i] /= norm;
    }
    return vec;
  }

  /// Cosine distance in [0, 2]. Lower = more similar.
  static double distance(Float32List a, Float32List b) {
    final n = math.min(a.length, b.length);
    double dot = 0;
    for (var i = 0; i < n; i++) {
      dot += a[i] * b[i];
    }
    return (1.0 - dot).clamp(0.0, 2.0);
  }
}
