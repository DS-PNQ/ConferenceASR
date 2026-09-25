import 'dart:math' as math;
import 'dart:typed_data';

import 'speaker_embedding.dart';
import 'vad.dart';

/// Online speaker diarization: VAD-gated utterance grouping + embedding
/// clustering.
///
/// Pipeline per 20 ms frame:
///   1. [Vad] decides speech vs silence.
///   2. Speech frames accumulate into the current utterance buffer.
///   3. A silence run longer than [silenceMs] closes the utterance.
///   4. The closed utterance is embedded ([SpeakerEmbedding]) and matched
///      against known speaker centroids (cosine distance < [threshold] →
///      same speaker, else new speaker, up to [maxSpeakers]).
///
/// The owning provider feeds mic frames via [pushFrame] and is notified of
/// speaker turns via [onUtteranceClosed] (utterance PCM + speaker label),
/// which it then sends to the ASR engine for transcription.
class DiarizationService {
  DiarizationService({
    Vad? vad,
    SpeakerEmbedding? embedding,
    this.silenceMs = 700,
    this.threshold = 0.32,
    this.maxSpeakers = 6,
    this.minUtteranceMs = 400,
    this.maxUtteranceSec = 20,
  })  : vad = vad ?? Vad(),
        embedding = embedding ?? SpeakerEmbedding();

  final Vad vad;
  final SpeakerEmbedding embedding;
  int silenceMs;
  double threshold;
  int maxSpeakers;
  final int minUtteranceMs;
  final int maxUtteranceSec;

  /// (speakerLabel, utterancePcm16kMono) — utterance ready for ASR.
  void Function(String speaker, Float32List utterance)? onUtteranceClosed;

  /// Live callback for UI level meter.
  void Function(double rms)? onLevel;

  final List<Float32List> _centroids = [];
  final List<String> _labels = [];
  final List<Float32List> _current = [];
  int _silenceFrames = 0;
  int _utteranceFrames = 0;

  int get speakerCount => _centroids.length;

  List<String> get speakers => List.unmodifiable(_labels);

  String _labelFor(int idx) => 'Speaker ${idx + 1}';

  /// Push one ~20 ms frame of 16 kHz mono float PCM.
  void pushFrame(Float32List frame) {
    final speech = vad.isSpeech(frame);
    onLevel?.call(Vad.rms(frame));

    if (speech) {
      _current.add(Float32List.fromList(frame));
      _silenceFrames = 0;
      _utteranceFrames++;
      // Safety cap: flush very long monologues so ASR stays responsive.
      final maxFrames = (maxUtteranceSec * 1000) ~/ vad.frameMs;
      if (_utteranceFrames >= maxFrames) {
        _closeUtterance();
      }
    } else {
      _silenceFrames++;
      final silenceRunMs = _silenceFrames * vad.frameMs;
      if (_current.isNotEmpty && silenceRunMs >= silenceMs) {
        _closeUtterance();
      }
    }
  }

  /// Force-close the in-progress utterance (e.g. on Stop).
  /// Returns true if something was flushed.
  bool flush() {
    if (_current.isEmpty) return false;
    _closeUtterance();
    return true;
  }

  void _closeUtterance() {
    final total = _current.fold<int>(0, (p, f) => p + f.length);
    final pcm = Float32List(total);
    var o = 0;
    for (final f in _current) {
      pcm.setRange(o, o + f.length, f);
      o += f.length;
    }
    _current.clear();
    _silenceFrames = 0;
    _utteranceFrames = 0;

    final durMs = (total / 16).round(); // 16 samples per ms @16kHz
    if (durMs < minUtteranceMs) return; // drop blips / clicks

    final speaker = _identify(pcm);
    onUtteranceClosed?.call(speaker, pcm);
  }

  String _identify(Float32List pcm) {
    final vec = embedding.embed(pcm);
    var best = -1;
    var bestDist = double.infinity;
    for (var i = 0; i < _centroids.length; i++) {
      final d = SpeakerEmbedding.distance(vec, _centroids[i]);
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    }
    if (best >= 0 && bestDist < threshold) {
      // Online centroid update (moving average, then re-normalize).
      final c = _centroids[best];
      for (var i = 0; i < c.length; i++) {
        c[i] = c[i] * 0.85 + vec[i] * 0.15;
      }
      double n = 0;
      for (final v in c) {
        n += v * v;
      }
      n = math.sqrt(n + 1e-9);
      for (var i = 0; i < c.length; i++) {
        c[i] /= n;
      }
      return _labels[best];
    }
    if (_centroids.length >= maxSpeakers) {
      // Full house: attach to nearest anyway (never silently drop speech).
      return best >= 0 ? _labels[best] : _labelFor(0);
    }
    _centroids.add(vec);
    final label = _labelFor(_centroids.length - 1);
    _labels.add(label);
    return label;
  }

  /// Rename a speaker everywhere (UI calls into provider, which rewrites
  /// past segments; this keeps future turns consistent).
  void rename(String oldLabel, String newLabel) {
    final i = _labels.indexOf(oldLabel);
    if (i >= 0) _labels[i] = newLabel;
  }

  void reset() {
    _centroids.clear();
    _labels.clear();
    _current.clear();
    _silenceFrames = 0;
    _utteranceFrames = 0;
    vad.reset();
  }
}
