import '../config/app_config.dart';

/// One finalized utterance shown in the transcript list.
class TranscriptSegment {
  TranscriptSegment({
    required this.id,
    required this.speaker,
    required this.text,
    required this.detectedLang,
    required this.startedAt,
    this.translation,
    this.translationTarget,
    this.isTranslating = false,
  });

  final int id;
  String speaker; // e.g. "Speaker 1" (renamable)
  String text; // ASR output, source language
  AppLang detectedLang; // best-effort (currently = session source setting)
  DateTime startedAt;
  String? translation; // translated text
  AppLang? translationTarget;
  bool isTranslating;

  TranscriptSegment copyWith({
    String? speaker,
    String? text,
    String? translation,
    AppLang? translationTarget,
    bool? isTranslating,
  }) {
    return TranscriptSegment(
      id: id,
      speaker: speaker ?? this.speaker,
      text: text ?? this.text,
      detectedLang: detectedLang,
      startedAt: startedAt,
      translation: translation ?? this.translation,
      translationTarget: translationTarget ?? this.translationTarget,
      isTranslating: isTranslating ?? this.isTranslating,
    );
  }

  /// Export line: "[00:12] Speaker 1 (EN): hello\n>>> (VI): xin chào"
  String toExportLine() {
    final t = startedAt;
    final ts =
        '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}:${t.second.toString().padLeft(2, '0')}';
    final buf = StringBuffer('[$ts] $speaker: $text');
    if (translation != null && translation!.isNotEmpty) {
      buf.write('\n>>> $translation');
    }
    return buf.toString();
  }
}
