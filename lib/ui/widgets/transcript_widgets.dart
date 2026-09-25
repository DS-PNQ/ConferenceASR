import 'package:flutter/material.dart';

import '../../config/app_config.dart';
import '../../models/transcript_segment.dart';

const _speakerColors = [
  Color(0xFF1565C0),
  Color(0xFF6A1B9A),
  Color(0xFF00695C),
  Color(0xFFEF6C00),
  Color(0xFFC62828),
  Color(0xFF283593),
];

Color speakerColor(String speaker) {
  final m = RegExp(r'(\d+)').firstMatch(speaker);
  final i = m == null ? speaker.hashCode : (int.parse(m.group(1)!) - 1);
  return _speakerColors[i.abs() % _speakerColors.length];
}

/// Avatar text: speaker number if present, else first letter.
String _avatarLabel(String speaker) {
  final digits = speaker.replaceAll(RegExp(r'[^0-9]'), '');
  if (digits.isNotEmpty) return digits.length > 2 ? digits.substring(0, 2) : digits;
  final t = speaker.trim();
  return t.isEmpty ? '?' : t.substring(0, 1).toUpperCase();
}

/// One transcript row: speaker chip + source text + translation.
class TranscriptBubble extends StatelessWidget {
  const TranscriptBubble({
    super.key,
    required this.segment,
    required this.showTranslation,
    this.onRename,
  });

  final TranscriptSegment segment;
  final bool showTranslation;
  final void Function(String newName)? onRename;

  @override
  Widget build(BuildContext context) {
    final color = speakerColor(segment.speaker);
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                CircleAvatar(
                  radius: 14,
                  backgroundColor: color,
                  child: Text(
                    _avatarLabel(segment.speaker),
                    style: const TextStyle(color: Colors.white, fontSize: 12),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: GestureDetector(
                    onLongPress: () => _renameDialog(context),
                    child: Text(
                      segment.speaker,
                      style: TextStyle(
                          fontWeight: FontWeight.bold, color: color,),
                    ),
                  ),
                ),
                Text(
                  TimeOfDay.fromDateTime(segment.startedAt).format(context),
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ],
            ),
            const SizedBox(height: 8),
            SelectableText(segment.text,
                style: Theme.of(context).textTheme.bodyLarge,),
            if (showTranslation) ...[
              const Divider(height: 16),
              if (segment.isTranslating)
                const Row(
                  children: [
                    SizedBox(
                        width: 14,
                        height: 14,
                        child:
                            CircularProgressIndicator(strokeWidth: 2),),
                    SizedBox(width: 8),
                    Text('Translating…',
                        style: TextStyle(fontStyle: FontStyle.italic),),
                  ],
                )
              else if (segment.translation != null)
                SelectableText(
                  segment.translation!,
                  style: Theme.of(context)
                      .textTheme
                      .bodyMedium
                      ?.copyWith(color: Colors.black87),
                ),
            ],
          ],
        ),
      ),
    );
  }

  Future<void> _renameDialog(BuildContext context) async {
    final ctrl = TextEditingController(text: segment.speaker);
    final name = await showDialog<String>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('Rename speaker'),
        content: TextField(
          controller: ctrl,
          decoration:
              const InputDecoration(hintText: 'e.g. Linh, John, 主持人'),
          autofocus: true,
        ),
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(c), child: const Text('Cancel'),),
          FilledButton(
              onPressed: () => Navigator.pop(c, ctrl.text.trim()),
              child: const Text('Save'),),
        ],
      ),
    );
    if (name != null && name.isNotEmpty && name != segment.speaker) {
      onRename?.call(name);
    }
  }
}

/// Source/target language dropdown pair for vi/en/zh (+ auto on source).
class LanguageSelector extends StatelessWidget {
  const LanguageSelector({
    super.key,
    required this.source,
    required this.target,
    required this.onSource,
    required this.onTarget,
  });

  final AppLang source;
  final AppLang target;
  final ValueChanged<AppLang> onSource;
  final ValueChanged<AppLang> onTarget;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: DropdownButtonFormField<AppLang>(
            initialValue: source,
            decoration: const InputDecoration(
                labelText: 'Speak', border: OutlineInputBorder(),),
            items: AppLang.values
                .map((l) => DropdownMenuItem(value: l, child: Text(l.label)))
                .toList(),
            onChanged: (v) {
              if (v != null) onSource(v);
            },
          ),
        ),
        const Padding(
          padding: EdgeInsets.symmetric(horizontal: 8),
          child: Icon(Icons.arrow_forward),
        ),
        Expanded(
          child: DropdownButtonFormField<AppLang>(
            initialValue: target,
            decoration: const InputDecoration(
                labelText: 'Translate to', border: OutlineInputBorder(),),
            items: [AppLang.vi, AppLang.en, AppLang.zh]
                .map((l) => DropdownMenuItem(value: l, child: Text(l.label)))
                .toList(),
            onChanged: (v) {
              if (v != null) onTarget(v);
            },
          ),
        ),
      ],
    );
  }
}

/// Live partial-text strip: stable in normal weight, provisional grey italic
/// (qwen_asr semantics — the tail may still change).
class LiveStrip extends StatelessWidget {
  const LiveStrip(
      {super.key, required this.stable, required this.provisional,});

  final String stable;
  final String provisional;

  @override
  Widget build(BuildContext context) {
    if (stable.isEmpty && provisional.isEmpty) return const SizedBox.shrink();
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.all(12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
      ),
      child: RichText(
        text: TextSpan(
          style: Theme.of(context).textTheme.bodyMedium,
          children: [
            TextSpan(text: stable),
            if (provisional.isNotEmpty)
              TextSpan(
                text: ' $provisional',
                style: const TextStyle(
                    fontStyle: FontStyle.italic, color: Colors.grey,),
              ),
            const TextSpan(text: ' ▍'),
          ],
        ),
      ),
    );
  }
}
