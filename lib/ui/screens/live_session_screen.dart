import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:share_plus/share_plus.dart';

import '../../providers/session_provider.dart';
import '../widgets/transcript_widgets.dart';

/// The "text app" screen: big transcript list + live partial strip +
/// record controls + language selection.
class LiveSessionScreen extends StatefulWidget {
  const LiveSessionScreen({super.key, required this.onBack});

  final VoidCallback onBack;

  @override
  State<LiveSessionScreen> createState() => _LiveSessionScreenState();
}

class _LiveSessionScreenState extends State<LiveSessionScreen> {
  final _scroll = ScrollController();

  void _jumpToBottom() {
    if (!_scroll.hasClients) return;
    _scroll.animateTo(
      _scroll.position.maxScrollExtent,
      duration: const Duration(milliseconds: 250),
      curve: Curves.easeOut,
    );
  }

  @override
  Widget build(BuildContext context) {
    final s = context.watch<SessionProvider>();

    // Auto-scroll when new segments arrive.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (s.segments.isNotEmpty) _jumpToBottom();
    });

    return Scaffold(
      appBar: AppBar(
        leading: IconButton(icon: const Icon(Icons.arrow_back), onPressed: () {
          if (s.sessionActive) {
            ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                content: Text('Stop recording before leaving.'),),);
            return;
          }
          widget.onBack();
        },),
        title: Text(s.sessionActive ? '● Recording' : 'Conference'),
        actions: [
          IconButton(
            tooltip: 'Copy transcript',
            icon: const Icon(Icons.copy),
            onPressed: s.segments.isEmpty
                ? null
                : () {
                    Clipboard.setData(
                        ClipboardData(text: s.exportText()),);
                    ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(content: Text('Copied.')),);
                  },
          ),
          IconButton(
            tooltip: 'Share transcript',
            icon: const Icon(Icons.share),
            onPressed: s.segments.isEmpty
                ? null
                : () => Share.share(s.exportText(),
                    subject: 'Conference transcript',),
          ),
          IconButton(
            tooltip: 'Clear',
            icon: const Icon(Icons.delete_outline),
            onPressed: s.segments.isEmpty || s.sessionActive ? null : s.clear,
          ),
        ],
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 12, 12, 0),
            child: LanguageSelector(
              source: s.sourceLang,
              target: s.targetLang,
              onSource: (v) async {
                s.setSource(v);
                if (s.asr.isReady) {
                  try {
                    await s.asr.setSourceLang(v);
                  } catch (_) {}
                }
              },
              onTarget: (v) {
                s.setTarget(v);
                s.retranslateAll();
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            child: Row(
              children: [
                // Mic level meter.
                Expanded(
                  child: LinearProgressIndicator(
                    value: s.micLevel,
                    minHeight: 6,
                    backgroundColor: Colors.grey.shade300,
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  s.speakers.isEmpty
                      ? 'No speakers yet'
                      : '${s.speakers.length} speaker(s): ${s.speakers.join(', ')}',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ],
            ),
          ),
          if (s.lowLatency)
            LiveStrip(
                stable: s.stableText, provisional: s.provisionalText,),
          if (s.status.isNotEmpty)
            Padding(
              padding:
                  const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(s.status,
                    style: Theme.of(context)
                        .textTheme
                        .bodySmall
                        ?.copyWith(color: Colors.grey.shade700),),
              ),
            ),
          Expanded(
            child: s.segments.isEmpty
                ? Center(
                    child: Text(
                      s.sessionActive
                          ? 'Listening… speak now.'
                          : 'Press Record to start the conference capture.',
                      style: Theme.of(context).textTheme.bodyLarge,
                      textAlign: TextAlign.center,
                    ),
                  )
                : ListView.builder(
                    controller: _scroll,
                    itemCount: s.segments.length,
                    itemBuilder: (c, i) {
                      final seg = s.segments[i];
                      return TranscriptBubble(
                        segment: seg,
                        showTranslation: s.translationEnabled,
                        onRename: (n) =>
                            s.renameSpeaker(seg.speaker, n),
                      );
                    },
                  ),
          ),
          _ControlBar(onToggled: _jumpToBottom),
        ],
      ),
    );
  }
}

class _ControlBar extends StatelessWidget {
  const _ControlBar({required this.onToggled});

  final VoidCallback onToggled;

  @override
  Widget build(BuildContext context) {
    final s = context.watch<SessionProvider>();
    return SafeArea(
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: Theme.of(context).colorScheme.surface,
          boxShadow: const [
            BoxShadow(blurRadius: 8, color: Colors.black12),
          ],
        ),
        child: Row(
          children: [
            Expanded(
              child: FilledButton.icon(
                style: FilledButton.styleFrom(
                  backgroundColor:
                      s.sessionActive ? Colors.red : null,
                  padding:
                      const EdgeInsets.symmetric(vertical: 14),
                ),
                onPressed: s.busy && !s.sessionActive
                    ? null
                    : () async {
                        if (s.sessionActive) {
                          await s.stop();
                        } else {
                          await s.start();
                        }
                        onToggled();
                      },
                icon: Icon(
                    s.sessionActive ? Icons.stop : Icons.fiber_manual_record,),
                label: Text(s.sessionActive
                    ? 'Stop'
                    : (s.busy ? 'Starting…' : 'Record'),),
              ),
            ),
            const SizedBox(width: 8),
            IconButton.outlined(
              tooltip: 'Re-translate all',
              onPressed: (!s.translationEnabled ||
                      s.segments.isEmpty ||
                      !s.mt.isLoaded)
                  ? null
                  : s.retranslateAll,
              icon: const Icon(Icons.translate),
            ),
          ],
        ),
      ),
    );
  }
}
