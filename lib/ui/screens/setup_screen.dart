import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/session_provider.dart';

/// First-run screen: model downloads, engine warm-up, session settings.
class SetupScreen extends StatefulWidget {
  const SetupScreen({super.key, required this.onReady});

  final VoidCallback onReady;

  @override
  State<SetupScreen> createState() => _SetupScreenState();
}

class _SetupScreenState extends State<SetupScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<SessionProvider>().refreshModelStatus();
    });
  }

  @override
  Widget build(BuildContext context) {
    final s = context.watch<SessionProvider>();
    return Scaffold(
      appBar: AppBar(title: const Text('Conference Translator — Setup')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text(
            'On-device ASR (sherpa-onnx Qwen3-ASR) + translation (Hy-MT2 1.8B) for '
            'Tiếng Việt · English · 中文. Everything runs offline after the '
            'one-time download below.',
          ),
          const SizedBox(height: 16),
          Card(
            child: ListTile(
              leading: Icon(
                s.asrReady ? Icons.check_circle : Icons.download,
                color: s.asrReady ? Colors.green : null,
              ),
              title: const Text('ASR model — Qwen3-ASR-0.6B int8 (~0.94 GB)'),
              subtitle: Text(
                s.asrReady
                    ? 'Ready'
                    : 'encoder + decoder + conv frontend + tokenizer/'
                        ' (auto-detects vi/en/zh)',
              ),
            ),
          ),
          Card(
            child: ListTile(
              leading: Icon(
                s.mtReady ? Icons.check_circle : Icons.download,
                color: s.mtReady ? Colors.green : null,
              ),
              title: const Text('MT model — Hy-MT2-1.8B Q4 (~1.13 GB)'),
              subtitle: Text(s.mtReady
                  ? 'Ready'
                  : 'Hy-MT2-1.8B-Q4_K_M.gguf from Hugging Face',),
            ),
          ),
          const SizedBox(height: 8),
          Text(s.modelStatus),
          if (s.modelsLoading) ...[
            if (s.downloadDetail.isNotEmpty) ...[
              const SizedBox(height: 4),
              Text(
                s.downloadDetail,
                style: Theme.of(context)
                    .textTheme
                    .bodyMedium
                    ?.copyWith(fontWeight: FontWeight.bold),
              ),
              const Text(
                'Large files — keep the app open and on Wi-Fi. '
                'Interrupted downloads resume automatically.',
                style: TextStyle(color: Colors.grey),
              ),
            ],
            const SizedBox(height: 8),
            LinearProgressIndicator(
                value: s.asrReady
                    ? (s.mtProgress == 0 ? null : s.mtProgress)
                    : (s.asrProgress == 0 ? null : s.asrProgress),),
          ],
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: s.modelsLoading
                      ? null
                      : () async {
                          await s.downloadModels();
                          if (context.mounted) {
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(content: Text(s.modelStatus)),
                            );
                          }
                        },
                  icon: const Icon(Icons.download),
                  label: const Text('Download models'),
                ),
              ),
              if (s.modelsLoading) ...[
                const SizedBox(width: 8),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: s.cancelDownload,
                    icon: const Icon(Icons.close),
                    label: const Text('Cancel'),
                  ),
                ),
              ],
            ],
          ),
          const Divider(height: 32),
          SwitchListTile(
            title: const Text('Enable translation'),
            subtitle: const Text('Requires the Hy-MT2 GGUF model'),
            value: s.translationEnabled,
            onChanged: s.setTranslationEnabled,
          ),
          SwitchListTile(
            title: const Text('Speaker diarization'),
            subtitle: const Text('VAD + on-device speaker clustering'),
            value: s.diarizationEnabled,
            onChanged: s.setDiarizationEnabled,
          ),
          SwitchListTile(
            title: const Text('Low-latency live preview'),
            subtitle: const Text(
              'Re-decodes the in-progress utterance every few seconds; '
              'utterance mode is steadier and cheaper',
            ),
            value: s.lowLatency,
            onChanged: s.setLowLatency,
          ),
          const SizedBox(height: 16),
          FilledButton.icon(
            onPressed: !s.asrReady || s.busy
                ? null
                : () async {
                    try {
                      await s.ensureEngines();
                      widget.onReady();
                    } catch (e) {
                      if (context.mounted) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(content: Text('Engine failed: $e')),
                        );
                      }
                    }
                  },
            icon: s.busy
                ? const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(strokeWidth: 2),)
                : const Icon(Icons.mic),
            label: Text(s.busy ? s.status : 'Start conference session'),
          ),
            if (!s.asrReady)
              const Padding(
                padding: EdgeInsets.only(top: 8),
                child: Text(
                  'Download the ASR model first. (~0.94 GB — use Wi-Fi. '
                  'Interrupted downloads resume automatically.)',
                  style: TextStyle(color: Colors.orange),
                ),
              ),
        ],
      ),
    );
  }
}
