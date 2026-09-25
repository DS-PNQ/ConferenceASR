import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';

import 'package:conference_asr_translator/config/app_config.dart';
import 'package:conference_asr_translator/services/asr_service.dart';
import 'package:conference_asr_translator/services/model_manager.dart';
import 'package:conference_asr_translator/services/translation_service.dart';
import 'package:conference_asr_translator/services/vad.dart';
import 'package:conference_asr_translator/models/transcript_segment.dart';

void main() {
  group('AppLang', () {
    test('hy-mt2 target names', () {
      expect(AppLang.vi.hyName, 'Vietnamese');
      expect(AppLang.zh.hyName, 'Chinese');
    });
  });

  group('Sherpa result language mapping', () {
    test('known codes map to vi/en/zh', () {
      expect(
        AsrService.appLangForResult('vi', AppLang.auto),
        AppLang.vi,
      );
      expect(
        AsrService.appLangForResult('en', AppLang.auto),
        AppLang.en,
      );
      expect(
        AsrService.appLangForResult('zh', AppLang.auto),
        AppLang.zh,
      );
      expect(
        AsrService.appLangForResult('yue', AppLang.auto),
        AppLang.zh,
      );
    });

    test('unknown codes fall back to the session source', () {
      expect(AsrService.appLangForResult('', AppLang.vi), AppLang.vi);
      expect(AsrService.appLangForResult('fr', AppLang.en), AppLang.en);
    });
  });

  group('Translation prompt', () {
    test('buildPrompt follows model-card template', () {
      final p = TranslationService.buildPrompt(
        text: 'Xin chào',
        source: AppLang.vi,
        target: AppLang.en,
      );
      expect(p, contains('Vietnamese'));
      expect(p, contains('English'));
      expect(p, contains('only output the translated result'));
      expect(p, contains('Xin chào'));
    });

    test('auto source omits source name', () {
      final p = TranslationService.buildPrompt(
        text: 'hello',
        source: AppLang.auto,
        target: AppLang.zh,
      );
      expect(p, contains('Chinese'));
      expect(p, isNot(contains('Auto text')));
    });
  });

  group('VAD', () {
    test('silence is not speech, loud sine is', () {
      final vad = Vad(energyThreshold: 0.008, adaptive: false);
      final silence = Float32List(320);
      expect(vad.isSpeech(silence), isFalse);
      // Realistic voiced signal: 200 Hz sine at 16 kHz (ZCR ~0.025).
      final tone = Float32List(320);
      for (var i = 0; i < tone.length; i++) {
        tone[i] = 0.5 * math.sin(2 * math.pi * 200 * i / 16000);
      }
      expect(vad.isSpeech(tone), isTrue);
    });
  });

  group('TranscriptSegment', () {
    test('export line contains speaker + translation', () {
      final s = TranscriptSegment(
        id: 1,
        speaker: 'Speaker 1',
        text: 'hello',
        detectedLang: AppLang.en,
        startedAt: DateTime(2026, 1, 1, 10, 0, 0),
        translation: 'xin chào',
      );
      expect(s.toExportLine(), contains('Speaker 1'));
      expect(s.toExportLine(), contains('xin chào'));
    });
  });

  group('Download progress', () {
    test('combineFileProgress blends file count + byte fraction', () {
      // Halfway through file 1 of 3 → 1/6.
      expect(ModelManager.combineFileProgress(0, 3, 50, 100),
          closeTo(1 / 6, 1e-9),);
      // 2 files done + current complete → 1.0.
      expect(ModelManager.combineFileProgress(2, 3, 100, 100), closeTo(1.0, 1e-9));
      // Unknown total → whole-file granularity only.
      expect(ModelManager.combineFileProgress(1, 3, 50, -1), closeTo(1 / 3, 1e-9));
      expect(ModelManager.combineFileProgress(0, 0, 0, 0), 0);
    });
  });
}
