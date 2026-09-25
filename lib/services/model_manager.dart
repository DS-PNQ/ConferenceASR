import 'dart:async';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

import '../config/app_config.dart';

/// Thrown when the user cancels an in-progress download via [ModelManager.cancel].
class DownloadCancelled implements Exception {
  const DownloadCancelled();
  @override
  String toString() => 'Download cancelled';
}

/// Download + verify the two on-device models, with progress callbacks.
///
/// * ASR dir `<appdocs>/models/qwen3-asr-0.6b/` must end up containing
///   model.safetensors, vocab.json, merges.txt (from Qwen/Qwen3-ASR-0.6B).
/// * MT file `<appdocs>/models/Hy-MT2-1.8B-1.25Bit.gguf` (~462 MB).
///
/// Both are downloaded once and reused offline afterwards.
///
/// Robustness for multi-GB phone downloads: interrupted files resume from
/// the `.part` file via HTTP Range, transient network errors are retried
/// (3 attempts), a 60 s no-data stall aborts the attempt (then retries),
/// and [cancel] aborts promptly.
class ModelManager {
  ModelManager({http.Client? client}) : _client = client ?? http.Client();

  final http.Client _client;
  bool _cancelRequested = false;

  /// Ask an in-progress [downloadAsr]/[downloadMt] to abort ASAP.
  /// The partial `.part` file is kept so the next run resumes.
  void cancel() => _cancelRequested = true;

  Future<Directory> get _modelsDir async {
    final docs = await getApplicationDocumentsDirectory();
    final dir = Directory('${docs.path}/models');
    if (!await dir.exists()) await dir.create(recursive: true);
    return dir;
  }

  Future<String> get asrDir async =>
      '${(await _modelsDir).path}/qwen3-asr-0.6b';

  Future<String> get mtPath async =>
      '${(await _modelsDir).path}/${AppConfig.mtFileName}';

  /// Which ASR files are still missing?
  Future<List<String>> missingAsrFiles() async {
    final dir = await asrDir;
    final missing = <String>[];
    for (final f in AppConfig.asrRequiredFiles) {
      if (!await File('$dir/$f').exists()) missing.add(f);
    }
    return missing;
  }

  Future<bool> isAsrReady() async => (await missingAsrFiles()).isEmpty;

  Future<bool> isMtReady() async {
    final p = await mtPath;
    final f = File(p);
    if (!await f.exists()) return false;
    // Guard against truncated downloads (allow 5% slack).
    final len = await f.length();
    return len > AppConfig.mtApproxBytes * 0.95;
  }

  /// Combine whole-file + intra-file progress into one 0..1 fraction.
  /// [done]/[total] count finished files; [received]/[totalBytes] track the
  /// current file ([totalBytes] <= 0 means unknown → whole-file granularity).
  static double combineFileProgress(
      int done, int total, int received, int totalBytes,) {
    if (total <= 0) return 0;
    final frac =
        totalBytes > 0 ? (received / totalBytes).clamp(0.0, 1.0) : 0.0;
    return ((done + frac) / total).clamp(0.0, 1.0);
  }

  /// Download missing ASR files. [onProgress] gets (doneFiles, totalFiles,
  /// fileName, receivedBytes, totalBytes).
  Future<void> downloadAsr({
    void Function(int done, int total, String file, int received, int totalBytes)?
        onProgress,
  }) async {
    _cancelRequested = false;
    final dir = await asrDir;
    await Directory(dir).create(recursive: true);
    final missing = await missingAsrFiles();
    var done = AppConfig.asrRequiredFiles.length - missing.length;
    for (final name in missing) {
      final url = '${AppConfig.asrHfBase}/$name';
      await _withRetry(() => _downloadFile(
            url,
            '$dir/$name',
            onChunk: (r, t) => onProgress?.call(
                done, AppConfig.asrRequiredFiles.length, name, r, t,),
          ),);
      done++;
      onProgress?.call(done, AppConfig.asrRequiredFiles.length, name, 1, 1);
    }
  }

  Future<void> downloadMt({
    String? urlOverride,
    void Function(int received, int total)? onProgress,
  }) async {
    _cancelRequested = false;
    final p = await mtPath;
    await _withRetry(() => _downloadFile(
          urlOverride ?? AppConfig.mtHfUrl,
          p,
          onChunk: onProgress,
        ),);
  }

  /// Retry transient network failures (not user cancellation).
  Future<void> _withRetry(Future<void> Function() fn) async {
    const maxAttempts = 3;
    for (var attempt = 1;; attempt++) {
      try {
        await fn();
        return;
      } on DownloadCancelled {
        rethrow;
      } on IOException catch (_) {
        if (attempt >= maxAttempts) rethrow;
        await Future.delayed(Duration(seconds: attempt * 2));
      } on TimeoutException catch (_) {
        if (attempt >= maxAttempts) rethrow;
        await Future.delayed(Duration(seconds: attempt * 2));
      } on http.ClientException catch (_) {
        if (attempt >= maxAttempts) rethrow;
        await Future.delayed(Duration(seconds: attempt * 2));
      }
    }
  }

  Future<void> _downloadFile(
    String url,
    String dest, {
    void Function(int received, int total)? onChunk,
    Duration stallTimeout = const Duration(seconds: 60),
  }) async {
    final part = File('$dest.part');
    var start = 0;
    if (await part.exists()) start = await part.length();

    final req = http.Request('GET', Uri.parse(url));
    if (start > 0) req.headers['Range'] = 'bytes=$start-';
    final resp = await _client.send(req);

    // 206 = resumed append; 200 after a Range request = server ignored it,
    // restart from scratch; anything else is a hard failure.
    final resumed = resp.statusCode == 206 && start > 0;
    if (!resumed) {
      if (resp.statusCode != 200) {
        throw HttpException(
            'Download failed ($url): HTTP ${resp.statusCode}',
            uri: Uri.parse(url),);
      }
      start = 0;
    }
    final remaining = resp.contentLength ?? -1;
    final total = remaining < 0 ? -1 : start + remaining;

    final sink = part.openWrite(
        mode: resumed ? FileMode.append : FileMode.write,);
    var received = start;
    onChunk?.call(received, total);
    try {
      // .timeout fires if the server goes quiet for [stallTimeout] —
      // without it a dead connection would hang here forever.
      await for (final chunk in resp.stream.timeout(stallTimeout)) {
        if (_cancelRequested) throw const DownloadCancelled();
        sink.add(chunk);
        received += chunk.length;
        onChunk?.call(received, total);
      }
    } finally {
      await sink.close();
    }
    if (_cancelRequested) throw const DownloadCancelled();
    await part.rename(dest);
  }

  void dispose() => _client.close();
}
