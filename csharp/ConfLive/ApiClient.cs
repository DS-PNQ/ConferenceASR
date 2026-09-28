using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace ConfLive;

/// <summary>
/// REST (/api/*) plus the streaming /ws/live protocol.
/// Incoming server messages are raised as events; the UI marshals them.
/// </summary>
public sealed class ApiClient : IDisposable
{
    private readonly HttpClient _http;
    private readonly string _base;
    private ClientWebSocket _ws;
    private CancellationTokenSource _wsCts;
    private readonly SemaphoreSlim _sendLock = new(1, 1);
    private bool _disposed;

    public event Action<PartialEvent> PartialReceived;
    public event Action<UtteranceEvent> UtteranceReceived;
    public event Action<string> StatusReceived;
    public event Action<string> ErrorReceived;

    public ApiClient(string baseUrl)
    {
        _base = baseUrl.TrimEnd('/');
        _http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
    }

    public async Task<List<UtteranceEvent>> TranscribeFileAsync(
        string path, string[] targets, string srcLang, bool? denoise,
        CancellationToken ct = default)
    {
        using var form = new MultipartFormDataContent();
        form.Add(new ByteArrayContent(await File.ReadAllBytesAsync(path, ct)),
            "file", Path.GetFileName(path));
        form.Add(new StringContent(string.Join(",", targets)), "targets");
        form.Add(new StringContent(srcLang ?? "auto"), "src_lang");
        form.Add(new StringContent(denoise == null ? "auto" : denoise.Value ? "true" : "false"), "denoise");
        using var resp = await _http.PostAsync(_base + "/api/transcribe", form, ct);
        resp.EnsureSuccessStatusCode();
        var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync(ct));
        var list = new List<UtteranceEvent>();
        if (doc.RootElement.TryGetProperty("utterances", out var arr))
            foreach (var el in arr.EnumerateArray())
                list.Add(ParseUtterance(el));
        return list;
    }

    // ---- streaming websocket ----
    public async Task ConnectLiveAsync(CancellationToken ct = default)
    {
        await CloseLiveAsync();
        _wsCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        _ws = new ClientWebSocket();
        _ws.Options.KeepAliveInterval = TimeSpan.FromSeconds(20);
        var uri = new Uri(_base.Replace("http", "ws") + "/ws/live");
        await _ws.ConnectAsync(uri, _wsCts.Token);
        _ = Task.Run(ReceiveLoop);
    }

    public async Task SendStartAsync(string[] targets, string srcLang, bool? denoise)
    {
        var msg = new Dictionary<string, object>
        {
            ["type"] = "stream_start",
            ["targets"] = targets,
            ["src_lang"] = srcLang,
            ["denoise"] = denoise == null ? "auto" : denoise.Value ? "true" : "false",
        };
        await SendJsonAsync(msg);
    }

    public async Task SendAudioAsync(byte[] pcm16, int sampleRate = 16000)
    {
        var msg = new Dictionary<string, object>
        {
            ["type"] = "stream_audio",
            ["audio_b64"] = Convert.ToBase64String(pcm16),
            ["sr"] = sampleRate,
        };
        await SendJsonAsync(msg);
    }

    public async Task SendStopAsync()
    {
        await SendJsonAsync(new Dictionary<string, object> { ["type"] = "stream_stop" });
    }

    private async Task SendJsonAsync(Dictionary<string, object> msg)
    {
        if (_ws == null || _ws.State != WebSocketState.Open) return;
        await _sendLock.WaitAsync();
        try
        {
            var bytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(msg));
            await _ws.SendAsync(bytes, WebSocketMessageType.Text, true,
                _wsCts?.Token ?? CancellationToken.None);
        }
        finally { _sendLock.Release(); }
    }

    private async Task ReceiveLoop()
    {
        var buf = new byte[1 << 20];
        var sb = new StringBuilder();
        try
        {
            while (_ws != null && _ws.State == WebSocketState.Open)
            {
                sb.Clear();
                WebSocketReceiveResult r;
                do
                {
                    r = await _ws.ReceiveAsync(buf, _wsCts.Token);
                    if (r.MessageType == WebSocketMessageType.Close) return;
                    sb.Append(Encoding.UTF8.GetString(buf, 0, r.Count));
                } while (!r.EndOfMessage);
                Dispatch(sb.ToString());
            }
        }
        catch { /* socket closed */ }
    }

    private void Dispatch(string text)
    {
        try
        {
            using var doc = JsonDocument.Parse(text);
            var el = doc.RootElement;
            if (!el.TryGetProperty("ok", out var ok) || !ok.GetBoolean())
            {
                ErrorReceived?.Invoke(el.TryGetProperty("error", out var e)
                    ? e.GetString() : "backend error");
                return;
            }
            string type = el.TryGetProperty("type", out var t) ? t.GetString() : "";
            if (type == "partial") PartialReceived?.Invoke(ParsePartial(el));
            else if (type == "utterance") UtteranceReceived?.Invoke(ParseUtterance(el));
            else if (type == "stream_started") StatusReceived?.Invoke("● Listening — streaming ASR + translation.");
            else if (type == "stream_stopped")
                StatusReceived?.Invoke($"Stopped ({el.GetProperty("finalized").GetInt32()} segments).");
            else if (type == "silence") { /* meter hint, ignored */ }
        }
        catch (Exception ex) { ErrorReceived?.Invoke("Bad server message: " + ex.Message); }
    }

    private static PartialEvent ParsePartial(JsonElement el) => new()
    {
        Id = el.GetProperty("id").GetInt64(),
        Speaker = el.TryGetProperty("speaker", out var s) ? s.GetString() : "",
        RmsDb = el.TryGetProperty("rms_db", out var r) ? r.GetDouble() : 0,
        Text = el.TryGetProperty("text", out var t) ? t.GetString() : "",
        Translations = ParseTranslations(el),
    };

    private static UtteranceEvent ParseUtterance(JsonElement el) => new()
    {
        Id = el.TryGetProperty("id", out var i) ? i.GetInt64() : 0,
        Speaker = el.TryGetProperty("speaker", out var s) ? s.GetString() : "",
        RmsDb = el.TryGetProperty("rms_db", out var r) ? r.GetDouble() : 0,
        Denoised = el.TryGetProperty("denoised", out var d) && d.GetBoolean(),
        SrcLang = el.TryGetProperty("src_lang", out var l) ? l.GetString() : "",
        Text = el.TryGetProperty("text", out var t) ? t.GetString() : "",
        Translations = ParseTranslations(el),
        Start = el.TryGetProperty("start", out var st) ? st.GetDouble() : 0,
    };

    private static Dictionary<string, string> ParseTranslations(JsonElement el)
    {
        var d = new Dictionary<string, string>();
        if (el.TryGetProperty("translations", out var tr) && tr.ValueKind == JsonValueKind.Object)
            foreach (var kv in tr.EnumerateObject())
                d[kv.Name] = kv.Value.GetString() ?? "";
        return d;
    }

    public async Task CloseLiveAsync()
    {
        try { _wsCts?.Cancel(); } catch { }
        if (_ws != null)
        {
            try
            {
                if (_ws.State == WebSocketState.Open)
                    await _ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "bye",
                        CancellationToken.None);
            }
            catch { }
            _ws.Dispose();
            _ws = null;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        CloseLiveAsync().GetAwaiter().GetResult();
        _http.Dispose();
    }
}
