using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Effects;
using Microsoft.Win32;

namespace ConfLive;

public partial class MainWindow : Window
{
    private static readonly string[] SpeakerColors =
        { "#7C3AED", "#0EA5E9", "#F59E0B", "#10B981", "#EC4899" };

    private readonly BackendManager _backend = new();
    private ApiClient _api;
    private readonly MicCapture _mic = new();
    private readonly List<UtteranceEvent> _history = new();
    private readonly Dictionary<long, PartialBubble> _partials = new();
    private readonly Dictionary<long, PartialBubble> _finals = new();
    private readonly SemaphoreSlim _sendLock = new(1, 1);
    private int _count;
    private bool _recording;
    private CancellationTokenSource _initCts = new();

    private sealed class PartialBubble
    {
        public Border Frame; public TextBlock Who; public TextBlock Orig;
        public StackPanel Body; public readonly List<TextBlock> Trs = new();
        public TextBlock Note; // "translating…" marker, finals only
    }

    public MainWindow()
    {
        InitializeComponent();
        Loaded += async (_, _) => await InitAsync();
        Closing += (_, _) => Shutdown();
    }

    // ---------- startup ----------
    private async Task InitAsync()
    {
        BackendManager.Log("ui: init start");
        ShowEmpty("Press  ● Record  or  ⇪ Upload —\nutterances appear here as speaker bubbles.");
        SetStatus("Starting backend…");
        try
        {
            var health = await _backend.EnsureRunningAsync(SetStatus, _initCts.Token);
            _api = new ApiClient(_backend.BaseUrl);
            _api.PartialReceived += p => Dispatcher.Invoke(() => UpsertPartial(p));
            _api.UtteranceReceived += u => Dispatcher.Invoke(() => AddFinal(u));
            _api.StatusReceived += s => Dispatcher.Invoke(() => SetStatus(s));
            _api.ErrorReceived += e => Dispatcher.Invoke(() => SetStatus("Error: " + e));
            ShowHealth(health);
            foreach (var (i, name) in MicCapture.ListInputs())
                MicBox.Items.Add($"{i}: {name}");
            if (MicBox.Items.Count > 0) MicBox.SelectedIndex = 0;
            _ = WarmupAsync(); // background; UI stays responsive
            BackendManager.Log("ui: init done");
        }
        catch (Exception ex)
        {
            BackendManager.Log("ui: init FAILED: " + ex.GetType().Name + ": " + ex.Message);
            SetStatus("Startup failed: " + ex.Message);
        }
    }

    private void ShowHealth(HealthInfo h)
    {
        string gpu = string.IsNullOrEmpty(h.Device.GpuName) ? "" : " · " + h.Device.GpuName.Split('(')[0].Trim();
        string fb = _backend.PreferredDevice == "cuda" && h.Device.Device == "cpu" ? " (CPU fallback active)" : "";
        DeviceBadge.Text = $"device: {h.Device.Device}{gpu}{fb}";
        ModelText.Text = $"ASR {h.Asr.Model}\nMT {h.Mt.Model}";
        SetStatus($"Ready — ASR {h.Asr.Model} · MT {h.Mt.Model}. Press Record.");
    }

    private async Task WarmupAsync()
    {
        try
        {
            SetStatus("Loading models (first run downloads weights)…");
            await _backend.WarmupAsync();
            ShowHealth(await _backend.GetHealthAsync());
        }
        catch (Exception ex) { SetStatus("Warmup failed: " + ex.Message); }
    }

    private string[] Targets()
    {
        var t = new List<string>();
        if (CbEn.IsChecked == true) t.Add("en");
        if (CbZh.IsChecked == true) t.Add("zh");
        if (CbVi.IsChecked == true) t.Add("vi");
        return t.Count > 0 ? t.ToArray() : new[] { "en" };
    }

    private string SrcLang() => SrcBox.SelectedIndex switch
    {
        1 => "Vietnamese", 2 => "English", 3 => "Chinese", _ => null,
    };

    // ---------- record / stop ----------
    private async void OnRecord(object sender, RoutedEventArgs e)
    {
        if (_api == null) { SetStatus("Backend not ready."); return; }
        try
        {
            await _api.ConnectLiveAsync();
            await _api.SendStartAsync(Targets(), SrcLang(), DenoiseBox.IsChecked);
            int dev = -1;
            if (MicBox.SelectedItem is string s && s.Contains(':')
                && int.TryParse(s.Split(':')[0], out int idx)) dev = idx;
            _mic.FrameReady += OnFrame;
            _mic.LevelChanged += OnLevel;
            _mic.Start(dev);
            _recording = true;
            BtnRecord.IsEnabled = false; BtnStop.IsEnabled = true;
            LiveText.Visibility = Visibility.Visible;
            SetStatus("● Recording — speak in vi / en / zh.");
        }
        catch (Exception ex) { SetStatus("Could not start: " + ex.Message); }
    }

    private async void OnStop(object sender, RoutedEventArgs e)
    {
        _recording = false;
        try
        {
            _mic.FrameReady -= OnFrame;
            _mic.LevelChanged -= OnLevel;
            _mic.Stop();
            if (_api != null) { await _api.SendStopAsync(); await _api.CloseLiveAsync(); }
        }
        catch { }
        BtnRecord.IsEnabled = true; BtnStop.IsEnabled = false;
        LiveText.Visibility = Visibility.Collapsed;
        Meter.Value = 0; MeterText.Text = "mic idle";
        SetStatus("Stopped.");
    }

    private async void OnFrame(byte[] frame)
    {
        if (!_recording || _api == null) return;
        await _sendLock.WaitAsync();
        try { await _api.SendAudioAsync(frame); }
        catch { }
        finally { _sendLock.Release(); }
    }

    private void OnLevel(double v) =>
        Dispatcher.Invoke(() =>
        {
            Meter.Value = v;
            MeterText.Text = !_recording ? "mic idle" : v > 0.02 ? "listening…" : "quiet…";
        });

    // ---------- upload / export ----------
    private async void OnUpload(object sender, RoutedEventArgs e)
    {
        if (_api == null) return;
        var dlg = new OpenFileDialog
        {
            Title = "Upload recording",
            Filter = "Audio|*.wav;*.mp3;*.ogg;*.flac;*.m4a;*.webm|All files|*.*",
        };
        if (dlg.ShowDialog() != true) return;
        SetStatus($"Transcribing {Path.GetFileName(dlg.FileName)}…");
        try
        {
            var list = await _api.TranscribeFileAsync(dlg.FileName, Targets(), SrcLang(), DenoiseBox.IsChecked);
            foreach (var u in list) AddFinal(u);
            SetStatus($"Done — {list.Count} utterances.");
        }
        catch (Exception ex) { SetStatus("Upload failed: " + ex.Message); }
    }

    private void OnExportTxt(object s, RoutedEventArgs e) => Export(".txt");
    private void OnExportSrt(object s, RoutedEventArgs e) => Export(".srt");

    private void Export(string ext)
    {
        if (_history.Count == 0) { SetStatus("Nothing to export yet."); return; }
        var dlg = new SaveFileDialog { DefaultExt = ext, Filter = $"{ext}|*{ext}" };
        if (dlg.ShowDialog() != true) return;
        var lines = new List<string>();
        if (ext == ".txt")
            foreach (var u in _history)
                lines.Add($"{u.Speaker} ({u.SrcLang}): {u.Text}\n  -> " +
                    string.Join(" | ", u.Translations.Select(kv => $"[{kv.Key}] {kv.Value}")));
        else
        {
            int i = 0;
            foreach (var u in _history)
            {
                i++;
                lines.Add($"{i}\n{Ts(i * 5)} --> {Ts(i * 5 + 4)}\n{u.Speaker}: {u.Text}\n" +
                    string.Join(" ", u.Translations.Values) + "\n");
            }
        }
        File.WriteAllLines(dlg.FileName, lines);
        SetStatus("Exported " + dlg.FileName);
    }

    private static string Ts(int s) => $"{s / 3600:00}:{(s % 3600) / 60:00}:{s % 60:00},000";

    private async void OnWarmup(object s, RoutedEventArgs e) => await WarmupAsync();

    private void OnClear(object s, RoutedEventArgs e)
    {
        _partials.Clear();
        _finals.Clear();
        Feed.Children.Clear();
        _history.Clear();
        _count = 0;
        CountText.Text = "0 utterances";
        ShowEmpty("Cleared — press  ● Record  for a new session.");
    }

    // ---------- feed ----------
    private static string ColorFor(string speaker)
    {
        int n = 0;
        foreach (char c in speaker ?? "")
            if (char.IsDigit(c)) n = n * 10 + (c - '0');
        return SpeakerColors[n <= 0 ? 0 : (n - 1) % SpeakerColors.Length];
    }

    private static Color Hex(string h) => (Color)ColorConverter.ConvertFromString(h);

    private Border MakeBubble(string speaker, string meta,
        out StackPanel body, out TextBlock orig, out TextBlock who, bool animate = false)
    {
        if (Feed.Children.Count == 1 && Feed.Children[0] is TextBlock tb && tb.Tag as string == "empty")
            Feed.Children.Clear();
        // System-like card: white fill, no outline, depth from a whisper of shadow.
        var bubble = new Border
        {
            Background = new SolidColorBrush(Hex("#FFFFFF")),
            BorderThickness = new Thickness(0), CornerRadius = new CornerRadius(14),
            Margin = new Thickness(10, 6, 10, 6),
            Effect = new DropShadowEffect
            {
                BlurRadius = 10, ShadowDepth = 1, Opacity = 0.12, Color = Colors.Black,
            },
        };
        body = new StackPanel { Margin = new Thickness(14, 12, 14, 12) };
        who = new TextBlock
        {
            Text = meta, FontWeight = FontWeights.SemiBold, FontSize = 12,
            Foreground = new SolidColorBrush(Hex(ColorFor(speaker))),
        };
        body.Children.Add(who);
        orig = new TextBlock
        {
            FontSize = 15, Foreground = new SolidColorBrush(Hex("#1C1917")),
            TextWrapping = TextWrapping.Wrap, Margin = new Thickness(0, 2, 0, 0),
        };
        body.Children.Add(orig);
        bubble.Child = body;
        Feed.Children.Add(bubble);
        FeedScroll.ScrollToBottom();
        if (animate)
        {
            // Calm fade-in: explains arrival, decorates nothing.
            var anim = new DoubleAnimation(0, 1, new Duration(TimeSpan.FromMilliseconds(250)))
            {
                EasingFunction = new QuadraticEase { EasingMode = EasingMode.EaseOut },
            };
            bubble.BeginAnimation(UIElement.OpacityProperty, anim);
        }
        return bubble;
    }

    private static TextBlock MakeTr(StackPanel body, string text)
    {
        var lb = new TextBlock
        {
            Text = text, FontSize = 13, Foreground = new SolidColorBrush(Hex("#44403C")),
            Background = new SolidColorBrush(Hex("#F5F5F4")),
            TextWrapping = TextWrapping.Wrap, Margin = new Thickness(0, 6, 0, 0),
            Padding = new Thickness(6),
        };
        body.Children.Add(lb);
        return lb;
    }

    private void ShowEmpty(string text) =>
        Feed.Children.Add(new TextBlock
        {
            Tag = "empty", Text = text, FontSize = 14,
            Foreground = new SolidColorBrush(Hex("#78716C")),
            HorizontalAlignment = HorizontalAlignment.Center,
            TextAlignment = TextAlignment.Center, Margin = new Thickness(0, 60, 0, 60),
        });

    private void UpsertPartial(PartialEvent p)
    {
        // drop stale partials from older segments
        foreach (var id in new List<long>(_partials.Keys))
            if (id != p.Id) { Feed.Children.Remove(_partials[id].Frame); _partials.Remove(id); }
        if (!_partials.TryGetValue(p.Id, out var b))
        {
            var frame = MakeBubble(p.Speaker, $"{p.Speaker} · listening…",
                out var body, out var orig, out var who);
            b = new PartialBubble { Frame = frame, Body = body, Orig = orig, Who = who };
            _partials[p.Id] = b;
        }
        b.Orig.Text = p.Text;
        try { b.Who.Text = $"{p.Speaker} · listening… · {p.RmsDb} dB"; } catch { }
        int i = 0;
        foreach (var kv in p.Translations)
        {
            string line = $"[{kv.Key}] {kv.Value}";
            if (i < b.Trs.Count) b.Trs[i].Text = line;
            else b.Trs.Add(MakeTr(b.Body, line));
            i++;
        }
        FeedScroll.ScrollToBottom();
    }

    private void AddFinal(UtteranceEvent u)
    {
        // Same id arrives twice (instant ASR text, then full translations):
        // upsert in place instead of duplicating.
        if (_partials.TryGetValue(u.Id, out var p))
        {
            Feed.Children.Remove(p.Frame);
            _partials.Remove(u.Id);
        }
        if (!_finals.TryGetValue(u.Id, out var b))
        {
            string badge0 = u.Denoised ? " · 🔇" : "";
            var frame = MakeBubble(u.Speaker, $"{u.Speaker} · {u.SrcLang} · {u.RmsDb} dB{badge0}",
                out var body, out var orig, out _, animate: true);
            b = new PartialBubble { Frame = frame, Body = body, Orig = orig };
            _finals[u.Id] = b;
            _history.Add(u);
            _count++;
            CountText.Text = $"{_count} utterances";
        }
        b.Orig.Text = u.Text;
        int i = 0;
        foreach (var kv in u.Translations)
        {
            string line = $"[{kv.Key}] {kv.Value}";
            if (i < b.Trs.Count) b.Trs[i].Text = line;
            else b.Trs.Add(MakeTr(b.Body, line));
            i++;
        }
        while (b.Trs.Count > i)
        {
            var extra = b.Trs[b.Trs.Count - 1];
            b.Body.Children.Remove(extra);
            b.Trs.RemoveAt(b.Trs.Count - 1);
        }
        if (u.Pending && b.Note == null)
        {
            b.Note = new TextBlock
            {
                Text = "translating…", FontSize = 11, FontStyle = FontStyles.Italic,
                Foreground = new SolidColorBrush(Hex("#7773cb")), Margin = new Thickness(0, 6, 0, 0),
            };
            b.Body.Children.Add(b.Note);
        }
        else if (!u.Pending && b.Note != null)
        {
            b.Body.Children.Remove(b.Note);
            b.Note = null;
        }
        FeedScroll.ScrollToBottom();
    }

    private void SetStatus(string s) { try { StatusText.Text = s; } catch { } }

    private void Shutdown()
    {
        try { _initCts.Cancel(); } catch { }
        _recording = false;
        try { _mic.Dispose(); } catch { }
        try { _api?.Dispose(); } catch { }
        try { _backend.Dispose(); } catch { }
    }
}
