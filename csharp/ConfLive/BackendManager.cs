using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace ConfLive;

/// <summary>
/// Finds Python, detects CUDA (nvidia-smi) for the CPU fallback decision,
/// launches the Python backend (run.py) when needed, and waits for /api/health.
/// If a backend is already listening on the port it is reused as-is.
/// </summary>
public sealed class BackendManager : IDisposable
{
    public const int Port = 8000;
    public string BaseUrl => $"http://127.0.0.1:{Port}";
    public string BackendDir { get; private set; } = "";
    public string PreferredDevice { get; private set; } = "auto"; // cuda | cpu
    public HealthInfo LastHealth { get; private set; }
    public bool OwnsBackend => _proc != null && !_proc.HasExited;

    private Process _proc;
    private readonly HttpClient _http = new() { Timeout = TimeSpan.FromSeconds(15) };
    private bool _disposed;

    public static bool CudaPresent()
    {
        // Primary: nvidia-smi lists a GPU. Fallback: WMI video controller name.
        try
        {
            var psi = new ProcessStartInfo("nvidia-smi", "-L")
            {
                RedirectStandardOutput = true, RedirectStandardError = true,
                UseShellExecute = false, CreateNoWindow = true,
            };
            using var p = Process.Start(psi);
            if (p == null) return false;
            string output = p.StandardOutput.ReadToEnd();
            p.WaitForExit(8000);
            if (p.ExitCode == 0 && output.Contains("GPU", StringComparison.OrdinalIgnoreCase))
                return true;
        }
        catch { /* nvidia-smi missing -> not CUDA */ }
        try
        {
            using var searcher = new System.Management.ManagementObjectSearcher(
                "SELECT Name FROM Win32_VideoController");
            foreach (var o in searcher.Get())
            {
                var name = o["Name"]?.ToString() ?? "";
                if (name.Contains("NVIDIA", StringComparison.OrdinalIgnoreCase))
                    return true;
            }
        }
        catch { }
        return false;
    }

    public string LocateBackend()
    {
        // Env override wins (installer / dev setups).
        var env = Environment.GetEnvironmentVariable("CONF_LIVE_BACKEND");
        if (!string.IsNullOrEmpty(env) && File.Exists(Path.Combine(env, "run.py")))
            return env;
        // Walk up from the exe: installer layout puts backend/ next to ConfLive.exe.
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (int i = 0; i < 5 && dir != null; i++, dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "run.py")))
                return dir.FullName;
            var sub = Path.Combine(dir.FullName, "backend");
            if (File.Exists(Path.Combine(sub, "run.py")))
                return sub;
            var sub2 = Path.Combine(dir.FullName, "conference-asr-translator");
            if (File.Exists(Path.Combine(sub2, "run.py")))
                return sub2;
        }
        throw new FileNotFoundException(
            "Python backend (run.py) not found. Set CONF_LIVE_BACKEND to its folder.");
    }

    public static string LocatePython(string backendDir)
    {
        // Prefer a venv inside the backend, then py launcher, then PATH python.
        var venv = Path.Combine(backendDir, ".venv", "Scripts", "python.exe");
        if (File.Exists(venv)) return venv;
        foreach (var candidate in new[] { "py", "python" })
        {
            try
            {
                var psi = new ProcessStartInfo(candidate, "--version")
                {
                    RedirectStandardOutput = true, RedirectStandardError = true,
                    UseShellExecute = false, CreateNoWindow = true,
                };
                using var p = Process.Start(psi);
                if (p == null) continue;
                p.WaitForExit(10000);
                if (p.ExitCode == 0) return candidate;
            }
            catch { }
        }
        throw new FileNotFoundException("No Python found. Install Python 3.11/3.12 and retry.");
    }

    public async Task<HealthInfo> EnsureRunningAsync(
        Action<string> log, CancellationToken ct = default)
    {
        BackendDir = LocateBackend();
        // Reuse a backend that is already up (e.g. started manually).
        try
        {
            var existing = await GetHealthAsync(ct);
            if (existing?.Ok == true)
            {
                LastHealth = existing;
                log("Attached to running backend on port " + Port + ".");
                return existing;
            }
        }
        catch { /* not running -> launch it */ }

        PreferredDevice = CudaPresent() ? "cuda" : "cpu";
        log(PreferredDevice == "cuda"
            ? "NVIDIA GPU detected — starting backend on CUDA."
            : "No NVIDIA GPU detected — starting backend on CPU (fallback).");

        string python = LocatePython(BackendDir);
        var psi = new ProcessStartInfo(python, "run.py")
        {
            WorkingDirectory = BackendDir,
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = true, RedirectStandardError = true,
        };
        psi.Environment["PYTHONPATH"] = BackendDir;
        psi.Environment["DEVICE"] = PreferredDevice; // Python still auto-falls back on OOM
        string logPath = Path.Combine(Path.GetTempPath(), "conflive_backend.log");
        _proc = Process.Start(psi);
        if (_proc == null) throw new InvalidOperationException("Could not start backend process.");
        _ = Task.Run(() =>
        {
            try
            {
                using var w = new StreamWriter(logPath, append: true);
                w.WriteLine($"--- backend started {DateTime.Now} ({python} run.py) ---");
                _proc.OutputDataReceived += (_, e) => { if (e.Data != null) w.WriteLine(e.Data); };
                _proc.ErrorDataReceived += (_, e) => { if (e.Data != null) w.WriteLine("ERR: " + e.Data); };
                _proc.BeginOutputReadLine();
                _proc.BeginErrorReadLine();
                _proc.WaitForExit();
            }
            catch { }
        });

        // Wait for /api/health (model imports take a while on first boot).
        var deadline = DateTime.UtcNow.AddMinutes(3);
        Exception last = new TimeoutException("backend did not answer /api/health");
        while (DateTime.UtcNow < deadline)
        {
            ct.ThrowIfCancellationRequested();
            if (_proc.HasExited)
                throw new InvalidOperationException(
                    $"Backend exited early (code {_proc.ExitCode}). See {logPath}.");
            try
            {
                var h = await GetHealthAsync(ct);
                if (h?.Ok == true) { LastHealth = h; return h; }
            }
            catch (Exception ex) { last = ex; }
            await Task.Delay(1500, ct);
        }
        throw last;
    }

    public async Task<HealthInfo> GetHealthAsync(CancellationToken ct = default)
    {
        using var resp = await _http.GetAsync(BaseUrl + "/api/health", ct);
        resp.EnsureSuccessStatusCode();
        var stream = await resp.Content.ReadAsStreamAsync(ct);
        return await JsonSerializer.DeserializeAsync<HealthInfo>(stream,
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true }, ct);
    }

    public async Task WarmupAsync(CancellationToken ct = default)
    {
        using var resp = await _http.PostAsync(BaseUrl + "/api/warmup",
            new StringContent(""), ct);
        resp.EnsureSuccessStatusCode();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        try
        {
            if (_proc != null && !_proc.HasExited) _proc.Kill(entireProcessTree: true);
        }
        catch { }
        _http.Dispose();
    }
}
