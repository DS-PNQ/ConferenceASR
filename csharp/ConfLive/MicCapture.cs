using System;
using System.Collections.Generic;
using NAudio.Wave;

namespace ConfLive;

/// <summary>
/// 16 kHz mono mic capture. Slices the stream into short frames for the
/// streaming /ws/live protocol and reports a 0..1 level for the meter.
/// </summary>
public sealed class MicCapture : IDisposable
{
    public const int SampleRate = 16000;
    public double FrameSeconds { get; set; } = 0.5;

    public event Action<byte[]> FrameReady;
    public event Action<double> LevelChanged;

    private WaveInEvent _waveIn;
    private readonly List<byte> _buffer = new();
    private int _device = -1;
    private bool _disposed;

    public static List<(int Index, string Name)> ListInputs()
    {
        var out_ = new List<(int, string)>();
        for (int i = 0; i < WaveIn.DeviceCount; i++)
        {
            try { out_.Add((i, WaveIn.GetCapabilities(i).ProductName)); }
            catch { }
        }
        return out_;
    }

    public void Start(int device = -1)
    {
        Stop();
        _device = device;
        _buffer.Clear();
        _waveIn = new WaveInEvent
        {
            DeviceNumber = device < 0 ? 0 : device,
            WaveFormat = new WaveFormat(SampleRate, 16, 1),
            BufferMilliseconds = 100,
            NumberOfBuffers = 3,
        };
        _waveIn.DataAvailable += OnData;
        _waveIn.StartRecording();
    }

    private void OnData(object sender, WaveInEventArgs e)
    {
        // level from this block (16-bit mono)
        double peak = 0;
        for (int i = 0; i + 1 < e.BytesRecorded; i += 64)
        {
            short s = (short)(e.Buffer[i] | (e.Buffer[i + 1] << 8));
            double v = Math.Abs(s / 32768.0);
            if (v > peak) peak = v;
        }
        try { LevelChanged?.Invoke(Math.Min(1.0, peak * 2.2)); } catch { }

        lock (_buffer)
        {
            for (int i = 0; i < e.BytesRecorded; i++) _buffer.Add(e.Buffer[i]);
            int need = (int)(SampleRate * FrameSeconds) * 2;
            while (_buffer.Count >= need)
            {
                var frame = _buffer.GetRange(0, need).ToArray();
                _buffer.RemoveRange(0, need);
                try { FrameReady?.Invoke(frame); } catch { }
            }
        }
    }

    public void Stop()
    {
        try { _waveIn?.StopRecording(); } catch { }
        try { _waveIn?.Dispose(); } catch { }
        _waveIn = null;
        lock (_buffer) { _buffer.Clear(); }
        try { LevelChanged?.Invoke(0); } catch { }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Stop();
    }
}
