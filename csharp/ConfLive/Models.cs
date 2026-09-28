using System.Text.Json.Serialization;

namespace ConfLive;

// Health / status DTOs (only the fields the UI needs)
public sealed class DeviceInfo
{
    [JsonPropertyName("device")] public string Device { get; set; } = "?";
    [JsonPropertyName("gpu_name")] public string GpuName { get; set; } = "";
    [JsonPropertyName("dtype")] public string Dtype { get; set; } = "";
}

public sealed class EngineInfo
{
    [JsonPropertyName("model")] public string Model { get; set; } = "?";
    [JsonPropertyName("backend")] public string Backend { get; set; } = "";
    [JsonPropertyName("device")] public string Device { get; set; } = "";
    [JsonPropertyName("ready")] public bool Ready { get; set; }
}

public sealed class HealthInfo
{
    [JsonPropertyName("ok")] public bool Ok { get; set; }
    [JsonPropertyName("device")] public DeviceInfo Device { get; set; } = new();
    [JsonPropertyName("asr")] public EngineInfo Asr { get; set; } = new();
    [JsonPropertyName("mt")] public EngineInfo Mt { get; set; } = new();
    [JsonPropertyName("denoise")] public EngineInfo Denoise { get; set; } = new();
    [JsonPropertyName("langs")] public string[] Langs { get; set; } = System.Array.Empty<string>();
    [JsonPropertyName("diarizer")] public string Diarizer { get; set; } = "";
}

// Live transcript events from /ws/live
public sealed class PartialEvent
{
    public long Id { get; set; }
    public string Speaker { get; set; } = "";
    public double RmsDb { get; set; }
    public string Text { get; set; } = "";
    public System.Collections.Generic.Dictionary<string, string> Translations { get; set; } = new();
}

public sealed class UtteranceEvent
{
    public long Id { get; set; }
    public string Speaker { get; set; } = "";
    public double RmsDb { get; set; }
    public bool Denoised { get; set; }
    public string SrcLang { get; set; } = "";
    public string Text { get; set; } = "";
    public System.Collections.Generic.Dictionary<string, string> Translations { get; set; } = new();
    public double Start { get; set; }
}
