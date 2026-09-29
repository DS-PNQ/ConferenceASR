// Live bridge to the ConfLive Python backend (REST + streaming WS + mic).
export const BASE = "http://127.0.0.1:8000";
export const WS_URL = "ws://127.0.0.1:8000/ws/live";

export interface Translations {
  [k: string]: string;
}

export interface Utterance {
  id: number;
  speaker: string;
  rms_db: number;
  denoised: boolean;
  src_lang: string;
  text: string;
  translations: Translations;
  start: number; // epoch seconds (server clock)
  diar_backend?: string;
}

export interface PartialMsg {
  id: number;
  speaker: string;
  rms_db: number;
  text: string;
  translations: Translations | null;
  final: false;
}

export interface Health {
  ok: boolean;
  device: { device: string; gpu_name?: string; dtype?: string };
  asr: { model: string; backend?: string; device?: string; ready: boolean };
  mt: { model: string; device?: string; ready: boolean };
  diarizer: string;
  diarizer_status: { backend: string; model?: string; device?: string; ready: boolean };
  denoise: { backend: string; ready: boolean };
  langs: string[];
}

export async function apiHealth(): Promise<Health> {
  const r = await fetch(`${BASE}/api/health`);
  if (!r.ok) throw new Error(`health ${r.status}`);
  return r.json();
}

export async function apiWarmup(): Promise<unknown> {
  const r = await fetch(`${BASE}/api/warmup`, { method: "POST" });
  if (!r.ok) throw new Error(`warmup ${r.status}`);
  return r.json();
}

export async function apiSetDiarizer(mode: "pyannote" | "volume" | "nemo"): Promise<unknown> {
  const r = await fetch(`${BASE}/api/settings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ diarizer: mode }),
  });
  if (!r.ok) throw new Error(`settings ${r.status}: ${await r.text()}`);
  return r.json();
}

export async function apiTranscribe(
  file: File,
  opts: { targets: string[]; srcLang: string; denoise: boolean; terms: Record<string, string> },
): Promise<Utterance[]> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("targets", opts.targets.join(","));
  fd.append("src_lang", opts.srcLang);
  fd.append("denoise", opts.denoise ? "true" : "false");
  fd.append("terms", JSON.stringify(opts.terms));
  const r = await fetch(`${BASE}/api/transcribe`, { method: "POST", body: fd });
  const j = await r.json();
  if (!j.ok) throw new Error(j.error || "transcribe failed");
  return j.utterances as Utterance[];
}

export async function apiTranslateText(
  text: string,
  target: string,
  terms: Record<string, string> = {},
): Promise<string> {
  const r = await fetch(`${BASE}/api/translate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, targets: [target], src: "auto", terms }),
  });
  const j = await r.json();
  if (!j.ok) throw new Error("translate failed");
  return (j.translations as Record<string, string>)[target] ?? "";
}

export type WSEvents = {
  onPartial: (p: PartialMsg) => void;
  onTok: (t: { id: number; tgt: string; seq: number; delta: string }) => void;
  onUtterance: (u: Utterance) => void;
  onStatus: (s: string) => void;
  onError: (e: string) => void;
};

export class LiveSocket {
  private ws: WebSocket | null = null;

  async connect(ev: WSEvents): Promise<void> {
    this.close();
    const ws = new WebSocket(WS_URL);
    this.ws = ws;
    await new Promise<void>((res, rej) => {
      ws.onopen = () => res();
      ws.onerror = () => rej(new Error("websocket failed — is the backend running?"));
    });
    ws.onmessage = (msg) => {
      let m: Record<string, unknown>;
      try {
        m = JSON.parse(msg.data);
      } catch {
        ev.onError("bad server message");
        return;
      }
      if (!m.ok) {
        ev.onError(String(m.error || "backend error"));
        return;
      }
      const t = m.type as string;
      if (t === "partial") ev.onPartial(m as unknown as PartialMsg);
      else if (t === "tok")
        ev.onTok({ id: Number(m.id), tgt: String(m.tgt), seq: Number(m.seq), delta: String(m.delta ?? "") });
      else if (t === "utterance") ev.onUtterance(m as unknown as Utterance);
      else if (t === "stream_started") ev.onStatus("listening");
      else if (t === "stream_stopped") ev.onStatus(`stopped (${String(m.finalized)} segments)`);
    };
    ws.onclose = () => {
      if (this.ws === ws) this.ws = null;
    };
  }

  send(obj: unknown): void {
    if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(obj));
  }

  start(opts: { targets: string[]; srcLang: string | null; denoise: boolean; terms: Record<string, string>; displayLang: string }): void {
    this.send({ type: "stream_start", targets: opts.targets, src_lang: opts.srcLang, denoise: opts.denoise, terms: opts.terms, display_lang: opts.displayLang });
  }

  config(opts: { targets?: string[]; denoise?: boolean; terms?: Record<string, string>; displayLang?: string }): void {
    const msg: Record<string, unknown> = { type: "stream_config" };
    if (opts.targets) msg.targets = opts.targets;
    if (opts.denoise !== undefined) msg.denoise = opts.denoise;
    if (opts.terms) msg.terms = opts.terms;
    if (opts.displayLang) msg.display_lang = opts.displayLang;
    this.send(msg);
  }

  audio(b64: string, sr = 16000): void {
    this.send({ type: "stream_audio", audio_b64: b64, sr });
  }

  stop(): void {
    this.send({ type: "stream_stop" });
  }

  close(): void {
    try {
      this.ws?.close();
    } catch {
      /* noop */
    }
    this.ws = null;
  }
}

// ---------- mic capture (16 kHz mono PCM16 frames + live level) ----------
export interface MicHandle {
  stop: () => void;
  setPaused: (p: boolean) => void;
  level: () => number;
  label: string;
}

function floatTo16(buf: Float32Array): Int16Array {
  const o = new Int16Array(buf.length);
  for (let i = 0; i < buf.length; i++) o[i] = Math.max(-32768, Math.min(32767, buf[i] * 32768));
  return o;
}

function b64pcm16(pcm: Int16Array): string {
  const u8 = new Uint8Array(pcm.buffer);
  let s = "";
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, Array.from(u8.subarray(i, i + 0x8000)));
  return btoa(s);
}

function resample16k(input: Float32Array, fromRate: number): Float32Array {
  if (fromRate === 16000) return input;
  const ratio = 16000 / fromRate;
  const out = new Float32Array(Math.max(1, Math.floor(input.length * ratio)));
  for (let i = 0; i < out.length; i++) out[i] = input[Math.min(input.length - 1, Math.floor(i / ratio))];
  return out;
}

export async function openMic(
  deviceId: string | undefined,
  onFrame: (b64: string) => void,
  frameSeconds = 0.5,
): Promise<MicHandle> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: deviceId ? { deviceId: { exact: deviceId }, echoCancellation: true } : { echoCancellation: true },
  });
  const track = stream.getAudioTracks()[0];
  const label = track?.label || "Microphone";
  const ctx = new AudioContext();
  const src = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  const peakBuf = new Float32Array(analyser.fftSize);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  let paused = false;
  let chunk: Float32Array[] = [];
  let chunkLen = 0;
  let peak = 0;
  proc.onaudioprocess = (e) => {
    analyser.getFloatTimeDomainData(peakBuf);
    let p = 0;
    for (let i = 0; i < peakBuf.length; i += 4) p = Math.max(p, Math.abs(peakBuf[i]));
    peak = p;
    if (paused) return;
    const r16 = resample16k(e.inputBuffer.getChannelData(0), ctx.sampleRate);
    chunk.push(r16);
    chunkLen += r16.length;
    const need = Math.floor(16000 * frameSeconds);
    while (chunkLen >= need) {
      const flat = new Float32Array(chunkLen);
      let o = 0;
      for (const b of chunk) {
        flat.set(b, o);
        o += b.length;
      }
      onFrame(b64pcm16(floatTo16(flat.slice(0, need))));
      const rest = flat.slice(need);
      chunk = rest.length ? [rest] : [];
      chunkLen = rest.length;
    }
  };
  src.connect(analyser);
  src.connect(proc);
  proc.connect(ctx.destination);
  return {
    label,
    setPaused: (p: boolean) => {
      paused = p;
      if (p) {
        chunk = [];
        chunkLen = 0;
      }
    },
    level: () => peak,
    stop: () => {
      try {
        proc.disconnect();
        void ctx.close();
        stream.getTracks().forEach((t) => t.stop());
      } catch {
        /* noop */
      }
    },
  };
}

export async function listMics(): Promise<{ id: string; label: string }[]> {
  try {
    const tmp = await navigator.mediaDevices.getUserMedia({ audio: true });
    tmp.getTracks().forEach((t) => t.stop());
  } catch {
    /* labels may stay empty without permission */
  }
  try {
    const devs = await navigator.mediaDevices.enumerateDevices();
    return devs
      .filter((d) => d.kind === "audioinput")
      .map((d, i) => ({ id: d.deviceId, label: d.label || `Microphone ${i + 1}` }));
  } catch {
    return [];
  }
}

// ---------- session archive (localStorage) ----------
export interface ArchivedUtterance {
  time: string;
  speaker: string;
  text: string;
  translations: Record<string, string>;
  /** pre-multilang archives stored one string; still readable */
  translation?: string;
}

export interface ArchivedSession {
  id: string;
  startedAt: number;
  seconds: number;
  utterances: ArchivedUtterance[];
}

const ARCHIVE_KEY = "conflive.sessions.v1";

export function loadArchive(): ArchivedSession[] {
  try {
    const raw = localStorage.getItem(ARCHIVE_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return [];
    // drop corrupt entries so one bad session can never blank Library
    return (arr as ArchivedSession[]).filter(
      (s) => s && typeof s === "object" && Array.isArray(s.utterances),
    );
  } catch {
    return [];
  }
}

export function saveArchive(sessions: ArchivedSession[]): void {
  try {
    localStorage.setItem(ARCHIVE_KEY, JSON.stringify(sessions.slice(0, 50)));
  } catch {
    /* quota — drop silently */
  }
}

// ---------- glossary (localStorage term pairs) ----------
const GLOSSARY_KEY = "conflive.glossary.v1";

export function loadGlossary(): Record<string, string> {
  try {
    const raw = localStorage.getItem(GLOSSARY_KEY);
    if (!raw) return {};
    const o = JSON.parse(raw);
    return o && typeof o === "object" ? (o as Record<string, string>) : {};
  } catch {
    return {};
  }
}

export function saveGlossary(terms: Record<string, string>): void {
  try {
    localStorage.setItem(GLOSSARY_KEY, JSON.stringify(terms));
  } catch {
    /* noop */
  }
}
