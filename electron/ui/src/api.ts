// Live bridge to the ConfLive Python backend (REST + streaming WS + mic).
export const BASE = "http://127.0.0.1:8000";
export const WS_URL = "ws://127.0.0.1:8000/ws/live";

export interface Translations {
  [k: string]: string;
}

export interface Timings {
  asr_ms: number;
  diar_ms: number;
  mt_ms: number;
  total_ms: number;
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
  timings?: Timings;
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
  diarizer_status: { backend: string; model?: string; device?: string; ready: boolean; latency_ms?: number };
  denoise: { backend: string; ready: boolean; device?: string };
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

export type DiarizerMode = "nemotron" | "pyannote" | "volume" | "nemo";

export async function apiSetDiarizer(mode: DiarizerMode): Promise<unknown> {
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

export async function apiTranslateText(  text: string,
  target: string,
  terms: Record<string, string> = {},
  src = "auto",
): Promise<string> {
  const r = await fetch(`${BASE}/api/translate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, targets: [target], src, terms }),
  });
  const j = await r.json();
  if (!j.ok) throw new Error("translate failed");
  return (j.translations as Record<string, string>)[target] ?? "";
}

// ---------- document OCR (PP-OCR blocks / PaddleOCR-VL + Hy-MT) ----------
export interface OcrBlock {
  id: number;
  text: string;
  conf: number;
  box: [number, number, number, number]; // relative x0,y0,x1,y1
  translation?: string;
  translation_error?: string;
}

export interface OcrPage {
  file: string;
  page: number;
  text?: string;
  blocks?: OcrBlock[];
  overall_conf?: number;
  backend?: string;
  preview?: string; // data-URL of the transformed page (overlay aligns to this)
  translation?: string;
  error?: string;
  translation_error?: string;
  ocr_ms?: number;
}

export interface OcrResult {
  ok: boolean;
  task: string;
  pages: OcrPage[];
}

export async function apiOcr(
  files: File[],
  opts: { task: string; translateTo: string; src: string; terms: Record<string, string>; rotate?: number; crop?: [number, number, number, number] | null },
): Promise<OcrResult> {
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  fd.append("task", opts.task);
  fd.append("translate_to", opts.translateTo);
  fd.append("src", opts.src);
  fd.append("terms", JSON.stringify(opts.terms));
  fd.append("rotate", String(opts.rotate ?? 0));
  fd.append("crop", opts.crop ? JSON.stringify(opts.crop) : "");
  const r = await fetch(`${BASE}/api/ocr`, { method: "POST", body: fd });
  const j = await r.json();
  if (!j.ok) throw new Error(j.error || "ocr failed");
  return j as OcrResult;
}

export type WSEvents = {
  onPartial: (p: PartialMsg) => void;
  onTok: (t: { id: number; tgt: string; seq: number; delta: string }) => void;
  onUtterance: (u: Utterance) => void;
  onStatus: (s: string) => void;
  onError: (e: string) => void;
  onClose?: () => void;
};

export class LiveSocket {
  private ws: WebSocket | null = null;
  private ev: WSEvents | null = null;
  private intentionalClose = false;
  private ready = false;
  private readyWaiters: Array<() => void> = [];
  private stopDone: (() => void) | null = null;

  async connect(ev: WSEvents): Promise<void> {
    this.close();
    this.ready = false;
    this.readyWaiters = [];
    this.close();
    const ws = new WebSocket(WS_URL);
    this.ws = ws;
    this.ev = ev;
    this.intentionalClose = false;
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
      else if (t === "stream_started") {
        this.markReady();
        ev.onStatus("listening");
      } else if (t === "stream_stopped") {
        ev.onStatus(`stopped (${String(m.finalized)} segments)`);
        this.stopDone?.();
      }
    };
    ws.onclose = () => {
      this.stopDone?.();
      if (this.ws === ws) this.ws = null;
      const cb = this.ev?.onClose;
      this.ev = null;
      if (!this.intentionalClose) {
        try {
          cb?.();
        } catch {
          /* noop */
        }
      }
      this.intentionalClose = false;
    };
  }

  send(obj: unknown): void {
    if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(obj));
  }

  // resume: reconnect mid-meeting, keep the server's speaker clusters
  start(opts: { targets: string[]; srcLang: string | null; denoise: boolean; terms: Record<string, string>; displayLang: string; resume?: boolean }): void {
    this.ready = false;
    this.send({ type: "stream_start", targets: opts.targets, src_lang: opts.srcLang, denoise: opts.denoise, terms: opts.terms, display_lang: opts.displayLang, resume: !!opts.resume });
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
    // Drop frames until the server confirms stream_started: it loads the ASR
    // model during stream_start, and anything sent earlier used to come back
    // as "no stream open (send stream_start first)".
    if (!this.ready) return;
    this.send({ type: "stream_audio", audio_b64: b64, sr });
  }

  /** Resolves when the server confirms stream_started (false on timeout). */
  waitForReady(timeoutMs = 120000): Promise<boolean> {
    if (this.ready) return Promise.resolve(true);
    return new Promise((res) => {
      const done = (ok: boolean) => {
        clearTimeout(timer);
        this.readyWaiters = this.readyWaiters.filter((w) => w !== fire);
        res(ok);
      };
      const fire = () => done(true);
      this.readyWaiters.push(fire);
      const timer = setTimeout(() => done(this.ready), timeoutMs);
    });
  }

  private markReady(): void {
    this.ready = true;
    const waiters = this.readyWaiters.splice(0);
    for (const w of waiters) {
      try {
        w();
      } catch {
        /* noop */
      }
    }
  }

  /** Finalize the open segment now (pause): no frames = no endpoint. */
  flush(): void {
    this.send({ type: "stream_flush" });
  }

  /** Resolves on stream_stopped: the server finalizes the open segment and
   *  drains its translations first (can take tens of seconds). */
  stop(timeoutMs = 310000): Promise<void> { // server drains MT for up to 300 s
    this.send({ type: "stream_stop" });
    return new Promise((res) => {
      const timer = setTimeout(() => this.stopDone?.(), timeoutMs);
      this.stopDone = () => {
        clearTimeout(timer);
        this.stopDone = null;
        res();
      };
    });
  }

  /** Bytes queued but unsent — grows when the server can't keep up. */
  pending(): number {
    try {
      return this.ws ? this.ws.bufferedAmount : 0;
    } catch {
      return 0;
    }
  }

  close(): void {
    this.intentionalClose = true;
    try {
      this.ws?.close();
    } catch {
      /* noop */
    }
    this.ws = null;
    this.ev = null;
  }
}

// ---------- mic capture (16 kHz mono PCM16 frames + live level) ----------
export interface MicHandle {
  stop: () => void;
  setPaused: (p: boolean) => void;
  /** Drop the next `seconds` of audio (resume pops, device switches). */
  dropNext: (seconds: number) => void;
  /** Seconds since the last emitted frame (watchdog input). */
  idleFor: () => number;
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

/** openMic device id for "what the computer plays" (Electron grants
 *  getDisplayMedia with Windows loopback audio, see electron/src/main.ts). */
export const SYSTEM_AUDIO = "system";

async function systemAudioStream(): Promise<MediaStream> {
  const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
  stream.getVideoTracks().forEach((t) => t.stop()); // audio only
  if (!stream.getAudioTracks().length) {
    stream.getTracks().forEach((t) => t.stop());
    throw new Error("no system audio track (share a screen with audio)");
  }
  return stream;
}

export async function openMic(
  deviceId: string | undefined,
  onFrame: (b64: string) => void,
  frameSeconds = 0.5,
  onTrackEnded?: () => void,
): Promise<MicHandle> {
  const stream = deviceId === SYSTEM_AUDIO ? await systemAudioStream() : await navigator.mediaDevices.getUserMedia({
    audio: deviceId ? { deviceId: { exact: deviceId }, echoCancellation: true } : { echoCancellation: true },
  });
  const track = stream.getAudioTracks()[0];
  const label = deviceId === SYSTEM_AUDIO ? "System audio" : track?.label || "Microphone";
  if (track) {
    track.onended = () => {
      try {
        onTrackEnded?.();
      } catch {
        /* noop */
      }
    };
  }
  const ctx = new AudioContext();
  await ctx.resume().catch(() => undefined); // autoplay policy: unlock on gesture
  const src = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  const peakBuf = new Float32Array(analyser.fftSize);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  let paused = false;
  let chunk: Float32Array[] = [];
  let chunkLen = 0;
  let dropUntil = 0;
  let lastEmit = performance.now();
  let peak = 0;
  proc.onaudioprocess = (e) => {
    analyser.getFloatTimeDomainData(peakBuf);
    let p = 0;
    for (let i = 0; i < peakBuf.length; i += 4) p = Math.max(p, Math.abs(peakBuf[i]));
    peak = p;
    if (paused) return;
    if (performance.now() < dropUntil) return; // resume/device pop guard
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
      lastEmit = performance.now();
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
      } else {
        dropUntil = performance.now() + 250; // swallow the resume pop
      }
    },
    dropNext: (seconds: number) => {
      dropUntil = performance.now() + seconds * 1000;
    },
    idleFor: () => (performance.now() - lastEmit) / 1000,
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
    return [
      ...devs
        .filter((d) => d.kind === "audioinput")
        .map((d, i) => ({ id: d.deviceId, label: d.label || `Microphone ${i + 1}` })),
      { id: SYSTEM_AUDIO, label: "System audio (what this computer plays)" },
    ];
  } catch {
    return [{ id: SYSTEM_AUDIO, label: "System audio (what this computer plays)" }];
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
  timings?: Timings;
}

export interface ArchivedSession {
  id: string;
  label: string;
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
  // Quota (~5 MB) is hit by a few hour-long sessions; the old silent drop
  // lost the NEWEST one. Shed the oldest sessions (list is newest-first).
  for (let n = Math.min(50, sessions.length); n > 0; n--) {
    try {
      localStorage.setItem(ARCHIVE_KEY, JSON.stringify(sessions.slice(0, n)));
      return;
    } catch {
      /* quota — try with one fewer */
    }
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
