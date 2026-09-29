// Renderer: sidebar controls + mic streaming + chat feed over /ws/live.
const BASE = "http://127.0.0.1:8000";
const WS_URL = "ws://127.0.0.1:8000/ws/live";

const SPEAKER_COLORS = ["#7C3AED", "#0EA5E9", "#F59E0B", "#10B981", "#EC4899"];

interface Translations { [k: string]: string }
interface Utterance {
  id: number; speaker: string; rms_db: number; denoised: boolean;
  src_lang: string; text: string; translations: Translations; start: number;
}
interface PartialMsg {
  id: number; speaker: string; rms_db: number; text: string;
  translations: Translations | null; final: false;
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

function speakerColor(speaker: string): string {
  const m = /(\d+)/.exec(speaker || "");
  const n = m ? parseInt(m[1], 10) : 1;
  return SPEAKER_COLORS[(Math.max(n, 1) - 1) % SPEAKER_COLORS.length];
}

function esc(s: unknown): string {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c] as string));
}

// ---------- state ----------
let ws: WebSocket | null = null;
let recording = false;
let audioCtx: AudioContext | null = null;
let micStream: MediaStream | null = null;
let proc: ScriptProcessorNode | null = null;
let frameBuf: Float32Array[] = [];
let frameLen = 0;
let utterances: Utterance[] = [];
let count = 0;
let partialEl: { id: number; root: HTMLElement; orig: HTMLElement; who: HTMLElement; trBox: HTMLElement } | null = null;

// ---------- controls ----------
function targets(): string[] {
  const t: string[] = [];
  if (($("tgt-en") as HTMLInputElement).checked) t.push("en");
  if (($("tgt-zh") as HTMLInputElement).checked) t.push("zh");
  if (($("tgt-vi") as HTMLInputElement).checked) t.push("vi");
  return t.length ? t : ["en"];
}

function srcLang(): string | null {
  const v = ($("src-lang") as HTMLSelectElement).value;
  return v === "auto" ? null : v;
}

function setStatus(s: string) {
  $("status").textContent = s;
}

// ---------- feed ----------
function clearEmpty() {
  const e = document.querySelector("#feed .empty");
  if (e) e.remove();
}

function scrollBottom() {
  const f = $("feed");
  f.scrollTop = f.scrollHeight;
}

function makeBubble(speaker: string, meta: string, animate: boolean): { root: HTMLElement; orig: HTMLElement; who: HTMLElement; trBox: HTMLElement } {
  clearEmpty();
  const root = document.createElement("div");
  root.className = "utt";
  root.style.borderLeftColor = speakerColor(speaker);
  if (animate) root.classList.add("fade-in");
  root.innerHTML = `<div class="who" style="color:${speakerColor(speaker)}">${esc(meta)}</div>
    <div class="orig"></div><div class="trs"></div>`;
  $("feed").appendChild(root);
  scrollBottom();
  return {
    root,
    orig: root.querySelector(".orig") as HTMLElement,
    who: root.querySelector(".who") as HTMLElement,
    trBox: root.querySelector(".trs") as HTMLElement,
  };
}

function setTranslations(box: HTMLElement, trs: Translations | null) {
  if (!trs) return;
  const rows = box.querySelectorAll(".tr");
  Object.entries(trs).forEach(([code, text], i) => {
    const line = `[${code}] ${text}`;
    if (i < rows.length) rows[i].textContent = line;
    else {
      const d = document.createElement("div");
      d.className = "tr";
      d.textContent = line;
      box.appendChild(d);
    }
  });
}

function upsertPartial(p: PartialMsg) {
  if (partialEl && partialEl.id !== p.id) {
    partialEl.root.remove();
    partialEl = null;
  }
  if (!partialEl) {
    const b = makeBubble(p.speaker, `${p.speaker} · listening…`, false);
    partialEl = { id: p.id, ...b };
  }
  partialEl.orig.textContent = p.text;
  partialEl.who.textContent = `${p.speaker} · listening… · ${p.rms_db} dB`;
  setTranslations(partialEl.trBox, p.translations);
  scrollBottom();
}

function addFinal(u: Utterance) {
  if (partialEl && partialEl.id === u.id) {
    partialEl.root.remove();
    partialEl = null;
  } else if (partialEl) {
    partialEl.root.remove();
    partialEl = null;
  }
  const badge = u.denoised ? " · 🔇" : "";
  const b = makeBubble(u.speaker, `${u.speaker} · ${u.src_lang} · ${u.rms_db} dB${badge}`, true);
  // streaming typewriter for the original
  const full = u.text;
  let i = 0;
  const step = Math.max(1, Math.floor(full.length / 60) + 1);
  const tick = () => {
    i = Math.min(full.length, i + step);
    b.orig.textContent = full.slice(0, i);
    scrollBottom();
    if (i < full.length) setTimeout(tick, 12);
    else {
      Object.entries(u.translations).forEach(([code, text]) => {
        const d = document.createElement("div");
        d.className = "tr";
        d.textContent = `[${code}] ${text}`;
        b.trBox.appendChild(d);
      });
      scrollBottom();
    }
  };
  tick();
  utterances.push(u);
  count++;
  $("feed-count").textContent = `${count} utterances`;
}

// ---------- backend ----------
async function refreshHealth() {
  try {
    const h = await (await fetch(`${BASE}/api/health`)).json();
    const d = h.device || {};
    ($("device-badge") as HTMLElement).innerHTML =
      `device: <b>${esc(d.device || "?")}</b> · ${esc(d.gpu_name || d.dtype || "")}`;
    ($("model-text") as HTMLElement).textContent = `ASR ${h.asr?.model || ""}\nMT ${h.mt?.model || ""}`;
  } catch {
    setStatus("Backend unreachable — start run.py or relaunch the app.");
  }
}

async function warmup() {
  setStatus("Loading models (first run downloads weights)…");
  try {
    const j = await (await fetch(`${BASE}/api/warmup`, { method: "POST" })).json();
    setStatus(`Ready — ASR ${j.asr.model} · MT ${j.mt.model}. Press Record.`);
    await refreshHealth();
  } catch (e) {
    setStatus("Warmup failed: " + e);
  }
}

// ---------- mic + streaming ----------
function floatTo16(buf: Float32Array): Int16Array {
  const o = new Int16Array(buf.length);
  for (let i = 0; i < buf.length; i++) {
    o[i] = Math.max(-32768, Math.min(32767, buf[i] * 32768));
  }
  return o;
}

function b64pcm16(pcm: Int16Array): string {
  const u8 = new Uint8Array(pcm.buffer);
  let s = "";
  for (let i = 0; i < u8.length; i += 0x8000) {
    s += String.fromCharCode.apply(null, Array.from(u8.subarray(i, i + 0x8000)));
  }
  return btoa(s);
}

function resample16k(input: Float32Array, fromRate: number): Float32Array {
  if (fromRate === 16000) return input;
  const ratio = 16000 / fromRate;
  const out = new Float32Array(Math.floor(input.length * ratio));
  for (let i = 0; i < out.length; i++) {
    const idx = Math.min(input.length - 1, Math.floor(i / ratio));
    out[i] = input[idx];
  }
  return out;
}

async function startRecording() {
  if (recording) return;
  ws = new WebSocket(WS_URL);
  await new Promise<void>((res, rej) => {
    ws!.onopen = () => res();
    ws!.onerror = () => rej(new Error("ws failed"));
  });
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (!m.ok) {
      setStatus("Error: " + (m.error || "backend error"));
      return;
    }
    if (m.type === "partial") upsertPartial(m as PartialMsg);
    else if (m.type === "utterance") addFinal(m as Utterance);
    else if (m.type === "stream_started") setStatus("● Listening — streaming ASR + translation.");
    else if (m.type === "stream_stopped") setStatus(`Stopped (${m.finalized} segments).`);
  };
  ws.send(JSON.stringify({
    type: "stream_start", targets: targets(), src_lang: srcLang(),
    denoise: ($("tgt-denoise") as HTMLInputElement).checked,
  }));

  const micId = ($("mic-list") as HTMLSelectElement).value || undefined;
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { deviceId: micId ? { exact: micId } : undefined, echoCancellation: true },
  });
  audioCtx = new AudioContext();
  const src = audioCtx.createMediaStreamSource(micStream);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  const peakBuf = new Float32Array(analyser.fftSize);
  proc = audioCtx.createScriptProcessor(4096, 1, 1);
  proc.onaudioprocess = (e) => {
    if (!recording) return;
    const ch = e.inputBuffer.getChannelData(0);
    analyser.getFloatTimeDomainData(peakBuf);
    let peak = 0;
    for (let i = 0; i < peakBuf.length; i += 4) peak = Math.max(peak, Math.abs(peakBuf[i]));
    ($("meter-fill") as HTMLElement).style.width = Math.min(100, peak * 220) + "%";
    const r16 = resample16k(ch, audioCtx!.sampleRate);
    frameBuf.push(r16);
    frameLen += r16.length;
    const need = Math.floor(16000 * 0.5);
    while (frameLen >= need) {
      const flat = new Float32Array(frameLen);
      let o = 0;
      for (const b of frameBuf) {
        flat.set(b, o);
        o += b.length;
      }
      const out = flat.slice(0, need);
      const rest = flat.slice(need);
      frameBuf = rest.length ? [rest] : [];
      frameLen = rest.length;
      if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({ type: "stream_audio", audio_b64: b64pcm16(floatTo16(out)), sr: 16000 }));
        ($("meter-label") as HTMLElement).textContent = "streaming…";
      }
    }
  };
  src.connect(analyser);
  src.connect(proc);
  proc.connect(audioCtx.destination);
  recording = true;
  frameBuf = [];
  frameLen = 0;
  ($("btn-record") as HTMLButtonElement).disabled = true;
  ($("btn-stop") as HTMLButtonElement).disabled = false;
  ($("live-pill") as HTMLElement).style.display = "";
  setStatus("● Recording — speak in vi / en / zh.");
}

async function stopRecording() {
  recording = false;
  try {
    proc?.disconnect();
    await audioCtx?.close();
    micStream?.getTracks().forEach((t) => t.stop());
    if (ws && ws.readyState === 1) {
      ws.send(JSON.stringify({ type: "stream_stop" }));
      setTimeout(() => ws?.close(), 4000);
    } else ws?.close();
  } catch { /* noop */ }
  audioCtx = null;
  micStream = null;
  proc = null;
  ($("btn-record") as HTMLButtonElement).disabled = false;
  ($("btn-stop") as HTMLButtonElement).disabled = true;
  ($("live-pill") as HTMLElement).style.display = "none";
  ($("meter-fill") as HTMLElement).style.width = "0%";
  ($("meter-label") as HTMLElement).textContent = "mic idle";
  setStatus("Stopped.");
}

async function listMics() {
  try {
    // labels need a granted stream first; failure just yields unlabeled entries
    const tmp = await navigator.mediaDevices.getUserMedia({ audio: true });
    tmp.getTracks().forEach((t) => t.stop());
  } catch { /* noop */ }
  try {
    const devs = await navigator.mediaDevices.enumerateDevices();
    const sel = $("mic-list") as HTMLSelectElement;
    sel.innerHTML = "";
    devs.filter((d) => d.kind === "audioinput").forEach((d, i) => {
      const o = document.createElement("option");
      o.value = d.deviceId;
      o.textContent = d.label || `Microphone ${i + 1}`;
      sel.appendChild(o);
    });
  } catch { /* noop */ }
}

// ---------- upload / export ----------
async function uploadFile(f: File) {
  setStatus(`Transcribing ${f.name}…`);
  const fd = new FormData();
  fd.append("file", f);
  fd.append("targets", targets().join(","));
  fd.append("src_lang", ($("src-lang") as HTMLSelectElement).value);
  fd.append("denoise", ($("tgt-denoise") as HTMLInputElement).checked ? "true" : "false");
  try {
    const j = await (await fetch(`${BASE}/api/transcribe`, { method: "POST", body: fd })).json();
    if (!j.ok) {
      setStatus("Error: " + (j.error || "failed"));
      return;
    }
    (j.utterances as Utterance[]).forEach(addFinal);
    setStatus(`Done — ${j.utterances.length} utterances.`);
  } catch (e) {
    setStatus("Upload failed: " + e);
  }
}

function download(name: string, text: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

function exportTxt() {
  if (!utterances.length) {
    setStatus("Nothing to export yet.");
    return;
  }
  const lines = utterances.map((u) =>
    `${u.speaker} (${u.src_lang}): ${u.text}\n  -> ` +
    Object.entries(u.translations).map(([k, v]) => `[${k}] ${v}`).join(" | "));
  download("transcript.txt", lines.join("\n"));
  setStatus("Exported transcript.txt");
}

function exportSrt() {
  if (!utterances.length) {
    setStatus("Nothing to export yet.");
    return;
  }
  const ts = (s: number) => {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    const p = (n: number) => String(n).padStart(2, "0");
    return `${p(h)}:${p(m)}:${p(sec)},000`;
  };
  const blocks = utterances.map((u, i) =>
    `${i + 1}\n${ts((i + 1) * 5)} --> ${ts((i + 1) * 5 + 4)}\n${u.speaker}: ${u.text}\n` +
    Object.values(u.translations).join(" ") + "\n");
  download("transcript.srt", blocks.join("\n"));
  setStatus("Exported transcript.srt");
}

// ---------- wire up ----------
window.addEventListener("DOMContentLoaded", () => {
  ($("btn-record") as HTMLButtonElement).onclick = () =>
    startRecording().catch((e) => setStatus("Mic failed: " + e));
  ($("btn-stop") as HTMLButtonElement).onclick = () => stopRecording();
  ($("btn-warmup") as HTMLButtonElement).onclick = () => warmup();
  ($("btn-txt") as HTMLButtonElement).onclick = () => exportTxt();
  ($("btn-srt") as HTMLButtonElement).onclick = () => exportSrt();
  ($("btn-clear") as HTMLButtonElement).onclick = () => {
    partialEl = null;
    utterances = [];
    count = 0;
    $("feed").innerHTML = '<div class="empty">Cleared. Press <b>Record</b> to start.</div>';
    $("feed-count").textContent = "0 utterances";
  };
  ($("file") as HTMLInputElement).onchange = (e) => {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (f) uploadFile(f);
  };
  listMics();
  refreshHealth();
  warmup();
  setInterval(refreshHealth, 15000);
});

