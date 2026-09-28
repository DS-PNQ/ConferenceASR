/* Landing streaming-text demo + live studio client (mic WS + upload + export). */
const $ = (id) => document.getElementById(id);
const feed = () => $("feed");

/* ---------- 1. Conversational preview with streaming text animation ---------- */
const DEMO_SCRIPT = [
  { who: "SPEAKER 01 · 中文", cls: "m-a", text: "各位早上好，今天我们先过一下季度目标。", tr: "Good morning everyone, let's start with quarterly goals." },
  { who: "SPEAKER 02 · EN", cls: "m-b", text: "Revenue is up 12% — the new onboarding flow worked.", tr: "Doanh thu tăng 12% — quy trình onboarding mới đã hiệu quả." },
  { who: "SPEAKER 01 · VI", cls: "m-a", text: "Tuyệt vời, vậy quý tới chúng ta mở rộng sang tiếng Trung nhé.", tr: "太棒了，下季度我们扩展中文支持。" },
];
function typeInto(el, text, speed = 22) {
  return new Promise((res) => {
    let i = 0;
    const tick = () => {
      el.textContent = text.slice(0, ++i);
      if (i < text.length) setTimeout(tick, speed);
      else res();
    };
    tick();
  });
}
async function runDemo() {
  const chat = $("demo-chat"), typed = $("demo-typed");
  while (true) {
    chat.innerHTML = "";
    for (const line of DEMO_SCRIPT) {
      const d = document.createElement("div");
      d.className = "msg " + line.cls;
      d.innerHTML = `<span class="who">${line.who}</span><span class="body"></span>`;
      chat.appendChild(d);
      typed.textContent = "▸ transcribing…";
      await typeInto(d.querySelector(".body"), line.text, 26);
      const tr = document.createElement("span");
      tr.className = "tr";
      d.appendChild(tr);
      await typeInto(tr, "→ " + line.tr, 14);
      typed.textContent = "";
      await new Promise((r) => setTimeout(r, 700));
    }
    await new Promise((r) => setTimeout(r, 2200));
  }
}

/* ---------- 2. Health / device badge ---------- */
async function refreshHealth() {
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    const d = h.device || {};
    $("device-badge").innerHTML =
      `device: <b>${d.device || "?"}</b> · ${d.gpu_name || d.dtype || ""} · ASR ${h.asr?.model || ""}`;
    const sd = $("stat-device");
    if (sd) sd.textContent = (d.device || "").toUpperCase() + (d.gpu_name ? " · " + d.gpu_name.split(" ").slice(0, 3).join(" ") : "");
    $("status").textContent = h.asr?.ready || h.mt?.ready
      ? "Models ready (or demo mode). Press Record."
      : "Models not loaded yet — press “Load models” (downloads on first run).";
  } catch { $("status").textContent = "Server unreachable — is `python run.py` running?"; }
}

/* ---------- 3. Transcript feed ---------- */
const SPEAKER_COLORS = ["s0", "s1", "s2", "s0", "s1"];
function speakerClass(spk) {
  const m = /(\d+)/.exec(spk || "");
  return SPEAKER_COLORS[m ? (parseInt(m[1], 10) - 1) % 3 : 0];
}
let count = 0;
function addUtterance(u) {
  const empty = feed().querySelector(".empty");
  if (empty) empty.remove();
  const d = document.createElement("div");
  d.className = "utt " + speakerClass(u.speaker);
  const trs = Object.entries(u.translations || {}).map(([k, v]) => `<span class="tr"><b>[${k}]</b> ${escapeHtml(v)}</span>`).join("");
  d.innerHTML = `<span class="who">${escapeHtml(u.speaker)} · ${escapeHtml(u.src_lang || "")} · ${u.rms_db ?? ""} dB${u.denoised ? " · 🔇" : ""}</span>
    <div class="orig"></div>${trs}`;
  feed().appendChild(d);
  feed().scrollTop = feed().scrollHeight;
  // streaming typewriter for the original text
  const body = d.querySelector(".orig");
  typeInto(body, u.text || "", 12);
  count++;
  $("feed-count").textContent = `${count} utterances`;
}
function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function targets() {
  const t = [];
  if ($("tgt-en").checked) t.push("en");
  if ($("tgt-zh").checked) t.push("zh");
  if ($("tgt-vi").checked) t.push("vi");
  return t.length ? t : ["en"];
}

/* ---------- 4. Live mic over WebSocket (16 kHz PCM16 base64 chunks) ---------- */
let ws = null, audioCtx = null, stream = null, proc = null, recording = false, chunkBuf = [];
function wsURL() { return (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/live"; }
function connectWS() {
  return new Promise((resolve, reject) => {
    ws = new WebSocket(wsURL());
    ws.onopen = resolve;
    ws.onerror = reject;
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.ok && m.type === "utterance") addUtterance(m);
      else if (m.ok && m.type === "silence") { $("meter-label").textContent = `… ${m.rms_db} dB (pause)`; }
    };
  });
}
function floatTo16(buf) {
  const o = new Int16Array(buf.length);
  for (let i = 0; i < buf.length; i++) o[i] = Math.max(-32768, Math.min(32767, buf[i] * 32768));
  return o;
}
function b64(bytes) {
  let s = "";
  const u8 = new Uint8Array(bytes.buffer);
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
}
async function startRecording() {
  try {
    await connectWS();
  } catch { $("status").textContent = "WebSocket failed — is the server running?"; return; }
  stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true } });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  const src = audioCtx.createMediaStreamSource(stream);
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  proc = audioCtx.createScriptProcessor(4096, 1, 1);
  const SEG = 4.0; // seconds per chunk ≈ config segment_seconds
  proc.onaudioprocess = (e) => {
    if (!recording) return;
    const ch = e.inputBuffer.getChannelData(0);
    // meter
    let peak = 0;
    for (let i = 0; i < ch.length; i += 4) peak = Math.max(peak, Math.abs(ch[i]));
    $("meter-fill").style.width = Math.min(100, peak * 220) + "%";
    chunkBuf.push(new Float32Array(ch));
    const total = chunkBuf.reduce((a, b) => a + b.length, 0);
    if (total >= 16000 * SEG) {
      const flat = new Float32Array(total);
      let o = 0;
      for (const b of chunkBuf) { flat.set(b, o); o += b.length; }
      chunkBuf = [];
      const pcm16 = floatTo16(flat);
      if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({ audio_b64: b64(pcm16), sr: 16000, targets: targets(), src_lang: $("src-lang").value === "auto" ? null : $("src-lang").value, denoise: $("tgt-denoise").checked }));
        $("meter-label").textContent = "streaming…";
      }
    }
  };
  src.connect(analyser);
  src.connect(proc);
  proc.connect(audioCtx.destination);
  recording = true;
  $("btn-record").disabled = true;
  $("btn-stop").disabled = false;
  $("status").textContent = "● Recording — speak in vi / en / zh.";
}

/* ---------- 5. Wire up ---------- */
window.addEventListener("DOMContentLoaded", () => {
  runDemo();
  refreshHealth();
  setInterval(refreshHealth, 15000);
  $("btn-record").onclick = startRecording;
  $("btn-stop").onclick = () => {
    recording = false;
    try { proc?.disconnect(); audioCtx?.close(); stream?.getTracks().forEach((t) => t.stop();); ws?.close(); } catch {}
    $("btn-record").disabled = false;
    $("btn-stop").disabled = true;
    $("meter-fill").style.width = "0%";
    $("meter-label").textContent = "mic idle";
    $("status").textContent = "Stopped.";
  };
  $("btn-warmup").onclick = async () => {
    $("status").textContent = "Loading models (first run downloads weights)…";
    const r = await fetch("/api/warmup", { method: "POST" });
    const j = await r.json();
    $("status").textContent = `ASR: ${j.asr.model} · MT: ${j.mt.model} · Denoise: ${j.denoise?.backend || "?"}`;
    refreshHealth();
  };
  $("file").onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    $("status").textContent = `Transcribing ${f.name}…`;
    const fd = new FormData();
    fd.append("file", f);
    fd.append("targets", targets().join(","));
    fd.append("src_lang", $("src-lang").value);
    fd.append("denoise", $("tgt-denoise").checked ? "true" : "false");
    const r = await fetch("/api/transcribe", { method: "POST", body: fd });
    const j = await r.json();
    if (!j.ok) { $("status").textContent = "Error: " + (j.error || "failed"); return; }
    j.utterances.forEach(addUtterance);
    $("status").textContent = `Done — ${j.utterances.length} utterances.`;
  };
  $("btn-txt").onclick = () => window.open("/api/export.txt", "_blank");
  $("btn-srt").onclick = () => window.open("/api/export.srt", "_blank");
  $("btn-clear").onclick = () => { feed().innerHTML = '<div class="empty">Cleared. Press <b>Record</b> to start a new session.</div>'; count = 0; $("feed-count").textContent = "0 utterances"; };
});
