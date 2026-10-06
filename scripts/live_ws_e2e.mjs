// Run with the backend up (py -3 run.py): node scripts/live_ws_e2e.mjs [file.wav]
// >1 min audio (PAUSE_AT=9999) catches the 1011 keepalive-timeout drop.
// Drives /ws/live exactly like electron/ui/src/api.ts (LiveSocket + openMic):
// stream_start -> wait stream_started -> 0.5 s PCM16 frames in real time ->
// a mid-session PAUSE (no frames) -> resume -> stream_stop.
import fs from "node:fs";

const WAV = process.argv[2] || "C:/Users/Asus/AppData/Local/Temp/opencode/asr_en.wav";
const PAUSE_AT = Number(process.env.PAUSE_AT ?? 12); // seconds of audio before pausing
const PAUSE_FOR = Number(process.env.PAUSE_FOR ?? 15);
const SPEED = Number(process.env.SPEED ?? 1);

function readWav(p) {
  const b = fs.readFileSync(p);
  let off = 12, fmt, data;
  while (off < b.length) {
    const id = b.toString("ascii", off, off + 4), sz = b.readUInt32LE(off + 4);
    if (id === "fmt ") fmt = { ch: b.readUInt16LE(off + 10), sr: b.readUInt32LE(off + 12), bits: b.readUInt16LE(off + 22) };
    if (id === "data") data = b.subarray(off + 8, off + 8 + sz);
    off += 8 + sz + (sz & 1);
  }
  const bs = fmt.bits / 8, n = Math.floor(data.length / bs / fmt.ch), out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = data.readIntLE(i * bs * fmt.ch, bs) / 2 ** (fmt.bits - 1);
  // same nearest-neighbour resample as api.ts resample16k
  const ratio = 16000 / fmt.sr, r = new Float32Array(Math.floor(n * ratio));
  for (let i = 0; i < r.length; i++) r[i] = out[Math.min(n - 1, Math.floor(i / ratio))];
  return r;
}
const b64 = (f) => {
  const o = new Int16Array(f.length);
  for (let i = 0; i < f.length; i++) o[i] = Math.max(-32768, Math.min(32767, f[i] * 32768));
  return Buffer.from(o.buffer).toString("base64");
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const pcm = readWav(WAV);
await fetch("http://127.0.0.1:8000/api/warmup", { method: "POST" });
const T0 = Date.now(), ts = () => ((Date.now() - T0) / 1000).toFixed(1).padStart(6);
let phase = "live";
const ev = { partial: 0, tok: 0, utterance: 0, errors: [], duringPause: 0, finals: new Map(), stopped: null };
const ws = new WebSocket("ws://127.0.0.1:8000/ws/live");
let started;
const ready = new Promise((r) => (started = r));
ws.onmessage = (m) => {
  const d = JSON.parse(m.data);
  if (!d.ok) { ev.errors.push(d.error); console.log(ts(), "ERROR", d.error); return; }
  if (phase === "paused" && d.type !== "pong") ev.duringPause++;
  if (d.type === "stream_started") started();
  else if (d.type in ev && typeof ev[d.type] === "number") ev[d.type]++;
  if (d.type === "partial" && ev.partial % 10 === 0) console.log(ts(), `sent=${globalThis.sentSec.toFixed(0)}s partial#${ev.partial} "${d.text.slice(-40)}"`);
  if (d.type === "utterance") {
    if (phase === "paused") ev.finalInPause = true;
    ev.finals.set(d.id, d);
    console.log(ts(), `sent=${globalThis.sentSec.toFixed(0)}s`, `[${phase}] utt ${d.id} pending=${d.pending} ${d.speaker} "${d.text.slice(0, 50)}" ->`, JSON.stringify(d.translations).slice(0, 60), JSON.stringify(d.timings));
  } else if (d.type === "stream_stopped") { ev.stopped = d; console.log(ts(), "stream_stopped", d.finalized); }
};
ws.onclose = (e) => { ev.closed = `${e.code} ${e.reason}`; console.log(ts(), "SOCKET CLOSED", e.code, e.reason); };
await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
ws.send(JSON.stringify({ type: "stream_start", targets: ["vi", "zh"], src_lang: null, denoise: true, terms: {}, display_lang: "vi" }));
await ready;
console.log(ts(), "stream_started; audio", (pcm.length / 16000).toFixed(1), "s");
const F = 8000; globalThis.sentSec = 0;
for (let i = 0; i < pcm.length; i += F) {
  if (phase === "live" && i / 16000 >= PAUSE_AT) {
    phase = "paused"; console.log(ts(), `--- PAUSE ${PAUSE_FOR}s (mic paused: no frames) ---`);
    ws.send(JSON.stringify({ type: "stream_flush" })); // App.tsx togglePause
    await sleep(PAUSE_FOR * 1000);
    console.log(ts(), `--- RESUME; events received while paused: ${ev.duringPause} ---`);
    phase = "resumed";
  }
  globalThis.sentSec = i / 16000;
  ws.send(JSON.stringify({ type: "stream_audio", audio_b64: b64(pcm.subarray(i, i + F)), sr: 16000 }));
  await sleep(500 / SPEED);
}
phase = "stopping";
ws.send(JSON.stringify({ type: "stream_stop" }));
console.log(ts(), "stream_stop sent");
const t1 = Date.now();
while (!ev.stopped && Date.now() - t1 < 320000) await sleep(200);
const stopLatency = (Date.now() - t1) / 1000;
await sleep(1500);
ws.onclose = null; ws.close();

const finals = [...ev.finals.values()];
const garbled = finals.filter((u) => JSON.stringify(u.translations).includes("�"));
const unfinished = finals.filter((u) => u.pending || !u.translations.vi || !u.translations.zh);
console.log(`\npartials=${ev.partial} tok=${ev.tok} utterance-events=${ev.utterance} rows=${finals.length} errors=${ev.errors.length}`);
console.log(`events while paused=${ev.duringPause} final-in-pause=${!!ev.finalInPause} stop->stream_stopped=${stopLatency.toFixed(1)}s unfinished=${unfinished.length} garbled=${garbled.length}`);
const ok = !ev.closed && ev.errors.length === 0 && finals.length > 0 && ev.stopped && unfinished.length === 0
  && (PAUSE_AT * 16000 >= pcm.length || ev.finalInPause) && garbled.length === 0;
console.log(ok ? "LIVE E2E PASSED" : "LIVE E2E FAILED");
process.exit(ok ? 0 : 1);
