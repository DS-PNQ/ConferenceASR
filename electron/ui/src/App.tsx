import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState, Component, type ReactNode } from "react";
import {
  apiHealth,
  apiOcr,
  apiSetDiarizer,
  apiTranscribe,
  apiTranslateText,
  apiWarmup,
  listMics,
  loadArchive,
  loadGlossary,
  saveArchive,
  saveGlossary,
  openMic,
  speak,
  stopSpeaking,
  type ArchivedSession,
  type ArchivedUtterance,
  type DiarizerMode,
  type Health,
  type MicHandle,
  type OcrPage,
  type PartialMsg,
  type Utterance,
  LiveSocket,
} from "./api";

type IconName =
  | "spark"
  | "waveform"
  | "library"
  | "clock"
  | "book"
  | "settings"
  | "plus"
  | "chevron"
  | "mic"
  | "pause"
  | "stop"
  | "more"
  | "translate"
  | "copy"
  | "download"
  | "search"
  | "sliders"
  | "arrow"
  | "check"
  | "wave"
  | "upload"
  | "sun"
  | "moon"
  | "scan"
  | "rotateL"
  | "rotateR"
  | "crop"
  | "zoomIn"
  | "zoomOut"
  | "fit"
  | "play"
  | "speaker"
  | "globe";

function Icon({ name, size = 18, stroke = 1.8 }: { name: IconName; size?: number; stroke?: number }) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: stroke,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };

  const paths: Record<IconName, ReactNode> = {
    spark: <><path d="m12 3-1.4 5.1L5.5 9.5l5.1 1.4L12 16l1.4-5.1 5.1-1.4-5.1-1.4L12 3Z" /><path d="m19 16-.6 2.2L16 19l2.4.8L19 22l.6-2.2L22 19l-2.4-.8L19 16Z" /></>,
    waveform: <path d="M4 10.5v3M8 7v10M12 3.5v17M16 6.5v11M20 10v4" />,
    library: <><rect x="3.5" y="4" width="4.2" height="16" rx="1" /><rect x="7.7" y="6.5" width="4.2" height="13.5" rx="1" /><rect x="13.3" y="4.4" width="4.2" height="15.6" rx="1" transform="rotate(-12 15.4 20)" /><path d="M5.6 7.5v2M9.8 10v2" /></>,
    clock: <><circle cx="12" cy="12" r="8.5" /><path d="M12 7v5l3.4 2" /></>,
    book: <><path d="M6.5 3.5h11.2a.8.8 0 0 1 .8.8v15.4a.8.8 0 0 1-.8.8H6.5a2 2 0 0 1-2-2v-13a2 2 0 0 1 2-2Z" /><path d="M4.5 18.5a2 2 0 0 1 2-2h12" /><path d="m9.2 13 2.3-6 2.3 6M10 11h3" /></>,
    settings: <><path d="M18.96 10.59 20.92 10.83 20.92 13.17 18.96 13.41 17.92 15.92 19.14 17.49 17.49 19.14 15.92 17.92 13.41 18.96 13.17 20.92 10.83 20.92 10.59 18.96 8.08 17.92 6.51 19.14 4.86 17.49 6.08 15.92 5.04 13.41 3.08 13.17 3.08 10.83 5.04 10.59 6.08 8.08 4.86 6.51 6.51 4.86 8.08 6.08 10.59 5.04 10.83 3.08 13.17 3.08 13.41 5.04 15.92 6.08 17.49 4.86 19.14 6.51 17.92 8.08Z" /><circle cx="12" cy="12" r="2.8" /></>,
    plus: <path d="M12 5v14M5 12h14" />,
    chevron: <path d="m7 10 5 5 5-5" />,
    mic: <><rect x="8.5" y="3" width="7" height="12" rx="3.5" /><path d="M5 11.5a7 7 0 0 0 14 0M12 18.5V22M8.5 22h7" /></>,
    pause: <path d="M8.5 5v14M15.5 5v14" />,
    stop: <rect x="6.5" y="6.5" width="11" height="11" rx="1" />,
    more: <><circle cx="5" cy="12" r="1" fill="currentColor" /><circle cx="12" cy="12" r="1" fill="currentColor" /><circle cx="19" cy="12" r="1" fill="currentColor" /></>,
    translate: <><path d="M3.5 6h9M8 3.5V6M5 6c.9 3.2 3 5.6 6 7M11 6c-.9 3.2-3 5.6-6 7" /><path d="m13 20.5 3.75-9 3.75 9M14.3 17.5h4.9" /></>,
    copy: <><rect x="8" y="8" width="11" height="12" rx="1.5" /><path d="M16 8V5.5A1.5 1.5 0 0 0 14.5 4h-9A1.5 1.5 0 0 0 4 5.5v9A1.5 1.5 0 0 0 5.5 16H8" /></>,
    download: <><path d="M12 3v12M7.5 10.5 12 15l4.5-4.5M5 20h14" /></>,
    upload: <><path d="M12 15V3M7.5 7.5 12 3l4.5 4.5M5 20h14" /></>,
    search: <><circle cx="10.8" cy="10.8" r="6.2" /><path d="m16 16 4.2 4.2" /></>,
    sliders: <><path d="M4 6h6M14 6h6M4 12h10M18 12h2M4 18h3M15 18h5" /><circle cx="12" cy="6" r="2" /><circle cx="16" cy="12" r="2" /><circle cx="9" cy="18" r="2" /></>,
    arrow: <><path d="M5 12h14" /><path d="m13 6 6 6-6 6" /></>,
    check: <path d="m5 12 4.3 4.3L19 6.8" />,
    sun: <><circle cx="12" cy="12" r="4" /><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4" /></>,
    moon: <path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5Z" />,
    wave: <><path d="M3 12h2l1.5-5 3 10 2.4-14L14.5 21l2.5-9H21" /></>,
    rotateL: <><path d="M4 4v5h5" /><path d="M5.1 13.5A7 7 0 1 0 6.6 7L4 9" /></>,
    rotateR: <><path d="M20 4v5h-5" /><path d="M18.9 13.5A7 7 0 1 1 17.4 7L20 9" /></>,
    crop: <><path d="M6 2.5V18h15.5" /><path d="M2.5 6H18v15.5" /></>,
    zoomIn: <><circle cx="10.8" cy="10.8" r="6.2" /><path d="m16 16 4.2 4.2M10.8 8.3v5M8.3 10.8h5" /></>,
    zoomOut: <><circle cx="10.8" cy="10.8" r="6.2" /><path d="m16 16 4.2 4.2M8.3 10.8h5" /></>,
    fit: <><path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5" /></>,
    play: <path d="M7.5 5.5v13l11-6.5-11-6.5Z" />,
    speaker: <><path d="M11 5 6.5 9H4v6h2.5l4.5 4V5Z" /><path d="M15.5 9.2a4 4 0 0 1 0 5.6M18.4 6.4a8 8 0 0 1 0 11.2" /></>,
    globe: <><circle cx="12" cy="12" r="8.5" /><path d="M3.5 12h17M12 3.5c2.3 2.4 3.4 5.2 3.4 8.5s-1.1 6.1-3.4 8.5c-2.3-2.4-3.4-5.2-3.4-8.5s1.1-6.1 3.4-8.5Z" /></>,
    scan: <><path d="M4 8.5V6a2 2 0 0 1 2-2h2.5M15.5 4H18a2 2 0 0 1 2 2v2.5M20 15.5V18a2 2 0 0 1-2 2h-2.5M8.5 20H6a2 2 0 0 1-2-2v-2.5" /><path d="M8 9h8M8 12h8M8 15h5" /></>,
  };

  return <svg {...common}>{paths[name]}</svg>;
}

// ---------- live row model ----------
interface Row {
  id: number;
  at: number; // session-relative seconds
  speaker: string;
  text: string;
  translations: Record<string, string>;
  timings?: { asr_ms: number; diar_ms: number; mt_ms: number; total_ms: number };
  pending?: boolean; // true while MT still catching up in background
}

function fmtLatency(t?: Row["timings"]): string | null {
  if (!t) return null;
  const s = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`);
  return `asr ${s(t.asr_ms)} · mt ${s(t.mt_ms)} · diar ${s(t.diar_ms)} · e2e ${s(t.total_ms)}`;
}

const DISPLAY_LANGS = [
  { code: "vi", label: "Vietnamese", native: "Tiếng Việt" },
  { code: "en", label: "English", native: "English" },
  { code: "zh", label: "Chinese", native: "中文" },
];

const LANG_NAMES = new Intl.DisplayNames(["en"], { type: "language" });
const langLabel = (code: string) => { try { return LANG_NAMES.of(code) ?? code; } catch { return code; } };
// lang= lets Chromium pick the right fallback font (CJK glyphs, Vietnamese stacked diacritics)
const htmlLang = (code: string) => (code === "zh" ? "zh-Hans" : code);

/** Listen + Copy under a finished translation. Listen: the backend's Piper voice
 *  for vi/en/zh, the OS voice otherwise; one voice plays at a time app-wide. */
function TrActions({ text, lang }: { text: string; lang: string }) {
  const [say, setSay] = useState<"idle" | "loading" | "playing" | "novoice">("idle");
  const [copied, setCopied] = useState(false);
  const name = langLabel(lang);
  const busy = say === "loading" || say === "playing";
  return <span className="tr-actions" onClick={(e) => e.stopPropagation()}>
    <button className={`tr-act ${say}`} aria-pressed={busy} disabled={say === "novoice"}
      aria-label={busy ? "Stop speaking" : say === "novoice" ? `No ${name} voice on this computer` : `Listen in ${name}`}
      title={busy ? "Stop" : say === "novoice" ? `No ${name} voice on this computer` : `Listen (${name})`}
      onClick={() => {
        if (busy) { stopSpeaking(); setSay("idle"); return; }
        setSay("loading");
        void speak(text, lang, { onStart: () => setSay("playing"), onEnd: () => setSay("idle") })
          .then((ok) => { if (!ok) setSay("novoice"); });
      }}>
      <Icon name={say === "playing" ? "stop" : "speaker"} size={15} />
    </button>
    <button className={`tr-act ${copied ? "done" : ""}`} aria-label={`Copy ${name} translation`} title={copied ? "Copied" : "Copy"}
      onClick={() => {
        void navigator.clipboard?.writeText(text);
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1500);
      }}>
      <Icon name={copied ? "check" : "copy"} size={15} />
    </button>
  </span>;
}

const AVATAR_COLORS = ["plum", "blue", "green", "amber"];

function speakerIdx(sp: string): number {
  const m = /(\d+)/.exec(sp || "");
  return m ? parseInt(m[1], 10) - 1 : 0;
}

function speakerName(sp: string): string {
  const m = /(\d+)/.exec(sp || "");
  return m ? `Speaker ${m[1].padStart(2, "0")}` : sp || "Speaker";
}

function speakerInitials(sp: string): string {
  const m = /(\d+)/.exec(sp || "");
  return m ? `S${parseInt(m[1], 10)}` : "S?";
}

// Display language first, then the rest in backend order (old-UI behavior:
// every requested target is shown, the selected one leads).
function orderedTranslations(entry: { translations: Record<string, string> }, displayLang: string): [string, string][] {
  const pairs = Object.entries(entry.translations);
  pairs.sort((a, b) => (a[0] === displayLang ? -1 : b[0] === displayLang ? 1 : 0));
  return pairs;
}

// Row translation rows = stored translations, with live token-stream text
// overlaid on the display language (present even before stored rows exist).
// Returns [code, text, streaming].
function rowTranslations(
  entry: { translations: Record<string, string> },
  displayLang: string,
  tokText: string | null,
): [string, string, boolean][] {
  const base: Record<string, string> = { ...(entry.translations || {}) };
  if (tokText) base[displayLang] = tokText;
  return orderedTranslations({ translations: base }, displayLang).map(
    ([code, text]) => [code, text, !!tokText && code === displayLang] as [string, string, boolean],
  );
}

// Archives from older builds stored a single `translation` string; normalize
// any shape here so opening them can never blank the view.
function archivedDict(u: { translations?: Record<string, string>; translation?: string }): Record<string, string> {
  if (u.translations && typeof u.translations === "object") return u.translations;
  if (u.translation) return { saved: u.translation };
  return {};
}

function formatTime(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds));
  const hours = Math.floor(s / 3600).toString().padStart(2, "0");
  const minutes = Math.floor((s % 3600) / 60).toString().padStart(2, "0");
  const seconds = (s % 60).toString().padStart(2, "0");
  return `${hours}:${minutes}:${seconds}`;
}

const VIEW_TITLES: Record<string, string> = { "Live session": "Live Session" };

const navigation = [
  { label: "Live session", icon: "waveform" as IconName },
  { label: "Library", icon: "library" as IconName },
  { label: "Glossary", icon: "book" as IconName },
  { label: "Translate", icon: "translate" as IconName },
  { label: "OCR", icon: "scan" as IconName },
];

// Last-resort guard: a crashing view shows a recovery card, never a white screen.
class ViewBoundary extends Component<{ onReset: () => void; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch() {
    /* rendered fallback below */
  }
  render() {
    if (this.state.failed) {
      return <section className="secondary-view wide">
        <h1>This view couldn’t be shown</h1>
        <p>Your live session and saved sessions are intact.</p>
        <button className="btn primary" onClick={() => { this.setState({ failed: false }); this.props.onReset(); }}>
          Back to Live Session
        </button>
      </section>;
    }
    return this.props.children;
  }
}

export default function App() {
  const [activeView, setActiveView] = useState("Live session");
  const [rows, setRows] = useState<Row[]>([]);
  const [partial, setPartial] = useState<PartialMsg | null>(null);
  const [live, setLive] = useState(false);
  const [paused, setPaused] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [micLevel, setMicLevel] = useState(0);
  const [status, setStatus] = useState("Starting…");
  const [health, setHealth] = useState<Health | null>(null);
  const [selectedSegment, setSelectedSegment] = useState<number | null>(null);
  const [showTranslation, setShowTranslation] = useState(true);
  const [displayLang, setDisplayLang] = useState("en");
  const [copied, setCopied] = useState(false);
  const [query, setQuery] = useState("");
  const [targets, setTargets] = useState<string[]>(["vi", "en", "zh"]);
  const [srcChoice, setSrcChoice] = useState("auto");
  const [denoise, setDenoise] = useState(true);
  const [micId, setMicId] = useState<string | undefined>(undefined);
  const [mics, setMics] = useState<{ id: string; label: string }[]>([]);
  const [terms, setTerms] = useState<Record<string, string>>(() => loadGlossary());
  const [archive, setArchive] = useState<ArchivedSession[]>(() => loadArchive());
  const [openArchived, setOpenArchived] = useState<string | null>(null);
  // light/dark: stored choice, else follow the OS
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    try {
      const t = localStorage.getItem("conflive.theme");
      if (t === "light" || t === "dark") return t;
    } catch { /* noop */ }
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    // Electron: recolour the native window buttons drawn over the toolbar
    (window as { conflive?: { setTheme?: (t: string) => void } }).conflive?.setTheme?.(theme);
    try { localStorage.setItem("conflive.theme", theme); } catch { /* noop */ }
  }, [theme]);
  // developer latency readout (per-segment ASR/MT/diar timings); on by default
  const [dev, setDev] = useState(() => {
    try {
      return localStorage.getItem("conflive.dev") !== "0";
    } catch {
      return true;
    }
  });
  // token-streamed translation for the segment currently finalizing
  const [tok, setTok] = useState<{ id: number; text: string } | null>(null);
  const tokAcc = useRef<Record<number, string>>({});

  const socketRef = useRef<LiveSocket | null>(null);
  const micRef = useRef<MicHandle | null>(null);
  const anchorRef = useRef<number | null>(null); // epoch seconds of at=0
  const displayLangRef = useRef(displayLang);
  displayLangRef.current = displayLang;
  const elapsedRef = useRef(0);
  const liveRef = useRef(false);
  const pausedRef = useRef(false);
  // rowsRef eagerly mirrors rows (appendUtterance updates both) so archiving
  // is race-free from timeouts, uploads and unload handlers alike.
  const rowsRef = useRef<Row[]>([]);
  const savedCountRef = useRef(0);
  // latest settings mirrored for async callbacks (reconnect, watchdog)
  const targetsRef = useRef(targets);
  const srcChoiceRef = useRef(srcChoice);
  const denoiseRef = useRef(denoise);
  const termsRef = useRef(terms);
  const micIdRef = useRef(micId);
  const reconnectingRef = useRef(false);
  targetsRef.current = targets;
  srcChoiceRef.current = srcChoice;
  denoiseRef.current = denoise;
  termsRef.current = terms;
  micIdRef.current = micId;
  liveRef.current = live;
  pausedRef.current = paused;
  elapsedRef.current = elapsed;

  // push display/target changes into a live session without resetting it
  useEffect(() => {
    if (liveRef.current && socketRef.current) {
      socketRef.current.config({ displayLang, targets });
    }
  }, [displayLang, targets]);

  // session clock
  useEffect(() => {
    if (!live || paused) return;
    // add real elapsed time, not +1 per tick: a minimized window throttles
    // timers and a +1 counter fell minutes behind over an hour
    let last = Date.now();
    const t = window.setInterval(() => {
      const now = Date.now();
      setElapsed((v) => v + (now - last) / 1000);
      last = now;
    }, 1000);
    return () => window.clearInterval(t);
  }, [live, paused]);

  // mic level meter pump + capture watchdog + backpressure indicator
  useEffect(() => {
    if (!live) {
      setMicLevel(0);
      return;
    }
    let restarts = 0;
    const t = window.setInterval(() => {
      const h = micRef.current;
      setMicLevel(h ? h.level() : 0);
      if (!h || pausedRef.current) return;
      // capture stall: frames stopped arriving -> rebuild the chain once
      if (h.idleFor() > 3 && restarts < 2) {
        restarts++;
        setStatus("Mic stalled — restarting capture…");
        void restartMic().then((ok) => {
          if (ok) setStatus("● Listening — streaming ASR + translation.");
        });
        return;
      }
      // server can't keep up: say so instead of silently lagging
      try {
        const pending = socketRef.current?.pending() ?? 0;
        if (pending > 512 * 1024) setStatus("Catching up… (server busy)");
      } catch {
        /* noop */
      }
    }, 500);
    return () => window.clearInterval(t);
  }, [live]);

  // boot: mics + health + warmup
  useEffect(() => {
    listMics().then(setMics).catch(() => undefined);
    (async () => {
      try {
        const h = await apiHealth();
        setHealth(h);
        setStatus("Loading models (first run downloads weights)…");
        await apiWarmup();
        setHealth(await apiHealth());
        // warmup can finish after Start was pressed: don't clobber "Recording"
        if (!liveRef.current) setStatus("Ready — press Record or New Session.");
      } catch {
        if (!liveRef.current) setStatus("Backend unreachable — launch run.py, then reload.");
      }
    })();
  }, []);

  useEffect(() => saveGlossary(terms), [terms]);

  // one scroller for every view: a new view starts at its top
  useEffect(() => { document.querySelector(".workspace")?.scrollTo(0, 0); }, [activeView, openArchived]);

  useEffect(() => {
    try {
      localStorage.setItem("conflive.dev", dev ? "1" : "0");
    } catch {
      /* noop */
    }
  }, [dev]);

  // flush unarchived rows if the window closes mid-session/upload
  useEffect(() => {
    const flush = () => {
      const all = rowsRef.current;
      if (all.length > savedCountRef.current) {
        const fresh = all.slice(savedCountRef.current);
        const s: ArchivedSession = {
          id: `${Date.now()}-${Math.floor(Math.random() * 1e6)}`,
          label: "Interrupted session",
          startedAt: (anchorRef.current ?? Date.now() / 1000) * 1000,
          seconds: elapsedRef.current,
          utterances: fresh.map((r) => ({
            time: formatTime(r.at),
            speaker: speakerName(r.speaker),
            text: r.text,
            translations: r.translations,
            timings: r.timings,
          })),
        };
        savedCountRef.current = all.length;
        const next = [s, ...loadArchive()].slice(0, 50);
        saveArchive(next);
      }
    };
    window.addEventListener("beforeunload", flush);
    return () => window.removeEventListener("beforeunload", flush);
  }, []);

  function anchorFor(serverStart: number | null): number {
    if (anchorRef.current === null) {
      anchorRef.current = serverStart ?? Date.now() / 1000;
    }
    return anchorRef.current;
  }

  function appendUtterance(u: Utterance) {
    const anchor = anchorFor(u.start);
    const row: Row = {
      id: u.id,
      at: Math.max(0, u.start - anchor),
      speaker: u.speaker,
      text: u.text,
      translations: u.translations,
      timings: u.timings,
      pending: (u as { pending?: boolean }).pending,
    };
    // upsert by id: the server emits each segment twice (instant ASR text,
    // then full translations) — merge in place, never duplicate or reorder.
    const prev = rowsRef.current;
    const i = prev.findIndex((r) => r.id === u.id);
    let next: Row[];
    if (i < 0) next = [...prev, row];
    else {
      next = prev.slice();
      next[i] = { ...next[i], ...row };
    }
    rowsRef.current = next;
    setRows(next);
    setSelectedSegment(u.id);
  }

  // (Re)open the mic for the current session. Returns false on failure.
  // Used by start, the stall watchdog, and mid-session device switches —
  // the WS session survives all of them (only a socket drop re-opens it).
  async function startMicCapture(device: string | undefined): Promise<boolean> {
    try {
      micRef.current?.stop();
    } catch {
      /* noop */
    }
    micRef.current = null;
    try {
      const mic = await openMic(
        device,
        (b64) => socketRef.current?.audio(b64),
        0.5,
        () => {
          // OS yanked the device mid-session: end cleanly, keep the transcript.
          if (liveRef.current) {
            setStatus("Microphone unplugged — session stopped, transcript kept.");
            void stopSession();
          }
        },
      );
      mic.dropNext(0.4); // swallow the open/device pop
      micRef.current = mic;
      return true;
    } catch (e) {
      setStatus("Microphone failed: " + (e as Error).message);
      return false;
    }
  }

  async function restartMic(): Promise<boolean> {
    if (!liveRef.current) return false;
    const ok = await startMicCapture(micIdRef.current);
    if (ok) micRef.current?.dropNext(0.5);
    return ok;
  }

  async function reconnectSocket(): Promise<void> {
    // Unexpected socket drop mid-session: re-open against a fresh server
    // session and keep the mic flowing (a few seconds of audio may land in
    // the new segment — continuity beats perfection here).
    if (!liveRef.current || reconnectingRef.current) return;
    reconnectingRef.current = true;
    setStatus("Connection lost — reconnecting…");
    try {
      const socket = new LiveSocket();
      socketRef.current = socket;
      await socket.connect({
        onPartial: (p) => setPartial(p),
        onTok: onTokHandler,
        onUtterance: onUtteranceHandler,
        onStatus: (s) => {
          if (s === "listening") setStatus("● Reconnected — listening.");
          else setStatus(s);
        },
        onError: (e) => setStatus("Error: " + e),
        onClose: () => void reconnectSocket(),
      });
      socket.start({
        targets: targetsRef.current,
        srcLang: srcChoiceRef.current === "auto" ? null : srcChoiceRef.current,
        denoise: denoiseRef.current,
        terms: termsRef.current,
        displayLang: displayLangRef.current,
        resume: true,
      });
    } catch {
      window.setTimeout(() => {
        reconnectingRef.current = false;
        if (liveRef.current) void reconnectSocket();
      }, 3000);
      return;
    }
    reconnectingRef.current = false;
  }

  function onTokHandler(t: { id: number; tgt: string; seq: number; delta: string }) {
    // live token stream for the display language only; the full
    // translations land with the utterance event right after.
    const want = displayLangRef.current;
    if (t.tgt !== want) return;
    // seq 0 = a new translation of this segment (next partial, or the final)
    tokAcc.current[t.id] = (t.seq === 0 ? "" : tokAcc.current[t.id] ?? "") + t.delta;
    const text = tokAcc.current[t.id];
    setTok((cur) => (cur && cur.id === t.id ? { id: t.id, text } : { id: t.id, text }));
  }

  function onUtteranceHandler(u: Utterance) {
    setPartial((cur) => (cur && cur.id === u.id ? null : cur));
    delete tokAcc.current[u.id];
    setTok((cur) => (cur && cur.id === u.id ? null : cur));
    appendUtterance(u);
  }

  async function startRecording() {
    if (liveRef.current) return;
    setStatus("Opening microphone…");
    const socket = new LiveSocket();
    socketRef.current = socket;
    try {
      await socket.connect({
        onPartial: (p) => setPartial(p),
        onTok: onTokHandler,
        onUtterance: onUtteranceHandler,
        onStatus: (s) => {
          if (s === "listening") setStatus("● Listening — streaming ASR + translation.");
          else setStatus(s);
        },
        onError: (e) => setStatus("Error: " + e),
        onClose: () => void reconnectSocket(),
      });
    } catch (e) {
      setStatus("Could not reach backend: " + (e as Error).message);
      return;
    }
    socket.start({
      targets,
      srcLang: srcChoice === "auto" ? null : srcChoice,
      denoise,
      terms,
      displayLang: displayLangRef.current,
    });
    // The server loads the ASR model during stream_start (seconds on first
    // run). Wait for stream_started before opening the mic so early audio is
    // held back, not rejected with "no stream open".
    setStatus("Starting models — warming up…");
    if (!(await socket.waitForReady())) {
      setStatus("Backend did not answer stream_start — is it still loading models? Try again.");
      socket.close();
      socketRef.current = null;
      return;
    }
    anchorRef.current = Date.now() / 1000;
    if (!(await startMicCapture(micId))) {
      socket.close();
      socketRef.current = null;
      return;
    }
    setStatus(`● Recording — speak in vi / en / zh.`);
    setPartial(null);
    setPaused(false);
    setLive(true);
  }

  function togglePause() {
    if (!liveRef.current) {
      void startRecording();
      return;
    }
    const next = !pausedRef.current;
    micRef.current?.setPaused(next);
    if (next) socketRef.current?.flush();
    setPaused(next);
    setStatus(next ? "Paused — resume to continue the same session." : "● Listening — streaming ASR + translation.");
  }

  function archiveDelta(label: string): ArchivedSession | null {
    // Archive only rows not archived before, so stop/upload/new can each
    // flush without ever duplicating. rowsRef mirrors state for unload flush.
    const all = rowsRef.current;
    const fresh = all.slice(savedCountRef.current);
    if (fresh.length === 0) return null;
    const s: ArchivedSession = {
      id: `${Date.now()}-${Math.floor(Math.random() * 1e6)}`,
      label,
      startedAt: (anchorRef.current ?? Date.now() / 1000) * 1000,
      seconds: elapsedRef.current,
      utterances: fresh.map((r) => ({
        time: formatTime(r.at),
        speaker: speakerName(r.speaker),
        text: r.text,
        translations: r.translations,
        timings: r.timings,
      })),
    };
    savedCountRef.current = all.length;
    setArchive((prev) => {
      const next = [s, ...prev].slice(0, 50);
      saveArchive(next);
      return next;
    });
    return s;
  }

  async function stopSession() {
    if (!liveRef.current) return;
    liveRef.current = false; // no reconnect while the socket winds down
    try {
      micRef.current?.stop();
    } catch {
      /* noop */
    }
    micRef.current = null;
    setLive(false);
    setPaused(false);
    setStatus("Finishing last segment and translations…");
    // The server finalizes the open segment and drains MT before
    // stream_stopped; closing earlier lost the tail and its translations.
    const socket = socketRef.current;
    await socket?.stop();
    socket?.close();
    if (socketRef.current === socket) socketRef.current = null;
    archiveDelta("Live session");
    setPartial(null);
    setStatus("Stopped — session archived to Library.");
  }

  async function newSession() {
    if (liveRef.current) await stopSession();
    else archiveDelta("Live session");
    setRows([]);
    rowsRef.current = [];
    savedCountRef.current = 0;
    setPartial(null);
    setSelectedSegment(null);
    setElapsed(0);
    anchorRef.current = null;
    setActiveView("Live session");
    void startRecording();
  }

  async function uploadFile(f: File) {
    setStatus(`Transcribing ${f.name}…`);
    try {
      const list = await apiTranscribe(f, {
        targets,
        srcLang: srcChoice,
        denoise,
        terms,
      });
      if (anchorRef.current === null && list.length) {
        anchorRef.current = Math.min(...list.map((u) => u.start));
      }
      list.forEach(appendUtterance);
      // rowsRef is updated eagerly, so flush synchronously: uploads never
      // hit stopSession, and without this they would never reach Library.
      const s = archiveDelta(`Upload · ${f.name}`);
      setStatus(s ? `Done — ${list.length} utterances, archived to Library.` : `Done — ${list.length} utterances.`);
    } catch (e) {
      setStatus("Upload failed: " + (e as Error).message);
    }
  }

  function copyTranscript() {
    const text = rows.map((r) => `${speakerName(r.speaker)}: ${r.text}`).join("\n\n");
    void navigator.clipboard?.writeText(text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  function exportTranscript() {
    const text = rows
      .map((r) => {
        const trs = orderedTranslations(r, displayLang)
          .map(([code, t]) => `[${code}] ${t}`)
          .join("\n");
        return `[${formatTime(r.at)}] ${speakerName(r.speaker)}\n${r.text}\n${trs}`;
      })
      .join("\n\n");
    const url = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "conference-transcript.txt";
    a.click();
    URL.revokeObjectURL(url);
  }

  const visibleRows = useMemo(
    () =>
      rows.filter(
        (r) =>
          r.text.toLowerCase().includes(query.toLowerCase()) ||
          speakerName(r.speaker).toLowerCase().includes(query.toLowerCase()),
      ),
    [rows, query],
  );

  const speakerCount = useMemo(() => new Set(rows.map((r) => r.speaker)).size, [rows]);
  const sessionDate = useMemo(() => {
    const base = anchorRef.current ? anchorRef.current * 1000 : Date.now();
    return new Date(base).toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" });
  }, [rows.length === 0]);

  return (
    <main className="app-shell">
      <aside className="sidebar glass">
        <div className="brand-lockup"><AppLogo size={24} /><span>ConfLive</span></div>
        <button className="new-session" onClick={newSession}><Icon name="plus" size={16} stroke={2.2} /><span>New Session</span></button>
        <nav className="primary-nav" aria-label="Main navigation">
          <p className="nav-kicker">Workspace</p>
          {navigation.map((item) => (
            <button className={`nav-item ${activeView === item.label ? "active" : ""}`} key={item.label} aria-current={activeView === item.label ? "page" : undefined}
              onClick={() => { setActiveView(item.label); setOpenArchived(null); }}>
              <Icon name={item.icon} size={20} stroke={1.6} /><span>{VIEW_TITLES[item.label] ?? item.label}</span>{item.label === "Live session" && live && <i className="nav-live-dot" aria-label="recording" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <button className={`nav-item ${activeView === "Settings" ? "active" : ""}`} aria-current={activeView === "Settings" ? "page" : undefined} onClick={() => setActiveView("Settings")}><Icon name="settings" size={20} stroke={1.6} /><span>Settings</span></button>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div className="title-block">
            <h2>{VIEW_TITLES[activeView] ?? activeView}</h2>
            <span>{status}</span>
          </div>
          <div className="topbar-actions">
            <div className="segmented glass" role="radiogroup" aria-label="Show translations in">
              {DISPLAY_LANGS.map((l) => (
                <button key={l.code} role="radio" aria-checked={displayLang === l.code} className={displayLang === l.code ? "on" : ""}
                  lang={htmlLang(l.code)} title={`Show translations in ${l.label}`} onClick={() => setDisplayLang(l.code)}>{l.native}</button>
              ))}
            </div>
            <button className="icon-button glass theme-toggle" aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"} title={theme === "dark" ? "Light mode" : "Dark mode"} onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}><Icon key={theme} name={theme === "dark" ? "sun" : "moon"} size={16} /></button>
          </div>
        </header>

        <ViewBoundary key={activeView} onReset={() => setActiveView("Live session")}>
        {activeView === "Live session" ? (
          <LiveSession
            copied={copied} elapsed={elapsed} exportTranscript={exportTranscript}
            live={live} paused={paused} micLevel={micLevel} micLabel={micRef.current?.label}
            onCopy={copyTranscript} onSelectSegment={setSelectedSegment}
            selectedSegment={selectedSegment} onTogglePause={togglePause} onStop={stopSession}
            onUpload={uploadFile} setQuery={setQuery} showTranslation={showTranslation}
            toggleTranslation={() => setShowTranslation((v) => !v)} dev={dev}
            visibleRows={visibleRows} partial={partial} tok={tok} displayLang={displayLang}
            speakerCount={speakerCount} sessionDate={sessionDate} status={status}
            health={health} targets={targets}
          />
        ) : activeView === "Library" ? (
          <LibraryView archive={archive} openId={openArchived} onOpen={setOpenArchived}
            onDelete={(id) => setArchive((prev) => { const n = prev.filter((s) => s.id !== id); saveArchive(n); return n; })}
            displayLang={displayLang} dev={dev} onNavigate={() => setActiveView("Live session")} />
        ) : activeView === "Glossary" ? (
          <GlossaryView terms={terms} onChange={setTerms} onNavigate={() => setActiveView("Live session")} />
        ) : activeView === "Translate" ? (
          <TranslateView displayLang={displayLang} terms={terms} status={status} setStatus={setStatus} />
        ) : activeView === "OCR" ? (
          <OcrView displayLang={displayLang} terms={terms} setStatus={setStatus} vietOcr={health?.ocr?.vietocr} />
        ) : (
          <SettingsView
            health={health} mics={mics} micId={micId} onMic={(id) => {
              setMicId(id);
              // hot-swap input mid-session: session + transcript survive
              if (liveRef.current) {
                setStatus("Switching microphone…");
                void restartMic().then((ok) => {
                  if (ok) setStatus("● Listening — streaming ASR + translation.");
                });
              }
            }}
            srcChoice={srcChoice} onSrc={setSrcChoice} targets={targets} onTargets={setTargets}
            denoise={denoise} onDenoise={setDenoise} dev={dev} onDev={setDev} status={status}
            onHealth={() => { apiHealth().then(setHealth).catch(() => undefined); }}
            onRefresh={async () => {
              try {
                setStatus("Loading models…");
                await apiWarmup();
                setHealth(await apiHealth());
                setStatus("Ready.");
              } catch (e) { setStatus("Warmup failed: " + (e as Error).message); }
            }}
            onDiarizer={async (mode) => {
              try {
                setStatus(`Switching diarizer to ${mode}…`);
                await apiSetDiarizer(mode);
                setHealth(await apiHealth());
                setStatus(`Diarizer: ${mode}. New sessions use it immediately.`);
              } catch (e) { setStatus("Switch failed: " + (e as Error).message); }
            }}
            onNavigate={() => setActiveView("Live session")}
          />
        )}
        </ViewBoundary>
      </section>
    </main>
  );
}

function LiveSession(props: {
  copied: boolean; elapsed: number; exportTranscript: () => void;
  live: boolean; paused: boolean; micLevel: number; micLabel?: string;
  onCopy: () => void; onSelectSegment: (id: number) => void;
  selectedSegment: number | null; onTogglePause: () => void; onStop: () => void;
  onUpload: (f: File) => void; setQuery: (v: string) => void;
  showTranslation: boolean; toggleTranslation: () => void;
  dev: boolean;
  visibleRows: Row[]; partial: PartialMsg | null;
  tok: { id: number; text: string } | null; displayLang: string;
  speakerCount: number; sessionDate: string; status: string;
  health: Health | null; targets: string[];
}) {
  const active = props.live && !props.paused;
  const fileRef = useRef<HTMLInputElement>(null);
  // auto-follow: stick to the newest row/partial while live; user scrolling
  // up pauses it (jump pill re-engages). `stuck` shows the mini control bar.
  const [follow, setFollow] = useState(true);
  const [stuck, setStuck] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ws = document.querySelector(".workspace");
    if (!ws) return;
    const onScroll = () => {
      const el = ws as HTMLElement;
      setStuck(el.scrollTop > 220);
      if (el.scrollHeight - el.scrollTop - el.clientHeight > 160) setFollow(false);
    };
    ws.addEventListener("scroll", onScroll, { passive: true });
    return () => ws.removeEventListener("scroll", onScroll);
  }, []);

  // jump on new rows / newly selected segment / partial id — not on every
  // token delta (position is already correct then; re-scrolling janks).
  const liveId = props.partial?.id ?? null;
  useEffect(() => {
    if (!follow) return;
    bottomRef.current?.scrollIntoView({ behavior: "auto", block: "end" });
  }, [props.visibleRows.length, props.selectedSegment, liveId, follow]);

  function jumpToLive() {
    setFollow(true);
    requestAnimationFrame(() => bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }));
  }

  function pickSegment(id: number) {
    setFollow(false); // reading history — stop pulling them back down
    props.onSelectSegment(id);
  }

  // (re)starting a session re-engages follow — they pressed Start to watch live
  const wasLive = useRef(props.live);
  useEffect(() => {
    if (props.live && !wasLive.current) jumpToLive();
    wasLive.current = props.live;
  }, [props.live]);

  return <>
    {stuck && (
      <div className="mini-stage glass" aria-label="Recording controls (compact)">
        <i className={active ? "recording-dot" : "idle-dot"} />
        <span className="mini-timer">{formatTime(props.elapsed)}</span>
        <span className="mini-status">{props.status}</span>
        <button className="mini-btn" onClick={props.onTogglePause} aria-label={active ? "Pause" : props.live ? "Resume" : "Start recording"}>
          <Icon name={active ? "pause" : "mic"} size={14} />
        </button>
        <button className="mini-btn" onClick={props.onStop} disabled={!props.live} aria-label="Stop recording">
          <Icon name="stop" size={13} />
        </button>
      </div>
    )}
    <section className="session-heading">
      <div>
        <h1>Conference session</h1>
        <p>{props.sessionDate}<span />{props.speakerCount} speaker{props.speakerCount === 1 ? "" : "s"}<span />{props.visibleRows.length} segment{props.visibleRows.length === 1 ? "" : "s"}</p>
      </div>
      <div className="session-actions">
        <button className="text-button" onClick={props.onCopy}><Icon name={props.copied ? "check" : "copy"} size={15} />{props.copied ? "Copied" : "Copy"}</button>
        <button className="text-button" onClick={() => fileRef.current?.click()}><Icon name="upload" size={15} />Upload</button>
        <input ref={fileRef} type="file" accept="audio/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) props.onUpload(f); e.target.value = ""; }} />
        <button className="text-button" onClick={props.exportTranscript}><Icon name="download" size={15} />Export</button>
      </div>
    </section>

    <section className={`recording-stage ${active ? "is-live" : ""}`} aria-label="Recording controls">
      <button className="record-core" onClick={props.onTogglePause}
        aria-label={active ? "Pause recording" : props.live ? "Resume recording" : "Start recording"}>
        <Icon name={active ? "pause" : "mic"} size={22} stroke={2} />
      </button>
      <div className="stage-clock">
        <div className="timer">{formatTime(props.elapsed)}</div>
        <div className="stage-meta"><i className={active ? "recording-dot" : "idle-dot"} />{active ? "Recording" : props.live ? "Paused" : "Idle"} · {props.micLabel ?? "Default input"}</div>
      </div>
      <div className="waveform" aria-hidden="true"><WaveBars level={props.micLevel} live={active} /></div>
      <div className="stage-controls">
        <button className="btn stop-button" onClick={props.onStop} disabled={!props.live} title="Stop and save to Library"><Icon name="stop" size={13} />Stop</button>
      </div>
    </section>

    <section className="transcript-toolbar">
      <div className="toolbar-title"><span>Transcript</span>{props.live && <b>Live</b>}</div>
      <div className="toolbar-actions">
        <label className="switch-wrap"><span>Translations</span><input type="checkbox" className="toggle" checked={props.showTranslation} onChange={props.toggleTranslation} /></label>
        <label className="search-box"><Icon name="search" size={15} /><input placeholder="Search" aria-label="Search transcript" onChange={(e) => props.setQuery(e.target.value)} /></label>
      </div>
    </section>

    <section className="transcript-area">
      <div className="transcript-list">
        {props.visibleRows.map((entry) => (
          <TranscriptRow key={entry.id} entry={entry} selected={props.selectedSegment === entry.id}
            onPick={pickSegment} showTranslation={props.showTranslation} displayLang={props.displayLang}
            tokText={props.tok && props.tok.id === entry.id ? props.tok.text : null} dev={props.dev} />
        ))}
        {props.partial && props.partial.text && (
          <div className="live-caption">
            <span className="caption-pulse" />
            <div className="caption-body">
              <p>{props.partial.text}<span className="typing-caret inline" /></p>
              {props.showTranslation && rowTranslations(
                { translations: props.partial.translations || {} },
                props.displayLang,
                props.tok && props.tok.id === props.partial.id ? props.tok.text : null,
              ).map(([code, text, streaming]) => {
                if (!text) return null;
                return (
                  <p className="translated-text live-tr" key={code} lang={htmlLang(code)}>
                    <b className="tr-lang">{code}</b>
                    {text}
                    {streaming ? <span className="typing-caret inline" /> : null}
                  </p>
                );
              })}
            </div>
          </div>
        )}
        {props.visibleRows.length === 0 && !props.partial && <div className="empty-search">
          <Icon name="wave" size={30} stroke={1.5} />
          <b>{props.live ? "Listening…" : "No transcript yet"}</b>
          <span>{props.live ? "Speak in Vietnamese, English or Chinese." : "Press Record, or Upload an audio file."}</span>
        </div>}
        <div ref={bottomRef} aria-hidden="true" />
        {!follow && props.live && (
          <button className="jump-live glass" onClick={jumpToLive}>↓ Jump to live</button>
        )}
      </div>
      <aside className="insight-panel">
        <div className="insight-heading">Session</div>
        <div className="stat-grid">
          <div><b>{formatTime(props.elapsed)}</b><span>Duration</span></div>
          <div><b>{props.visibleRows.length}</b><span>Segments</span></div>
          <div><b>{props.speakerCount}</b><span>Speakers</span></div>
          <div><b>{props.targets.map((t) => t.toUpperCase()).join(" · ")}</b><span>Targets</span></div>
        </div>
        <div className="insight-heading">Engines</div>
        <dl className="insight-list">
          <dt>Speech</dt><dd>Zipformer zh-en-vi · CPU int8</dd>
          <dt>Translation</dt><dd>Hy-MT2 1.8B FP8{props.health ? ` · ${props.health.device.device.toUpperCase()}` : ""}</dd>
          <dt>Speakers</dt><dd>{props.health ? backendLabel(props.health.diarizer_status) : "volume"}</dd>
          <dt>Voices</dt><dd>{props.health?.tts?.langs.length ? `Piper · ${props.health.tts.langs.map((l) => l.toUpperCase()).join(" ")}` : "System voice"}</dd>
        </dl>
      </aside>
    </section>
  </>;
}

const TranscriptRow = memo(function TranscriptRow({ entry, selected, onPick, showTranslation, displayLang, tokText, dev }: {
  entry: Row; selected: boolean; onPick: (id: number) => void; showTranslation: boolean;
  displayLang: string; tokText: string | null; dev: boolean;
}) {
  // div, not button: the Listen/Copy buttons inside can't nest in a button
  return (
    <div className={`transcript-row ${selected ? "selected" : ""}`} onClick={() => onPick(entry.id)} data-row-id={entry.id}>
      <time>{formatTime(entry.at)}</time>
      <span className={`speaker-avatar ${AVATAR_COLORS[speakerIdx(entry.speaker) % AVATAR_COLORS.length]}`}>{speakerInitials(entry.speaker)}</span>
      <div className="entry-copy">
        <span className="speaker-line"><b>{speakerName(entry.speaker)}</b></span>
        <p className="original-text">{entry.text}</p>
        {showTranslation && rowTranslations(entry, displayLang, tokText).map(([code, text, streaming]) => {
          if (!text) return null;
          return (
            <div className="translated-text" key={code} lang={htmlLang(code)}>
              <b className="tr-lang">{code}</b>
              <p>{text}{streaming ? <span className="typing-caret inline" /> : null}</p>
              {!streaming && <TrActions text={text} lang={code} />}
            </div>
          );
        })}
        {entry.pending ? <span className="translating">Translating…</span> : null}
        {dev && !entry.pending && fmtLatency(entry.timings) ? <span className="latency">{fmtLatency(entry.timings)}</span> : null}
      </div>
    </div>
  );
}, (a, b) => a.entry === b.entry && a.selected === b.selected && a.showTranslation === b.showTranslation
  && a.displayLang === b.displayLang && a.tokText === b.tokText && a.dev === b.dev); // onPick: stable behavior, new identity each render

function backendLabel(d: Health["diarizer_status"]): string {
  if (!d) return "volume";
  if (d.backend === "nemo-titanet") return `NeMo Titanet${d.device ? ` (${d.device})` : ""}`;
  if (d.backend === "nemotron") return `Nemotron streaming${d.latency_ms ? ` (${d.latency_ms} ms)` : ""}`;
  return d.backend;
}

function WaveBars({ level, live }: { level: number; live: boolean }) {
  const base = [18, 28, 40, 24, 55, 74, 52, 85, 44, 64, 78, 48, 31, 67, 52, 88, 70, 42, 60, 32, 75, 48, 23, 38, 56, 31, 21];
  return <>
    {base.map((h, i) => {
      const scaled = live ? Math.min(100, h * (0.35 + level * 2.4)) : h * 0.45;
      return <span key={i} className={live ? "active-bar" : ""} style={{ height: `${scaled}%`, animationDelay: `${i * 45}ms` }} />;
    })}
  </>;
}

/** Wrap case-insensitive matches of `needle` (already lowercased) in <mark>. */
function mark(text: string, needle: string): ReactNode {
  if (!needle) return text;
  const out: ReactNode[] = [];
  const low = text.toLowerCase();
  let i = 0;
  for (let j = low.indexOf(needle); j >= 0; j = low.indexOf(needle, i)) {
    out.push(text.slice(i, j), <mark key={j}>{text.slice(j, j + needle.length)}</mark>);
    i = j + needle.length;
  }
  out.push(text.slice(i));
  return out;
}

/** Same mark as electron/build/icon.png (scripts/make_icon.py), 1024 -> 100 units. */
function AppLogo({ size = 28 }: { size?: number }) {
  return <svg className="brand-mark" width={size} height={size} viewBox="0 0 100 100" aria-hidden="true">
    <defs><linearGradient id="logo-g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#474747" /><stop offset="1" stopColor="#141414" /></linearGradient></defs>
    <rect x="4" y="4" width="92" height="92" rx="22.5" fill="url(#logo-g)" />
    {[21.5, 39, 56.6, 39, 21.5].map((h, i) => <rect key={i} x={26.6 + i * 9.8} y={50 - h / 2} width="7" height={h} rx="3.5" fill="#fff" />)}
  </svg>;
}

function LibraryView(props: {
  archive: ArchivedSession[]; openId: string | null; onOpen: (id: string | null) => void;
  onDelete: (id: string) => void; displayLang: string; dev: boolean; onNavigate: () => void;
}) {
  const [q, setQ] = useState("");
  const needle = q.trim().toLowerCase();
  const hits = (u: ArchivedUtterance) => !needle ||
    [u.text, u.speaker, ...Object.values(archivedDict(u))].some((s) => (s || "").toLowerCase().includes(needle));
  const searchBox = <label className="search-box library-search">
    <Icon name="search" size={15} />
    <input placeholder="Search all sessions" aria-label="Search all sessions" value={q} onChange={(e) => setQ(e.target.value)} autoFocus />
    {q ? <button className="gt-clear" onClick={() => setQ("")} aria-label="Clear search">✕</button> : null}
  </label>;
  const open = props.archive.find((s) => s.id === props.openId) ?? null;
  if (open) {
    const rows = open.utterances.filter(hits);
    return <section className="secondary-view wide" key={open.id}>
      <button className="back-link" onClick={() => props.onOpen(null)}><Icon name="chevron" size={15} />Library</button>
      <h1>{open.label || "Live session"}</h1>
      <p>{new Date(open.startedAt).toLocaleString()} · {formatTime(open.seconds)} · {needle ? `${rows.length} of ${open.utterances.length} segments match “${q.trim()}”` : `${open.utterances.length} segments`}</p>
      <div className="archive-actions">
        {searchBox}
        <button className="btn destructive" onClick={() => { props.onDelete(open.id); props.onOpen(null); }}>Delete Session</button>
      </div>
      <div className="transcript-list archive-detail">
        {rows.map((u, i) => (
          <div className="transcript-row" key={i}>
            <time>{u.time}</time>
            <span className={`speaker-avatar ${AVATAR_COLORS[speakerIdx(u.speaker) % AVATAR_COLORS.length]}`}>{(u.speaker || "?").replace("Speaker ", "S").replace(/^S0/, "S")}</span>
            <div className="entry-copy">
              <span className="speaker-line"><b>{u.speaker || "Speaker"}</b></span>
              <p className="original-text">{mark(u.text || "", needle)}</p>
              {orderedTranslations({ translations: archivedDict(u) }, props.displayLang).map(([code, text]) => (
                <div className="translated-text" key={code} lang={htmlLang(code)}>
                  <b className="tr-lang">{code}</b>
                  <p>{mark(text, needle)}</p>
                  {text && <TrActions text={text} lang={code} />}
                </div>
              ))}
              {props.dev && fmtLatency(u.timings) ? (
                <span className="latency">{fmtLatency(u.timings)}</span>
              ) : null}
            </div>
          </div>
        ))}
      </div>
    </section>;
  }
  return <section className="secondary-view wide" key="library">
    <h1>Library</h1>
    <p>Sessions are saved on this computer when you stop recording or finish an upload.</p>
    {props.archive.length > 0 && searchBox}
    <div className="archive-list">
      {props.archive.length === 0 && <div className="empty-state">
        <Icon name="clock" size={28} stroke={1.5} />
        <b>No saved sessions</b>
        <span>Stop a recording to save it here.</span>
        <button className="btn primary" onClick={props.onNavigate}>Open Live Session</button>
      </div>}
      {props.archive.map((s) => {
        const n = needle ? s.utterances.filter(hits).length : 0;
        if (needle && !n && !(s.label || "").toLowerCase().includes(needle)) return null;
        return <button className="archive-row" key={s.id} onClick={() => props.onOpen(s.id)}>
          <span className="archive-icon"><Icon name="wave" size={16} /></span>
          <span className="archive-text">
            <span className="archive-title">{s.label || "Live session"}</span>
            <span className="archive-meta">{new Date(s.startedAt).toLocaleString()} · {needle ? `${n} match${n === 1 ? "" : "es"} · ` : ""}{s.utterances.length} segments · {formatTime(s.seconds)}</span>
          </span>
          <Icon name="chevron" size={15} />
        </button>;
      })}
      {needle && props.archive.every((s) => !s.utterances.some(hits) && !(s.label || "").toLowerCase().includes(needle))
        && <p className="archive-empty">No session mentions “{q.trim()}”.</p>}
    </div>
  </section>;
}

function GlossaryView(props: {
  terms: Record<string, string>; onChange: (t: Record<string, string>) => void; onNavigate: () => void;
}) {
  const [src, setSrc] = useState("");
  const [tgt, setTgt] = useState("");
  const entries = Object.entries(props.terms);
  const add = () => {
    if (!src.trim() || !tgt.trim()) return;
    props.onChange({ ...props.terms, [src.trim()]: tgt.trim() });
    setSrc("");
    setTgt("");
  };
  return <section className="secondary-view wide">
    <h1>Glossary</h1>
    <p>Each pair is sent with every translation, so product names and technical terms come out the way you want them.</p>
    <div className="glossary-add">
      <input placeholder="Term as spoken" aria-label="Term as spoken" value={src} onChange={(e) => setSrc(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") add(); }} />
      <Icon name="arrow" size={15} />
      <input placeholder="Preferred translation" aria-label="Preferred translation" value={tgt} onChange={(e) => setTgt(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") add(); }} />
      <button className="btn primary" onClick={add} disabled={!src.trim() || !tgt.trim()}><Icon name="plus" size={14} stroke={2.2} />Add</button>
    </div>
    <div className="glossary-list">
      {entries.length === 0 && <p className="archive-empty">No terms yet. Example: “standup” → “站会”.</p>}
      {entries.map(([s, t]) => (
        <div className="glossary-row" key={s}>
          <span className="glossary-src">{s}</span>
          <Icon name="arrow" size={14} />
          <span className="glossary-tgt">{t}</span>
          <button className="glossary-del" aria-label={`Remove ${s}`} onClick={() => {
            const n = { ...props.terms };
            delete n[s];
            props.onChange(n);
          }}>Remove</button>
        </div>
      ))}
    </div>
  </section>;
}

const EXTRA_LANGS = ["fr", "de", "ja", "ko", "es", "pt", "ru", "th", "ar", "it"];

function TranslateView(props: {
  displayLang: string; terms: Record<string, string>;
  status: string; setStatus: (s: string) => void;
}) {
  const [text, setText] = useState("");
  const [src, setSrc] = useState("auto");
  const [tgt, setTgt] = useState(props.displayLang);
  const [out, setOut] = useState("");
  const [busy, setBusy] = useState(false);
  const termCount = Object.keys(props.terms).length;

  async function go(override?: { text?: string; src?: string; tgt?: string }) {
    const t = (override?.text ?? text).trim();
    const s = override?.src ?? src;
    const g = override?.tgt ?? tgt;
    if (!t || busy) return;
    setBusy(true);
    props.setStatus("Translating phrase…");
    try {
      const r = await apiTranslateText(t, g, props.terms, s);
      setOut(r);
      props.setStatus("Ready.");
    } catch (e) {
      props.setStatus("Translate failed: " + (e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function swap() {
    if (src === "auto" || !out) return;
    const oldSrc = src;
    setSrc(tgt);
    setTgt(oldSrc);
    setText(out);
    setOut("");
    void go({ text: out, src: tgt, tgt: oldSrc });
  }

  const mainLangs = ["en", "zh", "vi"];
  return <section className="secondary-view wide">
    <h1>Translate</h1>
    <p>Same Hy-MT2 engine as the live session{termCount ? `, with ${termCount} glossary term${termCount === 1 ? "" : "s"} applied` : ""}. Press Return to translate, Shift-Return for a new line.</p>
    <div className="gt-bar">
      <div className="gt-tabs">
        {[["auto", "Detect language"], ...mainLangs.map((c) => [c, DISPLAY_LANGS.find((l) => l.code === c)?.label ?? c])].map(([code, label]) => (
          <button key={code} className={src === code ? "active" : ""} aria-pressed={src === code} onClick={() => setSrc(code)}>{label}</button>
        ))}
      </div>
      <button className="gt-swap" onClick={swap} disabled={src === "auto" || !out} aria-label="Swap languages" title="Swap languages">
        <Icon name="arrow" size={16} />
      </button>
      <div className="gt-tabs">
        {mainLangs.map((c) => (
          <button key={c} className={tgt === c ? "active" : ""} aria-pressed={tgt === c} onClick={() => setTgt(c)}>
            {DISPLAY_LANGS.find((l) => l.code === c)?.label ?? c}
          </button>
        ))}
        <select value={EXTRA_LANGS.includes(tgt) ? tgt : ""} onChange={(e) => { if (e.target.value) setTgt(e.target.value); }} aria-label="More languages">
          <option value="">{EXTRA_LANGS.includes(tgt) ? tgt.toUpperCase() : "More…"}</option>
          {EXTRA_LANGS.map((c) => <option key={c} value={c}>{c.toUpperCase()}</option>)}
        </select>
      </div>
    </div>
    <div className={"gt-grid" + (text.length > 160 || out.length > 160 ? " long" : "")}>
      <div className="gt-card">
        <textarea
          placeholder="Type or paste text of any length…"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void go(); } }}
        />
        <div className="gt-card-foot">
          <span>{text.trim() ? `${text.trim().split(/\s+/).length} words · ${text.trim().length} chars` : ""}</span>
          {text ? <button className="gt-clear" aria-label="Clear text" onClick={() => { setText(""); setOut(""); }}>✕</button> : null}
        </div>
      </div>
      <div className="gt-card result">
        {busy ? <span className="gt-thinking">Translating…</span>
          : out ? <p key={out} lang={htmlLang(tgt)}>{out}</p>
          : <span className="gt-placeholder">Translation</span>}
        {out && !busy ? (
          <div className="gt-card-foot">
            <span className="gt-terms-note">{termCount ? `glossary: ${termCount} terms` : ""}</span>
            <TrActions text={out} lang={tgt} />
          </div>
        ) : null}
      </div>
    </div>
  </section>;
}

const ZOOM_MIN = 0.25;
const ZOOM_MAX = 5;
const ZOOM_STEP = 1.25;
const CJK_RE = /[　-〿぀-ヿ㐀-鿿가-힯＀-￯]/;

// lang= on recognised text lets Chromium pick the right fallback font
// (CJK glyph variants, Vietnamese stacked diacritics)
function textLang(text: string, src: string): string | undefined {
  if (CJK_RE.test(text)) return "zh-Hans";
  return src === "auto" ? undefined : src;
}

function OcrView(props: {
  displayLang: string; terms: Record<string, string>;
  setStatus: (s: string) => void; vietOcr?: boolean;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [srcUrl, setSrcUrl] = useState<string | null>(null);
  const [task, setTask] = useState("ocr");
  const [wantTr, setWantTr] = useState(true);
  const [tgt, setTgt] = useState(props.displayLang);
  const [srcLang, setSrcLang] = useState("auto");
  const [rotate, setRotate] = useState(0);
  const [zoom, setZoom] = useState(1); // 1 = fit width
  const [cropMode, setCropMode] = useState(false);
  const [crop, setCrop] = useState<[number, number, number, number] | null>(null);
  const [busy, setBusy] = useState(false);
  const [pages, setPages] = useState<OcrPage[]>([]);
  const [sel, setSel] = useState(0);
  const [selBlock, setSelBlock] = useState<number | null>(null);
  const [tab, setTab] = useState<"text" | "regions" | "quality" | "export">("text");
  const [trace, setTrace] = useState<string[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);
  const viewRef = useRef<HTMLDivElement>(null);
  const previewRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<{ x0: number; y0: number } | null>(null);
  const panRef = useRef<{ x: number; y: number; l: number; t: number } | null>(null);
  const zoomRef = useRef(zoom);
  const anchorRef = useRef<{ ux: number; uy: number; px: number; py: number } | null>(null);
  zoomRef.current = zoom;

  const page = pages[sel] ?? null;
  const blocks = page?.blocks ?? [];
  const lowConf = blocks.filter((b) => b.conf < 0.85);
  const review = blocks.filter((b) => b.conf < 0.6);
  const overall = pages.length
    ? pages.reduce((a, p) => a + (p.overall_conf ?? 0), 0) / pages.length : 0;
  // Before a run: the picked image with live rotate/crop. After: the
  // backend's preview, already rotated + cropped, so no CSS rotate on it.
  const showing = page?.preview ?? srcUrl;
  const isSource = !page?.preview && !!srcUrl;
  const canCrop = !!srcUrl && rotate === 0;

  useEffect(() => () => { if (srcUrl) URL.revokeObjectURL(srcUrl); }, [srcUrl]);

  function stamp(msg: string) {
    const t = new Date().toTimeString().slice(0, 8);
    setTrace((prev) => [...prev.slice(-60), `${t}  ${msg}`]);
  }

  function pick(fs: File[]) {
    setFiles(fs);
    setPages([]);
    setSel(0);
    setSelBlock(null);
    setCrop(null);
    setRotate(0);
    setZoom(1);
    const img = fs.find((f) => f.type.startsWith("image/"));
    setSrcUrl(img ? URL.createObjectURL(img) : null);
    if (fs.length) stamp(`Document loaded — ${fs.map((f) => f.name).join(", ")}`);
  }

  // Rotate/crop edit the source; old results would no longer match it.
  function backToSource() {
    if (pages.length) { setPages([]); setSelBlock(null); stamp("Results cleared — edit the page, then Run OCR again"); }
  }

  function turn(deg: number) {
    backToSource();
    setRotate((r) => (r + deg + 360) % 360);
    if (crop) { setCrop(null); stamp("Crop cleared (rotation resets the crop area)"); }
  }

  // Zoom keeping the point under (cx, cy) — viewport coords — still.
  function zoomTo(next: number, cx?: number, cy?: number) {
    const v = viewRef.current;
    const z0 = zoomRef.current;
    const z = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, +next.toFixed(3)));
    if (!v || z === z0) return;
    const r = v.getBoundingClientRect();
    const px = (cx ?? r.left + r.width / 2) - r.left;
    const py = (cy ?? r.top + r.height / 2) - r.top;
    // anchor in zoom-1 units; ticks landing before the next render keep it
    const c = previewRef.current;
    anchorRef.current ??= { ux: (v.scrollLeft + px - (c?.offsetLeft ?? 0)) / z0,
                            uy: (v.scrollTop + py - (c?.offsetTop ?? 0)) / z0, px, py };
    zoomRef.current = z;
    setZoom(z);
  }

  useLayoutEffect(() => {
    const a = anchorRef.current;
    const v = viewRef.current;
    if (!a || !v) return;
    anchorRef.current = null;
    const c = previewRef.current;
    v.scrollLeft = a.ux * zoom + (c?.offsetLeft ?? 0) - a.px;
    v.scrollTop = a.uy * zoom + (c?.offsetTop ?? 0) - a.py;
  }, [zoom]);

  // Ctrl/⌘ + wheel and touchpad pinch (Chromium sends it as ctrl+wheel).
  // Needs a non-passive listener, which React's onWheel is not.
  useEffect(() => {
    const v = viewRef.current;
    if (!v) return;
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      zoomTo(zoomRef.current * Math.exp(-e.deltaY * 0.0025), e.clientX, e.clientY);
    };
    v.addEventListener("wheel", onWheel, { passive: false });
    return () => v.removeEventListener("wheel", onWheel);
  }, [showing]);

  function onViewKey(e: React.KeyboardEvent) {
    if (e.key === "+" || e.key === "=") zoomTo(zoomRef.current * ZOOM_STEP);
    else if (e.key === "-" || e.key === "_") zoomTo(zoomRef.current / ZOOM_STEP);
    else if (e.key === "0") zoomTo(1);
    else return;
    e.preventDefault();
  }

  async function run() {
    if (!files.length || busy) return;
    setBusy(true);
    const t0 = performance.now();
    stamp(`Run OCR — task=${task}, rotate=${rotate}°, crop=${crop ? "yes" : "no"}, translate=${wantTr ? tgt : "off"}`);
    props.setStatus("OCR reading pages (diarizer deloaded)…");
    try {
      const r = await apiOcr(files, {
        task, translateTo: wantTr ? tgt : "", src: srcLang, terms: props.terms,
        rotate, crop,
      });
      setPages(r.pages);
      setSel(0);
      setSelBlock(null);
      setCropMode(false);
      const n = r.pages.reduce((a, p) => a + (p.blocks?.length ?? 0), 0);
      const c = r.pages.length ? r.pages.reduce((a, p) => a + (p.overall_conf ?? 0), 0) / r.pages.length : 0;
      const errs = r.pages.filter((p) => p.error);
      stamp(`OCR completed — ${r.pages.length} page(s), ${n} blocks, ${Math.round(c * 100)}% overall confidence`);
      errs.forEach((p) => stamp(`Page ${p.page} of ${p.file}: ${p.error}`));
      props.setStatus(`OCR done — ${r.pages.length} page(s), ${n} blocks${errs.length ? `, ${errs.length} failed` : ""}.`);
    } catch (e) {
      stamp(`OCR failed — ${(e as Error).message}`);
      props.setStatus("OCR failed: " + (e as Error).message);
    } finally {
      stamp(`Job finished in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
      setBusy(false);
    }
  }

  function focusBlock(id: number) {
    setSelBlock(id);
    document.querySelector(`[data-ocrblock="${id}"]`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function relOf(e: React.MouseEvent): [number, number] {
    const el = previewRef.current!.getBoundingClientRect();
    return [
      Math.max(0, Math.min(1, (e.clientX - el.left) / el.width)),
      Math.max(0, Math.min(1, (e.clientY - el.top) / el.height)),
    ];
  }

  function download(name: string, text: string, mime: string) {
    const url = URL.createObjectURL(new Blob([text], { type: mime }));
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
    stamp(`Exported ${name}`);
  }

  function exportTxt(): string {
    return pages.map((p) => `=== ${p.file} · page ${p.page} ===\n${p.text ?? ""}`).join("\n\n");
  }

  async function copyPage() {
    if (!page?.text) return;
    await navigator.clipboard.writeText(page.translation ? `${page.text}\n\n${page.translation}` : page.text);
    stamp(`Copied page ${page.page} text`);
  }

  const zoomPct = Math.round(zoom * 100);
  const runLabel = busy ? "Reading…" : pages.length ? "Run again" : "Run OCR";

  return <section className="secondary-view wide ocr-scope">
    <header className="ocr-head">
      <h1>Document OCR</h1>
      <p>Images and PDFs, Vietnamese · English · Chinese. OCR mode switches speaker diarization off and frees its VRAM; optionally runs every block through Hy-MT2{Object.keys(props.terms).length ? " with your glossary applied" : ""}.</p>
      <p className={`ocr-vi ${props.vietOcr ? "ok" : ""}`}>
        {props.vietOcr === undefined ? "Checking Vietnamese reader…"
          : props.vietOcr ? <>Vietnamese reader: <b>VietOCR</b> — diacritics kept, with regions</>
          : <>Vietnamese reader not installed — Vietnamese loses its diacritics. Add it (45 MB): <code>python scripts/get_vietocr.py</code></>}
      </p>
    </header>

    <div className="ocr-toolbar" role="toolbar" aria-label="Document">
      <button className="ocr-tool" onClick={() => fileRef.current?.click()}><Icon name="plus" size={16} />Add document</button>
      <input ref={fileRef} type="file" accept="image/*,.pdf" multiple hidden onChange={(e) => { pick(Array.from(e.target.files ?? [])); e.target.value = ""; }} />
      {files.length > 0 && <span className="ocr-file" title={files.map((f) => f.name).join(", ")}>{files.length === 1 ? files[0].name : `${files.length} files`}</span>}
      <span className="ocr-sep" />
      <button className="ocr-tool icon" onClick={() => turn(-90)} disabled={!files.length} title="Rotate left" aria-label="Rotate left"><Icon name="rotateL" size={17} /></button>
      <button className="ocr-tool icon" onClick={() => turn(90)} disabled={!files.length} title="Rotate right" aria-label="Rotate right"><Icon name="rotateR" size={17} /></button>
      <button className={`ocr-tool ${cropMode ? "on" : ""}`} disabled={!canCrop} aria-pressed={cropMode}
        title={!srcUrl ? "Crop needs an image (PDF pages: crop after export)" : rotate !== 0 ? "Reset rotation to 0° to crop" : "Drag a region on the page"}
        onClick={() => { backToSource(); setCropMode((v) => !v); setCrop(null); }}><Icon name="crop" size={16} />{cropMode ? "Cropping" : "Crop"}</button>
      <label className="ocr-lang"><Icon name="globe" size={16} /><select value={srcLang} onChange={(e) => setSrcLang(e.target.value)} aria-label="Document language">
        <option value="auto">Auto-detect</option>
        <option value="vi">Vietnamese</option>
        <option value="en">English</option>
        <option value="zh">Chinese</option>
      </select></label>
      <button className="ocr-run" onClick={() => void run()} disabled={!files.length || busy}><Icon name="play" size={14} />{runLabel}</button>
    </div>

    <div className="gt-bar">
      <div className="gt-tabs" role="group" aria-label="Recognition task">
        {[["ocr", "Text"], ["table", "Table"], ["formula", "Formula"], ["chart", "Chart"]].map(([code, label]) => (
          <button key={code} className={task === code ? "active" : ""} aria-pressed={task === code} onClick={() => setTask(code)} title={code === "ocr" ? "Text with regions + confidence" : "VLM read (text only, no regions)"}>{label}</button>
        ))}
      </div>
      <label className="check inline ocr-tr-toggle">
        <input type="checkbox" checked={wantTr} onChange={(e) => setWantTr(e.target.checked)} />
        Translate to
      </label>
      {wantTr && (
        <div className="gt-tabs" role="group" aria-label="Translation language">
          {["vi", "en", "zh"].map((c) => (
            <button key={c} className={tgt === c ? "active" : ""} aria-pressed={tgt === c} onClick={() => setTgt(c)}>
              {DISPLAY_LANGS.find((l) => l.code === c)?.label ?? c}
            </button>
          ))}
        </div>
      )}
    </div>

    {pages.length > 1 && (
      <div className="ocr-thumbs" role="tablist" aria-label="Pages">
        {pages.map((p, i) => (
          <button key={i} role="tab" aria-selected={i === sel} className={`ocr-thumb ${i === sel ? "on" : ""}`} onClick={() => { setSel(i); setSelBlock(null); }} title={`${p.file} · page ${p.page}`}>
            {p.preview && <img src={p.preview} alt="" />}
            <span className="ocr-thumb-n">{p.page}</span>
          </button>
        ))}
      </div>
    )}

    <div className="ocr-grid">
      <div className="ocr-stage">
        <div ref={viewRef} className={`ocr-view ${zoom > 1 && !cropMode ? "pannable" : ""}`} tabIndex={0}
          aria-label="Page viewer. Ctrl + scroll or + / − to zoom, 0 to fit"
          onKeyDown={onViewKey}
          onMouseDown={(e) => {
            if (cropMode || zoom <= 1 || !viewRef.current) return;
            panRef.current = { x: e.clientX, y: e.clientY, l: viewRef.current.scrollLeft, t: viewRef.current.scrollTop };
          }}
          onMouseMove={(e) => {
            const p = panRef.current;
            if (!p || !viewRef.current) return;
            viewRef.current.scrollLeft = p.l - (e.clientX - p.x);
            viewRef.current.scrollTop = p.t - (e.clientY - p.y);
          }}
          onMouseUp={() => { panRef.current = null; }}
          onMouseLeave={() => { panRef.current = null; }}>
          {!showing && <div className="ocr-empty">
            <Icon name="scan" size={34} stroke={1.4} />
            <p>{files.length ? "PDF pages render here after Run OCR." : "Add an image or PDF to start."}</p>
            {!files.length && <button className="ocr-tool" onClick={() => fileRef.current?.click()}><Icon name="plus" size={16} />Add document</button>}
          </div>}
          {showing && (
            <div ref={previewRef} className="ocr-canvas"
              style={{ width: `${zoom * 100}%`, cursor: cropMode ? "crosshair" : undefined }}
              onDoubleClick={(e) => { if (!cropMode) zoomTo(zoom > 1 ? 1 : 2, e.clientX, e.clientY); }}
              onMouseDown={(e) => { if (!cropMode || !previewRef.current) return; dragRef.current = { x0: relOf(e)[0], y0: relOf(e)[1] }; setCrop(null); }}
              onMouseMove={(e) => {
                const d = dragRef.current;
                if (!d) return;
                const [x, y] = relOf(e);
                setCrop([Math.min(d.x0, x), Math.min(d.y0, y), Math.max(d.x0, x), Math.max(d.y0, y)]);
              }}
              onMouseUp={() => {
                if (!dragRef.current) return;
                dragRef.current = null;
                setCrop((c) => {
                  if (!c || c[2] - c[0] < 0.02 || c[3] - c[1] < 0.02) return null;
                  stamp(`Crop set — ${Math.round((c[2] - c[0]) * 100)}% × ${Math.round((c[3] - c[1]) * 100)}% of page`);
                  return c;
                });
              }}>
              <img src={showing} alt={isSource ? "Document to read" : `Page ${page?.page} with detected regions`} style={isSource && rotate ? { transform: `rotate(${rotate}deg)` } : undefined} draggable={false} />
              {!isSource && blocks.map((b, i) => (
                <button key={b.id} className={`ocr-box ob-${i % 5} ${selBlock === b.id ? "on" : ""}`}
                  style={{ left: `${b.box[0] * 100}%`, top: `${b.box[1] * 100}%`, width: `${(b.box[2] - b.box[0]) * 100}%`, height: `${(b.box[3] - b.box[1]) * 100}%` }}
                  onClick={() => focusBlock(b.id)} aria-label={`Block ${b.id}: ${b.text}`} title={`${b.id} · ${Math.round(b.conf * 100)}% · ${b.text}`} />
              ))}
              {crop && isSource && (
                <div className="ocr-crop" style={{ left: `${crop[0] * 100}%`, top: `${crop[1] * 100}%`, width: `${(crop[2] - crop[0]) * 100}%`, height: `${(crop[3] - crop[1]) * 100}%` }} />
              )}
            </div>
          )}
        </div>
        <div className="ocr-zoombar" role="group" aria-label="Zoom">
          <button onClick={() => zoomTo(zoom / ZOOM_STEP)} disabled={!showing || zoom <= ZOOM_MIN} title="Zoom out (−)" aria-label="Zoom out"><Icon name="zoomOut" size={17} /></button>
          <input type="range" min={ZOOM_MIN * 100} max={ZOOM_MAX * 100} step={5} value={zoomPct} disabled={!showing}
            onChange={(e) => zoomTo(+e.target.value / 100)} aria-label="Zoom level" />
          <button onClick={() => zoomTo(zoom * ZOOM_STEP)} disabled={!showing || zoom >= ZOOM_MAX} title="Zoom in (+)" aria-label="Zoom in"><Icon name="zoomIn" size={17} /></button>
          <output className="ocr-zoom" aria-live="polite">{zoomPct}%</output>
          <button className="ocr-fit" onClick={() => zoomTo(1)} disabled={!showing || zoom === 1} title="Fit width (0)"><Icon name="fit" size={15} />Fit</button>
        </div>
      </div>

      <aside className="ocr-side">
        <div className="ocr-tabs" role="tablist">
          {(["text", "regions", "quality", "export"] as const).map((t) => (
            <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{t[0].toUpperCase() + t.slice(1)}</button>
          ))}
        </div>
        {tab === "text" && (
          <>
            <div className="ocr-detected">
              <div><span>Detected text</span><p>{page?.backend ?? "—"}{page?.ocr_ms ? ` · ${(page.ocr_ms / 1000).toFixed(1)} s` : ""}</p></div>
              <div className="ocr-conf">Confidence <b>{pages.length ? `${Math.round(overall * 100)}%` : "—"}</b></div>
            </div>
            {page?.text && <button className="ocr-copy" onClick={() => void copyPage()}><Icon name="copy" size={15} />Copy page text</button>}
            {page?.error && <p className="ocr-warn">{page.error}</p>}
            {blocks.map((b, i) => (
              <div key={b.id} data-ocrblock={b.id} className={`ocr-block ${selBlock === b.id ? "on" : ""} ${b.conf < 0.6 ? "low" : ""}`} onClick={() => setSelBlock(b.id)}>
                <div className="ocr-block-head"><b className={`ocr-num ob-${i % 5}`}>{b.id}</b><span className="ocr-block-conf">{b.conf < 0.6 ? "check · " : ""}{Math.round(b.conf * 100)}%</span></div>
                <p lang={textLang(b.text, srcLang)}>{b.text}</p>
                {wantTr && b.translation && <p className="ocr-block-tr" lang={tgt === "zh" ? "zh-Hans" : tgt}><b className="tr-lang">{tgt}</b>{b.translation}</p>}
              </div>
            ))}
            {!blocks.length && page?.text && <div className="ocr-block"><p lang={textLang(page.text, srcLang)} style={{ whiteSpace: "pre-wrap" }}>{page.text}</p></div>}
            {!page && <p className="archive-empty">{busy ? "Reading…" : "Recognised text appears here, one block per region."}</p>}
            {page?.translation && !blocks.length && <div className="ocr-block"><div className="ocr-block-head"><b>Translation</b></div><p lang={tgt === "zh" ? "zh-Hans" : tgt} style={{ whiteSpace: "pre-wrap" }}>{page.translation}</p></div>}
            {page?.translation_error && <p className="ocr-warn">Translation failed: {page.translation_error}</p>}
          </>
        )}
        {tab === "regions" && (
          <>
            <p className="archive-meta">{blocks.length} regions {task !== "ocr" ? "(Table / Formula / Chart reads carry no regions — use Text)" : ""}</p>
            {blocks.map((b) => (
              <button key={b.id} className={`ocr-region ${selBlock === b.id ? "on" : ""}`} onClick={() => focusBlock(b.id)}>
                <b>#{b.id}</b><span>[{b.box.map((v) => v.toFixed(2)).join(", ")}]</span><i>{Math.round(b.conf * 100)}%</i>
              </button>
            ))}
          </>
        )}
        {tab === "quality" && (
          <>
            <div className="ocr-detected"><div><span>Quality</span><p>{blocks.length} blocks · {review.length} need review</p></div>
              <div className="ocr-conf"><b>{pages.length ? `${Math.round(overall * 100)}%` : "—"}</b></div></div>
            {review.length > 0 && <p className="ocr-warn">Low-confidence blocks — blurred, handwritten or cut off. Check them against the page:</p>}
            {lowConf.map((b) => (
              <button key={b.id} className="ocr-region" onClick={() => { setTab("text"); focusBlock(b.id); }}>
                <b>#{b.id}</b><span className="ocr-low-text" lang={textLang(b.text, srcLang)}>{b.text.slice(0, 42)}</span><i>{Math.round(b.conf * 100)}%</i>
              </button>
            ))}
            {lowConf.length === 0 && pages.length > 0 && <p className="archive-empty">All blocks above 85% — nothing to review.</p>}
            {!pages.length && <p className="archive-empty">Quality stats appear after Run OCR.</p>}
          </>
        )}
        {tab === "export" && (
          <>
            <p className="archive-meta">Downloads include all pages{wantTr ? " + translations" : ""}.</p>
            <div className="ocr-exports">
              <button disabled={!pages.length} onClick={() => download("ocr.txt", exportTxt(), "text/plain")}>Export text (.txt)</button>
              <button disabled={!pages.length} onClick={() => download("ocr.json", JSON.stringify(pages.map((p) => ({ file: p.file, page: p.page, text: p.text, blocks: p.blocks, overall_conf: p.overall_conf, translation: p.translation })), null, 2), "application/json")}>Export as JSON</button>
              <button disabled={!pages.length} onClick={() => download("ocr.csv", "﻿file,page,id,text,conf\n" + pages.flatMap((p) => (p.blocks ?? []).map((b) => `${p.file},${p.page},${b.id},"${(b.text ?? "").replace(/"/g, '""')}",${b.conf}`)).join("\n"), "text/csv")}>Export as CSV</button>
              <button disabled={!pages.length} onClick={() => download("ocr.md", pages.map((p) => `## ${p.file} · page ${p.page}\n\n${p.text ?? ""}${p.translation ? `\n\n> [${tgt}] ${p.translation}` : ""}`).join("\n\n"), "text/markdown")}>Export Markdown</button>
            </div>
          </>
        )}
      </aside>
    </div>

    <div className="ocr-bottom">
      <div className="ocr-trace">
        <div className="ocr-trace-tabs"><span className="on">OCR trace</span></div>
        {trace.length === 0 && <p className="archive-empty">Each step of a job is logged here.</p>}
        {trace.map((t, i) => <p key={i}>{t}</p>)}
      </div>
      <div className="ocr-actions">
        <button disabled={!pages.length} onClick={() => download("ocr.txt", exportTxt(), "text/plain")}><Icon name="download" size={15} />Export text</button>
        <button disabled={!files.length} onClick={() => { pick([]); setTrace([]); stamp("Cleared"); }}>Clear</button>
      </div>
    </div>
  </section>;
}

function SettingsView(props: {
  health: Health | null; mics: { id: string; label: string }[]; micId: string | undefined;
  onMic: (id: string | undefined) => void; srcChoice: string; onSrc: (v: string) => void;
  targets: string[]; onTargets: (t: string[]) => void;
  denoise: boolean; onDenoise: (v: boolean) => void; status: string;
  dev: boolean; onDev: (v: boolean) => void; onHealth: () => void;
  onRefresh: () => void; onDiarizer: (m: DiarizerMode) => void; onNavigate: () => void;
}) {
  const h = props.health;
  const dBackend = h?.diarizer_status?.backend ?? "";
  // the selected class first: a Nemotron diarizer on its Titanet fallback is still "nemotron"
  const diarMode: DiarizerMode = h?.diarizer === "NemotronDiarizer" ? "nemotron"
    : dBackend === "pyannote" || h?.diarizer === "PyannoteDiarizer" ? "pyannote"
    : dBackend === "nemo-titanet" || h?.diarizer === "NeMoDiarizer" ? "nemo" : "volume";
  // memory readout stays current while Settings is open (voices unload when idle)
  useEffect(() => {
    props.onHealth();
    const t = window.setInterval(props.onHealth, 5000);
    return () => window.clearInterval(t);
  }, []);
  const gb = (mb?: number) => (mb === undefined ? "—" : `${(mb / 1024).toFixed(1)} GB`);
  const voices = h?.tts?.langs ?? [];
  return <section className="secondary-view wide settings-view">
    <h1>Settings</h1>
    <p>Input, languages and the engines running on this computer.</p>

    <h3 className="group-title">Input</h3>
    <div className="form-group">
      <label className="form-row"><span>Microphone</span>
        <select value={props.micId ?? ""} onChange={(e) => props.onMic(e.target.value || undefined)}>
          <option value="">System default</option>
          {props.mics.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
        </select>
      </label>
      <label className="form-row"><span>Spoken language</span>
        <select value={props.srcChoice} onChange={(e) => props.onSrc(e.target.value)}>
          <option value="auto">Detect automatically</option>
          <option value="Vietnamese">Tiếng Việt</option>
          <option value="English">English</option>
          <option value="Chinese">中文</option>
        </select>
      </label>
      <label className="form-row"><span>Noise reduction<small>Cleans each segment before the final transcript</small></span>
        <input type="checkbox" className="toggle" checked={props.denoise} onChange={(e) => props.onDenoise(e.target.checked)} />
      </label>
    </div>

    <h3 className="group-title">Translation</h3>
    <div className="form-group">
      <div className="form-row"><span>Translate into<small>Applies to new segments and uploads</small></span>
        <div className="target-row" role="group" aria-label="Translate into">
          {DISPLAY_LANGS.map((l) => {
            const on = props.targets.includes(l.code);
            return <button key={l.code} className={`chip ${on ? "on" : ""}`} aria-pressed={on} lang={htmlLang(l.code)}
              onClick={() => {
                const next = on ? props.targets.filter((t) => t !== l.code) : [...props.targets, l.code];
                props.onTargets(next.length ? next : [l.code]);
              }}>{on && <Icon name="check" size={13} stroke={2.4} />}{l.native}</button>;
          })}
        </div>
      </div>
      <div className="form-row"><span>Speaker voices<small>Listen buttons under each translation</small></span>
        <span className="form-value">{voices.length ? `Piper (offline, CPU) · ${voices.map((v) => v.toUpperCase()).join(" ")}` : <>System voice only — install: <code>python scripts/get_tts.py</code></>}</span>
      </div>
    </div>

    <h3 className="group-title">Speakers</h3>
    <div className="form-group">
      <label className="form-row"><span>Speaker detection</span>
        <select value={diarMode} onChange={(e) => props.onDiarizer(e.target.value as DiarizerMode)}>
          <option value="nemotron">Nemotron streaming (switches mid-sentence)</option>
          <option value="pyannote">pyannote embeddings (GPU)</option>
          <option value="nemo">NeMo Titanet (neural)</option>
          <option value="volume">Volume levels (instant)</option>
        </select>
      </label>
    </div>

    <h3 className="group-title">Engines</h3>
    <div className="form-group">
      <div className="form-row"><span>Device</span><span className="form-value">{h ? `${h.device.device.toUpperCase()}${h.device.gpu_name ? ` · ${h.device.gpu_name}` : ""}` : "…"}</span></div>
      <div className="form-row"><span>Speech recognition</span><span className="form-value">{h ? `${h.asr.model}${h.asr.ready ? "" : " · loading"}` : "…"}</span></div>
      <div className="form-row"><span>Translation</span><span className="form-value">{h ? `${h.mt.model}${h.mt.ready ? "" : " · loading"}` : "…"}</span></div>
      <div className="form-row"><span>Speaker detection</span><span className="form-value">{h ? backendLabel(h.diarizer_status) : "…"}</span></div>
      <div className="form-row"><span>Noise reduction</span><span className="form-value">{h ? (h.denoise.backend === "passthrough" ? "Off — DeepFilterNet not installed" : `${h.denoise.backend} · ${h.denoise.device}`) : "…"}</span></div>
      <div className="form-row"><span>Memory<small>Backend and its helpers</small></span><span className="form-value mono">RAM {gb(h?.mem?.ram_mb)} · GPU {gb(h?.mem?.vram_mb)}{h?.mem?.gpu_used_mb !== undefined ? ` (device ${gb(h.mem.gpu_used_mb)})` : ""}</span></div>
      <label className="form-row"><span>Latency readout<small>Per-segment ASR / MT / speaker timings</small></span>
        <input type="checkbox" className="toggle" checked={props.dev} onChange={(e) => props.onDev(e.target.checked)} />
      </label>
    </div>
    <p className="status-line" role="status">{props.status}</p>
    <div className="settings-actions">
      <button className="btn" onClick={props.onRefresh}>Reload Models</button>
      <button className="btn primary" onClick={props.onNavigate}>Open Live Session</button>
    </div>
  </section>;
}
