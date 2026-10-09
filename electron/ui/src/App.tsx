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
  | "grid"
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
    grid: <><rect x="3.5" y="3.5" width="7" height="7" rx="1.3" /><rect x="13.5" y="3.5" width="7" height="7" rx="1.3" /><rect x="3.5" y="13.5" width="7" height="7" rx="1.3" /><rect x="13.5" y="13.5" width="7" height="7" rx="1.3" /></>,
    clock: <><circle cx="12" cy="12" r="8.5" /><path d="M12 7v5l3.4 2" /></>,
    book: <><path d="M4.5 5.5A2.5 2.5 0 0 1 7 3h12v16H7a2.5 2.5 0 0 0-2.5 2.5v-16Z" /><path d="M7 3v16" /><path d="M10.5 7H16" /><path d="M10.5 10.5H16" /></>,
    settings: <><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.7 1.7 0 0 0 .34 1.88l.06.06-2.6 2.6-.06-.06a1.7 1.7 0 0 0-1.88-.34 1.7 1.7 0 0 0-1.03 1.56v.09h-3.68v-.09A1.7 1.7 0 0 0 9.52 19a1.7 1.7 0 0 0-1.88.34l-.06.06-2.6-2.6.06-.06A1.7 1.7 0 0 0 5.38 15a1.7 1.7 0 0 0-1.56-1.03h-.09v-3.68h.09A1.7 1.7 0 0 0 5.38 9.3a1.7 1.7 0 0 0-.34-1.88l-.06-.06 2.6-2.6.06.06A1.7 1.7 0 0 0 9.52 5.2a1.7 1.7 0 0 0 1.03-1.56v-.09h3.68v.09a1.7 1.7 0 0 0 1.03 1.56 1.7 1.7 0 0 0 1.88-.34l.06-.06 2.6 2.6-.06.06a1.7 1.7 0 0 0-.34 1.88 1.7 1.7 0 0 0 1.56 1.03h.09v3.68h-.09A1.7 1.7 0 0 0 19.4 15Z" /></>,
    plus: <path d="M12 5v14M5 12h14" />,
    chevron: <path d="m7 10 5 5 5-5" />,
    mic: <><rect x="8.5" y="3" width="7" height="12" rx="3.5" /><path d="M5 11.5a7 7 0 0 0 14 0M12 18.5V22M8.5 22h7" /></>,
    pause: <path d="M8.5 5v14M15.5 5v14" />,
    stop: <rect x="6.5" y="6.5" width="11" height="11" rx="1" />,
    more: <><circle cx="5" cy="12" r="1" fill="currentColor" /><circle cx="12" cy="12" r="1" fill="currentColor" /><circle cx="19" cy="12" r="1" fill="currentColor" /></>,
    translate: <><path d="M5 5h8M9 3v2c0 4-2.2 7.1-5.5 8.8M5 12c1.1 1 2.4 1.8 4 2.2" /><path d="M14 19h6M17 5l4 14M13 19l4-14" /></>,
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
    globe: <><circle cx="12" cy="12" r="8.5" /><path d="M3.5 12h17M12 3.5c2.3 2.4 3.4 5.2 3.4 8.5s-1.1 6.1-3.4 8.5c-2.3-2.4-3.4-5.2-3.4-8.5s1.1-6.1 3.4-8.5Z" /></>,
    scan: <><path d="M4 8V5.5A1.5 1.5 0 0 1 5.5 4H8M16 4h2.5A1.5 1.5 0 0 1 20 5.5V8M20 16v2.5a1.5 1.5 0 0 1-1.5 1.5H16M8 20H5.5A1.5 1.5 0 0 1 4 18.5V16" /><path d="M4 12h16" /></>,
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
  { code: "vi", label: "Vietnamese" },
  { code: "en", label: "English" },
  { code: "zh", label: "Chinese" },
];

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

const navigation = [
  { label: "Live session", icon: "grid" as IconName },
  { label: "Library", icon: "clock" as IconName },
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
        <p className="secondary-label">Hiccup</p>
        <h1>This view hit bad data.</h1>
        <p>Your live session and archive are intact — go back and carry on.</p>
        <button onClick={() => { this.setState({ failed: false }); this.props.onReset(); }}>
          Back to live session <Icon name="arrow" size={17} />
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
        if (!liveRef.current) setStatus("Ready — press the mic core or New session to record.");
      } catch {
        if (!liveRef.current) setStatus("Backend unreachable — launch run.py, then reload.");
      }
    })();
  }, []);

  useEffect(() => saveGlossary(terms), [terms]);

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
      <aside className="sidebar">
        <div className="brand-lockup"><AppLogo /><span>ConfLive</span></div>
        <button className="new-session" onClick={newSession}><Icon name="plus" size={17} /><span>New session</span></button>
        <nav className="primary-nav" aria-label="Main navigation">
          <p className="nav-kicker">Workspace</p>
          {navigation.map((item) => (
            <button className={`nav-item ${activeView === item.label ? "active" : ""}`} key={item.label} onClick={() => { setActiveView(item.label); setOpenArchived(null); }}>
              <Icon name={item.icon} size={18} /><span>{item.label}</span>{item.label === "Live session" && live && <i className="nav-live-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <button className={`nav-item ${activeView === "Settings" ? "active" : ""}`} onClick={() => setActiveView("Settings")}><Icon name="settings" size={18} /><span>Settings</span></button>
          <button className="profile-button"><span className="profile-avatar">CL</span><span className="profile-name">Local session</span><Icon name="chevron" size={16} /></button>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div className="crumbs"><span>All sessions</span><b>/</b><strong>{activeView === "Live session" ? "Live transcription" : activeView}</strong></div>
          <div className="topbar-actions">
            <label className="language-select"><Icon name="translate" size={16} /><select value={displayLang} onChange={(e) => setDisplayLang(e.target.value)} aria-label="Translation language">{DISPLAY_LANGS.map((l) => <option key={l.code} value={l.code}>{l.label}</option>)}</select><Icon name="chevron" size={15} /></label>
            <button className="icon-button theme-toggle" aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"} title={theme === "dark" ? "Light mode" : "Dark mode"} onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}><Icon key={theme} name={theme === "dark" ? "sun" : "moon"} size={17} /></button>
            <button className="icon-button" aria-label="Session settings" onClick={() => setActiveView("Settings")}><Icon name="sliders" size={18} /></button>
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
      setStuck(el.scrollTop > 380);
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
      <div className="mini-stage" aria-label="Recording controls (compact)">
        <i className={active ? "recording-dot" : "idle-dot"} />
        <span className="mini-timer">{formatTime(props.elapsed)}</span>
        <span className="mini-status">{props.status}</span>
        <button className="mini-btn" onClick={props.onTogglePause} aria-label={active ? "Pause" : "Resume"}>
          <Icon name={active ? "pause" : "mic"} size={14} />
        </button>
        <button className="mini-btn" onClick={props.onStop} aria-label="Stop recording">
          <Icon name="stop" size={13} />
        </button>
      </div>
    )}
    <section className="session-heading">
      <div>
        <div className="live-status"><i className={active ? "recording-dot" : "idle-dot"} />{active ? "Recording live" : props.live ? "Session paused" : "Session idle"}</div>
        <h1>Conference session</h1>
        <p>{props.sessionDate} <span /> {props.speakerCount} speaker{props.speakerCount === 1 ? "" : "s"}</p>
      </div>
      <div className="session-actions">
        <button className="text-button" onClick={props.onCopy}>{props.copied ? <Icon name="check" size={16} /> : <Icon name="copy" size={16} />}{props.copied ? "Copied" : "Copy"}</button>
        <button className="text-button" onClick={() => fileRef.current?.click()}><Icon name="upload" size={17} />Upload</button>
        <input ref={fileRef} type="file" accept="audio/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) props.onUpload(f); e.target.value = ""; }} />
        <button className="text-button" onClick={props.exportTranscript}><Icon name="download" size={17} />Export</button>
        <button className="icon-button more-button" aria-label="More session actions"><Icon name="more" size={20} /></button>
      </div>
    </section>

    <section className="recording-stage" aria-label="Recording controls">
      <div className="stage-meta"><span>Microphone</span><b>{props.micLabel ?? "Default input"}</b><i /></div>
      <div className={`sound-orbit ${active ? "is-live" : ""}`}><div className="orbit-line orbit-one" /><div className="orbit-line orbit-two" /><div className="record-core"><Icon name={active ? "pause" : "mic"} size={26} stroke={2} /></div></div>
      <div className="timer">{formatTime(props.elapsed)}</div>
      <div className="waveform" aria-hidden="true"><WaveBars level={props.micLevel} live={active} /></div>
      <div className="stage-controls">
        <button className="stop-button" onClick={props.onStop} aria-label="Stop recording"><Icon name="stop" size={15} /></button>
        <button className={`record-toggle ${active ? "pause-state" : ""}`} onClick={props.onTogglePause}><Icon name={active ? "pause" : "mic"} size={17} /><span>{active ? "Pause recording" : props.live ? "Resume recording" : "Start recording"}</span></button>
      </div>
      <div className="stage-status">{props.status}</div>
    </section>

    <section className="transcript-toolbar">
      <div className="toolbar-title"><span>Transcript</span><b>Live</b></div>
      <div className="toolbar-actions"><label className="switch-wrap"><span>Translation</span><input type="checkbox" checked={props.showTranslation} onChange={props.toggleTranslation} /><i className="switch" /></label><label className="search-box"><Icon name="search" size={16} /><input placeholder="Search transcript" onChange={(e) => props.setQuery(e.target.value)} /></label></div>
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
              <p>{props.partial.text}</p>
              {props.showTranslation && rowTranslations(
                { translations: props.partial.translations || {} },
                props.displayLang,
                props.tok && props.tok.id === props.partial.id ? props.tok.text : null,
              ).map(([code, text, streaming]) => {
                if (!text) return null;
                return (
                  <p className="translated-text live-tr" key={code}>
                    <b className="tr-lang">{code}</b>
                    {text}
                    {streaming ? <span className="typing-caret inline" /> : null}
                  </p>
                );
              })}
            </div>
            <span className="typing-caret" />
          </div>
        )}
        {props.visibleRows.length === 0 && !props.partial && <div className="empty-search">No moments yet — press Start recording or Upload audio.</div>}
        <div ref={bottomRef} aria-hidden="true" />
        {!follow && props.live && (
          <button className="jump-live" onClick={jumpToLive}>↓ Jump to live</button>
        )}
      </div>
      <aside className="insight-panel">
        <div className="insight-heading"><span>Session</span></div>
        <div className="stat-grid">
          <div><b>{formatTime(props.elapsed)}</b><span>duration</span></div>
          <div><b>{props.visibleRows.length}</b><span>segments</span></div>
          <div><b>{props.speakerCount}</b><span>speakers</span></div>
          <div><b>{props.targets.join(" · ")}</b><span>targets</span></div>
        </div>
        <div className="insight-section"><span>ASR</span><p>Zipformer zh-en-vi (sherpa-onnx, CPU int8)</p></div>
        <div className="insight-section"><span>Translation</span><p>Hy-MT2 FP8{props.health ? ` · ${props.health.device.device}` : ""}</p></div>
        <div className="insight-section"><span>Diarization</span><p>{props.health ? backendLabel(props.health.diarizer_status) : "volume"}</p></div>
      </aside>
    </section>
  </>;
}

const TranscriptRow = memo(function TranscriptRow({ entry, selected, onPick, showTranslation, displayLang, tokText, dev }: {
  entry: Row; selected: boolean; onPick: (id: number) => void; showTranslation: boolean;
  displayLang: string; tokText: string | null; dev: boolean;
}) {
  return (
    <button className={`transcript-row ${selected ? "selected" : ""}`} onClick={() => onPick(entry.id)} data-row-id={entry.id}>
      <time>{formatTime(entry.at)}</time>
      <span className={`speaker-avatar ${AVATAR_COLORS[speakerIdx(entry.speaker) % AVATAR_COLORS.length]}`}>{speakerInitials(entry.speaker)}</span>
      <span className="entry-copy">
        <span className="speaker-line"><b>{speakerName(entry.speaker)}</b></span>
        <span className="original-text">{entry.text}</span>
        {showTranslation && rowTranslations(entry, displayLang, tokText).map(([code, text, streaming]) => {
          if (!text && !entry.pending) return null;
          return (
            <span className="translated-text" key={code}>
              <b className="tr-lang">{code}</b>
              {text}
              {streaming ? <span className="typing-caret inline" /> : null}
            </span>
          );
        })}
        {entry.pending ? <span className="translating">translating…</span> : null}
        {dev && !entry.pending && fmtLatency(entry.timings) ? <span className="latency">{fmtLatency(entry.timings)}</span> : null}
      </span>
      <span className="row-more"><Icon name="more" size={18} /></span>
    </button>
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
    <defs><linearGradient id="logo-g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#7C78E0" /><stop offset="1" stopColor="#1D2640" /></linearGradient></defs>
    <rect x="4" y="4" width="92" height="92" rx="22.5" fill="url(#logo-g)" />
    <rect x="19.5" y="23.5" width="61" height="45" rx="14.6" fill="#fff" />
    <path d="M32.2 64.5 29.3 81 46.9 67.4Z" fill="#fff" />
    {[9, 19.5, 29.3, 19.5, 9].map((h, i) => <rect key={i} x={32.7 + i * 8.8} y={45.9 - h / 2} width="5.3" height={h} rx="2.6" fill="#635FBD" />)}
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
    <Icon name="search" size={16} />
    <input placeholder="Search words in every session" value={q} onChange={(e) => setQ(e.target.value)} autoFocus />
    {q ? <button className="gt-clear" onClick={() => setQ("")} aria-label="Clear search">✕</button> : null}
  </label>;
  const open = props.archive.find((s) => s.id === props.openId) ?? null;
  if (open) {
    const rows = open.utterances.filter(hits);
    return <section className="secondary-view wide" key={open.id}>
      <p className="secondary-label">Archive · {open.label || "Live session"} · {new Date(open.startedAt).toLocaleString()}</p>
      <h1>{formatTime(open.seconds)} session</h1>
      <p>{needle ? `${rows.length} of ${open.utterances.length} segments match “${q.trim()}”.` : `${open.utterances.length} segments.`}</p>
      <div className="archive-actions">
        <button onClick={() => props.onOpen(null)}>← Back to library</button>
        <button className="danger" onClick={() => { props.onDelete(open.id); props.onOpen(null); }}>Delete session</button>
      </div>
      {searchBox}
      <div className="transcript-list archive-list">
        {rows.map((u, i) => (
          <div className="transcript-row" key={i}>
            <time>{u.time}</time>
            <span className="speaker-avatar plum">{(u.speaker || "?").replace("Speaker ", "S")}</span>
            <span className="entry-copy">
              <span className="speaker-line"><b>{u.speaker || "Speaker"}</b></span>
              <span className="original-text">{mark(u.text || "", needle)}</span>
              {orderedTranslations({ translations: archivedDict(u) }, props.displayLang).map(([code, text]) => (
                <span className="translated-text" key={code}>
                  <b className="tr-lang">{code}</b>
                  {mark(text, needle)}
                </span>
              ))}
              {props.dev && fmtLatency(u.timings) ? (
                <span className="latency">{fmtLatency(u.timings)}</span>
              ) : null}
            </span>
          </div>
        ))}
      </div>
    </section>;
  }
  return <section className="secondary-view wide" key="library">
    <p className="secondary-label">Your archive</p>
    <h1>Every conversation, ready when you are.</h1>
    <p>Sessions stay on this device and are automatically organized by date and speaker.</p>
    <button onClick={props.onNavigate}>Open live session <Icon name="arrow" size={17} /></button>
    {props.archive.length > 0 && searchBox}
    <div className="archive-list">
      {props.archive.length === 0 && <p className="archive-empty">No archived sessions yet — stop a session to save it here.</p>}
      {props.archive.map((s) => {
        const n = needle ? s.utterances.filter(hits).length : 0;
        if (needle && !n && !(s.label || "").toLowerCase().includes(needle)) return null;
        return <button className="archive-row" key={s.id} onClick={() => props.onOpen(s.id)}>
          <span className="archive-title">{s.label || "Live session"} · {new Date(s.startedAt).toLocaleString()}</span>
          <span className="archive-meta">{needle ? `${n} match${n === 1 ? "" : "es"} · ` : ""}{s.utterances.length} segments · {formatTime(s.seconds)}</span>
          <Icon name="arrow" size={16} />
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
  return <section className="secondary-view wide">
    <p className="secondary-label">Language</p>
    <h1>Make every technical term land correctly.</h1>
    <p>These pairs are sent with every translation (Hy-MT2 terminology mode), so product words and names survive interpretation.</p>
    <div className="glossary-add">
      <input placeholder="Source phrase (as spoken)" value={src} onChange={(e) => setSrc(e.target.value)} />
      <input placeholder={`Preferred translation`} value={tgt} onChange={(e) => setTgt(e.target.value)} />
      <button onClick={() => {
        if (!src.trim() || !tgt.trim()) return;
        props.onChange({ ...props.terms, [src.trim()]: tgt.trim() });
        setSrc("");
        setTgt("");
      }}>Add a glossary term <Icon name="arrow" size={17} /></button>
    </div>
    <div className="glossary-list">
      {entries.length === 0 && <p className="archive-empty">No terms yet — e.g. source “standup”, target “站会”.</p>}
      {entries.map(([s, t]) => (
        <div className="glossary-row" key={s}>
          <span className="glossary-src">{s}</span>
          <Icon name="arrow" size={15} />
          <span className="glossary-tgt">{t}</span>
          <button className="glossary-del" onClick={() => {
            const n = { ...props.terms };
            delete n[s];
            props.onChange(n);
          }}>Remove</button>
        </div>
      ))}
    </div>
    <div className="secondary-wave"><Icon name="wave" size={46} stroke={1.25} /></div>
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
  const [copiedOut, setCopiedOut] = useState(false);
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

  function copyOut() {
    if (!out) return;
    void navigator.clipboard?.writeText(out);
    setCopiedOut(true);
    window.setTimeout(() => setCopiedOut(false), 1500);
  }

  const mainLangs = ["en", "zh", "vi"];
  return <section className="secondary-view wide">
    <p className="secondary-label">Phrasebook</p>
    <h1>Type it, get it back translated.</h1>
    <p>Same Hy-MT2 engine as the live session{termCount ? `, with ${termCount} glossary term${termCount === 1 ? "" : "s"} applied` : ""}. Enter to translate.</p>
    <div className="gt-bar">
      <div className="gt-tabs">
        {[["auto", "Detect language"], ...mainLangs.map((c) => [c, DISPLAY_LANGS.find((l) => l.code === c)?.label ?? c])].map(([code, label]) => (
          <button key={code} className={src === code ? "active" : ""} onClick={() => setSrc(code)}>{label}</button>
        ))}
      </div>
      <button className="gt-swap" onClick={swap} disabled={src === "auto" || !out} aria-label="Swap languages" title="Swap languages">
        <Icon name="arrow" size={16} />
      </button>
      <div className="gt-tabs">
        {mainLangs.map((c) => (
          <button key={c} className={tgt === c ? "active" : ""} onClick={() => setTgt(c)}>
            {DISPLAY_LANGS.find((l) => l.code === c)?.label ?? c}
          </button>
        ))}
        <select value={EXTRA_LANGS.includes(tgt) ? tgt : ""} onChange={(e) => { if (e.target.value) setTgt(e.target.value); }} aria-label="More languages">
          <option value="">{EXTRA_LANGS.includes(tgt) ? tgt.toUpperCase() : "More ▾"}</option>
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
          {text ? <button className="gt-clear" onClick={() => { setText(""); setOut(""); }}>✕</button> : null}
        </div>
      </div>
      <div className="gt-card result">
        {busy ? <span className="gt-thinking">Translating…</span>
          : out ? <p key={out}>{out}</p>
          : <span className="gt-placeholder">Translation</span>}
        {out && !busy ? (
          <div className="gt-card-foot">
            <span className="gt-terms-note">{termCount ? `glossary: ${termCount} terms` : ""}</span>
            <button className="gt-copy" onClick={copyOut} aria-label="Copy translation">
              {copiedOut ? <Icon name="check" size={16} /> : <Icon name="copy" size={16} />}
            </button>
          </div>
        ) : null}
      </div>
    </div>
    <div className="secondary-wave"><Icon name="wave" size={46} stroke={1.25} /></div>
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
  dev: boolean; onDev: (v: boolean) => void;
  onRefresh: () => void; onDiarizer: (m: DiarizerMode) => void; onNavigate: () => void;
}) {
  const h = props.health;
  const dBackend = h?.diarizer_status?.backend ?? "";
  // the selected class first: a Nemotron diarizer on its Titanet fallback is still "nemotron"
  const diarMode: DiarizerMode = h?.diarizer === "NemotronDiarizer" ? "nemotron"
    : dBackend === "pyannote" || h?.diarizer === "PyannoteDiarizer" ? "pyannote"
    : dBackend === "nemo-titanet" || h?.diarizer === "NeMoDiarizer" ? "nemo" : "volume";
  return <section className="secondary-view wide">
    <p className="secondary-label">Preferences</p>
    <h1>Make ConfLive work around your room.</h1>
    <p>Devices, languages, diarization backend, and model status.</p>
    <div className="settings-grid">
      <label>Microphone
        <select value={props.micId ?? ""} onChange={(e) => props.onMic(e.target.value || undefined)}>
          <option value="">System default</option>
          {props.mics.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
        </select>
      </label>
      <label>Spoken language
        <select value={props.srcChoice} onChange={(e) => props.onSrc(e.target.value)}>
          <option value="auto">auto-detect</option>
          <option value="Vietnamese">Tiếng Việt</option>
          <option value="English">English</option>
          <option value="Chinese">中文</option>
        </select>
      </label>
      <label>Speaker diarization
        <select value={diarMode} onChange={(e) => props.onDiarizer(e.target.value as DiarizerMode)}>
          <option value="nemotron">Nemotron streaming (switches mid-sentence)</option>
          <option value="pyannote">pyannote embeddings (GPU)</option>
          <option value="nemo">NeMo Titanet (neural)</option>
          <option value="volume">Volume levels (instant)</option>
        </select>
      </label>
      <div className="target-pick">
        <span>Translate to (applies to new segments & uploads)</span>
        <div className="target-row">
          {DISPLAY_LANGS.map((l) => (
            <label key={l.code} className="check inline">
              <input
                type="checkbox"
                checked={props.targets.includes(l.code)}
                onChange={(e) => {
                  const next = e.target.checked
                    ? [...props.targets, l.code]
                    : props.targets.filter((t) => t !== l.code);
                  props.onTargets(next.length ? next : [l.code]);
                }}
              />
              {l.label}
            </label>
          ))}
        </div>
      </div>
      <label className="check">DeepFilterNet denoise
        <input type="checkbox" checked={props.denoise} onChange={(e) => props.onDenoise(e.target.checked)} />
      </label>
      <label className="check">Developer latency readout
        <input type="checkbox" checked={props.dev} onChange={(e) => props.onDev(e.target.checked)} />
      </label>
    </div>
    <div className="backend-status">
      <p><b>Device:</b> {h ? `${h.device.device}${h.device.gpu_name ? ` · ${h.device.gpu_name}` : ""}` : "…"}</p>
      <p><b>ASR:</b> {h ? `${h.asr.model} · ready=${String(h.asr.ready)}` : "…"}</p>
      <p><b>MT:</b> {h ? `${h.mt.model} · ready=${String(h.mt.ready)}` : "…"}</p>
      <p><b>Diarizer:</b> {h ? backendLabel(h.diarizer_status) : "…"}</p>
      <p><b>Denoise:</b> {h ? (h.denoise.backend === "passthrough" ? "OFF — passthrough (deepfilternet not installed; run setup.bat with Python 3.11)" : `${h.denoise.backend} on ${h.denoise.device}`) : "…"}</p>
      <p className="status-line">{props.status}</p>
    </div>
    <div className="settings-actions">
      <button onClick={props.onRefresh}>Reload models <Icon name="arrow" size={17} /></button>
      <button className="ghost" onClick={props.onNavigate}>Open live session <Icon name="arrow" size={17} /></button>
    </div>
  </section>;
}
