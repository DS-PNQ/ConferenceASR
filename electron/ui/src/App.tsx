import { useEffect, useMemo, useRef, useState, Component, type ReactNode } from "react";
import {
  apiHealth,
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
  type Health,
  type MicHandle,
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
  | "upload";

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
    wave: <><path d="M3 12h2l1.5-5 3 10 2.4-14L14.5 21l2.5-9H21" /></>,
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
    const t = window.setInterval(() => setElapsed((v) => v + 1), 1000);
    return () => window.clearInterval(t);
  }, [live, paused]);

  // mic level meter pump
  useEffect(() => {
    if (!live) {
      setMicLevel(0);
      return;
    }
    const t = window.setInterval(() => {
      const h = micRef.current;
      setMicLevel(h ? h.level() : 0);
    }, 120);
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
        setStatus("Ready — press the mic core or New session to record.");
      } catch {
        setStatus("Backend unreachable — launch run.py, then reload.");
      }
    })();
  }, []);

  useEffect(() => saveGlossary(terms), [terms]);

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
    };
    setRows((prev) => [...prev, row]);
    setSelectedSegment(u.id);
  }

  async function startRecording() {
    if (liveRef.current) return;
    setStatus("Opening microphone…");
    const socket = new LiveSocket();
    socketRef.current = socket;
    try {
      await socket.connect({
        onPartial: (p) => setPartial(p),
        onTok: (t) => {
          // live token stream for the display language only; the full
          // translations land with the utterance event right after.
          const want = displayLangRef.current;
          if (t.tgt !== want) return;
          tokAcc.current[t.id] = (tokAcc.current[t.id] ?? "") + t.delta;
          const text = tokAcc.current[t.id];
          setTok((cur) => (cur && cur.id === t.id ? { id: t.id, text } : { id: t.id, text }));
        },
        onUtterance: (u) => {
          setPartial((cur) => (cur && cur.id === u.id ? null : cur));
          delete tokAcc.current[u.id];
          setTok((cur) => (cur && cur.id === u.id ? null : cur));
          appendUtterance(u);
        },
        onStatus: (s) => {
          if (s === "listening") setStatus("● Listening — streaming ASR + translation.");
          else setStatus(s);
        },
        onError: (e) => setStatus("Error: " + e),
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
    anchorRef.current = Date.now() / 1000;
    try {
      const mic = await openMic(micId, (b64) => socket.audio(b64));
      micRef.current = mic;
      setStatus(`● Recording via ${mic.label} — speak in vi / en / zh.`);
    } catch (e) {
      setStatus("Microphone blocked: " + (e as Error).message);
      socket.close();
      socketRef.current = null;
      return;
    }
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
    setPaused(next);
    setStatus(next ? "Paused — resume to continue the same session." : "● Listening — streaming ASR + translation.");
  }

  function archiveCurrent(): ArchivedSession | null {
    if (rows.length === 0) return null;
    const s: ArchivedSession = {
      id: String(Date.now()),
      startedAt: (anchorRef.current ?? Date.now() / 1000) * 1000,
      seconds: elapsedRef.current,
      utterances: rows.map((r) => ({
        time: formatTime(r.at),
        speaker: speakerName(r.speaker),
        text: r.text,
        translations: r.translations,
      })),
    };
    setArchive((prev) => {
      const next = [s, ...prev].slice(0, 50);
      saveArchive(next);
      return next;
    });
    return s;
  }

  function stopSession() {
    if (!liveRef.current) return;
    try {
      micRef.current?.stop();
    } catch {
      /* noop */
    }
    micRef.current = null;
    try {
      socketRef.current?.stop();
    } catch {
      /* noop */
    }
    window.setTimeout(() => {
      socketRef.current?.close();
      socketRef.current = null;
    }, 4000);
    archiveCurrent();
    setLive(false);
    setPaused(false);
    setPartial(null);
    setStatus("Stopped — session archived to Library.");
  }

  function newSession() {
    if (liveRef.current) stopSession();
    else archiveCurrent();
    setRows([]);
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
      setStatus(`Done — ${list.length} utterances.`);
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
        <div className="brand-lockup"><div className="brand-mark"><Icon name="spark" size={18} stroke={1.7} /></div><span>ConfLive</span></div>
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
            toggleTranslation={() => setShowTranslation((v) => !v)}
            visibleRows={visibleRows} partial={partial} tok={tok} displayLang={displayLang}
            speakerCount={speakerCount} sessionDate={sessionDate} status={status}
            health={health} targets={targets}
          />
        ) : activeView === "Library" ? (
          <LibraryView archive={archive} openId={openArchived} onOpen={setOpenArchived}
            onDelete={(id) => setArchive((prev) => { const n = prev.filter((s) => s.id !== id); saveArchive(n); return n; })}
            displayLang={displayLang} onNavigate={() => setActiveView("Live session")} />
        ) : activeView === "Glossary" ? (
          <GlossaryView terms={terms} onChange={setTerms} onNavigate={() => setActiveView("Live session")} />
        ) : activeView === "Translate" ? (
          <TranslateView displayLang={displayLang} terms={terms} status={status} setStatus={setStatus} />
        ) : (
          <SettingsView
            health={health} mics={mics} micId={micId} onMic={setMicId}
            srcChoice={srcChoice} onSrc={setSrcChoice} targets={targets} onTargets={setTargets}
            denoise={denoise} onDenoise={setDenoise} status={status}
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
  visibleRows: Row[]; partial: PartialMsg | null;
  tok: { id: number; text: string } | null; displayLang: string;
  speakerCount: number; sessionDate: string; status: string;
  health: Health | null; targets: string[];
}) {
  const active = props.live && !props.paused;
  const fileRef = useRef<HTMLInputElement>(null);
  return <>
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
          <button className={`transcript-row ${props.selectedSegment === entry.id ? "selected" : ""}`} onClick={() => props.onSelectSegment(entry.id)} key={entry.id}>
            <time>{formatTime(entry.at)}</time>
            <span className={`speaker-avatar ${AVATAR_COLORS[speakerIdx(entry.speaker) % AVATAR_COLORS.length]}`}>{speakerInitials(entry.speaker)}</span>
            <span className="entry-copy">
              <span className="speaker-line"><b>{speakerName(entry.speaker)}</b></span>
              <span className="original-text">{entry.text}</span>
              {props.showTranslation && orderedTranslations(entry, props.displayLang).map(([code, text]) => (
                <span className="translated-text" key={code}>
                  <b className="tr-lang">{code}</b>
                  {text}
                </span>
              ))}
            </span>
            <span className="row-more"><Icon name="more" size={18} /></span>
          </button>
        ))}
        {props.partial && props.partial.text && (
          <div className="live-caption">
            <span className="caption-pulse" />
            <div className="caption-body">
              <p>{props.partial.text}</p>
              {props.showTranslation && props.partial.translations && orderedTranslations(
                { translations: props.partial.translations }, props.displayLang,
              ).map(([code, text]) => {
                const streaming = props.tok && props.tok.id === props.partial!.id && code === props.displayLang;
                const shown = streaming ? props.tok!.text : text;
                if (!shown) return null;
                return (
                  <p className="translated-text live-tr" key={code}>
                    <b className="tr-lang">{code}</b>
                    {shown}
                    {streaming ? <span className="typing-caret inline" /> : null}
                  </p>
                );
              })}
            </div>
            <span className="typing-caret" />
          </div>
        )}
        {props.visibleRows.length === 0 && !props.partial && <div className="empty-search">No moments yet — press Start recording or Upload audio.</div>}
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

function backendLabel(d: Health["diarizer_status"]): string {
  if (!d) return "volume";
  if (d.backend === "nemo-titanet") return `NeMo Titanet${d.device ? ` (${d.device})` : ""}`;
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

function LibraryView(props: {
  archive: ArchivedSession[]; openId: string | null; onOpen: (id: string | null) => void;
  onDelete: (id: string) => void; displayLang: string; onNavigate: () => void;
}) {
  const open = props.archive.find((s) => s.id === props.openId) ?? null;
  if (open) {
    return <section className="secondary-view wide">
      <p className="secondary-label">Archive · {new Date(open.startedAt).toLocaleString()}</p>
      <h1>{formatTime(open.seconds)} session</h1>
      <p>{open.utterances.length} segments.</p>
      <div className="archive-actions">
        <button onClick={() => props.onOpen(null)}>← Back to library</button>
        <button className="danger" onClick={() => { props.onDelete(open.id); props.onOpen(null); }}>Delete session</button>
      </div>
      <div className="transcript-list archive-list">
        {open.utterances.map((u, i) => (
          <div className="transcript-row" key={i}>
            <time>{u.time}</time>
            <span className="speaker-avatar plum">{(u.speaker || "?").replace("Speaker ", "S")}</span>
            <span className="entry-copy">
              <span className="speaker-line"><b>{u.speaker || "Speaker"}</b></span>
              <span className="original-text">{u.text || ""}</span>
              {orderedTranslations({ translations: archivedDict(u) }, props.displayLang).map(([code, text]) => (
                <span className="translated-text" key={code}>
                  <b className="tr-lang">{code}</b>
                  {text}
                </span>
              ))}
            </span>
          </div>
        ))}
      </div>
    </section>;
  }
  return <section className="secondary-view wide">
    <p className="secondary-label">Your archive</p>
    <h1>Every conversation, ready when you are.</h1>
    <p>Sessions stay on this device and are automatically organized by date and speaker.</p>
    <button onClick={props.onNavigate}>Open live session <Icon name="arrow" size={17} /></button>
    <div className="archive-list">
      {props.archive.length === 0 && <p className="archive-empty">No archived sessions yet — stop a session to save it here.</p>}
      {props.archive.map((s) => (
        <button className="archive-row" key={s.id} onClick={() => props.onOpen(s.id)}>
          <span className="archive-title">{new Date(s.startedAt).toLocaleString()}</span>
          <span className="archive-meta">{s.utterances.length} segments · {formatTime(s.seconds)}</span>
          <Icon name="arrow" size={16} />
        </button>
      ))}
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

function TranslateView(props: {
  displayLang: string; terms: Record<string, string>;
  status: string; setStatus: (s: string) => void;
}) {
  const [text, setText] = useState("");
  const [tgt, setTgt] = useState(props.displayLang);
  const [out, setOut] = useState("");
  const [busy, setBusy] = useState(false);
  async function go() {
    if (!text.trim() || busy) return;
    setBusy(true);
    props.setStatus("Translating phrase…");
    try {
      const r = await apiTranslateText(text.trim(), tgt, props.terms);
      setOut(r);
      props.setStatus("Ready.");
    } catch (e) {
      props.setStatus("Translate failed: " + (e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return <section className="secondary-view wide">
    <p className="secondary-label">Phrasebook</p>
    <h1>Type it, get it back translated.</h1>
    <p>Same Hy-MT2 engine as the live session, with your glossary applied. Enter to translate.</p>
    <div className="phrase-grid">
      <textarea
        placeholder="Type a phrase in any language…"
        value={text}
        rows={4}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void go(); } }}
      />
      <div className="phrase-row">
        <select value={tgt} onChange={(e) => setTgt(e.target.value)} aria-label="Target language">
          {DISPLAY_LANGS.map((l) => <option key={l.code} value={l.code}>{l.label}</option>)}
          {["fr", "de", "ja", "ko", "es", "pt", "ru", "th", "ar", "it"].map((c) => (
            <option key={c} value={c}>{c.toUpperCase()}</option>
          ))}
        </select>
        <button onClick={() => void go()} disabled={busy || !text.trim()}>
          {busy ? "Translating…" : "Translate"} <Icon name="arrow" size={17} />
        </button>
      </div>
      {out ? (
        <div className="translated-text phrase-out"><b className="tr-lang">{tgt}</b>{out}</div>
      ) : null}
    </div>
    <div className="secondary-wave"><Icon name="wave" size={46} stroke={1.25} /></div>
  </section>;
}

function SettingsView(props: {
  health: Health | null; mics: { id: string; label: string }[]; micId: string | undefined;
  onMic: (id: string | undefined) => void; srcChoice: string; onSrc: (v: string) => void;
  targets: string[]; onTargets: (t: string[]) => void;
  denoise: boolean; onDenoise: (v: boolean) => void; status: string;
  onRefresh: () => void; onDiarizer: (m: "pyannote" | "volume" | "nemo") => void; onNavigate: () => void;
}) {
  const h = props.health;
  const dBackend = h?.diarizer_status?.backend ?? "";
  const diarMode = dBackend === "pyannote" || h?.diarizer === "PyannoteDiarizer" ? "pyannote"
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
        <select value={diarMode} onChange={(e) => props.onDiarizer(e.target.value as "pyannote" | "volume" | "nemo")}>
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
    </div>
    <div className="backend-status">
      <p><b>Device:</b> {h ? `${h.device.device}${h.device.gpu_name ? ` · ${h.device.gpu_name}` : ""}` : "…"}</p>
      <p><b>ASR:</b> {h ? `${h.asr.model} · ready=${String(h.asr.ready)}` : "…"}</p>
      <p><b>MT:</b> {h ? `${h.mt.model} · ready=${String(h.mt.ready)}` : "…"}</p>
      <p><b>Diarizer:</b> {h ? backendLabel(h.diarizer_status) : "…"}</p>
      <p><b>Denoise:</b> {h ? `${h.denoise.backend} (Python 3.11 runtime)` : "…"}</p>
      <p className="status-line">{props.status}</p>
    </div>
    <div className="settings-actions">
      <button onClick={props.onRefresh}>Reload models <Icon name="arrow" size={17} /></button>
      <button className="ghost" onClick={props.onNavigate}>Open live session <Icon name="arrow" size={17} /></button>
    </div>
  </section>;
}
