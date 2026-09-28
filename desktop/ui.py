"""Native desktop window (CustomTkinter): sidebar controls + chat-style transcript feed.

Neutral stone tones with AI-purple accent (#7C3AED). All inference runs in
desktop/worker.py; the mic in desktop/recorder.py. This module only touches
widgets, and only from the main thread (background results arrive via queue
and are drained by a polling loop).
"""
from __future__ import annotations

import logging
import tkinter.filedialog as filedialog
from pathlib import Path

import customtkinter as ctk

from app.audio_io import decode_bytes
from desktop.recorder import MicRecorder, list_input_devices

log = logging.getLogger("conf.ui")

PURPLE = "#7C3AED"
PURPLE_HOVER = "#6D28D9"
PURPLE_SOFT = "#EDE9FE"
BG = "#FAFAF9"
CARD = "#FFFFFF"
INK = "#1C1917"
MUTED = "#78716C"
LINE = "#E7E5E4"
TR_BG = "#F5F5F4"
SPEAKER_COLORS = ["#7C3AED", "#0EA5E9", "#F59E0B", "#10B981", "#EC4899"]

SRC_OPTIONS = {"auto-detect": None, "Tiếng Việt": "Vietnamese",
               "English": "English", "中文": "Chinese"}


def speaker_color(speaker: str) -> str:
    import re

    m = re.search(r"(\d+)", speaker or "")
    return SPEAKER_COLORS[(int(m.group(1)) - 1) % len(SPEAKER_COLORS)] if m else SPEAKER_COLORS[0]


class ConferenceApp(ctk.CTk):
    def __init__(self, ctx: dict):
        super().__init__()
        self.ctx = ctx
        cfg = ctx["cfg"]
        self.cfg = cfg
        self.pipe = ctx["pipe"]
        self.worker: object = ctx["worker"]

        ctk.set_appearance_mode("light")
        self.configure(fg_color=BG)
        self.title("ConfLive — Conference ASR + Translator")
        self.geometry("1200x780")
        self.minsize(980, 640)

        self.n_utterances = 0
        self._build_layout()
        self._build_sidebar()
        self._build_main()

        self.recorder: MicRecorder | None = None
        self.worker.start()
        self.worker.warmup()  # load models in background; UI stays responsive
        self._set_status("Starting — loading models in background…")
        self.after(80, self._pump_results)
        self.after(120, self._pump_meter)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- layout ------------------------------------------------------------
    def _build_layout(self):
        self.grid_columnconfigure(0, minsize=300)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def _build_sidebar(self):
        self.side = ctk.CTkScrollableFrame(self, fg_color=CARD, corner_radius=0, width=300)
        self.side.grid(row=0, column=0, sticky="nsew")
        brand = ctk.CTkLabel(self.side, text="●  ConfLive",
                             font=ctk.CTkFont(size=22, weight="bold"), text_color=INK, anchor="w")
        brand.pack(fill="x", padx=18, pady=(18, 0))
        ctk.CTkLabel(self.side, text="Zipformer zh-en-vi · Hy-MT2 FP8",
                     font=ctk.CTkFont(size=12), text_color=MUTED, anchor="w").pack(fill="x", padx=18)

        self.device_lbl = ctk.CTkLabel(self.side, text="device: …",
                                       font=ctk.CTkFont(size=12), text_color=PURPLE,
                                       fg_color=PURPLE_SOFT, corner_radius=8, anchor="w")
        self.device_lbl.pack(fill="x", padx=18, pady=(12, 4))
        dev = self.ctx.get("device", {})
        gpu = f" · {dev.get('gpu_name', '').split('(')[0].strip()}" if dev.get("gpu_name") else ""
        self.device_lbl.configure(text=f"device: {dev.get('device', '?')}{gpu}")

        ctk.CTkLabel(self.side, text="Microphone", font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=INK, anchor="w").pack(fill="x", padx=18, pady=(12, 2))
        self.mic_names = ["System default"]
        self.mic_ids: list[int | None] = [None]
        try:
            for i, name in list_input_devices():
                self.mic_names.append(name[:42])
                self.mic_ids.append(i)
        except Exception as e:
            log.warning("mic listing failed: %s", e)
        self.mic_var = ctk.StringVar(value=self.mic_names[0])
        ctk.CTkOptionMenu(self.side, values=self.mic_names, variable=self.mic_var,
                          fg_color=CARD, button_color=PURPLE, text_color=INK,
                          button_hover_color=PURPLE_HOVER).pack(fill="x", padx=18)

        ctk.CTkLabel(self.side, text="Spoken language", font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=INK, anchor="w").pack(fill="x", padx=18, pady=(12, 2))
        self.src_var = ctk.StringVar(value="auto-detect")
        ctk.CTkOptionMenu(self.side, values=list(SRC_OPTIONS), variable=self.src_var,
                          fg_color=CARD, button_color=PURPLE, text_color=INK,
                          button_hover_color=PURPLE_HOVER).pack(fill="x", padx=18)

        ctk.CTkLabel(self.side, text="Translate to", font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=INK, anchor="w").pack(fill="x", padx=18, pady=(12, 2))
        trow = ctk.CTkFrame(self.side, fg_color="transparent")
        trow.pack(fill="x", padx=18)
        self.cb_en = ctk.CTkCheckBox(trow, text="EN", onvalue=True, offvalue=False)
        self.cb_en.select()
        self.cb_en.pack(side="left", padx=(0, 10))
        self.cb_zh = ctk.CTkCheckBox(trow, text="中文", onvalue=True, offvalue=False)
        self.cb_zh.select()
        self.cb_zh.pack(side="left", padx=(0, 10))
        self.cb_vi = ctk.CTkCheckBox(trow, text="VI", onvalue=True, offvalue=False)
        self.cb_vi.pack(side="left")

        self.denoise_sw = ctk.CTkSwitch(self.side, text="🔇 DeepFilterNet denoise",
                                        progress_color=PURPLE)
        self.denoise_sw.select()
        self.denoise_sw.pack(fill="x", padx=18, pady=(12, 2))
        ctk.CTkLabel(self.side, text=f"ASR {self.ctx['asr'].model_id}\nMT {self.ctx['mt'].model_id}",
                     font=ctk.CTkFont(size=11), text_color=MUTED, anchor="w",
                     justify="left").pack(fill="x", padx=18, pady=(2, 0))

        self.btn_record = ctk.CTkButton(self.side, text="●  Record", fg_color=PURPLE,
                                        hover_color=PURPLE_HOVER, height=38,
                                        font=ctk.CTkFont(size=14, weight="bold"),
                                        command=self._on_record)
        self.btn_record.pack(fill="x", padx=18, pady=(14, 6))
        self.btn_stop = ctk.CTkButton(self.side, text="■  Stop", height=34, state="disabled",
                                      fg_color="#57534E", command=self._on_stop)
        self.btn_stop.pack(fill="x", padx=18, pady=(0, 6))
        brow = ctk.CTkFrame(self.side, fg_color="transparent")
        brow.pack(fill="x", padx=18, pady=(0, 4))
        ctk.CTkButton(brow, text="⇪ Upload", fg_color=CARD, text_color=INK, border_width=1,
                      border_color=LINE, command=self._on_upload).pack(side="left", expand=True,
                                                                       fill="x", padx=(0, 4))
        ctk.CTkButton(brow, text="Load models", fg_color=CARD, text_color=INK, border_width=1,
                      border_color=LINE, command=self.worker.warmup).pack(side="left", expand=True,
                                                                          fill="x", padx=(4, 0))
        erow = ctk.CTkFrame(self.side, fg_color="transparent")
        erow.pack(fill="x", padx=18, pady=(0, 4))
        ctk.CTkButton(erow, text=".txt", fg_color=CARD, text_color=INK, border_width=1,
                      border_color=LINE, command=lambda: self._on_export("txt")).pack(
                          side="left", expand=True, fill="x", padx=(0, 2))
        ctk.CTkButton(erow, text=".srt", fg_color=CARD, text_color=INK, border_width=1,
                      border_color=LINE, command=lambda: self._on_export("srt")).pack(
                          side="left", expand=True, fill="x", padx=(2, 2))
        ctk.CTkButton(erow, text="Clear", fg_color=CARD, text_color=INK, border_width=1,
                      border_color=LINE, command=self._on_clear).pack(
                          side="left", expand=True, fill="x", padx=(2, 0))

        self.status_lbl = ctk.CTkLabel(self.side, text="…", font=ctk.CTkFont(size=12),
                                       text_color=MUTED, anchor="w", justify="left",
                                       wraplength=250)
        self.status_lbl.pack(fill="x", padx=18, pady=(10, 18))

    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=16, pady=16)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(1, weight=1)

        head = ctk.CTkFrame(main, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ctk.CTkLabel(head, text="Live transcript",
                     font=ctk.CTkFont(size=20, weight="bold"), text_color=INK).pack(side="left")
        self.count_lbl = ctk.CTkLabel(head, text="0 utterances",
                                      font=ctk.CTkFont(size=13), text_color=MUTED)
        self.count_lbl.pack(side="left", padx=12)
        self.live_lbl = ctk.CTkLabel(head, text="● LIVE", font=ctk.CTkFont(size=13, weight="bold"),
                                     text_color="#16A34A")
        # packed only while recording

        self.feed = ctk.CTkScrollableFrame(main, fg_color=CARD, corner_radius=14,
                                           border_width=1, border_color=LINE)
        self.feed.grid(row=1, column=0, sticky="nsew")
        self._feed_empty = ctk.CTkLabel(self.feed, text="Press  ● Record  or  ⇪ Upload —\n"
                                                        "utterances appear here as speaker bubbles.",
                                        font=ctk.CTkFont(size=14), text_color=MUTED,
                                        justify="center")
        self._feed_empty.pack(expand=True, pady=60)

        meter = ctk.CTkFrame(main, fg_color="transparent")
        meter.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.meter = ctk.CTkProgressBar(meter, height=10, progress_color=PURPLE,
                                        fg_color=LINE, corner_radius=5)
        self.meter.set(0.0)
        self.meter.pack(side="left", fill="x", expand=True)
        self.meter_lbl = ctk.CTkLabel(meter, text="mic idle", width=110,
                                      font=ctk.CTkFont(size=12), text_color=MUTED)
        self.meter_lbl.pack(side="left", padx=(10, 0))

    # -- settings ----------------------------------------------------------
    def _settings(self) -> dict:
        targets = []
        if self.cb_en.get():
            targets.append("en")
        if self.cb_zh.get():
            targets.append("zh")
        if self.cb_vi.get():
            targets.append("vi")
        idx = self.mic_names.index(self.mic_var.get()) if self.mic_var.get() in self.mic_names else 0
        return {"targets": targets or ["en"],
                "src_lang": SRC_OPTIONS[self.src_var.get()],
                "denoise": bool(self.denoise_sw.get()),
                "device": self.mic_ids[idx]}

    # -- actions -----------------------------------------------------------
    def _on_record(self):
        if self.recorder is None:
            try:
                self.recorder = MicRecorder(self.worker.tasks, self._settings,
                                            samplerate=int(self.cfg.get("sample_rate", 16000)),
                                            frame_seconds=float(self.cfg.get("frame_seconds", 0.5)))
            except Exception as e:
                self._set_status(f"Mic unavailable: {e}")
                self.recorder = None
                return
        self.worker.submit(("stream_start", self._settings()))
        try:
            self.recorder.start()
        except Exception as e:
            self._set_status(f"Could not open microphone: {e}")
            return
        self.btn_record.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.live_lbl.pack(side="left", padx=8)
        self._set_status("● Recording — speak in vi / en / zh.")

    def _on_stop(self):
        if self.recorder is not None:
            self.recorder.stop()
        try:
            self.worker.submit(("stream_stop",))
        except Exception:
            pass
        self.btn_record.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.live_lbl.pack_forget()
        self.meter.set(0.0)
        self.meter_lbl.configure(text="mic idle")
        self._set_status("Stopped.")

    def _on_upload(self):
        path = filedialog.askopenfilename(title="Upload recording",
                                          filetypes=[("Audio", "*.wav *.mp3 *.ogg *.flac *.m4a *.webm"),
                                                     ("All files", "*.*")])
        if not path:
            return
        try:
            raw = Path(path).read_bytes()
            pcm, sr = decode_bytes(raw)
        except Exception as e:
            self._set_status(f"Could not decode audio: {e}")
            return
        s = self._settings()
        self.worker.submit(("file", pcm, sr, s["targets"], s["src_lang"], s["denoise"]))
        self._set_status(f"Transcribing {Path(path).name}…")

    def _on_export(self, kind: str):
        if not self.pipe.history:
            self._set_status("Nothing to export yet.")
            return
        path = filedialog.asksaveasfilename(defaultextension=f".{kind}",
                                            filetypes=[(kind.upper(), f"*.{kind}")])
        if not path:
            return
        text = self.pipe.export_txt() if kind == "txt" else self.pipe.export_srt()
        Path(path).write_text(text, encoding="utf-8")
        self._set_status(f"Exported {path}")

    def _on_clear(self):
        self._drop_partial()
        for w in self.feed.winfo_children():
            w.destroy()
        self.n_utterances = 0
        self.count_lbl.configure(text="0 utterances")
        self._feed_empty = ctk.CTkLabel(self.feed, text="Cleared — press  ● Record  for a new session.",
                                        font=ctk.CTkFont(size=14), text_color=MUTED)
        self._feed_empty.pack(expand=True, pady=60)

    def _on_close(self):
        try:
            if self.recorder is not None:
                self.recorder.stop()
            self.worker.submit(("stop",))
        except Exception:
            pass
        self.destroy()

    # -- background result pump (main thread only) --------------------------
    def _pump_results(self):
        for _ in range(25):
            try:
                msg = self.worker.results.get_nowait()
            except Exception:
                break
            kind, payload = msg[0], msg[1]
            if kind == "status":
                self._set_status(payload)
            elif kind == "ready":
                a, m, d = payload["asr"], payload["mt"], payload.get("denoise", {})
                self.device_lbl.configure(
                    text=f"{self.ctx['device']['device']} · ASR {a['backend']} · "
                         f"MT ready · denoise {d.get('backend', '?')}")
                self._set_status(f"Ready — ASR {a['model']} · MT {m['model']}. Press Record.")
            elif kind == "utterance":
                self._finalize_partial(payload)
                self._add_bubble(payload)
            elif kind == "partial":
                self._upsert_partial(payload)
            elif kind == "file_done":
                self._set_status(f"Done — {payload} utterances.")
            elif kind == "error":
                self._set_status(f"Error: {payload}")
        self.after(80, self._pump_results)

    def _pump_meter(self):
        if self.recorder is not None and self.recorder.running:
            lv = self.recorder.level
            self.meter.set(lv)
            self.meter_lbl.configure(text="listening…" if lv > 0.02 else "quiet…")
        self.after(120, self._pump_meter)

    # -- transcript bubbles --------------------------------------------------
    def _clear_empty(self):
        if self._feed_empty is not None:
            try:
                self._feed_empty.destroy()
            except Exception:
                pass
            self._feed_empty = None

    def _make_bubble(self, speaker: str, meta: str):
        """Shell bubble; returns (frame, body, orig_label, who_label)."""
        self._clear_empty()
        color = speaker_color(speaker)
        bubble = ctk.CTkFrame(self.feed, fg_color=CARD, corner_radius=12,
                              border_width=1, border_color=LINE)
        bubble.pack(fill="x", padx=10, pady=6)
        strip = ctk.CTkFrame(bubble, fg_color=color, width=6, corner_radius=6)
        strip.pack(side="left", fill="y", padx=(0, 0), pady=0)
        body = ctk.CTkFrame(bubble, fg_color="transparent")
        body.pack(side="left", fill="x", expand=True, padx=12, pady=10)
        who = ctk.CTkLabel(body, text=meta, font=ctk.CTkFont(size=11, weight="bold"),
                           text_color=color, anchor="w")
        who.pack(fill="x")
        orig = ctk.CTkLabel(body, text="", font=ctk.CTkFont(size=14), text_color=INK,
                            anchor="w", justify="left", wraplength=660)
        orig.pack(fill="x", pady=(2, 0))
        return bubble, body, orig, who

    def _add_tr_label(self, body):
        lb = ctk.CTkLabel(body, text="", font=ctk.CTkFont(size=13), text_color="#44403C",
                          fg_color=TR_BG, corner_radius=8, anchor="w", justify="left",
                          wraplength=660)
        lb.pack(fill="x", pady=(6, 0))
        return lb

    def _add_bubble(self, u: dict):
        badge = f" · 🔇" if u.get("denoised") else ""
        bubble, body, orig, _ = self._make_bubble(
            u.get("speaker", ""),
            f"{u.get('speaker','')} · {u.get('src_lang','')} · {u.get('rms_db','')} dB{badge}")
        tr_labels = []
        for code, text in (u.get("translations") or {}).items():
            lb = self._add_tr_label(body)
            tr_labels.append((lb, f"[{code}] {text}"))
        self.n_utterances += 1
        self.count_lbl.configure(text=f"{self.n_utterances} utterances")
        self._scroll_bottom()
        # streaming typewriter for the original, then reveal translations
        self._typewriter(orig, u.get("text", ""), lambda: self._reveal(tr_labels))

    # -- live partial bubble (streaming ASR + translation) ----------------------
    def _upsert_partial(self, p: dict):
        cur = getattr(self, "_partial", None)
        if cur is not None and cur.get("id") != p.get("id"):
            self._drop_partial()  # stale (finalize should have beaten it here)
            cur = None
        if cur is None:
            bubble, body, orig, who = self._make_bubble(
                p.get("speaker", ""), f"{p.get('speaker','')} · listening…")
            self._partial = {"id": p.get("id"), "frame": bubble, "body": body,
                             "orig": orig, "who": who, "trs": []}
            cur = self._partial
        try:
            cur["orig"].configure(text=p.get("text", ""))
            cur["who"].configure(
                text=f"{p.get('speaker','')} · listening… · {p.get('rms_db','')} dB")
            for i, (code, text) in enumerate((p.get("translations") or {}).items()):
                if i < len(cur["trs"]):
                    cur["trs"][i].configure(text=f"[{code}] {text}")
                else:
                    lb = self._add_tr_label(cur["body"])
                    lb.configure(text=f"[{code}] {text}")
                    cur["trs"].append(lb)
        except Exception:
            pass
        self._scroll_bottom()

    def _drop_partial(self):
        cur = getattr(self, "_partial", None)
        if cur is not None:
            try:
                cur["frame"].destroy()
            except Exception:
                pass
            self._partial = None

    def _finalize_partial(self, u: dict):
        cur = getattr(self, "_partial", None)
        if cur is not None and cur.get("id") == u.get("id"):
            self._drop_partial()
        elif cur is not None:
            self._drop_partial()  # orphan partial (e.g. after Clear)

    def _typewriter(self, label, text: str, done=None, i: int = 0):
        step = max(1, len(text) // 60 + 1)
        nxt = min(len(text), i + step)
        try:
            label.configure(text=text[:nxt])
        except Exception:
            return
        self._scroll_bottom()
        if nxt < len(text):
            self.after(12, lambda: self._typewriter(label, text, done, nxt))
        elif done:
            done()

    def _reveal(self, pairs):
        for lb, text in pairs:
            try:
                lb.configure(text=text)
            except Exception:
                pass
        self._scroll_bottom()

    def _scroll_bottom(self):
        try:
            self.feed.update_idletasks()
            self.feed._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    def _set_status(self, msg: str):
        try:
            self.status_lbl.configure(text=msg)
        except Exception:
            pass
