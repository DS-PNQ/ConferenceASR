"""Offline text-to-speech for the speaker buttons: sherpa-onnx Piper VITS, CPU, no VRAM.

One int8 voice per language in tts_dir/<lang>/ (scripts/get_tts.py fills it):
vi vais1000-medium, en lessac-medium, zh chaowen-medium (14-22 MB each).
A voice loads on its first request (~1.3 s, ~100-120 MB RAM) and every voice
is dropped after tts_idle_unload seconds without a request.
"""
from __future__ import annotations

import io
import logging
import threading
from pathlib import Path

log = logging.getLogger("conf.tts")


class PiperTTS:
    def __init__(self, cfg: dict):
        self.dir = Path(cfg.get("tts_dir") or "models/tts")
        self.threads = int(cfg.get("tts_threads", 2))
        self.idle = float(cfg.get("tts_idle_unload", 120))
        self._voices: dict = {}
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def langs(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        return sorted(d.name for d in self.dir.iterdir() if d.is_dir() and any(d.glob("*.onnx")))

    def _voice(self, lang: str):
        v = self._voices.get(lang)
        if v is not None:
            return v
        d = self.dir / lang
        onnx = next(d.glob("*.onnx"), None) if d.is_dir() else None
        if onnx is None:
            raise LookupError(f"no TTS voice for '{lang}' (run scripts/get_tts.py)")
        import sherpa_onnx

        f = lambda name: str(d / name) if (d / name).exists() else ""  # noqa: E731
        vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(onnx), tokens=f("tokens.txt"), lexicon=f("lexicon.txt"),
            data_dir=f("espeak-ng-data"))  # en/vi: espeak phonemes; zh: lexicon + rule FSTs
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=self.threads, provider="cpu"),
            rule_fsts=",".join(p for p in (f("phone.fst"), f("date.fst"), f("number.fst")) if p))
        v = self._voices[lang] = sherpa_onnx.OfflineTts(cfg)
        log.info("TTS voice loaded: %s (%s)", lang, onnx.name)
        return v

    def synth(self, text: str, lang: str) -> bytes:
        """-> 16-bit mono WAV bytes. LookupError when the language has no voice."""
        import numpy as np
        import soundfile as sf

        if text.isupper():
            text = text.lower()  # same-language copies are raw UPPERCASE ASR; espeak would spell them
        with self._lock:
            audio = self._voice(lang).generate(text, sid=0, speed=1.0)
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.idle, self.unload)
            self._timer.daemon = True
            self._timer.start()
        buf = io.BytesIO()
        sf.write(buf, np.asarray(audio.samples, dtype=np.float32), audio.sample_rate,
                 format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def unload(self) -> dict:
        with self._lock:
            was = sorted(self._voices)
            self._voices.clear()
        if was:
            log.info("TTS voices unloaded: %s", ", ".join(was))
        return {"unloaded": was}

    def status(self) -> dict:
        return {"backend": "sherpa-onnx piper", "device": "cpu", "langs": self.langs(),
                "loaded": sorted(self._voices)}


if __name__ == "__main__":  # python -m app.tts_engine  (needs scripts/get_tts.py voices)
    t = PiperTTS({"tts_idle_unload": 5})
    for lang, s in (("en", "Good morning."), ("vi", "Xin chào."), ("zh", "大家好。")):
        if lang in t.langs():
            wav = t.synth(s, lang)
            assert wav[:4] == b"RIFF" and len(wav) > 8000, lang
            print(lang, len(wav), "bytes ok")
    try:
        t.synth("Bonjour", "fr")
        raise AssertionError("fr should have no voice")
    except LookupError:
        print("missing voice -> LookupError ok")
