"""Fast language ID for vi/en/zh (lingua, ms-level, lazy singleton).

Lets the pipeline skip MT entirely when a target matches the detected source
(e.g. Vietnamese speech -> Vietnamese row is a copy, not a 3 s generate).
Returns None when unsure — callers then translate everything (old behavior).
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger("conf.lid")

_lock = threading.Lock()
_detector = None
_failed = False

_MIN_CHARS = 4


def _get():
    global _detector, _failed
    if _detector is not None or _failed:
        return _detector
    with _lock:
        if _detector is not None or _failed:
            return _detector
        try:
            from lingua import Language, LanguageDetectorBuilder

            _detector = (
                LanguageDetectorBuilder.from_languages(
                    Language.ENGLISH, Language.VIETNAMESE, Language.CHINESE
                ).build(),
                {Language.ENGLISH: "en", Language.VIETNAMESE: "vi",
                 Language.CHINESE: "zh"},
            )
        except Exception as e:
            log.warning("LID unavailable (%s) — translating all targets.", e)
            _failed = True
    return _detector


def detect(text: str) -> str | None:
    """'vi' | 'en' | 'zh' | None (None = unsure, translate everything)."""
    text = (text or "").strip()
    if len(text) < _MIN_CHARS:
        return None
    det = _get()
    if det is None:
        return None
    detector, mapping = det
    try:
        lang = detector.detect_language_of(text)
    except Exception:
        return None
    return mapping.get(lang)
