"""Script & language verification.

Primary engine: FastText ``lid.176.ftz`` (optional dependency).  When the model
file is absent (offline CI, slim container), we fall back to a deterministic
Unicode-block classifier that is *exactly* what Phase 1 needs for sa/hi/mr/gu/en
— Devanagari vs Gujarati vs Latin — and then disambiguates hi/mr/sa inside the
Devanagari block with high-frequency function-word profiles.

SOLID notes:
- DIP: pipeline depends on the ``LanguageDetector`` protocol, not on FastText.
- OCP: new engines plug in by implementing the protocol.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from indic_rag.schemas import SupportedLang

# Characteristic function words (unambiguous markers only).
_MARKER_WORDS: dict[SupportedLang, tuple[str, ...]] = {
    "sa": ("च", "अतः", "यत्", "तस्मात्", "भवति", "इति", "हे", "ो", "सः", "सा", "ते"),
    "hi": ("है", "और", "में", "से", "का", "की", "के", "यह", "कि", "लिए", "नहीं", "हुआ"),
    "mr": ("आहे", "आहो", "म्हणजे", "या", "झाले", "करीता", "नाही", "असते", "मराठी"),
    "gu": ("છે", "અને", "થાય", "કરવા", "માટે", "નથી", "છું", "છો", "ગુજરાતી"),
    "en": ("the", "and", "of", "to", "is", "in", "for", "with", "that", "as"),
}

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_GUJARATI = re.compile(r"[\u0A80-\u0AFF]")
_LATIN = re.compile(r"[A-Za-z]")


class LanguageDetector(Protocol):
    """Abstract detector contract (Strategy Pattern)."""

    def detect(self, text: str) -> tuple[SupportedLang, float]:
        """Return (language_code, confidence in [0, 1])."""

    def script_of(self, text: str) -> str:
        """Dominant Unicode script name of the buffer."""


def script_share(text: str) -> dict[str, float]:
    total = max(len(re.sub(r"\s", "", text)), 1)
    dev = len(_DEVANAGARI.findall(text))
    guj = len(_GUJARATI.findall(text))
    lat = len(_LATIN.findall(text))
    return {
        "devanagari": dev / total,
        "gujarati": guj / total,
        "latin": lat / total,
    }


class ScriptAwareFallbackDetector:
    """Deterministic Unicode-block + marker-word detector (no ML weights)."""

    def script_of(self, text: str) -> str:
        shares = script_share(text)
        return max(shares, key=shares.get) if any(shares.values()) else "unknown"

    def detect(self, text: str) -> tuple[SupportedLang, float]:
        shares = script_share(text)
        dominant_script = max(shares, key=shares.get)

        if dominant_script == "gujarati" and shares["gujarati"] >= 0.25:
            return "gu", round(min(1.0, shares["gujarati"] * 2), 3)
        if dominant_script == "latin" and shares["latin"] >= 0.6:
            return "en", round(min(1.0, shares["latin"]), 3)
        if dominant_script == "devanagari" and shares["devanagari"] >= 0.25:
            return self._disambiguate_devanagari(text)
        # Mixed / unknown — treat as English only when latin clearly leads.
        if shares["latin"] >= shares["devanagari"]:
            return "en", round(max(shares["latin"], 0.4), 3)
        return self._disambiguate_devanagari(text)

    @staticmethod
    def _disambiguate_devanagari(text: str) -> tuple[SupportedLang, float]:
        lowered = text
        scores: dict[str, int] = {"sa": 0, "hi": 0, "mr": 0}
        for lang, markers in _MARKER_WORDS.items():
            if lang in scores:
                scores[lang] += sum(lowered.count(m) for m in markers)
        best = max(scores, key=scores.get)
        total = sum(scores.values()) or 1
        conf = scores[best] / total
        return best, round(min(0.5 + conf / 2, 0.99), 3)  # floor 0.5 confidence


class FastTextLanguageDetector:
    """FastText lid.176 wrapper with automatic fallback delegation."""

    MODEL_FILENAME = "lid.176.ftz"

    def __init__(self, model_path: str | Path, fallback: LanguageDetector | None = None) -> None:
        self.fallback = fallback or ScriptAwareFallbackDetector()
        self._model = None
        path = Path(model_path)
        if path.is_file():
            try:
                import fasttext  # optional dependency

                self._model = fasttext.load_model(str(path))
            except Exception:
                self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def script_of(self, text: str) -> str:
        return self.fallback.script_of(text)

    def detect(self, text: str) -> tuple[SupportedLang, float]:
        if self._model is None:
            return self.fallback.detect(text)
        clean = re.sub(r"\s+", " ", text)[:2048].strip()
        if not clean:
            return "en", 0.0
        labels, probs = self._model.predict(clean)
        code = labels[0].replace("__label__", "")
        prob = float(probs[0])
        mapping = {"sa": "sa", "hi": "hi", "mr": "mr", "gu": "gu", "en": "en"}
        if code in mapping:
            return mapping[code], round(prob, 3)
        # FastText said something outside our set — trust the script layer.
        fb_lang, fb_conf = self.fallback.detect(text)
        return fb_lang, round(fb_conf * 0.8, 3)


def build_detector(models_dir: str | Path) -> LanguageDetector:
    """Factory honouring DIP: callers receive the protocol, not a concrete class."""
    return FastTextLanguageDetector(Path(models_dir) / FastTextLanguageDetector.MODEL_FILENAME)
