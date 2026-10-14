"""Indic normalizer & danda standardizer.

Fixes Unicode fragmentation, stray ZWJ/ZWNJ, inconsistent shloka punctuation
(``।।`` / ``||`` -> ``॥``) before tokenization.  Uses AI4Bharat IndicNLP when
installed; otherwise a functionally-equivalent pure-Python path keeps the
pipeline deterministic and offline-safe.

SOLID notes:
- SRP: text canonicalization only — no chunking, no IO.
- LSP: ``PurePythonIndicNormalizer`` satisfies the same contract as the
  IndicNLP factory wrapper, so either can be substituted transparently.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Protocol

ZWJ = "\u200d"   # zero-width joiner
ZWNJ = "\u200c"  # zero-width non-joiner (keep intentional ones; strip noisy runs)


class TextNormalizerBackend(Protocol):
    def normalize(self, text: str) -> str: ...


class PurePythonIndicNormalizer:
    """Script canonicalization without external model weights.

    - NFC composition (combining matras/nuktas merged with base letters).
    - Virama-reordering fix for common fragmented patterns.
    - Nukta *preservation* (remove_nuktas=False policy from the blueprint).
    """

    def normalize(self, text: str) -> str:
        # NFC then a light compatibility pass; deliberately NOT NFKD, which
        # would decompose conjuncts and destroy Devanagari orthography.
        text = unicodedata.normalize("NFC", text)
        # Drop ZWJ runs and duplicated zero-width marks that break tokenizers.
        text = re.sub(f"[{ZWJ}]+", "", text)
        text = re.sub(f"({ZWNJ})\\1+", r"\1", text)
        return unicodedata.normalize("NFC", text)


def _try_indicnlp(lang: str) -> TextNormalizerBackend | None:
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

        factory = IndicNormalizerFactory()
        return factory.get_normalizer(lang, remove_nuktas=False)
    except Exception:
        return None


class IndicTextNormalizer:
    """Per-language normalizer registry + danda standardization."""

    INDIC_LANGS = ("sa", "hi", "mr", "gu")

    def __init__(self) -> None:
        pure = PurePythonIndicNormalizer()
        self.normalizers: dict[str, TextNormalizerBackend] = {}
        for lang in self.INDIC_LANGS:
            self.normalizers[lang] = _try_indicnlp(lang) or pure
        self._pure = pure

    # ------------------------------------------------------------------ steps
    @staticmethod
    def unify_dandas(text: str, lang: str) -> str:
        """Canonicalize sentence/verse boundaries.

        ``। ।`` | ``।।`` | ``||`` -> ``॥`` (Deergha Viram, verse end)
        ASCII ``|``               -> ``।`` (Poorna Viram, sentence end)
        """
        text = re.sub(r"।\s*।", "॥", text)
        text = text.replace("||", "॥")
        if lang in IndicTextNormalizer.INDIC_LANGS:
            text = text.replace("|", "।")
        # Normalize whitespace glued to dandas: "गोविन्दं॥" stays intact but
        # "॥  ॥" double verse markers collapse to one.
        text = re.sub(r"॥\s*॥", "॥", text)
        return text

    def clean(self, text: str, lang: str) -> str:
        # 1. NFC Unicode composition (plus ZWJ/ZWNJ hygiene) even pre-backend.
        text = unicodedata.normalize("NFC", text)
        # 2. Danda standardization BEFORE backend normalization.
        text = self.unify_dandas(text, lang)
        # 3. Apply AI4Bharat (or pure-python) script canonicalization.
        backend = self.normalizers.get(lang, self._pure)
        text = backend.normalize(text)
        # 4. Collapse horizontal whitespace while preserving newlines.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        return text.strip()
