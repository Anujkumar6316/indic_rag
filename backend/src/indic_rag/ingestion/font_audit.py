"""Legacy-font gatekeeper.

Detects text extracted from PDFs/typesetting systems that used ASCII-mapped
legacy Indic fonts (KrutiDev, Shree-Lipi/Akshar Unicode, Gujarati Shraddha,
Mangal-ASCII hacks).  Such files store Devanagari/Gujarati as *Latin code
points* with quirky consonant clusters, so they must be quarantined before
they poison the embedding index.

SOLID notes:
- SRP: only classification + quarantine mechanics live here.
- OCP: add new font signatures by appending to ``_FONT_SIGNATURES`` — no logic
  changes required.
- LSP: ``audit_text`` is pure and side-effect free; file movement lives in
  ``quarantine_file``.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from indic_rag.schemas import FontAuditReport

# Each entry: (font_name, regex over characteristic ASCII bigrams/trigrams).
# Signatures are deliberately conservative: they require several hits of
# KrutiDev/Shree-Lipi-only clusters such as "z" => ज़, "K^" => क्ष,
# "[~]" => ऱ्, "vkk" => व্ক which essentially never occur in English prose.
_FONT_SIGNATURES: list[tuple[str, re.Pattern[str]]] = [
    (
        "KrutiDev",
        re.compile(
            r"(?:"
            r"[aeoAEO][kghjdttnpdmb][hmlrsv]"   # "Bkr", "dkh", "gny"... style clusters
            r"|[Kk]\^|z[Z^]|[\[\]][~`]"          # क्ष, ज्ञ/ज़ markers, ऱ् markers
            r"|y[yk]"                            # य्य hybrid conjunct marker
            r")"
        ),
    ),
    (
        "Shree-Lipi / Akshar Unicode",
        re.compile(
            r"(?:"
            r"[Hh][iI][tT]"                     # हित् artefact
            r"|R[iI]"                           # रि prefix hack
            r"|[Ll]{2}[pvmbtk]"                 # ल्ल cluster hack
            r")"
        ),
    ),
    (
        "Gujarati Shraddha ASCII",
        re.compile(r"(?:[jJ][^\x00-\x7F]?[~^]|[Ss][~^][a-z])"),
    ),
]

# Characters that legitimately indicate proper Unicode Indic text.
_INDIC_UNICODE = re.compile(r"[\u0900-\u097F\u0A80-\u0AFF\u0C00-\u0C7F]")

# Suspicious ASCII tokens: heavy punctuation/number noise typical of bad OCR
# of legacy-font PDFs (e.g. "¤", "¦", "«", repeated digits inside words).
_NOISE_CHARS = set("¤¦«»¬¯°±§¶")


def audit_text(path: str | Path, text: str) -> FontAuditReport:
    """Classify a decoded text buffer as Unicode-clean or legacy-ASCII-mapped."""
    sample = text[:200_000]
    if not sample.strip():
        return FontAuditReport(path=str(path), action="skip")

    # If the text already carries real Devanagari/Gujarati code points, it is
    # Unicode-encoded and cannot be an ASCII-mapped legacy font.
    indic_hits = len(_INDIC_UNICODE.findall(sample))
    latin_letters = sum(c.isascii() and c.isalpha() for c in sample)
    total_alpha = indic_hits + latin_letters or 1

    best_font: str | None = None
    best_score = 0.0
    best_samples: list[str] = []
    for font, pattern in _FONT_SIGNATURES:
        matches = pattern.findall(sample)
        score = len(matches) / max(total_alpha, 1)
        if score > best_score:
            best_score, best_font = score, font
            best_samples = [m if isinstance(m, str) else m[0] for m in matches][:10]

    noise_ratio = sum(1 for c in sample if c in _NOISE_CHARS) / max(len(sample), 1)

    # Threshold tuned so genuine English/Latin documents pass untouched.
    is_legacy = (best_score >= 0.004 and best_samples) and indic_hits == 0
    is_legacy = is_legacy or noise_ratio > 0.01

    return FontAuditReport(
        path=str(path),
        is_legacy_font=bool(is_legacy),
        detected_font=best_font if is_legacy else None,
        suspicious_ratio=round(max(best_score, noise_ratio), 5),
        suspicious_samples=best_samples[:10],
        action="quarantine" if is_legacy else "pass",
    )


def quarantine_file(src: Path, quarantine_dir: Path) -> Path:
    """Move a flagged file into ``data/quarantine/`` preserving its name."""
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    dest = quarantine_dir / src.name
    if dest.exists():  # avoid clobbering on repeated runs
        dest = quarantine_dir / f"{src.stem}.dup{len(list(quarantine_dir.glob(src.name)))}{src.suffix}"
    shutil.move(str(src), str(dest))
    return dest


class FontAuditor:
    """Directory-level orchestration around the pure functions above."""

    SUPPORTED_SUFFIXES = {".txt", ".jsonl", ".md"}

    def __init__(self, quarantine_legacy_fonts: bool = True) -> None:
        self.quarantine_enabled = quarantine_legacy_fonts

    def iter_raw_files(self, raw_dir: Path):
        for p in sorted(raw_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_SUFFIXES | {".pdf"}:
                yield p

    @staticmethod
    def _extract_text(path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            try:  # optional dependency
                from pypdf import PdfReader

                reader = PdfReader(str(path))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception:
                return ""
        return path.read_text(encoding="utf-8", errors="replace")

    def run(self, raw_dir: Path, quarantine_dir: Path) -> list[FontAuditReport]:
        reports: list[FontAuditReport] = []
        for file_path in self.iter_raw_files(raw_dir):
            text = self._extract_text(file_path)
            report = audit_text(file_path, text)
            if report.action == "quarantine" and self.quarantine_enabled:
                moved = quarantine_file(file_path, quarantine_dir)
                report.path = str(moved)
            reports.append(report)
        return reports
