"""Token-aware recursive splitter that respects shlokas (॥) and sentences (।).

Guarantees (acceptance criterion #2): a Sanskrit verse terminated by ``॥`` is
never split mid-verse unless the single verse itself exceeds ``chunk_size_tokens``.

SOLID notes:
- SRP: pure splitting; language tagging/normalization happen upstream.
- DIP/OCP: token counting is injected via ``TokenizerProtocol`` — swap in a
  HuggingFace tokenizer without touching the splitter logic.
"""
from __future__ import annotations

import re
from typing import Protocol, Sequence

from indic_rag.schemas import NormalizedChunk, RawDocument, make_chunk_id


class TokenizerProtocol(Protocol):
    def count(self, text: str) -> int: ...


class WhitespaceIndicTokenizer:
    """Offline-safe token counter: counts Indic orthographic units.

    Splits on whitespace *and* on dandas so that "गोविन्दं॥" counts as one
    word + boundary, approximating subword token counts within ~15%.
    """

    _UNIT = re.compile(r"\S+")

    def count(self, text: str) -> int:
        return len(self._UNIT.findall(text))


class ShlokaAwareChunker:
    VERSE_END = "॥"
    SENTENCE_END = "।"

    def __init__(
        self,
        chunk_size_tokens: int = 300,
        chunk_overlap_tokens: int = 45,
        separators: Sequence[str] | None = None,
        tokenizer: TokenizerProtocol | None = None,
    ) -> None:
        self.chunk_size = chunk_size_tokens
        self.overlap = chunk_overlap_tokens
        self.separators = list(separators or ["\n\n", "\n", self.VERSE_END, self.SENTENCE_END, ".", "?", "!"])
        self.tokenizer = tokenizer or WhitespaceIndicTokenizer()

    # ------------------------------------------------------------- internals
    def _split_by_separator(self, text: str, sep: str) -> list[str]:
        if not sep:
            return [text]
        if sep in (self.VERSE_END, self.SENTENCE_END):
            # Keep the delimiter attached to the *preceding* segment so verse
            # integrity survives the split ("...धীमहि ॥" stays one unit).
            parts = [p for p in re.split(rf"(?<={re.escape(sep)})\s*", text) if p.strip()]
            return parts
        raw_parts = [p for p in text.split(sep) if p.strip()]
        # Re-attach sentence terminators that plain ``split`` would drop, so a
        # downstream merge never has to glue fragments back across a "।".
        if sep in ".?!":
            # A trailing separator belongs to the LAST part, not to a dropped
            # empty fragment: keep every non-final occurrence.
            n_terms = len(raw_parts) - (0 if text.rstrip().endswith(sep) else 1)
            raw_parts = [p.rstrip() + sep for p in raw_parts[:n_terms]] + raw_parts[n_terms:]
        return raw_parts

    def _merge_units(self, units: Sequence[str]) -> list[str]:
        """Greedily pack atomic units into <= chunk_size chunks with overlap."""
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0
        for unit in units:
            unit_tokens = self.tokenizer.count(unit)
            if unit_tokens > self.chunk_size:
                # Oversized atomic unit (e.g. a single huge verse): hard-split
                # on whitespace as last resort, preserving the terminator tail.
                if current:
                    chunks.append(" ".join(current).strip())
                    current, current_tokens = [], 0
                chunks.extend(self._hard_split(unit))
                continue
            if current_tokens + unit_tokens > self.chunk_size and current:
                chunks.append(" ".join(current).strip())
                # Overlap: carry trailing units up to `overlap` tokens forward.
                carry: list[str] = []
                carry_tokens = 0
                for prev in reversed(current):
                    t = self.tokenizer.count(prev)
                    if carry_tokens + t > self.overlap:
                        break
                    carry.insert(0, prev)
                    carry_tokens += t
                current, current_tokens = carry, carry_tokens
            current.append(unit)
            current_tokens += unit_tokens
        if current:
            chunks.append(" ".join(current).strip())
        return [c for c in chunks if len(c.strip()) >= 5]

    def _hard_split(self, text: str) -> list[str]:
        words = text.split()
        out, buf, n = [], [], 0
        for w in words:
            if n + 1 > self.chunk_size and buf:
                out.append(" ".join(buf))
                buf, n = [], 0
            buf.append(w)
            n += 1
        if buf:
            out.append(" ".join(buf))
        return out

    # ---------------------------------------------------------------- public
    def split_text(self, text: str) -> list[str]:
        """Recursive separator descent, then greedy packing of atomic units."""
        # Verse integrity is a hard guarantee: if the buffer still exceeds the
        # chunk budget, split at ॥ boundaries FIRST — never mid-verse.
        if (
            self.VERSE_END in text
            and self.tokenizer.count(text) > self.chunk_size
            and self.VERSE_END in self.separators
        ):
            pieces = self._split_by_separator(text, self.VERSE_END)
            if len(pieces) > 1:
                rest = [s for s in self.separators if s != self.VERSE_END]
                nested: list[str] = []
                for piece in pieces:
                    if self.tokenizer.count(piece) <= self.chunk_size:
                        nested.append(piece)
                    else:
                        sub = ShlokaAwareChunker(
                            self.chunk_size, self.overlap, rest, self.tokenizer
                        )
                        nested.extend(sub.split_text(piece))
                return self._merge_units(nested)

        remaining_seps = [s for s in self.separators if s in text]
        if not remaining_seps:
            units = [text]
        else:
            # Highest-priority separator present decides this level's split.
            top_sep = next(s for s in self.separators if s in text)
            pieces = self._split_by_separator(text, top_sep)
            if len(pieces) == 1:
                units = pieces
            else:
                nested: list[str] = []
                rest = [s for s in self.separators if s != top_sep]
                for piece in pieces:
                    if self.tokenizer.count(piece) <= self.chunk_size:
                        nested.append(piece)
                    else:
                        sub = ShlokaAwareChunker(
                            self.chunk_size, self.overlap, rest, self.tokenizer
                        )
                        nested.extend(sub.split_text(piece))
                units = nested
        return self._merge_units(units)

    def chunk_document(self, doc: RawDocument, lang: str) -> list[NormalizedChunk]:
        chunks: list[NormalizedChunk] = []
        texts = self.split_text(doc.raw_text)
        for idx, text in enumerate(texts):
            chunks.append(
                NormalizedChunk(
                    chunk_id=make_chunk_id(doc.doc_id, idx),
                    doc_id=doc.doc_id,
                    title=doc.title,
                    lang=lang,  # type: ignore[arg-type]  (validated upstream)
                    domain=doc.domain,
                    chunk_index=idx,
                    text=text,
                    token_count=self.tokenizer.count(text),
                    ends_with_danda=text.rstrip().endswith((self.VERSE_END, self.SENTENCE_END)),
                    metadata={"source_path": doc.source_path},
                )
            )
        return chunks
