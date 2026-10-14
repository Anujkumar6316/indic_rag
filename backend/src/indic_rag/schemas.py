"""Data contracts enforced across every step of the ingestion pipeline.

SOLID notes:
- SRP: pure data models, zero behaviour beyond validation/derivation.
- OCP: new payload fields can be added without touching pipeline code
  (``payload`` is a free-form dict carried through to Qdrant).
"""
from __future__ import annotations

import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SupportedLang = Literal["sa", "hi", "mr", "gu", "en"]

# Unicode blocks that legitimately carry our supported languages.
_DEVANAGARI = (0x0900, 0x097F)
_GUJARATI = (0x0A80, 0x0AFF)


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic chunk id so re-ingestion is idempotent."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}#{chunk_index}"))


class RawDocument(BaseModel):
    """A document as read from ``data/raw/`` before any processing."""

    model_config = ConfigDict(str_strip_whitespace=True)

    doc_id: str
    title: str = "none"
    lang: Optional[SupportedLang] = None
    domain: str = "general"
    raw_text: str
    source_path: Optional[str] = None

    @field_validator("raw_text")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("raw_text must contain non-whitespace content")
        return v


class FontAuditReport(BaseModel):
    """Outcome of the legacy-font gatekeeper for one file."""

    path: str
    is_legacy_font: bool = False
    detected_font: Optional[str] = None
    suspicious_ratio: float = 0.0
    suspicious_samples: list[str] = Field(default_factory=list)
    action: Literal["pass", "quarantine", "skip"] = "pass"


class NormalizedChunk(BaseModel):
    """A language-tagged, Unicode-normalized chunk ready for embedding."""

    chunk_id: str
    doc_id: str
    title: str
    lang: SupportedLang
    domain: str
    chunk_index: int
    text: str = Field(..., min_length=5)
    token_count: int = 0
    ends_with_danda: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorRecord(BaseModel):
    """One point written into the vector store."""

    point_id: str
    chunk: NormalizedChunk
    dense_dim: int
    provider: str


class SparseVector(BaseModel):
    """Backend-agnostic learned-sparse representation (BGE-M3 / BM25)."""

    indices: list[int]
    values: list[float]

    def as_dict(self) -> dict[str, list[float]]:
        return {"indices": self.indices, "values": self.values}


class ScoredChunk(BaseModel):
    """Retrieval result surfaced by the hybrid search / RRF fusion."""

    chunk_id: str
    doc_id: str
    title: str
    lang: str
    domain: str
    text: str
    score: float
    rank: int
    sources: list[str] = Field(default_factory=list)  # e.g. ["dense", "bm25"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionStats(BaseModel):
    """Batch-level counters returned by the orchestrator and CLI."""

    files_seen: int = 0
    files_quarantined: int = 0
    documents: int = 0
    chunks: int = 0
    embedded: int = 0
    upserted: int = 0
    errors: list[str] = Field(default_factory=list)


class GoldenQuery(BaseModel):
    """One entry of the cross-lingual Recall@5 benchmark."""

    query_id: str
    query: str
    query_lang: SupportedLang
    target_lang: SupportedLang
    target_doc_id: str
    target_chunk_ids: list[str] = Field(default_factory=list)


class RetrievalResponse(BaseModel):
    """API-level response envelope for search endpoints."""

    query: str
    results: list[ScoredChunk]
    provider: str
    took_ms: float
