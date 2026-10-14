"""End-to-end batch ingestion orchestrator (Phase 1).

Graph of responsibilities (each step delegates to a single-purpose collaborator):

    data/raw/*.{txt,jsonl,md,pdf}
        │ FontAuditor          (gatekeeper: KrutiDev / Shree-Lipi → quarantine)
        ▼
    RawDocument                (schemas.RawDocument)
        │ LanguageDetector     (FastText lid.176.ftz or script-aware fallback)
        │ IndicTextNormalizer  (NFC + AI4Bharat + danda standardization)
        │ ShlokaAwareChunker   (॥/।-respecting token-aware splitter)
        ▼
    NormalizedChunk            (written to data/staging/*.jsonl)
        │ BaseEmbedder         (EmbeddingGemma-2 / BGE-M3 / hash-fallback)
        ▼
    dense vectors (+ sparse)   ──► VectorStore.upsert_batch (Qdrant hybrid)

SOLID notes:
- SRP: this module only sequences steps; every algorithm lives behind its own
  interface.
- DIP/OCP: collaborators are injected via the constructor; concrete strategies
  are chosen only by the factories in ``build_pipeline_components``.
- LSP: any ``VectorStore`` / ``BaseEmbedder`` implementation drops in unchanged.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List

import numpy as np

from indic_rag.config import Settings
from indic_rag.embeddings.base import BaseEmbedder
from indic_rag.ingestion.chunker import ShlokaAwareChunker
from indic_rag.ingestion.font_audit import FontAuditReport, FontAuditor
from indic_rag.ingestion.lang_detect import LanguageDetector
from indic_rag.ingestion.normalizer import IndicTextNormalizer
from indic_rag.schemas import IngestionStats, NormalizedChunk, RawDocument
from indic_rag.vectorstore.base import VectorStore

logger = logging.getLogger("indic_rag.pipeline")

_SUPPORTED_SUFFIXES = {".txt", ".jsonl", ".md"}


class IngestionPipeline:
    """Deterministic, idempotent batch ingestion for the five supported languages."""

    def __init__(
        self,
        settings: Settings,
        embedder: BaseEmbedder,
        store: VectorStore,
        detector: LanguageDetector,
        normalizer: IndicTextNormalizer | None = None,
        chunker: ShlokaAwareChunker | None = None,
        auditor: FontAuditor | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.detector = detector
        self.normalizer = normalizer or IndicTextNormalizer()
        self.chunker = chunker or ShlokaAwareChunker(
            chunk_size_tokens=settings.ingestion.chunk_size_tokens,
            chunk_overlap_tokens=settings.ingestion.chunk_overlap_tokens,
            separators=settings.ingestion.separators,
        )
        self.auditor = auditor or FontAuditor(
            quarantine_legacy_fonts=settings.ingestion.quarantine_legacy_fonts
        )

    # ------------------------------------------------------------- stage: audit
    def run_audit(self) -> list[FontAuditReport]:
        raw_dir = Path(self.settings.paths.raw_dir)
        qdir = Path(self.settings.paths.quarantine_dir)
        reports = self.auditor.run(raw_dir, qdir)
        for r in reports:
            logger.info(
                "audit %s -> %s (%s)", r.path, r.action, r.detected_font or "unicode-clean"
            )
        return reports

    # -------------------------------------------------------- stage: document IO
    @staticmethod
    def _iter_raw_documents(raw_dir: Path) -> Iterable[RawDocument]:
        """Yield RawDocuments from .txt/.md (one doc per file) and .jsonl (one per line)."""
        if not raw_dir.is_dir():
            return
        for path in sorted(raw_dir.rglob("*")):
            if not (path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if path.suffix.lower() == ".jsonl":
                for line_no, line in enumerate(text.splitlines(), start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        yield RawDocument(
                            doc_id=obj.get("doc_id") or f"{path.stem}-{line_no}",
                            title=obj.get("title", path.stem),
                            lang=obj.get("lang"),
                            domain=obj.get("domain", "general"),
                            raw_text=obj.get("raw_text") or obj.get("text", ""),
                            source_path=str(path),
                        )
                    except (json.JSONDecodeError, ValueError) as exc:
                        logger.warning("skipping %s:%s — %s", path.name, line_no, exc)
            else:
                try:
                    yield RawDocument(
                        doc_id=path.stem,
                        title=path.stem,
                        raw_text=text,
                        source_path=str(path),
                    )
                except ValueError as exc:
                    logger.warning("skipping empty file %s — %s", path.name, exc)

    # ---------------------------------------------------- stage: normalize+chunk
    def process_document(self, doc: RawDocument) -> List[NormalizedChunk]:
        sample = doc.raw_text[:2000]
        if doc.lang is None:
            lang, confidence = self.detector.detect(sample)
            logger.debug("detected lang=%s conf=%.2f for %s", lang, confidence, doc.doc_id)
        else:
            lang = doc.lang
        clean = self.normalizer.clean(doc.raw_text, lang)
        chunks = self.chunker.chunk_document(
            doc.model_copy(update={"lang": lang, "raw_text": clean}), lang
        )
        return chunks

    # ------------------------------------------------------------------ staging
    def write_staging(self, chunks: List[NormalizedChunk]) -> Path:
        staging = Path(self.settings.paths.staging_dir)
        staging.mkdir(parents=True, exist_ok=True)
        path = staging / "chunks.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")
        return path

    # -------------------------------------------------------------------- embed
    def embed_and_store(self, chunks: List[NormalizedChunk]) -> int:
        upserted = 0
        bs = max(1, self.settings.embedding.batch_size)
        use_sparse = self.embedder.supports_sparse()
        for i in range(0, len(chunks), bs):
            batch = chunks[i : i + bs]
            vecs: np.ndarray = self.embedder.embed_chunks(batch)
            sparse = self.embedder.embed_sparse([c.text for c in batch]) if use_sparse else None
            upserted += self.store.upsert_batch(batch, vecs, sparse_vectors=sparse)
        return upserted

    # ------------------------------------------------------------------ run all
    def run_ingest(self) -> IngestionStats:
        stats = IngestionStats()
        raw_dir = Path(self.settings.paths.raw_dir)
        docs = list(self._iter_raw_documents(raw_dir))
        stats.documents = len(docs)
        all_chunks: List[NormalizedChunk] = []
        for doc in docs:
            try:
                all_chunks.extend(self.process_document(doc))
            except Exception as exc:  # keep batch resilient
                stats.errors.append(f"{doc.doc_id}: {exc}")
                logger.exception("document %s failed", doc.doc_id)
        stats.chunks = len(all_chunks)
        if all_chunks:
            self.write_staging(all_chunks)
            stats.embedded = len(all_chunks)
            stats.upserted = self.embed_and_store(all_chunks)
        logger.info(
            "ingest done: docs=%s chunks=%s upserted=%s backend=%s",
            stats.documents, stats.chunks, stats.upserted, self.store.backend_name,
        )
        return stats

    def run_full(self) -> dict:
        """audit → ingest, returning both reports for CLI/API consumption."""
        reports = self.run_audit()
        stats = self.run_ingest()
        stats.files_seen = len(reports)
        stats.files_quarantined = sum(1 for r in reports if r.action == "quarantine")
        return {"audit": [r.model_dump() for r in reports], "stats": stats.model_dump()}


# ---------------------------------------------------------------------------
# Composition root — the ONLY place that knows every concrete strategy.
# ---------------------------------------------------------------------------
def build_pipeline_components(settings: Settings, force_memory_store: bool = False) -> dict:
    from indic_rag.embeddings.factory import build_embedder
    from indic_rag.ingestion.lang_detect import build_detector
    from indic_rag.vectorstore.factory import build_vector_store

    embedder = build_embedder(settings.embedding)
    store = build_vector_store(
        settings.vector_db, dense_dim=embedder.dimension, force_memory=force_memory_store
    )
    detector = build_detector(settings.paths.models_dir)
    return {"settings": settings, "embedder": embedder, "store": store, "detector": detector}


def build_pipeline(settings: Settings, force_memory_store: bool = False) -> IngestionPipeline:
    return IngestionPipeline(**build_pipeline_components(settings, force_memory_store))
