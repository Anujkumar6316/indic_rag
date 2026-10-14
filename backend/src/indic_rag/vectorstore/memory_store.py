"""In-memory hybrid store used for dev, CI and single-process evaluation.

Satisfies the same ``VectorStore`` contract as Qdrant (LSP), combining cosine
similarity over dense vectors with a pure-Python Okapi-BM25 lexical index so
hybrid RRF search works fully offline.  Optional on-disk persistence to JSONL
keeps state across CLI stages within one container run.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

from indic_rag.schemas import NormalizedChunk, ScoredChunk
from indic_rag.vectorstore.base import VectorStore
from indic_rag.vectorstore.qdrant_store import _LocalBM25Index


class InMemoryVectorStore(VectorStore):
    backend_name = "in-memory"

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._chunks: dict[str, NormalizedChunk] = {}       # point_id -> chunk
        self._vectors: dict[str, np.ndarray] = {}           # point_id -> dense vec
        self._bm25 = _LocalBM25Index()
        self._lock = threading.RLock()
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.is_file():
            self._load()

    # ------------------------------------------------------------------ write
    def upsert_batch(
        self,
        chunks: List[NormalizedChunk],
        dense_vectors: np.ndarray,
        sparse_vectors: list | None = None,
    ) -> int:
        dense = np.asarray(dense_vectors, dtype=np.float32)
        with self._lock:
            for chunk, vec in zip(chunks, dense):
                pid = chunk.chunk_id
                self._chunks[pid] = chunk
                self._vectors[pid] = vec
                self._bm25.add(pid, chunk.text)
            self._persist()
        return len(chunks)

    # ------------------------------------------------------------------ read
    def _filter(self, lang: Optional[str], domain: Optional[str]) -> list[str]:
        return [
            pid
            for pid, c in self._chunks.items()
            if (lang is None or c.lang == lang) and (domain is None or c.domain == domain)
        ]

    def _scored(self, pid: str, score: float, rank: int, source: str) -> ScoredChunk:
        c = self._chunks[pid]
        return ScoredChunk(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            title=c.title,
            lang=c.lang,
            domain=c.domain,
            text=c.text,
            score=float(score),
            rank=rank,
            sources=[source],
            metadata=c.metadata,
        )

    def search_dense(
        self,
        query_vector: np.ndarray,
        limit: int = 20,
        lang: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[ScoredChunk]:
        with self._lock:
            ids = self._filter(lang, domain)
            if not ids:
                return []
            mat = np.stack([self._vectors[i] for i in ids])
            q = np.asarray(query_vector, dtype=np.float32)
            qn = np.linalg.norm(q) or 1.0
            sims = (mat @ q) / qn  # stored vectors are already L2-normalized
            order = np.argsort(-sims)[:limit]
            return [self._scored(ids[i], sims[i], r, "dense") for r, i in enumerate(order)]

    def search_bm25(
        self,
        query_text: str,
        limit: int = 20,
        lang: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[ScoredChunk]:
        with self._lock:
            allowed = set(self._filter(lang, domain))
            ranked = [
                (pid, s) for pid, s in self._bm25.search(query_text, limit * 4 + 20)
                if pid in allowed
            ][:limit]
            return [self._scored(pid, s, r, "bm25") for r, (pid, s) in enumerate(ranked)]

    def count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def scroll_all(self, limit: int = 10_000) -> List[NormalizedChunk]:
        with self._lock:
            return list(self._chunks.values())[:limit]

    # ------------------------------------------------------------ persistence
    def _persist(self) -> None:
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self.persist_path.open("w", encoding="utf-8") as fh:
            for pid, chunk in self._chunks.items():
                rec = chunk.model_dump()
                rec["_point_id"] = pid
                rec["_dense"] = self._vectors[pid].tolist()
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _load(self) -> None:
        with self.persist_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                dense = np.asarray(rec.pop("_dense"), dtype=np.float32)
                pid = rec.pop("_point_id")
                chunk = NormalizedChunk.model_validate(rec)
                self._chunks[pid] = chunk
                self._vectors[pid] = dense
                self._bm25.add(pid, chunk.text)
