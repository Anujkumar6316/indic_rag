"""Abstract vector-store contract (Dependency Inversion boundary).

The pipeline, retrieval graph, API and Studio all depend on this ABC only.
Qdrant is the production implementation; an in-memory store keeps dev/CI green.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from indic_rag.schemas import NormalizedChunk, ScoredChunk


class VectorStore(ABC):
    """Hybrid-capable store contract: dense + sparse search with payload filters."""

    backend_name: str = "abstract"

    @abstractmethod
    def upsert_batch(
        self,
        chunks: List[NormalizedChunk],
        dense_vectors: np.ndarray,
        sparse_vectors: list | None = None,
    ) -> int:
        """Idempotently write points; returns number of points upserted."""

    @abstractmethod
    def search_dense(
        self,
        query_vector: np.ndarray,
        limit: int = 20,
        lang: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[ScoredChunk]:
        ...

    @abstractmethod
    def search_bm25(
        self,
        query_text: str,
        limit: int = 20,
        lang: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[ScoredChunk]:
        """Sparse lexical search over BM25-encoded document text."""

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def scroll_all(self, limit: int = 10_000) -> List[NormalizedChunk]:
        """Dump stored chunks (used by evaluation scripts & studio inspection)."""
