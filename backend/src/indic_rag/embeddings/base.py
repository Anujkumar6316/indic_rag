"""Abstract Embedder interface (Strategy Pattern).

SOLID notes:
- ISP: ``supports_sparse`` keeps sparse-capable engines (BGE-M3) from forcing a
  null method onto dense-only engines; callers query capability instead.
- DIP: the pipeline and vector store depend only on this ABC.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from indic_rag.schemas import NormalizedChunk


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization with a zero-vector guard (no NaNs)."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


class BaseEmbedder(ABC):
    """Dense embedding contract shared by every provider adapter."""

    #: Public attribute every strategy must expose for collection sizing.
    dimension: int
    provider_name: str = "base"

    @abstractmethod
    def embed_chunks(self, chunks: List[NormalizedChunk]) -> np.ndarray:
        """Embed document chunks (with the provider's document-side prefix)."""

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query (with the provider's query-side prefix)."""

    # ------------------------------------------------------------- capabilities
    def supports_sparse(self) -> bool:
        return False

    def embed_sparse(self, texts: List[str]) -> list | None:  # noqa: UP006 - py310 compat
        """Learned-sparse weights; only BGE-M3 implements this meaningfully."""
        raise NotImplementedError(f"{self.provider_name} has no sparse encoder")
