"""``BAAI/bge-m3`` adapter — 1024d dense + learned sparse lexical weights.

BGE-M3 is the fallback engine that additionally emits *learned sparse*
representations, which we map onto Qdrant sparse vectors for exact-term hybrid
retrieval alongside the dense HNSW index.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from indic_rag.embeddings.base import BaseEmbedder, l2_normalize
from indic_rag.schemas import NormalizedChunk, SparseVector

logger = logging.getLogger("indic_rag.embeddings.bgem3")


class BGEM3Adapter(BaseEmbedder):
    provider_name = "bge-m3"
    DENSE_DIM = 1024

    def __init__(self, model_id: str = "BAAI/bge-m3", batch_size: int = 32) -> None:
        self.batch_size = batch_size
        self.dimension = self.DENSE_DIM
        self.model_id = model_id

        try:
            from FlagEmbedding import BGEM3FlagModel  # optional heavy dependency
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "BGE-M3 requires `pip install FlagEmbedding` (or switch provider to "
                "'embeddinggemma-2' / 'hash-fallback' in configs/settings.yaml)"
            ) from exc

        self.model = BGEM3FlagModel(model_id, use_fp16=False)  # fp16 disabled on purpose

    # ------------------------------------------------------------------ API
    def _encode(self, texts: List[str]) -> dict:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

    def embed_chunks(self, chunks: List[NormalizedChunk]) -> np.ndarray:
        out = self._encode([c.text for c in chunks])["dense_vecs"]
        vecs = np.asarray(out, dtype=np.float32)
        if np.isnan(vecs).any():
            raise RuntimeError("NaN detected in BGE-M3 embeddings.")
        return l2_normalize(vecs)

    def embed_query(self, query: str) -> np.ndarray:
        vec = np.asarray(self._encode([query])["dense_vecs"], dtype=np.float32)[0]
        if np.isnan(vec).any():
            raise RuntimeError("NaN detected in BGE-M3 query embedding.")
        return l2_normalize(vec.reshape(1, -1))[0]

    # ------------------------------------------------------------- sparse side
    def supports_sparse(self) -> bool:
        return True

    def embed_sparse(self, texts: List[str]) -> list[SparseVector]:
        """Learned-sparse lexical weights (token-id -> weight)."""
        raw = self._encode(texts)["lexical_weights"]
        result: list[SparseVector] = []
        for weights in raw:
            items = sorted((int(k), float(v)) for k, v in weights.items() if v > 0)
            result.append(
                SparseVector(indices=[i for i, _ in items], values=[w for _, w in items])
            )
        return result
