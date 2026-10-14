"""Hybrid retrieval service: dense + BM25 sparse search fused with RRF.

SOLID notes:
- SRP: fusion & orchestration only; vector math lives in the store, prefixes
  live in the embedder strategies.
- DIP: depends on the ``BaseEmbedder`` and ``VectorStore`` abstractions.
- OCP: add new rankers by extending ``ReciprocalRankFusion`` subclasses
  without touching ``HybridRetriever``.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from indic_rag.embeddings.base import BaseEmbedder
from indic_rag.schemas import ScoredChunk
from indic_rag.vectorstore.base import VectorStore

logger = logging.getLogger("indic_rag.retrieval")


class ReciprocalRankFusion:
    """Cormack et al. RRF: score(d) = Σ_lists 1 / (k + rank)."""

    #: Labels attached to each ranking list passed to ``fuse`` (in order).
    LIST_NAMES = ("dense", "sparse")

    def __init__(self, k: int = 60) -> None:
        self.k = max(1, k)

    def fuse(self, *rankings: List[ScoredChunk]) -> List[ScoredChunk]:
        fused: dict[str, float] = {}
        best: dict[str, ScoredChunk] = {}
        for ranking in rankings:
            for hit in ranking:
                key = hit.chunk_id
                fused[key] = fused.get(key, 0.0) + 1.0 / (self.k + hit.rank)
                if key not in best or len(hit.text) > len(best[key].text):
                    best[key] = hit
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        results: List[ScoredChunk] = []
        for rank, (cid, score) in enumerate(ordered, start=1):
            contributing = [
                name
                for name, r in zip(self.LIST_NAMES, rankings)
                if any(h.chunk_id == cid for h in r)
            ]
            chunk = best[cid].model_copy(
                update={"score": round(score, 6), "rank": rank, "sources": contributing}
            )
            results.append(chunk)
        return results


class HybridRetriever:
    """Query-side pipeline: embed → parallel dense/BM25 recall → RRF fusion."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: VectorStore,
        fusion: ReciprocalRankFusion | None = None,
        top_k_dense: int = 20,
        top_k_sparse: int = 20,
        final_k: int = 5,
        enable_bm25: bool = True,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.fusion = fusion or ReciprocalRankFusion()
        self.top_k_dense = top_k_dense
        self.top_k_sparse = top_k_sparse
        self.final_k = final_k
        self.enable_bm25 = enable_bm25

    def search(
        self,
        query: str,
        lang: Optional[str] = None,
        domain: Optional[str] = None,
        final_k: Optional[int] = None,
    ) -> List[ScoredChunk]:
        k = final_k or self.final_k

        q_vec = self.embedder.embed_query(query)
        dense_hits = self.store.search_dense(q_vec, limit=self.top_k_dense, lang=lang, domain=domain)

        rankings = [dense_hits]
        if self.enable_bm25:
            try:
                sparse_hits = self.store.search_bm25(
                    query, limit=self.top_k_sparse, lang=lang, domain=domain
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Sparse search failed (%s); dense-only fallback.", exc)
                sparse_hits = []
            rankings.append(sparse_hits)

        fused = self.fusion.fuse(*rankings)
        return fused[:k]
