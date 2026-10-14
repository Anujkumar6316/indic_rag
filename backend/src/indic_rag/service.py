"""Application service container — lazy composition root for every entrypoint.

FastAPI, Gradio Studio, LangGraph and the CLI all ask this singleton for the
same long-lived collaborators (embedder, vector store, retriever, pipeline),
so heavy models load at most once per process and tests can override any
component (Dependency Inversion + Interface Segregation: consumers receive
narrow interfaces, never the container's internals).
"""
from __future__ import annotations

import logging
import threading
from functools import cached_property
from typing import Optional

from indic_rag.config import Settings, get_settings
from indic_rag.embeddings.base import BaseEmbedder
from indic_rag.pipeline import IngestionPipeline, build_pipeline_components
from indic_rag.retrieval import HybridRetriever, ReciprocalRankFusion
from indic_rag.vectorstore.base import VectorStore

logger = logging.getLogger("indic_rag.service")


class RagService:
    """Lazily wires settings → embedder → store → retriever → pipeline."""

    def __init__(self, settings: Optional[Settings] = None, force_memory_store: bool = False) -> None:
        self.settings = settings or get_settings()
        self._force_memory_store = force_memory_store
        self._lock = threading.Lock()
        self._components: dict | None = None

    @property
    def components(self) -> dict:
        with self._lock:
            if self._components is None:
                logger.info(
                    "Initializing RAG service (provider=%s, dim=%s)",
                    self.settings.embedding.provider,
                    self.settings.embedding.mrl_dimension,
                )
                self._components = build_pipeline_components(
                    self.settings, force_memory_store=self._force_memory_store
                )
            return self._components

    @cached_property
    def embedder(self) -> BaseEmbedder:
        return self.components["embedder"]

    @cached_property
    def store(self) -> VectorStore:
        return self.components["store"]

    @cached_property
    def pipeline(self) -> IngestionPipeline:
        return IngestionPipeline(**self.components)

    @cached_property
    def retriever(self) -> HybridRetriever:
        rc = self.settings.retrieval
        return HybridRetriever(
            embedder=self.embedder,
            store=self.store,
            fusion=ReciprocalRankFusion(k=self.settings.vector_db.rrf_k),
            top_k_dense=rc.top_k_dense,
            top_k_sparse=rc.top_k_sparse,
            final_k=rc.final_k,
            enable_bm25=self.settings.vector_db.enable_bm25_sparse,
        )

    @cached_property
    def query_service(self):
        """LangGraph-backed ask pipeline (import kept lazy to isolate langgraph)."""
        from indic_rag.graph.factory import build_graph_query_service

        return build_graph_query_service(self)

    # ------------------------------------------------------------------ health
    def health(self) -> dict:
        try:
            points = self.store.count()
        except Exception as exc:  # pragma: no cover - defensive
            points = -1
            logger.warning("store count failed: %s", exc)
        return {
            "status": "ok" if points >= 0 else "degraded",
            "embedding_provider": self.embedder.provider_name,
            "dense_dim": self.embedder.dimension,
            "sparse_enabled": self.embedder.supports_sparse()
            or self.settings.vector_db.enable_bm25_sparse,
            "vector_backend": self.store.backend_name,
            "collection": self.settings.vector_db.collection_name,
            "points": points,
            "supported_languages": self.settings.supported_languages,
        }


_service: RagService | None = None
_service_lock = threading.Lock()


def get_service() -> RagService:
    """Process-wide singleton accessor used as the FastAPI dependency."""
    global _service
    with _service_lock:
        if _service is None:
            _service = RagService()
        return _service


def set_service(service: RagService) -> None:
    """Test/demo hook to inject a preconfigured container."""
    global _service
    with _service_lock:
        _service = service
