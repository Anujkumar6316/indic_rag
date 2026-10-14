"""Vector-store factory with graceful degradation.

Preference order:
1. Qdrant server (docker-compose) — production path.
2. In-memory hybrid store persisted to ``vector_db.local_path`` — dev/CI path,
   fully compatible with the same ``VectorStore`` contract (LSP).
"""
from __future__ import annotations

import logging
from pathlib import Path

from indic_rag.config import VectorDbConfig
from indic_rag.vectorstore.base import VectorStore
from indic_rag.vectorstore.memory_store import InMemoryVectorStore

logger = logging.getLogger("indic_rag.vectorstore.factory")


def build_vector_store(cfg: VectorDbConfig, dense_dim: int, force_memory: bool = False) -> VectorStore:
    if not force_memory:
        try:
            from indic_rag.vectorstore.qdrant_store import IndicQdrantStore

            store = IndicQdrantStore(
                host=cfg.host,
                port=cfg.port,
                collection_name=cfg.collection_name,
                dense_dim=dense_dim,
                enable_bm25_sparse=cfg.enable_bm25_sparse,
                grpc_port=cfg.grpc_port,
                api_key=cfg.api_key,
                prefer_grpc=cfg.prefer_grpc,
            )
            logger.info("Connected to Qdrant at %s:%s", cfg.host, cfg.port)
            return store
        except Exception as exc:
            logger.warning(
                "Qdrant unavailable (%s). Using persisted in-memory hybrid store at %s.",
                str(exc)[:160],
                cfg.local_path,
            )
    persist = None
    if cfg.local_path:
        Path(cfg.local_path).mkdir(parents=True, exist_ok=True)
        persist = Path(cfg.local_path) / f"{cfg.collection_name}.jsonl"
    return InMemoryVectorStore(persist_path=persist)
