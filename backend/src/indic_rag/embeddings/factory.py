"""Embedder factory — the single place allowed to know about concrete strategies.

Resolution order for ``embedding.provider``:
1. Requested provider (``embeddinggemma-2`` / ``bge-m3``).
2. If its heavy dependencies or HF weights are unavailable AND
   ``allow_hash_fallback`` is true → deterministic ``hash-fallback``.
This keeps dev/CI green while production stays on the real models.
"""
from __future__ import annotations

import logging

from indic_rag.config import EmbeddingConfig
from indic_rag.embeddings.base import BaseEmbedder
from indic_rag.embeddings.fallback import HashingFallbackEmbedder

logger = logging.getLogger("indic_rag.embeddings.factory")


def build_embedder(cfg: EmbeddingConfig) -> BaseEmbedder:
    if cfg.provider == "hash-fallback":
        return HashingFallbackEmbedder(dimension=cfg.mrl_dimension)

    try:
        if cfg.provider == "embeddinggemma-2":
            from indic_rag.embeddings.gemma2 import EmbeddingGemma2Adapter

            return EmbeddingGemma2Adapter(
                model_id=cfg.model_id,
                mrl_dim=cfg.mrl_dimension,
                batch_size=cfg.batch_size,
                precision=cfg.precision,
            )
        if cfg.provider == "bge-m3":
            from indic_rag.embeddings.bgem3 import BGEM3Adapter

            return BGEM3Adapter(model_id=cfg.model_id, batch_size=cfg.batch_size)
        raise ValueError(f"Unknown embedding provider: {cfg.provider}")
    except (ImportError, RuntimeError, OSError) as exc:
        if not cfg.allow_hash_fallback:
            raise
        logger.warning(
            "Provider '%s' unavailable (%s). Falling back to deterministic hash embedder.",
            cfg.provider,
            str(exc)[:200],
        )
        return HashingFallbackEmbedder(dimension=cfg.mrl_dimension)
