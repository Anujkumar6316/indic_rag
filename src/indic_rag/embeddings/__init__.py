"""Pluggable embedding layer (Strategy Pattern).

Concrete strategies: EmbeddingGemma-2, BGE-M3, and a deterministic hash-based
fallback so the whole microservice runs offline/in-CI without HF weights.
"""
from indic_rag.embeddings.base import BaseEmbedder
from indic_rag.embeddings.factory import build_embedder
from indic_rag.embeddings.fallback import HashingFallbackEmbedder

__all__ = ["BaseEmbedder", "build_embedder", "HashingFallbackEmbedder"]
