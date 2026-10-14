"""``google/embeddinggemma-2`` adapter — BF16, asymmetric task prefixes, MRL slicing.

Document side : ``title: {title} | text: {content}``
Query side    : ``task: search result | query: {query}``

CRITICAL: precision is bfloat16 (or float32 on unsupported CPUs) — never
float16, which overflows/NaNs the Gemma attention stack.
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np

from indic_rag.embeddings.base import BaseEmbedder, l2_normalize
from indic_rag.schemas import NormalizedChunk

logger = logging.getLogger("indic_rag.embeddings.gemma2")


class EmbeddingGemma2Adapter(BaseEmbedder):
    provider_name = "embeddinggemma-2"

    def __init__(
        self,
        model_id: str = "google/embeddinggemma-2",
        mrl_dim: int = 512,
        batch_size: int = 32,
        precision: str = "bfloat16",
    ) -> None:
        if mrl_dim not in (768, 512, 256, 128):
            raise ValueError("EmbeddingGemma 2 MRL dimension must be 768, 512, 256, or 128.")
        if precision == "float16":
            raise ValueError("float16 is forbidden for EmbeddingGemma 2 — use bfloat16/float32.")
        self.batch_size = batch_size
        self.dimension = mrl_dim
        self.model_id = model_id

        import torch  # heavy dependency imported lazily (ISP at module level)
        from sentence_transformers import SentenceTransformer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        bf16_ok = self.device == "cuda" or (
            hasattr(torch.cpu, "is_bf16_supported") and torch.cpu.is_bf16_supported()
        )
        dtype = torch.bfloat16 if (precision == "bfloat16" and bf16_ok) else torch.float32
        logger.info("Loading %s on %s with %s", model_id, self.device, dtype)

        self.model = SentenceTransformer(
            model_id,
            device=self.device,
            model_kwargs={"torch_dtype": dtype},
            truncate_dim=mrl_dim,
        )

    # ------------------------------------------------------------------ utils
    @staticmethod
    def format_document(title: str, text: str) -> str:
        return f"title: {title or 'none'} | text: {text}"

    @staticmethod
    def format_query(query: str) -> str:
        return f"task: search result | query: {query}"

    # ------------------------------------------------------------------ API
    def embed_chunks(self, chunks: List[NormalizedChunk]) -> np.ndarray:
        formatted = [self.format_document(c.title, c.text) for c in chunks]
        vecs = self.model.encode(
            formatted,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        if np.isnan(vecs).any():
            raise RuntimeError("NaN detected in embeddings. Verify torch_dtype is bfloat16/float32.")
        return l2_normalize(vecs)

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(
            [self.format_query(query)],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)[0]
        if np.isnan(vec).any():
            raise RuntimeError("NaN detected in query embedding.")
        return l2_normalize(vec.reshape(1, -1))[0]
