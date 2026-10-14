"""Deterministic offline embedding fallback.

Implements the same ``BaseEmbedder`` contract using signed feature hashing over
Unicode-aware character n-grams + word unigrams.  It is *not* a semantic model,
but it keeps the whole ingestion → storage → retrieval graph exercisable in CI
and air-gapped environments, and its shared-ngram overlap gives lexical-ish
recall that is enough to smoke-test hybrid plumbing end-to-end.

Switch back by setting ``embedding.provider: hash-fallback`` in settings.yaml —
zero code changes (OCP).
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List

import numpy as np

from indic_rag.embeddings.base import BaseEmbedder, l2_normalize
from indic_rag.schemas import NormalizedChunk

_WORD = re.compile(r"\w+", re.UNICODE)


def _token_stream(text: str) -> list[str]:
    tokens = [t.lower() for t in _WORD.findall(text)]
    # Character 3-grams capture Indic conjuncts even under whitespace noise.
    compact = re.sub(r"\s+", "", text).lower()
    grams = [compact[i : i + 3] for i in range(max(len(compact) - 2, 0))]
    return tokens + grams


class HashingFallbackEmbedder(BaseEmbedder):
    provider_name = "hash-fallback"

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float64)
        for tok in _token_stream(text):
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "big")
            idx = h % self.dimension
            sign = 1.0 if (h >> 63) & 1 else -1.0
            weight = 1.0 + math.log1p(1) / (1 + len(tok))
            vec[idx] += sign * weight
        norm = np.linalg.norm(vec)
        if norm == 0:
            vec[0] = 1.0  # deterministic unit vector; never emit all-zeros/NaN
            norm = 1.0
        return (vec / norm).astype(np.float32)

    def embed_chunks(self, chunks: List[NormalizedChunk]) -> np.ndarray:
        if not chunks:
            return np.empty((0, self.dimension), dtype=np.float32)
        return l2_normalize(np.stack([self._embed_one(c.text) for c in chunks]))

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed_one(query)
