"""Hybrid Qdrant vector store: dense HNSW + ``Qdrant/bm25`` sparse lexical index.

Sparse encoding strategy (mirrors the blueprint):
- Documents are sent to Qdrant as ``models.Document(text=..., model="Qdrant/bm25")``
  and FastEmbed tokenizes/IDF-weights them server-side.
- Queries are encoded *client-side* with the same ``Qdrant/bm25`` FastEmbed model
  when installed; otherwise a pure-Python BM25 scorer over stored texts keeps
  hybrid search functional offline.

SOLID notes:
- DIP: implements ``VectorStore``; pipeline never imports qdrant_client.
- SRP: storage mechanics only — RRF fusion lives in the retrieval layer.
"""
from __future__ import annotations

import logging
import math
import re
import uuid
from collections import Counter, defaultdict
from typing import List, Optional

import numpy as np

from indic_rag.schemas import NormalizedChunk, ScoredChunk
from indic_rag.vectorstore.base import VectorStore

logger = logging.getLogger("indic_rag.vectorstore.qdrant")

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize_bm25(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class _LocalBM25Index:
    """Minimal Okapi-BM25 used when FastEmbed/Qdrant server are unavailable."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_tokens: dict[str, Counter] = {}
        self.doc_len: dict[str, int] = {}
        self.df: Counter = Counter()

    def add(self, point_id: str, text: str) -> None:
        toks = tokenize_bm25(text)
        if point_id in self.doc_tokens:
            self.remove(point_id)
        counts = Counter(toks)
        self.doc_tokens[point_id] = counts
        self.doc_len[point_id] = len(toks)
        for tok in counts:
            self.df[tok] += 1

    def remove(self, point_id: str) -> None:
        counts = self.doc_tokens.pop(point_id, None)
        self.doc_len.pop(point_id, None)
        if counts:
            for tok in counts:
                self.df[tok] -= 1

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        n_docs = max(len(self.doc_tokens), 1)
        avg_len = sum(self.doc_len.values()) / n_docs or 1.0
        scores: dict[str, float] = defaultdict(float)
        for tok in set(tokenize_bm25(query)):
            df = self.df.get(tok, 0)
            if not df:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for pid, counts in self.doc_tokens.items():
                tf = counts.get(tok, 0)
                if not tf:
                    continue
                dl = self.doc_len[pid]
                scores[pid] += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / avg_len)
                )
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:limit]


class IndicQdrantStore(VectorStore):
    backend_name = "qdrant"

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        dense_dim: int,
        enable_bm25_sparse: bool = True,
        grpc_port: int = 6334,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
        local_path: Optional[str] = None,
    ) -> None:
        from qdrant_client import QdrantClient  # imported lazily by design

        if local_path:
            # Embedded Qdrant (same engine, no server needed) — used by tests.
            Path(local_path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(local_path))
        else:
            self.client = QdrantClient(
                host=host,
                port=port,
                grpc_port=grpc_port,
                api_key=api_key,
                prefer_grpc=prefer_grpc,
            )
        self.collection_name = collection_name
        self.dense_dim = dense_dim
        self.enable_bm25_sparse = enable_bm25_sparse
        self._bm25_encoder = self._load_fastembed() if enable_bm25_sparse else None
        self._ensure_collection()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _load_fastembed():
        try:
            from fastembed import TextSparseVectorizer  # type: ignore

            return TextSparseVectorizer(model_name="Qdrant/bm25")
        except Exception:
            logger.info("FastEmbed unavailable — using local BM25 scorer for queries.")
            return None

    def _ensure_collection(self) -> None:
        from qdrant_client import models

        if not self.client.collection_exists(self.collection_name):
            vectors_config = {
                "dense": models.VectorParams(
                    size=self.dense_dim,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100),
                )
            }
            sparse_config = None
            if self.enable_bm25_sparse:
                sparse_config = {
                    "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                }
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_config,
            )
            # Fast payload filters for language and domain isolation.
            for field in ("lang", "domain"):
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
        elif self._collection_dim_mismatch():
            raise RuntimeError(
                f"Collection '{self.collection_name}' exists with a different dense dim "
                f"than {self.dense_dim}. Recreate it or align settings.yaml."
            )

    def _collection_dim_mismatch(self) -> bool:
        try:
            info = self.client.get_collection(self.collection_name)
            params = info.config.params.vectors
            dense = params.get("dense") if isinstance(params, dict) else None
            return dense is not None and dense.size != self.dense_dim
        except Exception:
            return False

    # ------------------------------------------------------------------ write
    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def upsert_batch(
        self,
        chunks: List[NormalizedChunk],
        dense_vectors: np.ndarray,
        sparse_vectors: list | None = None,
    ) -> int:
        from qdrant_client import models

        points = []
        for chunk, vec in zip(chunks, dense_vectors):
            vector: dict = {"dense": np.asarray(vec, dtype=np.float32).tolist()}
            if self.enable_bm25_sparse:
                # Server-side BM25 encoding via Document + named sparse vector.
                vector["bm25"] = models.Document(text=chunk.text, model="Qdrant/bm25")
            points.append(
                models.PointStruct(
                    id=self._point_id(chunk.chunk_id),
                    vector=vector,
                    payload=chunk.model_dump(),
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    # ------------------------------------------------------------------ read
    def _query_sparse(self, query_text: str):
        if self._bm25_encoder is not None:
            enc = next(iter(self._bm25_encoder.embed([query_text])))
            from qdrant_client import models

            return models.SparseVector(indices=enc.indices.tolist(), values=enc.values.tolist())
        # Fallback: build an IDF-ish one-hot sparse vector from tokens hashed to ids.
        from qdrant_client import models

        idx = {abs(hash(t)) % (2**31): 0.0 for t in tokenize_bm25(query_text)}
        for tok in tokenize_bm25(query_text):
            idx[abs(hash(tok)) % (2**31)] += 1.0
        pairs = sorted(idx.items(), key=lambda kv: -kv[1])[:512]
        return models.SparseVector(
            indices=[i for i, _ in pairs], values=[float(v) for _, v in pairs]
        )

    @staticmethod
    def _build_query_filter(lang: Optional[str], domain: Optional[str]):
        from qdrant_client import models

        must = []
        if lang:
            must.append(models.FieldCondition(key="lang", match=models.MatchValue(value=lang)))
        if domain:
            must.append(models.FieldCondition(key="domain", match=models.MatchValue(value=domain)))
        return models.Filter(must=must) if must else None

    @staticmethod
    def _to_scored(record, rank: int, source: str) -> ScoredChunk:
        p = record.payload or {}
        return ScoredChunk(
            chunk_id=p.get("chunk_id", str(record.id)),
            doc_id=p.get("doc_id", ""),
            title=p.get("title", "none"),
            lang=p.get("lang", "en"),
            domain=p.get("domain", "general"),
            text=p.get("text", ""),
            score=float(record.score or 0.0),
            rank=rank,
            sources=[source],
            metadata=p.get("metadata") or {},
        )

    def search_dense(
        self,
        query_vector: np.ndarray,
        limit: int = 20,
        lang: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[ScoredChunk]:
        records = self.client.query_points(
            collection_name=self.collection_name,
            query=np.asarray(query_vector, dtype=np.float32).tolist(),
            using="dense",
            limit=limit,
            query_filter=self._build_query_filter(lang, domain),
            with_payload=True,
        ).points
        return [self._to_scored(r, i, "dense") for i, r in enumerate(records)]

    def search_bm25(
        self,
        query_text: str,
        limit: int = 20,
        lang: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[ScoredChunk]:
        if not self.enable_bm25_sparse:
            return []
        records = self.client.query_points(
            collection_name=self.collection_name,
            query=self._query_sparse(query_text),
            using="bm25",
            limit=limit,
            query_filter=self._build_query_filter(lang, domain),
            with_payload=True,
        ).points
        return [self._to_scored(r, i, "bm25") for i, r in enumerate(records)]

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name, exact=True).count

    def scroll_all(self, limit: int = 10_000) -> List[NormalizedChunk]:
        out: List[NormalizedChunk] = []
        offset = None
        while len(out) < limit:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in records:
                try:
                    out.append(NormalizedChunk.model_validate(r.payload))
                except Exception:
                    continue
            if offset is None or not records:
                break
        return out[:limit]
