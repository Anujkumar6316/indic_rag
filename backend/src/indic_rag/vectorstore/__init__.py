"""Vector store backends (Qdrant hybrid server + in-memory dev fallback)."""
from indic_rag.vectorstore.base import VectorStore
from indic_rag.vectorstore.factory import build_vector_store
from indic_rag.vectorstore.memory_store import InMemoryVectorStore
from indic_rag.vectorstore.qdrant_store import IndicQdrantStore

__all__ = ["VectorStore", "IndicQdrantStore", "InMemoryVectorStore", "build_vector_store"]
