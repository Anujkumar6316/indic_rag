"""LangGraph orchestration for the query path (Phase 1 retrieval + Phase-2-ready LLM node).

Graph topology::

    START → preprocess (normalize + detect language)
          → retrieve   (hybrid dense+BM25 with RRF via HybridRetriever)
          → augment    (build grounded context block)
          → generate   (optional LLM answer; deterministic extractive fallback
                         when no chat model is configured — Phase 2 slot)
          → END

SOLID notes:
- SRP: each node is one small function; state is a typed dict.
- DIP/OCP: ``build_query_graph`` receives narrow *ports* (PreprocessorPort,
  RetrieverPort, ChatModelPort); FastAPI/Studio/tests inject their own impls.
  Swapping in a real LLM later = pass a new ChatModelPort, zero graph edits.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Protocol

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from indic_rag.schemas import ScoredChunk

logger = logging.getLogger("indic_rag.graph")


class QueryState(TypedDict, total=False):
    query: str
    lang: Optional[str]
    domain: Optional[str]
    final_k: Optional[int]
    normalized_query: str
    detected_lang: str
    results: List[ScoredChunk]
    context: str
    answer: str
    provider: str
    took_ms: float


# --------------------------------------------------------------------- ports
class PreprocessorPort(Protocol):
    def clean(self, text: str, lang: str) -> str: ...
    def detect(self, text: str) -> tuple[str, float]: ...


class RetrieverPort(Protocol):
    def search(
        self, query: str, lang: Optional[str] = None, domain: Optional[str] = None,
        final_k: Optional[int] = None,
    ) -> List[ScoredChunk]: ...


class ChatModelPort(Protocol):
    """Narrow interface for the Phase-2 generation step."""

    def invoke(self, prompt: str) -> str: ...


CONTEXT_TEMPLATE = (
    "You are an assistant for Indic multilingual documents. "
    "Answer strictly from the context; reply in the question's language.\n\n"
    "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
)


def default_prompt(query: str, chunks: List[ScoredChunk]) -> str:
    context = "\n\n".join(
        f"[{c.lang}] {c.title} #{c.chunk_id[:8]}: {c.text}" for c in chunks
    ) or "(no context retrieved)"
    return CONTEXT_TEMPLATE.format(context=context, query=query)


def extractive_answer(chunks: List[ScoredChunk], max_sentences: int = 3) -> str:
    """Deterministic offline 'answer': top-ranked sentences from best chunks."""
    lines: list[str] = []
    for c in chunks[:2]:
        sents = [s.strip() for s in _split_sentences(c.text) if s.strip()]
        lines.extend(sents[:max_sentences])
    return " ".join(lines) if lines else "No relevant passages found in the knowledge base."


def _split_sentences(text: str) -> list[str]:
    import re

    return re.split(r"(?<=॥)|(?<=।)|(?<=[.!?])\s+", text)


# ---------------------------------------------------------------------- graph
def build_query_graph(
    preprocessor: PreprocessorPort,
    retriever: RetrieverPort,
    chat_model: Optional[ChatModelPort] = None,
    default_final_k: int = 5,
):
    """Compile the retrieval graph over injected ports (compiled once, thread-safe)."""

    def preprocess_node(state: QueryState) -> Dict[str, Any]:
        query = state["query"].strip()
        detected, confidence = preprocessor.detect(query[:500])
        lang_hint = state.get("lang") or detected
        normalized = preprocessor.clean(query, lang_hint)
        logger.debug("preprocess: lang=%s conf=%.2f", detected, confidence)
        return {
            "normalized_query": normalized,
            "detected_lang": detected,
            # Filter corpus by language ONLY when the caller explicitly asked.
            "lang": state.get("lang"),
        }

    def retrieve_node(state: QueryState) -> Dict[str, Any]:
        results = retriever.search(
            state["normalized_query"],
            lang=state.get("lang"),
            domain=state.get("domain"),
            final_k=state.get("final_k") or default_final_k,
        )
        return {"results": results}

    def augment_node(state: QueryState) -> Dict[str, Any]:
        results = state.get("results", [])
        context = "\n\n".join(
            f"[{c.lang}] {c.title}: {c.text}" for c in results
        )
        return {"context": context}

    def generate_node(state: QueryState) -> Dict[str, Any]:
        results = state.get("results", [])
        if chat_model is not None:
            prompt = default_prompt(state["normalized_query"], results)
            try:
                return {"answer": chat_model.invoke(prompt)}
            except Exception as exc:  # pragma: no cover - network safety
                logger.warning("chat model failed (%s); using extractive answer.", exc)
        return {"answer": extractive_answer(results)}

    graph = StateGraph(QueryState)
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("augment", augment_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "retrieve")
    graph.add_edge("retrieve", "augment")
    graph.add_edge("augment", "generate")
    graph.add_edge("generate", END)
    return graph.compile(name="indic-rag-query-graph")


# ------------------------------------------------------------------ facade
class QueryService:
    """Application-facing facade hiding the raw LangGraph from API/Studio."""

    def __init__(self, graph, provider: str) -> None:
        self._graph = graph
        self.provider = provider

    def ask(
        self,
        query: str,
        lang: Optional[str] = None,
        domain: Optional[str] = None,
        final_k: Optional[int] = None,
        with_answer: bool = True,
    ) -> QueryState:
        state: QueryState = {"query": query, "lang": lang, "domain": domain, "final_k": final_k}
        out = self._graph.invoke(state)
        out.setdefault("provider", self.provider)
        if not with_answer:
            out.pop("answer", None)
        return out


def build_query_service(
    settings,
    preprocessor: PreprocessorPort,
    retriever,
    chat_model: Optional[ChatModelPort] = None,
) -> QueryService:
    """Compile the graph and wrap it; ``retriever`` may be a HybridRetriever."""
    graph = build_query_graph(
        preprocessor=preprocessor,
        retriever=retriever,
        chat_model=chat_model,
        default_final_k=settings.retrieval.final_k,
    )
    provider = getattr(getattr(retriever, "embedder", None), "provider_name", "unknown")
    return QueryService(graph, provider=provider)
