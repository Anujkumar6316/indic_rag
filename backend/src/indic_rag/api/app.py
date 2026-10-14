"""FastAPI service exposing Phase-1 ingestion + hybrid retrieval.

Layering (SOLID):
- Routes are thin adapters; all domain logic lives behind ``RagService`` ports.
- Dependency injection via ``Depends`` keeps endpoints testable with an
  overridden container (``app.dependency_overrides``).
"""
from __future__ import annotations

import time
from typing import List, Literal, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from indic_rag.config import Settings
from indic_rag.schemas import ScoredChunk
from indic_rag.service import RagService, get_service

LangFilter = Optional[Literal["sa", "hi", "mr", "gu", "en"]]


# ------------------------------------------------------------------- request models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    lang: LangFilter = None
    domain: Optional[str] = None
    final_k: int = Field(default=5, ge=1, le=50)


class AskRequest(SearchRequest):
    """Search + grounded answer through the LangGraph query pipeline."""


class IngestRequest(BaseModel):
    run_audit: bool = True


class TextIngestRequest(BaseModel):
    text: str = Field(..., min_length=5)
    doc_id: str = "adhoc-doc"
    title: str = "none"
    lang: LangFilter = None
    domain: str = "general"


# ---------------------------------------------------------------------- app factory
def create_app(service: RagService | None = None) -> FastAPI:
    # Never touch the global singleton at construction time (keeps tests cheap);
    # heavy components load lazily on first request via ``svc()``.
    settings_for_app = service.settings if service is not None else Settings.load()

    app = FastAPI(
        title="Indic Multilingual RAG — Phase 1",
        description=(
            "Hybrid (dense EmbeddingGemma-2/BGE-M3 + BM25 sparse, RRF fusion) "
            "ingestion & retrieval API for sa/hi/mr/gu/en corpora."
        ),
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings_for_app.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def svc() -> RagService:
        return service or get_service()

    @app.get("/health", tags=["ops"])
    def health(s: RagService = Depends(svc)) -> dict:
        return s.health()

    @app.get("/settings", tags=["ops"])
    def current_settings(s: RagService = Depends(svc)) -> dict:
        """Non-secret view of runtime configuration (model toggles live here)."""
        cfg = s.settings.model_dump()
        cfg["vector_db"].pop("api_key", None)
        return cfg

    # -------------------------------------------------------------- retrieval
    @app.post("/search", response_model=dict, tags=["retrieval"])
    def search(req: SearchRequest, s: RagService = Depends(svc)) -> dict:
        t0 = time.perf_counter()
        try:
            results: List[ScoredChunk] = s.retriever.search(
                req.query, lang=req.lang, domain=req.domain, final_k=req.final_k
            )
        except Exception as exc:  # pragma: no cover - infra failure
            raise HTTPException(status_code=503, detail=f"retrieval failed: {exc}") from exc
        took_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "query": req.query,
            "results": [r.model_dump() for r in results],
            "provider": s.embedder.provider_name,
            "backend": s.store.backend_name,
            "took_ms": took_ms,
        }

    @app.post("/ask", response_model=dict, tags=["retrieval"])
    def ask(req: AskRequest, s: RagService = Depends(svc)) -> dict:
        """Run the LangGraph: preprocess → retrieve → augment → generate."""
        t0 = time.perf_counter()
        state = s.query_service.ask(
            req.query, lang=req.lang, domain=req.domain, final_k=req.final_k
        )
        took_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "query": req.query,
            "normalized_query": state.get("normalized_query", req.query),
            "detected_lang": state.get("detected_lang"),
            "answer": state.get("answer", ""),
            "results": [r.model_dump() for r in state.get("results", [])],
            "provider": state.get("provider", s.embedder.provider_name),
            "took_ms": took_ms,
        }

    # ------------------------------------------------------------- ingestion
    @app.post("/ingest/audit", tags=["ingestion"])
    def ingest_audit(s: RagService = Depends(svc)) -> dict:
        reports = s.pipeline.run_audit()
        return {"reports": [r.model_dump() for r in reports]}

    @app.post("/ingest/batch", tags=["ingestion"])
    def ingest_batch(req: IngestRequest, s: RagService = Depends(svc)) -> dict:
        if req.run_audit:
            return s.pipeline.run_full()
        return {"stats": s.pipeline.run_ingest().model_dump()}

    @app.post("/ingest/text", tags=["ingestion"])
    def ingest_text(req: TextIngestRequest, s: RagService = Depends(svc)) -> dict:
        """Quick single-document ingestion used by the Gradio Studio upload tab."""
        from indic_rag.schemas import RawDocument

        doc = RawDocument(
            doc_id=req.doc_id, title=req.title, lang=req.lang,
            domain=req.domain, raw_text=req.text,
        )
        chunks = s.pipeline.process_document(doc)
        upserted = s.pipeline.embed_and_store(chunks)
        return {"doc_id": doc.doc_id, "chunks": len(chunks), "upserted": upserted}

    @app.get("/corpus/stats", tags=["ops"])
    def corpus_stats(lang: LangFilter = None, s: RagService = Depends(svc)) -> dict:
        chunks = s.store.scroll_all()
        by_lang: dict[str, int] = {}
        by_domain: dict[str, int] = {}
        for c in chunks:
            if lang and c.lang != lang:
                continue
            by_lang[c.lang] = by_lang.get(c.lang, 0) + 1
            by_domain[c.domain] = by_domain.get(c.domain, 0) + 1
        return {"total_chunks": len(chunks), "by_lang": by_lang, "by_domain": by_domain}

    return app


def build_app(settings: Settings | None = None) -> FastAPI:
    svc = RagService(settings=settings) if settings else None
    app = create_app(svc)
    return app
