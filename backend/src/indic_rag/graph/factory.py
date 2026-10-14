"""Query-service factory wired to the shared ``RagService`` container."""
from __future__ import annotations

from indic_rag.config import Settings
from indic_rag.graph.query_graph import QueryService, build_query_service
from indic_rag.ingestion.lang_detect import LanguageDetector, build_detector
from indic_rag.ingestion.normalizer import IndicTextNormalizer
from indic_rag.service import RagService


def build_graph_query_service(
    service: RagService,
    normalizer: IndicTextNormalizer | None = None,
    detector: LanguageDetector | None = None,
    settings: Settings | None = None,
) -> QueryService:
    """Compose the LangGraph over the container's ports (DIP at the boundary).

    The preprocessor port needs BOTH ``clean`` (normalizer) and ``detect``
    (language detector); we adapt them into one object so each collaborator
    keeps a single responsibility.
    """
    settings = settings or service.settings
    normalizer = normalizer or IndicTextNormalizer()
    detector = detector or build_detector(settings.paths.models_dir)

    class _Preprocessor:
        def clean(self, text: str, lang: str) -> str:
            return normalizer.clean(text, lang)

        def detect(self, text: str) -> tuple[str, float]:
            return detector.detect(text)

    return build_query_service(
        settings=settings,
        preprocessor=_Preprocessor(),
        retriever=service.retriever,
        chat_model=None,  # Phase 2: inject a ChatModelPort here
    )
