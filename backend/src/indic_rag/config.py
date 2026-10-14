"""Pydantic settings loader.

Single source of truth for runtime configuration (``configs/settings.yaml``),
with environment-variable overrides prefixed with ``INDIC_RAG_``
(e.g. ``INDIC_RAG_EMBEDDING__PROVIDER=bge-m3``).

SOLID notes:
- SRP: this module only loads/validates configuration; no pipeline logic.
- DIP: consumers depend on the ``Settings`` contract, not on YAML files.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

SupportedLang = Literal["sa", "hi", "mr", "gu", "en"]
EmbeddingProvider = Literal["embeddinggemma-2", "bge-m3", "hash-fallback"]


class PathsConfig(BaseModel):
    raw_dir: str = "data/raw"
    quarantine_dir: str = "data/quarantine"
    staging_dir: str = "data/staging"
    eval_dir: str = "data/eval"
    models_dir: str = "data/models"


class IngestionConfig(BaseModel):
    chunk_size_tokens: int = 300
    chunk_overlap_tokens: int = 45
    quarantine_legacy_fonts: bool = True
    separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", "॥", "।", ".", "?", "!"]
    )

    @model_validator(mode="after")
    def _check_overlap(self) -> "IngestionConfig":
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        return self


class EmbeddingConfig(BaseModel):
    provider: EmbeddingProvider = "embeddinggemma-2"
    model_id: str = "google/embeddinggemma-2"
    precision: Literal["bfloat16", "float32"] = "bfloat16"
    mrl_dimension: int = 512
    batch_size: int = 32
    allow_hash_fallback: bool = True

    @field_validator("precision")
    @classmethod
    def _reject_float16(cls, v: str) -> str:
        # CRITICAL guardrail from the blueprint: float16 corrupts EmbeddingGemma 2.
        if v == "float16":
            raise ValueError("float16 is forbidden; use bfloat16 or float32")
        return v

    @model_validator(mode="after")
    def _check_model_pairing(self) -> "EmbeddingConfig":
        if self.provider == "embeddinggemma-2" and self.mrl_dimension not in (768, 512, 256, 128):
            raise ValueError(
                "EmbeddingGemma 2 MRL dimension must be one of 768, 512, 256, 128"
            )
        if self.provider == "bge-m3" and self.mrl_dimension != 1024:
            raise ValueError("BGE-M3 dense dimension must be 1024")
        return self


class VectorDbConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    api_key: str | None = None
    prefer_grpc: bool = False
    collection_name: str = "indic_knowledge_v1"
    enable_bm25_sparse: bool = True
    local_path: str = "data/qdrant_local"
    rrf_k: int = 60


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class StudioConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False


class RetrievalConfig(BaseModel):
    top_k_dense: int = 20
    top_k_sparse: int = 20
    final_k: int = 5


class LoggingConfig(BaseModel):
    config_path: str = "configs/logging.yaml"


class Settings(BaseModel):
    """Root application settings."""

    project_name: str = "indic-multilingual-rag"
    supported_languages: list[SupportedLang] = Field(
        default_factory=lambda: ["sa", "hi", "mr", "gu", "en"]
    )
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_db: VectorDbConfig = Field(default_factory=VectorDbConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    studio: StudioConfig = Field(default_factory=StudioConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @staticmethod
    def _env_overrides(data: dict) -> dict:
        """Apply INDIC_RAG_SECTION__KEY environment overrides onto a raw dict."""
        import os

        for env_key, env_val in os.environ.items():
            if not env_key.startswith("INDIC_RAG_"):
                continue
            dotted = env_key[len("INDIC_RAG_"):].lower().replace("__", ".")
            parts = dotted.split(".")
            node = data
            for p in parts[:-1]:
                node = node.setdefault(p, {})
                if not isinstance(node, dict):
                    break
            else:
                raw: object = env_val
                if env_val.lower() in ("true", "false"):
                    raw = env_val.lower() == "true"
                else:
                    try:
                        raw = yaml.safe_load(env_val)
                    except yaml.YAMLError:
                        raw = env_val
                node[parts[-1]] = raw
        return data

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        """Load settings from YAML (if present) + env overrides + defaults."""
        data: dict = {}
        candidates = [
            Path(path) if path else None,
            Path("configs/settings.yaml"),
            Path(__file__).resolve().parents[2] / "configs" / "settings.yaml",
        ]
        for cand in candidates:
            if cand and cand.is_file():
                loaded = yaml.safe_load(cand.read_text(encoding="utf-8")) or {}
                data = cls._env_overrides(loaded)
                break
        else:
            data = cls._env_overrides({})
        return cls.model_validate(data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached accessor used by FastAPI dependency injection."""
    return Settings.load()
