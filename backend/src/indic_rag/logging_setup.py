"""Structured JSON logging bootstrap (referenced by ``configs/logging.yaml``)."""
from __future__ import annotations

import json
import logging
import logging.config
from datetime import timezone
from pathlib import Path

from indic_rag.config import get_settings


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line with ISO-8601 timestamps."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if extra := getattr(record, "context", None):
            payload["context"] = extra
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(config_path: str | Path | None = None) -> logging.Logger:
    """Configure root logging from the YAML file; safe no-op fallback otherwise."""
    settings = get_settings()
    path = Path(config_path or settings.logging.config_path)
    if path.is_file():
        try:
            import yaml

            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            logging.config.dictConfig(cfg)
        except Exception as exc:  # pragma: no cover - defensive
            logging.basicConfig(level=logging.INFO)
            logging.getLogger(__name__).warning("Falling back to basic logging: %s", exc)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        )
    logger = logging.getLogger("indic_rag")
    logger.debug("Logging initialised (utc=%s)", timezone.utc)
    return logger
