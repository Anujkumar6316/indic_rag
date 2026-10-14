"""Uvicorn entrypoint: ``python -m indic_rag.api`` or ``scripts/run_api.py``."""
from __future__ import annotations

import uvicorn

from indic_rag.api.app import create_app
from indic_rag.config import get_settings
from indic_rag.logging_setup import setup_logging

app = create_app()


def main() -> None:
    settings = get_settings()
    setup_logging(settings.logging.config_path)
    uvicorn.run(
        "indic_rag.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
