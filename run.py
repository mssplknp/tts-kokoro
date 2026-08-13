"""Development entrypoint: ``python run.py``.

For production prefer a process manager, e.g.
``uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 1``.
Note: keep ``--workers 1`` — a single worker owns the model.
"""

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
