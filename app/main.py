"""FastAPI application factory."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import __version__
from app.api.tts import router as tts_router
from app.config import Settings, get_settings
from app.core.concurrency import ConcurrencyLimiter
from app.exceptions import ProviderError
from app.logging_config import setup_logging
from app.middleware import RequestContextMiddleware
from app.providers.base import TTSProvider
from app.providers.kokoro import KokoroProvider

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. ``settings`` overrides the default for tests."""
    settings = settings or get_settings()
    setup_logging(settings.LOG_LEVEL)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Provider may be injected in tests; otherwise load + warm up Kokoro.
        # Model loading is blocking, so it happens off the event loop.
        if not hasattr(app.state, "provider"):
            logger.info("initializing Kokoro provider ...")
            app.state.provider = await asyncio.to_thread(KokoroProvider, settings)
        if not hasattr(app.state, "limiter"):
            app.state.limiter = ConcurrencyLimiter(
                limit=settings.CONCURRENCY_LIMIT,
                timeout=settings.QUEUE_WAIT_TIMEOUT,
            )
        yield
        logger.info("server shutting down")

    app = FastAPI(
        title="Kokoro TTS Server",
        version=__version__,
        description=(
            "Streaming TTS service for the LiveKit voice agent. "
            "Consumes complete sentences and returns raw s16le mono PCM."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(RequestContextMiddleware)
    app.include_router(tts_router, prefix="/v1/tts", tags=["TTS"])

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # The API contract calls for 400 on malformed fields, not 422. Keep the
        # payload JSON-serializable (pydantic ctx can hold exception objects).
        errors = [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
            for e in exc.errors()
        ]
        return JSONResponse(status_code=400, content={"detail": errors})

    @app.exception_handler(ProviderError)
    async def provider_error_handler(
        request: Request, exc: ProviderError
    ) -> JSONResponse:
        logger.error("provider error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {
            "service": "kokoro-tts",
            "version": __version__,
            "docs": "/docs",
            "health": "/v1/tts/health",
            "voices": "/v1/tts/voices",
        }

    return app


app = create_app()
