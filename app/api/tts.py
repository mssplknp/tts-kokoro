"""TTS HTTP endpoints.

Implements the API contract in the project README. All PCM responses are raw
little-endian signed 16-bit mono samples, streamed as ``application/octet-stream``.
"""

import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from app.context import get_request_id, new_request_id
from app.core.audio import PcmFrameAccumulator
from app.core.concurrency import ConcurrencyLimiter
from app.exceptions import ProviderBusy, ProviderError
from app.providers.base import TTSProvider
from app.schemas.tts import TTSRequest

logger = logging.getLogger(__name__)

router = APIRouter()


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #


def get_provider(request: Request) -> TTSProvider:
    """Resolve the configured TTS backend (attached at startup)."""
    provider = getattr(request.app.state, "provider", None)
    if provider is None:
        raise HTTPException(status_code=503, detail="TTS provider not ready")
    return provider


def get_limiter(request: Request) -> ConcurrencyLimiter:
    """Resolve the per-process concurrency limiter."""
    limiter = getattr(request.app.state, "limiter", None)
    if limiter is None:
        raise HTTPException(status_code=503, detail="server not ready")
    return limiter


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _resolve_request_id(body: TTSRequest, request: Request) -> str:
    """Correlation id: explicit body field, then header, then generated."""
    return body.request_id or request.headers.get("x-request-id") or new_request_id()


def _validate_language(language: str | None, provider: TTSProvider) -> None:
    if language is None:
        return
    normalized = language.strip().lower()
    if normalized not in provider.supported_languages:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported language {language!r}; "
                f"this server serves {sorted(provider.supported_languages)}"
            ),
        )


def _validate_voice(voice: str | None, provider: TTSProvider) -> None:
    if voice is None:
        return
    for part in voice.split(","):
        name = part.strip()
        if name and name not in provider.voices:
            raise HTTPException(
                status_code=400,
                detail=f"unknown voice {name!r}; available: {provider.voices}",
            )


def _pcm_headers(output_rate: int, request_id: str) -> dict[str, str]:
    return {
        "X-Sample-Rate": str(output_rate),
        "X-Channels": "1",
        "X-Sample-Format": "s16le",
        "X-Request-Id": request_id,
    }


async def _acquire_permit(limiter: ConcurrencyLimiter) -> None:
    try:
        await limiter.acquire()
    except ProviderBusy as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


async def _pcm_stream(
    audio_stream: AsyncIterator[bytes],
    accumulator: PcmFrameAccumulator,
    limiter: ConcurrencyLimiter,
    request_id: str,
) -> AsyncIterator[bytes]:
    """Consume the provider's async stream, emit frame-aligned PCM, always
    release the concurrency slot — including on client disconnect, which
    propagates a cancellation here and lets the provider stop early."""
    try:
        async for chunk in audio_stream:
            for frame in accumulator.push(chunk):
                yield frame
        remainder = accumulator.flush()
        if remainder:
            yield remainder
    except ProviderError as exc:
        logger.error("synthesis failed (request %s): %s", request_id, exc)
        raise
    except Exception:
        logger.exception("unexpected synthesis error (request %s)", request_id)
        raise
    finally:
        limiter.release()
        await audio_stream.aclose()


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.post("/stream")
async def stream_tts(
    body: TTSRequest,
    request: Request,
    provider: TTSProvider = Depends(get_provider),
    limiter: ConcurrencyLimiter = Depends(get_limiter),
) -> StreamingResponse:
    """Synthesize a sentence and stream raw s16le PCM.

    Audio is generated incrementally; the first chunk arrives as soon as the
    first sub-unit is ready. The response is truncated if the client
    disconnects, which stops synthesis early to free the model.
    """
    text = body.text.strip()
    _validate_language(body.language, provider)
    _validate_voice(body.voice, provider)
    await _acquire_permit(limiter)

    output_rate = body.sample_rate or provider.sample_rate
    frame_size = output_rate * 20 // 1000 * 2  # 20 ms frame, mono s16le
    request_id = _resolve_request_id(body, request)

    audio_stream = provider.astream(
        text,
        voice=body.voice,
        speed=body.speed,
        sample_rate=body.sample_rate,
    )

    return StreamingResponse(
        _pcm_stream(
            audio_stream,
            PcmFrameAccumulator(frame_size),
            limiter,
            request_id,
        ),
        media_type="application/octet-stream",
        headers=_pcm_headers(output_rate, request_id),
    )


@router.post("/synthesize")
async def synthesize_tts(
    body: TTSRequest,
    request: Request,
    provider: TTSProvider = Depends(get_provider),
    limiter: ConcurrencyLimiter = Depends(get_limiter),
) -> Response:
    """Non-streaming convenience: return one complete PCM buffer."""
    text = body.text.strip()
    _validate_language(body.language, provider)
    _validate_voice(body.voice, provider)
    await _acquire_permit(limiter)

    output_rate = body.sample_rate or provider.sample_rate
    request_id = _resolve_request_id(body, request)

    try:
        pcm = await provider.asynthesize(
            text,
            voice=body.voice,
            speed=body.speed,
            sample_rate=body.sample_rate,
        )
    except ProviderError as exc:
        logger.error("synthesis failed (request %s): %s", request_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception:
        logger.exception("unexpected synthesis error (request %s)", request_id)
        raise
    finally:
        limiter.release()

    return Response(
        content=pcm,
        media_type="application/octet-stream",
        headers=_pcm_headers(output_rate, request_id),
    )


@router.get("/health")
async def health(
    provider: TTSProvider = Depends(get_provider),
    limiter: ConcurrencyLimiter = Depends(get_limiter),
) -> dict:
    """Liveness + capacity so the agent can gate on readiness."""
    return {
        "status": "ok",
        "model": provider.model_name,
        "device": provider.device,
        "language": provider.language,
        "native_sample_rate": provider.native_sample_rate,
        "sample_rate": provider.sample_rate,
        "default_voice": provider.default_voice,
        "voices": provider.voices,
        "limit": limiter.limit,
        "active": limiter.active,
        "waiting": limiter.waiting,
    }


@router.get("/voices")
async def list_voices(
    provider: TTSProvider = Depends(get_provider),
) -> dict:
    """Available voices/descriptions for client configuration."""
    return {
        "language": provider.language,
        "default_voice": provider.default_voice,
        "voices": provider.voices,
    }
