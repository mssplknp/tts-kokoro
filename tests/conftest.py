"""Shared fixtures: a fake provider so tests never load the real model."""

import asyncio
import time

import httpx
import pytest

from app.config import Settings
from app.core.concurrency import ConcurrencyLimiter
from app.exceptions import ProviderError
from app.main import create_app
from app.providers.base import TTSProvider


class FakeProvider(TTSProvider):
    """Deterministic provider used in API tests."""

    native_sample_rate = 24000
    sample_rate = 44100
    device = "fake"
    model_name = "fake-kokoro"
    language = "h"
    default_voice = "hm_psi"
    default_speed = 1.0
    voices = ["hm_psi", "hm_omega"]
    supported_languages = {"h", "hi"}

    def __init__(self, *, chunk_size: int = 1000, delay: float = 0.0):
        self.chunk_size = chunk_size
        self.delay = delay

    def stream(self, text, *, voice=None, speed=None, sample_rate=None):
        # Two samples of PCM per char; never empty for non-blank text.
        data = b"\x01\x00" * max(1, len(text))
        for i in range(0, len(data), self.chunk_size):
            if self.delay:
                time.sleep(self.delay)
            yield data[i : i + self.chunk_size]

    async def astream(self, text, *, voice=None, speed=None, sample_rate=None):
        # Async variant that yields on the loop without blocking it, so an
        # open stream keeps its concurrency slot until the client finishes.
        data = b"\x01\x00" * max(1, len(text))
        for i in range(0, len(data), self.chunk_size):
            if self.delay:
                await asyncio.sleep(self.delay)
            yield data[i : i + self.chunk_size]


def _settings(**overrides) -> Settings:
    defaults = {
        "CONCURRENCY_LIMIT": 8,
        "QUEUE_WAIT_TIMEOUT": 0.5,
    }
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture
def app():
    application = create_app(_settings())
    application.state.provider = FakeProvider()
    return application


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client():
    """ASGI transport client for concurrency tests (no lifespan run).

    The provider and limiter are injected directly; the limiter allows exactly
    one in-flight request so overload can be exercised deterministically.
    """
    application = create_app(_settings(CONCURRENCY_LIMIT=1, QUEUE_WAIT_TIMEOUT=0.05))
    application.state.provider = FakeProvider(delay=0.05)
    application.state.limiter = ConcurrencyLimiter(limit=1, timeout=0.05)
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        yield client, application
