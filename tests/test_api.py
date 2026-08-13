"""API-level tests using the fake provider (no model required)."""

import asyncio

import pytest

from app.core.audio import PcmFrameAccumulator
from app.core.concurrency import ConcurrencyLimiter


def _pcm_length(text: str) -> int:
    return 2 * len(text)  # FakeProvider emits 2 bytes per char


# --------------------------------------------------------------------------- #
# /stream
# --------------------------------------------------------------------------- #


def test_stream_returns_pcm_with_expected_headers(client):
    resp = client.post("/v1/tts/stream", json={"text": "नमस्ते किसान भाई।"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"
    assert resp.headers["X-Sample-Rate"] == "44100"
    assert resp.headers["X-Channels"] == "1"
    assert resp.headers["X-Sample-Format"] == "s16le"
    assert resp.headers["X-Request-Id"]
    assert len(resp.content) == _pcm_length("नमस्ते किसान भाई।")


def test_stream_echoes_request_id(client):
    resp = client.post(
        "/v1/tts/stream",
        json={"text": "नमस्ते।", "request_id": "speech_abc123"},
    )
    assert resp.headers["X-Request-Id"] == "speech_abc123"


def test_stream_honours_requested_sample_rate(client):
    resp = client.post(
        "/v1/tts/stream",
        json={"text": "नमस्ते।", "sample_rate": 24000},
    )
    assert resp.headers["X-Sample-Rate"] == "24000"


def test_stream_accepts_optional_fields(client):
    resp = client.post(
        "/v1/tts/stream",
        json={"text": "नमस्ते।", "language": "hi", "voice": "hm_omega", "speed": 1.2},
    )
    assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# Validation -> 400
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("body", [
    {"text": ""},
    {"text": "   "},
    {"text": "\n\t"},
])
def test_stream_rejects_blank_text(client, body):
    assert client.post("/v1/tts/stream", json=body).status_code == 400


@pytest.mark.parametrize("speed", [0, -1, 3.1, 100])
def test_stream_rejects_invalid_speed(client, speed):
    resp = client.post("/v1/tts/stream", json={"text": "नमस्ते।", "speed": speed})
    assert resp.status_code == 400


def test_stream_rejects_unsupported_language(client):
    resp = client.post(
        "/v1/tts/stream", json={"text": "नमस्ते।", "language": "en"}
    )
    assert resp.status_code == 400


def test_stream_rejects_unknown_voice(client):
    resp = client.post(
        "/v1/tts/stream", json={"text": "नमस्ते।", "voice": "not_a_voice"}
    )
    assert resp.status_code == 400


def test_stream_rejects_invalid_sample_rate(client):
    resp = client.post(
        "/v1/tts/stream", json={"text": "नमस्ते।", "sample_rate": 100}
    )
    assert resp.status_code == 400


def test_stream_rejects_malformed_body(client):
    assert client.post("/v1/tts/stream", json={"textt": "oops"}).status_code == 400
    assert client.post("/v1/tts/stream", content="not json").status_code in (400, 415)


# --------------------------------------------------------------------------- #
# /synthesize
# --------------------------------------------------------------------------- #


def test_synthesize_returns_full_buffer(client):
    text = "आप चाहें तो मुझे अपनी फसल की उम्र बता सकते हैं।"
    resp = client.post("/v1/tts/synthesize", json={"text": text})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/octet-stream"
    assert resp.headers["X-Sample-Format"] == "s16le"
    assert len(resp.content) == _pcm_length(text)


def test_synthesize_rejects_blank_text(client):
    assert client.post("/v1/tts/synthesize", json={"text": " "}).status_code == 400


# --------------------------------------------------------------------------- #
# /health and /voices
# --------------------------------------------------------------------------- #


def test_health_reports_ok(client):
    resp = client.get("/v1/tts/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model"] == "fake-kokoro"
    assert body["language"] == "h"
    assert body["sample_rate"] == 44100
    assert "active" in body and "waiting" in body and "limit" in body


def test_voices_lists_available_voices(client):
    resp = client.get("/v1/tts/voices")
    assert resp.status_code == 200
    body = resp.json()
    assert body["language"] == "h"
    assert "hm_psi" in body["voices"]
    assert body["default_voice"] == "hm_psi"


# --------------------------------------------------------------------------- #
# Concurrency / overload -> 429
# --------------------------------------------------------------------------- #


async def test_overload_returns_429(async_client):
    client, application = async_client
    # Occupy the only slot directly; a new request must be refused with 429.
    await application.state.limiter.acquire()
    second = await client.post("/v1/tts/stream", json={"text": "y" * 10})
    assert second.status_code == 429
    application.state.limiter.release()
    # Once freed, the slot serves requests again.
    ok = await client.post("/v1/tts/stream", json={"text": "z" * 10})
    assert ok.status_code == 200


# --------------------------------------------------------------------------- #
# Cancellation: the streaming helper must release the slot and stop the worker
# --------------------------------------------------------------------------- #


async def test_pcm_stream_releases_limiter_when_cancelled():
    from app.api.tts import _pcm_stream

    limiter = ConcurrencyLimiter(limit=1, timeout=1)
    await limiter.acquire()

    async def slow_astream():
        while True:
            await asyncio.sleep(0.05)  # awaits inside the generator
            yield b"\x00\x01" * 50  # one chunk == one frame

    accumulator = PcmFrameAccumulator(frame_size=100)
    stream = _pcm_stream(slow_astream(), accumulator, limiter, "test")

    first = await asyncio.create_task(stream.__anext__())
    assert first

    # Second pull is now parked inside the provider's sleep; cancel mid-await.
    task = asyncio.create_task(stream.__anext__())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert limiter.active == 0  # permit released on cancellation
