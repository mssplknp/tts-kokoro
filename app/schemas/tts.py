"""Request models for the TTS API (contract in the project README)."""

from pydantic import BaseModel, Field, field_validator


class TTSRequest(BaseModel):
    """Body accepted by ``POST /v1/tts/stream`` and ``POST /v1/tts/synthesize``.

    Only ``text`` is required — that is all the current LiveKit clients send.
    Every other field is an optional extension that overrides a server default.
    """

    text: str
    language: str | None = None
    voice: str | None = None
    speed: float | None = Field(default=None, gt=0, le=3.0)
    sample_rate: int | None = Field(default=None, ge=8000, le=192_000)
    request_id: str | None = None

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must be a non-empty string")
        return value
