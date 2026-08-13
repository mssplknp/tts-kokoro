"""Application configuration.

Settings are read from environment variables or a local ``.env`` file and can
be overridden at the process level (e.g. ``PORT=9000 uvicorn app.main:app``).
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Server configuration. See ``.env.example`` for every field."""

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8001

    # Model
    MODEL_REPO_ID: str = "hexgrad/Kokoro-82M"
    LANG_CODE: str = "h"
    DEFAULT_VOICE: str = "hm_psi"
    DEVICE: str = "auto"  # auto | cpu | mps | cuda
    SPEED: float = 1.0
    WARMUP_TEXT: str = "नमस्ते, मैं किसान मित्र हूँ।"

    # Audio contract
    NATIVE_SAMPLE_RATE: int = 24000
    SAMPLE_RATE: int = 44100
    FRAME_SIZE: int = 1764  # 20 ms @ 44100 Hz mono s16le

    # Load / concurrency
    CONCURRENCY_LIMIT: int = Field(default=4, ge=1)
    QUEUE_WAIT_TIMEOUT: float = Field(default=10.0, gt=0)

    # Observability
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @property
    def supported_language_aliases(self) -> set[str]:
        """Accepted values for the request ``language`` field."""
        return {"h", "hi"}


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (settings are immutable at runtime)."""
    return Settings()
