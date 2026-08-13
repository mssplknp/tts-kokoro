"""TTS backends (Kokoro)."""

from app.providers.base import TTSProvider
from app.providers.kokoro import KokoroProvider

__all__ = ["TTSProvider", "KokoroProvider"]
