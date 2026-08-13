"""Kokoro TTS backend (hexgrad/Kokoro-82M).

Inference notes
---------------
* Kokoro generates at 24000 Hz; the server resamples to the requested output
  rate (44100 Hz by default so the existing LiveKit clients work unchanged).
* Kokoro's pipeline is a *synchronous* generator, so streaming is done at
  chunk granularity: the API layer drives it from a worker thread and the
  first chunk is ready as soon as the first sub-unit is synthesized.
* Both MPS and the espeak-ng G2P layer are not thread-safe. Every synthesis is
  serialized on a :class:`threading.Lock`; the lock is taken per sub-unit so
  concurrent requests interleave fairly instead of blocking each other.
"""

import asyncio
import logging
import threading
import time
from typing import AsyncIterator, Iterator

import torch
from kokoro import KPipeline

from app.config import Settings
from app.core.audio import to_pcm16
from app.core.resampler import AudioResampler
from app.core.text_split import split_for_synthesis
from app.providers.base import TTSProvider

logger = logging.getLogger(__name__)

# Voices shipped with Kokoro-82M per language (subset we advertise).
_KOKORO_VOICES: dict[str, list[str]] = {
    "a": [
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah",
        "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
        "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa",
    ],
    "b": [
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    ],
    "h": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"],
}


def resolve_device(device: str) -> str:
    """Resolve the configured device name to a concrete torch device."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda requested but CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        logger.warning("DEVICE=mps requested but MPS is unavailable; using CPU")
        return "cpu"
    return device


class KokoroProvider(TTSProvider):
    """Streaming Kokoro synthesis backed by a single KPipeline."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = "Kokoro-82M"
        self.language = settings.LANG_CODE
        self.default_voice = settings.DEFAULT_VOICE
        self.default_speed = settings.SPEED
        self.native_sample_rate = settings.NATIVE_SAMPLE_RATE
        self.sample_rate = settings.SAMPLE_RATE
        self.device = resolve_device(settings.DEVICE)
        self.voices = _KOKORO_VOICES.get(
            self.language, [self.default_voice]
        )
        self.supported_languages = settings.supported_language_aliases

        # Serializes pipeline + model inference (MPS and espeak are not
        # thread-safe). Acquired per sub-unit, not per request.
        self._lock = threading.Lock()
        self._resampler = AudioResampler(
            self.native_sample_rate, self.sample_rate
        )

        logger.info(
            "loading Kokoro model (lang=%s voice=%s device=%s repo=%s)",
            self.language,
            self.default_voice,
            self.device,
            settings.MODEL_REPO_ID,
        )
        self._pipeline = KPipeline(
            lang_code=self.language,
            repo_id=settings.MODEL_REPO_ID,
            device=self.device,
        )
        # Preload the default voice so the first request is not stalled on a
        # Hugging Face download.
        self._pipeline.load_voice(self.default_voice)
        self._warmup(settings.WARMUP_TEXT)
        logger.info("Kokoro provider ready (voices=%s)", self.voices)

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
        sample_rate: int | None = None,
    ) -> Iterator[bytes]:
        """Synchronous streaming (used by warm-up, scripts and tests).

        Heavy per-segment work runs inline; prefer :meth:`astream` from the
        event loop so the server stays responsive under concurrent requests.
        """
        voice = voice or self.default_voice
        speed = speed if speed is not None else self.default_speed
        output_rate = sample_rate or self.sample_rate
        resampler = self._resampler_for(output_rate)

        for segment in split_for_synthesis(text):
            pcm = self._segment_to_pcm(segment, voice, speed, resampler)
            if pcm:
                yield pcm

    async def astream(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
        sample_rate: int | None = None,
    ) -> AsyncIterator[bytes]:
        """Async streaming for the HTTP layer.

        Each sub-unit is synthesized off the event loop via
        ``asyncio.to_thread``, so the loop never blocks on inference and
        cancellation between sub-units takes effect as soon as the current
        chunk finishes.
        """
        voice = voice or self.default_voice
        speed = speed if speed is not None else self.default_speed
        output_rate = sample_rate or self.sample_rate
        resampler = self._resampler_for(output_rate)

        for segment in split_for_synthesis(text):
            pcm = await asyncio.to_thread(
                self._segment_to_pcm, segment, voice, speed, resampler
            )
            if pcm:
                yield pcm

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _resampler_for(self, output_rate: int) -> AudioResampler:
        if output_rate == self.sample_rate:
            return self._resampler
        return AudioResampler(self.native_sample_rate, output_rate)

    def _segment_to_pcm(
        self,
        text: str,
        voice: str,
        speed: float,
        resampler: AudioResampler,
    ) -> bytes | None:
        """Synthesize one sub-unit into PCM bytes (runs in a worker thread).

        Phonemization + inference are serialized on :attr:`_lock` because MPS
        and espeak-ng are not thread-safe; resampling and the int16 conversion
        are cheap and stay in the same worker thread.
        """
        audio = self._synthesize(text, voice, speed)
        if audio is None:
            return None
        if resampler.enabled:
            audio = resampler(audio)
        return to_pcm16(audio)

    def _synthesize(
        self,
        text: str,
        voice: str,
        speed: float,
    ) -> torch.Tensor | None:
        """Phonemize + generate one text sub-unit (serialized on the lock)."""
        with self._lock:
            audios = []
            for result in self._pipeline(
                [text],
                voice=voice,
                speed=speed,
            ):
                if result.audio is not None:
                    audios.append(result.audio)
            if not audios:
                return None
            if len(audios) == 1:
                return audios[0]
            return torch.cat(audios, dim=0)

    def _warmup(self, text: str) -> None:
        """Synthesize a short phrase so weights, MPS kernels and the resampler
        are compiled before the first live request arrives."""
        logger.info("warming up model ...")
        started = time.monotonic()
        for _ in self.stream(text):
            pass
        logger.info("warm-up finished in %.2fs", time.monotonic() - started)
