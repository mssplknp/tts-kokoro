"""Provider interface shared by every TTS backend."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator


class TTSProvider(ABC):
    """Backend contract.

    Implementations expose read-only metadata attributes (below) plus two
    streaming methods that yield raw ``s16le`` mono PCM bytes incrementally.
    """

    #: Native sample rate the model produces (e.g. 24000 for Kokoro).
    native_sample_rate: int
    #: Default output sample rate (the client contract, e.g. 44100).
    sample_rate: int
    #: torch device in use ("cpu", "mps", "cuda:0", ...).
    device: str
    model_name: str
    language: str
    default_voice: str
    default_speed: float
    voices: list[str]
    supported_languages: set[str]

    @abstractmethod
    def stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
        sample_rate: int | None = None,
    ) -> Iterator[bytes]:
        """Synthesize ``text`` and yield raw ``s16le`` PCM chunks.

        Chunks are produced incrementally so the first bytes can start playing
        before the full utterance is ready. ``sample_rate`` selects the output
        rate; None uses :attr:`sample_rate`.
        """

    async def astream(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
        sample_rate: int | None = None,
    ) -> AsyncIterator[bytes]:
        """Async variant of :meth:`stream` for the HTTP layer.

        The default implementation iterates the synchronous stream on the event
        loop and is only suitable for backends that are fast and safe to call
        there. Backends doing heavy work (e.g. Kokoro) override this to run
        each chunk off the loop via ``asyncio.to_thread``.
        """
        for chunk in self.stream(
            text,
            voice=voice,
            speed=speed,
            sample_rate=sample_rate,
        ):
            yield chunk

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
        sample_rate: int | None = None,
    ) -> bytes:
        """Non-streaming convenience: return the whole utterance as PCM bytes."""
        return b"".join(
            self.stream(
                text,
                voice=voice,
                speed=speed,
                sample_rate=sample_rate,
            )
        )

    async def asynthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
        sample_rate: int | None = None,
    ) -> bytes:
        """Async non-streaming convenience for the HTTP layer."""
        chunks: list[bytes] = []
        async for chunk in self.astream(
            text,
            voice=voice,
            speed=speed,
            sample_rate=sample_rate,
        ):
            chunks.append(chunk)
        return b"".join(chunks)
