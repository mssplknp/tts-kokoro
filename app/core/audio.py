"""Float audio -> signed 16-bit little-endian PCM, and frame alignment."""

import torch


def to_pcm16(audio: torch.Tensor) -> bytes:
    """Convert a float32 mono tensor in ``[-1, 1]`` to ``s16le`` PCM bytes.

    Rounding (rather than truncation) keeps the conversion unbiased; values are
    clamped so out-of-range samples cannot wrap on cast.
    """
    if audio.ndim != 1:
        raise ValueError(
            f"expected 1-D mono audio, got shape {tuple(audio.shape)}"
        )
    pcm = (
        audio.detach()
        .float()
        .clamp(-1.0, 1.0)
        .mul(32767.0)
        .round()
        .to(torch.int16)
    )
    return pcm.cpu().numpy().tobytes()


class PcmFrameAccumulator:
    """Re-chunk raw PCM bytes into fixed-size frames.

    Mirrors the agent-side ``AudioChunker``: 20 ms at 44100 Hz mono s16le is
    ``44100 * 0.02 * 1 * 2 = 1764`` bytes/frame. Keeping wire chunks aligned to
    this size avoids partial-frame buffering latency in the agent.
    """

    def __init__(self, frame_size: int):
        if frame_size <= 0:
            raise ValueError("frame_size must be positive")
        self._frame_size = frame_size
        self._buffer = bytearray()

    def push(self, chunk: bytes) -> list[bytes]:
        """Add PCM bytes; return complete frames. Partial data is buffered."""
        self._buffer.extend(chunk)
        frames: list[bytes] = []
        while len(self._buffer) >= self._frame_size:
            frames.append(bytes(self._buffer[: self._frame_size]))
            del self._buffer[: self._frame_size]
        return frames

    def flush(self) -> bytes:
        """Return the remaining partial frame (empty bytes if none)."""
        if not self._buffer:
            return b""
        remainder = bytes(self._buffer)
        self._buffer.clear()
        return remainder
