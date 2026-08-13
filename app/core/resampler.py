"""Sample-rate conversion via torchaudio's windowed-sinc resampler.

torchaudio ships the same high-quality resampler the reference ``tts-server``
relied on (Hann-windowed sinc with a low-pass filter). Resampling is applied
per synthesis unit; the kernel and MPS/CUDA kernels are compiled once during
startup warm-up so the first live request is not penalized.
"""

import torch
import torchaudio


class AudioResampler:
    """Resample a 1-D mono tensor between two fixed sample rates."""

    def __init__(self, orig_freq: int, new_freq: int):
        if orig_freq <= 0 or new_freq <= 0:
            raise ValueError("sample rates must be positive")
        self._orig_freq = orig_freq
        self._new_freq = new_freq

    @property
    def enabled(self) -> bool:
        return self._orig_freq != self._new_freq

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return audio
        return torchaudio.functional.resample(
            audio.unsqueeze(0), self._orig_freq, self._new_freq
        ).squeeze(0)
