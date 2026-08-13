"""Unit tests for PCM conversion and frame alignment."""

import struct

import numpy as np
import pytest
import torch

from app.core.audio import PcmFrameAccumulator, to_pcm16


def test_to_pcm16_matches_reference_conversion():
    # Reference formula: clip(x, -1, 1) * 32767, cast to int16.
    x = torch.tensor([0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0])
    pcm = to_pcm16(x)
    expected = np.clip(x.numpy(), -1.0, 1.0) * 32767
    expected = np.round(expected).astype(np.int16)
    assert pcm == expected.tobytes()


def test_to_pcm16_is_little_endian_s16():
    x = torch.tensor([0.5])
    raw = to_pcm16(x)
    assert struct.unpack("<h", raw) == (16384,)


def test_to_pcm16_rejects_non_mono():
    with pytest.raises(ValueError):
        to_pcm16(torch.zeros(2, 4))


def test_accumulator_frames_and_remainder():
    acc = PcmFrameAccumulator(frame_size=4)
    assert acc.push(b"ab") == []          # too small, buffered
    assert acc.push(b"cde") == [b"abcd"]  # one frame from 5 bytes
    assert acc.flush() == b"e"            # remainder


def test_accumulator_multi_frame_push():
    acc = PcmFrameAccumulator(frame_size=4)
    frames = acc.push(b"aabbccddee")
    assert frames == [b"aabb", b"ccdd"]
    assert acc.flush() == b"ee"


def test_accumulator_exact_frames_no_remainder():
    acc = PcmFrameAccumulator(frame_size=4)
    assert acc.push(b"aabbccdd") == [b"aabb", b"ccdd"]
    assert acc.flush() == b""


def test_accumulator_rejects_invalid_frame_size():
    with pytest.raises(ValueError):
        PcmFrameAccumulator(frame_size=0)
