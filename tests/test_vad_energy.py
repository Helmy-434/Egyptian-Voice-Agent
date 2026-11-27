"""Unit tests for the energy-based VAD fallback."""

from __future__ import annotations

import struct

import pytest

from app.pipeline.vad_energy import EnergyVAD


def _silent_frame(sample_rate: int = 16000) -> bytes:
    n = 512 if sample_rate == 16000 else 256
    return b"\x00\x00" * n


def _loud_frame(sample_rate: int = 16000, amplitude: int = 16000) -> bytes:
    n = 512 if sample_rate == 16000 else 256
    return struct.pack(f"<{n}h", *([amplitude, -amplitude] * (n // 2)))


def test_silent_frame_returns_zero_prob() -> None:
    vad = EnergyVAD(16000)
    assert vad.prob(_silent_frame()) == 0.0


def test_loud_frame_returns_high_prob() -> None:
    vad = EnergyVAD(16000)
    prob = vad.prob(_loud_frame())
    assert prob > 0.5


def test_very_loud_frame_returns_one() -> None:
    vad = EnergyVAD(16000, energy_threshold=0.01)
    prob = vad.prob(_loud_frame(amplitude=30000))
    assert prob == 1.0


def test_reset_does_not_raise() -> None:
    vad = EnergyVAD(16000)
    vad.reset()  # Should be a no-op


def test_wrong_frame_size_raises() -> None:
    vad = EnergyVAD(16000)
    with pytest.raises(ValueError, match="Expected"):
        vad.prob(b"\x00" * 100)


def test_8khz_accepts_correct_frame_size() -> None:
    vad = EnergyVAD(8000)
    assert vad.prob(_silent_frame(8000)) == 0.0


def test_unsupported_sample_rate_raises() -> None:
    with pytest.raises(ValueError, match="supports 8000 or 16000"):
        EnergyVAD(44100)  # type: ignore[arg-type]
