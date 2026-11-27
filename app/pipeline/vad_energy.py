
from __future__ import annotations

import struct
from typing import Literal


class EnergyVAD:
    """Energy-based VAD: speech when RMS exceeds a fixed threshold.

    Satisfies the ``VADModel`` protocol (``prob(frame) -> float`` and
    ``reset() -> None``).
    """

    def __init__(
        self,
        sample_rate: Literal[8000, 16000] = 16000,
        *,
        energy_threshold: float = 0.02,
    ) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError(f"EnergyVAD supports 8000 or 16000 Hz, got {sample_rate}")
        self._sample_rate = sample_rate
        self._expected_bytes = (512 if sample_rate == 16000 else 256) * 2
        self._threshold = energy_threshold

    def prob(self, frame: bytes) -> float:
        """Return a pseudo-probability (0.0–1.0) based on RMS energy."""
        if len(frame) != self._expected_bytes:
            raise ValueError(
                f"Expected {self._expected_bytes} bytes for {self._sample_rate} Hz, "
                f"got {len(frame)} bytes"
            )
        n_samples = len(frame) // 2
        samples = struct.unpack(f"<{n_samples}h", frame)

        # RMS in normalized float range [-1, 1]
        sum_sq = sum(s * s for s in samples)
        rms = (sum_sq / n_samples) ** 0.5 / 32768.0

        # Map RMS to a 0–1 probability using a simple sigmoid-like curve
        if rms < self._threshold * 0.5:
            return 0.0
        if rms > self._threshold * 2.0:
            return 1.0
        # Linear ramp in the transition zone
        return min(1.0, max(0.0, (rms - self._threshold * 0.5) / (self._threshold * 1.5)))

    def reset(self) -> None:
        """No internal state to reset."""
        pass
