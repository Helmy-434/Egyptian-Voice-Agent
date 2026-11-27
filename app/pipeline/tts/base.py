from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from app.configs import Settings
from app.contracts import MAX_SPEAKABLE_CHARS

__all__ = [
    "AudioChunk",
    "MAX_SPEAKABLE_CHARS",
    "TTSEngine",
    "TTSError",
    "Voice",
    "create_tts_engine",
]

Voice = Literal["male", "female"]


class TTSError(Exception):
    """Raised by a TTS engine on model/provider failure, invalid input, or timeout."""


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """One mono signed 16-bit little-endian PCM audio chunk."""

    pcm: bytes
    sample_rate: int


@runtime_checkable
class TTSEngine(Protocol):
    """Async interface shared by local and cloud text-to-speech drivers."""

    sample_rate: int

    async def warmup(self) -> None:
        """Load the synthesis backend and set ``sample_rate`` before serving calls."""
        ...

    def synthesize_stream(self, text: str, voice: Voice) -> AsyncIterator[AudioChunk]:
        """Synthesize one speakable agent chunk of at most ``MAX_SPEAKABLE_CHARS``.

        Yield audio as soon as it exists; the first yield marks time-to-first-audio.
        Empty, whitespace-only, and unpronounceable text yields nothing. Longer text
        or an unknown voice raises ``TTSError``. Cancellation or stopping iteration
        must stop synthesis promptly and leave the engine reusable.

        Every yielded chunk is mono PCM16 little-endian; implementations convert
        other model formats internally and report the actual output sample rate.
        """
        ...

    async def close(self) -> None:
        """Release backend resources without blocking the event loop."""
        ...


def create_tts_engine(settings: Settings) -> TTSEngine:
    """Create the selected driver without importing local GPU dependencies eagerly."""
    if settings.tts_provider == "voicetut":
        from app.pipeline.tts.voicetut import VoiceTutTTS

        return VoiceTutTTS(settings)

    if settings.tts_provider == "cloud":
        from app.pipeline.tts.cloud import CloudTTS

        return CloudTTS(settings)

    raise ValueError(f"Unsupported tts_provider: {settings.tts_provider!r}")
