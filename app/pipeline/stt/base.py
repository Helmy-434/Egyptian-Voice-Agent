from dataclasses import dataclass
import io
from typing import Literal, Protocol, runtime_checkable
import wave

from app.configs import Settings

SampleRate = Literal[8000, 16000]


class STTError(Exception):
    """Raised by any STTEngine implementation on inference/API failure or timeout."""


@dataclass(frozen=True, slots=True)
class STTResult:
    """Normalized transcript and optional confidence in the range 0.0-1.0."""

    text: str
    confidence: float | None = None


@runtime_checkable
class STTEngine(Protocol):
    """Async interface shared by local GPU and cloud speech-to-text drivers."""

    async def warmup(self) -> None:
        """Preload model weights or initialize HTTP clients during app startup."""
        ...

    async def transcribe(
        self,
        audio: bytes,
        sample_rate: SampleRate = 16000,
    ) -> STTResult:
        """Transcribe one complete utterance of mono little-endian PCM16 audio.

        Contract for implementations:
        - Input format: Raw mono signed 16-bit little-endian PCM bytes (browser
          AudioWorklet Float32 samples are converted to Int16 on the client).
        - Non-blocking & GPU-safe: Local synchronous models (e.g., QwenCleo-ASR)
          MUST run off the event loop (e.g., via a single-worker
          `ThreadPoolExecutor`) and be serialized so only one inference runs on
          the GPU at a time, even after the awaiting task is cancelled.
        - Empty/too-short audio: Return `STTResult(text="", confidence=None)`
          without raising so the orchestrator drops it cleanly.
        - Output hygiene: Return `text` stripped of leading/trailing whitespace
          and any model-specific special tokens.
        - Errors: Raise `STTError` on provider/model failure or timeout.
        """
        ...

    async def close(self) -> None:
        """Release GPU memory or close HTTP connection pools on shutdown."""
        ...


def pcm16_to_wav_bytes(audio: bytes, sample_rate: SampleRate = 16000) -> bytes:
    """Wrap raw mono PCM16 frames in an in-memory RIFF/WAV header for cloud APIs."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)
    return buf.getvalue()


def create_stt_engine(settings: Settings) -> STTEngine:
    """Lazy factory so CPU cloud deployments never import torch or qwen-asr."""
    if settings.stt_provider == "qwencleo":
        from app.pipeline.stt.qwencleo import QwenCleoSTT

        return QwenCleoSTT(settings)

    if settings.stt_provider == "cloud":
        from app.pipeline.stt.cloud import CloudSTT

        return CloudSTT(settings)

    raise ValueError(f"Unsupported stt_provider: {settings.stt_provider!r}")