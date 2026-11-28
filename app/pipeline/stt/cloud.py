import asyncio
from typing import Any
from google.genai import types
from app.configs import Settings
from app.pipeline.stt.base import (
    STTError,
    STTResult,
    SampleRate,
    pcm16_to_wav_bytes,
)

_MIN_AUDIO_SEC = 0.10


class CloudSTT:
    """Cloud speech-to-text driver backed by Gemini's transcription model."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None
        self._aio_client: Any | None = None

    async def warmup(self) -> None:
        """Initialize the async Gemini client without loading local model weights."""
        if self._aio_client is not None:
            return

        api_key = self._settings.gemini_api_key.get_secret_value()
        if not api_key:
            raise STTError(
                "GEMINI_API_KEY is required when STT_PROVIDER=cloud."
            )

        try:
            from google import genai
        except ImportError as exc:
            raise STTError(
                "google-genai is required for cloud STT. "
                "Install requirements.txt."
            ) from exc

        try:
            client = genai.Client(
                api_key=api_key,
                http_options={"api_version": "v1beta"},
            )
        except Exception as exc:
            raise STTError(
                f"Failed to initialize the Gemini STT client: {exc}"
            ) from exc

        self._client = client
        self._aio_client = client.aio

    async def transcribe(
        self,
        audio: bytes,
        sample_rate: SampleRate = 16000,
    ) -> STTResult:
        """Transcribe one utterance using Gemini's GenerateContent API."""

        aligned_len = len(audio) - (len(audio) % 2)
        audio_sec = aligned_len / (sample_rate * 2)

        if audio_sec < _MIN_AUDIO_SEC:
            return STTResult(text="", confidence=None)

        if self._aio_client is None:
            raise STTError(
                "Gemini STT is not initialized; call warmup() at startup."
            )

        wav_audio = pcm16_to_wav_bytes(
            audio[:aligned_len],
            sample_rate,
        )

        try:
            response = await asyncio.wait_for(
                self._aio_client.models.generate_content(
                    model=self._settings.gemini_stt_model,
                    contents=[
                        types.Part.from_bytes(
                            data=wav_audio,
                            mime_type="audio/wav",
                        )
                    ],
                    config=types.GenerateContentConfig(
                        audio_transcription_config=types.AudioTranscriptionConfig(
                            language_codes=["ar-EG"],
                        )
                    ),
                ),
                timeout=self._settings.stt_cloud_timeout_sec,
            )

        except asyncio.TimeoutError as exc:
            raise STTError(
                f"Gemini STT timed out after "
                f"{self._settings.stt_cloud_timeout_sec:.2f}s "
                f"(audio={audio_sec:.2f}s)"
            ) from exc

        except Exception as exc:
            raise STTError(
                f"Gemini STT request failed: {exc}"
            ) from exc

        parts = getattr(response, "parts", []) or []
        transcript_parts: list[str] = []

        for part in parts:
            part_text = getattr(part, "text", None)

            if part_text:
                transcript_parts.append(part_text)
                continue

            audio_transcription = getattr(
                part,
                "audio_transcription",
                None,
            )
            transcription_text = getattr(
                audio_transcription,
                "text",
                None,
            )

            if transcription_text:
                transcript_parts.append(transcription_text)

        text = "".join(transcript_parts).strip()

        if not text:
            raise STTError(
                "Gemini returned an empty transcription."
            )

        return STTResult(
            text=text,
            confidence=None,
        )

    async def close(self) -> None:
        """Close the async and synchronous clients used by the Gemini SDK."""

        client = self._client
        aio_client = self._aio_client

        self._client = None
        self._aio_client = None

        close_errors: list[Exception] = []

        if aio_client is not None:
            try:
                await aio_client.aclose()
            except Exception as exc:
                close_errors.append(exc)

        if client is not None:
            try:
                await asyncio.to_thread(client.close)
            except Exception as exc:
                close_errors.append(exc)

        if close_errors:
            raise STTError(
                f"Failed to close the Gemini STT client: "
                f"{close_errors[0]}"
            ) from close_errors[0]