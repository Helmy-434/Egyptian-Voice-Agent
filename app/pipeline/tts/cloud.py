from __future__ import annotations
import asyncio
import io
import logging
import unicodedata
from collections.abc import AsyncIterator
from typing import Any

from app.configs import Settings
from app.contracts import MAX_SPEAKABLE_CHARS
from app.pipeline.tts.base import AudioChunk, TTSError, Voice

logger = logging.getLogger(__name__)

_EDGE_SAMPLE_RATE = 24000
_AZURE_SAMPLE_RATE = 24000
_PCM_CHUNK_SAMPLES = 4800  # 200 ms at 24 kHz


def _has_pronounceable_text(text: str) -> bool:
    return any(unicodedata.category(c).startswith("L") or c.isdigit() for c in text)


# ---------------------------------------------------------------------------
# Edge TTS (free, no API key)
# ---------------------------------------------------------------------------

class _EdgeTTSDriver:
    """Thin async wrapper around the ``edge-tts`` library."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._edge_tts: Any = None
        self.sample_rate: int = 0

    async def warmup(self) -> None:
        try:
            import edge_tts  # noqa: F401
        except ImportError as exc:
            raise TTSError(
                "edge-tts is required for TTS_CLOUD_PROVIDER=edge. "
                "Install it with: pip install edge-tts"
            ) from exc
        self._edge_tts = edge_tts
        self.sample_rate = _EDGE_SAMPLE_RATE

    async def synthesize_stream(
            self, text: str, voice: Voice
    ) -> AsyncIterator[AudioChunk]:
        if self._edge_tts is None:
            raise TTSError(
                "Edge TTS is not initialized; call warmup() at startup."
            )

        speaker = self._speaker_for_voice(voice)
        communicate = self._edge_tts.Communicate(text, speaker)

        mp3_buffer = bytearray()

        async for event in communicate.stream():
            if event["type"] == "audio":
                raw = event["data"]
                if raw:
                    mp3_buffer.extend(raw)

        if not mp3_buffer:
            return

        # Decode the complete MP3 once instead of decoding every fragment.
        pcm = await asyncio.to_thread(
            _mp3_to_pcm16,
            bytes(mp3_buffer),
            _EDGE_SAMPLE_RATE,
        )

        chunk_bytes = _PCM_CHUNK_SAMPLES * 2

        for offset in range(0, len(pcm), chunk_bytes):
            chunk = pcm[offset:offset + chunk_bytes]

            if len(chunk) % 2:
                chunk = chunk[:-1]

            if chunk:
                yield AudioChunk(
                    pcm=chunk,
                    sample_rate=_EDGE_SAMPLE_RATE,
                )

    async def close(self) -> None:
        self._edge_tts = None

    def _speaker_for_voice(self, voice: Voice) -> str:
        if voice == "male":
            return self._settings.edge_male_voice
        if voice == "female":
            return self._settings.edge_female_voice
        raise TTSError(f"Unknown voice: {voice!r}. Expected 'male' or 'female'.")


def _mp3_to_pcm16(mp3_data: bytes, target_rate: int) -> bytes:
    """Decode MP3 bytes to mono 16-bit PCM using the ``io`` + ``wave`` fallback.

    edge-tts returns audio/mpeg chunks. We decode them with the ``pydub``
    library if available, otherwise fall through to raw pass-through (the
    edge-tts library can be configured to return PCM directly via SSML
    output format, but that is fragile across versions).
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise TTSError(
            "pydub is required to decode Edge TTS MP3 audio. "
            "Install it with: pip install pydub"
        )

    try:
        seg = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")
    except Exception as exc:
        raise TTSError(f"Failed to decode Edge TTS MP3 audio: {exc}") from exc

    seg = seg.set_channels(1).set_sample_width(2).set_frame_rate(target_rate)
    return seg.raw_data


# ---------------------------------------------------------------------------
# Azure Cognitive Services Speech
# ---------------------------------------------------------------------------

class _AzureTTSDriver:
    """Wrapper around ``azure-cognitiveservices-speech`` with pull-stream output."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._speechsdk: Any = None
        self.sample_rate: int = 0

    async def warmup(self) -> None:
        key = self._settings.azure_speech_key.get_secret_value()
        if not key:
            raise TTSError("AZURE_SPEECH_KEY is required for TTS_CLOUD_PROVIDER=azure.")

        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as exc:
            raise TTSError(
                "azure-cognitiveservices-speech is required for TTS_CLOUD_PROVIDER=azure. "
                "Install it with: pip install azure-cognitiveservices-speech"
            ) from exc

        self._speechsdk = speechsdk
        self.sample_rate = _AZURE_SAMPLE_RATE

    async def synthesize_stream(self, text: str, voice: Voice) -> AsyncIterator[AudioChunk]:
        if self._speechsdk is None:
            raise TTSError("Azure TTS is not initialized; call warmup() at startup.")

        speaker = self._speaker_for_voice(voice)
        pcm = await asyncio.to_thread(self._synthesize_sync, text, speaker)
        if not pcm:
            return

        chunk_bytes = _PCM_CHUNK_SAMPLES * 2
        for offset in range(0, len(pcm), chunk_bytes):
            chunk = pcm[offset: offset + chunk_bytes]
            if len(chunk) % 2:
                chunk = chunk[:-1]
            if chunk:
                yield AudioChunk(pcm=chunk, sample_rate=_AZURE_SAMPLE_RATE)

    async def close(self) -> None:
        self._speechsdk = None

    def _synthesize_sync(self, text: str, speaker: str) -> bytes:
        """Run Azure synthesis in a worker thread (blocking SDK call)."""
        speechsdk = self._speechsdk
        key = self._settings.azure_speech_key.get_secret_value()
        region = self._settings.azure_speech_region

        speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
        )
        speech_config.speech_synthesis_voice_name = speaker

        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config, audio_config=None
        )

        result = synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            raise TTSError(
                f"Azure TTS synthesis canceled: {cancellation.reason} — "
                f"{cancellation.error_details}"
            )
        else:
            raise TTSError(f"Azure TTS unexpected result: {result.reason}")

    def _speaker_for_voice(self, voice: Voice) -> str:
        if voice == "male":
            return self._settings.azure_male_voice
        if voice == "female":
            return self._settings.azure_female_voice
        raise TTSError(f"Unknown voice: {voice!r}. Expected 'male' or 'female'.")


# ---------------------------------------------------------------------------
# Unified CloudTTS façade
# ---------------------------------------------------------------------------

class CloudTTS:
    """Dispatches to the configured cloud TTS provider (Azure or Edge).

    Satisfies the ``TTSEngine`` protocol.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        provider = settings.tts_cloud_provider

        if provider == "edge":
            self._driver: _EdgeTTSDriver | _AzureTTSDriver = _EdgeTTSDriver(settings)
        elif provider == "azure":
            self._driver = _AzureTTSDriver(settings)
        else:
            raise TTSError(f"Unknown tts_cloud_provider: {provider!r}")

        self.sample_rate: int = 0

    async def warmup(self) -> None:
        await self._driver.warmup()
        self.sample_rate = self._driver.sample_rate

    async def synthesize_stream(self, text: str, voice: Voice) -> AsyncIterator[AudioChunk]:
        if len(text) > MAX_SPEAKABLE_CHARS:
            raise TTSError(f"Text exceeds MAX_SPEAKABLE_CHARS ({MAX_SPEAKABLE_CHARS}).")
        if not text.strip() or not _has_pronounceable_text(text):
            return
        if self.sample_rate == 0:
            raise TTSError("Cloud TTS is not initialized; call warmup() at startup.")

        try:
            async for chunk in self._driver.synthesize_stream(text, voice):
                yield chunk
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"Cloud TTS synthesis failed: {exc}") from exc

    async def close(self) -> None:
        await self._driver.close()
