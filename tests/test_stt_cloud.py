import asyncio
import base64
from io import BytesIO
from types import SimpleNamespace
import wave

import pytest

from app.configs import Settings
from app.pipeline.stt.base import STTResult
from app.pipeline.stt.cloud import CloudSTT


class _FakeInteractions:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.request: dict | None = None

    async def create(self, **request: object) -> SimpleNamespace:
        self.request = request
        return SimpleNamespace(output_text=self.output_text)


def test_gemini_transcription_wraps_pcm_as_wav_and_strips_text() -> None:
    interactions = _FakeInteractions("  أهلا وسهلا  ")
    engine = CloudSTT(Settings(gemini_api_key="test-key"))
    engine._aio_client = SimpleNamespace(interactions=interactions)

    result = asyncio.run(engine.transcribe(b"\x00\x00" * 1600, sample_rate=16000))

    assert result == STTResult(text="أهلا وسهلا", confidence=None)
    assert interactions.request is not None
    assert interactions.request["model"] == "gemini-3.5-transcribe"

    input_parts = interactions.request["input"]
    assert isinstance(input_parts, list)
    audio_part = input_parts[0]
    assert isinstance(audio_part, dict)
    encoded_audio = audio_part["data"]
    assert isinstance(encoded_audio, str)
    wav_data = base64.b64decode(encoded_audio)
    with wave.open(BytesIO(wav_data), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000

    assert interactions.request["generation_config"] == {
        "transcription_config": {"mode": {"type": "verbatim"}}
    }


def test_gemini_transcription_skips_audio_shorter_than_minimum() -> None:
    engine = CloudSTT(Settings(gemini_api_key="test-key"))

    result = asyncio.run(engine.transcribe(b"\x00\x00" * 1599))

    assert result == STTResult(text="", confidence=None)


def test_cloud_stt_requires_gemini_api_key() -> None:
    settings = Settings(stt_provider="cloud")

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        settings.validate_for_runtime()
