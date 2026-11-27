"""Unit tests for the cloud TTS drivers (Azure + Edge)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from app.configs import Settings
from app.contracts import MAX_SPEAKABLE_CHARS
from app.pipeline.tts.base import AudioChunk, TTSError, Voice, create_tts_engine
from app.pipeline.tts.cloud import CloudTTS, _AzureTTSDriver, _EdgeTTSDriver


def _make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "_env_file": None,
        "tts_provider": "cloud",
        "tts_cloud_provider": "edge",
        "edge_male_voice": "ar-EG-ShakirNeural",
        "edge_female_voice": "ar-EG-SalmaNeural",
        "azure_male_voice": "ar-EG-ShakirNeural",
        "azure_female_voice": "ar-EG-SalmaNeural",
        "azure_speech_key": "test-azure-key",
        "azure_speech_region": "eastus",
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Factory routing
# ---------------------------------------------------------------------------

def test_factory_creates_cloud_tts_for_cloud_provider() -> None:
    s = _make_settings(tts_provider="cloud", tts_cloud_provider="edge")
    engine = create_tts_engine(s)
    assert isinstance(engine, CloudTTS)


def test_factory_routes_voicetut_not_cloud() -> None:
    from app.pipeline.tts.voicetut import VoiceTutTTS

    s = _make_settings(tts_provider="voicetut")
    engine = create_tts_engine(s)
    assert isinstance(engine, VoiceTutTTS)


# ---------------------------------------------------------------------------
# CloudTTS validation
# ---------------------------------------------------------------------------

def test_cloud_tts_rejects_text_exceeding_max_speakable_chars() -> None:
    engine = CloudTTS(_make_settings())
    engine.sample_rate = 24000

    async def _run() -> None:
        with pytest.raises(TTSError, match="MAX_SPEAKABLE_CHARS"):
            async for _ in engine.synthesize_stream("أ" * (MAX_SPEAKABLE_CHARS + 1), "male"):
                pass

    asyncio.run(_run())


def test_cloud_tts_skips_empty_and_unpronounceable_text() -> None:
    engine = CloudTTS(_make_settings())
    engine.sample_rate = 24000

    async def _run() -> None:
        for empty in ("", "   \n\t  ", "؟!.,..."):
            chunks = [c async for c in engine.synthesize_stream(empty, "female")]
            assert chunks == []

    asyncio.run(_run())


def test_cloud_tts_raises_when_not_warmed_up() -> None:
    engine = CloudTTS(_make_settings())

    async def _run() -> None:
        with pytest.raises(TTSError, match="not initialized"):
            async for _ in engine.synthesize_stream("مرحبا", "male"):
                pass

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Edge TTS driver
# ---------------------------------------------------------------------------

def test_edge_driver_selects_correct_voice() -> None:
    s = _make_settings()
    driver = _EdgeTTSDriver(s)
    assert driver._speaker_for_voice("male") == "ar-EG-ShakirNeural"
    assert driver._speaker_for_voice("female") == "ar-EG-SalmaNeural"
    with pytest.raises(TTSError, match="Unknown voice"):
        driver._speaker_for_voice("robot")  # type: ignore[arg-type]


def test_edge_driver_raises_on_missing_library() -> None:
    s = _make_settings()
    driver = _EdgeTTSDriver(s)

    # Simulate edge_tts not being installed by patching __init__ state
    driver._edge_tts = None

    async def _run() -> None:
        with pytest.raises(TTSError, match="not initialized"):
            async for _ in driver.synthesize_stream("مرحبا", "male"):
                pass

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Azure TTS driver
# ---------------------------------------------------------------------------

def test_azure_driver_selects_correct_voice() -> None:
    s = _make_settings()
    driver = _AzureTTSDriver(s)
    assert driver._speaker_for_voice("male") == "ar-EG-ShakirNeural"
    assert driver._speaker_for_voice("female") == "ar-EG-SalmaNeural"
    with pytest.raises(TTSError, match="Unknown voice"):
        driver._speaker_for_voice("robot")  # type: ignore[arg-type]


def test_azure_driver_requires_api_key() -> None:
    s = _make_settings(azure_speech_key="")
    driver = _AzureTTSDriver(s)

    async def _run() -> None:
        with pytest.raises(TTSError, match="AZURE_SPEECH_KEY"):
            await driver.warmup()

    asyncio.run(_run())


def test_azure_tts_validation_in_config() -> None:
    """Config validator catches missing Azure key when tts_provider=cloud, tts_cloud_provider=azure."""
    s = _make_settings(tts_provider="cloud", tts_cloud_provider="azure", azure_speech_key="")
    with pytest.raises(ValueError, match="AZURE_SPEECH_KEY"):
        s.validate_for_runtime()


# ---------------------------------------------------------------------------
# CloudTTS dispatching
# ---------------------------------------------------------------------------

def test_cloud_tts_uses_edge_driver_by_default() -> None:
    engine = CloudTTS(_make_settings(tts_cloud_provider="edge"))
    assert isinstance(engine._driver, _EdgeTTSDriver)


def test_cloud_tts_uses_azure_driver_when_configured() -> None:
    engine = CloudTTS(_make_settings(tts_cloud_provider="azure"))
    assert isinstance(engine._driver, _AzureTTSDriver)


def test_cloud_tts_close_is_idempotent() -> None:
    engine = CloudTTS(_make_settings())

    async def _run() -> None:
        await engine.close()
        await engine.close()  # Should not raise

    asyncio.run(_run())
