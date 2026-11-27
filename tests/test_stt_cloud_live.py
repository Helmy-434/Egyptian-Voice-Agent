import asyncio
import os
from pathlib import Path
import wave

import pytest

from app.configs import Settings
from app.pipeline.stt.base import STTResult
from app.pipeline.stt.cloud import CloudSTT

_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
_CLIP_PATH = Path("scripts/eval_clips/clip_001.wav")


@pytest.mark.live
@pytest.mark.skipif(not _GEMINI_API_KEY, reason="GEMINI_API_KEY is not set")
def test_live_gemini_transcribes_eval_clip() -> None:
    if not _CLIP_PATH.is_file():
        pytest.skip("scripts/eval_clips/clip_001.wav is not present")

    settings = Settings(_env_file=None, gemini_api_key=_GEMINI_API_KEY)
    engine = CloudSTT(settings)

    with wave.open(str(_CLIP_PATH), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        audio = wav_file.readframes(wav_file.getnframes())

    async def transcribe_clip() -> STTResult:
        await engine.warmup()
        try:
            return await engine.transcribe(audio, sample_rate=sample_rate)
        finally:
            await engine.close()

    result = asyncio.run(transcribe_clip())

    assert isinstance(result.text, str)
    assert result.text.strip()
