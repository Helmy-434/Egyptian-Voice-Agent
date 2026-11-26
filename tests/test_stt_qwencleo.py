import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from typing import cast
from urllib.request import Request
import wave

from pytest import MonkeyPatch

from app.config import Settings
from app.pipeline.stt.base import STTResult
from app.pipeline.stt.qwencleo import QwenCleoSTT


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class _RunningProcess:
    def poll(self) -> None:
        return None


def test_qwencleo_transcription_posts_a_temporary_wav(monkeypatch: MonkeyPatch) -> None:
    engine = QwenCleoSTT(Settings(stt_timeout_sec=5.0))
    engine._server_process = cast(subprocess.Popen[bytes], _RunningProcess())
    server_temp = tempfile.TemporaryDirectory(prefix="qwencleo-test-")
    engine._server_temp = server_temp
    captured: dict[str, object] = {}

    def fake_urlopen(request: Request, timeout: float) -> _FakeResponse:
        assert timeout > 0
        assert request.full_url.endswith("/v1/audio/transcriptions")
        assert request.data is not None
        payload = json.loads(request.data.decode("utf-8"))
        audio_path = Path(payload["audio"])
        assert audio_path.is_file()
        with wave.open(str(audio_path), "rb") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2
            assert wav_file.getframerate() == 16000
            assert wav_file.readframes(1600) == b"\x01\x00" * 1600
        captured.update(payload)
        return _FakeResponse(json.dumps({"text": "  أهلا <|endoftext|> "}).encode("utf-8"))

    monkeypatch.setattr("app.pipeline.stt.qwencleo.urlopen", fake_urlopen)
    try:
        result = asyncio.run(engine.transcribe(b"\x01\x00" * 1600))
        assert result == STTResult(text="أهلا", confidence=None)
        assert captured["model"] == "qwencleo-local"
        assert captured["language"] == "Arabic"
        assert not Path(str(captured["audio"])).exists()
    finally:
        engine._executor.shutdown(wait=True, cancel_futures=True)
        server_temp.cleanup()


def test_qwencleo_skips_audio_shorter_than_minimum_without_startup() -> None:
    engine = QwenCleoSTT(Settings())
    try:
        assert asyncio.run(engine.transcribe(b"\x00\x00" * 1599)) == STTResult(
            text="",
            confidence=None,
        )
    finally:
        engine._executor.shutdown(wait=True, cancel_futures=True)
