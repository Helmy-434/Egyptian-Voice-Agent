import asyncio
import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import cast
from urllib.error import URLError
from urllib.request import Request
import wave

import pytest
from pytest import MonkeyPatch

<<<<<<< Updated upstream
from app.config import Settings
from app.pipeline.stt.base import STTResult
=======
from app.configs import Settings
from app.pipeline.stt.base import STTError, STTResult
>>>>>>> Stashed changes
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
    engine = QwenCleoSTT(Settings(_env_file=None, llm_provider="groq", stt_timeout_sec=5.0))
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
    engine = QwenCleoSTT(Settings(_env_file=None, llm_provider="groq"))
    try:
        assert asyncio.run(engine.transcribe(b"\x00\x00" * 1599)) == STTResult(
            text="",
            confidence=None,
        )
    finally:
        engine._executor.shutdown(wait=True, cancel_futures=True)


def test_qwencleo_timeout_is_fail_fast_while_worker_remains_busy(monkeypatch: MonkeyPatch) -> None:
    engine = QwenCleoSTT(
        Settings(
            _env_file=None,
            llm_provider="groq",
            stt_timeout_sec=0.05,
            stt_timeout_per_audio_sec=0,
        )
    )
    engine._server_process = cast(subprocess.Popen[bytes], _RunningProcess())
    server_temp = tempfile.TemporaryDirectory(prefix="qwencleo-timeout-test-")
    engine._server_temp = server_temp
    request_started = threading.Event()
    release_request = threading.Event()

    def hanging_urlopen(request: Request, timeout: float) -> _FakeResponse:
        request_started.set()
        release_request.wait(timeout=2)
        return _FakeResponse(b'{"text":"done"}')

    monkeypatch.setattr("app.pipeline.stt.qwencleo.urlopen", hanging_urlopen)

    async def verify_timeout_and_busy_guard() -> None:
        with pytest.raises(STTError, match="timed out"):
            await engine.transcribe(b"\x01\x00" * 1600)

        assert request_started.is_set()
        assert engine._worker_busy.is_set()
        with pytest.raises(STTError, match="still busy"):
            await engine.transcribe(b"\x01\x00" * 1600)

        release_request.set()
        deadline = time.monotonic() + 1
        while engine._worker_busy.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert not engine._worker_busy.is_set()

    try:
        asyncio.run(verify_timeout_and_busy_guard())
    finally:
        release_request.set()
        engine._executor.shutdown(wait=True, cancel_futures=True)
        server_temp.cleanup()


def test_qwencleo_server_crash_is_reported_as_stt_error(monkeypatch: MonkeyPatch) -> None:
    engine = QwenCleoSTT(Settings(_env_file=None, llm_provider="groq", stt_timeout_sec=1.0))
    engine._server_process = cast(subprocess.Popen[bytes], _RunningProcess())
    server_temp = tempfile.TemporaryDirectory(prefix="qwencleo-crash-test-")
    engine._server_temp = server_temp

    def crashed_server(request: Request, timeout: float) -> _FakeResponse:
        raise URLError("server process exited")

    monkeypatch.setattr("app.pipeline.stt.qwencleo.urlopen", crashed_server)
    try:
        with pytest.raises(STTError, match="Could not reach local QwenCleo-ASR server"):
            asyncio.run(engine.transcribe(b"\x01\x00" * 1600))
    finally:
        engine._executor.shutdown(wait=True, cancel_futures=True)
        server_temp.cleanup()
