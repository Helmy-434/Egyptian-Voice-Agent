"""Unit tests for the TTS contract and local VoiceTut audio.cpp driver."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from app.configs import Settings
from app.contracts import MAX_SPEAKABLE_CHARS as CONTRACT_MAX_SPEAKABLE_CHARS
from app.pipeline.tts.base import (
    MAX_SPEAKABLE_CHARS,
    AudioChunk,
    TTSEngine,
    TTSError,
    Voice,
    create_tts_engine,
)
from app.pipeline.tts import voicetut
from app.pipeline.tts.voicetut import VoiceTutTTS


class _FakeTTSEngine:
    sample_rate = 24000

    async def warmup(self) -> None:
        return None

    async def synthesize_stream(self, text: str, voice: Voice) -> AsyncIterator[AudioChunk]:
        if text.strip():
            yield AudioChunk(pcm=b"\x01\x00" * 100, sample_rate=self.sample_rate)

    async def close(self) -> None:
        return None


def _make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "_env_file": None,
        "tts_provider": "voicetut",
        "voicetut_male_voice": "mohammed",
        "voicetut_female_voice": "yasmin",
        "voicetut_allow_cpu": True,
        "tts_first_chunk_timeout_sec": 0.5,
        "tts_total_timeout_sec": 1.0,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


def _ready_engine(settings: Settings | None = None) -> VoiceTutTTS:
    engine = VoiceTutTTS(settings or _make_settings())
    engine._ready = True
    engine.sample_rate = 24000
    return engine


def test_fake_engine_satisfies_protocol_and_contract_cap() -> None:
    assert isinstance(_FakeTTSEngine(), TTSEngine)
    assert MAX_SPEAKABLE_CHARS == CONTRACT_MAX_SPEAKABLE_CHARS


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported tts_provider"):
        create_tts_engine(SimpleNamespace(tts_provider="invalid_provider"))  # type: ignore[arg-type]


def test_input_validation_happens_before_server_work() -> None:
    engine = VoiceTutTTS(_make_settings())

    async def _run() -> None:
        for empty_input in ("", "   \n\t  ", "؟!.,..."):
            assert [chunk async for chunk in engine.synthesize_stream(empty_input, "male")] == []

        with pytest.raises(TTSError, match="MAX_SPEAKABLE_CHARS"):
            async for _ in engine.synthesize_stream("أ" * (MAX_SPEAKABLE_CHARS + 1), "male"):
                pass

        with pytest.raises(TTSError, match="Unknown voice"):
            async for _ in engine.synthesize_stream("أهلاً", "robot"):  # type: ignore[arg-type]
                pass

        await engine.close()

    asyncio.run(_run())


def test_sse_audio_deltas_are_decoded(monkeypatch: pytest.MonkeyPatch) -> None:
    pcm = b"\x01\x00\x02\x00"
    events = [
        {
            "type": "speech.audio.delta",
            "audio": base64.b64encode(pcm).decode("ascii"),
        },
        {"type": "speech.audio.done"},
    ]
    response = io.BytesIO(
        "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode("utf-8")
    )
    calls: list[object] = []

    def fake_urlopen(request: object, timeout: float):
        calls.append((request, timeout))
        return response

    monkeypatch.setattr(voicetut, "urlopen", fake_urlopen)
    engine = VoiceTutTTS(_make_settings())

    assert list(engine._read_pcm_sync("أهلاً", "mohammed", None, None, None)) == [pcm]
    request, timeout = calls[0]  # type: ignore[misc]
    assert timeout == 1.0
    assert request.full_url == "http://127.0.0.1:18762/v1/audio/speech"
    assert request.get_header("Content-type") == "application/json"
    assert json.loads(request.data.decode("utf-8")) == {
        "model": "voicetut-local",
        "input": "أهلاً",
        "voice": "mohammed",
        "language": "arz",
        "response_format": "pcm",
        "stream_format": "sse",
    }


def test_streaming_chunks_preserve_pcm_and_sample_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    pcm = bytes(range(256)) * 32
    engine = _ready_engine()
    speakers: list[str] = []

    def fake_read_pcm(text, speaker, cancel, response_holder, response_lock):
        speakers.append(speaker)
        for offset in range(0, len(pcm), 1024):
            yield pcm[offset : offset + 1024]

    monkeypatch.setattr(engine, "_read_pcm_sync", fake_read_pcm)

    async def _run() -> None:
        chunks = [chunk async for chunk in engine.synthesize_stream("صباح الخير", "female")]
        assert b"".join(chunk.pcm for chunk in chunks) == pcm
        assert len(chunks) == 2
        assert len(chunks[0].pcm) <= 1024
        assert all(chunk.sample_rate == 24000 for chunk in chunks)
        assert speakers == ["yasmin"]
        await engine.close()

    asyncio.run(_run())


def test_warmup_raises_when_speaker_wav_or_transcript_missing(tmp_path: Path) -> None:
    model_path = tmp_path / "voicetut.gguf"
    server_path = tmp_path / "audiocpp_server.exe"
    model_path.write_bytes(b"gguf")
    server_path.write_bytes(b"exe")
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir()
    (voice_dir / "mohammed.wav").write_bytes(b"RIFF")
    (voice_dir / "prompt_text").write_text("mohammed|مرحبا\n", encoding="utf-8")

    engine = VoiceTutTTS(
        _make_settings(
            voicetut_model_path=str(model_path),
            voicetut_server_path=str(server_path),
            voicetut_voice_dir=str(voice_dir),
        )
    )
    with pytest.raises(TTSError, match="missing yasmin.wav"):
        asyncio.run(engine.warmup())

    (voice_dir / "yasmin.wav").write_bytes(b"RIFF")
    with pytest.raises(TTSError, match="missing a transcript"):
        asyncio.run(engine.warmup())


def test_server_config_uses_streaming_gguf_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "voicetut.gguf"
    server_path = tmp_path / "audiocpp_server.exe"
    model_path.write_bytes(b"gguf")
    server_path.write_bytes(b"server")

    class FakeProcess:
        returncode: int | None = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout: float | None = None):
            return self.returncode

    process = FakeProcess()
    monkeypatch.setattr(voicetut, "has_cuda_driver", lambda: True)
    monkeypatch.setattr(
        voicetut.subprocess,
        "Popen",
        lambda *args, **kwargs: process,
    )
    monkeypatch.setattr(voicetut, "urlopen", lambda *args, **kwargs: io.BytesIO(b"ok"))
    engine = VoiceTutTTS(
        _make_settings(
            voicetut_model_path=str(model_path),
            voicetut_server_path=str(server_path),
        )
    )

    def fake_read_pcm(text, speaker, cancel, response_holder, response_lock):
        yield b"\x00\x00"

    monkeypatch.setattr(engine, "_read_pcm_sync", fake_read_pcm)

    async def _run() -> None:
        await engine.warmup()
        assert engine.sample_rate == 24000
        assert engine.device == "cuda:0"
        assert engine._server_temp is not None
        config_path = Path(engine._server_temp.name) / "server.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["port"] == 18762
        assert config["models"] == [
            {
                "id": "voicetut-local",
                "family": "omnivoice",
                "path": str(model_path),
                "task": "tts",
                "mode": "streaming",
            }
        ]
        await engine.close()

    asyncio.run(_run())


def test_abandoned_worker_fails_fast_then_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    release_worker = threading.Event()
    worker_started = threading.Event()
    engine = _ready_engine(
        _make_settings(
            tts_first_chunk_timeout_sec=0.05,
            tts_total_timeout_sec=0.1,
        )
    )

    def fake_read_pcm(text, speaker, cancel, response_holder, response_lock):
        if text == "جملة بطيئة":
            worker_started.set()
            release_worker.wait(timeout=2.0)
        yield b"\x00\x00" * 10

    monkeypatch.setattr(engine, "_read_pcm_sync", fake_read_pcm)

    async def _run() -> None:
        with pytest.raises(TTSError, match="timeout"):
            async for _ in engine.synthesize_stream("جملة بطيئة", "male"):
                pass
        assert worker_started.is_set()
        abandoned_job = engine._abandoned_job
        assert abandoned_job is not None and not abandoned_job.is_set()

        with pytest.raises(TTSError, match="abandoned"):
            async for _ in engine.synthesize_stream("جملة تانية", "female"):
                pass

        release_worker.set()
        for _ in range(50):
            if abandoned_job.is_set() and not engine._worker_busy.is_set():
                break
            await asyncio.sleep(0.01)

        chunks = [chunk async for chunk in engine.synthesize_stream("جملة سريعة", "male")]
        assert chunks
        await engine.close()

    asyncio.run(_run())


@pytest.mark.skipif(
    not Path("QwenCleo/models/voicetut-tts-q8_0.gguf").is_file(),
    reason="Local VoiceTut GGUF unavailable",
)
@pytest.mark.live
def test_voicetut_local_gpu_integration() -> None:
    if not voicetut.has_cuda_driver():
        pytest.skip("CUDA is unavailable for local GPU integration")

    async def _run() -> None:
        engine = create_tts_engine(
            _make_settings(
                voicetut_allow_cpu=False,
                tts_first_chunk_timeout_sec=3.0,
                tts_total_timeout_sec=30.0,
            )
        )
        await engine.warmup()
        try:
            for voice in ("male", "female"):
                chunks = [
                    chunk
                    async for chunk in engine.synthesize_stream(
                        "أهلاً بيك، الأوردر بتاعك جاهز للتوصيل.",
                        voice,  # type: ignore[arg-type]
                    )
                ]
                assert chunks
                assert all(chunk.sample_rate == engine.sample_rate for chunk in chunks)
        finally:
            await engine.close()

    asyncio.run(_run())
