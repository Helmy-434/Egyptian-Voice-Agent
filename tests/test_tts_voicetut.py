import asyncio
import base64
import json
import threading
import time
from collections.abc import Coroutine, Iterator
from typing import Any

import pytest

from app.configs import Settings
from app.contracts import MAX_SPEAKABLE_CHARS
from app.pipeline.tts.base import TTSError
from app.pipeline.tts.voicetut import VoiceTutTTS


class FakeSSE:
    def __init__(self, pcm_parts: list[bytes], gate: threading.Event | None = None) -> None:
        self._lines = [
            b"data: "
            + json.dumps(
                {
                    "type": "speech.audio.delta",
                    "audio": base64.b64encode(part).decode(),
                }
            ).encode()
            + b"\n"
            for part in pcm_parts
        ] + [b'data: {"type": "speech.audio.done"}\n', b"data: [DONE]\n"]
        self._gate = gate

    def __enter__(self) -> "FakeSSE":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def __iter__(self) -> Iterator[bytes]:
        for index, line in enumerate(self._lines):
            if self._gate and index == 1:
                self._gate.wait(timeout=2)
            yield line

    def close(self) -> None:
        pass


def _ready_engine(**overrides: object) -> VoiceTutTTS:
    settings = Settings(_env_file=None, **overrides)
    engine = VoiceTutTTS(settings)
    engine._ready = True
    engine.sample_rate = 24000
    return engine


def _run_and_shutdown(
    engine: VoiceTutTTS,
    coroutine: Coroutine[Any, Any, None],
) -> None:
    try:
        asyncio.run(coroutine)
    finally:
        engine._executor.shutdown(wait=True, cancel_futures=True)


def test_streaming_preserves_pcm_and_emits_small_first_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pcm_parts = [b"\x01\x00" * 512, b"\x02\x00" * 512]
    monkeypatch.setattr(
        "app.pipeline.tts.voicetut.urlopen",
        lambda *args, **kwargs: FakeSSE(pcm_parts),
    )
    engine = _ready_engine()

    async def verify() -> None:
        chunks = [chunk async for chunk in engine.synthesize_stream("صباح الخير", "male")]

        assert chunks
        assert len(chunks[0].pcm) <= 1024
        assert all(chunk.sample_rate == 24000 for chunk in chunks)
        assert b"".join(chunk.pcm for chunk in chunks) == b"".join(pcm_parts)

    _run_and_shutdown(engine, verify())


@pytest.mark.parametrize("text", ["", "   \n\t", "؟!.,..."])
def test_unpronounceable_text_yields_nothing(text: str) -> None:
    engine = _ready_engine()

    async def verify() -> None:
        assert [chunk async for chunk in engine.synthesize_stream(text, "male")] == []

    _run_and_shutdown(engine, verify())


def test_overlong_text_and_unknown_voice_raise() -> None:
    engine = _ready_engine()

    async def verify() -> None:
        with pytest.raises(TTSError, match="MAX_SPEAKABLE_CHARS"):
            async for _ in engine.synthesize_stream(
                "أ" * (MAX_SPEAKABLE_CHARS + 1),
                "male",
            ):
                pass
        with pytest.raises(TTSError, match="Unknown voice"):
            async for _ in engine.synthesize_stream("أهلاً", "robot"):  # type: ignore[arg-type]
                pass

    _run_and_shutdown(engine, verify())


def test_two_sequential_healthy_calls_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    pcm = b"\x01\x00" * 512
    monkeypatch.setattr(
        "app.pipeline.tts.voicetut.urlopen",
        lambda *args, **kwargs: FakeSSE([pcm]),
    )
    engine = _ready_engine()

    async def verify() -> None:
        first = [chunk async for chunk in engine.synthesize_stream("الأولى", "male")]
        second = [chunk async for chunk in engine.synthesize_stream("الثانية", "female")]

        assert first and second
        assert b"".join(chunk.pcm for chunk in first) == pcm
        assert b"".join(chunk.pcm for chunk in second) == pcm

    _run_and_shutdown(engine, verify())


def test_abandoned_worker_fails_fast_then_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = threading.Event()
    pcm = b"\x01\x00" * 512
    responses = [FakeSSE([pcm, pcm], gate), FakeSSE([pcm])]
    monkeypatch.setattr(
        "app.pipeline.tts.voicetut.urlopen",
        lambda *args, **kwargs: responses.pop(0),
    )
    engine = _ready_engine()

    async def verify() -> None:
        stream = engine.synthesize_stream("جملة طويلة", "male")
        try:
            async for _ in stream:
                break
            await stream.aclose()

            with pytest.raises(TTSError, match="abandoned synthesis"):
                async for _ in engine.synthesize_stream("محاولة تانية", "female"):
                    pass
        finally:
            await stream.aclose()
            gate.set()

        deadline = time.monotonic() + 2
        while engine._worker_busy.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert not engine._worker_busy.is_set()

        chunks = [
            chunk async for chunk in engine.synthesize_stream("محاولة أخيرة", "female")
        ]
        assert chunks

    _run_and_shutdown(engine, verify())


def test_first_chunk_timeout_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = threading.Event()
    monkeypatch.setattr(
        "app.pipeline.tts.voicetut.urlopen",
        lambda *args, **kwargs: FakeSSE([b"", b"\x01\x00" * 512], gate),
    )
    engine = _ready_engine(tts_first_chunk_timeout_sec=0.05)

    async def verify() -> None:
        try:
            with pytest.raises(TTSError, match="first-chunk timeout"):
                async for _ in engine.synthesize_stream("جملة بطيئة", "male"):
                    pass
        finally:
            gate.set()

    _run_and_shutdown(engine, verify())
