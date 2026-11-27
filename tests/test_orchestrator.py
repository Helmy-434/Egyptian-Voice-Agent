import asyncio
from typing import AsyncGenerator

from app.configs import Settings
from app.pipeline.latency import LatencyTracker
from app.pipeline.orchestrator import VoiceSession
from app.pipeline.stt.base import STTError, STTResult
from app.pipeline.tts.base import AudioChunk
from dev.stub_agent import stream_agent_turn


class FakeSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    async def send_json(self, data: dict) -> None:
        self.events.append(("json", data))

    async def send_bytes(self, data: bytes) -> None:
        self.events.append(("bytes", len(data)))

    def json_events(self) -> list[dict]:
        return [d for k, d in self.events if k == "json"]  # type: ignore[misc]


class FakeSTT:
    def __init__(self, text: str = "أهلا", fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.gate: asyncio.Event | None = None

    async def transcribe(self, audio_pcm: bytes, sample_rate: int) -> STTResult:
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise STTError("boom")
        return STTResult(text=self.text, confidence=0.95)


class FakeTTS:
    sample_rate = 24000

    def __init__(self, chunks_per_call: int = 1) -> None:
        self.chunks_per_call = chunks_per_call
        self.voices_seen: list[str] = []
        self.gate: asyncio.Event | None = None
        self.calls = 0

    async def synthesize_stream(
        self, text: str, voice: str
    ) -> AsyncGenerator[AudioChunk, None]:
        self.calls += 1
        self.voices_seen.append(voice)
        if self.calls == 2 and self.gate is not None:
            await self.gate.wait()
        for _ in range(self.chunks_per_call):
            yield AudioChunk(pcm=b"\x00\x01" * 240, sample_rate=24000)


class FakeVAD:
    def prob(self, frame: bytes) -> float:
        return 1.0 if any(frame) else 0.0

    def reset(self) -> None:
        pass


def _settings(**kw) -> Settings:
    defaults = dict(
        _env_file=None,
        vad_min_speech_ms=64,
        vad_silence_duration_ms=96,
        vad_preroll_ms=0,
        barge_in_min_speech_ms=64,
    )
    defaults.update(kw)
    return Settings(**defaults)


def test_normal_turn_and_latency():
    async def run():
        sink = FakeSink()
        stt = FakeSTT("أهلا")
        tts = FakeTTS()
        lat = LatencyTracker(None)
        session = VoiceSession(
            session_id="s1",
            settings=_settings(),
            stt=stt,
            tts=tts,
            stream_turn=stream_agent_turn,
            vad=FakeVAD(),
            sink=sink,
            latency=lat,
            voice="female",
        )
        await session.start()

        # Auto-ack chunks as they end
        async def auto_ack():
            seen = set()
            while True:
                for d in sink.json_events():
                    if d.get("type") == "tts_chunk_end" and d["chunk_id"] not in seen:
                        seen.add(d["chunk_id"])
                        await session.handle_message({"type": "played", "chunk_id": d["chunk_id"]})
                await asyncio.sleep(0.01)

        acker = asyncio.create_task(auto_ack())
        timer = lat.new_turn("s1", 1, 96)
        timer.mark("speech_end")
        await session._run_turn(b"\x01\x00" * 512, timer)
        acker.cancel()
        await asyncio.gather(acker, return_exceptions=True)

        types = [d["type"] for d in sink.json_events()]
        assert "transcript" in types
        assert "agent_text" in types
        assert "tts_chunk_start" in types
        assert "tts_chunk_end" in types
        assert lat.summary()["count"] == 1

    asyncio.run(run())


def test_empty_stt_dropped_silently():
    async def run():
        sink = FakeSink()
        lat = LatencyTracker(None)
        called = False

        async def spy_agent(req):
            nonlocal called
            called = True
            yield

        session = VoiceSession(
            session_id="s1",
            settings=_settings(),
            stt=FakeSTT("   "),
            tts=FakeTTS(),
            stream_turn=spy_agent,
            vad=FakeVAD(),
            sink=sink,
            latency=lat,
            voice="female",
        )
        await session.start()
        timer = lat.new_turn("s1", 1, 96)
        timer.mark("speech_end")
        await session._run_turn(b"\x01\x00" * 512, timer)

        assert not called
        assert lat.summary()["count"] == 0
        assert all(d.get("type") != "transcript" for d in sink.json_events())
        assert sink.json_events()[-1] == {"type": "state", "state": "listening"}

    asyncio.run(run())


def test_stt_failure_speaks_apology():
    async def run():
        sink = FakeSink()
        lat = LatencyTracker(None)
        session = VoiceSession(
            session_id="s1",
            settings=_settings(),
            stt=FakeSTT(fail=True),
            tts=FakeTTS(),
            stream_turn=stream_agent_turn,
            vad=FakeVAD(),
            sink=sink,
            latency=lat,
            voice="female",
        )
        session._playback_ack_timeout_pad = 0.05
        await session.start()
        timer = lat.new_turn("s1", 1, 96)
        timer.mark("speech_end")
        await session._run_turn(b"\x01\x00" * 512, timer)

        errors = [d for d in sink.json_events() if d.get("type") == "error"]
        assert errors and errors[0]["code"] == "stt_failed"
        starts = [d for d in sink.json_events() if d.get("type") == "tts_chunk_start"]
        assert starts and "ممكن تعيد" in starts[0]["text"]

    asyncio.run(run())


def test_barge_in_mid_tts_tracks_truncated_text():
    async def run():
        sink = FakeSink()
        tts = FakeTTS()
        tts.gate = asyncio.Event()
        lat = LatencyTracker(None)

        seen_requests = []

        async def two_chunk_agent(req):
            seen_requests.append(req)
            from app.contracts import AgentChunk
            yield AgentChunk(type="text_delta", content="أول جملة.")
            yield AgentChunk(type="text_delta", content="تاني جملة.")
            yield AgentChunk(type="done")

        session = VoiceSession(
            session_id="s1",
            settings=_settings(),
            stt=FakeSTT("سؤال"),
            tts=tts,
            stream_turn=two_chunk_agent,
            vad=FakeVAD(),
            sink=sink,
            latency=lat,
            voice="female",
        )
        await session.start()
        timer = lat.new_turn("s1", 1, 96)
        timer.mark("speech_end")
        await session._set_state("thinking")
        session._turn_task = asyncio.create_task(session._run_turn(b"\x01\x00" * 512, timer))

        # Wait for chunk 0 to finish and chunk 1 to block on gate
        for _ in range(50):
            ends = [d for d in sink.json_events() if d.get("type") == "tts_chunk_end"]
            if ends:
                await session.handle_message({"type": "played", "chunk_id": ends[0]["chunk_id"]})
                break
            await asyncio.sleep(0.01)

        await asyncio.sleep(0.02)
        await session._barge_in()
        assert any(d.get("type") == "barge_in" for d in sink.json_events())

        # Next turn receives truncated_assistant_text == "أول جملة."
        tts.gate.set()
        session._playback_ack_timeout_pad = 0.05
        t2 = lat.new_turn("s1", 2, 96)
        t2.mark("speech_end")
        await session._run_turn(b"\x01\x00" * 512, t2)
        assert seen_requests[1].truncated_assistant_text == "أول جملة."

        # Following turn with no barge-in receives None
        t3 = lat.new_turn("s1", 3, 96)
        t3.mark("speech_end")
        await session._run_turn(b"\x01\x00" * 512, t3)
        assert seen_requests[2].truncated_assistant_text is None

    asyncio.run(run())


def test_barge_in_during_thinking_passes_empty_truncated():
    async def run():
        sink = FakeSink()
        stt = FakeSTT("أهلا")
        stt.gate = asyncio.Event()
        lat = LatencyTracker(None)
        seen_requests = []

        async def rec_agent(req):
            seen_requests.append(req)
            from app.contracts import AgentChunk
            yield AgentChunk(type="done")

        session = VoiceSession(
            session_id="s1",
            settings=_settings(),
            stt=stt,
            tts=FakeTTS(),
            stream_turn=rec_agent,
            vad=FakeVAD(),
            sink=sink,
            latency=lat,
            voice="female",
        )
        await session.start()
        t1 = lat.new_turn("s1", 1, 96)
        t1.mark("speech_end")
        await session._set_state("thinking")
        session._turn_task = asyncio.create_task(session._run_turn(b"\x01\x00" * 512, t1))
        await asyncio.sleep(0.02)
        await session._barge_in()

        stt.gate.set()
        t2 = lat.new_turn("s1", 2, 96)
        t2.mark("speech_end")
        await session._run_turn(b"\x01\x00" * 512, t2)
        assert seen_requests[0].truncated_assistant_text == ""

    asyncio.run(run())


def test_voice_change_and_unknown_ack_ignored():
    async def run():
        sink = FakeSink()
        tts = FakeTTS()
        lat = LatencyTracker(None)
        session = VoiceSession(
            session_id="s1",
            settings=_settings(),
            stt=FakeSTT("أهلا"),
            tts=tts,
            stream_turn=stream_agent_turn,
            vad=FakeVAD(),
            sink=sink,
            latency=lat,
            voice="female",
        )
        session._playback_ack_timeout_pad = 0.05
        await session.start()
        await session.handle_message({"type": "played", "chunk_id": 9999})
        await session.handle_message({"type": "voice", "voice": "male"})

        t = lat.new_turn("s1", 1, 96)
        t.mark("speech_end")
        await session._run_turn(b"\x01\x00" * 512, t)
        assert tts.voices_seen and all(v == "male" for v in tts.voices_seen)

    asyncio.run(run())
