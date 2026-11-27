from pathlib import Path
from typing import AsyncGenerator

from starlette.testclient import TestClient

from app.configs import Settings
from app.main import create_app
from app.pipeline.latency import LatencyTracker
from app.pipeline.stt.base import STTResult
from app.pipeline.tts.base import AudioChunk
from dev.stub_agent import stream_agent_turn


class FakeSTT:
    device = "cpu"

    async def transcribe(self, audio_pcm: bytes, sample_rate: int) -> STTResult:
        return STTResult(text="أهلا", confidence=0.95)


class FakeTTS:
    device = "cpu"
    sample_rate = 24000

    def __init__(self) -> None:
        self.voices_seen: list[str] = []

    async def synthesize_stream(
        self, text: str, voice: str
    ) -> AsyncGenerator[AudioChunk, None]:
        self.voices_seen.append(voice)
        yield AudioChunk(pcm=b"\x00\x01" * 240, sample_rate=24000)


class FakeVAD:
    def prob(self, frame: bytes) -> float:
        return 1.0 if any(frame) else 0.0

    def reset(self) -> None:
        pass


def test_ws_voice_flow_and_busy_and_health(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        vad_min_speech_ms=64,
        vad_silence_duration_ms=96,
        vad_preroll_ms=0,
        barge_in_min_speech_ms=64,
    )
    tts = FakeTTS()
    closed_sessions: list[str] = []

    async def close_fn(sid: str) -> None:
        closed_sessions.append(sid)

    app = create_app(
        settings=settings,
        stt=FakeSTT(),
        tts=tts,
        stream_turn=stream_agent_turn,
        close_session_fn=close_fn,
        vad=FakeVAD(),
        latency=LatencyTracker(tmp_path / "lat.jsonl"),
    )

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
        assert client.get("/metrics").status_code == 200

        with client.websocket_connect("/ws/voice") as ws:
            ws.send_json({"type": "start", "voice": "male", "email": "u@example.com"})
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            assert ready["voice"] == "male"
            sid = ready["session_id"]

            # Second concurrent connection gets busy
            with client.websocket_connect("/ws/voice") as ws2:
                busy = ws2.receive_json()
                assert busy["type"] == "error"
                assert busy["code"] == "busy"

            speech_frame = b"\x01\x00" * 512
            silence_frame = b"\x00\x00" * 512
            for _ in range(6):
                ws.send_bytes(speech_frame)
            for _ in range(5):
                ws.send_bytes(silence_frame)

            seen_types: list[str] = []
            while True:
                msg = ws.receive()
                if "text" in msg and msg["text"] is not None:
                    import json
                    data = json.loads(msg["text"])
                    t = data.get("type")
                    seen_types.append(t)
                    if t == "tts_chunk_end":
                        ws.send_json({"type": "played", "chunk_id": data["chunk_id"]})
                        break
                elif "bytes" in msg and msg["bytes"] is not None:
                    seen_types.append("bytes")

            for expected in ("transcript", "agent_text", "tts_chunk_start", "bytes", "tts_chunk_end"):
                assert expected in seen_types
            assert tts.voices_seen and all(v == "male" for v in tts.voices_seen)

            ws.send_json({"type": "stop"})

        assert sid in closed_sessions
