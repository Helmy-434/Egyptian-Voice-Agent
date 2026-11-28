import asyncio
import contextlib
import json
import logging
from uuid import uuid4
from app.agent.state import mark_confirmation_played
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.pipeline.orchestrator import ClientSink, VoiceSession

logger = logging.getLogger(__name__)
router = APIRouter()


class WebSocketSink(ClientSink):
    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket

    async def send_json(self, data: dict) -> None:
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await self._ws.send_json(data)

    async def send_bytes(self, data: bytes) -> None:
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            await self._ws.send_bytes(data)


@router.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket) -> None:
    await websocket.accept()
    state = websocket.app.state
    settings = state.settings
    session_lock: asyncio.Lock = state.session_lock

    if session_lock.locked():
        with contextlib.suppress(Exception):
            await websocket.send_json(
                {"type": "error", "code": "busy", "message": "Another voice session is active."}
            )
            await websocket.close(code=1013)
        return

    async with session_lock:
        voice = settings.default_voice
        email: str | None = None
        try:
            start_msg = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
            if isinstance(start_msg, dict) and start_msg.get("type") == "start":
                v = start_msg.get("voice")
                if v in ("male", "female"):
                    voice = v
                e = start_msg.get("email")
                if isinstance(e, str) and e.strip():
                    email = e.strip()
        except (asyncio.TimeoutError, WebSocketDisconnect, ValueError, TypeError):
            pass

        session_id = uuid4().hex
        sink = WebSocketSink(websocket)
        session = VoiceSession(
            session_id=session_id,
            settings=settings,
            stt=state.stt,
            tts=state.tts,
            stream_turn=state.stream_turn,
            vad=state.vad,
            sink=sink,
            latency=state.latency,
            voice=voice,
            email=email,
        )

        frame_bytes = settings.audio_sample_rate * settings.audio_frame_ms // 1000 * 2
        await sink.send_json(
            {
                "type": "ready",
                "session_id": session_id,
                "capture_rate": settings.audio_sample_rate,
                "frame_bytes": frame_bytes,
                "voice": voice,
            }
        )
        await session.start()

        async def _watchdog() -> None:
            await asyncio.sleep(settings.max_session_duration_sec)
            await sink.send_json(
                {
                    "type": "error",
                    "code": "session_timeout",
                    "message": "Maximum session duration reached.",
                }
            )
            with contextlib.suppress(Exception):
                await websocket.close(code=1000)

        watchdog_task = asyncio.create_task(_watchdog())
        try:
            while True:
                msg = await websocket.receive()

                if msg.get("type") == "websocket.disconnect":
                    break

                if "bytes" in msg and msg["bytes"] is not None:
                    await session.feed_audio(msg["bytes"])

                elif "text" in msg and msg["text"] is not None:
                    try:
                        data = json.loads(msg["text"])
                    except ValueError:
                        logger.debug(
                            "Malformed JSON from client: %r",
                            msg["text"],
                        )
                        continue

                    if isinstance(data, dict):
                        if data.get("type") == "stop":
                            break

                        elif data.get("type") == "playback_complete":
                            action_id = data.get("action_id")

                            if action_id:
                                played = mark_confirmation_played(
                                    session_id,
                                    action_id,
                                )

                                logger.info(
                                    "Confirmation playback acknowledgement: "
                                    "session=%s action=%s played=%s",
                                    session_id,
                                    action_id,
                                    played,
                                )

                        else:
                            await session.handle_message(data)

        except WebSocketDisconnect:
            pass
        finally:
            watchdog_task.cancel()
