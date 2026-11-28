import asyncio
import contextlib
import logging
from typing import AsyncGenerator, Callable, Literal, Protocol
from app.agent.chunker import chunk_text, SENTENCE_BOUNDARY
from app.agent.state import (
    set_confirmation_playback,
    invalidate_confirmation_playback,
)
from app.configs import Settings
from app.contracts import AgentChunk, AgentRequest, SessionContext
from app.pipeline.latency import LatencyTracker, TurnTimer
from app.pipeline.stt.base import STTEngine, STTError
from app.pipeline.tts.base import TTSEngine, TTSError
from app.pipeline.vad import VADModel, VADStateMachine

logger = logging.getLogger(__name__)


class ClientSink(Protocol):
    async def send_json(self, data: dict) -> None: ...
    async def send_bytes(self, data: bytes) -> None: ...


class VoiceSession:
    def __init__(
        self,
        *,
        session_id: str,
        settings: Settings,
        stt: STTEngine,
        tts: TTSEngine,
        stream_turn: Callable[[AgentRequest], AsyncGenerator[AgentChunk, None]],
        vad: VADModel,
        sink: ClientSink,
        latency: LatencyTracker,
        voice: Literal["male", "female"],
        email: str | None = None,
    ) -> None:

        self._confirmation_action_id: str | None = None

        self.session_id = session_id
        self._settings = settings
        self._stt = stt
        self._tts = tts
        self._stream_turn = stream_turn
        self._vad = vad
        self._sink = sink
        self._latency = latency
        self._voice: Literal["male", "female"] = voice
        self._email = email

        self._sm = VADStateMachine(
            frame_ms=settings.audio_frame_ms,
            threshold=settings.vad_speech_threshold,
            hysteresis=settings.vad_hysteresis,
            min_speech_ms=settings.vad_min_speech_ms,
            silence_ms=settings.vad_silence_duration_ms,
            preroll_ms=settings.vad_preroll_ms,
            max_utterance_sec=settings.max_utterance_sec,
        )

        self._frame_bytes = (
            settings.audio_sample_rate
            * settings.audio_frame_ms
            // 1000
            * 2
        )
        self._rx_buf = bytearray()

        self._state: str = ""
        self._turn_counter: int = 0
        self._turn_task: asyncio.Task[None] | None = None
        self._first_turn: bool = True
        self._pending_truncated: str | None = None

        self._next_chunk_id: int = 0

        self._vad.reset()

    async def start(self) -> None:
        await self._set_state("listening")

    async def _set_state(self, new_state: str) -> None:
        if new_state != self._state:
            self._state = new_state
            await self._sink.send_json(
                {"type": "state", "state": new_state}
            )

    def _reset_turn_playback_state(self) -> None:
        self._confirmation_action_id = None

    async def feed_audio(self, data: bytes) -> None:
        self._rx_buf.extend(data)

        while len(self._rx_buf) >= self._frame_bytes:
            frame = bytes(self._rx_buf[:self._frame_bytes])
            del self._rx_buf[:self._frame_bytes]

            if self._state in ("thinking", "speaking"):
                self._sm.min_speech_ms = (
                    self._settings.barge_in_min_speech_ms
                )
            else:
                self._sm.min_speech_ms = (
                    self._settings.vad_min_speech_ms
                )

            prob = self._vad.prob(frame)
            event = self._sm.update(frame, prob)

            if event is None:
                continue

            if event.kind == "speech_start":
                if self._state in ("thinking", "speaking"):
                    await self._barge_in()

                await self._set_state("hearing")

            elif event.kind == "speech_end":
                self._turn_counter += 1

                timer = self._latency.new_turn(
                    self.session_id,
                    self._turn_counter,
                    self._settings.vad_silence_duration_ms,
                )

                timer.mark("speech_end")
                await self._set_state("thinking")

                self._turn_task = asyncio.create_task(
                    self._run_turn(event.audio, timer)
                )

    async def handle_message(self, msg: dict) -> None:
        if msg.get("type") == "voice":
            voice = msg.get("voice")

            if voice in ("male", "female"):
                self._voice = voice

    async def _barge_in(self) -> None:
        action_id = self._confirmation_action_id

        if action_id:
            invalidate_confirmation_playback(
                self.session_id,
                action_id,
            )

        task = self._turn_task

        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

            await self._sink.send_json({"type": "barge_in"})

            self._pending_truncated = None
            self._reset_turn_playback_state()

    async def _run_turn(
        self,
        pcm: bytes,
        timer: TurnTimer,
    ) -> None:
        self._reset_turn_playback_state()

        text_q: asyncio.Queue[
            tuple[str, str | None] | None
        ] = asyncio.Queue()

        text_buffer = ""
        buffer_action_id: str | None = None

        confirmation_text = ""
        confirmation_action_id: str | None = None

        async def flush_text_buffer() -> None:
            nonlocal text_buffer, buffer_action_id

            if not text_buffer.strip():
                text_buffer = ""
                buffer_action_id = None
                return

            for text_part in chunk_text(text_buffer):
                await text_q.put((text_part, buffer_action_id))

            text_buffer = ""
            buffer_action_id = None

        async def add_text_delta(
            content: str,
            action_id: str | None,
        ) -> None:
            nonlocal text_buffer, buffer_action_id

            # Never combine normal speech with confirmation speech.
            if text_buffer and action_id != buffer_action_id:
                await flush_text_buffer()

            if not text_buffer:
                buffer_action_id = action_id

            text_buffer += content

            # Send complete sentences without waiting for the full response.
            sentences = SENTENCE_BOUNDARY.split(text_buffer)

            if len(sentences) > 1:
                complete_text = " ".join(sentences[:-1])
                text_buffer = sentences[-1]

                for text_part in chunk_text(complete_text):
                    await text_q.put((text_part, buffer_action_id))

        tts_task = asyncio.create_task(
            self._tts_worker(text_q, timer)
        )

        agen: AsyncGenerator[AgentChunk, None] | None = None
        cancelled_by_barge_in = False

        try:
            result = await self._stt.transcribe(
                pcm,
                self._settings.audio_sample_rate,
            )

            timer.mark("stt_done")

            text = result.text.strip()

            if not text:
                timer.cancelled = True
                return

            await self._sink.send_json(
                {"type": "transcript", "text": text}
            )

            ctx = (
                SessionContext(user_email=self._email)
                if self._first_turn and self._email
                else None
            )

            request = AgentRequest(
                session_id=self.session_id,
                user_text=text,
                stt_confidence=result.confidence,
                truncated_assistant_text=self._pending_truncated,
                context=ctx,
            )

            self._pending_truncated = None
            self._first_turn = False

            agen = self._stream_turn(request)

            async for chunk in agen:

                if chunk.type == "text_delta":
                    action_id = (chunk.metadata or {}).get(
                        "confirmation_action_id"
                    )

                    if action_id:
                        confirmation_action_id = action_id
                        confirmation_text += chunk.content

                    await add_text_delta(
                        chunk.content,
                        action_id,
                    )

                elif chunk.type == "confirm":
                    action_id = (chunk.metadata or {}).get("action_id")

                    if not action_id:
                        raise ValueError(
                            "Confirmation event is missing action_id"
                        )

                    if confirmation_action_id != action_id:
                        raise ValueError(
                            "Confirmation action ID does not match "
                            "the read-back text"
                        )

                    if not confirmation_text.strip():
                        raise ValueError(
                            "Confirmation read-back text is empty"
                        )

                    # Queue all remaining confirmation text first.
                    await flush_text_buffer()

                    # Register the exact text that must be played.
                    set_confirmation_playback(
                        self.session_id,
                        action_id,
                        confirmation_text,
                    )

                    self._confirmation_action_id = action_id

                    await self._sink.send_json(
                        {
                            "type": "agent_event",
                            "event": "confirm",
                            "content": chunk.content,
                            "metadata": chunk.metadata,
                        }
                    )

                elif chunk.type == "done":
                    await flush_text_buffer()
                    timer.agent_timing = chunk.timing
                    break

                else:
                    await self._sink.send_json(
                        {
                            "type": "agent_event",
                            "event": chunk.type,
                            "content": chunk.content,
                            "metadata": chunk.metadata,
                        }
                    )

            # Finish TTS generation and sending.
            await text_q.put(None)
            await tts_task

            self._latency.record(timer)

        except asyncio.CancelledError:
            timer.cancelled = True
            cancelled_by_barge_in = True
            raise

        except STTError as exc:
            timer.cancelled = True

            await self._sink.send_json(
                {
                    "type": "error",
                    "code": "stt_failed",
                    "message": str(exc),
                }
            )

            with contextlib.suppress(Exception):
                await text_q.put(("ممكن تعيد كلامك تاني؟", None))
                await text_q.put(None)
                await tts_task

        except Exception as exc:
            timer.cancelled = True

            logger.exception("Agent turn failed: %s", exc)

            await self._sink.send_json(
                {
                    "type": "error",
                    "code": "agent_failed",
                    "message": str(exc),
                }
            )

            with contextlib.suppress(Exception):
                await text_q.put(("آسف، حصلت مشكلة", None))
                await text_q.put(None)
                await tts_task

        finally:
            if not tts_task.done():
                tts_task.cancel()

            await asyncio.gather(
                tts_task,
                return_exceptions=True,
            )

            if agen is not None:
                with contextlib.suppress(Exception):
                    await agen.aclose()

            if not cancelled_by_barge_in:
                await self._set_state("listening")

    async def _tts_worker(
        self,
        text_q: asyncio.Queue[tuple[str, str | None] | None],
        timer: TurnTimer,
    ) -> None:
        first_chunk_of_turn = True
        first_audio_of_turn = True

        while True:
            item = await text_q.get()

            if item is None:
                break

            text, action_id = item

            chunk_id = self._next_chunk_id
            self._next_chunk_id += 1

            voice = self._voice
            announced_rate = self._tts.sample_rate

            await self._sink.send_json(
                {
                    "type": "tts_chunk_start",
                    "chunk_id": chunk_id,
                    "action_id": action_id,
                    "text": text,
                    "sample_rate": announced_rate,
                }
            )

            if first_chunk_of_turn:
                first_chunk_of_turn = False
                await self._set_state("speaking")

            synthesis_succeeded = False

            for attempt in range(2):
                try:
                    async for audio in self._tts.synthesize_stream(
                        text,
                        voice,
                    ):
                        if audio.sample_rate != announced_rate:
                            raise TTSError(
                                f"TTS sample rate mismatch: "
                                f"expected {announced_rate}, "
                                f"received {audio.sample_rate}"
                            )

                        if first_audio_of_turn:
                            first_audio_of_turn = False
                            timer.mark("tts_first_audio")

                        await self._sink.send_bytes(audio.pcm)

                    synthesis_succeeded = True
                    break

                except TTSError:
                    if attempt == 0:
                        await asyncio.sleep(0.3)
                        continue

                    logger.exception("TTS synthesis failed")

                    await self._sink.send_json(
                        {
                            "type": "error",
                            "code": "tts_failed",
                            "message": "TTS synthesis failed",
                        }
                    )

            await self._sink.send_json(
                {
                    "type": "tts_chunk_end",
                    "chunk_id": chunk_id,
                    "action_id": action_id,
                    "success": synthesis_succeeded,
                }
            )

            if not synthesis_succeeded:
                logger.error(
                    "Skipping failed TTS chunk "
                    "(session=%s, chunk=%s)",
                    self.session_id,
                    chunk_id,
                )

    async def close(self) -> None:
        action_id = self._confirmation_action_id

        if action_id:
            invalidate_confirmation_playback(
                self.session_id,
                action_id,
            )

        task = self._turn_task

        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        self._turn_task = None
