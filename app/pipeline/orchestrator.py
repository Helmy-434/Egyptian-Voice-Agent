import asyncio
import contextlib
import logging
from typing import AsyncGenerator, Callable, Literal, Protocol
from app.agent.state import mark_confirmation_played
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

        self._confirmation_chunks: dict[str, set[int]] = {}
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
        self._frame_bytes = settings.audio_sample_rate * settings.audio_frame_ms // 1000 * 2
        self._rx_buf = bytearray()

        self._state: str = ""
        self._turn_counter: int = 0
        self._turn_task: asyncio.Task[None] | None = None
        self._first_turn: bool = True
        self._pending_truncated: str | None = None

        self._next_chunk_id: int = 0
        self._chunk_texts: dict[int, str] = {}
        self._sent_order: list[int] = []
        self._acked: set[int] = set()
        self._all_acked = asyncio.Event()
        self._worker_done: bool = False
        self._audio_seconds_sent: float = 0.0
        self._playback_ack_timeout_pad: float = 5.0

        self._vad.reset()

    async def start(self) -> None:
        await self._set_state("listening")

    async def _set_state(self, new_state: str) -> None:
        if new_state != self._state:
            self._state = new_state
            await self._sink.send_json({"type": "state", "state": new_state})

    def _reset_turn_playback_state(self) -> None:
        self._confirmation_chunks.clear()
        self._confirmation_action_id = None
        self._sent_order.clear()
        self._acked.clear()
        self._chunk_texts.clear()
        self._all_acked.clear()
        self._worker_done = False
        self._audio_seconds_sent = 0.0

    def _maybe_set_all_acked(self) -> None:
        if self._worker_done and set(self._sent_order) <= self._acked:
            self._all_acked.set()

    def _truncated_text(self) -> str:
        played: list[str] = []
        for cid in self._sent_order:
            if cid not in self._acked:
                break
            text = self._chunk_texts.get(cid)
            if text is not None:
                played.append(text)
        return " ".join(played)

    async def feed_audio(self, data: bytes) -> None:
        self._rx_buf.extend(data)
        while len(self._rx_buf) >= self._frame_bytes:
            frame = bytes(self._rx_buf[: self._frame_bytes])
            del self._rx_buf[: self._frame_bytes]

            if self._state in ("thinking", "speaking"):
                self._sm.min_speech_ms = self._settings.barge_in_min_speech_ms
            else:
                self._sm.min_speech_ms = self._settings.vad_min_speech_ms

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
                self._turn_task = asyncio.create_task(self._run_turn(event.audio, timer))

    async def handle_message(self, msg: dict) -> None:
        mtype = msg.get("type")
        if mtype == "voice":
            v = msg.get("voice")
            if v in ("male", "female"):
                self._voice = v
        elif mtype == "played":
            cid = msg.get("chunk_id")
            if isinstance(cid, int) and cid in self._chunk_texts:
                self._acked.add(cid)
                self._maybe_set_all_acked()

    async def _barge_in(self) -> None:
        task = self._turn_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await self._sink.send_json({"type": "barge_in"})
            self._pending_truncated = self._truncated_text()
            self._reset_turn_playback_state()

    async def _run_turn(self, pcm: bytes, timer: TurnTimer) -> None:
        self._reset_turn_playback_state()
        text_q: asyncio.Queue[tuple[str, str | None] | None] = asyncio.Queue()
        tts_task = asyncio.create_task(self._tts_worker(text_q, timer))
        agen: AsyncGenerator[AgentChunk, None] | None = None
        cancelled_by_barge_in = False

        try:
            result = await self._stt.transcribe(pcm, self._settings.audio_sample_rate)
            timer.mark("stt_done")
            text = result.text.strip()
            if not text:
                timer.cancelled = True
                return

            await self._sink.send_json({"type": "transcript", "text": text})
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
                    timer.mark("agent_first_chunk")

                    confirmation_action_id = None
                    if chunk.metadata:
                        confirmation_action_id = chunk.metadata.get(
                            "confirmation_action_id"
                        )

                    await self._sink.send_json(
                        {"type": "agent_text", "text": chunk.content}
                    )

                    await text_q.put((chunk.content, confirmation_action_id))

                elif chunk.type == "confirm":
                    action_id = (chunk.metadata or {}).get("action_id")

                    if not action_id:
                        raise ValueError("Confirmation event is missing action_id")

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

            await text_q.put(None)
            await tts_task
            playback_complete = await self._wait_playback_done()

            self._pending_truncated = self._truncated_text()

            # Mark the email read-back as played only when every chunk
            # belonging to that confirmation has been acknowledged.
            action_id = self._confirmation_action_id

            if playback_complete and action_id:
                confirmation_chunk_ids = self._confirmation_chunks.get(
                    action_id, set()
                )

                confirmation_fully_played = (
                        bool(confirmation_chunk_ids)
                        and confirmation_chunk_ids.issubset(self._acked)
                )

                if confirmation_fully_played:
                    mark_confirmation_played(
                        self.session_id,
                        action_id,
                    )
                    logger.info(
                        "Email confirmation read-back fully played (session=%s)",
                        self.session_id,
                    )
            if not playback_complete:
                logger.warning(
                    "Assistant playback was not fully acknowledged (session=%s)",
                    self.session_id,
                )
            self._latency.record(timer)

        except asyncio.CancelledError:
            timer.cancelled = True
            cancelled_by_barge_in = True
            raise
        except STTError as exc:
            timer.cancelled = True
            await self._sink.send_json(
                {"type": "error", "code": "stt_failed", "message": str(exc)}
            )
            with contextlib.suppress(Exception):
                await text_q.put(("ممكن تعيد كلامك تاني؟",None))
                await text_q.put(None)
                await tts_task
                await self._wait_playback_done()
        except Exception as exc:
            timer.cancelled = True
            logger.exception("Agent turn failed: %s", exc)
            await self._sink.send_json(
                {"type": "error", "code": "agent_failed", "message": str(exc)}
            )
            with contextlib.suppress(Exception):
                await text_q.put(("آسف، حصلت مشكلة",None))
                await text_q.put(None)
                await tts_task
                await self._wait_playback_done()
        finally:
            if not tts_task.done():
                tts_task.cancel()
            await asyncio.gather(tts_task, return_exceptions=True)
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

            text, confirmation_action_id = item

            chunk_id = self._next_chunk_id
            self._next_chunk_id += 1
            self._chunk_texts[chunk_id] = text
            self._sent_order.append(chunk_id)

            if confirmation_action_id:
                self._confirmation_chunks.setdefault(
                    confirmation_action_id, set()
                ).add(chunk_id)

            voice = self._voice
            announced_rate = self._tts.sample_rate

            await self._sink.send_json(
                {
                    "type": "tts_chunk_start",
                    "chunk_id": chunk_id,
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
                    async for audio in self._tts.synthesize_stream(text, voice):
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

                        self._audio_seconds_sent += (
                                len(audio.pcm) / (2 * audio.sample_rate)
                        )

                    synthesis_succeeded = True
                    break

                except TTSError as exc:
                    if attempt == 0:
                        await asyncio.sleep(0.3)
                        continue

                    logger.exception("TTS synthesis failed")

                    await self._sink.send_json(
                        {
                            "type": "error",
                            "code": "tts_failed",
                            "message": str(exc),
                        }
                    )

            if not synthesis_succeeded:
                logger.error(
                    "Skipping failed TTS chunk (session=%s, chunk=%s)",
                    self.session_id,
                    chunk_id,
                )

            await self._sink.send_json(
                {"type": "tts_chunk_end", "chunk_id": chunk_id}
            )

        self._worker_done = True
        self._maybe_set_all_acked()

    async def _wait_playback_done(self) -> bool:
        """Return True only when all generated audio chunks are acknowledged."""

        if self._audio_seconds_sent <= 0:
            return False

        timeout = self._audio_seconds_sent + self._playback_ack_timeout_pad

        try:
            await asyncio.wait_for(
                self._all_acked.wait(),
                timeout=timeout,
            )
            return True

        except asyncio.TimeoutError:
            logger.warning(
                "Timed out waiting for playback acks (session=%s, timeout=%.2fs)",
                self.session_id,
                timeout,
            )
            return False


    async def close(self) -> None:
        task = self._turn_task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._turn_task = None
