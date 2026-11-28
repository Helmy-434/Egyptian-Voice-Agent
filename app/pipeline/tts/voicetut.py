import asyncio
import base64
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unicodedata
from collections.abc import AsyncIterator, Iterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.configs import Settings
from app.contracts import MAX_SPEAKABLE_CHARS
from app.pipeline.net import _port_in_use, has_cuda_driver
from app.pipeline.tts.base import AudioChunk, TTSError, Voice

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MODEL_ID = "voicetut-local"
_SAMPLE_RATE = 24000
_PCM_CHUNK_MS = 150
_FIRST_EMIT_BYTES = _SAMPLE_RATE * 2 * 20 // 1000
_CLOSE_TIMEOUT_SEC = 3.0
_WARMUP_TEXT = "أهلاً بيك، إزاي أقدر أساعدك النهاردة؟"
_END_OF_STREAM = object()


@dataclass(frozen=True, slots=True)
class _WorkerError:
    exception: Exception


def _has_pronounceable_text(text: str) -> bool:
    return any(unicodedata.category(char).startswith("L") or char.isdigit() for char in text)


class VoiceTutTTS:
    """Local VoiceTut GGUF driver backed by the audio.cpp server."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._server_process: subprocess.Popen[bytes] | None = None
        self._server_temp: tempfile.TemporaryDirectory[str] | None = None
        self._server_log_path: Path | None = None
        self._server_url = f"http://127.0.0.1:{settings.voicetut_server_port}"
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voicetut-tts")
        self._worker_busy = threading.Event()
        self._abandoned_job: threading.Event | None = None
        self._active_calls: dict[threading.Event, tuple[list[Any], Any]] = {}
        self._active_lock = threading.Lock()
        self._closed = False
        self._ready = False
        self.device = "uninitialized"
        self.sample_rate = 0

    async def warmup(self) -> None:
        """Start the local server and synthesize a short phrase through both voices."""
        if self._closed:
            raise TTSError("VoiceTut-TTS engine is closed.")
        if self._ready:
            return

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._warmup_sync)
        except TTSError:
            await loop.run_in_executor(self._executor, self._stop_server_sync)
            raise
        except Exception as exc:
            await loop.run_in_executor(self._executor, self._stop_server_sync)
            raise TTSError(f"Failed to warm up local VoiceTut-TTS: {exc}") from exc

        self.sample_rate = _SAMPLE_RATE
        self._ready = True

    async def synthesize_stream(self, text: str, voice: Voice) -> AsyncIterator[AudioChunk]:
        """Stream PCM16 chunks for one speakable sentence/clause from the agent.

        Text must be at most ``MAX_SPEAKABLE_CHARS``. Empty or unpronounceable
        input yields nothing; overlong input or an unknown voice raises ``TTSError``.
        The audio.cpp SSE stream is forwarded as soon as audio deltas arrive.
        Cancellation closes the active HTTP response so the server can stop its
        generation; if it is still busy, the next call fails fast for cloud fallback.
        """
        if len(text) > MAX_SPEAKABLE_CHARS:
            raise TTSError(f"Text exceeds MAX_SPEAKABLE_CHARS ({MAX_SPEAKABLE_CHARS}).")

        speaker = self._speaker_for_voice(voice)
        if not text.strip() or not _has_pronounceable_text(text):
            return
        if not self._ready or self._closed:
            raise TTSError("VoiceTut-TTS is not ready; call warmup() before synthesis.")
        job = self._abandoned_job
        if job is not None and not job.is_set():
            raise TTSError("VoiceTut-TTS worker is still finishing an abandoned synthesis.")

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[object] = asyncio.Queue(maxsize=1)
        cancel = threading.Event()
        started = threading.Event()
        worker_done = threading.Event()
        response_holder: list[Any] = []
        response_lock = threading.Lock()
        started_at = time.perf_counter()
        first_chunk_seen = False
        stream_finished = False

        with self._active_lock:
            self._active_calls[cancel] = (response_holder, response_lock)

        try:
            try:
                loop.run_in_executor(
                    self._executor,
                    self._synthesize_sync,
                    text,
                    speaker,
                    cancel,
                    started,
                    worker_done,
                    queue,
                    loop,
                    response_holder,
                    response_lock,
                )
            except RuntimeError as exc:
                raise TTSError(f"VoiceTut-TTS worker is unavailable: {exc}") from exc

            first_timeout = self._settings.tts_first_chunk_timeout_sec
            total_timeout = self._settings.tts_total_timeout_sec
            while True:
                remaining = total_timeout - (time.perf_counter() - started_at)
                if remaining <= 0:
                    raise TTSError(f"VoiceTut-TTS synthesis timed out after {total_timeout:.2f}s.")
                timeout = min(first_timeout, remaining) if not first_chunk_seen else remaining
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=timeout)
                except asyncio.TimeoutError as exc:
                    elapsed = time.perf_counter() - started_at
                    if elapsed >= total_timeout:
                        reason = f"total timeout after {total_timeout:.2f}s"
                    else:
                        reason = f"first-chunk timeout after {first_timeout:.2f}s"
                    raise TTSError(f"VoiceTut-TTS {reason}.") from exc

                if item is _END_OF_STREAM:
                    stream_finished = True
                    break
                if isinstance(item, _WorkerError):
                    raise TTSError(f"VoiceTut-TTS synthesis failed: {item.exception}") from item.exception
                if not isinstance(item, AudioChunk):
                    raise TTSError("VoiceTut-TTS worker returned an invalid audio chunk.")
                if not first_chunk_seen:
                    first_chunk_seen = True
                    logger.debug(
                        "VoiceTut-TTS first audio chunk ready in %.1f ms.",
                        (time.perf_counter() - started_at) * 1000.0,
                    )
                yield item
        finally:
            cancel.set()
            self._close_response(response_holder, response_lock)
            if not stream_finished and started.is_set() and not worker_done.is_set():
                self._abandoned_job = worker_done
            with self._active_lock:
                self._active_calls.pop(cancel, None)

    async def close(self) -> None:
        """Stop the managed audio.cpp server and release its temporary config."""
        if self._closed:
            return
        self._closed = True

        with self._active_lock:
            active_calls = list(self._active_calls.items())
        for cancel, (response_holder, response_lock) in active_calls:
            cancel.set()
            self._close_response(response_holder, response_lock)

        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, self._stop_server_sync),
                timeout=_CLOSE_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError as exc:
            raise TTSError("Timed out while stopping the local VoiceTut-TTS server.") from exc
        except Exception as exc:
            raise TTSError(f"Failed to stop the local VoiceTut-TTS server: {exc}") from exc
        finally:
            self._ready = False
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _warmup_sync(self) -> None:
        model_path = self._resolve_project_path(self._settings.voicetut_model_path)
        server_path = self._resolve_project_path(self._settings.voicetut_server_path)
        for path in (model_path, server_path):
            if not path.is_file():
                raise TTSError(f"Required VoiceTut/audio.cpp file was not found: {path}")
        voice_dir = self._validate_voice_dir()

        if has_cuda_driver():
            backend = "cuda"
            self.device = "cuda:0"
        elif self._settings.voicetut_allow_cpu:
            backend = "cpu"
            self.device = "cpu"
            logger.warning("CUDA unavailable; running VoiceTut-TTS on CPU (VOICETUT_ALLOW_CPU=true).")
        else:
            raise TTSError(
                "The NVIDIA CUDA driver is unavailable for VoiceTut-TTS. "
                "Set VOICETUT_ALLOW_CPU=true to override "
                "or switch TTS_PROVIDER=cloud."
            )

        self._server_temp = tempfile.TemporaryDirectory(prefix="voicetut-server-")
        temp_dir = Path(self._server_temp.name)
        config_path = temp_dir / "server.json"
        self._server_log_path = temp_dir / "server.log"
        config = {
            "host": "127.0.0.1",
            "port": self._settings.voicetut_server_port,
            "backend": backend,
            "device": 0,
            "threads": 2,
            "lazy_load": False,
            "max_loaded_models": 1,
            "voice_dir": str(voice_dir),
            "models": [
                {
                    "id": _MODEL_ID,
                    "family": "omnivoice",
                    "path": str(model_path),
                    "task": "tts",
                    "mode": "streaming",
                }
            ],
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")
        port = self._settings.voicetut_server_port
        if _port_in_use(port):
            raise TTSError(
                f"Port {port} is already in use (a stale audiocpp_server process?). "
                "Stop it or change the port."
            )
        self._start_server(server_path, config_path)
        self._wait_until_ready()

        for speaker in (self._settings.voicetut_male_voice, self._settings.voicetut_female_voice):
            pcm = b"".join(self._read_pcm_sync(_WARMUP_TEXT, speaker, None, None, None))
            if not pcm:
                raise TTSError(f"VoiceTut-TTS warmup produced no audio for speaker {speaker!r}.")
            seconds = len(pcm) / (2 * _SAMPLE_RATE)
            logger.info("VoiceTut warmup speaker=%s audio=%.2fs", speaker, seconds)
            if not 0.5 <= seconds <= 10.0:
                logger.warning("Warmup audio duration %.2fs looks wrong; verify _SAMPLE_RATE.", seconds)

        self.sample_rate = _SAMPLE_RATE
        logger.info(
            "Local VoiceTut-TTS warmup complete: backend=%s device=%s sample_rate=%d Hz",
            backend,
            self.device,
            self.sample_rate,
        )

    def _start_server(self, server_path: Path, config_path: Path) -> None:
        if self._server_temp is None or self._server_log_path is None:
            raise TTSError("VoiceTut-TTS server temporary directory was not initialized.")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with self._server_log_path.open("wb") as log_file:
            self._server_process = subprocess.Popen(
                [str(server_path), "--config", str(config_path), "--no-ui", "--log"],
                cwd=server_path.parent,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )

    def _wait_until_ready(self) -> None:
        process = self._server_process
        if process is None:
            raise TTSError("Local VoiceTut-TTS server process was not created.")

        timeout_sec = self._settings.voicetut_server_startup_timeout_sec
        deadline = time.monotonic() + timeout_sec
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            exit_code = process.poll()
            if exit_code is not None:
                raise TTSError(
                    f"Local VoiceTut-TTS server exited with code {exit_code}. "
                    f"Server log: {self._read_server_log()}"
                )
            try:
                with urlopen(f"{self._server_url}/health", timeout=1.0):
                    return
            except (OSError, URLError, TimeoutError) as exc:
                last_error = exc
                time.sleep(0.25)
        raise TTSError(
            f"Local VoiceTut-TTS server did not become ready within {timeout_sec:.1f}s "
            f"({last_error}). Server log: {self._read_server_log()}"
        )

    def _read_pcm_sync(
        self,
        text: str,
        speaker: str,
        cancel: threading.Event | None,
        response_holder: list[Any] | None,
        response_lock: Any,
    ) -> Iterator[bytes]:
        if cancel is not None and cancel.is_set():
            return
        payload = {
            "model": _MODEL_ID,
            "input": text,
            "voice": speaker,
            "language": "arz",
            "response_format": "pcm",
            "stream_format": "sse",
        }
        request = Request(
            f"{self._server_url}/v1/audio/speech",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        try:
            response = urlopen(request, timeout=self._settings.tts_total_timeout_sec)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise TTSError(f"VoiceTut-TTS server returned HTTP {exc.code}: {body}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise TTSError(f"Could not reach local VoiceTut-TTS server: {exc}") from exc

        if response_holder is not None and response_lock is not None:
            with response_lock:
                response_holder.append(response)

        saw_audio = False
        saw_done = False
        try:
            with response:
                for raw_line in response:
                    if cancel is not None and cancel.is_set():
                        return
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise TTSError(f"VoiceTut-TTS server returned invalid SSE JSON: {exc}") from exc
                    event_type = event.get("type")
                    if event_type == "speech.audio.delta":
                        try:
                            pcm = base64.b64decode(event["audio"], validate=True)
                        except (KeyError, ValueError) as exc:
                            raise TTSError(f"VoiceTut-TTS server returned invalid audio data: {exc}") from exc
                        if pcm:
                            saw_audio = True
                            yield pcm
                    elif event_type == "speech.audio.done":
                        saw_done = True
                    elif event_type == "error":
                        raise TTSError(f"VoiceTut-TTS server error: {event.get('error', event)}")
        except (OSError, TimeoutError, URLError) as exc:
            if cancel is None or not cancel.is_set():
                raise TTSError(f"VoiceTut-TTS audio stream failed: {exc}") from exc
        finally:
            if response_holder is not None and response_lock is not None:
                with response_lock:
                    if response_holder and response_holder[0] is response:
                        response_holder.clear()

        if cancel is None or not cancel.is_set():
            if not saw_audio:
                raise TTSError("VoiceTut-TTS server returned no audio.")
            if not saw_done:
                raise TTSError("VoiceTut-TTS server ended without a speech.audio.done event.")

    def _synthesize_sync(
        self,
        text: str,
        speaker: str,
        cancel: threading.Event,
        started: threading.Event,
        worker_done: threading.Event,
        queue: asyncio.Queue[object],
        loop: asyncio.AbstractEventLoop,
        response_holder: list[Any],
        response_lock: Any,
    ) -> None:
        started.set()
        self._worker_busy.set()
        try:
            if cancel.is_set():
                return
            pcm_buffer = bytearray()
            chunk_size = _SAMPLE_RATE * 2 * _PCM_CHUNK_MS // 1000
            audio_seen = False
            first_sent = False
            for pcm_delta in self._read_pcm_sync(
                text,
                speaker,
                cancel,
                response_holder,
                response_lock,
            ):
                if cancel.is_set():
                    break
                pcm_buffer.extend(pcm_delta)
                audio_seen = True
                if not first_sent and len(pcm_buffer) >= _FIRST_EMIT_BYTES:
                    n = len(pcm_buffer) - (len(pcm_buffer) % 2)
                    first = bytes(pcm_buffer[:n])
                    del pcm_buffer[:n]
                    if not self._put_from_worker(
                        loop,
                        queue,
                        AudioChunk(pcm=first, sample_rate=_SAMPLE_RATE),
                        cancel,
                    ):
                        break
                    first_sent = True
                while len(pcm_buffer) >= chunk_size and not cancel.is_set():
                    chunk_bytes = bytes(pcm_buffer[:chunk_size])
                    del pcm_buffer[:chunk_size]
                    if not self._put_from_worker(
                        loop,
                        queue,
                        AudioChunk(pcm=chunk_bytes, sample_rate=_SAMPLE_RATE),
                        cancel,
                    ):
                        break
            if cancel.is_set():
                return
            if not audio_seen:
                raise TTSError("VoiceTut-TTS server returned no audio.")
            if len(pcm_buffer) % 2:
                raise TTSError("VoiceTut-TTS server returned an incomplete PCM16 sample.")
            if pcm_buffer:
                self._put_from_worker(
                    loop,
                    queue,
                    AudioChunk(pcm=bytes(pcm_buffer), sample_rate=_SAMPLE_RATE),
                    cancel,
                )
            if not cancel.is_set():
                self._put_from_worker(loop, queue, _END_OF_STREAM, cancel)
        except Exception as exc:
            if not cancel.is_set():
                self._put_from_worker(loop, queue, _WorkerError(exc), cancel)
        finally:
            self._worker_busy.clear()
            worker_done.set()

    @staticmethod
    def _put_from_worker(
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[object],
        item: object,
        cancel: threading.Event,
    ) -> bool:
        if loop.is_closed():
            return False
        try:
            pending = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
        except RuntimeError:
            return False
        while not cancel.is_set():
            try:
                pending.result(timeout=0.05)
                return True
            except FutureTimeoutError:
                continue
            except Exception:
                return False
        pending.cancel()
        return False

    @staticmethod
    def _close_response(response_holder: list[Any], response_lock: Any) -> None:
        with response_lock:
            response = response_holder[0] if response_holder else None
        if response is not None:
            try:
                response.close()
            except OSError as exc:
                logger.debug("Error closing VoiceTut-TTS response: %s", exc)

    def _stop_server_sync(self) -> None:
        process = self._server_process
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=_CLOSE_TIMEOUT_SEC)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=_CLOSE_TIMEOUT_SEC)
        finally:
            if process is None or process.poll() is not None:
                self._server_process = None
                if self._server_temp is not None:
                    self._server_temp.cleanup()
                    self._server_temp = None

    def _validate_voice_dir(self) -> Path:
        voice_dir = self._resolve_project_path(self._settings.voicetut_voice_dir)
        if not voice_dir.is_dir():
            raise TTSError(f"VoiceTut reference speakers directory was not found: {voice_dir}")

        available = sorted(p.stem for p in voice_dir.glob("*.wav") if p.is_file() and p.stat().st_size > 0)
        prompt_path = voice_dir / "prompt_text"
        if not prompt_path.is_file():
            raise TTSError(
                f"VoiceTut prompt_text file was not found in {voice_dir}. "
                f"Available .wav speakers: {available}"
            )

        transcripts: dict[str, str] = {}
        for line in prompt_path.read_text(encoding="utf-8").splitlines():
            if "|" in line:
                name, text = line.split("|", 1)
                if name.strip() and text.strip():
                    transcripts[name.strip()] = text.strip()

        for role, speaker in (
            ("male", self._settings.voicetut_male_voice),
            ("female", self._settings.voicetut_female_voice),
        ):
            wav_path = voice_dir / f"{speaker}.wav"
            if not wav_path.is_file() or wav_path.stat().st_size == 0:
                raise TTSError(
                    f"Configured {role} speaker {speaker!r} is missing {wav_path.name} in {voice_dir}. "
                    f"Available speakers: {available}"
                )
            if speaker not in transcripts:
                raise TTSError(
                    f"Configured {role} speaker {speaker!r} is missing a transcript in {prompt_path}. "
                    f"Available speakers: {available}"
                )
        return voice_dir

    def _speaker_for_voice(self, voice: Voice) -> str:
        if voice == "male":
            return self._settings.voicetut_male_voice
        if voice == "female":
            return self._settings.voicetut_female_voice
        raise TTSError(f"Unknown voice: {voice!r}. Expected 'male' or 'female'.")

    @staticmethod
    def _resolve_project_path(raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        return path.resolve()

    def _read_server_log(self) -> str:
        if self._server_log_path is None:
            return "<no server log>"
        try:
            lines = self._server_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return f"<could not read server log: {exc}>"
        return "\n".join(lines[-30:]) or "<empty>"
