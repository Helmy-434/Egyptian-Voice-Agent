import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import wave

from app.configs import Settings
from app.pipeline.stt.base import STTError, STTResult, SampleRate

logger = logging.getLogger(__name__)

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")
_MIN_AUDIO_SEC = 0.10
_CLOSE_TIMEOUT_SEC = 3.0
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class QwenCleoSTT:
    """Local QwenCleo GGUF driver backed by the bundled audio.cpp server."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base_timeout_sec = settings.stt_timeout_sec
        self._per_audio_sec = settings.stt_timeout_per_audio_sec
        self._server_process: Optional[subprocess.Popen[bytes]] = None
        self._server_temp: Optional[tempfile.TemporaryDirectory[str]] = None
        self._server_url = f"http://127.0.0.1:{settings.qwencleo_server_port}"
        self.device = "uninitialized"
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwencleo-stt")
        self._worker_busy = threading.Event()

    async def warmup(self) -> None:
        """Start the local server, load its GGUF, and run a short warmup request."""
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(self._executor, self._warmup_sync)
        except STTError:
            await loop.run_in_executor(self._executor, self._stop_server_sync)
            raise
        except Exception as exc:
            await loop.run_in_executor(self._executor, self._stop_server_sync)
            raise STTError(f"Failed to warm up local QwenCleo-ASR: {exc}") from exc

    async def transcribe(
        self,
        audio: bytes,
        sample_rate: SampleRate = 16000,
    ) -> STTResult:
        """Transcribe one mono little-endian PCM16 utterance off the event loop."""
        if not audio:
            return STTResult(text="", confidence=None)

        aligned_len = len(audio) - (len(audio) % 2)
        audio_sec = aligned_len / (sample_rate * 2)
        if audio_sec < _MIN_AUDIO_SEC:
            return STTResult(text="", confidence=None)

        process = self._server_process
        if process is None or process.poll() is not None:
            raise STTError("Local QwenCleo-ASR is not running; call warmup() at startup.")

        if self._worker_busy.is_set():
            raise STTError("Local QwenCleo-ASR worker is still busy from an earlier turn.")

        self._worker_busy.set()
        pcm_bytes = audio[:aligned_len]
        timeout_sec = self._base_timeout_sec + self._per_audio_sec * audio_sec
        loop = asyncio.get_running_loop()
        try:
            future = loop.run_in_executor(
                self._executor,
                self._infer_sync,
                pcm_bytes,
                sample_rate,
            )
        except RuntimeError as exc:
            self._worker_busy.clear()
            raise STTError(f"Local QwenCleo-ASR worker is unavailable: {exc}") from exc

        try:
            text = await asyncio.wait_for(future, timeout=timeout_sec)
        except asyncio.TimeoutError as exc:
            raise STTError(
                f"Local QwenCleo-ASR timed out after {timeout_sec:.2f}s "
                f"(audio={audio_sec:.2f}s)"
            ) from exc
        except STTError:
            raise
        except Exception as exc:
            raise STTError(f"Local QwenCleo-ASR inference failed: {exc}") from exc

        return STTResult(text=text, confidence=None)

    async def close(self) -> None:
        """Stop the managed server and clean up its temporary configuration."""
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, self._stop_server_sync),
                timeout=_CLOSE_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError as exc:
            raise STTError("Timed out while stopping the local QwenCleo-ASR server.") from exc
        finally:
            self._executor.shutdown(wait=False, cancel_futures=True)

    def _warmup_sync(self) -> None:
        model_path = self._resolve_project_path(self._settings.qwencleo_model_path)
        server_path = self._resolve_project_path(self._settings.qwencleo_server_path)
        for path in (model_path, server_path):
            if not path.is_file():
                raise STTError(f"Required local QwenCleo file was not found: {path}")

        try:
            import torch
        except ImportError as exc:
            raise STTError(
                "torch is required to detect CUDA for STT_PROVIDER=qwencleo. "
                "Install requirements-gpu.txt or switch to STT_PROVIDER=cloud."
            ) from exc

        if torch.cuda.is_available():
            backend = "cuda"
            self.device = "cuda:0"
        elif self._settings.qwencleo_allow_cpu:
            backend = "cpu"
            self.device = "cpu"
            logger.warning("CUDA unavailable; running local QwenCleo-ASR on CPU (QWENCLEO_ALLOW_CPU=true)")
        else:
            raise STTError(
                "CUDA is unavailable for QwenCleo-ASR. Set QWENCLEO_ALLOW_CPU=true to override "
                "or switch STT_PROVIDER=cloud."
            )

        self._server_temp = tempfile.TemporaryDirectory(prefix="qwencleo-server-")
        temp_dir = Path(self._server_temp.name)
        config_path = temp_dir / "server.json"
        log_path = temp_dir / "server.log"
        config = {
            "host": "127.0.0.1",
            "port": self._settings.qwencleo_server_port,
            "backend": backend,
            "device": 0,
            "threads": 2,
            "lazy_load": False,
            "max_loaded_models": 1,
            "models": [
                {
                    "id": "qwencleo-local",
                    "family": "qwen3_asr",
                    "path": str(model_path),
                    "task": "asr",
                    "mode": "offline",
                }
            ],
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        with log_path.open("wb") as log_file:
            self._server_process = subprocess.Popen(
                [
                    str(server_path),
                    "--config",
                    str(config_path),
                    "--no-ui",
                    "--log",
                ],
                cwd=server_path.parent,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )

        self._wait_until_ready(log_path)
        warmup_audio = self._load_warmup_audio(self._settings.audio_sample_rate)
        wav_path = self._write_wav(warmup_audio, self._settings.audio_sample_rate)
        try:
            self._request_transcription(wav_path, timeout_sec=30.0)
        finally:
            wav_path.unlink(missing_ok=True)

        logger.info("Local QwenCleo-ASR warmup complete on %s.", self.device)

    def _wait_until_ready(self, log_path: Path) -> None:
        process = self._server_process
        if process is None:
            raise STTError("Local QwenCleo-ASR server process was not created.")

        deadline = time.monotonic() + self._settings.qwencleo_server_startup_timeout_sec
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            exit_code = process.poll()
            if exit_code is not None:
                raise STTError(
                    f"Local QwenCleo-ASR server exited with code {exit_code}. "
                    f"Server log: {self._read_server_log(log_path)}"
                )
            try:
                with urlopen(f"{self._server_url}/health", timeout=1.0):
                    return
            except (OSError, URLError, TimeoutError) as exc:
                last_error = exc
                time.sleep(0.25)

        raise STTError(
            f"Local QwenCleo-ASR server did not become ready within "
            f"{self._settings.qwencleo_server_startup_timeout_sec:.1f}s"
            f" ({last_error}). Server log: {self._read_server_log(log_path)}"
        )

    def _load_warmup_audio(self, sample_rate: int) -> bytes:
        raw_path = self._settings.qwencleo_warmup_clip_path
        if not raw_path:
            logger.info("QWENCLEO_WARMUP_CLIP_PATH unset; warming up with 2.0s silence.")
            return bytes(sample_rate * 2 * 2)

        clip_path = self._resolve_project_path(raw_path)
        if not clip_path.is_file():
            logger.warning("Warmup clip not found at %s; falling back to 2.0s silence.", clip_path)
            return bytes(sample_rate * 2 * 2)

        try:
            with wave.open(str(clip_path), "rb") as wav_file:
                actual_format = (
                    wav_file.getnchannels(),
                    wav_file.getsampwidth(),
                    wav_file.getframerate(),
                )
                if actual_format != (1, 2, sample_rate):
                    logger.warning(
                        "Warmup clip %s has format %s; expected (1, 2, %d). "
                        "Falling back to 2.0s silence.",
                        clip_path,
                        actual_format,
                        sample_rate,
                    )
                    return bytes(sample_rate * 2 * 2)
                raw = wav_file.readframes(sample_rate * 3)
                if not raw:
                    logger.warning("Warmup clip %s is empty; falling back to 2.0s silence.", clip_path)
                    return bytes(sample_rate * 2 * 2)
                logger.info(
                    "Warming up QwenCleo-ASR with clip %s (%.2fs).",
                    clip_path,
                    len(raw) / (sample_rate * 2),
                )
                return raw
        except (wave.Error, OSError) as exc:
            logger.warning(
                "Failed to read warmup clip %s (%s); falling back to 2.0s silence.",
                clip_path,
                exc,
            )
            return bytes(sample_rate * 2 * 2)

    def _write_wav(self, pcm_bytes: bytes, sample_rate: int) -> Path:
        if self._server_temp is None:
            raise STTError("Local QwenCleo-ASR server is not initialized.")

        with tempfile.NamedTemporaryFile(
            dir=self._server_temp.name,
            prefix="audio-",
            suffix=".wav",
            delete=False,
        ) as temp_file:
            wav_path = Path(temp_file.name)

        try:
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm_bytes)
        except Exception:
            wav_path.unlink(missing_ok=True)
            raise
        return wav_path

    def _request_transcription(self, wav_path: Path, timeout_sec: float) -> str:
        request_body = json.dumps(
            {
                "model": "qwencleo-local",
                "audio": str(wav_path.resolve()),
                "language": "Arabic",
            }
        ).encode("utf-8")
        request = Request(
            f"{self._server_url}/v1/audio/transcriptions",
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            raise STTError(
                f"Local QwenCleo-ASR server returned HTTP {exc.code}: {response_body}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise STTError(f"Could not reach local QwenCleo-ASR server: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise STTError(f"Local QwenCleo-ASR server returned invalid JSON: {exc}") from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise STTError("Local QwenCleo-ASR response did not contain a string 'text' field.")
        return _SPECIAL_TOKEN_RE.sub("", payload["text"]).strip()

    def _infer_sync(self, pcm_bytes: bytes, sample_rate: int) -> str:
        try:
            wav_path = self._write_wav(pcm_bytes, sample_rate)
            try:
                audio_sec = len(pcm_bytes) / (sample_rate * 2)
                timeout_sec = self._base_timeout_sec + self._per_audio_sec * audio_sec
                return self._request_transcription(wav_path, timeout_sec)
            finally:
                wav_path.unlink(missing_ok=True)
        finally:
            self._worker_busy.clear()

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

    @staticmethod
    def _resolve_project_path(raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        return path.resolve()

    @staticmethod
    def _read_server_log(log_path: Path) -> str:
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return f"<could not read server log: {exc}>"
        return "\n".join(lines[-30:]) or "<empty>"
