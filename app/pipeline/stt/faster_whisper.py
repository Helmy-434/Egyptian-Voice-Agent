import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
from faster_whisper import WhisperModel

from app.configs import Settings
from app.pipeline.stt.base import STTError, STTResult, SampleRate

logger = logging.getLogger(__name__)

_MIN_AUDIO_SEC = 0.10


class FasterWhisperSTT:
    """Local Faster-Whisper STT driver."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: WhisperModel | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="faster-whisper-stt",
        )
        self._worker_busy = threading.Event()
        self.device = "uninitialized"

    async def warmup(self) -> None:
        """Load the model and run a short warmup inference."""
        loop = asyncio.get_running_loop()

        try:
            await loop.run_in_executor(
                self._executor,
                self._load_model_sync,
            )

            # Short silent audio initializes the inference path.
            warmup_audio = np.zeros(16000, dtype=np.float32)

            await loop.run_in_executor(
                self._executor,
                self._transcribe_sync,
                warmup_audio,
            )

            logger.info(
                "Faster-Whisper warmup complete: model=%s device=%s",
                self._settings.faster_whisper_model,
                self.device,
            )

        except Exception as exc:
            if isinstance(exc, STTError):
                raise

            raise STTError(
                f"Failed to warm up Faster-Whisper: {exc}"
            ) from exc

    async def transcribe(
        self,
        audio: bytes,
        sample_rate: SampleRate = 16000,
    ) -> STTResult:
        """Transcribe mono little-endian PCM16 audio."""

        if not audio:
            return STTResult(text="", confidence=None)

        aligned_len = len(audio) - (len(audio) % 2)
        audio = audio[:aligned_len]

        audio_sec = aligned_len / (sample_rate * 2)

        if audio_sec < _MIN_AUDIO_SEC:
            return STTResult(text="", confidence=None)

        if self._model is None:
            raise STTError(
                "Faster-Whisper model is not loaded; call warmup() at startup."
            )

        if self._worker_busy.is_set():
            raise STTError(
                "Faster-Whisper worker is still busy from an earlier turn."
            )

        self._worker_busy.set()

        try:
            # Convert PCM16 bytes to normalized float32 waveform.
            waveform = (
                np.frombuffer(audio, dtype=np.int16)
                .astype(np.float32)
                / 32768.0
            )

            loop = asyncio.get_running_loop()

            future = loop.run_in_executor(
                self._executor,
                self._transcribe_sync,
                waveform,
            )

            text = await asyncio.wait_for(
                future,
                timeout=self._settings.faster_whisper_timeout_sec,
            )

            return STTResult(text=text, confidence=None)

        except asyncio.TimeoutError as exc:
            raise STTError(
                "Faster-Whisper timed out after "
                f"{self._settings.faster_whisper_timeout_sec:.2f}s "
                f"(audio={audio_sec:.2f}s)"
            ) from exc

        except STTError:
            raise

        except Exception as exc:
            raise STTError(
                f"Faster-Whisper inference failed: {exc}"
            ) from exc

        finally:
            # The inference may still be running after asyncio timeout
            # or cancellation. Clear busy only when the worker finishes.
            pass

    def _load_model_sync(self) -> None:
        """Load the model inside the dedicated worker thread."""

        try:
            self._model = WhisperModel(
                self._settings.faster_whisper_model,
                device=self._settings.faster_whisper_device,
                compute_type=self._settings.faster_whisper_compute_type,
                cpu_threads=self._settings.faster_whisper_cpu_threads,
            )

            self.device = self._settings.faster_whisper_device

        except Exception as exc:
            raise STTError(
                f"Could not load Faster-Whisper model: {exc}"
            ) from exc

    def _transcribe_sync(self, waveform: np.ndarray) -> str:
        """Run inference and consume the segments generator."""

        try:
            if self._model is None:
                raise STTError("Faster-Whisper model is not loaded.")

            segments, _ = self._model.transcribe(
                waveform,
                language="ar",
                beam_size=5,
                vad_filter=False,
                condition_on_previous_text=False,
            )

            return " ".join(
                segment.text.strip()
                for segment in segments
                if segment.text.strip()
            ).strip()

        except STTError:
            raise

        except Exception as exc:
            raise STTError(
                f"Faster-Whisper transcription failed: {exc}"
            ) from exc

        finally:
            self._worker_busy.clear()

    async def close(self) -> None:
        """Release the model worker."""

        loop = asyncio.get_running_loop()

        try:
            await loop.run_in_executor(
                None,
                lambda: self._executor.shutdown(
                    wait=True,
                    cancel_futures=True,
                ),
            )
        except Exception as exc:
            raise STTError(
                f"Failed to close Faster-Whisper: {exc}"
            ) from exc

        self._model = None
        self.device = "closed"