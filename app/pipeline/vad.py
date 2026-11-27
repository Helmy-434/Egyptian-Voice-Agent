from collections import deque
from typing import Literal, NamedTuple, Protocol


class VADEvent(NamedTuple):
    kind: Literal["speech_start", "speech_end"]
    audio: bytes = b""


class VADModel(Protocol):
    def prob(self, frame: bytes) -> float: ...
    def reset(self) -> None: ...


class VADStateMachine:
    def __init__(
        self,
        *,
        frame_ms: int,
        threshold: float,
        hysteresis: float,
        min_speech_ms: int,
        silence_ms: int,
        preroll_ms: int,
        max_utterance_sec: float,
    ) -> None:
        self.frame_ms = frame_ms
        self.threshold = threshold
        self.hysteresis = hysteresis
        self.min_speech_ms = min_speech_ms
        self.silence_ms = silence_ms
        self.preroll_ms = preroll_ms
        self.max_utterance_sec = max_utterance_sec

        preroll_frames = max(0, preroll_ms // frame_ms)
        self._preroll: deque[bytes] = deque(maxlen=preroll_frames)
        self._candidate: list[bytes] = []
        self._buffer: bytearray = bytearray()
        self._buffer_frames: int = 0
        self._silence_frames: int = 0
        self._in_speech: bool = False

    def reset(self) -> None:
        self._preroll.clear()
        self._candidate.clear()
        self._buffer.clear()
        self._buffer_frames = 0
        self._silence_frames = 0
        self._in_speech = False

    def update(self, frame: bytes, prob: float) -> VADEvent | None:
        if not self._in_speech:
            if prob >= self.threshold:
                self._candidate.append(frame)
                if len(self._candidate) * self.frame_ms >= self.min_speech_ms:
                    self._in_speech = True
                    self._buffer = bytearray()
                    for f in self._preroll:
                        self._buffer.extend(f)
                    for f in self._candidate:
                        self._buffer.extend(f)
                    self._buffer_frames = len(self._preroll) + len(self._candidate)
                    self._preroll.clear()
                    self._candidate.clear()
                    self._silence_frames = 0
                    return VADEvent("speech_start", bytes(self._buffer))
            else:
                for f in self._candidate:
                    self._preroll.append(f)
                self._candidate.clear()
                self._preroll.append(frame)
            return None

        # State: SPEECH
        self._buffer.extend(frame)
        self._buffer_frames += 1

        if prob < self.threshold - self.hysteresis:
            self._silence_frames += 1
        else:
            self._silence_frames = 0

        if (
            self._silence_frames * self.frame_ms >= self.silence_ms
            or self._buffer_frames * self.frame_ms >= self.max_utterance_sec * 1000
        ):
            audio = bytes(self._buffer)
            self.reset()
            return VADEvent("speech_end", audio)

        return None


class SileroVAD:
    def __init__(self, sample_rate: int) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError(f"SileroVAD supports 8000 or 16000 Hz, got {sample_rate}")
        import numpy as np
        import torch
        from silero_vad import load_silero_vad

        self._np = np
        self._torch = torch
        self._sample_rate = sample_rate
        self._expected_samples = 512 if sample_rate == 16000 else 256
        self._model = load_silero_vad()

    def prob(self, frame: bytes) -> float:
        expected_bytes = self._expected_samples * 2
        if len(frame) != expected_bytes:
            raise ValueError(
                f"Expected {expected_bytes} bytes ({self._expected_samples} samples) "
                f"for {self._sample_rate} Hz, got {len(frame)} bytes"
            )
        audio_np = self._np.frombuffer(frame, "<i2").astype(self._np.float32) / 32768.0
        tensor = self._torch.from_numpy(audio_np)
        with self._torch.no_grad():
            return float(self._model(tensor, self._sample_rate).item())

    def reset(self) -> None:
        self._model.reset_states()
