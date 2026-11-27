from collections import deque
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any

from app.contracts import Timing

logger = logging.getLogger(__name__)


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    idx = max(0, min(n - 1, math.ceil(p / 100.0 * n) - 1))
    return sorted_vals[idx]


class TurnTimer:
    def __init__(self, session_id: str, turn: int, silence_ms: int) -> None:
        self.session_id = session_id
        self.turn = turn
        self.silence_ms = silence_ms
        self.marks: dict[str, float] = {}
        self.agent_timing: Timing | None = None
        self.cancelled: bool = False

    def mark(self, name: str, t: float | None = None) -> None:
        if name not in self.marks:
            self.marks[name] = time.perf_counter() if t is None else t


class LatencyTracker:
    _METRIC_KEYS = (
        "stt_ms",
        "agent_first_chunk_ms",
        "tts_first_audio_ms",
        "e2e_ms",
        "e2e_with_silence_ms",
    )

    def __init__(self, path: str | Path | None, ring_size: int = 200) -> None:
        self.path = Path(path) if path is not None else None
        self._ring: deque[dict[str, Any]] = deque(maxlen=ring_size)
        self._lock = threading.Lock()

    def new_turn(self, session_id: str, turn: int, silence_ms: int) -> TurnTimer:
        return TurnTimer(session_id=session_id, turn=turn, silence_ms=silence_ms)

    def record(self, timer: TurnTimer) -> dict[str, Any] | None:
        if timer.cancelled:
            return None
        m = timer.marks
        required = ("speech_end", "stt_done", "agent_first_chunk", "tts_first_audio")
        if any(k not in m for k in required):
            return None

        stt_ms = round((m["stt_done"] - m["speech_end"]) * 1000.0, 1)
        agent_first_chunk_ms = round((m["agent_first_chunk"] - m["stt_done"]) * 1000.0, 1)
        tts_first_audio_ms = round((m["tts_first_audio"] - m["agent_first_chunk"]) * 1000.0, 1)
        e2e_ms = round((m["tts_first_audio"] - m["speech_end"]) * 1000.0, 1)
        e2e_with_silence_ms = round(e2e_ms + timer.silence_ms, 1)

        row: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "session_id": timer.session_id,
            "turn": timer.turn,
            "stt_ms": stt_ms,
            "agent_first_chunk_ms": agent_first_chunk_ms,
            "tts_first_audio_ms": tts_first_audio_ms,
            "e2e_ms": e2e_ms,
            "e2e_with_silence_ms": e2e_with_silence_ms,
        }
        if timer.agent_timing is not None:
            row["agent_reported"] = timer.agent_timing.model_dump()

        with self._lock:
            self._ring.append(row)
            if self.path is not None:
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                except Exception as exc:
                    logger.warning("Failed to write latency record to %s: %s", self.path, exc)

        return row

    def summary(self) -> dict[str, Any]:
        with self._lock:
            rows = list(self._ring)

        metrics: dict[str, dict[str, float]] = {}
        for key in self._METRIC_KEYS:
            vals = sorted(float(r[key]) for r in rows if key in r)
            if vals:
                metrics[key] = {
                    "p50": _percentile(vals, 50),
                    "p95": _percentile(vals, 95),
                    "max": vals[-1],
                }
            else:
                metrics[key] = {"p50": 0.0, "p95": 0.0, "max": 0.0}

        return {
            "count": len(rows),
            "metrics": metrics,
        }
