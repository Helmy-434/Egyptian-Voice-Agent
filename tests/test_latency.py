import json
from pathlib import Path

from app.contracts import Timing
from app.pipeline.latency import LatencyTracker


def test_latency_metrics_and_jsonl(tmp_path: Path):
    log_path = tmp_path / "sub" / "latency.jsonl"
    tracker = LatencyTracker(log_path, ring_size=5)

    t = tracker.new_turn("s1", 1, silence_ms=450)
    t.mark("speech_end", 1.000)
    t.mark("stt_done", 1.200)
    t.mark("agent_first_chunk", 1.350)
    t.mark("tts_first_audio", 1.500)
    t.agent_timing = Timing(first_token_ms=50, first_chunk_ms=100, total_ms=200)

    row = tracker.record(t)
    assert row is not None
    assert row["stt_ms"] == 200.0
    assert row["agent_first_chunk_ms"] == 150.0
    assert row["tts_first_audio_ms"] == 150.0
    assert row["e2e_ms"] == 500.0
    assert row["e2e_with_silence_ms"] == 950.0
    assert row["agent_reported"]["first_chunk_ms"] == 100

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["session_id"] == "s1"


def test_cancelled_and_incomplete_timers_skipped(tmp_path: Path):
    tracker = LatencyTracker(tmp_path / "lat.jsonl")
    t1 = tracker.new_turn("s1", 1, 450)
    t1.mark("speech_end", 1.0)
    t1.mark("stt_done", 1.1)
    t1.mark("agent_first_chunk", 1.2)
    t1.mark("tts_first_audio", 1.3)
    t1.cancelled = True
    assert tracker.record(t1) is None

    t2 = tracker.new_turn("s1", 2, 450)
    t2.mark("speech_end", 1.0)
    assert tracker.record(t2) is None
    assert tracker.summary()["count"] == 0


def test_percentiles_and_ring_size():
    tracker = LatencyTracker(None, ring_size=5)
    for i in range(10):
        t = tracker.new_turn("s1", i, silence_ms=100)
        t.mark("speech_end", 0.0)
        t.mark("stt_done", 0.1)
        t.mark("agent_first_chunk", 0.2)
        t.mark("tts_first_audio", 0.3 + i * 0.1)
        tracker.record(t)

    summary = tracker.summary()
    assert summary["count"] == 5
    # Last 5 e2e_ms values: 800, 900, 1000, 1100, 1200
    assert summary["metrics"]["e2e_ms"]["p50"] == 1000.0
    assert summary["metrics"]["e2e_ms"]["p95"] == 1200.0
    assert summary["metrics"]["e2e_ms"]["max"] == 1200.0


def test_io_error_does_not_raise(tmp_path: Path):
    # Point path to an existing directory so open() fails
    bad_path = tmp_path / "adir"
    bad_path.mkdir()
    tracker = LatencyTracker(bad_path)
    t = tracker.new_turn("s1", 1, 100)
    t.mark("speech_end", 0.0)
    t.mark("stt_done", 0.1)
    t.mark("agent_first_chunk", 0.2)
    t.mark("tts_first_audio", 0.3)
    row = tracker.record(t)
    assert row is not None
