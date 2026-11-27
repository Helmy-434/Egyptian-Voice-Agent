from app.pipeline.vad import VADStateMachine

FRAME = b"\x01\x00" * 512


def _make_sm(**overrides) -> VADStateMachine:
    defaults = dict(
        frame_ms=32,
        threshold=0.5,
        hysteresis=0.15,
        min_speech_ms=64,
        silence_ms=96,
        preroll_ms=64,
        max_utterance_sec=2.0,
    )
    defaults.update(overrides)
    return VADStateMachine(**defaults)


def test_short_speech_never_emits_start():
    sm = _make_sm(min_speech_ms=96)
    assert sm.update(FRAME, 0.8) is None
    assert sm.update(FRAME, 0.8) is None
    assert sm.update(FRAME, 0.1) is None


def test_speech_start_includes_preroll():
    sm = _make_sm(min_speech_ms=64, preroll_ms=64)
    p1 = b"\x02\x00" * 512
    p2 = b"\x03\x00" * 512
    s1 = b"\x04\x00" * 512
    s2 = b"\x05\x00" * 512

    assert sm.update(p1, 0.1) is None
    assert sm.update(p2, 0.1) is None
    assert sm.update(s1, 0.9) is None
    ev = sm.update(s2, 0.9)
    assert ev is not None
    assert ev.kind == "speech_start"
    assert ev.audio == p1 + p2 + s1 + s2


def test_speech_end_after_silence_and_second_utterance():
    sm = _make_sm(min_speech_ms=64, silence_ms=96, preroll_ms=32)
    assert sm.update(FRAME, 0.9) is None
    ev_start = sm.update(FRAME, 0.9)
    assert ev_start is not None and ev_start.kind == "speech_start"

    assert sm.update(FRAME, 0.1) is None
    assert sm.update(FRAME, 0.1) is None
    ev_end = sm.update(FRAME, 0.1)
    assert ev_end is not None and ev_end.kind == "speech_end"
    assert len(ev_end.audio) == 5 * len(FRAME)

    # Second utterance works after reset
    assert sm.update(FRAME, 0.9) is None
    ev_start2 = sm.update(FRAME, 0.9)
    assert ev_start2 is not None and ev_start2.kind == "speech_start"


def test_short_pause_does_not_end_utterance():
    sm = _make_sm(min_speech_ms=64, silence_ms=96, preroll_ms=0)
    assert sm.update(FRAME, 0.9) is None
    assert sm.update(FRAME, 0.9).kind == "speech_start"

    assert sm.update(FRAME, 0.1) is None
    assert sm.update(FRAME, 0.1) is None
    # Speech resumes before 3rd silence frame
    assert sm.update(FRAME, 0.8) is None
    assert sm.update(FRAME, 0.1) is None


def test_hysteresis_band_not_counted_as_silence():
    sm = _make_sm(threshold=0.5, hysteresis=0.15, min_speech_ms=64, silence_ms=64, preroll_ms=0)
    assert sm.update(FRAME, 0.9) is None
    assert sm.update(FRAME, 0.9).kind == "speech_start"

    # 0.40 is between 0.35 (threshold - hysteresis) and 0.50 (threshold)
    for _ in range(5):
        assert sm.update(FRAME, 0.40) is None


def test_max_utterance_sec_forces_end():
    sm = _make_sm(min_speech_ms=32, silence_ms=500, preroll_ms=0, max_utterance_sec=0.1)
    assert sm.update(FRAME, 0.9).kind == "speech_start"
    assert sm.update(FRAME, 0.9) is None
    assert sm.update(FRAME, 0.9) is None
    ev = sm.update(FRAME, 0.9)
    assert ev is not None and ev.kind == "speech_end"


def test_dynamic_min_speech_ms_change():
    sm = _make_sm(min_speech_ms=96, preroll_ms=0)
    assert sm.update(FRAME, 0.9) is None
    sm.min_speech_ms = 64
    ev = sm.update(FRAME, 0.9)
    assert ev is not None and ev.kind == "speech_start"
