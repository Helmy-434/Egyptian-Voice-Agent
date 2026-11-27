"""Benchmark end-to-end voice loop latency (VAD -> STT -> Agent -> TTS).

Note: The bundled clips in `scripts/eval_clips` are synthetic Edge-TTS audio
(fine for latency, not for accuracy). Real recordings should be used for the
final README numbers.
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.configs import PROJECT_ROOT, get_settings
from app.main import load_agent_backend
from app.pipeline.latency import LatencyTracker
from app.pipeline.orchestrator import ClientSink, VoiceSession
from app.pipeline.stt.base import create_stt_engine
from app.pipeline.tts.base import create_tts_engine
from app.pipeline.vad import SileroVAD


class NullSink(ClientSink):
    def __init__(self) -> None:
        self.session: VoiceSession | None = None
        self.listening_event = asyncio.Event()

    async def send_json(self, data: dict) -> None:
        t = data.get("type")
        if t == "tts_chunk_end" and self.session is not None:
            await self.session.handle_message({"type": "played", "chunk_id": data["chunk_id"]})
        elif t == "state" and data.get("state") == "listening":
            self.listening_event.set()

    async def send_bytes(self, data: bytes) -> None:
        pass


def _load_pcm16_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise ValueError(f"{path} must be 16 kHz mono PCM16")
        return wf.readframes(wf.getnframes())


async def run_benchmark(turns: int, clips_dir: Path, agent_backend: str) -> None:
    settings = get_settings()
    settings.validate_for_runtime()

    clips = sorted(clips_dir.glob("*.wav"))
    if not clips:
        raise FileNotFoundError(f"No *.wav files found in {clips_dir}")
    pcm_clips = [_load_pcm16_wav(c) for c in clips]

    stt = create_stt_engine(settings)
    t0 = time.perf_counter()
    await stt.warmup()
    stt_warmup_s = time.perf_counter() - t0

    tts = create_tts_engine(settings)
    t0 = time.perf_counter()
    await tts.warmup()
    tts_warmup_s = time.perf_counter() - t0

    print(f"Warmup: STT={stt_warmup_s:.2f}s, TTS={tts_warmup_s:.2f}s")

    vad = SileroVAD(settings.audio_sample_rate)
    stream_turn, close_session_fn = load_agent_backend(agent_backend)
    latency = LatencyTracker(None, ring_size=max(200, turns + 10))

    sink = NullSink()
    session = VoiceSession(
        session_id="bench",
        settings=settings,
        stt=stt,
        tts=tts,
        stream_turn=stream_turn,
        vad=vad,
        sink=sink,
        latency=latency,
        voice=settings.default_voice,
    )
    sink.session = session
    await session.start()

    frame_bytes = settings.audio_sample_rate * settings.audio_frame_ms // 1000 * 2
    silence_frame = b"\x00" * frame_bytes
    trailing_silence_frames = (settings.vad_silence_duration_ms + 200) // settings.audio_frame_ms + 1

    rows: list[dict] = []
    total_turns = 3 + turns
    try:
        for i in range(total_turns):
            clip = pcm_clips[i % len(pcm_clips)]
            sink.listening_event.clear()

            for off in range(0, len(clip), frame_bytes):
                chunk = clip[off : off + frame_bytes]
                if len(chunk) == frame_bytes:
                    await session.feed_audio(chunk)
                    await asyncio.sleep(0)
            for _ in range(trailing_silence_frames):
                await session.feed_audio(silence_frame)
                await asyncio.sleep(0)

            await asyncio.wait_for(sink.listening_event.wait(), timeout=30.0)
            if i == 2:
                # Discard the 3 throwaway warm turns
                latency = LatencyTracker(None, ring_size=max(200, turns + 10))
                session._latency = latency
            elif i >= 3:
                pass

        summary = latency.summary()
        rows = list(latency._ring)
    finally:
        await session.close()
        await close_session_fn("bench")
        await stt.close()
        await tts.close()

    out_path = PROJECT_ROOT / "scripts" / "probe_out" / "benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nVAD silence window: {settings.vad_silence_duration_ms} ms | Measured turns: {len(rows)}")
    if rows:
        first = rows[0]
        print(
            f"First measured turn: e2e_ms={first['e2e_ms']} ms, "
            f"e2e_with_silence_ms={first['e2e_with_silence_ms']} ms"
        )
    print(f"\n{'Metric':<24} {'p50 (ms)':>10} {'p95 (ms)':>10} {'max (ms)':>10}")
    print("-" * 58)
    for k, v in summary["metrics"].items():
        print(f"{k:<24} {v['p50']:>10.1f} {v['p95']:>10.1f} {v['max']:>10.1f}")

    p95_total = summary["metrics"]["e2e_with_silence_ms"]["p95"]
    status = "PASS" if p95_total < 3000.0 else "FAIL"
    print(f"\nTarget p95(e2e_with_silence_ms) < 3000 ms: {status} ({p95_total:.1f} ms)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark voice loop latency")
    parser.add_argument("--turns", type=int, default=30)
    parser.add_argument("--clips-dir", type=Path, default=PROJECT_ROOT / "scripts" / "eval_clips")
    parser.add_argument("--agent", type=str, default="stub")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.turns, args.clips_dir, args.agent))


if __name__ == "__main__":
    main()
