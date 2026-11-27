"""TTS latency, RTF, VRAM, and soak stability probe for VoiceTut-TTS."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import wave

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.configs import get_settings
from app.contracts import MAX_SPEAKABLE_CHARS
from app.pipeline.tts.base import TTSError, Voice, create_tts_engine

SENTENCES: dict[str, str] = {
    "short": "أهلاً بيك، أقدر أساعدك؟",
    "medium": "الأوردر بتاعك اتشحن امبارح وهيوصلك النهاردة بالليل.",
    "long": "للأسف الطلب اتأخر عن الميعاد المتوقع، بس أنا هتابع مع شركة الشحن وأبلغك أول ما يوصل إن شاء الله.",
}
VOICES: tuple[Voice, ...] = ("male", "female")
OUT_DIR = PROJECT_ROOT / "scripts" / "probe_out"
CLIP_PATH = PROJECT_ROOT / "scripts" / "eval_clips" / "clip_001.wav"

_nvidia_smi_warned = False


def gpu_mem_mb() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return None


def log_gpu_mem(label: str) -> int | None:
    global _nvidia_smi_warned
    mem = gpu_mem_mb()
    if mem is None:
        if not _nvidia_smi_warned:
            print("nvidia-smi not available")
            _nvidia_smi_warned = True
    else:
        print(f"GPU memory ({label}): {mem} MB")
    return mem


def save_pcm16_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def load_clip_pcm16(path: Path, expected_rate: int) -> bytes | None:
    if not path.exists():
        return None
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != expected_rate:
            print(
                f"Note: {path} format ({wf.getnchannels()}ch, {wf.getsampwidth()*8}bit, "
                f"{wf.getframerate()}Hz) does not match expected mono PCM16 {expected_rate}Hz; skipping interleave."
            )
            return None
        return wf.readframes(wf.getnframes())


async def run_one_synthesis(
    engine,
    text: str,
    voice: Voice,
    save_wav_path: Path | None = None,
) -> dict[str, float | int | str | bool]:
    t0 = time.perf_counter()
    first: float | None = None
    total_bytes = 0
    chunks = 0
    rate: int | None = None
    pcm_parts: list[bytes] = []

    async for chunk in engine.synthesize_stream(text, voice):
        if first is None:
            first = time.perf_counter() - t0
        rate = chunk.sample_rate
        total_bytes += len(chunk.pcm)
        chunks += 1
        if save_wav_path is not None:
            pcm_parts.append(chunk.pcm)

    total = time.perf_counter() - t0
    if rate is None or total_bytes == 0 or first is None:
        raise TTSError("No audio chunks yielded")

    audio_sec = total_bytes / (2 * rate)
    rtf = total / audio_sec

    if save_wav_path is not None:
        save_pcm16_wav(save_wav_path, b"".join(pcm_parts), rate)

    return {
        "ok": True,
        "ttfa_ms": round(first * 1000.0, 2),
        "total_ms": round(total * 1000.0, 2),
        "audio_sec": round(audio_sec, 3),
        "rtf": round(rtf, 3),
        "chunks": chunks,
        "sample_rate": rate,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="VoiceTut-TTS latency and stability probe.")
    parser.add_argument("--repeats", type=int, default=3, help="Measured repetitions per (sentence, voice).")
    parser.add_argument("--soak", type=int, default=0, help="Extra medium-sentence syntheses alternating voices.")
    parser.add_argument("--with-stt", action="store_true", help="Also warm up QwenCleoSTT and interleave clips.")
    args = parser.parse_args()

    for name, text in SENTENCES.items():
        assert len(text) <= MAX_SPEAKABLE_CHARS, (
            f"Sentence {name!r} exceeds MAX_SPEAKABLE_CHARS ({len(text)} > {MAX_SPEAKABLE_CHARS})"
        )

    settings = get_settings()
    if settings.tts_provider != "voicetut":
        print(
            f"Error: tts_provider is {settings.tts_provider!r}, expected 'voicetut'. "
            "Set TTS_PROVIDER=voicetut in .env or environment and retry."
        )
        return 1

    log_gpu_mem("before engines")

    engine = create_tts_engine(settings)
    stt = None
    clip_pcm: bytes | None = None

    try:
        t_warm = time.perf_counter()
        await engine.warmup()
        print(f"TTS warmup: {(time.perf_counter() - t_warm) * 1000.0:.1f} ms (sample_rate={engine.sample_rate} Hz)")
        log_gpu_mem("after TTS warmup")

        if args.with_stt:
            from app.pipeline.stt.qwencleo import QwenCleoSTT

            stt = QwenCleoSTT(settings)
            t_stt_warm = time.perf_counter()
            await stt.warmup()
            print(f"STT warmup: {(time.perf_counter() - t_stt_warm) * 1000.0:.1f} ms")
            log_gpu_mem("after STT warmup")

            clip_pcm = load_clip_pcm16(CLIP_PATH, settings.audio_sample_rate)
            if clip_pcm is None:
                print(f"Note: {CLIP_PATH} not found or invalid; skipping STT interleaving.")

        # Throwaway synthesis per voice before measuring
        for voice in VOICES:
            try:
                async for _ in engine.synthesize_stream(SENTENCES["short"], voice):
                    pass
            except TTSError as exc:
                print(f"Throwaway warmup ({voice}) failed: {exc}")

        raw_results: list[dict] = []
        summary_rows: list[dict] = []
        saved_wavs: set[tuple[str, str]] = set()

        for sent_name, text in SENTENCES.items():
            for voice in VOICES:
                runs_ok: list[dict] = []
                for rep in range(1, args.repeats + 1):
                    if stt is not None and clip_pcm is not None:
                        await stt.transcribe(clip_pcm, settings.audio_sample_rate)

                    wav_path = (
                        OUT_DIR / f"{voice}_{sent_name}.wav"
                        if (voice, sent_name) not in saved_wavs
                        else None
                    )
                    try:
                        res = await run_one_synthesis(engine, text, voice, save_wav_path=wav_path)
                        if wav_path is not None:
                            saved_wavs.add((voice, sent_name))
                        runs_ok.append(res)
                        raw_results.append(
                            {
                                "sentence": sent_name,
                                "voice": voice,
                                "repeat": rep,
                                "chars": len(text),
                                **res,
                            }
                        )
                    except TTSError as exc:
                        print(f"TTSError [{sent_name}, {voice}, rep {rep}]: {exc}")
                        raw_results.append(
                            {
                                "sentence": sent_name,
                                "voice": voice,
                                "repeat": rep,
                                "chars": len(text),
                                "ok": False,
                                "error": str(exc),
                            }
                        )

                if runs_ok:
                    summary_rows.append(
                        {
                            "sentence": sent_name,
                            "voice": voice,
                            "chars": len(text),
                            "ttfa_ms": statistics.median(r["ttfa_ms"] for r in runs_ok),
                            "total_ms": statistics.median(r["total_ms"] for r in runs_ok),
                            "audio_sec": statistics.median(r["audio_sec"] for r in runs_ok),
                            "rtf": statistics.median(r["rtf"] for r in runs_ok),
                            "chunks": statistics.median(r["chunks"] for r in runs_ok),
                        }
                    )

        log_gpu_mem("after measured runs")

        print()
        header = (
            f"{'sentence':<8} | {'voice':<6} | {'chars':>5} | {'TTFA ms':>8} | "
            f"{'total ms':>8} | {'audio s':>7} | {'RTF':>5} | {'chunks':>6}"
        )
        print(header)
        print("-" * len(header))
        for row in summary_rows:
            print(
                f"{row['sentence']:<8} | {row['voice']:<6} | {row['chars']:>5} | "
                f"{row['ttfa_ms']:>8.1f} | {row['total_ms']:>8.1f} | "
                f"{row['audio_sec']:>7.2f} | {row['rtf']:>5.3f} | {row['chunks']:>6.1f}"
            )

        print()
        for voice in VOICES:
            voice_rows = [r for r in summary_rows if r["voice"] == voice]
            if voice_rows:
                worst = max(r["ttfa_ms"] for r in voice_rows)
                print(f"Summary ({voice}): worst-case median TTFA = {worst:.1f} ms")
            else:
                print(f"Summary ({voice}): no successful runs")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "results.json").write_text(
            json.dumps(raw_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if args.soak > 0:
            print(f"\n--- Soak test ({args.soak} runs, medium sentence, alternating voices) ---")
            soak_ttfas: list[float] = []
            soak_mems: list[int] = []
            soak_failures = 0

            mem_start = gpu_mem_mb()
            if mem_start is not None:
                soak_mems.append(mem_start)

            for i in range(args.soak):
                voice = VOICES[i % len(VOICES)]
                if stt is not None and clip_pcm is not None:
                    await stt.transcribe(clip_pcm, settings.audio_sample_rate)
                try:
                    res = await run_one_synthesis(engine, SENTENCES["medium"], voice)
                    soak_ttfas.append(float(res["ttfa_ms"]))
                except TTSError as exc:
                    soak_failures += 1
                    print(f"Soak TTSError [run {i+1}, {voice}]: {exc}")

                mem_now = gpu_mem_mb()
                if mem_now is not None:
                    soak_mems.append(mem_now)

            mem_end = soak_mems[-1] if soak_mems else None
            if len(soak_ttfas) >= 5:
                first5 = statistics.median(soak_ttfas[:5])
                last5 = statistics.median(soak_ttfas[-5:])
                pct = ((last5 - first5) / first5 * 100.0) if first5 > 0 else 0.0
                print(f"Soak TTFA: first-5 median = {first5:.1f} ms, last-5 median = {last5:.1f} ms ({pct:+.1f}%)")
                if last5 > first5 * 1.25:
                    print(f"WARNING: last-5 median TTFA grew by {pct:.1f}% (> 25% threshold)")
            elif soak_ttfas:
                print(f"Soak TTFA: median = {statistics.median(soak_ttfas):.1f} ms (fewer than 5 successful runs)")
            else:
                print("Soak TTFA: no successful runs")

            if mem_start is not None and mem_end is not None:
                mem_delta = mem_end - mem_start
                print(f"Soak GPU memory: start = {mem_start} MB, end = {mem_end} MB ({mem_delta:+d} MB)")
                if mem_delta > 300:
                    print(f"WARNING: GPU memory grew by {mem_delta} MB (> 300 MB threshold)")

            print(f"Soak TTSError failures: {soak_failures}")

        print("\nInterpretation:")
        print(
            "- RTF < 1.0 means synthesis is faster than playback; RTF > 1.0 means later "
            "sentences can underrun playback and cause audible gaps."
        )
        print("- TTFA of the short sentence is your floor for time-to-first-audio.")
        print(
            "- Check the saved WAVs by ear: voices sound natural and not slowed/sped up, "
            "otherwise _SAMPLE_RATE in voicetut.py is wrong."
        )

    finally:
        await engine.close()
        if stt is not None:
            await stt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
