"""Latency, RTF, and VRAM benchmark probe for VoiceTut-TTS across sentence lengths and voices."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import time
import wave

sys.stdout.reconfigure(encoding="utf-8")

from app.configs import get_settings
from app.pipeline.tts.base import Voice, create_tts_engine

OUT_DIR = Path("scripts/eval_outputs")

PROBE_CASES: list[tuple[str, str]] = [
    # Short (~2-4 words)
    ("short", "أهلاً بيك يا فندم."),
    ("short", "تمام، فهمت حضرتك."),
    ("short", "ثانية واحدة معايا."),
    # Medium (~8-12 words, Egyptian + English code-switching)
    ("medium", "الأوردر بتاعك اتشحن النهاردة وهيوصل إسكندرية بكرة الصبح."),
    ("medium", "ممكن تبعتلي رقم الـ tracking على الـ WhatsApp لو سمحت؟"),
    ("medium", "لو حابب تعمل cancel للطلب نقدر ننفذه حالاً."),
    ("medium", "مصاريف الشحن خمسين جنيه والدفع متاح عند الاستلام."),
    # Long (~18-24 words, near speakable-chunk cap)
    (
        "long",
        "أهلاً بيك في خدمة العملاء، تقدر تتابع حالة الشحنة من خلال الـ app "
        "أو تكلمنا في أي وقت لو احتجت تعدل عنوان التوصيل.",
    ),
    (
        "long",
        "للأسف المنتج ده خلص من المخزن حالياً، بس هيتوفر تاني يوم الحد الجاي "
        "ونقدر نبعتلك notification أول ما يرجع.",
    ),
    (
        "long",
        "تم تسجيل طلب الاسترجاع بنجاح والمندوب هيتواصل معاك خلال أربعة وعشرين ساعة "
        "عشان يستلم الطرد ويرجعلك المبلغ كامل.",
    ),
]


def save_pcm16_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


async def main() -> None:
    settings = get_settings()
    engine = create_tts_engine(settings)

    print("Warming up VoiceTut-TTS (excluded from benchmark timings)...")
    t_warm = time.perf_counter()
    await engine.warmup()
    print(f"Warmup finished in {(time.perf_counter() - t_warm) * 1000:.1f} ms (rate={engine.sample_rate} Hz)\n")

    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        torch = None  # type: ignore[assignment]
        has_cuda = False

    header = (
        f"{'Tier':<7} | {'Voice':<6} | {'Chars':>5} | {'TTFA (ms)':>9} | "
        f"{'Total (ms)':>10} | {'Audio (ms)':>10} | {'RTF':>6} | {'Peak VRAM (MB)':>14}"
    )
    print(header)
    print("-" * len(header))

    saved_voices: set[Voice] = set()

    try:
        for tier, sentence in PROBE_CASES:
            for voice in ("male", "female"):
                if has_cuda and torch is not None:
                    torch.cuda.reset_peak_memory_stats()

                t0 = time.perf_counter()
                ttfa_ms: float | None = None
                pcm_parts: list[bytes] = []

                async for chunk in engine.synthesize_stream(sentence, voice):  # type: ignore[arg-type]
                    if ttfa_ms is None:
                        ttfa_ms = (time.perf_counter() - t0) * 1000.0
                    pcm_parts.append(chunk.pcm)

                total_ms = (time.perf_counter() - t0) * 1000.0
                full_pcm = b"".join(pcm_parts)
                audio_ms = (len(full_pcm) / (engine.sample_rate * 2)) * 1000.0
                rtf = (total_ms / audio_ms) if audio_ms > 0 else 0.0
                peak_mb = (
                    torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                    if has_cuda and torch is not None
                    else 0.0
                )

                print(
                    f"{tier:<7} | {voice:<6} | {len(sentence):>5} | "
                    f"{(ttfa_ms or 0.0):>9.1f} | {total_ms:>10.1f} | "
                    f"{audio_ms:>10.1f} | {rtf:>6.2f} | {peak_mb:>14.1f}"
                )

                if voice not in saved_voices and full_pcm:
                    out_wav = OUT_DIR / f"probe_{voice}.wav"
                    save_pcm16_wav(out_wav, full_pcm, engine.sample_rate)
                    saved_voices.add(voice)  # type: ignore[arg-type]
    finally:
        await engine.close()

    print(f"\nSaved listening samples to {OUT_DIR / 'probe_male.wav'} and {OUT_DIR / 'probe_female.wav'}.")


if __name__ == "__main__":
    asyncio.run(main())