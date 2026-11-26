import asyncio
import json
from pathlib import Path
import wave

import edge_tts
import librosa
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")
SAMPLE_RATE = 16000
OUT_DIR = Path("scripts/eval_clips")
MANIFEST_PATH = OUT_DIR / "manifest.jsonl"

CLIPS = [
    ("eg_general", "ar-EG-ShakirNeural", "أهلاً بيك، الأوردر بتاعي اتأخر أوي بقالي تلات أيام مستني"),
    ("code_switch", "ar-EG-SalmaNeural", "عايز أعمل cancel للـ order بتاعي لو سمحت"),
    ("control_yes", "ar-EG-ShakirNeural", "أيوه تمام ابعته على الإيميل دلوقتي"),
    ("control_no", "ar-EG-SalmaNeural", "لأ ألغي الموضوع ده مش عايز أبعت حاجة"),
    ("escalation", "ar-EG-ShakirNeural", "حولني لموظف خدمة عملاء حالاً الموضوع مستعجل"),
]


async def synth_to_pcm16_wav(text: str, voice: str, out_wav: Path) -> float:
    tmp_mp3 = out_wav.with_suffix(".tmp.mp3")
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(str(tmp_mp3))

        # Decode MP3 and resample directly to 16 kHz mono float32
        samples, _ = librosa.load(str(tmp_mp3), sr=SAMPLE_RATE, mono=True)
        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")

        with wave.open(str(out_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm16.tobytes())

        return len(pcm16) / SAMPLE_RATE
    finally:
        if tmp_mp3.exists():
            tmp_mp3.unlink()


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []

    for idx, (category, voice, text) in enumerate(CLIPS, start=1):
        fname = f"clip_{idx:03d}.wav"
        fpath = OUT_DIR / fname
        dur = await synth_to_pcm16_wav(text, voice, fpath)
        print(f"Generated {fname} ({dur:.2f}s, {voice}): {text}")
        rows.append(
            json.dumps(
                {
                    "file": fname,
                    "category": category,
                    "voice": voice,
                    "reference": text,
                    "duration_sec": round(dur, 2),
                },
                ensure_ascii=False,
            )
        )

    MANIFEST_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nSaved {len(CLIPS)} clips and {MANIFEST_PATH}.")


if __name__ == "__main__":
    asyncio.run(main())