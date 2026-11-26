import json
from pathlib import Path
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "QwenCleo" / "runtime" / "audiocpp_cli.exe"
MODEL_PATH = PROJECT_ROOT / "QwenCleo" / "models" / "qwencleo-asr-q8_0.gguf"
CLIP_DIR = PROJECT_ROOT / "scripts" / "eval_clips"
OUT_DIR = PROJECT_ROOT / "scripts" / "eval_outputs"


def main() -> None:
    for path in (CLI_PATH, MODEL_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required local QwenCleo file was not found: {path}")

    if not any(CLIP_DIR.glob("*.wav")):
        raise FileNotFoundError(f"No WAV clips found in {CLIP_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        str(CLI_PATH),
        "--task",
        "asr",
        "--family",
        "qwen3_asr",
        "--model",
        str(MODEL_PATH),
        "--backend",
        "cuda",
        "--language",
        "Arabic",
        "--batch-audio-dir",
        str(CLIP_DIR),
        "--out-dir",
        str(OUT_DIR),
        "--metrics",
    ]
    proc = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )

    records: dict[str, dict[str, object]] = {}
    current_id: str | None = None

    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("request_id="):
            current_id = line.split("=", 1)[1]
            records.setdefault(current_id, {"id": current_id})
        elif line.startswith("text_output=") and current_id:
            records[current_id]["transcript"] = line.split("=", 1)[1]
        elif current_id and line.startswith(f"metrics[{current_id}].wall_ms="):
            records[current_id]["wall_ms"] = float(line.split("=", 1)[1])
        elif current_id and line.startswith(f"metrics[{current_id}].audio_duration_ms="):
            records[current_id]["audio_duration_ms"] = int(line.split("=", 1)[1])

    results = list(records.values())
    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    for row in results:
        print(f"[{row['id']}] ({row.get('wall_ms', 0):.1f} ms): {row.get('transcript', '')}")


if __name__ == "__main__":
    main()