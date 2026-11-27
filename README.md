# Arabic AI Voice Agent

Real-time Egyptian Arabic and Modern Standard Arabic (MSA) customer service voice agent (`STT -> LLM -> TTS`) supporting barge-in interruption and guarded Gmail and Telegram tool execution.

## Evaluation vs. Hosting Setup
* **Latency Evaluation (`< 3s` target):** Measured locally on GPU using `QwenCleo-ASR` and `VoiceTut-TTS`.
* **Hosted Demo:** Runs via a lightweight CPU Docker image (`requirements.txt`) with Gemini cloud STT and `DRY_RUN=true`.

For hosted STT, set `STT_PROVIDER=cloud` and provide `GEMINI_API_KEY`. The
Gemini provider defaults to `gemini-3.5-transcribe`; adjust
`GEMINI_STT_MODEL` or `STT_CLOUD_TIMEOUT_SEC` if needed.

## Local QwenCleo Setup

The local `qwencleo` provider uses the `QwenCleo-ASR` GGUF fine-tune, quantized
as **Q8_0** (about 2.48 GB), with the audio.cpp server. Model files and runtime
binaries are intentionally excluded from Git by `.gitignore`; install them
locally for each clone:

1. Download `qwencleo-asr-q8_0.gguf` from the
   [QwenCleo-ASR-GGUF model repository](https://huggingface.co/mohammedaly22/QwenCleo-ASR-GGUF)
   and place it at `QwenCleo/models/qwencleo-asr-q8_0.gguf`.
2. From the
   [audio.cpp v0.9.6 release](https://github.com/0xShug0/audio.cpp/releases/tag/v0.9.6),
   download and extract `audio-v0.9.6-bin-windows-x64-cuda12.4.zip` into
   `QwenCleo/runtime`.
3. Extract the matching
   `audio-v0.9.6-cudart-windows-x64-cuda12.4.zip` into the same
   `QwenCleo/runtime` directory so the CUDA runtime DLLs are beside
   `audiocpp_server.exe`.

For example, after downloading both archives to `Downloads`, run in
PowerShell from the repository root:

```powershell
New-Item -ItemType Directory -Force .\QwenCleo\runtime, .\QwenCleo\models
Expand-Archive "$HOME\Downloads\audio-v0.9.6-bin-windows-x64-cuda12.4.zip" .\QwenCleo\runtime
Expand-Archive "$HOME\Downloads\audio-v0.9.6-cudart-windows-x64-cuda12.4.zip" .\QwenCleo\runtime -Force
```

Ensure the model and `audiocpp_server.exe` exist at the paths configured by
`QWENCLEO_MODEL_PATH` and `QWENCLEO_SERVER_PATH`, then set
`STT_PROVIDER=qwencleo`. CUDA is used by default; set
`QWENCLEO_ALLOW_CPU=true` to use the CPU backend.
The Python driver also needs PyTorch installed to detect CUDA and select the
server backend; inference itself runs in audio.cpp. Install a PyTorch build
compatible with your Python and GPU setup using the
[official PyTorch instructions](https://pytorch.org/get-started/locally/).

The app manages `audiocpp_server.exe` as a separate process, keeping native
runtime crashes isolated from the Python server. Qwen inference runs in a
single-worker executor off the event loop. Concurrent requests fail fast while
an inference is busy, and local inference timeouts are surfaced as `STTError`.
The server loads the model once at startup and is stopped during app shutdown.

## Project Structure

```text
arabic-voice-agent/
├── .gitignore
├── .env.example
├── requirements.txt
├── requirements-gpu.txt         # Local GPU dependencies (voicetut, silero-vad)
├── Dockerfile
├── docker-compose.yml
├── README.md
│
├── app/
│   ├── main.py                  # FastAPI app factory, static mount, lifespan warmup
│   ├── config.py                # Pydantic settings, voice mappings, & tunable thresholds
│   ├── contracts.py             # Shared streaming interface (stream_agent_turn, close_session)
│   │
│   ├── api/
│   │   ├── ws_voice.py          # /ws/voice: PCM audio stream, barge-in, & session lifecycle
│   │   ├── ws_text.py           # /ws/text: text-only WebSocket interface
│   │   └── health.py            # /health & /metrics endpoints
│   │
│   ├── pipeline/
│   │   ├── orchestrator.py      # Per-turn loop: VAD -> STT -> Agent -> TTS
│   │   ├── vad.py               # Silero VAD (end-of-speech & barge-in detection)
│   │   ├── latency.py           # Per-stage latency instrumentation
│   │   ├── stt/
│   │   │   ├── base.py          # STTEngine protocol
│   │   │   ├── qwencleo.py      # Local GPU ASR driver (QwenCleo-ASR)
│   │   │   └── cloud.py         # Cloud ASR driver (Gemini)
│   │   └── tts/
│   │       ├── base.py          # TTSEngine protocol (male / female voice selection)
│   │       ├── voicetut.py      # Local GPU TTS driver (VoiceTut-TTS)
│   │       └── cloud.py         # Cloud TTS driver (Azure / Edge TTS)
│   │
│   ├── agent/
│   │   ├── graph.py             # Conversational routing, confirmation state, & LLM streaming
│   │   ├── state.py             # In-memory per-session state store
│   │   ├── chunker.py           # Prefix route tag parser & speakable clause chunker
│   │   ├── prompts.py           # Egyptian Arabic & MSA customer service prompts
│   │   ├── kb.py                # FAQ loader & ID validator
│   │   └── tools/
│   │       ├── guardrails.py    # Confirmation read-back builder, allowlists, rate limits, & audit log
│   │       ├── email.py         # SMTP/Gmail integration
│   │       └── telegram.py      # Telegram notification & escalation alerts
│   │
│   └── data/
│       └── faq_ar.json          # Customer service FAQ knowledge base
│
├── static/                      # Web client (microphone capture, voice toggle, audio playback)
├── tests/                       # Automated contract, guardrail, chunker, & VAD tests
├── scripts/                     # Latency benchmarking & STT WER evaluation scripts
└── logs/                        # Audit & latency JSONL logs (gitignored)