# Arabic AI Voice Agent

Real-time Egyptian Arabic and Modern Standard Arabic (MSA) customer service voice agent (`STT -> LLM -> TTS`) supporting barge-in interruption and guarded Gmail and Telegram tool execution.

## Evaluation vs. Hosting Setup
* **Latency Evaluation (`< 3s` target):** Measured locally on GPU using `QwenCleo-ASR` and `VoiceTut-TTS`.
* **Hosted Demo:** Runs via a lightweight CPU Docker image (`requirements.txt`) with Gemini cloud STT and `DRY_RUN=true`.

For hosted STT, set `STT_PROVIDER=cloud` and provide `GEMINI_API_KEY`. The
Gemini provider defaults to `gemini-3.5-transcribe`; adjust
`GEMINI_STT_MODEL` or `STT_CLOUD_TIMEOUT_SEC` if needed.

## Local QwenCleo Setup

The local `qwencleo` provider uses the existing GGUF model and
`QwenCleo/runtime/audiocpp_server.exe`; it does not download Hugging Face model
weights. Ensure both files exist at the paths configured by
`QWENCLEO_MODEL_PATH` and `QWENCLEO_SERVER_PATH` (their defaults point to the
repository's `QwenCleo` directory), then set `STT_PROVIDER=qwencleo`. At app
startup, the driver starts the server and loads the model once; it stops the
server during app shutdown. CUDA is used by default. Set
`QWENCLEO_ALLOW_CPU=true` to explicitly use the CPU backend.

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