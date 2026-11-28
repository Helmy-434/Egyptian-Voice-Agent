# Arabic AI Voice Agent

A real-time AI-powered customer service voice agent supporting **Egyptian Arabic** and **Modern Standard Arabic (MSA)**.

The application combines speech recognition, conversational AI, and speech synthesis in a streaming pipeline:

**STT → LLM Agent → TTS**

It supports natural multi-turn conversations, voice interruption (barge-in), FAQ-based responses, guarded tool execution, and human escalation through Telegram.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Core Features](#core-features)
3. [System Architecture](#system-architecture)
4. [Conversation Workflow](#conversation-workflow)
5. [Agent and Tool Execution](#agent-and-tool-execution)
6. [Confirmation and Safety](#confirmation-and-safety)
7. [Speech Processing](#speech-processing)
8. [Technology Stack](#technology-stack)
9. [Project Structure](#project-structure)
10. [Requirements](#requirements)
11. [Installation](#installation)
12. [Environment Configuration](#environment-configuration)
13. [Running the Application](#running-the-application)
14. [WebSocket API](#websocket-api)
15. [Knowledge Base](#knowledge-base)
16. [Testing](#testing)
17. [Latency Evaluation](#latency-evaluation)
18. [Docker and Hosted Deployment](#docker-and-hosted-deployment)
19. [Troubleshooting](#troubleshooting)
20. [Limitations and Future Improvements](#limitations-and-future-improvements)

---

## Project Overview

Traditional customer service systems often rely on static menus, predefined responses, or human operators to handle customer requests.

This project introduces a voice-first conversational agent that allows customers to communicate naturally in Arabic.

The agent can:

- Understand spoken Egyptian Arabic and MSA.
- Respond using natural synthesized Arabic speech.
- Answer frequently asked questions using a predefined knowledge base.
- Maintain conversational context across multiple turns.
- Handle customer requests that require clarification.
- Interrupt its response when the user starts speaking.
- Perform selected actions through external tools.
- Request customer confirmation before executing sensitive actions.
- Escalate unresolved issues to a human support team.

The application is designed around a modular architecture so that speech recognition, language models, speech synthesis, and external integrations can be changed independently.

### Project Goals

- Provide a natural Arabic voice customer service experience.
- Reduce response latency through streaming and asynchronous processing.
- Support both local AI models and cloud-based providers.
- Keep external actions controlled and explicitly authorized.
- Maintain a clear separation between conversation orchestration and individual service components.

---

## Core Features

### 1. Arabic Voice Interaction

- Egyptian Arabic and Modern Standard Arabic support.
- Speech-to-text transcription.
- Text-to-speech audio generation.
- Streaming responses instead of waiting for the entire response.
- Male and female voice selection.
- Microphone input through the web client.

### 2. Real-Time Conversation

- Multi-turn conversations.
- Session-based context management.
- Incremental agent response streaming.
- Separate text and audio streaming.
- Voice activity detection.
- Automatic speech endpoint detection.
- Barge-in interruption while the agent is thinking or speaking.

### 3. AI Agent

- LLM-powered conversational reasoning.
- FAQ-based knowledge retrieval.
- Intent-based conversational routing.
- Customer clarification handling.
- Tool selection and execution.
- Human escalation when required.

### 4. Guarded Tool Execution

The agent can interact with external services through controlled tools:

- Gmail/SMTP for customer-facing email communication.
- Telegram Bot API for internal support notifications and escalation.

Sensitive customer-facing actions require explicit confirmation before execution.

### 5. Configurable AI Providers

The application separates provider configuration from the main application logic.

Supported or integrated provider options include:

| Component | Providers |
|---|---|
| STT | QwenCleo, Faster-Whisper, Gemini |
| LLM | OpenRouter, OpenAI, Groq |
| TTS | VoiceTut, Azure Speech, Edge TTS |
| VAD | Silero VAD, Energy-based VAD |

Provider availability depends on the installed dependencies, local model files, credentials, and runtime configuration.

### 6. Monitoring and Evaluation

- Per-stage latency measurements.
- JSONL latency and audit logs.
- STT evaluation scripts.
- TTS latency and real-time factor measurements.
- Automated tests for core application components.

---

## System Architecture

The application follows a modular pipeline architecture.

Each component has a specific responsibility and communicates through shared contracts.

### High-Level Architecture

```text
                  WEB CLIENT
                      |
                      | Microphone Audio
                      v
              FastAPI WebSocket
                      |
                      v
             Voice Orchestrator
                      |
                      v
                 VAD
                      |
              Speech Detected
                      |
                      v
                     STT
                      |
               Transcribed Text
                      |
                      v
                AI AGENT
                      |
             LangGraph Workflow
                      |
           +----------+----------+
           |          |          |
           v          v          v
          FAQ        LLM       Tools
           |          |          |
           +----------+----------+
                      |
              Agent Response
                      |
                      v
                     TTS
                      |
                Audio Chunks
                      |
                      v
                 WebSocket
                      |
                      v
                  WEB CLIENT
```

### Main Components

**FastAPI Application**

Responsible for application startup, dependency initialization, provider warmup, API routing, and resource cleanup.

**Voice Orchestrator**

Coordinates the real-time voice pipeline:

- Receives microphone audio.
- Detects speech and silence.
- Calls the STT engine.
- Sends transcribed text to the agent.
- Streams agent text to TTS.
- Sends generated audio to the client.
- Handles interruption and session lifecycle.

**STT Engine**

Converts incoming audio into text.

**AI Agent**

Processes the user's request, maintains conversational state, retrieves relevant information, and decides whether a tool is needed.

**TTS Engine**

Converts generated response text into audio chunks and streams them to the client.

**Web Client**

Captures microphone input, sends audio to the backend, plays generated audio, and displays conversation events.

---

## Conversation Workflow

A typical voice interaction follows these steps:

1. The client establishes a WebSocket connection.
2. The backend initializes the voice session.
3. The user speaks into the microphone.
4. Audio frames are sent to the backend.
5. VAD detects speech and determines when the utterance has ended.
6. The STT engine transcribes the recorded audio.
7. The transcribed text is submitted to the agent.
8. The agent processes the request using its conversation state and knowledge base.
9. If necessary, the agent requests clarification or prepares a tool action.
10. The response is streamed as text chunks.
11. Each speakable text chunk is passed to TTS.
12. Generated audio chunks are streamed back to the client.
13. The client plays the audio.
14. The session returns to the listening state.

### Barge-In

Barge-in allows the user to interrupt the agent while it is speaking or thinking.

When VAD detects new speech during an active response:

- The current turn is cancelled.
- The active TTS stream is stopped.
- The current agent stream is closed.
- The session returns to listening.
- The new user utterance is processed as the next turn.

This makes the interaction more natural than a traditional request-and-response voice interface.

---

## Agent and Tool Execution

The agent is implemented using LangGraph to organize conversational processing and tool execution.

The agent is responsible for deciding how to handle each customer request.

Possible outcomes include:

- Answering directly.
- Retrieving information from the FAQ knowledge base.
- Asking the customer for clarification.
- Preparing a customer-facing action.
- Escalating the issue to human support.

### LLM Configuration

The LLM provider is configurable through environment variables.

Supported providers:

- OpenRouter
- OpenAI
- Groq

The selected provider and model are configured independently of the rest of the application.

### FAQ Retrieval

The initial knowledge base is a JSON file containing frequently asked questions and their answers.

The agent uses this information to answer supported customer questions without requiring a separate vector database.

### Tool Integrations

**Gmail / SMTP**

Used for customer-facing email communication, such as sending a summary or follow-up after the customer explicitly approves the action.

**Telegram**

Used to notify the internal support team when a request needs human intervention.

Telegram escalation is an internal notification and does not require customer confirmation.

---

## Confirmation and Safety

External actions must follow controlled execution rules.

### Customer-Facing Actions

Before sending an email or performing another confirmation-protected action:

1. The agent prepares the intended action.
2. The application generates a read-back describing the recipient and content.
3. The read-back is delivered to the customer through TTS.
4. The customer is asked to confirm.
5. The application verifies that the corresponding confirmation is valid.
6. The action is executed only after authorization.

A confirmation must be associated with the correct action ID.

A positive response must not authorize an action if the corresponding read-back was not fully played.

If playback was interrupted, the system must not treat the confirmation as completed. The read-back must be repeated before accepting approval.

The backend also verifies confirmation state before executing the action. The LLM cannot independently bypass this check.

### Barge-In and Pending Actions

If the user interrupts the agent:

- Pending unconfirmed actions must not execute.
- Incomplete playback must not count as a completed read-back.
- The session must retain consistent confirmation state.
- Session cleanup must not execute pending actions.

### Human Escalation

When the agent cannot resolve a request, it can send an internal Telegram notification to the configured support chat.

The support destination is controlled by server-side configuration.

---

## Speech Processing

### Speech-to-Text (STT)

The STT layer uses a shared engine interface so that different recognition providers can be selected without changing the voice orchestration logic.

#### QwenCleo-ASR

Local Arabic speech recognition using a GGUF model and the audio.cpp runtime.

- Local inference.
- CUDA acceleration when available.
- Q8_0 quantization.
- Single-worker inference.
- Model initialization during application startup.
- Separate native server process.

#### Faster-Whisper

Local speech recognition using the Faster-Whisper implementation.

- Runs locally without a cloud API.
- Supports CPU inference.
- Uses configurable model size and compute type.
- Uses a single-worker executor to keep inference off the event loop.
- Suitable as a local alternative when cloud STT quotas are unavailable.

The worker rejects concurrent inference requests while a previous inference is still running. A timed-out asynchronous request does not necessarily stop the underlying CPU inference thread immediately.

#### Gemini STT

Cloud-based speech recognition using the Google Gemini API.

- Requires a Gemini API key.
- Uses configurable model and timeout settings.
- Useful for hosted deployments without local ASR model files.

The default model is `gemini-3.5-transcribe`.

Cloud usage is subject to the provider's rate limits and quota restrictions.

### Text-to-Speech (TTS)

The TTS layer uses a shared streaming interface.

#### VoiceTut-TTS

Local Arabic speech synthesis using a GGUF model and audio.cpp.

- GPU-based inference.
- Streaming PCM audio output.
- Configurable male and female speaker IDs.
- Separate runtime process from the STT server.

Default speakers:

- Male: `Mohammed`
- Female: `Yasmin`

#### Azure Speech

Cloud speech synthesis using Azure Speech Services.

#### Edge TTS

Cloud-connected speech synthesis using Edge TTS.

The selected voice is configurable through the application settings.

### Voice Activity Detection

VAD detects speech activity from incoming audio frames.

Supported implementations:

- Silero VAD.
- Energy-based VAD.

VAD is used for speech detection, silence detection, utterance segmentation, and barge-in handling.

---

## Technology Stack

| Area | Technology |
|---|---|
| Backend | Python, FastAPI |
| Communication | WebSockets |
| Agent orchestration | LangGraph |
| Language models | OpenAI, OpenRouter, Groq |
| Local STT | QwenCleo-ASR, Faster-Whisper |
| Cloud STT | Gemini |
| Local TTS | VoiceTut-TTS |
| Cloud TTS | Azure Speech, Edge TTS |
| VAD | Silero VAD, Energy-based VAD |
| Validation and settings | Pydantic |
| Email | SMTP / Gmail |
| Notifications | Telegram Bot API |
| Testing | Pytest |
| Dependency management | uv |
| Deployment | Docker |

---

## Project Structure

```text
arabic-voice-agent/
│
├── .gitignore
├── .env.example
├── requirements.txt
├── requirements-gpu.txt
├── Dockerfile
├── Dockerfile.gpu
├── docker-compose.yml
├── README.md
│
├── app/
│   ├── main.py
│   ├── configs.py
│   ├── contracts.py
│   │
│   ├── api/
│   │   ├── ws_voice.py
│   │   ├── ws_text.py
│   │   └── health.py
│   │
│   ├── pipeline/
│   │   ├── orchestrator.py
│   │   ├── vad.py
│   │   ├── latency.py
│   │   │
│   │   ├── stt/
│   │   │   ├── base.py
│   │   │   ├── qwencleo.py
│   │   │   ├── faster_whisper.py
│   │   │   └── cloud.py
│   │   │
│   │   └── tts/
│   │       ├── base.py
│   │       ├── voicetut.py
│   │       └── cloud.py
│   │
│   ├── agent/
│   │   ├── graph.py
│   │   ├── state.py
│   │   ├── chunker.py
│   │   ├── prompts.py
│   │   ├── kb.py
│   │   │
│   │   └── tools/
│   │       ├── guardrails.py
│   │       ├── email.py
│   │       └── telegram.py
│   │
│   └── data/
│       └── faq_ar.json
│
├── static/
│
├── tests/
│
├── scripts/
│
├── dev/
│
├── QwenCleo/
│   ├── models/
│   ├── runtime/
│   └── voices/
│
└── logs/
```

### Important Directories

| Directory | Responsibility |
|---|---|
| `app/api/` | HTTP and WebSocket endpoints |
| `app/pipeline/` | Audio processing and orchestration |
| `app/pipeline/stt/` | Speech recognition providers |
| `app/pipeline/tts/` | Speech synthesis providers |
| `app/agent/` | Agent workflow, prompts, state, and tools |
| `app/data/` | FAQ knowledge base |
| `static/` | Browser-based voice client |
| `tests/` | Automated tests |
| `scripts/` | Evaluation and benchmarking scripts |
| `dev/` | Development-only stub implementations |
| `QwenCleo/` | Local model and runtime assets |
| `logs/` | Runtime logs and evaluation outputs |

**Note:** The `dev/` stub agent is for development only and is intentionally excluded from the Docker image.

Model files, native runtime binaries, secrets, and generated logs should not be committed to Git.

---

## Requirements

### General Requirements

- Python 3.11 or a compatible Python version supported by the installed dependencies.
- `uv` or pip.
- A Gemini, OpenAI, OpenRouter, or Groq API key depending on the selected provider.
- A modern browser with microphone access for the web client.

### Local GPU Requirements

For QwenCleo-ASR and VoiceTut-TTS:

- NVIDIA GPU.
- Compatible NVIDIA drivers and CUDA runtime.
- Compatible PyTorch installation.
- Required GGUF models.
- Matching audio.cpp runtime binaries.

### CPU Requirements

For Faster-Whisper CPU inference:

- CPU with sufficient available memory.
- Faster-Whisper installed.
- A supported model size and compute type.

The project can use cloud speech providers when local GPU inference is unavailable.

---

## Installation

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd arabic-voice-agent
```

### 2. Create a Virtual Environment

Using `uv`:

```bash
uv sync
```

If the project does not define dependencies in `pyproject.toml`, install from the requirements file instead:

```bash
uv pip install -r requirements.txt
```

For pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Install GPU Dependencies (Optional)

For local GPU-based inference:

```bash
pip install -r requirements-gpu.txt
```

Install a PyTorch build compatible with your Python version, operating system, and CUDA environment using the official instructions:

https://pytorch.org/get-started/locally/

### 4. Configure Environment Variables

Create your local environment file:

```bash
cp .env.example .env
```

Edit `.env` and configure the required API keys, providers, and runtime settings.

Never commit `.env` or expose API keys in source code.

---

## Environment Configuration

The application uses Pydantic settings to load and validate configuration.

The following are important configuration groups. The exact available fields and defaults are defined in `app/configs.py` and `.env.example`.

### Agent Configuration

| Variable | Purpose |
|---|---|
| `AGENT_BACKEND` | Selects the agent implementation |
| `LLM_PROVIDER` | Selects OpenRouter, OpenAI, or Groq |
| `LLM_MODEL` | Model identifier |
| `LLM_BASE_URL` | Optional custom API endpoint |
| `LLM_MAX_OUTPUT_TOKENS` | Maximum generated tokens |
| `LLM_TIMEOUT_SEC` | LLM request timeout |

### STT Configuration

| Variable | Purpose |
|---|---|
| `STT_PROVIDER` | Selects the speech recognition provider |
| `STT_CLOUD_PROVIDER` | Selects the cloud STT provider |
| `GEMINI_API_KEY` | Gemini API credential |
| `GEMINI_STT_MODEL` | Gemini transcription model |
| `STT_TIMEOUT_SEC` | Local STT timeout |
| `STT_CLOUD_TIMEOUT_SEC` | Cloud STT timeout |

For Faster-Whisper:

| Variable | Purpose |
|---|---|
| `FASTER_WHISPER_MODEL` | Model size, such as `base` |
| `FASTER_WHISPER_DEVICE` | `cpu` or `cuda` |
| `FASTER_WHISPER_COMPUTE_TYPE` | Inference precision, such as `int8` |
| `FASTER_WHISPER_CPU_THREADS` | CPU thread count |
| `FASTER_WHISPER_TIMEOUT_SEC` | Local inference timeout |

For QwenCleo:

| Variable | Purpose |
|---|---|
| `QWENCLEO_MODEL_PATH` | Local GGUF model path |
| `QWENCLEO_SERVER_PATH` | audio.cpp executable path |
| `QWENCLEO_SERVER_PORT` | Local STT server port |
| `QWENCLEO_ALLOW_CPU` | Enables CPU backend when supported |

### TTS Configuration

| Variable | Purpose |
|---|---|
| `TTS_PROVIDER` | Selects the TTS implementation |
| `TTS_CLOUD_PROVIDER` | Selects cloud TTS |
| `DEFAULT_VOICE` | Default male or female voice |
| `VOICETUT_MODEL_PATH` | Local VoiceTut model |
| `VOICETUT_SERVER_PATH` | Local audio.cpp executable |
| `VOICETUT_SERVER_PORT` | Local TTS server port |
| `VOICETUT_MALE_VOICE` | Male speaker ID |
| `VOICETUT_FEMALE_VOICE` | Female speaker ID |
| `VOICETUT_ALLOW_CPU` | Enables CPU inference when supported |

### VAD and Audio Configuration

The application also exposes settings for:

- Audio sample rate.
- Audio frame duration.
- Speech probability threshold.
- VAD hysteresis.
- Minimum speech duration.
- Silence duration.
- Pre-roll duration.
- Maximum utterance duration.
- Barge-in minimum speech duration.

These settings control how microphone audio is segmented and when a new conversational turn begins.

### Email and Telegram Configuration

The integrations require server-side credentials.

Typical settings include:

- Gmail sender address.
- Gmail application password.
- Telegram bot token.
- Support Telegram chat ID.

When `DRY_RUN=true`, external actions should operate in the configured dry-run mode rather than performing real external side effects.

For exact variable names and validation requirements, use `.env.example` and `app/configs.py` as the source of truth.

---

## Running the Application

### Start the Backend

From the repository root:

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Use the port configured by your application if it differs from `8000`.

### Check Application Health

Open:

```text
http://localhost:8000/health
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

### Open the Voice Client

If the application serves the static directory through FastAPI, open the root URL:

```text
http://localhost:8000/
```

If the frontend is served separately, use the configured frontend server URL.

The browser must have microphone permission enabled.

---

## WebSocket API

The application exposes two WebSocket interfaces.

### Voice WebSocket

```text
/ws/voice
```

Used for real-time audio conversations.

Responsibilities:

- Receiving PCM audio frames.
- Voice activity detection.
- Speech transcription.
- Agent interaction.
- Streaming synthesized audio.
- Voice selection.
- Barge-in interruption.
- Session management.

The backend sends structured JSON events and binary audio frames.

Common event categories include:

| Event | Purpose |
|---|---|
| `state` | Indicates the current session state |
| `transcript` | Returns recognized user speech |
| `agent_text` | Sends generated assistant text |
| `agent_event` | Sends non-text agent events |
| `tts_chunk_start` | Announces a new audio chunk |
| `tts_chunk_end` | Announces the end of an audio chunk |
| `barge_in` | Indicates that the current turn was interrupted |
| `error` | Reports processing failures |

The client should distinguish between JSON control messages and binary PCM audio messages.

### Text WebSocket

```text
/ws/text
```

Provides a text-only interface to the conversational agent.

It is useful for testing agent behavior independently of microphone input, STT, and TTS.

### Health and Metrics

The health module exposes application health and metrics endpoints.

Check `app/api/health.py` for the exact route names and response formats.

---

## Knowledge Base

The initial FAQ knowledge base is located at:

```text
app/data/faq_ar.json
```

It contains Arabic questions and answers used by the agent to answer common customer requests.

The FAQ loader and validation logic are implemented in:

```text
app/agent/kb.py
```

To add new information:

1. Add the new FAQ entry to the JSON file.
2. Follow the existing schema.
3. Ensure the entry has a valid identifier.
4. Run the relevant tests.
5. Restart or reload the application so the updated knowledge base is loaded.

The initial implementation does not require a separate vector database.

---

## Testing

The project includes automated tests for core application behavior.

### Run the Test Suite

```bash
uv run pytest -q
```

The hermetic test suite does not require a GPU or live API credentials.

### Run Live Tests

```bash
uv run pytest -q -m live
```

Live tests may require:

- API credentials.
- Local model files.
- GPU access.
- External service availability.

### Recommended Testing Areas

- Shared contracts.
- Agent streaming behavior.
- Conversation state management.
- Confirmation guardrails.
- Tool execution authorization.
- Text chunking.
- VAD speech detection.
- Barge-in cancellation.
- STT and TTS interfaces.
- WebSocket session lifecycle.

---

## Latency Evaluation

The application includes latency instrumentation to measure the time spent in different stages of a conversational turn.

The main latency implementation is:

```text
app/pipeline/latency.py
```

It tracks timestamps for important events, including:

- Speech end.
- STT completion.
- First agent response chunk.
- First generated audio chunk.
- Agent completion.

This allows the application to identify which component contributes most to response latency.

### Target

The project targets a response latency below **3 seconds** in the local GPU evaluation environment.

This is an evaluation target, not a guarantee for every provider, utterance, or hardware configuration.

### TTS Benchmark

Run:

```bash
uv run python scripts/tts_probe.py --repeats 3 --soak 0
```

Optional STT benchmarking:

```bash
uv run python scripts/tts_probe.py --repeats 3 --soak 0 --with-stt
```

The benchmark measures relevant performance characteristics such as:

- Time to first audio.
- Total synthesis time.
- Real-time factor (RTF).
- VRAM usage.
- Soak stability.

### STT Evaluation

The project includes scripts for evaluating speech recognition accuracy.

The clips generated by:

```bash
scripts/generate_audio_files.py
```

are synthetic Edge-TTS smoke-test recordings.

They are useful for checking whether the transcription pipeline works, but real Word Error Rate (WER) evaluation requires genuine recordings with reference transcripts.

---

## Docker and Hosted Deployment

The default Docker image is for cloud providers and does not include CUDA, PyTorch, or local model weights. The GPU image builds a Linux CUDA-enabled audio.cpp server for local QwenCleo/VoiceTut inference. Both run Uvicorn with one worker.

### Cloud Quick Start

Set `OPENROUTER_API_KEY` and `GEMINI_API_KEY` in `.env`, then run:

```bash
docker compose up --build
```

This uses cloud STT/TTS providers and energy-based VAD by default. It builds with the lightweight root `Dockerfile` and does not require NVIDIA Container Toolkit.

### Local GPU Quick Start

On Windows, use Docker Desktop with the WSL2 backend and NVIDIA GPU support enabled. Ensure both GGUF models and the reference-speaker WAVs are present under `QwenCleo/`; model downloads are described in [Local QwenCleo Setup](#local-qwencleo-setup). Set `OPENROUTER_API_KEY` in `.env` for the graph agent, then run:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The GPU overlay selects `Dockerfile.gpu`. Ensure both GGUF models and the reference-speaker WAVs are present under `QwenCleo/`; they are mounted at runtime and are not baked into the image.

Check `http://localhost:8000/health`; both `stt.device` and `tts.device` should report `cuda:0`. To directly test the NVIDIA driver from the app container:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml exec voice-agent python -c "from app.pipeline.net import has_cuda_driver; print(has_cuda_driver())"
```

The app reads secrets from `.env`; Compose does not bake that file into the image. Logs are persisted in `./logs`. The GPU container exposes port 8000 and runs with one worker because session state is in-process. PyTorch is CPU-only for Silero VAD; CUDA detection and model inference use the NVIDIA driver and audio.cpp, so `torch.cuda.is_available()` is expected to be false.

### Image Size

The cloud image excludes CUDA and PyTorch; its exact size depends on dependency updates and has not been measured. The GPU image includes the CUDA runtime, Linux audio.cpp server, Python 3.12, and CPU PyTorch; expect a **multi-GB image** (roughly **2–4 GB**, not measured). The local QwenCleo-ASR and VoiceTut-TTS GGUF models are about **3.83 GB (3.56 GiB) combined** (2.48 GB ASR + 1.35 GB TTS); they are mounted from the host, not copied into either image. The host needs enough disk space for both.

Run in the background with the same two Compose files plus `-d`; view logs with `docker compose -f docker-compose.yml -f docker-compose.gpu.yml logs -f`, and stop with the corresponding `down` command. The current RTX 4060 has 8 GB VRAM; both model processes load at startup, so actual VRAM headroom depends on runtime allocations.

The development-only `dev/` stub agent and local model files/native runtimes are intentionally excluded from the image.

---

## Local QwenCleo Setup

The local `qwencleo` provider uses the QwenCleo-ASR GGUF fine-tune, quantized as **Q8_0**, approximately 2.48 GB.

It uses the audio.cpp runtime.

Model files and runtime binaries are excluded from Git. Install them locally for each clone.

### 1. Download the Model

Download:

[QwenCleo-ASR-GGUF](https://huggingface.co/mohammedaly22/QwenCleo-ASR-GGUF)

Place the model at:

```text
QwenCleo/models/qwencleo-asr-q8_0.gguf
```

### 2. Download the Runtime

Download the matching audio.cpp release:

[audio.cpp v0.9.6](https://github.com/0xShug0/audio.cpp/releases/tag/v0.9.6)

The original Windows GPU setup uses:

```text
audio-v0.9.6-bin-windows-x64-cuda12.4.zip
audio-v0.9.6-cudart-windows-x64-cuda12.4.zip
```

Extract both archives into:

```text
QwenCleo/runtime/
```

Ensure the CUDA runtime DLLs are beside:

```text
audiocpp_server.exe
```

**Platform note:** audio.cpp supports Linux, but these downloaded archives are Windows binaries. The Docker GPU setup above builds a Linux CUDA server.

### 3. Configure the Provider

```env
STT_PROVIDER=qwencleo
```

Set the model and server paths if your installation differs from the defaults:

```env
QWENCLEO_MODEL_PATH=QwenCleo/models/qwencleo-asr-q8_0.gguf
QWENCLEO_SERVER_PATH=QwenCleo/runtime/audiocpp_server.exe
QWENCLEO_SERVER_PORT=18761
```

CUDA is used by default.

Set:

```env
QWENCLEO_ALLOW_CPU=true
```

to enable the CPU backend when supported by the installed runtime.

The Python driver uses PyTorch to detect CUDA and select the server backend. Actual model inference runs inside audio.cpp.

### Runtime Behavior

The application:

- Starts the native server as a separate process.
- Loads the model during application startup.
- Runs inference through a single-worker executor.
- Keeps blocking inference work off the asynchronous event loop.
- Rejects concurrent inference while the worker is busy.
- Reports local inference timeouts as `STTError`.
- Stops the native process during application shutdown.

Keeping the native runtime separate helps isolate native runtime failures from the Python application process.

---

## Local Faster-Whisper Setup

Faster-Whisper provides a local STT alternative that does not require a cloud transcription API.

It is particularly useful for CPU-based development and testing.

### 1. Install Faster-Whisper

```bash
uv pip install faster-whisper
```

Or:

```bash
pip install faster-whisper
```

### 2. Configure the Provider

```env
STT_PROVIDER=faster_whisper
```

Example CPU configuration:

```env
FASTER_WHISPER_MODEL=base
FASTER_WHISPER_DEVICE=cpu
FASTER_WHISPER_COMPUTE_TYPE=int8
FASTER_WHISPER_CPU_THREADS=4
FASTER_WHISPER_TIMEOUT_SEC=30
```

The model size can be changed depending on available memory and the desired accuracy/latency trade-off.

### 3. Runtime Behavior

The implementation:

- Loads the Whisper model during warmup.
- Uses a single-worker executor.
- Converts incoming PCM audio into the format expected by Faster-Whisper.
- Performs transcription without blocking the event loop.
- Uses Arabic language configuration.
- Rejects overlapping inference requests while the previous request is still running.

**Important:** An asynchronous timeout does not forcibly terminate a running CPU inference thread. If a transcription times out, the worker may remain busy until the underlying inference actually finishes. A subsequent request can therefore receive a worker-busy error.

This is intentional protection against running concurrent inference on the same model instance.

---

## Local VoiceTut-TTS Setup

VoiceTut-TTS provides local speech synthesis using a GGUF model and the audio.cpp runtime.

### 1. Install GPU Dependencies

Install a compatible PyTorch build:

https://pytorch.org/get-started/locally/

Then install:

```bash
pip install -r requirements-gpu.txt
```

### 2. Configure the Model

The default model path is:

```text
QwenCleo/models/voicetut-tts-q8_0.gguf
```

The default runtime executable is:

```text
QwenCleo/runtime/audiocpp_server.exe
```

Override the paths if necessary:

```env
VOICETUT_MODEL_PATH=QwenCleo/models/voicetut-tts-q8_0.gguf
VOICETUT_SERVER_PATH=QwenCleo/runtime/audiocpp_server.exe
VOICETUT_SERVER_PORT=18762
```

The TTS port is separate from the default QwenCleo STT port (`18761`).

### 3. Configure Voices

Default speakers:

```env
VOICETUT_MALE_VOICE=Mohammed
VOICETUT_FEMALE_VOICE=Yasmin
```

The speaker IDs must be supported by the installed GGUF.

CPU inference is disabled by default. Enable it only when intentionally using a compatible CPU runtime:

```env
VOICETUT_ALLOW_CPU=true
```

### Runtime Behavior

The TTS engine:

- Starts its own audio.cpp server.
- Streams generated PCM audio from its speech endpoint.
- Supports male and female speaker selection.
- Uses configurable first-chunk and total timeouts.
- Runs independently from the STT runtime.

---

## Troubleshooting

### STT: Faster-Whisper Worker Is Still Busy

Example:

```text
stt_failed: Faster-Whisper worker is still busy from an earlier turn.
```

**Cause:**

A previous inference is still running, or the worker has not been released after an error.

A timeout on the asynchronous request does not necessarily stop the underlying thread.

**What to check:**

1. Look for an earlier Faster-Whisper timeout.
2. Check whether the worker eventually finishes.
3. Check the inference start and finish logs.
4. Restart the application if the worker is genuinely stuck.

Do not manually clear the busy flag while inference is running. This could allow multiple inferences to execute concurrently.

### STT: Gemini Quota Exceeded

Example:

```text
RESOURCE_EXHAUSTED
generate_content_free_tier_requests
```

**Cause:**

The configured Gemini model has exceeded its available API quota.

**Solutions:**

- Wait for the quota to reset.
- Check the Gemini API usage and billing configuration.
- Use Faster-Whisper locally.
- Use QwenCleo-ASR when the required local runtime is available.

### STT: Transcription Timeout

**Possible causes:**

- Slow local CPU inference.
- Large audio input.
- Cloud API latency.
- Provider quota or network problems.
- Incorrect timeout configuration.

Check the logs to determine whether the timeout occurred in the STT provider or elsewhere in the pipeline.

### TTS: Audio Is Choppy

**Possible causes:**

- Slow audio generation.
- Network or WebSocket delays.
- Audio chunks arriving faster or slower than the client can play them.
- Incorrect audio sample-rate handling.
- TTS retries or timeouts.

Check the TTS first-audio latency and total generation time.

### QwenCleo or VoiceTut Server Does Not Start

Check:

- The configured model path.
- The native executable path.
- Runtime libraries and dependencies.
- CUDA driver compatibility.
- Whether the configured port is already in use.

The local runtime must be compatible with the operating system.

### Missing API Credentials

Check `.env` and the validation rules in:

```text
app/configs.py
```

Make sure all credentials required by the selected providers and deployment mode are configured.

### WebSocket Disconnects

Check:

- Backend logs.
- Browser console.
- WebSocket connection URL.
- Audio format and sample rate.
- Client microphone permissions.
- Session cleanup and cancellation behavior.

### Audio Is Not Detected

Check the VAD configuration:

- Speech threshold.
- Hysteresis.
- Minimum speech duration.
- Silence duration.
- Audio sample rate.
- Frame size.

Incorrect VAD settings can cause speech to be ignored or utterances to be segmented incorrectly.

---

## Limitations and Future Improvements

Potential future improvements include:

- More robust handling of long-running local inference.
- Better STT accuracy evaluation using real Egyptian Arabic recordings.
- Improved speech playback synchronization.
- More comprehensive playback acknowledgement tests.
- Persistent conversation storage.
- Expanded FAQ knowledge retrieval.
- Better monitoring and performance dashboards.
- Additional Arabic dialect support.
- More deployment options for Linux GPU environments.

---

## Development Notes

### Stub Agent

The `dev/` directory contains a development-only stub agent.

It is useful for testing pipeline behavior independently of the actual LLM agent.

It is not included in the Docker image.

### Contracts

Shared interfaces and data structures are defined in:

```text
app/contracts.py
```

The STT and TTS engine interfaces are defined in their respective `base.py` files.

When adding a new provider, implement the existing interface and register it in the corresponding factory rather than changing the orchestrator unnecessarily.

### Logging

Runtime logs and generated evaluation outputs are stored under:

```text
logs/
```

The directory is excluded from Git.

Use the logs to investigate:

- STT failures.
- TTS failures.
- Agent errors.
- Tool execution.
- Confirmation state.
- Turn latency.

---

## License

Add the project's chosen license before public distribution.

---

## Acknowledgements and References

- [QwenCleo-ASR-GGUF](https://huggingface.co/mohammedaly22/QwenCleo-ASR-GGUF)
- [audio.cpp](https://github.com/0xShug0/audio.cpp)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs)
- [Azure Speech](https://learn.microsoft.com/azure/ai-services/speech-service/)