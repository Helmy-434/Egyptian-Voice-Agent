from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Annotated, Literal, Optional
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.contracts import MAX_SPEAKABLE_CHARS

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PROVIDER_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
_PROVIDER_HOSTS = {
    "groq": "groq.com",
    "openai": "openai.com",
    "openrouter": "openrouter.ai",
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE_PATH = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    
    model_config = SettingsConfigDict(env_file=ENV_FILE_PATH, extra="ignore")

    # --- Runtime Mode & Providers ---
    agent_backend: Literal["stub", "graph"] = "stub"
    stt_provider: Literal["qwencleo", "cloud"] = "qwencleo"
    stt_cloud_provider: Literal["gemini"] = "gemini"
    tts_provider: Literal["voicetut", "cloud"] = "voicetut"
    tts_cloud_provider: Literal["azure", "edge"] = "edge"

    # --- Voice Selection & Provider Speaker Mapping ---
    default_voice: Literal["male", "female"] = "female"
    voicetut_male_voice: str = "Mohammed"
    voicetut_female_voice: str = "Yasmin"
    azure_male_voice: str = "ar-EG-ShakirNeural"
    azure_female_voice: str = "ar-EG-SalmaNeural"
    edge_male_voice: str = "ar-EG-ShakirNeural"
    edge_female_voice: str = "ar-EG-SalmaNeural"

    # --- Audio Transport & Session Limits (Silero VAD requires 512@16kHz or 256@8kHz = 32ms) ---
    audio_sample_rate: Literal[8000, 16000] = 16000
    audio_frame_ms: int = 32
    max_session_duration_sec: int = 600

    # --- LLM Configuration ---
    llm_provider: Literal["groq", "openai", "openrouter"] = "openrouter"
    llm_base_url: Optional[str] = None
    # Verify this slug on openrouter.ai/models before relying on it.
    llm_model: str = "meta-llama/llama-3.3-70b-instruct"
    llm_max_output_tokens: int = 180
    llm_timeout_sec: float = 8.0

    # --- Secrets ---
    groq_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    openrouter_api_key: SecretStr = SecretStr("")
    azure_speech_key: SecretStr = SecretStr("")
    azure_speech_region: str = "eastus"
    gemini_api_key: SecretStr = SecretStr("")

    # --- Email (SMTP) & Telegram ---
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    gmail_sender: str = ""
    gmail_app_password: SecretStr = SecretStr("")
    support_email_allowlist: Annotated[list[str], NoDecode] = Field(default_factory=list)

    telegram_bot_token: SecretStr = SecretStr("")
    support_telegram_chat_id: str = ""

    # --- VAD, Barge-In, & Idle Timer ---
    vad_speech_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_silence_duration_ms: int = 450
    barge_in_min_speech_ms: int = 300
    stt_min_confidence: float = 0.45
    enable_idle_offer: bool = False
    idle_timeout_sec: float = 15.0

    # --- Chunker Tuning ---
    chunk_first_min_chars: int = 25
    chunk_next_min_chars: int = 60
    chunk_max_chars: int = MAX_SPEAKABLE_CHARS

    # --- Guardrails & Logging ---
    dry_run: bool = True
    max_user_sends_per_session: int = 3
    audit_log_path: str = "logs/audit.jsonl"
    latency_log_path: str = "logs/latency.jsonl"
    
    # --- Local STT Tuning ---
    qwencleo_model_path: str = "QwenCleo/models/qwencleo-asr-q8_0.gguf"
    qwencleo_server_path: str = "QwenCleo/runtime/audiocpp_server.exe"
    qwencleo_server_port: int = Field(default=18761, ge=1, le=65535)
    qwencleo_server_startup_timeout_sec: float = Field(default=120.0, gt=0)
    qwencleo_allow_cpu: bool = False
    qwencleo_warmup_clip_path: str = ""
    stt_timeout_sec: float = 1.2                # Base timeout; calibrate just above measured p95
    stt_timeout_per_audio_sec: float = 0.15     # Extra allowance per second of speech
    gemini_stt_model: str = "gemini-3.5-transcribe"
    stt_cloud_timeout_sec: float = 15.0

    # --- Local TTS Tuning ---
    voicetut_model_path: str = "QwenCleo/models/voicetut-tts-q8_0.gguf"
    voicetut_server_path: str = "QwenCleo/runtime/audiocpp_server.exe"
    voicetut_server_port: int = Field(default=18762, ge=1, le=65535)
    voicetut_server_startup_timeout_sec: float = Field(default=120.0, gt=0)
    voicetut_allow_cpu: bool = False
    tts_first_chunk_timeout_sec: float = Field(default=3.5, gt=0)
    tts_total_timeout_sec: float = Field(default=5.0, gt=0)

    @field_validator(
        "groq_api_key",
        "openai_api_key",
        "openrouter_api_key",
        "azure_speech_key",
        "gemini_api_key",
        "gmail_app_password",
        "telegram_bot_token",
        "gmail_sender",
        "support_telegram_chat_id",
        mode="before",
    )
    @classmethod
    def strip_credentials(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, SecretStr):
            return v.get_secret_value().strip()
        return v

    @field_validator("support_email_allowlist", mode="before")
    @classmethod
    def parse_allowlist(cls, v: object) -> list[str]:
        if isinstance(v, str):
            raw = v.strip()
            if not raw:
                return []
            items = json.loads(raw) if raw.startswith("[") else raw.split(",")
        elif v is None:
            return []
        else:
            items = list(v)

        cleaned: list[str] = []
        for item in items:
            if not isinstance(item, str):
                raise ValueError(f"support_email_allowlist entries must be strings, got {type(item).__name__}")
            email = item.strip().lower()
            if not email:
                continue
            if not _EMAIL_RE.match(email):
                raise ValueError(f"Invalid email in support_email_allowlist: {email!r}")
            cleaned.append(email)
        return cleaned

    @model_validator(mode="after")
    def validate_structure(self) -> "Settings":
        """Pure structural checks safe to run during tests, evals, and stub runs."""
        if (self.audio_sample_rate, self.audio_frame_ms) not in {(16000, 32), (8000, 32)}:
            raise ValueError(
                f"Unsupported (audio_sample_rate={self.audio_sample_rate}, audio_frame_ms={self.audio_frame_ms}); "
                "Silero VAD requires (16000, 32) [512 samples] or (8000, 32) [256 samples]"
            )

        if self.chunk_max_chars > MAX_SPEAKABLE_CHARS:
            raise ValueError(f"chunk_max_chars exceeds MAX_SPEAKABLE_CHARS ({MAX_SPEAKABLE_CHARS})")
        if not (0 < self.chunk_first_min_chars <= self.chunk_next_min_chars <= self.chunk_max_chars):
            raise ValueError("Require 0 < chunk_first_min_chars <= chunk_next_min_chars <= chunk_max_chars")

        if not self.llm_base_url or not self.llm_base_url.strip():
            self.llm_base_url = _PROVIDER_URLS[self.llm_provider]
        else:
            host = (urlparse(self.llm_base_url.strip()).hostname or "").lower()
            for provider, provider_host in _PROVIDER_HOSTS.items():
                if provider == self.llm_provider:
                    continue
                if host == provider_host or host.endswith("." + provider_host):
                    raise ValueError(
                        f"llm_provider is {self.llm_provider!r} but llm_base_url points to {provider!r}"
                    )

        return self

    def validate_for_runtime(self) -> None:
        """Called in FastAPI lifespan startup; checks secrets for active backends only."""
        groq_key = self.groq_api_key.get_secret_value()
        openai_key = self.openai_api_key.get_secret_value()
        openrouter_key = self.openrouter_api_key.get_secret_value()
        azure_key = self.azure_speech_key.get_secret_value()

        if self.agent_backend == "graph":
            if self.llm_provider == "groq" and not groq_key:
                raise ValueError("GROQ_API_KEY is required when agent_backend='graph' and llm_provider='groq'")
            if self.llm_provider == "openai" and not openai_key:
                raise ValueError("OPENAI_API_KEY is required when agent_backend='graph' and llm_provider='openai'")
            if self.llm_provider == "openrouter" and not openrouter_key:
                raise ValueError(
                    "OPENROUTER_API_KEY is required when agent_backend='graph' and llm_provider='openrouter'"
                )

        if self.stt_provider == "cloud":
            if self.stt_cloud_provider == "gemini" and not self.gemini_api_key.get_secret_value():
                raise ValueError("GEMINI_API_KEY is required for cloud Gemini STT")

        if self.tts_provider == "cloud" and self.tts_cloud_provider == "azure" and not azure_key:
            raise ValueError("AZURE_SPEECH_KEY is required for cloud Azure TTS")

        if not self.dry_run:
            missing = [
                name for name, val in [
                    ("GMAIL_SENDER", self.gmail_sender),
                    ("GMAIL_APP_PASSWORD", self.gmail_app_password.get_secret_value()),
                    ("TELEGRAM_BOT_TOKEN", self.telegram_bot_token.get_secret_value()),
                    ("SUPPORT_TELEGRAM_CHAT_ID", self.support_telegram_chat_id),
                ] if not val
            ]
            if missing:
                raise ValueError(f"DRY_RUN=false requires: {', '.join(missing)}")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()