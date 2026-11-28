from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

from app.configs import Settings


def test_settings_default_to_openrouter() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "openrouter"
    assert settings.llm_base_url == "https://openrouter.ai/api/v1"


def test_local_server_defaults_use_platform_binary_name() -> None:
    settings = Settings(_env_file=None)
    expected_name = (
        "audiocpp_server.exe" if sys.platform == "win32" else "audiocpp_server"
    )

    assert Path(settings.qwencleo_server_path).name == expected_name
    assert Path(settings.voicetut_server_path).name == expected_name


def test_provider_rejects_another_providers_base_url() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            llm_provider="groq",
            llm_base_url="https://openrouter.ai/api/v1",
        )


def test_chunk_thresholds_are_validated() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_first_min_chars=999)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A@x.com, b@x.com", ["a@x.com", "b@x.com"]),
        ('["a@x.com"]', ["a@x.com"]),
    ],
)
def test_allowlist_parses_supported_formats(raw: str, expected: list[str]) -> None:
    assert Settings(_env_file=None, support_email_allowlist=raw).support_email_allowlist == expected


def test_allowlist_rejects_invalid_email() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, support_email_allowlist="not-an-email")


def test_runtime_key_validation_depends_on_agent_backend() -> None:
    graph_settings = Settings(_env_file=None, agent_backend="graph")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        graph_settings.validate_for_runtime()

    Settings(_env_file=None, agent_backend="stub").validate_for_runtime()


def test_live_sends_report_all_missing_credentials() -> None:
    settings = Settings(_env_file=None, dry_run=False)

    with pytest.raises(ValueError) as exc_info:
        settings.validate_for_runtime()

    assert "GMAIL_SENDER" in str(exc_info.value)
    assert "TELEGRAM_BOT_TOKEN" in str(exc_info.value)
