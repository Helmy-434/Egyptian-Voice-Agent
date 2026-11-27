from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import app.agent.tools.telegram as telegram_tools


@pytest.fixture
def fake_settings(monkeypatch):
    settings = SimpleNamespace(
        telegram_bot_token=SimpleNamespace(
            get_secret_value=lambda: "fake-token"
        ),
        support_telegram_chat_id="123456",
        dry_run=True,
    )

    monkeypatch.setattr(
        telegram_tools,
        "get_settings",
        lambda: settings,
    )

    return settings


def test_send_telegram_alert_dry_run(fake_settings):
    result = telegram_tools.send_telegram_alert(
        message="A customer needs human support.",
        idempotency_key="escalation:test-session",
    )

    assert result["success"] is True
    assert result["dry_run"] is True


def test_send_telegram_alert_empty_message():
    with pytest.raises(ValueError, match="cannot be empty"):
        telegram_tools.send_telegram_alert(
            message=" ",
            idempotency_key="test-key",
        )


def test_send_telegram_alert_missing_idempotency_key():
    with pytest.raises(ValueError, match="Idempotency key"):
        telegram_tools.send_telegram_alert(
            message="Escalation required",
            idempotency_key=" ",
        )


def test_send_telegram_alert_missing_configuration(
    monkeypatch,
):
    settings = SimpleNamespace(
        telegram_bot_token=SimpleNamespace(
            get_secret_value=lambda: ""
        ),
        support_telegram_chat_id="123456",
        dry_run=False,
    )

    monkeypatch.setattr(
        telegram_tools,
        "get_settings",
        lambda: settings,
    )

    with pytest.raises(
        RuntimeError,
        match="configuration is incomplete",
    ):
        telegram_tools.send_telegram_alert(
            message="Escalation required",
            idempotency_key="test-key",
        )


def test_send_telegram_alert_api_success(monkeypatch):
    settings = SimpleNamespace(
        telegram_bot_token=SimpleNamespace(
            get_secret_value=lambda: "fake-token"
        ),
        support_telegram_chat_id="123456",
        dry_run=False,
    )

    monkeypatch.setattr(
        telegram_tools,
        "get_settings",
        lambda: settings,
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return (
                b'{"ok": true, "result": {"message_id": 55}}'
            )

    urlopen_mock = Mock(return_value=FakeResponse())

    monkeypatch.setattr(
        telegram_tools.urllib.request,
        "urlopen",
        urlopen_mock,
    )

    result = telegram_tools.send_telegram_alert(
        message="Escalation required",
        idempotency_key="escalation:test-session",
    )

    assert result["success"] is True
    assert result["message_id"] == 55
    assert result["dry_run"] is False

    request = urlopen_mock.call_args.args[0]

    assert "fake-token" in request.full_url
    assert request.get_method() == "POST"


def test_request_escalation_success(
    monkeypatch,
    fake_settings,
):
    session = SimpleNamespace(escalated=False)

    monkeypatch.setattr(
        telegram_tools.session_store,
        "get_session",
        lambda session_id: session,
    )

    mark_escalated = Mock(return_value=True)

    monkeypatch.setattr(
        telegram_tools.session_store,
        "mark_escalated",
        mark_escalated,
    )

    send_alert = Mock(
        return_value={
            "success": True,
            "dry_run": True,
        }
    )

    monkeypatch.setattr(
        telegram_tools,
        "send_telegram_alert",
        send_alert,
    )

    result = telegram_tools.request_escalation.invoke(
        {
            "reason": "Customer requested a human agent.",
            "conversation_summary": "The customer needs technical assistance.",
        },
        config={
            "configurable": {
                "session_id": "test-session",
            }
        },
    )

    assert result["success"] is True
    assert result["status"] == "escalated"

    send_alert.assert_called_once()
    mark_escalated.assert_called_once_with("test-session")


def test_request_escalation_missing_session():
    with pytest.raises(ValueError, match="Missing session ID"):
        telegram_tools.request_escalation.invoke(
            {
                "reason": "Needs human support",
                "conversation_summary": "Customer has an unresolved issue.",
            },
            config={},
        )


def test_request_escalation_empty_reason():
    with pytest.raises(ValueError, match="reason cannot be empty"):
        telegram_tools.request_escalation.invoke(
            {
                "reason": " ",
                "conversation_summary": "Customer needs assistance.",
            },
            config={
                "configurable": {
                    "session_id": "test-session",
                }
            },
        )


def test_request_escalation_unknown_session(
    monkeypatch,
):
    monkeypatch.setattr(
        telegram_tools.session_store,
        "get_session",
        lambda session_id: None,
    )

    with pytest.raises(KeyError, match="Session not found"):
        telegram_tools.request_escalation.invoke(
            {
                "reason": "Needs human support",
                "conversation_summary": "Customer needs assistance.",
            },
            config={
                "configurable": {
                    "session_id": "test-session",
                }
            },
        )


def test_request_escalation_already_escalated(
    monkeypatch,
):
    session = SimpleNamespace(escalated=True)

    monkeypatch.setattr(
        telegram_tools.session_store,
        "get_session",
        lambda session_id: session,
    )

    send_alert = Mock()

    monkeypatch.setattr(
        telegram_tools,
        "send_telegram_alert",
        send_alert,
    )

    result = telegram_tools.request_escalation.invoke(
        {
            "reason": "Needs human support",
            "conversation_summary": "Customer needs assistance.",
        },
        config={
            "configurable": {
                "session_id": "test-session",
            }
        },
    )

    assert result["status"] == "already_escalated"
    send_alert.assert_not_called()