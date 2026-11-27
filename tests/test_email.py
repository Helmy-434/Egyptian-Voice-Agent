from types import SimpleNamespace
from unittest.mock import Mock
from app.configs import get_settings
import pytest
from langchain_core.runnables import RunnableConfig

import app.agent.tools.email as email_tools


@pytest.fixture
def fake_settings(monkeypatch):
    settings = SimpleNamespace(
        gmail_sender="support@example.com",
        gmail_app_password=SimpleNamespace(
            get_secret_value=lambda: "fake-password"
        ),
        smtp_host="smtp.example.com",
        smtp_port=587,
        dry_run=True,
        max_user_sends_per_session=3,
    )

    monkeypatch.setattr(
        email_tools,
        "get_settings",
        lambda: settings,
    )

    return settings


@pytest.fixture
def tool_config():
    return {
        "configurable": {
            "session_id": "test-session",
        }
    }


def test_prepare_email_success(monkeypatch, tool_config):
    expected_action = SimpleNamespace(
        action_id="action-123",
        tool="email",
        recipient="customer@example.com",
        subject="Support Follow-up",
        content="Your issue has been resolved.",
        summary="A follow-up about your resolved issue.",
    )

    set_action = Mock(return_value=expected_action)

    monkeypatch.setattr(
        email_tools,
        "set_pending_action",
        set_action,
    )

    result = email_tools.prepare_email.invoke(
        {
            "recipient": "customer@example.com",
            "subject": "Support Follow-up",
            "content": "Your issue has been resolved.",
            "summary": "A follow-up about your resolved issue.",
        },
        config=tool_config,
    )

    assert result["success"] is True
    assert result["status"] == "pending_confirmation"
    assert result["action_id"] == "action-123"

    set_action.assert_called_once()


def test_prepare_email_missing_session():
    with pytest.raises(ValueError, match="Missing session ID"):
        email_tools.prepare_email.invoke(
            {
                "recipient": "customer@example.com",
                "subject": "Support",
                "content": "Hello",
                "summary": "A support follow-up.",
            },
            config={},
        )


@pytest.mark.parametrize(
    "recipient",
    [
        "",
        "invalid-email",
        "customer@",
    ],
)
def test_prepare_email_invalid_recipient(
    recipient,
    tool_config,
):
    with pytest.raises(ValueError, match="Invalid email recipient"):
        email_tools.prepare_email.invoke(
            {
                "recipient": recipient,
                "subject": "Support",
                "content": "Hello",
                "summary": "A support follow-up.",
            },
            config=tool_config,
        )


def test_prepare_email_invalid_content(tool_config):
    with pytest.raises(
        ValueError,
        match="Invalid email subject or content",
    ):
        email_tools.prepare_email.invoke(
            {
                "recipient": "customer@example.com",
                "subject": "",
                "content": "Hello",
                "summary": "A support follow-up.",
            },
            config=tool_config,
        )


def test_prepare_email_empty_summary(tool_config):
    with pytest.raises(
        ValueError,
        match="Email summary cannot be empty",
    ):
        email_tools.prepare_email.invoke(
            {
                "recipient": "customer@example.com",
                "subject": "Support",
                "content": "Hello",
                "summary": " ",
            },
            config=tool_config,
        )


def test_send_email_dry_run(fake_settings):
    result = email_tools.send_email(
        recipient="customer@example.com",
        subject="Support",
        content="Your issue has been resolved.",
        confirmed=True,
        send_count=0,
    )

    assert result["success"] is True
    assert result["dry_run"] is True


def test_send_email_requires_confirmation(fake_settings):
    with pytest.raises(PermissionError):
        email_tools.send_email(
            recipient="customer@example.com",
            subject="Support",
            content="Hello",
            confirmed=False,
            send_count=0,
        )


def test_execute_pending_email_success(
    monkeypatch,
    fake_settings,
):
    action = SimpleNamespace(
        action_id="action-123",
        tool="email",
        recipient="customer@example.com",
        subject="Support",
        content="Your issue has been resolved.",
        confirmed=True,
    )

    claimed_action = action

    monkeypatch.setattr(
        email_tools,
        "get_pending_action",
        lambda session_id: action,
    )

    monkeypatch.setattr(
        email_tools,
        "get_send_count",
        lambda session_id: 0,
    )

    monkeypatch.setattr(
        email_tools,
        "claim_confirmed_action",
        lambda **kwargs: claimed_action,
    )

    send_mock = Mock(
        return_value={
            "success": True,
            "dry_run": True,
            "message": "Email validated.",
        }
    )

    monkeypatch.setattr(
        email_tools,
        "send_email",
        send_mock,
    )

    mark_sent = Mock()

    monkeypatch.setattr(
        email_tools,
        "mark_action_sent",
        mark_sent,
    )

    result = email_tools.execute_pending_email(
        session_id="test-session",
        action_id="action-123",
        readback_fully_played=True,
    )

    assert result["success"] is True
    assert result["status"] == "sent"
    assert result["action_id"] == "action-123"

    send_mock.assert_called_once()
    mark_sent.assert_called_once_with(
        "test-session",
        "action-123",
    )


def test_execute_email_rejects_unplayed_readback():
    with pytest.raises(
        PermissionError,
        match="read-back was not fully played",
    ):
        email_tools.execute_pending_email(
            session_id="test-session",
            action_id="action-123",
            readback_fully_played=False,
        )


def test_execute_email_rejects_unconfirmed_action(
    monkeypatch,
):
    action = SimpleNamespace(
        action_id="action-123",
        tool="email",
        confirmed=False,
    )

    monkeypatch.setattr(
        email_tools,
        "get_pending_action",
        lambda session_id: action,
    )

    with pytest.raises(
        PermissionError,
        match="not been confirmed",
    ):
        email_tools.execute_pending_email(
            session_id="test-session",
            action_id="action-123",
            readback_fully_played=True,
        )


def test_execute_email_rejects_wrong_action_id(
    monkeypatch,
):
    action = SimpleNamespace(
        action_id="different-action",
        tool="email",
        confirmed=True,
    )

    monkeypatch.setattr(
        email_tools,
        "get_pending_action",
        lambda session_id: action,
    )

    with pytest.raises(
        ValueError,
        match="No matching pending email action",
    ):
        email_tools.execute_pending_email(
            session_id="test-session",
            action_id="action-123",
            readback_fully_played=True,
        )


def test_execute_email_marks_failure(
    monkeypatch,
    fake_settings,
):
    action = SimpleNamespace(
        action_id="action-123",
        tool="email",
        recipient="customer@example.com",
        subject="Support",
        content="Hello",
        confirmed=True,
    )

    monkeypatch.setattr(
        email_tools,
        "get_pending_action",
        lambda session_id: action,
    )
    monkeypatch.setattr(
        email_tools,
        "get_send_count",
        lambda session_id: 0,
    )
    monkeypatch.setattr(
        email_tools,
        "claim_confirmed_action",
        lambda **kwargs: action,
    )

    monkeypatch.setattr(
        email_tools,
        "send_email",
        Mock(side_effect=RuntimeError("SMTP failed")),
    )

    mark_failed = Mock()

    monkeypatch.setattr(
        email_tools,
        "mark_action_failed",
        mark_failed,
    )

    with pytest.raises(RuntimeError, match="SMTP failed"):
        email_tools.execute_pending_email(
            session_id="test-session",
            action_id="action-123",
            readback_fully_played=True,
        )

    mark_failed.assert_called_once_with(
        "test-session",
        "action-123",
    )