import smtplib
from email.message import EmailMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from app.agent.state import set_pending_action
from app.configs import get_settings
from app.agent.tools.guardrails import validate_email_content, validate_email_recipient, validate_send_permission, validate_email_address
from app.agent.state import (
    get_pending_action,
    get_send_count,
    claim_confirmed_action,
    mark_action_sent,
    mark_action_failed,
)

def send_email(*,recipient: str,subject: str,content: str,confirmed: bool,send_count: int) -> dict:
    """Send a customer email after the action workflow authorizes it.

       The caller must first claim the confirmed pending action
       through session state.

       SMTP failures are treated as uncertain delivery outcomes.
       Do not automatically retry them.
       """

    settings = get_settings()

    validate_send_permission(
        confirmed=confirmed,
        send_count=send_count,
    )

    if not validate_email_recipient(recipient):
        raise ValueError("Invalid email recipient")

    if not validate_email_content(subject, content):
        raise ValueError("Invalid email content")

    sender = settings.gmail_sender
    password = settings.gmail_app_password.get_secret_value()

    if not validate_email_address(sender):
        raise RuntimeError("Invalid sender email configuration")

    if not password:
        raise RuntimeError("Email credentials are missing")

    if settings.dry_run:
        return {
            "success": True,
            "dry_run": True,
            "message": "Email validated but not actually sent.",
        }

    message = EmailMessage()
    message["Subject"] = subject.strip()
    message["From"] = sender.strip()
    message["To"] = recipient.strip()
    message.set_content(content.strip())

    with smtplib.SMTP(
            settings.smtp_host,
            settings.smtp_port,
            timeout=15,
    ) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender, password)
        server.send_message(message)

    return {
        "success": True,
        "dry_run": False,
        "message": "Email sent successfully.",
    }




def execute_pending_email(*, session_id: str, action_id: str, readback_fully_played: bool) -> dict:
    """Execute an email only after its read-back was fully played, and It's matching pending action was confirmed.

    readback_fully_played must be calculated by trusted application
    code using the playback information from the voice pipeline.
    It must never come directly from the LLM.
    """

    if not readback_fully_played:
        raise PermissionError(
            "Email confirmation read-back was not fully played"
        )

    action = get_pending_action(session_id)

    if action is None or action.action_id != action_id:
        raise ValueError("No matching pending email action")

    if action.tool != "email":
        raise ValueError("Pending action is not an email action")

    if not action.confirmed:
        raise PermissionError("Email action has not been confirmed")

    send_count = get_send_count(session_id)

    settings = get_settings()

    claimed_action = claim_confirmed_action(
        session_id=session_id,
        action_id=action_id,
        max_sends=settings.max_user_sends_per_session,
    )

    try:
        result = send_email(
            recipient=claimed_action.recipient,
            subject=claimed_action.subject,
            content=claimed_action.content,
            confirmed=claimed_action.confirmed,
            send_count=send_count,
        )

    except Exception:
        mark_action_failed(session_id, action_id)
        raise

    mark_action_sent(session_id, action_id)

    return {
        **result,
        "action_id": action_id,
        "status": "sent",
    }


@tool
def prepare_email(
    recipient: str,
    subject: str,
    content: str,
    summary: str,
    config: RunnableConfig,
) -> dict:
    """Prepare an email for the customer.

    Use this tool when the customer explicitly requests an email.
    This tool does NOT send the email. It creates a pending action
    that must be read back to the customer and explicitly confirmed.

    Args:
        recipient: Customer's email address.
        subject: Email subject.
        content: Complete email body.
        summary: Concise summary of the email content for confirmation.
    """
    session_id = config.get("configurable", {}).get("session_id")

    if not session_id:
        raise ValueError("Missing session ID in agent configuration")

    if not validate_email_recipient(recipient):
        raise ValueError("Invalid email recipient")

    if not validate_email_content(subject, content):
        raise ValueError("Invalid email subject or content")

    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("Email summary cannot be empty")

    if len(summary) > 1000:
        raise ValueError("Email summary is too long")

    recipient = recipient.strip()
    subject = subject.strip()
    content = content.strip()
    summary = summary.strip()

    try:
        existing = get_pending_action(session_id)
    except KeyError:
        existing = None
    if existing is not None:
        same_action = (
            existing.tool == "email"
            and existing.recipient == recipient
            and existing.subject == subject
            and existing.content == content
            and existing.summary == summary
        )
        if not same_action:
            raise ValueError("An active pending action already exists")
        action = existing
    else:
        action = set_pending_action(
            session_id=session_id,
            tool="email",
            recipient=recipient,
            subject=subject,
            content=content,
            summary=summary,
        )

    return {
        "success": True,
        "status": "pending_confirmation",
        "action_id": action.action_id,
        "tool": action.tool,
        "recipient": action.recipient,
        "subject": action.subject,
        "summary": action.summary,
        "message": (
            "Email prepared successfully. "
            "Read its details to the customer and request confirmation. "
            "Do not claim that the email has been sent."
        ),
    }