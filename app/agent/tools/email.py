import smtplib
from email.message import EmailMessage

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
        # Do not automatically retry an uncertain SMTP operation.
        mark_action_failed(session_id, action_id)
        raise

    # 5. Mark successful execution.
    mark_action_sent(session_id, action_id)

    return {
        **result,
        "action_id": action_id,
        "status": "sent",
    }