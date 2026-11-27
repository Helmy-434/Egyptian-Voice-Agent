import smtplib
from email.message import EmailMessage

from app.configs import get_settings
from app.agent.tools.guardrails import (
    validate_email_content,
    validate_email_recipient,
    validate_send_permission,
)


def send_email(*,recipient: str,subject: str,content: str,confirmed: bool,send_count: int) -> dict:
    """Send one email. Call only after state.claim_confirmed_action()."""
    settings = get_settings()

    validate_send_permission(confirmed=confirmed, send_count=send_count)

    if not validate_email_recipient(recipient):
        raise PermissionError("Recipient is not authorized")

    if not validate_email_content(subject, content):
        raise ValueError("Invalid email content")

    if settings.dry_run:
        return {
            "success": True,
            "dry_run": True,
            "message": "Email validated but not actually sent.",
        }

    sender = settings.gmail_sender
    password = settings.gmail_app_password.get_secret_value()

    if not sender or not password:
        raise RuntimeError("Email configuration is incomplete")

    message = EmailMessage()
    message["Subject"] = subject.strip()
    message["From"] = sender
    message["To"] = recipient.strip()
    message.set_content(content)

    # The caller must treat delivery as uncertain on SMTP failures
    # (smtplib.SMTPException, OSError). Do not automatically retry.
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as server:
        server.starttls()
        server.login(sender, password)
        server.send_message(message)

    return {
        "success": True,
        "dry_run": False,
        "message": "Email sent successfully.",
    }