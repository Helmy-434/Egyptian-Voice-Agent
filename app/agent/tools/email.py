import smtplib
from email.message import EmailMessage

from app.configs import get_settings
from app.agent.tools.guardrails import validate_email_content, validate_email_recipient, validate_send_permission, validate_email_address


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