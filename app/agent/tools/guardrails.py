## Guardrails are rules and checks that prevent the AI agent from doing something unsafe, unauthorized, or outside the project's allowed behavior.
import re
from email.utils import parseaddr
from app.configs import get_settings

EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$"
)

MAX_SUBJECT_CHARS = 200
MAX_CONTENT_CHARS = 10_000


def validate_email_address(email: str) -> bool:

    if not email:
        return False

    email = email.strip()
    _, parsed_email = parseaddr(email)

    if parsed_email != email:
        return False

    return bool(EMAIL_PATTERN.fullmatch(parsed_email))


def validate_email_recipient(email: str) -> bool:
    """Validate a customer's email address.

    Authorization and customer confirmation must be handled
    separately by the agent's action workflow.
    """

    return validate_email_address(email)


def validate_email_content(subject: str, content: str) -> bool:

    if not isinstance(subject, str) or not isinstance(content, str):
        return False

    subject = subject.strip()
    content = content.strip()

    if not subject or not content:
        return False

    if len(subject) > MAX_SUBJECT_CHARS or len(content) > MAX_CONTENT_CHARS:
        return False

    # Block header injection through the subject line.
    if "\r" in subject or "\n" in subject:
        return False

    return True


def validate_send_permission(*, confirmed: bool, send_count: int) -> None:
    """Reject sending without confirmation or beyond the session limit."""

    if not confirmed:
        raise PermissionError(
            "Email sending requires explicit confirmation"
        )

    if not isinstance(send_count, int) or isinstance(send_count, bool):
        raise ValueError("Invalid email send count")

    if send_count < 0:
        raise ValueError("Email send count cannot be negative")

    settings = get_settings()

    if send_count >= settings.max_user_sends_per_session:
        raise PermissionError(
            "Email send limit reached for this session"
        )
