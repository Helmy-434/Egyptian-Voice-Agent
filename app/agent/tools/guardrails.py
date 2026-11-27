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

    if not validate_email_address(email):
        return False

    allowed = get_settings().support_email_allowlist

    if not allowed:
        return False

    return email.strip().lower() in allowed


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
    """Raise PermissionError unless to send is confirmed and within limits."""
    if not confirmed:
        raise PermissionError("Email sending requires explicit confirmation")

    if send_count >= get_settings().max_user_sends_per_session:
        raise PermissionError("Email send limit reached for this session")