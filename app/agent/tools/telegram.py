# Email: Communicate with customers or send summaries and follow-ups.
# Telegram: Notify your internal support team when human intervention is needed.
import json
import urllib.error
import urllib.parse
import urllib.request
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from app.agent import state as session_store
from app.configs import get_settings

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"

# note for me to remember : The * means these arguments must be passed by name.
def send_telegram_alert(*,message: str, idempotency_key: str) -> dict:
    """Send an escalation alert to the support chat.

    The chat ID always comes from server configuration, never from the
    customer or the LLM. Call state.mark_escalated() first and send only
    if it returns True; that is what prevents duplicate alerts.
    """
    if not message or not message.strip():
        raise ValueError("Telegram message cannot be empty")

    if not idempotency_key or not idempotency_key.strip():
        raise ValueError("Idempotency key is required")

    settings = get_settings()

    if settings.dry_run:
        return {
            "success": True,
            "dry_run": True,
            "idempotency_key": idempotency_key,
        }

    token = settings.telegram_bot_token.get_secret_value()
    chat_id = settings.support_telegram_chat_id

    if not token or not chat_id:
        raise RuntimeError("Telegram configuration is incomplete")

    payload = urllib.parse.urlencode(
        {"chat_id": chat_id, "text": message}
    ).encode("utf-8")

    request = urllib.request.Request(
        TELEGRAM_API_URL.format(token=token),
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = json.loads(response.read().decode("utf-8"))

    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        raise RuntimeError("Telegram notification failed") from None

    if not body.get("ok"):
        raise RuntimeError("Telegram API rejected the notification")

    return {
        "success": True,
        "dry_run": False,
        "message_id": body.get("result", {}).get("message_id"),
        "idempotency_key": idempotency_key,
    }



@tool
def request_escalation(reason: str, conversation_summary: str, config: RunnableConfig) -> dict:
    """Request human support when the customer needs assistance
    that the AI agent cannot provide.

    Use this tool when:
    - The customer explicitly requests a human agent.
    - The issue requires human intervention.
    - The agent cannot resolve the issue reliably.

    Args:
        reason: Why human intervention is required.
        conversation_summary: A concise summary of the customer's issue.
    """

    session_id = config.get("configurable", {}).get("session_id")

    if not session_id:
        raise ValueError("Missing session ID in agent configuration")

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Escalation reason cannot be empty")

    if not isinstance(conversation_summary, str):
        raise ValueError("Invalid conversation summary")

    reason = reason.strip()
    conversation_summary = conversation_summary.strip()

    if len(reason) > 1000:
        raise ValueError("Escalation reason is too long")

    if not conversation_summary or len(conversation_summary) > 3000:
        raise ValueError("Invalid conversation summary")

    session = session_store.get_session(session_id)

    if session is None:
        raise KeyError("Session not found")

    if session.escalated:
        return {
            "success": True,
            "status": "already_escalated",
            "message": "This session has already been escalated.",
        }

    customer_email = None

    notification = (
        "Human Support Escalation\n\n"
        f"Session ID: {session_id}\n"
        f"Reason: {reason}\n"
        f"Conversation Summary: {conversation_summary}\n"
    )

    if customer_email:
        notification += f"Customer Email: {customer_email}\n"

    idempotency_key = f"escalation:{session_id}"

    try:
        result = send_telegram_alert(
            message=notification,
            idempotency_key=idempotency_key,
        )
    except Exception:
        raise RuntimeError(
            "Unable to notify the support team"
        ) from None

    marked = session_store.mark_escalated(session_id)

    if not marked:
        return {
            "success": True,
            "status": "already_escalated",
            "message": "This session has already been escalated.",
        }

    return {
        "success": True,
        "status": "escalated",
        "message": "The support team has been notified.",
        "session_id": session_id,
        "dry_run": result.get("dry_run", False),
    }