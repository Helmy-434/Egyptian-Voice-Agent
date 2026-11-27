# Email: Communicate with customers or send summaries and follow-ups.
# Telegram: Notify your internal support team when human intervention is needed.
import json
import urllib.error
import urllib.parse
import urllib.request

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