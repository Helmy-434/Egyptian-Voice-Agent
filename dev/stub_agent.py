import re
import time
from collections.abc import AsyncGenerator
from uuid import uuid4
from typing import Any

from app.contracts import (
    MAX_SPEAKABLE_CHARS,
    AgentChunk,
    AgentRequest,
    Timing,
)

_SESSIONS: dict[str, dict] = {}
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_NO_WORDS = {"لأ", "لا", "الغي", "cancel", "no"}
_YES_WORDS = {"أيوه", "ايوه", "اه", "تمام", "yes", "ok"}


def _readback(action: dict[str, Any]) -> tuple[AgentChunk, AgentChunk]:
    recipient = action["recipient"]
    summary = action["summary"]
    return (
        AgentChunk(
            type="text_delta",
            content=f"هبعت {summary} على {recipient}، تمام؟"[:MAX_SPEAKABLE_CHARS],
        ),
        AgentChunk(
            type="confirm",
            metadata={
                "action_id": action["action_id"],
                "tool": "email",
                "recipient": recipient,
                "summary": summary,
            },
        ),
    )


async def stream_agent_turn(request: AgentRequest) -> AsyncGenerator[AgentChunk, None]:
    t0 = time.perf_counter()
    first_chunk_ms: int | None = None

    def done() -> AgentChunk:
        return AgentChunk(
            type="done",
            timing=Timing(
                first_chunk_ms=first_chunk_ms,
                total_ms=int((time.perf_counter() - t0) * 1000),
            ),
        )

    async def emit(chunk: AgentChunk) -> AsyncGenerator[AgentChunk, None]:
        nonlocal first_chunk_ms
        if chunk.type == "text_delta" and first_chunk_ms is None:
            first_chunk_ms = int((time.perf_counter() - t0) * 1000)
        yield chunk

    text = request.user_text.strip()
    if not text:
        yield done()
        return

    if request.event == "idle":
        yield done()
        return

    session = _SESSIONS.get(request.session_id)
    if session is None:
        session = {}
        _SESSIONS[request.session_id] = session

    pending = session.get("pending")
    normalized = text.casefold()
    if pending and normalized in _NO_WORDS:
        session.pop("pending", None)
        async for chunk in emit(AgentChunk(type="text_delta", content="تمام، ما بعتش حاجة")):
            yield chunk
        yield done()
        return

    if pending and normalized in _YES_WORDS:
        if request.truncated_assistant_text is not None:
            readback, confirm = _readback(pending)
            for chunk in (readback, confirm):
                async for emitted in emit(chunk):
                    yield emitted
            yield done()
            return
        yield AgentChunk(type="tool_status", content="sent (dry run)")
        async for chunk in emit(AgentChunk(type="text_delta", content="تمام، اتبعت")):
            yield chunk
        session.pop("pending", None)
        yield done()
        return

    if "trigger error" in normalized:
        async for chunk in emit(AgentChunk(type="text_delta", content="آسف، حصلت مشكلة")):
            yield chunk
        yield AgentChunk(type="error", metadata={"code": "stub_error"})
        yield done()
        return

    if "موظف" in text or "human" in normalized:
        async for chunk in emit(
            AgentChunk(type="text_delta", content="هحولك لموظف خدمة العملاء")
        ):
            yield chunk
        yield AgentChunk(
            type="escalate",
            metadata={"reason": "user_requested_human", "ticket_id": None},
        )
        yield done()
        return

    if any(term in normalized for term in ("إيميل", "ايميل", "email")):
        recipient_match = _EMAIL_RE.search(text)
        context_email = request.context.user_email if request.context else None
        if context_email:
            session["user_email"] = context_email
        recipient = (
            recipient_match.group(0)
            if recipient_match
            else context_email or session.get("user_email")
        )
        if not recipient:
            async for chunk in emit(
                AgentChunk(type="text_delta", content="ممكن تبعتلي عنوان الإيميل؟")
            ):
                yield chunk
            yield done()
            return

        action: dict[str, Any] = {
            "action_id": uuid4().hex,
            "recipient": recipient,
            "summary": "ملخص الطلب",
        }
        session["pending"] = action
        for chunk in _readback(action):
            async for emitted in emit(chunk):
                yield emitted
        yield done()
        return

    async for chunk in emit(AgentChunk(type="text_delta", content="تمام، أنا معاك")):
        yield chunk
    yield done()


async def close_session(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)
