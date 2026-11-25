import asyncio
import re

from app.agent.stub import close_session, stream_agent_turn
from app.config import get_settings
from app.contracts import (
    AgentChunk,
    AgentRequest,
    MAX_SPEAKABLE_CHARS,
    SessionContext,
)

_MARKDOWN_RE = re.compile(r"(\*\*|__|`|#{1,6}\s)")


async def _collect(req: AgentRequest) -> list[AgentChunk]:
    return [chunk async for chunk in stream_agent_turn(req)]


def _assert_stream_invariants(chunks: list[AgentChunk]) -> None:
    assert len(chunks) >= 1, "Stream must yield at least `done`"
    assert chunks[-1].type == "done", f"Last chunk must be `done`, got {chunks[-1].type}"

    done_count = sum(1 for c in chunks if c.type == "done")
    assert done_count == 1, "`done` must be yielded exactly once"

    for c in chunks:
        if c.type == "text_delta":
            assert c.content.strip(), "`text_delta` must not be empty"
            assert len(c.content) <= MAX_SPEAKABLE_CHARS, (
                f"`text_delta` length {len(c.content)} exceeds {MAX_SPEAKABLE_CHARS}"
            )
            assert not _MARKDOWN_RE.search(c.content), f"Markdown forbidden in TTS text: {c.content!r}"


def test_config_chunk_cap_matches_contract() -> None:
    settings = get_settings()
    assert settings.chunk_max_chars <= MAX_SPEAKABLE_CHARS


def test_normal_turn_invariants_and_timing() -> None:
    sid = "sess-normal"
    chunks = asyncio.run(_collect(AgentRequest(session_id=sid, user_text="عايز أعرف مواعيد العمل")))
    _assert_stream_invariants(chunks)

    text_chunks = [c for c in chunks if c.type == "text_delta"]
    assert len(text_chunks) >= 1
    assert chunks[-1].timing is not None
    assert chunks[-1].timing.total_ms is not None
    asyncio.run(close_session(sid))


def test_empty_or_whitespace_user_text_yields_only_done() -> None:
    sid = "sess-empty"
    for blank in ("", "   ", "\n\t "):
        chunks = asyncio.run(_collect(AgentRequest(session_id=sid, user_text=blank)))
        assert len(chunks) == 1
        assert chunks[0].type == "done"
    asyncio.run(close_session(sid))


def test_confirmation_flow_and_barge_in_reread() -> None:
    sid = "sess-confirm"
    # Step 1: Request email -> must speak read-back BEFORE emitting `confirm`
    chunks = asyncio.run(
        _collect(
            AgentRequest(
                session_id=sid,
                user_text="ابعتلي التفاصيل على الإيميل",
                context=SessionContext(user_email="helmy@example.com"),
            )
        )
    )
    _assert_stream_invariants(chunks)
    types = [c.type for c in chunks]
    assert "confirm" in types
    confirm_idx = types.index("confirm")
    assert "text_delta" in types[:confirm_idx], "`confirm` must be preceded by a spoken `text_delta`"

    confirm_chunk = chunks[confirm_idx]
    assert confirm_chunk.metadata is not None
    for key in ("action_id", "tool", "recipient", "summary"):
        assert key in confirm_chunk.metadata
    assert confirm_chunk.metadata["recipient"] in "".join(
        c.content for c in chunks[:confirm_idx] if c.type == "text_delta"
    )

    # Step 2: Simulate barge-in truncation on the read-back -> saying "yes" must re-read, not send
    reread_chunks = asyncio.run(
        _collect(
            AgentRequest(
                session_id=sid,
                user_text="أيوه",
                truncated_assistant_text="تحب أبعت تفاصيل طلبك على...",
            )
        )
    )
    _assert_stream_invariants(reread_chunks)
    assert "confirm" in [c.type for c in reread_chunks], "Truncated read-back must re-confirm on 'yes'"

    # Step 3: Full read-back completed -> saying "yes" executes action
    exec_chunks = asyncio.run(_collect(AgentRequest(session_id=sid, user_text="أيوه")))
    _assert_stream_invariants(exec_chunks)
    assert "confirm" not in [c.type for c in exec_chunks]
    asyncio.run(close_session(sid))


def test_escalation_order() -> None:
    sid = "sess-escalate"
    chunks = asyncio.run(_collect(AgentRequest(session_id=sid, user_text="عايز أكلم موظف خدمة عملاء")))
    _assert_stream_invariants(chunks)

    types = [c.type for c in chunks]
    assert types == ["text_delta", "escalate", "done"]
    assert chunks[1].metadata is not None and "reason" in chunks[1].metadata
    asyncio.run(close_session(sid))


def test_error_yields_apology_before_error_and_done() -> None:
    sid = "sess-error"
    chunks = asyncio.run(_collect(AgentRequest(session_id=sid, user_text="trigger error")))
    _assert_stream_invariants(chunks)

    types = [c.type for c in chunks]
    assert types == ["text_delta", "error", "done"]
    assert chunks[1].metadata is not None and "code" in chunks[1].metadata
    asyncio.run(close_session(sid))


def test_close_session_is_idempotent_and_discards_pending() -> None:
    sid = "sess-close"
    asyncio.run(
        _collect(
            AgentRequest(
                session_id=sid,
                user_text="ابعت إيميل",
                context=SessionContext(user_email="user@example.com"),
            )
        )
    )
    # Safe to call multiple times
    asyncio.run(close_session(sid))
    asyncio.run(close_session(sid))

    # Pending confirmation was discarded, so "yes" does not execute a send
    after_chunks = asyncio.run(_collect(AgentRequest(session_id=sid, user_text="yes")))
    _assert_stream_invariants(after_chunks)
    assert "tool_status" not in [c.type for c in after_chunks]
    asyncio.run(close_session(sid))