from typing import AsyncGenerator, Literal, Optional

from pydantic import BaseModel

# Max length of one speakable chunk. The agent splits at sentence/clause
# boundaries; the pipeline's chunker may rely on this cap.
MAX_SPEAKABLE_CHARS = 200


# ---------------------------------------------------------------------------
# Request side: Pipeline -> Agent
# ---------------------------------------------------------------------------

class SessionContext(BaseModel):
    """Optional, sent on the first turn only.

    All values are UNVERIFIED hints. The agent must still read back and get a
    spoken confirmation before using any address.

    The support Telegram chat id is NOT here: it is server config
    (allowlist) and never comes from the client.
    """
    user_email: Optional[str] = None


class AgentRequest(BaseModel):
    session_id: str

    user_text: str
    # STT transcript. Empty/whitespace-only transcripts (VAD noise) are
    # dropped by the pipeline before calling the agent. If one arrives anyway,
    # the agent yields only `done` and changes NO state: no history entry,
    # no counters, no effect on pending actions.

    stt_confidence: Optional[float] = None
    # 0-1. A non-empty but low-confidence transcript should make the agent
    # ask the user to repeat rather than guess.

    truncated_assistant_text: Optional[str] = None
    # None -> the previous reply was fully played.
    # str  -> the text ACTUALLY played before barge-in (pipeline owns this
    #         truth, since the agent only knows what it yielded, and playback
    #         lags generation). The agent records this in history instead of
    #         what it yielded. If absent after a cancellation, the agent falls
    #         back to chunks it fully yielded (degraded default).
    #         If a confirmation was pending and its read-back was truncated,
    #         the agent must re-read it before accepting a "yes".

    event: Literal["user_turn", "idle"] = "user_turn"
    # "idle": no user speech (silence timeout). The agent may make the one-time
    # summary offer, otherwise yields only `done`. Never mutates history.

    context: Optional[SessionContext] = None


# ---------------------------------------------------------------------------
# Response side: Agent -> Pipeline
# ---------------------------------------------------------------------------

class Timing(BaseModel):
    """Typed so both sides can feed the latency report. Sent on `done` only."""
    first_token_ms: Optional[int] = None   # request received -> first LLM token
    first_chunk_ms: Optional[int] = None   # request received -> first text_delta yielded
    total_ms: Optional[int] = None         # request received -> done


class AgentChunk(BaseModel):
    type: Literal["text_delta", "tool_status", "confirm", "escalate", "done", "error"]
    content: str = ""
    # text_delta: speakable text (see rules). Others: short UI/log message.

    metadata: Optional[dict] = None
    # confirm  : {"action_id": str, "tool": "email"|"telegram",
    #             "recipient": str, "summary": str}
    # escalate : {"reason": str, "ticket_id": Optional[str]}
    # error    : {"code": str}
    # Others   : free-form, for UI/logging only. Never required by the pipeline.

    timing: Optional[Timing] = None        # only on `done`


async def stream_agent_turn(request: AgentRequest) -> AsyncGenerator[AgentChunk, None]:
    """
    Takes one user turn and yields chunks as soon as they are ready.

    WHAT GETS SPOKEN
    - ONLY `text_delta` goes to TTS. `tool_status`, `confirm`, `escalate`,
      and `error` are UI/logging only.
    - `text_delta` is plain text: no markdown, no emoji, no URLs read aloud.
      One sentence or clause, at most MAX_SPEAKABLE_CHARS characters.

    STREAM INVARIANTS
    - `done` is ALWAYS the last chunk, including after `error` or `escalate`.
    - On error: yield an apologetic `text_delta` first (so the user never
      hears silence), then `error`, then `done`.
    - `escalate` is a signal, not a terminator. Order: spoken `text_delta`
      handoff message -> `escalate` -> `done`.
    - Empty/whitespace `user_text`: yield only `done`; touch nothing.

    CONFIRMATION FLOW (email / Telegram sends requested by or for the user)
    - `confirm` means a send is pending and the next user turn is a yes/no.
    - `confirm` is ALWAYS preceded by at least one `text_delta` read-back that
      contains the recipient and a summary of the content. The read-back is
      generated IN CODE from the same data as the `confirm` metadata (tool,
      recipient, summary, action_id), never free-form by the LLM, so what is
      spoken always matches what will be sent.
    - A "yes" is valid only for that `action_id`, and only if the read-back
      was fully played (see `truncated_assistant_text`). Otherwise re-read.
    - A "no"/"cancel" is honored immediately, even after a truncated read-back.
    - The send function additionally enforces `confirmed=True` in state, so
      the LLM cannot bypass the gate.

    SUMMARY EMAIL
    - Offered at most once per session (`summary_offered` in state), when the
      issue is resolved or the user signals they are done (or on an `idle`
      event). Sent only after the normal confirm flow. Never sent at disconnect.

    ESCALATION ALERTS
    - Alerts to the allowlisted support Telegram chat fire at escalation time
      from server config, with an idempotency key (a retry never double-sends).
      They need no user confirmation because the user is not the recipient.

    CANCELLATION (barge-in)
    - The pipeline may cancel this generator at any time. On
      CancelledError/GeneratorExit the agent must NOT complete a pending tool
      action and must leave state consistent. What counts as "spoken" is
      decided by the NEXT request's `truncated_assistant_text`.

    TIMING
    - The final `done` chunk carries a populated `Timing`.
    """
    raise NotImplementedError
    yield  # pragma: no cover


async def close_session(session_id: str) -> None:
    """
    Called on disconnect or timeout. CLEANUP ONLY: free memory, flush the audit
    log, discard any pending (unconfirmed) action.

    Must NEVER send email or Telegram to the user. Anything unconfirmed at this
    point is dropped. Safe to call more than once.
    """
    raise NotImplementedError