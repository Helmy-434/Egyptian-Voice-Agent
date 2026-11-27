
import pytest
from uuid import uuid4

from app.agent import state


@pytest.fixture
def session_id():
    """Create a unique session and clean it up after each test."""
    session_id = str(uuid4())
    state.create_session(session_id, user_email=None)

    yield session_id

    state.close_session(session_id)


# --------------------------------------------------
# Session tests
# --------------------------------------------------

def test_create_and_get_session(session_id):
    session = state.get_session(session_id)

    assert session is not None
    assert session.session_id == session_id
    assert session.send_count == 0
    assert session.messages == []


def test_duplicate_session_creation(session_id):
    with pytest.raises(ValueError):
        state.create_session(session_id, user_email=None)


def test_get_nonexistent_session():
    assert state.get_session("unknown-session") is None


def test_session_returns_deep_copy(session_id):
    session = state.get_session(session_id)
    session.send_count = 100

    stored_session = state.get_session(session_id)

    assert stored_session.send_count == 0


# --------------------------------------------------
# Message tests
# --------------------------------------------------

def test_add_and_get_message(session_id):
    message = state.add_message(
        session_id=session_id,
        content="Hello",
        role="user",
    )

    assert message.content == "Hello"
    assert message.role == "user"

    messages = state.get_messages(session_id)

    assert len(messages) == 1
    assert messages[0].content == "Hello"


def test_add_multiple_messages(session_id):
    state.add_message(session_id, "Hello", "user")
    state.add_message(session_id, "Welcome!", "assistant")

    messages = state.get_messages(session_id)

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"


def test_add_empty_message(session_id):
    with pytest.raises(ValueError):
        state.add_message(session_id, "   ", "user")


def test_add_message_to_unknown_session():
    with pytest.raises(KeyError):
        state.add_message("unknown-session", "Hello", "user")


# --------------------------------------------------
# Pending action tests
# --------------------------------------------------

def create_test_action(session_id):
    return state.set_pending_action(
        session_id=session_id,
        tool="email",
        recipient="customer@example.com",
        subject="Appointment confirmation",
        content="Your appointment is confirmed.",
        summary="Confirm the customer's appointment.",
    )


def test_create_pending_action(session_id):
    action = create_test_action(session_id)

    assert action.action_id
    assert action.tool == "email"
    assert action.status == state.ActionStatus.PENDING_CONFIRMATION
    assert action.confirmed is False


def test_get_pending_action(session_id):
    created = create_test_action(session_id)
    retrieved = state.get_pending_action(session_id)

    assert retrieved is not None
    assert retrieved.action_id == created.action_id


def test_prevent_multiple_active_actions(session_id):
    create_test_action(session_id)

    with pytest.raises(ValueError):
        create_test_action(session_id)


def test_confirm_pending_action(session_id):
    action = create_test_action(session_id)

    confirmed = state.confirm_pending_action(
        session_id,
        action.action_id,
    )

    assert confirmed.confirmed is True
    assert confirmed.status == state.ActionStatus.CONFIRMED


def test_cannot_confirm_same_action_twice(session_id):
    action = create_test_action(session_id)

    state.confirm_pending_action(session_id, action.action_id)

    with pytest.raises(ValueError):
        state.confirm_pending_action(session_id, action.action_id)


def test_cancel_pending_action(session_id):
    action = create_test_action(session_id)

    cancelled = state.cancel_pending_action(
        session_id,
        action.action_id,
    )

    assert cancelled.status == state.ActionStatus.CANCELLED
    assert cancelled.confirmed is False


def test_cannot_confirm_wrong_action_id(session_id):
    create_test_action(session_id)

    with pytest.raises(ValueError):
        state.confirm_pending_action(session_id, "wrong-id")


# --------------------------------------------------
# Action execution tests
# --------------------------------------------------

def test_complete_action_lifecycle(session_id):
    action = create_test_action(session_id)

    # Confirmation
    state.confirm_pending_action(session_id, action.action_id)

    # Claim action for execution
    claimed = state.claim_confirmed_action(
        session_id,
        action.action_id,
        max_sends=3,
    )

    assert claimed.status == state.ActionStatus.EXECUTING

    # Mark successful send
    sent = state.mark_action_sent(session_id, action.action_id)

    assert sent.status == state.ActionStatus.SENT
    assert state.get_send_count(session_id) == 1


def test_cannot_claim_unconfirmed_action(session_id):
    action = create_test_action(session_id)

    with pytest.raises(PermissionError):
        state.claim_confirmed_action(
            session_id,
            action.action_id,
            max_sends=3,
        )


def test_cannot_claim_same_action_twice(session_id):
    action = create_test_action(session_id)

    state.confirm_pending_action(session_id, action.action_id)

    state.claim_confirmed_action(
        session_id,
        action.action_id,
        max_sends=3,
    )

    with pytest.raises(ValueError):
        state.claim_confirmed_action(
            session_id,
            action.action_id,
            max_sends=3,
        )


def test_send_limit(session_id):
    action = create_test_action(session_id)

    state.confirm_pending_action(session_id, action.action_id)
    state.claim_confirmed_action(
        session_id,
        action.action_id,
        max_sends=0,
    ) if False else None

    # A zero send limit must reject execution.
    # Use a fresh action because the previous one is still pending.
    with pytest.raises(PermissionError):
        state.claim_confirmed_action(
            session_id,
            action.action_id,
            max_sends=0,
        )


def test_mark_action_failed(session_id):
    action = create_test_action(session_id)

    state.confirm_pending_action(session_id, action.action_id)
    state.claim_confirmed_action(
        session_id,
        action.action_id,
        max_sends=3,
    )

    failed = state.mark_action_failed(session_id, action.action_id)

    assert failed.status == state.ActionStatus.FAILED
    assert failed.confirmed is False
    assert state.get_send_count(session_id) == 0


# --------------------------------------------------
# Session flags
# --------------------------------------------------

def test_summary_offered_only_once(session_id):
    assert state.mark_summary_offered(session_id) is True
    assert state.mark_summary_offered(session_id) is False


def test_escalation_only_once(session_id):
    assert state.mark_escalated(session_id) is True
    assert state.mark_escalated(session_id) is False


# --------------------------------------------------
# Session cleanup
# --------------------------------------------------

def test_close_session(session_id):
    assert state.close_session(session_id) is True
    assert state.get_session(session_id) is None


def test_close_nonexistent_session():
    assert state.close_session("unknown-session") is False