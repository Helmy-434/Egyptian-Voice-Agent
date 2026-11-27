from langchain_core.messages import AIMessage

import app.agent.graph as graph_module
from app.agent import state as session_store


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, config=None):
        self.calls += 1
        return next(self.responses)


def create_test_session(session_id):
    session_store.create_session(
        session_id=session_id,
        user_email=None,
    )


def mock_faq(monkeypatch):
    monkeypatch.setattr(
        graph_module.faq_kb,
        "search",
        lambda query: [],
    )

    monkeypatch.setattr(
        graph_module.faq_kb,
        "format_results",
        lambda results: "No relevant FAQ found.",
    )


def test_graph_normal_response(monkeypatch):
    session_id = "graph-normal-test"

    create_test_session(session_id)
    mock_faq(monkeypatch)

    fake_llm = FakeLLM(
        [
            AIMessage(
                content="مواعيد العمل من التاسعة صباحاً حتى الخامسة مساءً."
            )
        ]
    )

    monkeypatch.setattr(
        graph_module,
        "get_llm",
        lambda: fake_llm,
    )

    answer = graph_module.run_agent(
        session_id=session_id,
        user_text="ما هي مواعيد العمل؟",
    )

    assert answer
    assert "مواعيد العمل" in answer
    assert fake_llm.calls == 1

    history = session_store.get_messages(session_id)

    assert history[-2].role == "user"
    assert history[-1].role == "assistant"


def test_graph_prepare_email_tool_cycle(monkeypatch):
    session_id = "graph-email-test"

    create_test_session(session_id)
    mock_faq(monkeypatch)

    fake_llm = FakeLLM(
        [
            # First LLM response: request the email tool.
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "prepare_email",
                        "args": {
                            "recipient": "customer@example.com",
                            "subject": "Support Follow-up",
                            "content": "Your issue has been resolved.",
                            "summary": "A follow-up about your resolved issue.",
                        },
                        "id": "call-email-123",
                        "type": "tool_call",
                    }
                ],
            ),

            # Second LLM response: respond after tool execution.
            AIMessage(
                content=(
                    "تم تجهيز البريد الإلكتروني. "
                    "هل تؤكد إرساله؟"
                )
            ),
        ]
    )

    monkeypatch.setattr(
        graph_module,
        "get_llm",
        lambda: fake_llm,
    )

    answer = graph_module.run_agent(
        session_id=session_id,
        user_text="ابعتلي ملخص المحادثة على الإيميل.",
    )

    assert answer
    assert "البريد الإلكتروني" in answer
    assert fake_llm.calls == 2

    # Verify that the tool prepared an action,
    # but did not send the email.
    action = session_store.get_pending_action(session_id)

    assert action is not None
    assert action.tool == "email"
    assert action.recipient == "customer@example.com"
    assert action.confirmed is False
    assert action.status == "pending_confirmation"


def test_graph_tool_result_is_followed_by_llm(monkeypatch):
    session_id = "graph-tool-loop-test"

    create_test_session(session_id)
    mock_faq(monkeypatch)

    fake_llm = FakeLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "prepare_email",
                        "args": {
                            "recipient": "customer@example.com",
                            "subject": "Support",
                            "content": "Hello",
                            "summary": "A short follow-up.",
                        },
                        "id": "call-email-456",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="تم تجهيز البريد الإلكتروني بنجاح."
            ),
        ]
    )

    monkeypatch.setattr(
        graph_module,
        "get_llm",
        lambda: fake_llm,
    )

    graph_module.run_agent(
        session_id=session_id,
        user_text="أرسل لي رسالة على بريدي الإلكتروني.",
    )

    # One LLM call to request the tool,
    # another LLM call after receiving the ToolMessage.
    assert fake_llm.calls == 2