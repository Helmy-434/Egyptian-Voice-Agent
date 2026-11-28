from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from app.agent import state as session_store
from app.agent.kb import faq_kb
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools.email import prepare_email
from app.agent.tools.telegram import request_escalation
from app.configs import get_settings
from app.contracts import AgentChunk, AgentRequest, MAX_SPEAKABLE_CHARS


PROVIDER_API_KEYS = {
    "openrouter": "openrouter_api_key",
    "openai": "openai_api_key",
    "groq": "groq_api_key",
}


AGENT_TOOLS = [
    prepare_email,
    request_escalation,
]


class AgentGraphState(TypedDict):
    session_id: str
    user_text: str
    messages: Annotated[list[BaseMessage], add_messages]
    faq_context: str


def get_llm() -> ChatOpenAI:

    settings = get_settings()
    provider = settings.llm_provider

    if provider not in PROVIDER_API_KEYS:
        raise ValueError(
            f"Unsupported LLM provider: {provider}"
        )

    api_key = getattr(
        settings,
        PROVIDER_API_KEYS[provider],
    ).get_secret_value()

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=api_key,
        base_url=settings.llm_base_url,
        temperature=0.2,
        timeout=settings.llm_timeout_sec,
        max_tokens=settings.llm_max_output_tokens,
    )


def load_context(state: AgentGraphState) -> dict:

    history = session_store.get_messages(
        state["session_id"]
    )

    messages: list[BaseMessage] = []

    for message in history:
        if message.role == "user":
            messages.append(
                HumanMessage(content=message.content)
            )

        elif message.role == "assistant":
            messages.append(
                AIMessage(content=message.content)
            )

    messages.append(
        HumanMessage(content=state["user_text"])
    )

    return {"messages": messages}



def retrieve_faq(state: AgentGraphState) -> dict:

    results = faq_kb.search(state["user_text"])
    return {
        "faq_context": faq_kb.format_results(results),
    }


def generate_response(
    state: AgentGraphState,
    config: RunnableConfig,
) -> dict:

    llm = get_llm().bind_tools(AGENT_TOOLS)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"FAQ KNOWLEDGE:\n{state['faq_context']}\n\n"

        "EMAIL RULES:\n"
        "- If the customer explicitly requests an email, "
        "use prepare_email.\n"
        "- Preparing an email does not mean it was sent.\n"
        "- Never claim an email was sent before the backend "
        "confirms execution.\n"
        "- Do not invent a confirmation action ID.\n\n"

        "ESCALATION RULES:\n"
        "- If the customer explicitly requests human support, "
        "use request_escalation.\n"
        "- Escalate when the issue requires human intervention "
        "or cannot be reliably resolved.\n"
        "- Never claim support was notified unless the tool succeeds.\n\n"

        "GENERAL RULES:\n"
        "- Use the FAQ information when answering customer questions.\n"
        "- If the FAQ does not contain enough information, "
        "do not invent an answer.\n"
    )

    conversation = [
        SystemMessage(content=prompt),
        *state["messages"],
    ]

    response = llm.invoke(
        conversation,
        config=config,
    )

    return {"messages": [response]}



def build_graph():

    graph = StateGraph(AgentGraphState)

    graph.add_node("load_context", load_context)
    graph.add_node("retrieve_faq", retrieve_faq)
    graph.add_node("generate_response", generate_response)
    graph.add_node("tools", ToolNode(AGENT_TOOLS))

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "retrieve_faq")
    graph.add_edge("retrieve_faq", "generate_response")

    graph.add_conditional_edges(
        "generate_response",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    graph.add_edge("tools", "generate_response")

    return graph.compile()


agent_graph = build_graph()


def _get_final_answer(messages: list[BaseMessage]) -> str:

    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            if isinstance(message.content, str):
                return message.content.strip()

            return str(message.content).strip()

    return ""


def _split_speech(text: str) -> list[str]:
    """Split text into speakable chunks without exceeding the limit."""
    text = " ".join(text.split())

    if not text:
        return []

    chunks: list[str] = []
    remaining = text

    while len(remaining) > MAX_SPEAKABLE_CHARS:
        boundary = remaining.rfind(
            " ",
            0,
            MAX_SPEAKABLE_CHARS + 1,
        )

        if boundary <= 0:
            boundary = MAX_SPEAKABLE_CHARS

        chunks.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


def _make_email_readback(action) -> str:
    """Build confirmation speech from trusted pending-action data."""
    return (
        f"هجهز رسالة إلى {action.recipient}. "
        f"عنوان الرسالة: {action.subject}. "
        f"ملخص الرسالة: {action.summary}. "
        "هل تؤكد إرسال الرسالة؟"
    )


def _persist_turn(session_id: str, user_text: str, assistant_text: str) -> None:
    """save messages of user & assistant"""
    session_store.add_message(
        session_id,
        content=user_text,
        role="user",
    )
    if assistant_text:
        session_store.add_message(
            session_id,
            content=assistant_text,
            role="assistant",
        )


def _graph_input(session_id: str, user_text: str) -> dict:
    return {
        "session_id": session_id,
        "user_text": user_text,
        "messages": [],
        "faq_context": "",
    }


def _graph_config(session_id: str) -> RunnableConfig:
    config: RunnableConfig = {
        "configurable": {
            "session_id": session_id,
        },
        "recursion_limit": 12,
    }

    return config


def run_agent(session_id: str, user_text: str) -> str:
    """Synchronous entry point, useful for tests and non-voice callers."""

    result = agent_graph.invoke(
        _graph_input(session_id, user_text),
        config=_graph_config(session_id),
    )

    answer = _get_final_answer(result["messages"])

    _persist_turn(
        session_id,
        user_text,
        answer,
    )

    return answer


async def stream_agent_turn(request: AgentRequest):
    """Adapt LangGraph output to the voice pipeline AgentChunk interface.

    This first implementation uses async graph invocation and then
    emits the completed response in speech-sized chunks.
    """

    session_id = request.session_id
    user_text = request.user_text.strip()

    try:
        if not user_text:
            yield AgentChunk(
                type="done",
                content="",
            )
            return

        # Create the session on its first request.
        session = session_store.get_session(session_id)
        if session is None:
            email = request.context.user_email if request.context else None
            session_store.create_session(session_id, email)

        previous_action = session_store.get_pending_action(
            session_id
        )

        result = await agent_graph.ainvoke(
            _graph_input(session_id, user_text),
            config=_graph_config(session_id),
        )

        messages = result["messages"]

        current_action = session_store.get_pending_action(
            session_id
        )

        # A newly prepared email requires a deterministic read-back.
        if (
            current_action is not None
            and (
                previous_action is None
                or previous_action.action_id != current_action.action_id
            )
            and current_action.tool == "email"
        ):
            readback = _make_email_readback(current_action)

            _persist_turn(
                session_id,
                user_text,
                readback,
            )

            for chunk in _split_speech(readback):
                yield AgentChunk(
                    type="text_delta",
                    content=chunk,
                    metadata={
                        "confirmation_action_id": current_action.action_id,
                    },
                )

            yield AgentChunk(
                type="confirm",
                content="Email confirmation required",
                metadata={
                    "action_id": current_action.action_id,
                    "tool": "email",
                    "recipient": current_action.recipient,
                    "subject": current_action.subject,
                    "summary": current_action.summary,
                },
            )

            yield AgentChunk(
                type="done",
                content="",
            )
            return

        escalation_succeeded = any(
            isinstance(message, ToolMessage)
            and message.name == "request_escalation"
            for message in messages
        )

        if escalation_succeeded:
            yield AgentChunk(
                type="escalate",
                content="Human support escalation requested",
                metadata={
                    "session_id": session_id,
                },
            )

        answer = _get_final_answer(messages)

        _persist_turn(
            session_id,
            user_text,
            answer,
        )

        for chunk in _split_speech(answer):
            yield AgentChunk(
                type="text_delta",
                content=chunk,
            )

    except Exception:
        import logging

        logging.getLogger(__name__).exception(
            "Agent turn failed (session=%s)",
            session_id,
        )

        yield AgentChunk(
            type="error",
            content="تعذر معالجة الطلب حالياً.",
            metadata={
                "code": "agent_failed",
            },
        )

    finally:
        yield AgentChunk(
            type="done",
            content="",
        )


def close_session(session_id: str) -> None:
    session_store.close_session(session_id)
