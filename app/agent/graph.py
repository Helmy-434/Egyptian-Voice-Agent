from typing import Annotated , TypedDict
from typing_extensions import TypedDict
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from langgraph.graph import START , END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent import state as session_store
from app.agent.kb import faq_kb
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools.email import prepare_email
from app.agent.tools.telegram import request_escalation
from app.configs import get_settings


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
        raise ValueError(f"Unsupported LLM provider: {provider}")

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

    history = session_store.get_messages(state["session_id"])

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

    return {
        "messages": messages,
    }


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
        "- Never claim an email was sent before the backend confirms execution.\n\n"

        "ESCALATION RULES:\n"
        "- If the customer explicitly requests human support, "
        "use request_escalation.\n"
        "- Escalate when the issue requires human intervention "
        "or cannot be reliably resolved.\n"
        "- Never claim that support was notified unless the tool succeeds.\n\n"

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

    return {
        "messages": [response],
    }


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


def run_agent(session_id: str, user_text: str) -> str:

    config : RunnableConfig = {
        "configurable": {
            "session_id": session_id,
        },
        "recursion_limit": 12,
    }

    result = agent_graph.invoke(
        {
            "session_id": session_id,
            "user_text": user_text,
            "messages": [],
            "faq_context": "",
        },
        config=config,
    )

    # Find the final assistant response, ignoring intermediate
    # assistant messages that only contain tool calls.
    answer = ""

    for message in reversed(result["messages"]):

        if isinstance(message, AIMessage) and not message.tool_calls:

            if isinstance(message.content, str):
                answer = message.content.strip()
            else:
                answer = str(message.content).strip()

            break

    session_store.add_message(
        session_id,
        content=user_text,
        role="user",
    )

    if answer:
        session_store.add_message(
            session_id,
            content=answer,
            role="assistant",
        )

    return answer