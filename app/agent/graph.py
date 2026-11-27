from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage , BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.agent import state as session_store
from app.agent.kb import faq_kb
from app.agent.prompts import SYSTEM_PROMPT
from app.configs import get_settings


PROVIDER_API_KEYS = {
    "openrouter": "openrouter_api_key",
    "openai": "openai_api_key",
    "groq": "groq_api_key",
}


class AgentGraphState(TypedDict):
    session_id: str
    user_text: str
    messages: list[dict]
    faq_context: str
    response: str
    should_escalate: bool
    error: str


def get_llm() -> ChatOpenAI:

    settings = get_settings()
    provider = settings.llm_provider

    if provider not in PROVIDER_API_KEYS:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    api_key = getattr(settings, PROVIDER_API_KEYS[provider]).get_secret_value()

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

    messages = [
        {"role": message.role, "content": message.content}
        for message in history
    ]

    return {"messages": messages}



def retrieve_faq(state: AgentGraphState) -> dict:
    results = faq_kb.search(state["user_text"])
    return {"faq_context": faq_kb.format_results(results)}



def generate_response(state: AgentGraphState) -> dict:

    llm = get_llm()

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"FAQ KNOWLEDGE:\n{state['faq_context']}\n\n"
        "If the FAQ does not contain enough information, "
        "do not invent an answer."
    )

    conversation : list[BaseMessage] = [SystemMessage(content=prompt)]

    for message in state["messages"]:
        if message["role"] == "user":
            conversation.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            conversation.append(AIMessage(content=message["content"]))

    conversation.append(HumanMessage(content=state["user_text"]))

    response = llm.invoke(conversation)
    answer = response.content

    if not isinstance(answer, str):
        answer = str(answer)

    return {
        "response": answer.strip(),
        "should_escalate": False,
    }


def build_graph():

    workflow = StateGraph(AgentGraphState)

    workflow.add_node("load_context", load_context)
    workflow.add_node("retrieve_faq", retrieve_faq)
    workflow.add_node("generate_response", generate_response)

    workflow.set_entry_point("load_context")
    workflow.add_edge("load_context", "retrieve_faq")
    workflow.add_edge("retrieve_faq", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()


agent_graph = build_graph()


def run_agent(session_id: str, user_text: str) -> str:

    result = agent_graph.invoke(
        {
            "session_id": session_id,
            "user_text": user_text,
            "messages": [],
            "faq_context": "",
            "response": "",
            "should_escalate": False,
            "error": "",
        }
    )

    answer = result["response"]

    session_store.add_message(session_id, content=user_text, role="user")

    if answer:
        session_store.add_message(session_id, content=answer, role="assistant")

    return answer