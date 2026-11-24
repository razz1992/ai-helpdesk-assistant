"""Build and compile the helpdesk StateGraph."""
from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import Client
from langgraph.graph import END, START, StateGraph

from src.config import settings
from src.langgraph_workflow.nodes import (
    classify_query, fallback_answer, generate_knowledge_answer,
    greeting_answer, lookup_ticket, prepare_escalation, retrieve_knowledge,
)
from src.langgraph_workflow.state import HelpdeskState


def route_by_intent(state: HelpdeskState) -> str:
    return {
        "greeting": "greeting_answer",
        "knowledge_question": "retrieve_knowledge",
        "ticket_status": "lookup_ticket",
        "escalation_request": "prepare_escalation",
        "out_of_scope": "fallback_answer",
    }.get(state.get("intent", "out_of_scope"), "fallback_answer")


def build_graph():
    builder = StateGraph(HelpdeskState)
    builder.add_node("classify_query", classify_query)
    builder.add_node("greeting_answer", greeting_answer)
    builder.add_node("retrieve_knowledge", retrieve_knowledge)
    builder.add_node("generate_knowledge_answer", generate_knowledge_answer)
    builder.add_node("lookup_ticket", lookup_ticket)
    builder.add_node("prepare_escalation", prepare_escalation)
    builder.add_node("fallback_answer", fallback_answer)
    builder.add_edge(START, "classify_query")
    builder.add_conditional_edges("classify_query", route_by_intent, {
        "greeting_answer": "greeting_answer",
        "retrieve_knowledge": "retrieve_knowledge",
        "lookup_ticket": "lookup_ticket",
        "prepare_escalation": "prepare_escalation",
        "fallback_answer": "fallback_answer",
    })
    builder.add_edge("retrieve_knowledge", "generate_knowledge_answer")
    for node in ("greeting_answer", "generate_knowledge_answer", "lookup_ticket",
                 "prepare_escalation", "fallback_answer"):
        builder.add_edge(node, END)
    return builder.compile()


def _build_langsmith_client() -> Client:
    """Build an explicitly workspace-scoped client for an org-scoped Service Key."""
    missing = [
        name for name, value in (
            ("LANGSMITH_API_KEY", settings.langsmith_api_key),
            ("LANGSMITH_WORKSPACE_ID", settings.langsmith_workspace_id),
            ("LANGSMITH_ORGANIZATION_ID", settings.langsmith_organization_id),
        ) if not value.strip()
    ]
    if missing:
        raise RuntimeError(
            "LangSmith tracing is enabled but these settings are missing: "
            + ", ".join(missing)
        )

    return Client(
        api_key=settings.langsmith_api_key,
        workspace_id=settings.langsmith_workspace_id,
        headers={"X-Organization-Id": settings.langsmith_organization_id},
    )


def run_helpdesk(question: str) -> HelpdeskState:
    """Run one conversation turn and trace the graph plus nested LLM calls."""
    tags = [
        "ai-helpdesk-assistant",
        "langgraph",
        f"llm-provider:{settings.provider}",
    ]
    metadata = {
        "llm_provider": settings.provider,
        "llm_model": (
            settings.google_model if settings.provider == "google"
            else settings.deepseek_model if settings.provider == "deepseek"
            else settings.openai_model if settings.provider == "openai"
            else "ollama"
        ),
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.sentence_transformer_model,
        "langsmith_project": settings.langsmith_project,
        "langsmith_tracing_enabled": settings.langsmith_tracing,
    }
    run_config = {
        "run_name": "helpdesk_conversation",
        "tags": tags,
        "metadata": metadata,
    }
    graph = build_graph()
    initial_state = {"question": question, "errors": []}

    if not settings.langsmith_tracing:
        return graph.invoke(initial_state, config=run_config)

    # Pass an explicitly scoped client so org-scoped Service Keys send both
    # X-Organization-Id and X-Tenant-Id on trace and query requests.
    client = _build_langsmith_client()
    with tracing_v2_enabled(
        project_name=settings.langsmith_project,
        tags=tags,
        client=client,
    ):
        return graph.invoke(initial_state, config=run_config)
