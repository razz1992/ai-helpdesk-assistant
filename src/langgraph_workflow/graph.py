"""Build, compile and invoke the helpdesk StateGraph."""
from langchain_core.tracers.context import tracing_v2_enabled
from langsmith import Client
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.config import settings
from src.langgraph_workflow.nodes import (
    capture_assistant_message, capture_user_message, classify_query,
    fallback_answer, generate_knowledge_answer, greeting_answer, lookup_ticket,
    prepare_escalation, reject_escalation, request_human_approval,
    retrieve_knowledge, route_after_approval,
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


def build_graph(checkpointer=None):
    """Compile the graph, optionally attaching a checkpointer for interrupts."""
    builder = StateGraph(HelpdeskState)
    builder.add_node("capture_user_message", capture_user_message)
    builder.add_node("classify_query", classify_query)
    builder.add_node("greeting_answer", greeting_answer)
    builder.add_node("retrieve_knowledge", retrieve_knowledge)
    builder.add_node("generate_knowledge_answer", generate_knowledge_answer)
    builder.add_node("lookup_ticket", lookup_ticket)
    builder.add_node("request_human_approval", request_human_approval)
    builder.add_node("prepare_escalation", prepare_escalation)
    builder.add_node("reject_escalation", reject_escalation)
    builder.add_node("fallback_answer", fallback_answer)
    builder.add_node("capture_assistant_message", capture_assistant_message)

    builder.add_edge(START, "capture_user_message")
    builder.add_edge("capture_user_message", "classify_query")
    builder.add_conditional_edges("classify_query", route_by_intent, {
        "greeting_answer": "greeting_answer",
        "retrieve_knowledge": "retrieve_knowledge",
        "lookup_ticket": "lookup_ticket",
        "prepare_escalation": "request_human_approval",
        "fallback_answer": "fallback_answer",
    })
    builder.add_edge("retrieve_knowledge", "generate_knowledge_answer")
    builder.add_conditional_edges("request_human_approval", route_after_approval, {
        "approved": "prepare_escalation",
        "rejected": "reject_escalation",
    })
    for node in (
        "greeting_answer", "generate_knowledge_answer", "lookup_ticket",
        "prepare_escalation", "reject_escalation", "fallback_answer",
    ):
        builder.add_edge(node, "capture_assistant_message")
    builder.add_edge("capture_assistant_message", END)
    return builder.compile(checkpointer=checkpointer)


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


def run_helpdesk(
    question: str | None = None,
    *,
    thread_id: str | None = None,
    graph=None,
    resume: dict | None = None,
) -> HelpdeskState:
    """Run a new turn or resume a paused turn using its existing thread_id."""
    tags = [
        "ai-helpdesk-assistant",
        "langgraph",
        f"llm-provider:{settings.provider}",
        "persistent-conversation" if thread_id else "single-turn",
        "human-in-the-loop" if resume is not None else "normal-invocation",
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
    if thread_id:
        run_config["configurable"] = {"thread_id": thread_id}
        metadata["conversation_thread_id"] = thread_id

    compiled_graph = graph or build_graph()
    graph_input = (
        Command(resume=resume)
        if resume is not None
        else {"question": question or "", "errors": []}
    )

    if not settings.langsmith_tracing:
        return compiled_graph.invoke(graph_input, config=run_config)

    client = _build_langsmith_client()
    with tracing_v2_enabled(
        project_name=settings.langsmith_project,
        tags=tags,
        client=client,
    ):
        return compiled_graph.invoke(graph_input, config=run_config)
