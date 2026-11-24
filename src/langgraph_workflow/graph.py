"""Build and compile the helpdesk StateGraph."""
from langgraph.graph import END, START, StateGraph
from src.langgraph_workflow.nodes import (
    classify_query, fallback_answer, generate_knowledge_answer,
    lookup_ticket, prepare_escalation, retrieve_knowledge,
)
from src.langgraph_workflow.state import HelpdeskState

def route_by_intent(state: HelpdeskState) -> str:
    return {
        "knowledge_question": "retrieve_knowledge",
        "ticket_status": "lookup_ticket",
        "escalation_request": "prepare_escalation",
        "out_of_scope": "fallback_answer",
    }.get(state.get("intent", "out_of_scope"), "fallback_answer")

def build_graph():
    builder = StateGraph(HelpdeskState)
    builder.add_node("classify_query", classify_query)
    builder.add_node("retrieve_knowledge", retrieve_knowledge)
    builder.add_node("generate_knowledge_answer", generate_knowledge_answer)
    builder.add_node("lookup_ticket", lookup_ticket)
    builder.add_node("prepare_escalation", prepare_escalation)
    builder.add_node("fallback_answer", fallback_answer)
    builder.add_edge(START, "classify_query")
    builder.add_conditional_edges("classify_query", route_by_intent, {
        "retrieve_knowledge": "retrieve_knowledge",
        "lookup_ticket": "lookup_ticket",
        "prepare_escalation": "prepare_escalation",
        "fallback_answer": "fallback_answer",
    })
    builder.add_edge("retrieve_knowledge", "generate_knowledge_answer")
    for node in ("generate_knowledge_answer", "lookup_ticket",
                 "prepare_escalation", "fallback_answer"):
        builder.add_edge(node, END)
    return builder.compile()

def run_helpdesk(question: str) -> HelpdeskState:
    return build_graph().invoke({"question": question, "errors": []})
