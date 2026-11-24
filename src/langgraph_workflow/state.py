"""Typed shared state that flows between LangGraph nodes."""
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class HelpdeskState(TypedDict, total=False):
    question: str
    intent: str
    ticket_id: str | None
    retrieved_documents: list[Any]
    ticket_record: dict[str, Any] | None
    answer: str
    needs_human_review: bool
    human_approval: bool
    approval_comment: str
    escalation_reason: str | None
    errors: list[str]
    # add_messages appends new turns while preserving prior conversation messages.
    messages: Annotated[list[AnyMessage], add_messages]
