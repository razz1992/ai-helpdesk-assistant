"""Typed shared state that flows between LangGraph nodes."""
from typing import Any, TypedDict


class HelpdeskState(TypedDict, total=False):
    question: str
    intent: str
    ticket_id: str | None
    retrieved_documents: list[Any]
    ticket_record: dict[str, Any] | None
    answer: str
    needs_human_review: bool
    escalation_reason: str | None
    errors: list[str]
