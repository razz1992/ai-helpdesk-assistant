"""Graph node functions. Each node receives state and returns state updates."""
import re

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.types import interrupt

from src.langchain_components.llm import get_chat_model
from src.langchain_components.prompts import KNOWLEDGE_ANSWER_PROMPT
from src.langchain_components.retriever import get_retriever
from src.langchain_components.tools import lookup_ticket_record
from src.langgraph_workflow.state import HelpdeskState

TICKET_PATTERN = re.compile(r"\bINC\d{4}\b", re.IGNORECASE)
GREETING_PATTERN = re.compile(
    r"^(?:hi|hello|hey|good morning|good afternoon|good evening)"
    r"(?: there)?[!.\s]*$",
    re.IGNORECASE,
)
FOLLOW_UP_PATTERN = re.compile(
    r"\b(already tried|first two steps|those steps|that didn't work|"
    r"that did not work|still not working|still doesn't work|"
    r"still does not work|what next|what should i try next|"
    r"what should i do next|what do i try next|same issue|it failed|"
    r"not resolved)\b",
    re.IGNORECASE,
)


def capture_user_message(state: HelpdeskState) -> dict:
    """Append the current user turn to the reducer-managed message history."""
    return {"messages": [HumanMessage(content=state["question"])]}


def classify_query(state: HelpdeskState) -> dict:
    """Rule-based baseline with simple conversational follow-up resolution."""
    question = state["question"].strip()
    messages = state.get("messages", [])
    previous_user_questions = [
        str(message.content)
        for message in messages[:-1]
        if isinstance(message, HumanMessage)
    ]

    if FOLLOW_UP_PATTERN.search(question) and previous_user_questions:
        topic_terms = (
            "vpn", "password", "laptop", "access", "email", "outlook",
            "software", "install", "application", "account", "login",
        )
        previous_question = next(
            (
                item for item in reversed(previous_user_questions)
                if any(term in item.lower() for term in topic_terms)
            ),
            None,
        )
        if previous_question:
            contextual_question = (
                f"Follow-up to the previous helpdesk question: {previous_question}\n"
                f"Current follow-up: {question}"
            )
            return {"intent": "knowledge_question", "question": contextual_question}

    if GREETING_PATTERN.fullmatch(question):
        return {"intent": "greeting"}
    lower = question.lower()
    if (
        any(term in lower for term in ("password", "credential", "login", "secret"))
        and any(term in lower for term in (
            "another employee", "other employee", "someone else",
            "someone else's", "someone elses", "colleague",
        ))
    ):
        return {"intent": "out_of_scope"}
    match = TICKET_PATTERN.search(question)
    if match:
        return {"intent": "ticket_status", "ticket_id": match.group(0).upper()}
    if any(x in lower for x in (
        "human", "person", "escalate", "still cannot", "still can't",
        "unresolved", "blocking my work", "cannot work", "can't work",
    )):
        return {"intent": "escalation_request"}
    if any(x in lower for x in (
        "password", "vpn", "laptop", "access", "email", "outlook",
        "software", "install", "application", "account", "login",
    )):
        return {"intent": "knowledge_question"}
    return {"intent": "out_of_scope"}


def greeting_answer(state: HelpdeskState) -> dict:
    return {
        "answer": (
            "Hello! I'm the AI Helpdesk Assistant. I can help with demo IT topics "
            "such as VPN, passwords, laptop and application access, email, software "
            "installation, and demo ticket status. What do you need help with?"
        )
    }


def retrieve_knowledge(state: HelpdeskState) -> dict:
    return {"retrieved_documents": get_retriever().invoke(state["question"])}


def generate_knowledge_answer(state: HelpdeskState) -> dict:
    docs = state.get("retrieved_documents", [])
    context = "\n\n".join(doc.page_content for doc in docs)
    history = state.get("messages", [])[:-1]
    chain = KNOWLEDGE_ANSWER_PROMPT | get_chat_model() | StrOutputParser()
    return {
        "answer": chain.invoke({
            "question": state["question"],
            "context": context,
            "history": history,
        })
    }


def lookup_ticket(state: HelpdeskState) -> dict:
    ticket_id = state.get("ticket_id") or ""
    record = lookup_ticket_record(ticket_id)
    if record is None:
        return {
            "ticket_record": None,
            "answer": f"Ticket {ticket_id} was not found in the demo dataset. Please verify the ID.",
        }
    return {
        "ticket_record": {"ticket_id": ticket_id, **record},
        "answer": (
            f"Ticket {ticket_id}: {record['status']}. "
            f"Assigned team: {record['assigned_team']}. "
            f"Priority: {record['priority']}. Summary: {record['summary']}."
        ),
    }


def request_human_approval(state: HelpdeskState) -> dict:
    """Pause execution until a human approves or rejects the simulated escalation.

    Keep interrupt() before side effects: LangGraph re-enters this node when resumed.
    """
    decision = interrupt({
        "kind": "escalation_approval",
        "question": state.get("question", ""),
        "reason": (
            "The user requested human support or reports an unresolved/work-blocking issue."
        ),
        "message": "Approve this simulated escalation?",
    })
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else False
    return {
        "human_approval": approved,
        "approval_comment": (
            str(decision.get("comment", "")) if isinstance(decision, dict) else ""
        ),
    }


def route_after_approval(state: HelpdeskState) -> str:
    return "approved" if state.get("human_approval") else "rejected"


def prepare_escalation(state: HelpdeskState) -> dict:
    return {
        "needs_human_review": True,
        "answer": (
            "Escalation approved. This demo marks the conversation for simulated "
            "human support; no real ticket is created."
        ),
    }


def reject_escalation(state: HelpdeskState) -> dict:
    return {
        "needs_human_review": False,
        "answer": (
            "Escalation was not approved. No ticket or external action was created. "
            "You can continue troubleshooting or request another review."
        ),
    }


def fallback_answer(state: HelpdeskState) -> dict:
    question = state["question"].lower()
    credential_request = (
        any(term in question for term in ("password", "credential", "login", "secret"))
        and any(term in question for term in (
            "another employee", "other employee", "someone else",
            "someone else's", "someone elses", "colleague",
        ))
    )
    if credential_request:
        return {"answer": (
            "I can't provide another person's password or credentials. "
            "Use the approved identity and access-management process for authorized access, "
            "and never share passwords or MFA codes."
        )}
    return {"answer": (
        "I can help with demo IT topics: passwords, VPN, laptop/access, "
        "email, software installation and checking a demo ticket ID."
    )}


def capture_assistant_message(state: HelpdeskState) -> dict:
    """Append the assistant response so the next turn has complete history."""
    return {"messages": [AIMessage(content=state.get("answer", ""))]}
