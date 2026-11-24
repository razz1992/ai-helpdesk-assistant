"""Graph node functions. Each node receives state and returns state updates."""
import re
from langchain_core.output_parsers import StrOutputParser
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


def classify_query(state: HelpdeskState) -> dict:
    """Rule-based baseline; replace with structured LLM classification later."""
    question = state["question"].strip()
    if GREETING_PATTERN.fullmatch(question):
        return {"intent": "greeting"}
    lower = question.lower()
    # Do not treat requests for another person's credentials as routine IT help.
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
    if any(x in lower for x in ("human", "person", "escalate", "still cannot",
                                "still can't", "unresolved", "blocking my work",
                                "cannot work", "can't work")):
        return {"intent": "escalation_request"}
    if any(x in lower for x in ("password", "vpn", "laptop", "access", "email",
                                "outlook", "software", "install", "application",
                                "account", "login")):
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
    chain = KNOWLEDGE_ANSWER_PROMPT | get_chat_model() | StrOutputParser()
    return {"answer": chain.invoke({"question": state["question"], "context": context})}


def lookup_ticket(state: HelpdeskState) -> dict:
    ticket_id = state.get("ticket_id") or ""
    record = lookup_ticket_record(ticket_id)
    if record is None:
        return {"ticket_record": None,
                "answer": f"Ticket {ticket_id} was not found in the demo dataset. Please verify the ID."}
    return {"ticket_record": {"ticket_id": ticket_id, **record},
            "answer": (f"Ticket {ticket_id}: {record['status']}. "
                       f"Assigned team: {record['assigned_team']}. "
                       f"Priority: {record['priority']}. Summary: {record['summary']}.")}


def prepare_escalation(state: HelpdeskState) -> dict:
    return {
        "needs_human_review": True,
        "escalation_reason": "User requested human support or reports an unresolved/work-blocking issue. This is simulated.",
        "answer": ("This needs review by a human IT support representative. "
                   "This demo marks the conversation for simulated escalation; no real ticket is created.")
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
    return {"answer": ("I can help with demo IT topics: passwords, VPN, laptop/access, "
                       "email, software installation and checking a demo ticket ID.")}
