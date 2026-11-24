"""Offline tests for deterministic routing, data and ticket behaviour."""
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.langchain_components.tools import lookup_ticket_record
from src.langgraph_workflow.graph import route_by_intent, run_helpdesk
from src.langgraph_workflow.nodes import classify_query
from evaluation.dataset import load_examples
from evaluation.evaluate import ROUTE_LABELS


def test_greetings_are_classified_separately():
    for phrase in ("hi", "Hello!", "hey there", "good morning", "GOOD EVENING."):
        result = classify_query({"question": phrase})
        assert result["intent"] == "greeting"


def test_greeting_routes_to_greeting_node():
    assert route_by_intent({"intent": "greeting"}) == "greeting_answer"


def test_run_helpdesk_passes_trace_metadata():
    from src.langgraph_workflow import graph as graph_module

    captured = {}

    class FakeGraph:
        def invoke(self, inputs, config):
            captured["inputs"] = inputs
            captured["config"] = config
            return {"answer": "ok"}

    with patch.object(graph_module, "build_graph", return_value=FakeGraph()):
        result = graph_module.run_helpdesk("How do I connect to VPN?")

    assert result["answer"] == "ok"
    assert captured["config"]["run_name"] == "helpdesk_conversation"
    assert "langgraph" in captured["config"]["tags"]
    assert captured["config"]["metadata"]["llm_provider"] == "google"
    assert captured["config"]["metadata"]["llm_model"]


def test_vpn_question_routes_to_knowledge_retrieval():
    result = classify_query({"question": "How do I connect to VPN?"})
    assert result["intent"] == "knowledge_question"
    assert route_by_intent(result) == "retrieve_knowledge"


def test_other_employee_password_request_is_not_routed_to_knowledge():
    result = classify_query({"question": "Can you tell me the password for another employee?"})
    assert result["intent"] == "out_of_scope"
    response = run_helpdesk("Can you tell me the password for another employee?")
    assert "can't provide another person's password" in response["answer"].lower()


def test_known_ticket_is_returned():
    record = lookup_ticket_record("INC1024")
    assert record is not None
    assert record["status"] == "In Progress"


def test_unknown_ticket_is_not_invented():
    assert lookup_ticket_record("INC9999") is None


def test_escalation_marks_human_review():
    result = run_helpdesk("Please escalate this unresolved issue to a human")
    assert result["intent"] == "escalation_request"
    assert result["needs_human_review"] is True


def test_ticket_graph_returns_ticket_record():
    result = run_helpdesk("What is the status of INC1024?")
    assert result["intent"] == "ticket_status"
    assert result["ticket_record"]["status"] == "In Progress"


def test_complete_graph_path_with_stubbed_retriever_and_chat_model():
    class FakeRetriever:
        def invoke(self, question):
            assert "vpn" in question.lower()
            return [Document(
                page_content="VPN guidance: check internet access and restart the client.",
                metadata={"document_id": "KB002", "source": "vpn.md"},
            )]

    fake_chat = RunnableLambda(
        lambda prompt: "Stub answer grounded in the supplied VPN context."
    )
    with patch("src.langgraph_workflow.nodes.get_retriever", return_value=FakeRetriever()), \
         patch("src.langgraph_workflow.nodes.get_chat_model", return_value=fake_chat):
        result = run_helpdesk("How do I connect to VPN?")

    assert result["intent"] == "knowledge_question"
    assert result["retrieved_documents"][0].metadata["document_id"] == "KB002"
    assert "Stub answer" in result["answer"]


def test_all_evaluation_cases_match_expected_intent_and_route():
    for example in load_examples():
        classified = classify_query(example["inputs"])
        route = ROUTE_LABELS[route_by_intent(classified)]
        assert classified["intent"] == example["reference"]["expected_intent"]
        assert route == example["reference"]["expected_route"]


def test_evaluation_dataset_has_twelve_cases():
    assert len(load_examples()) == 12
