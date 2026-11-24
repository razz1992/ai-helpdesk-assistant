"""Persistence and conversational-context tests for the SQLite checkpointer."""
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.sqlite import SqliteSaver

from src.langgraph_workflow.graph import build_graph


class FakeRetriever:
    def invoke(self, question):
        assert "vpn" in question.lower()
        return [Document(
            page_content="VPN steps: verify internet, restart the VPN client, then sign in.",
            metadata={"document_id": "KB002"},
        )]


def test_sqlite_checkpoint_resumes_conversation_after_reopening_database(
    tmp_path: Path, monkeypatch
):
    from src.langgraph_workflow import nodes

    monkeypatch.setattr(nodes, "get_retriever", lambda: FakeRetriever())
    monkeypatch.setattr(
        nodes,
        "get_chat_model",
        lambda: RunnableLambda(lambda prompt: "Try checking your network and VPN client."),
    )
    database = tmp_path / "helpdesk-checkpoints.sqlite"
    thread_id = "vpn-conversation-001"
    config = {"configurable": {"thread_id": thread_id}}

    # First application process: start a new conversation and persist turn one.
    with SqliteSaver.from_conn_string(str(database)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        first = graph.invoke(
            {"question": "My VPN is not working.", "errors": []},
            config=config,
        )
        assert first["intent"] == "knowledge_question"
        assert len(first["messages"]) == 2
        assert isinstance(first["messages"][0], HumanMessage)
        assert isinstance(first["messages"][1], AIMessage)

    # Second application process: reopen the same SQLite file and resume the thread.
    with SqliteSaver.from_conn_string(str(database)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        second = graph.invoke(
            {"question": "I already tried the first two steps.", "errors": []},
            config=config,
        )
        assert second["intent"] == "knowledge_question"
        assert "VPN" in second["question"]
        assert len(second["messages"]) == 4
        assert isinstance(second["messages"][2], HumanMessage)
        assert second["messages"][2].content == "I already tried the first two steps."
        assert isinstance(second["messages"][3], AIMessage)

        # Inspect persisted graph state, not just the returned value.
        snapshot = graph.get_state(config)
        assert len(snapshot.values["messages"]) == 4
        assert snapshot.values["intent"] == "knowledge_question"


def test_checkpoint_threads_are_isolated(tmp_path: Path):
    database = tmp_path / "thread-isolation.sqlite"
    with SqliteSaver.from_conn_string(str(database)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        graph.invoke(
            {"question": "hello", "errors": []},
            config={"configurable": {"thread_id": "thread-A"}},
        )
        graph.invoke(
            {"question": "What is the status of INC1024?", "errors": []},
            config={"configurable": {"thread_id": "thread-B"}},
        )

        state_a = graph.get_state({"configurable": {"thread_id": "thread-A"}})
        state_b = graph.get_state({"configurable": {"thread_id": "thread-B"}})
        assert len(state_a.values["messages"]) == 2
        assert len(state_b.values["messages"]) == 2
        assert state_a.values["messages"][0].content == "hello"
        assert state_b.values["messages"][0].content == "What is the status of INC1024?"


def test_natural_language_next_step_follow_up_uses_previous_vpn_question():
    from src.langgraph_workflow.nodes import classify_query

    state = {
        "question": "What should I try next?",
        "messages": [
            HumanMessage(content="My VPN is not working."),
            AIMessage(content="Try the first two VPN troubleshooting steps."),
            HumanMessage(content="What should I try next?"),
        ],
    }
    update = classify_query(state)
    assert update["intent"] == "knowledge_question"
    assert "My VPN is not working." in update["question"]
    assert "What should I try next?" in update["question"]


def test_next_step_follow_up_skips_intermediate_short_follow_up_turn():
    from src.langgraph_workflow.nodes import classify_query

    state = {
        "question": "What should I try next?",
        "messages": [
            HumanMessage(content="My VPN is not working."),
            AIMessage(content="Try the first two VPN troubleshooting steps."),
            HumanMessage(content="I already tried the first two steps."),
            AIMessage(content="Please continue with the remaining VPN steps."),
            HumanMessage(content="What should I try next?"),
        ],
    }
    update = classify_query(state)
    assert update["intent"] == "knowledge_question"
    assert "My VPN is not working." in update["question"]
