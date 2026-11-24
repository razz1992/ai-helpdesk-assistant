"""Live two-turn SQLite conversation test with LangSmith trace verification.

Run from the project root:
    python -m scripts.test_persistent_conversation
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import time
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver


def main() -> int:
    from src.config import settings
    from src.langgraph_workflow.graph import (
        _build_langsmith_client, build_graph, run_helpdesk,
    )

    if not settings.langsmith_tracing:
        print("FAIL: Enable LANGSMITH_TRACING in the project .env.")
        return 1

    client = _build_langsmith_client()
    try:
        next(iter(client.list_projects(limit=1)), None)
    except Exception as exc:
        print(f"FAIL: LangSmith authentication ({type(exc).__name__}).")
        return 1

    thread_id = f"module1-{uuid4()}"
    started_at = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory(prefix="helpdesk-module1-") as temp_dir:
        database = Path(temp_dir) / "conversation.sqlite"
        try:
            with SqliteSaver.from_conn_string(str(database)) as checkpointer:
                graph = build_graph(checkpointer=checkpointer)
                first = run_helpdesk(
                    "My VPN is not working.",
                    thread_id=thread_id,
                    graph=graph,
                )
                second = run_helpdesk(
                    "I already tried the first two steps.",
                    thread_id=thread_id,
                    graph=graph,
                )
                snapshot = graph.get_state(
                    {"configurable": {"thread_id": thread_id}}
                )
                messages = snapshot.values.get("messages", [])
                if first.get("intent") != "knowledge_question":
                    print("FAIL: Initial VPN question did not use knowledge retrieval.")
                    return 1
                if second.get("intent") != "knowledge_question":
                    print("FAIL: Follow-up did not resolve to the VPN knowledge route.")
                    return 1
                if len(messages) != 4:
                    print("FAIL: Expected two user and two assistant messages in checkpoint.")
                    return 1
            print("PASS: SQLite checkpoint retained both turns.")
            print("PASS: Follow-up was routed with prior VPN context.")
        except Exception as exc:
            print(f"FAIL: Persistent conversation invocation ({type(exc).__name__}).")
            return 1

    cutoff = started_at - timedelta(seconds=5)
    last_error = None
    for attempt in range(20):
        try:
            runs = list(client.list_runs(
                project_name=settings.langsmith_project,
                start_time=cutoff,
                limit=200,
            ))
            last_error = None
        except Exception as exc:
            last_error = type(exc).__name__
            runs = []
        roots = [run for run in runs if run.name == "helpdesk_conversation"]
        llms = [run for run in runs if (run.run_type or "").lower() == "llm"]
        if len(roots) >= 2 and len(llms) >= 2:
            print(f"PASS: LangSmith contains {len(roots)} conversation root runs.")
            print(f"PASS: LangSmith contains {len(llms)} nested LLM runs.")
            print("PASS: Both turns were traced in the configured LangSmith project.")
            print("Conversation contents and credentials were not printed.")
            return 0
        if attempt < 19:
            time.sleep(2)

    print("FAIL: Expected two root runs and two nested LLM runs were not queryable.")
    if last_error:
        print(f"Last LangSmith query error: {last_error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
