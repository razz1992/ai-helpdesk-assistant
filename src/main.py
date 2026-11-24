"""CLI entry point for the persistent AI Helpdesk Assistant."""
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import DATA_DIR
from src.langgraph_workflow.graph import build_graph, run_helpdesk


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = DATA_DIR / "conversations.sqlite"
    print("\nAI Helpdesk Assistant — LangChain + LangGraph + LangSmith")
    print("SQLite-backed conversations. Type 'exit' to quit.\n")
    resume_id = input("Conversation ID to resume (leave blank for new): ").strip()
    thread_id = resume_id or str(uuid4())
    print(f"Conversation ID: {thread_id}")
    print("Keep this ID to resume this conversation after restarting the app.\n")

    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if question.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break
            if not question:
                continue
            try:
                result = run_helpdesk(question, thread_id=thread_id, graph=graph)
                print(f"\nIntent: {result.get('intent', 'unknown')}")
                print(f"Assistant: {result.get('answer', 'No answer was produced.')}")
                if result.get("needs_human_review"):
                    print(f"Escalation reason: {result.get('escalation_reason')}")
                print()
            except Exception as exc:
                print(f"\nApplication error: {exc}")
                print("Check .env, provider availability and installed dependencies.\n")


if __name__ == "__main__":
    main()
