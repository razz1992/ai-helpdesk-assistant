"""CLI entry point for the AI Helpdesk Assistant."""
from src.langgraph_workflow.graph import run_helpdesk

def main() -> None:
    print("\nAI Helpdesk Assistant — LangChain + LangGraph + LangSmith")
    print("Demo data only. Type 'exit' to quit.\n")
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
            result = run_helpdesk(question)
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
