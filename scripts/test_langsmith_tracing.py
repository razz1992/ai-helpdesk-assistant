"""Run one live Gemini request and verify its LangSmith trace was uploaded.

Run from the project root:
    python -m scripts.test_langsmith_tracing
"""
from datetime import datetime, timedelta, timezone
import os
import sys
import time

from dotenv import load_dotenv


def main() -> int:
    # Load project configuration before importing modules that create Settings.
    from src.config import ROOT_DIR

    load_dotenv(ROOT_DIR / ".env", override=True)

    from src.config import settings

    if not settings.langsmith_api_key.strip():
        print("FAIL: LANGSMITH_API_KEY is empty in the project .env.")
        return 1
    if not settings.langsmith_tracing:
        print("FAIL: Set LANGSMITH_TRACING=true in the project .env.")
        return 1
    if settings.provider != "google":
        print("FAIL: This live trace test expects LLM_PROVIDER=google.")
        return 1

    missing = [
        name for name, value in (
            ("LANGSMITH_WORKSPACE_ID", settings.langsmith_workspace_id),
            ("LANGSMITH_ORGANIZATION_ID", settings.langsmith_organization_id),
        ) if not value.strip()
    ]
    if missing:
        print("FAIL: Missing LangSmith scope settings: " + ", ".join(missing))
        return 1

    from langsmith import Client
    from src.langgraph_workflow.graph import run_helpdesk

    project_name = settings.langsmith_project
    client = Client(
        api_key=settings.langsmith_api_key,
        workspace_id=settings.langsmith_workspace_id,
        headers={"X-Organization-Id": settings.langsmith_organization_id},
    )

    try:
        next(iter(client.list_projects(limit=1)), None)
        print("PASS: LangSmith Service Key authenticated for the configured workspace.")
    except Exception as exc:
        print(f"FAIL: LangSmith authentication failed ({type(exc).__name__}).")
        return 1

    started_at = datetime.now(timezone.utc)
    question = "How do I connect to the company VPN?"
    print(f"Running one live Gemini request; LangSmith project: {project_name}")

    try:
        result = run_helpdesk(question)
    except Exception as exc:
        print(f"FAIL: Helpdesk/Gemini invocation failed ({type(exc).__name__}).")
        return 1

    if not result.get("answer"):
        print("FAIL: Gemini workflow returned no answer.")
        return 1

    cutoff = started_at - timedelta(seconds=5)
    last_query_error = None
    for attempt in range(15):
        try:
            runs = list(client.list_runs(
                project_name=project_name,
                start_time=cutoff,
                limit=100,
            ))
            last_query_error = None
        except Exception as exc:
            # Trace ingestion and project indexing are asynchronous. Retry
            # transient API failures during the bounded verification window.
            last_query_error = type(exc).__name__
            runs = []

        root_runs = [
            run for run in runs
            if run.name == "helpdesk_conversation"
        ]
        llm_runs = [
            run for run in runs
            if (run.run_type or "").lower() == "llm"
        ]
        if root_runs and llm_runs:
            print("PASS: Root LangGraph conversation run found.")
            print(f"PASS: Nested LLM run found ({len(llm_runs)} LLM run(s)).")
            print("PASS: Trace is queryable in the configured LangSmith workspace.")
            print("Trace inputs/outputs and credentials were not printed.")
            return 0
        if attempt < 14:
            time.sleep(2)

    print("FAIL: Gemini returned an answer, but the expected root/LLM trace")
    print("was not visible in LangSmith after 30 seconds.")
    if last_query_error:
        print(f"Last LangSmith query error: {last_query_error}")
    print("Check project permissions, endpoint and trace ingestion.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
