"""Local workflow evaluation plus optional LangSmith experiment.

Local mode evaluates intent, route and retrieval@3 without calling a chat API:
    python -m evaluation.evaluate

LangSmith mode uploads the synthetic dataset and runs the same deterministic
workflow target as an experiment:
    python -m evaluation.evaluate --langsmith
"""
import os
import sys

from src.langchain_components.retriever import get_retriever
from src.langgraph_workflow.graph import route_by_intent
from src.langgraph_workflow.nodes import classify_query
from evaluation.dataset import load_examples

DATASET_NAME = "ai-helpdesk-assistant-routing-v1"

ROUTE_LABELS = {
    "greeting_answer": "greeting",
    "retrieve_knowledge": "knowledge",
    "lookup_ticket": "ticket_lookup",
    "prepare_escalation": "escalation",
    "fallback_answer": "fallback",
}


def evaluate_case(example: dict, include_retrieval: bool = True) -> dict:
    question = example["inputs"]["question"]
    reference = example["reference"]
    classified = classify_query({"question": question})
    route = route_by_intent(classified)
    actual = {
        "intent": classified["intent"],
        "route": ROUTE_LABELS[route],
        "retrieved_kb": [],
    }
    expected_kb = reference.get("expected_kb", [])
    # Retrieval is only executed on the knowledge route. Escalation cases may
    # mention a KB article in their reference but intentionally bypass retrieval.
    if include_retrieval and actual["intent"] == "knowledge_question":
        docs = get_retriever().invoke(question)
        actual["retrieved_kb"] = sorted({
            doc.metadata.get("document_id")
            for doc in docs
            if doc.metadata.get("document_id")
        })
    actual["intent_correct"] = actual["intent"] == reference["expected_intent"]
    actual["route_correct"] = actual["route"] == reference["expected_route"]
    actual["retrieval_hit"] = (
        bool(set(expected_kb).intersection(actual["retrieved_kb"]))
        if expected_kb and actual["intent"] == "knowledge_question" else None
    )
    return actual


def run_local_evaluation() -> None:
    examples = load_examples()
    results = []
    for example in examples:
        result = evaluate_case(example)
        results.append((example["metadata"]["case_id"], result))

    intent_correct = sum(result["intent_correct"] for _, result in results)
    route_correct = sum(result["route_correct"] for _, result in results)
    retrieval_cases = [result for _, result in results if result["retrieval_hit"] is not None]
    retrieval_hits = sum(result["retrieval_hit"] for result in retrieval_cases)

    print("AI Helpdesk Assistant — Local Evaluation")
    print("-" * 68)
    print(f"{'Case':<8}{'Intent':<12}{'Route':<14}{'Retrieval@3':<14}")
    for case_id, result in results:
        retrieval = (
            "N/A" if result["retrieval_hit"] is None
            else ("PASS" if result["retrieval_hit"] else "MISS")
        )
        print(
            f"{case_id:<8}{'PASS' if result['intent_correct'] else 'FAIL':<12}"
            f"{'PASS' if result['route_correct'] else 'FAIL':<14}{retrieval:<14}"
        )
    print("-" * 68)
    print(f"Intent accuracy: {intent_correct}/{len(results)} ({intent_correct / len(results):.1%})")
    print(f"Route accuracy: {route_correct}/{len(results)} ({route_correct / len(results):.1%})")
    if retrieval_cases:
        print(
            f"Retrieval hit@3: {retrieval_hits}/{len(retrieval_cases)} "
            f"({retrieval_hits / len(retrieval_cases):.1%})"
        )


def run_langsmith_evaluation() -> None:
    if not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError(
            "LANGSMITH_API_KEY is not configured in the project .env file. "
            "Add a LangSmith API key, then rerun with --langsmith."
        )

    from langsmith import Client
    from langsmith.evaluation import evaluate

    client = Client()
    examples = load_examples()
    dataset = next(
        client.list_datasets(dataset_name=DATASET_NAME, limit=1),
        None,
    )
    if dataset is None:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=(
                "Synthetic IT helpdesk routing and retrieval cases for the "
                "LangGraph learning project."
            ),
        )
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "inputs": item["inputs"],
                    "outputs": {
                        "intent": item["reference"]["expected_intent"],
                        "route": item["reference"]["expected_route"],
                        "expected_kb": item["reference"].get("expected_kb", []),
                    },
                    "metadata": item["metadata"],
                }
                for item in examples
            ],
        )
        print(f"Created LangSmith dataset: {DATASET_NAME}")
    else:
        print(f"Using existing LangSmith dataset: {DATASET_NAME}")

    def target(inputs: dict) -> dict:
        question = inputs["question"]
        classified = classify_query({"question": question})
        route = route_by_intent(classified)
        retrieved_kb = []
        if classified["intent"] == "knowledge_question":
            docs = get_retriever().invoke(question)
            retrieved_kb = sorted({
                doc.metadata.get("document_id")
                for doc in docs
                if doc.metadata.get("document_id")
            })
        return {
            "intent": classified["intent"],
            "route": ROUTE_LABELS[route],
            "retrieved_kb": retrieved_kb,
        }

    def workflow_correctness(run, example) -> dict:
        prediction = run.outputs or {}
        reference = example.outputs or {}
        intent_ok = prediction.get("intent") == reference.get("intent")
        route_ok = prediction.get("route") == reference.get("route")
        expected_kb = set(reference.get("expected_kb", []))
        retrieved_kb = set(prediction.get("retrieved_kb", []))
        retrieval_ok = (
            not expected_kb
            or prediction.get("intent") != "knowledge_question"
            or bool(expected_kb.intersection(retrieved_kb))
        )
        return {
            "key": "workflow_correctness",
            "score": intent_ok and route_ok and retrieval_ok,
            "comment": (
                f"intent={intent_ok}; route={route_ok}; "
                f"retrieval_hit={retrieval_ok if expected_kb else 'not_applicable'}"
            ),
        }

    results = evaluate(
        target,
        data=dataset.name,
        evaluators=[workflow_correctness],
        client=client,
        experiment_prefix="helpdesk-routing-retrieval",
        description=(
            "Evaluates deterministic LangGraph intent/route decisions and "
            "MiniLM + FAISS retrieval against the synthetic 12-case dataset. "
            "Does not invoke the chat model."
        ),
        max_concurrency=1,
        metadata={"embedding_model": "sentence-transformers/all-MiniLM-L6-v2"},
    )
    print(f"LangSmith experiment: {results.experiment_name}")


if __name__ == "__main__":
    if "--langsmith" in sys.argv:
        run_langsmith_evaluation()
    else:
        run_local_evaluation()
