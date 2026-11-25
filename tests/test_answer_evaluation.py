from evaluation.evaluate_answers import DIMENSIONS, evaluate_dimensions, retrieval_metrics


def test_evaluation_matrix_has_five_distinct_dimensions():
    assert DIMENSIONS == (
        "answer_correctness", "reasoning_quality", "groundedness",
        "completeness", "safety",
    )


def test_retrieval_precision_recall_and_hit():
    metrics = retrieval_metrics(
        {"retrieved_kb": ["KB001", "KB002", "KB003"]},
        {"expected_kb": ["KB002"]},
    )
    assert metrics == {
        "precision_at_3": 1 / 3,
        "recall_at_3": 1.0,
        "retrieval_hit": True,
    }


def test_retrieval_miss_and_not_applicable():
    miss = retrieval_metrics({"retrieved_kb": ["KB004"]}, {"expected_kb": ["KB001"]})
    assert miss["retrieval_hit"] is False
    assert miss["recall_at_3"] == 0.0
    n_a = retrieval_metrics({"retrieved_kb": []}, {})
    assert n_a == {"precision_at_3": None, "recall_at_3": None, "retrieval_hit": None}


def test_each_matrix_dimension_is_scored_separately(monkeypatch):
    from evaluation import evaluate_answers

    monkeypatch.setattr(
        evaluate_answers, "judge",
        lambda dimension, *args: {"score": 4, "rationale": f"{dimension} checked"},
    )
    example = {
        "inputs": {"question": "How do I reset my password?"},
        "outputs": {"reference_answer": "Use approved reset process."},
        "metadata": {"expected_kb": ["KB001"]},
    }
    prediction = {
        "answer": "Use approved reset process.",
        "context": "Approved password reset knowledge article.",
        "retrieved_kb": ["KB001", "KB002"],
    }
    scores = evaluate_dimensions(prediction, example)
    for dimension in DIMENSIONS:
        assert scores[dimension]["score"] == 4
    assert scores["retrieval"]["retrieval_hit"] is True
