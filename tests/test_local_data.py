"""Offline tests for deterministic data and ticket behaviour."""
from src.langchain_components.tools import lookup_ticket_record
from evaluation.dataset import load_examples

def test_known_ticket_is_returned():
    record = lookup_ticket_record("INC1024")
    assert record is not None
    assert record["status"] == "In Progress"

def test_unknown_ticket_is_not_invented():
    assert lookup_ticket_record("INC9999") is None

def test_evaluation_dataset_has_twelve_cases():
    assert len(load_examples()) == 12
