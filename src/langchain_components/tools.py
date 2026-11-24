"""Deterministic local mock-ticket tool; no external system is contacted."""
import json
from pathlib import Path

from langchain_core.tools import tool

from src.config import TICKETS_FILE


def lookup_ticket_record(ticket_id: str) -> dict | None:
    """Look up one ticket in the local synthetic JSON dataset."""
    records = json.loads(Path(TICKETS_FILE).read_text(encoding="utf-8"))
    return records.get(ticket_id.strip().upper())


@tool
def check_ticket_status(ticket_id: str) -> str:
    """Check a support ticket by ID, for example INC1024."""
    record = lookup_ticket_record(ticket_id)
    if record is None:
        return f"Ticket {ticket_id.strip().upper()} was not found in the demo dataset."
    return json.dumps(
        {"ticket_id": ticket_id.strip().upper(), **record},
        ensure_ascii=False,
    )
