import json

import pytest

from open_deep_research.harness.checklist import (
    ChecklistDimension,
    ChecklistItem,
    ResearchChecklist,
)
from open_deep_research.harness.ledger import ResearchLedger
from open_deep_research.harness.notes import NoteLocationStatus, create_note


def checklist():
    return ResearchChecklist(
        topic="Any topic",
        items=(
            ChecklistItem(
                item_id="what-1",
                dimension=ChecklistDimension.WHAT,
                question="What happened?",
                priority=1,
                required_source_count=2,
            ),
        ),
    )


def test_ledger_serializes_rounds_sources_notes_and_rejected_deletions():
    ledger = ResearchLedger(research_id="run-1", topic="Any topic")
    full_text = "First paragraph.\n\nLast paragraph that must remain cached."
    ledger.record_round(
        round_number=1,
        action="search",
        query="a neutral query",
        result_summary="Two results",
        token_count=17,
        cost_usd=0.004,
    )
    ledger.cache_source("https://example.com/source", full_text)
    note = create_note(
        item_id="what-1",
        finding="The attempted quote is retained.",
        quote="Wording not present.",
        url="https://example.com/source",
        source_text=full_text,
    )
    ledger.add_note(note)

    accepted = checklist().request_delete(
        "what-1", reason="Requested by a later model turn", ledger=ledger
    )
    payload = json.loads(ledger.to_audit_json())

    assert accepted is False
    assert payload["rounds"][0]["token_count"] == 17
    assert payload["rounds"][0]["cost_usd"] == 0.004
    assert payload["source_cache"]["https://example.com/source"] == full_text
    assert payload["notes"][0]["location_status"] == "unlocatable"
    assert payload["notes"][0]["quote"] == "Wording not present."
    assert payload["checklist_history"] == [
        {
            "accepted": False,
            "event": "delete",
            "from_status": "unexplored",
            "item_id": "what-1",
            "reason": "Requested by a later model turn",
            "to_status": None,
        }
    ]
    assert ledger.total_tokens == 17
    assert ledger.total_cost_usd == 0.004
    assert note.location_status is NoteLocationStatus.UNLOCATABLE


def test_source_cache_is_idempotent_but_does_not_overwrite_changed_text():
    ledger = ResearchLedger()

    assert ledger.cache_source("https://example.com", "original") is True
    assert ledger.cache_source("https://example.com", "original") is False
    with pytest.raises(ValueError, match="cached source changed"):
        ledger.cache_source("https://example.com", "replacement")

    assert ledger.get_source("https://example.com") == "original"
