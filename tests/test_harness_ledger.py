import json

import pytest

from open_deep_research.harness.checklist import (
    ChecklistDimension,
    ChecklistItem,
    ResearchChecklist,
)
from open_deep_research.harness.ledger import (
    ResearchLedger,
    SettlementEvidence,
)
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
    assert payload["notes"][0]["model_quote"] == "Wording not present."
    assert payload["notes"][0]["source_quote"] is None
    assert payload["notes"][0]["failure_reason"] == "quote_not_found"
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


def test_legacy_serialized_notes_receive_stable_ids_when_loaded():
    note = create_note(
        item_id="what-1",
        finding="A finding.",
        quote="Original source wording.",
        url="https://example.com/source",
        source_text="Original source wording.",
    )
    legacy_note = note.model_dump(
        mode="json",
        exclude={"note_id", "source_id"},
    )

    first_load = ResearchLedger.model_validate({"notes": [legacy_note]})
    second_load = ResearchLedger.model_validate({"notes": [legacy_note]})

    assert first_load.notes[0].note_id == "note-000001"
    assert second_load.notes[0].note_id == "note-000001"
    assert first_load.notes[0].source_id == second_load.notes[0].source_id


def test_gap_notes_do_not_rewrite_settle_time_evidence_snapshot():
    ledger = ResearchLedger()
    ledger.record_checklist_change(
        event="status_update",
        item_id="what-1",
        accepted=True,
        reason="settled during initial collection",
        from_status="has_material",
        to_status="settled",
        settlement_evidence=SettlementEvidence(),
    )
    text = "Later source wording."
    url = "https://later.example/source"
    ledger.cache_source(url, text)
    note = ledger.add_note(
        create_note(
            item_id="what-1",
            finding="A post-draft finding.",
            quote=text,
            url=url,
            source_text=text,
        )
    )
    ledger.record_evidence_gap(
        event="source_acquired",
        url=url,
        note_ids=(note.note_id,),
    )

    assert ledger.settled_without_located_evidence_item_ids == ("what-1",)
    assert ledger.settled_without_located_evidence == 1
    assert ledger.evidence_gap_history[0].event == "source_acquired"
    assert ledger.rounds == []
