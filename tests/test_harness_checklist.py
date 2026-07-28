import asyncio

import pytest
from pydantic import ValidationError

from open_deep_research.harness.checklist import (
    ChecklistDimension,
    ChecklistItem,
    ChecklistStatus,
    ResearchChecklist,
    generate_checklist,
)


class FakeModel:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return self.payload


class FakeLedger:
    def __init__(self):
        self.changes = []

    def record_checklist_change(self, **change):
        self.changes.append(change)


def item(item_id="who-1", status=ChecklistStatus.UNEXPLORED):
    return ChecklistItem(
        item_id=item_id,
        dimension=ChecklistDimension.WHO,
        question="Who is involved?",
        priority=1,
        required_source_count=1,
        status=status,
    )


def test_model_generated_checklist_is_parsed_and_prompt_is_topic_neutral():
    model = FakeModel(
        '{"items":[{"item_id":"who-1","dimension":"who",'
        '"question":"Who is involved?","priority":1,'
        '"corroboration_target":1}]}'
    )

    checklist = asyncio.run(
        generate_checklist("A topic supplied at runtime", model_client=model)
    )

    assert checklist.items == (item(),)
    assert checklist.items[0].corroboration_target == 1
    assert "who, what, when, where, why, and how" in model.prompts[0]
    assert "corroboration_target" in model.prompts[0]
    assert "required_source_count" not in model.prompts[0]
    assert model.prompts[0].endswith("Topic:\nA topic supplied at runtime\n")


def test_legacy_required_source_count_is_read_but_not_reemitted() -> None:
    legacy = item()

    assert legacy.required_source_count == 1
    assert legacy.model_dump(mode="json")["corroboration_target"] == 1
    assert "required_source_count" not in legacy.model_dump(mode="json")


def test_generated_membership_is_immutable_and_delete_is_rejected_and_audited():
    checklist = ResearchChecklist(topic="Any topic", items=(item(),))
    ledger = FakeLedger()

    accepted = checklist.request_delete(
        "who-1", reason="No longer wanted", ledger=ledger
    )

    assert accepted is False
    assert [entry.item_id for entry in checklist.items] == ["who-1"]
    assert ledger.changes == [
        {
            "event": "delete",
            "item_id": "who-1",
            "accepted": False,
            "reason": "No longer wanted",
            "from_status": "unexplored",
        }
    ]
    with pytest.raises(ValidationError):
        checklist.items = ()


def test_even_unknown_delete_request_is_rejected_and_audited():
    checklist = ResearchChecklist(topic="Any topic", items=(item(),))
    ledger = FakeLedger()

    assert checklist.request_delete(
        "missing", reason="Requested anyway", ledger=ledger
    ) is False
    assert ledger.changes[0]["accepted"] is False
    assert ledger.changes[0]["from_status"] is None


def test_append_requires_reason_and_records_accepted_change():
    checklist = ResearchChecklist(topic="Any topic", items=(item(),))
    ledger = FakeLedger()
    addition = item("who-2")

    with pytest.raises(ValueError, match="requires a reason"):
        checklist.append_item(addition, reason=" ", ledger=ledger)

    expanded = checklist.append_item(
        addition, reason="A remaining question emerged", ledger=ledger
    )

    assert [entry.item_id for entry in checklist.items] == ["who-1"]
    assert [entry.item_id for entry in expanded.items] == ["who-1", "who-2"]
    assert ledger.changes[-1]["event"] == "append"
    assert ledger.changes[-1]["accepted"] is True
    assert ledger.changes[-1]["reason"] == "A remaining question emerged"


def test_status_change_returns_new_checklist_and_is_audited():
    checklist = ResearchChecklist(topic="Any topic", items=(item(),))
    ledger = FakeLedger()

    updated = checklist.set_status(
        "who-1",
        ChecklistStatus.HAS_MATERIAL,
        reason="A source was read",
        ledger=ledger,
    )

    assert checklist.get("who-1").status is ChecklistStatus.UNEXPLORED
    assert updated.get("who-1").status is ChecklistStatus.HAS_MATERIAL
    assert ledger.changes == [
        {
            "event": "status_change",
            "item_id": "who-1",
            "accepted": True,
            "reason": "A source was read",
            "from_status": "unexplored",
            "to_status": "has_material",
        }
    ]


def test_exhausted_not_found_is_complete_and_explicitly_not_failure():
    exhausted = item(status=ChecklistStatus.EXHAUSTED_NOT_FOUND)
    checklist = ResearchChecklist(topic="Any topic", items=(exhausted,))

    assert ChecklistStatus.EXHAUSTED_NOT_FOUND.is_terminal is True
    assert ChecklistStatus.EXHAUSTED_NOT_FOUND.is_failure is False
    assert exhausted.is_complete is True
    assert exhausted.is_failed is False
    assert checklist.is_complete is True
    assert checklist.has_failures is False
