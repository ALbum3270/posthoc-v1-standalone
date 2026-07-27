from open_deep_research.harness.assemble import assemble_notes
from open_deep_research.harness.checklist import (
    ChecklistDimension,
    ChecklistItem,
    ResearchChecklist,
)
from open_deep_research.harness.notes import create_note


def item(item_id, dimension, priority):
    return ChecklistItem(
        item_id=item_id,
        dimension=dimension,
        question=f"Question for {item_id}?",
        priority=priority,
        required_source_count=1,
    )


def test_assembly_orders_dimension_then_priority_then_item_id():
    checklist = ResearchChecklist(
        topic="Any topic",
        items=(
            item("how-z", ChecklistDimension.HOW, 1),
            item("what-b", ChecklistDimension.WHAT, 2),
            item("who-c", ChecklistDimension.WHO, 3),
            item("what-a", ChecklistDimension.WHAT, 1),
            item("who-a", ChecklistDimension.WHO, 1),
        ),
    )

    assembled = assemble_notes(checklist, [])

    headings = [
        "## who | priority=1 | item_id=who-a",
        "## who | priority=3 | item_id=who-c",
        "## what | priority=1 | item_id=what-a",
        "## what | priority=2 | item_id=what-b",
        "## how | priority=1 | item_id=how-z",
    ]
    assert [assembled.index(heading) for heading in headings] == sorted(
        assembled.index(heading) for heading in headings
    )


def test_assembly_is_byte_reproducible_and_preserves_all_notes():
    checklist = ResearchChecklist(
        topic="Any topic",
        items=(
            item("what-1", ChecklistDimension.WHAT, 1),
            item("who-1", ChecklistDimension.WHO, 1),
        ),
    )
    source = "First exact quote.\n\nSecond exact quote."
    notes = [
        create_note(
            item_id="what-1",
            finding="Second finding",
            quote="Second exact quote.",
            url="https://b.example.com/page",
            source_text=source,
        ),
        create_note(
            item_id="what-1",
            finding="First finding",
            quote="First exact quote.",
            url="https://a.example.com/page",
            source_text=source,
        ),
        create_note(
            item_id="who-1",
            finding="An unlocatable finding is still retained",
            quote="Absent quote",
            url="https://example.net/page",
            source_text=source,
        ),
        create_note(
            item_id="unmatched-1",
            finding="An unmatched note is not silently discarded",
            quote="First exact quote.",
            url="https://example.org/page",
            source_text=source,
        ),
    ]

    first = assemble_notes(checklist, notes)
    second = assemble_notes(checklist, notes)
    reordered = assemble_notes(checklist, reversed(notes))

    assert first == second
    assert first == reordered
    assert first.index("item_id=who-1") < first.index("item_id=what-1")
    assert first.index('"First finding"') < first.index('"Second finding"')
    assert "Absent quote" in first
    assert "Location: unlocatable" in first
    assert "## unmatched notes" in first
    assert "An unmatched note is not silently discarded" in first
