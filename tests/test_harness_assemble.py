from open_deep_research.harness.assemble import assemble_notes
from open_deep_research.harness.checklist import (
    ChecklistDimension,
    ChecklistItem,
    ResearchChecklist,
)
from open_deep_research.harness.notes import create_note, source_evidence
from open_deep_research.harness.write import build_write_prompt


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
    assert "Absent quote" not in first
    assert "Source quote: unavailable" in first
    assert "Location: unlocatable" in first
    assert "## unmatched notes" in first
    assert "An unmatched note is not silently discarded" in first


def test_writing_and_verification_consume_source_quote_not_model_quote():
    checklist = ResearchChecklist(
        topic="Any topic",
        items=(item("what-1", ChecklistDimension.WHAT, 1),),
    )
    note = create_note(
        item_id="what-1",
        finding="A repaired note has authoritative source text.",
        quote="alpha beta.",
        url="https://example.com/page",
        source_text="AlphaBeta",
    )

    assembled = assemble_notes(checklist, [note])
    writing_prompt = build_write_prompt(assembled)
    verification_evidence = source_evidence(note)

    assert note.model_quote == "alpha beta."
    assert note.source_quote == "AlphaBeta"
    assert "AlphaBeta" in writing_prompt
    assert "alpha beta." not in writing_prompt
    assert verification_evidence is not None
    assert verification_evidence.quote == "AlphaBeta"
    assert verification_evidence.quote != note.model_quote
