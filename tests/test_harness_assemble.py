from open_deep_research.harness.assemble import assemble_notes
from open_deep_research.harness.checklist import (
    ChecklistDimension,
    ChecklistItem,
    ChecklistStatus,
    ResearchChecklist,
)
from open_deep_research.harness.ledger import ResearchLedger
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
    assert "Required source count" not in assembled
    assert "corroboration_target" not in assembled


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
    assert "note_id=note-material-" in first
    assert "- Source ID: source-" in first
    assert "## unmatched notes" in first
    assert "An unmatched note is not silently discarded" in first


def test_writing_and_verification_consume_source_quote_not_model_quote():
    checklist = ResearchChecklist(
        topic="Any topic",
        items=(item("what-1", ChecklistDimension.WHAT, 1),),
    )
    ledger = ResearchLedger()
    note = ledger.add_note(
        create_note(
            item_id="what-1",
            finding="A repaired note has authoritative source text.",
            quote="alpha beta.",
            url="https://example.com/page",
            source_text="AlphaBeta",
        )
    )

    assembled = assemble_notes(checklist, ledger.notes)
    writing_prompt = build_write_prompt(assembled)
    verification_evidence = source_evidence(note)

    assert note.model_quote == "alpha beta."
    assert note.source_quote == "AlphaBeta"
    assert "note_id=note-000001" in writing_prompt
    assert note.source_id in writing_prompt
    assert "AlphaBeta" in writing_prompt
    assert "alpha beta." not in writing_prompt
    assert verification_evidence is not None
    assert verification_evidence.quote == "AlphaBeta"
    assert verification_evidence.quote != note.model_quote


def test_unlocatable_note_gives_writer_finding_but_never_model_quote():
    checklist = ResearchChecklist(
        topic="Any topic",
        items=(item("what-1", ChecklistDimension.WHAT, 1),),
    )
    source = "The cached source contains different wording."
    note = create_note(
        item_id="what-1",
        finding="The finding remains useful drafting context.",
        quote="MODEL-PROPOSED WORDING THAT IS NOT IN THE SOURCE",
        url="https://example.com/page",
        source_text=source,
    )

    assembled = assemble_notes(checklist, [note])
    writing_prompt = build_write_prompt(assembled)

    assert "The finding remains useful drafting context." in writing_prompt
    assert note.source_id in writing_prompt
    assert "MODEL-PROPOSED WORDING THAT IS NOT IN THE SOURCE" not in (
        writing_prompt
    )
    assert source not in writing_prompt
    assert "Source quote: unavailable" in writing_prompt


def test_scope_excluded_question_stays_out_but_its_evidence_is_not_deleted():
    active = item("what-1", ChecklistDimension.WHAT, 1)
    excluded = item("where-1", ChecklistDimension.WHERE, 2).model_copy(
        update={"status": ChecklistStatus.OUT_OF_SCOPE}
    )
    checklist = ResearchChecklist(
        topic="Explain the event's cause",
        items=(active, excluded),
    )
    notes = (
        create_note(
            item_id="what-1",
            finding="The active causal finding.",
            quote="Active source quote.",
            url="https://active.example/report",
            source_text="Active source quote.",
        ),
        create_note(
            item_id="where-1",
            finding="The filing also identifies the liquidity shortfall.",
            quote="The filing identifies the liquidity shortfall.",
            url="https://excluded.example/office",
            source_text="The filing identifies the liquidity shortfall.",
        ),
    )

    assembled = assemble_notes(checklist, notes)

    assert "The active causal finding." in assembled
    assert excluded.question not in assembled
    assert (
        "## evidence candidates from excluded checklist provenance" in assembled
    )
    assert "provenance_item_id=where-1" in assembled
    assert "The filing also identifies the liquidity shortfall." in assembled
    assert "originating questions are not report requirements" in assembled
    assert "## unmatched notes" not in assembled
