from open_deep_research.harness.notes import (
    NoteLocationStatus,
    create_note,
)


def test_locatable_note_uses_exact_source_slice_and_offsets():
    source = "Opening paragraph.\n\nA source-backed sentence.\n\nClosing paragraph."

    note = create_note(
        item_id="what-1",
        finding="The source contains a relevant statement.",
        quote="A source-backed sentence.",
        url="https://www.example.com/article",
        source_text=source,
    )

    assert note.location_status is NoteLocationStatus.LOCATABLE
    assert note.is_locatable is True
    assert note.publisher == "example.com"
    assert note.span is not None
    assert source[note.span.start_char : note.span.end_char] == note.quote
    assert note.span.model_dump() == {
        "start_char": source.index("A source-backed sentence."),
        "end_char": source.index("A source-backed sentence.")
        + len("A source-backed sentence."),
    }


def test_whitespace_tolerant_location_still_stores_verbatim_source_text():
    source = "The source has\nvariable spacing."

    note = create_note(
        item_id="how-1",
        finding="Spacing does not change the quote's location.",
        quote="The source has variable spacing.",
        url="https://example.com/article",
        source_text=source,
    )

    assert note.is_locatable is True
    assert note.quote == "The source has\nvariable spacing."


def test_unlocatable_note_is_marked_and_preserved_without_exception():
    requested_quote = "  This wording is absent from the source.  "

    note = create_note(
        item_id="why-1",
        finding="A model-produced finding is retained for later audit.",
        quote=requested_quote,
        url="https://example.com/article",
        source_text="The cached source says something else.",
    )

    assert note.location_status is NoteLocationStatus.UNLOCATABLE
    assert note.is_locatable is False
    assert note.quote == requested_quote
    assert note.span is None
