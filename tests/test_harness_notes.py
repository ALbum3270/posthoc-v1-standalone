from open_deep_research.harness.notes import (
    NoteExtractionMode,
    NoteLocationStatus,
    QuoteFailureReason,
    QuoteRepairMethod,
    create_note,
    create_note_from_segment_range,
    source_evidence,
)
from open_deep_research.harness.source_spans import build_source_span_registry


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
    assert source[note.span.start_char : note.span.end_char] == note.source_quote
    assert note.model_quote == "A source-backed sentence."
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
    assert note.model_quote == "The source has variable spacing."
    assert note.source_quote == "The source has\nvariable spacing."


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
    assert note.model_quote == requested_quote
    assert note.source_quote is None
    assert note.span is None
    assert note.failure_reason is QuoteFailureReason.QUOTE_NOT_FOUND


def test_unique_format_repair_keeps_model_and_source_quotes_separate():
    source = "Ａlpha—Beta 2024"

    note = create_note(
        item_id="what-1",
        finding="A formatting-only difference can be repaired.",
        quote="alpha beta 2024.",
        url="https://example.com/article",
        source_text=source,
    )

    assert note.location_status is NoteLocationStatus.REPAIRED_LOCATABLE
    assert note.is_locatable is False
    assert note.has_usable_source_span is True
    assert note.model_quote == "alpha beta 2024."
    assert note.source_quote == source
    assert note.repair_method is (
        QuoteRepairMethod.NFKC_CASEFOLD_ALNUM_UNIQUE_CONTIGUOUS
    )
    assert note.span is not None
    assert source[note.span.start_char : note.span.end_char] == note.source_quote


def test_repair_rejects_ambiguous_numeric_and_overexpanded_candidates():
    ambiguous = create_note(
        item_id="what-1",
        finding="An ambiguous match is not evidence.",
        quote="ALPHA BETA!",
        url="https://example.com/ambiguous",
        source_text="Alpha beta. Later, alpha-beta appears again.",
    )
    numeric = create_note(
        item_id="what-1",
        finding="Numeric token boundaries cannot change.",
        quote="Value 12-34",
        url="https://example.com/numeric",
        source_text="Value 1234",
    )
    expanded = create_note(
        item_id="what-1",
        finding="Removed punctuation cannot hide an unbounded source span.",
        quote="AB",
        url="https://example.com/expanded",
        source_text="A" + ("-" * 65) + "B",
    )

    assert ambiguous.failure_reason is (
        QuoteFailureReason.AMBIGUOUS_FORMAT_MATCH
    )
    assert numeric.failure_reason is (
        QuoteFailureReason.NUMBER_SEQUENCE_MISMATCH
    )
    assert expanded.failure_reason is QuoteFailureReason.REPAIR_SPAN_TOO_LARGE
    assert all(
        note.location_status is NoteLocationStatus.UNLOCATABLE
        for note in (ambiguous, numeric, expanded)
    )


def test_noncontiguous_diagnostic_never_creates_fragment_evidence():
    source = "First continuous passage.\n\nIntervening text.\n\nSecond passage."
    note = create_note(
        item_id="how-1",
        finding="One finding was backed by two separate passages.",
        quote="First continuous passage. ... Second passage.",
        url="https://example.com/composite",
        source_text=source,
    )

    assert note.location_status is NoteLocationStatus.UNLOCATABLE
    assert note.failure_reason is (
        QuoteFailureReason.NONCONTIGUOUS_COMPOSITE
    )
    assert note.located_fragment_count == 2
    assert note.source_quote is None
    assert note.span is None
    assert source_evidence(note) is None
    assert "fragments" not in note.model_dump()


def test_segment_pointer_note_uses_only_the_authoritative_source_slice():
    source = "First sentence. Second sentence.\n\nAnother paragraph."
    registry = build_source_span_registry(source)

    note = create_note_from_segment_range(
        item_id="what-1",
        finding="A continuous two-sentence passage supports the finding.",
        start_segment_id="S000001",
        end_segment_id="S000002",
        url="https://example.com/article",
        source_text=source,
        registry=registry,
    )

    assert note.extraction_mode is NoteExtractionMode.SEGMENT_POINTER
    assert note.model_quote is None
    assert note.source_quote == "First sentence. Second sentence."
    assert note.span is not None
    assert source[note.span.start_char : note.span.end_char] == note.source_quote
    assert note.start_segment_id == "S000001"
    assert note.end_segment_id == "S000002"
    assert note.span_registry_id == registry.registry_id
    assert note.source_text_sha256 == registry.source_text_sha256
    assert note.segmentation_version == registry.segmentation_version


def test_legacy_free_text_note_remains_explicitly_legacy():
    note = create_note(
        item_id="what-1",
        finding="The historical extraction path remains auditable.",
        quote="Exact historical quote.",
        url="https://example.com/legacy",
        source_text="Exact historical quote.",
    )

    assert note.extraction_mode is NoteExtractionMode.LEGACY_FREE_TEXT
    assert note.model_quote == "Exact historical quote."
    assert note.start_segment_id is None
    assert note.span_registry_id is None
