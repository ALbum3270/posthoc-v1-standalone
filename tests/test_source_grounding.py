from open_deep_research.graphrag.ontology import OntologySlot
from open_deep_research.graphrag.schemas import SourceDocument
from open_deep_research.graphrag.validation.grounding import (
    ground_extracted_row,
    locate_source_quote,
    normalized_numbers,
    row_is_numerically_supported,
)


SLOT = OntologySlot(
    slot_id="what.scale",
    dimension="WHAT",
    label="Scale",
    question="What was the scale?",
)


def document(content: str) -> SourceDocument:
    return SourceDocument(
        document_id="doc-1",
        title="Source",
        url="https://example.com/source",
        content=content,
    )


def test_exact_quote_returns_offsets_into_source() -> None:
    content = "Before.\nFTX reported liabilities of $8 billion.\nAfter."

    span = locate_source_quote(content, "FTX reported liabilities of $8 billion.")

    assert span is not None
    assert content[span.start_char : span.end_char] == span.quote


def test_whitespace_only_differences_are_allowed_but_source_text_wins() -> None:
    content = "FTX reported\nliabilities   of $8 billion."

    span = locate_source_quote(content, "FTX reported liabilities of $8 billion.")

    assert span is not None
    assert span.quote == "FTX reported\nliabilities   of $8 billion."


def test_quote_not_present_is_rejected() -> None:
    assert locate_source_quote("The page says something else.", "Invented fact") is None


def test_numbers_are_normalized_for_subset_comparison() -> None:
    assert normalized_numbers("about $8,000 and 12 %") == {"8000", "12%"}
    row = {
        "subject": "FTX",
        "predicate": "owed",
        "object": "$8,000",
    }
    assert row_is_numerically_supported(row, "The filing says FTX owed $8,000.")


def test_invented_number_is_rejected_even_when_quote_is_real() -> None:
    source = document("The filing says FTX owed $8 billion.")
    row = {
        "subject": "FTX",
        "predicate": "owed",
        "object": "$9 billion",
        "quote": "The filing says FTX owed $8 billion.",
    }

    assert ground_extracted_row(document=source, slot=SLOT, row=row) is None


def test_grounded_row_uses_verbatim_source_quote() -> None:
    source = document("The filing says FTX owed $8 billion to customers.")
    row = {
        "subject": "FTX",
        "predicate": "owed",
        "object": "$8 billion",
        "quote": "The filing says FTX owed $8 billion to customers.",
    }

    triple = ground_extracted_row(document=source, slot=SLOT, row=row)

    assert triple is not None
    assert triple.source_span is not None
    assert triple.source_span.quote == row["quote"]
    assert source.content[
        triple.source_span.start_char : triple.source_span.end_char
    ] == row["quote"]


def test_missing_quote_is_not_treated_as_grounded() -> None:
    row = {"subject": "FTX", "predicate": "owed", "object": "$8 billion"}

    assert ground_extracted_row(
        document=document("FTX owed $8 billion."),
        slot=SLOT,
        row=row,
    ) is None
