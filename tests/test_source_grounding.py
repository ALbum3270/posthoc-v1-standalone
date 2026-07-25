import pytest

from open_deep_research.graphrag.ontology import OntologySlot
from open_deep_research.graphrag.schemas import SourceDocument
from open_deep_research.graphrag.validation.grounding import (
    expand_span_to_sentence,
    ground_extracted_row,
    locate_source_quote,
    normalized_numbers,
    quote_names_subject,
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


def test_missing_subject_clause_expands_to_verbatim_source_sentence() -> None:
    content = "FTX\nFiled for Chapter 11 bankruptcy in November 2022."
    source = document(content)
    row = {
        "subject": "FTX",
        "predicate": "filed for",
        "object": "Chapter 11 bankruptcy in November 2022",
        "quote": "Filed for Chapter 11 bankruptcy in November 2022",
    }

    triple = ground_extracted_row(document=source, slot=SLOT, row=row)

    assert triple is not None
    assert triple.source_span is not None
    assert triple.source_span.quote == content
    assert content[
        triple.source_span.start_char : triple.source_span.end_char
    ] == triple.source_span.quote


def test_dangling_and_clause_expands_to_a_self_contained_sentence() -> None:
    content = (
        "On November 11, Alameda Research and FTX declared bankruptcy, "
        "and Bankman-Fried stepped down as CEO of FTX."
    )
    row = {
        "subject": "Bankman-Fried",
        "predicate": "stepped down",
        "object": "as CEO of FTX",
        "quote": "and Bankman-Fried stepped down as CEO of FTX",
    }

    triple = ground_extracted_row(document=document(content), slot=SLOT, row=row)

    assert triple is not None
    assert triple.source_span is not None
    assert triple.source_span.quote == content


def test_dangling_but_clause_is_rejected_when_full_sentence_stays_anaphoric() -> None:
    content = (
        "Their competitor agreed to buy FTX on November 8, 2022, "
        "but backed out of the deal on November 9."
    )
    row = {
        "subject": "Binance",
        "predicate": "backed out",
        "object": "of the deal on November 9",
        "quote": "but backed out of the deal on November 9",
    }

    assert ground_extracted_row(
        document=document(content),
        slot=SLOT,
        row=row,
    ) is None


@pytest.mark.parametrize(
    ("content", "subject", "quote"),
    [
        ("They also asked Binance for help.", "FTX", "They also asked Binance for help"),
        (
            "Their competitor agreed to buy FTX on November 8, 2022.",
            "Binance",
            "Their competitor agreed to buy FTX on November 8, 2022",
        ),
        (
            "This triggered a competing exchange, Binance, to sell its holdings.",
            "Binance",
            "This triggered a competing exchange, Binance, to sell its holdings",
        ),
        ("该公司随后发布了公告。", "北京大学", "该公司随后发布了公告"),
        ("其随后离开了现场。", "张伟", "其随后离开了现场"),
    ],
)
def test_unresolved_english_and_chinese_pronoun_subjects_are_rejected(
    content: str,
    subject: str,
    quote: str,
) -> None:
    row = {
        "subject": subject,
        "predicate": "acted",
        "object": "event",
        "quote": quote,
    }

    assert ground_extracted_row(
        document=document(content),
        slot=SLOT,
        row=row,
    ) is None


def test_anaphoric_company_possessive_is_rejected() -> None:
    content = (
        "The company's top fifty creditors, which included large financial "
        "firms, were owed over $3 billion."
    )
    row = {
        "subject": "FTX's top fifty creditors",
        "predicate": "were owed",
        "object": "over $3 billion",
        "quote": content,
    }

    assert ground_extracted_row(
        document=document(content),
        slot=SLOT,
        row=row,
    ) is None


@pytest.mark.parametrize(
    ("subject", "quote"),
    [
        ("张伟", "张伟于周一提交了报告。"),
        ("北京大学", "北京大学发布了研究结果。"),
        ("Silicon Valley Bank", "SVB announced the decision."),
        ("Sam Bankman-Fried", "Bankman-Fried stepped down."),
    ],
)
def test_subject_naming_is_language_aware_and_supports_structural_short_names(
    subject: str,
    quote: str,
) -> None:
    assert quote_names_subject(quote, subject)


def test_subject_name_cannot_be_weakened_to_an_arbitrary_name_fragment() -> None:
    assert not quote_names_subject(
        "New York approved the proposal.",
        "Bank of New York Mellon",
    )


def test_sentence_expansion_preserves_the_exact_source_slice() -> None:
    content = "Before. FTX filed for bankruptcy in 2022. After."
    partial = locate_source_quote(content, "filed for bankruptcy")

    assert partial is not None
    expanded = expand_span_to_sentence(content, partial)
    assert expanded.quote == "FTX filed for bankruptcy in 2022."
    assert content[expanded.start_char : expanded.end_char] == expanded.quote
