"""Tests for boilerplate stripping and passage selection (§3.11 constraint 2)."""

from __future__ import annotations

from open_deep_research.graphrag.adapters.content import (
    clean_text,
    query_terms,
    select_relevant_text,
    split_chunks,
)

# The real shape of what Tavily returns for a Wikipedia page: the first two
# thousand characters are navigation, and the answer is far below.
WIKI_LIKE = (
    "[Jump to content](#bodyContent)\n\n"
    "![](/static/images/icons/enwiki-25.svg) ![Wikipedia](/static/images/wordmark.svg)\n\n"
    "[Search](/wiki/Special:Search \"Search Wikipedia\")\n\n"
    "Main menu\n\n"
    + "[Navigation link](/wiki/Some_Page) " * 300
    + "\n\nFTX faced a liquidity shortfall of 8 billion USD when withdrawals spiked.\n\n"
    + "Unrelated filler paragraph about corporate history. " * 40
)


def test_navigation_chrome_is_removed() -> None:
    cleaned = clean_text(WIKI_LIKE)

    assert "Jump to content" not in cleaned
    assert "enwiki-25.svg" not in cleaned
    assert "Main menu" not in cleaned
    assert "FTX faced a liquidity shortfall" in cleaned


def test_the_answer_survives_where_head_truncation_would_lose_it() -> None:
    """The exact V1 failure: the fact is nowhere near the first 2000 chars."""

    assert "liquidity shortfall" not in WIKI_LIKE[:2000]

    selected = select_relevant_text(
        WIKI_LIKE, focus="What was the scale of the losses?", max_chars=2000
    )

    assert "liquidity shortfall of 8 billion USD" in selected
    assert len(selected) <= 2000


def test_short_documents_pass_through_cleaned() -> None:
    text = "FTX filed for bankruptcy.\n\nIt happened in November 2022."
    assert select_relevant_text(text, focus="when?") == text


def test_selection_respects_the_character_cap() -> None:
    selected = select_relevant_text(WIKI_LIKE, focus="liquidity", max_chars=500)
    assert len(selected) <= 500


def test_selected_passages_keep_document_order() -> None:
    text = "\n\n".join(
        [
            "Alpha paragraph mentions losses once.",
            "Filler. " * 200,
            "Beta paragraph mentions losses losses losses repeatedly.",
        ]
    )

    selected = select_relevant_text(text, focus="losses", max_chars=200)

    assert selected.index("Alpha") < selected.index("Beta")


def test_no_focus_falls_back_to_the_cleaned_head() -> None:
    selected = select_relevant_text(WIKI_LIKE, focus="", max_chars=300)

    assert selected
    assert "Jump to content" not in selected


def test_empty_input_is_empty_output() -> None:
    assert select_relevant_text("", focus="anything") == ""


def test_query_terms_drop_stopwords_and_handle_cjk() -> None:
    assert query_terms("What was the scale of the losses?") == {"scale", "losses"}
    assert "破" in query_terms("FTX 破产规模")


def test_chunks_split_on_paragraphs_not_mid_sentence() -> None:
    text = "\n\n".join(f"Paragraph number {i} with some text." for i in range(10))
    chunks = split_chunks(text, target_chars=100)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    # Nothing is lost in the split.
    for i in range(10):
        assert f"Paragraph number {i}" in "\n\n".join(chunks)


def test_oversized_single_paragraph_is_still_split() -> None:
    chunks = split_chunks("x" * 2500, target_chars=900)
    assert len(chunks) >= 3


def test_chunk_size_is_clamped_to_the_budget() -> None:
    """A chunk bigger than max_chars could never be selected.

    Without the clamp, selection silently degrades back to "return the head" --
    the very behaviour this module exists to replace.
    """

    text = "\n\n".join(
        ["Intro paragraph, not relevant."]
        + ["Filler sentence. " * 60]
        + ["The losses reached 8 billion USD in total."]
    )

    selected = select_relevant_text(text, focus="losses", max_chars=300)

    assert "8 billion USD" in selected
    assert len(selected) <= 300
