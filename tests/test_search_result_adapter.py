"""Tests for search-provider -> SourceDocument mapping.

The response shapes below were captured from live Tavily calls on 2026-07-24,
including the asymmetry that drives the design: ``topic="news"`` carries
``published_date``, ``topic="general"`` has no such key at all.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from open_deep_research.graphrag.adapters.search_results import (
    parse_published_at,
    published_at_from_url,
    tavily_response_to_source_documents,
    tavily_result_to_source_document,
)
from open_deep_research.graphrag.schemas import SourceType

# Captured verbatim: topic="general" returns no published_date key.
GENERAL_RESULT = {
    "url": "https://en.wikipedia.org/wiki/FTX",
    "title": "FTX - Wikipedia",
    "content": "FTX was a cryptocurrency exchange...",
    "raw_content": "[Jump to content](#bodyContent)\n\nFTX was...",
    "score": 0.83,
}

# Captured verbatim: topic="news" does carry it, RFC 2822 formatted.
NEWS_RESULT = {
    "url": "https://www.motherjones.com/politics/2026/07/sam-bankman-fried-pardon/",
    "title": "Sam Bankman-Fried...",
    "content": "...",
    "raw_content": "...",
    "published_date": "Sat, 18 Jul 2026 07:01:00 GMT",
    "score": 0.91,
}


def test_news_publication_date_is_parsed() -> None:
    doc = tavily_result_to_source_document(NEWS_RESULT, topic="news")

    assert doc.published_at == datetime(2026, 7, 18, 7, 1, tzinfo=timezone.utc)
    assert doc.metadata["published_at_source"] == "provider"
    assert doc.source_type is SourceType.NEWS


def test_general_result_without_a_date_stays_unknown() -> None:
    """Measured: topic="general" has no published_date. None must mean unknown.

    Substituting now() here is precisely the bug that produced 2026 dates, so the
    adapter has to leave the gap visible rather than fill it.
    """

    doc = tavily_result_to_source_document(GENERAL_RESULT, topic="general")

    assert doc.published_at is None
    assert doc.source_type is SourceType.WEB
    assert "published_at_source" not in doc.metadata


def test_date_is_recovered_from_a_dated_url_path() -> None:
    item = dict(NEWS_RESULT)
    item.pop("published_date")

    doc = tavily_result_to_source_document(item, topic="general")

    assert doc.published_at == datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert doc.metadata["published_at_source"] == "url_path"


def test_provider_date_wins_over_the_url_path() -> None:
    doc = tavily_result_to_source_document(NEWS_RESULT, topic="news")
    assert doc.published_at.day == 18  # from published_date, not the /07/ path


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://x.com/2026/07/24/slug", datetime(2026, 7, 24, tzinfo=timezone.utc)),
        ("https://x.com/2026-07-24-slug", datetime(2026, 7, 24, tzinfo=timezone.utc)),
        ("https://x.com/politics/2022/11/", datetime(2022, 11, 1, tzinfo=timezone.utc)),
        # Not dates: out-of-range month, version-like paths, bare ids.
        ("https://x.com/2026/13/slug", None),
        ("https://x.com/v1/2/3", None),
        ("https://en.wikipedia.org/wiki/FTX", None),
        (None, None),
    ],
)
def test_url_date_extraction(url: str | None, expected: datetime | None) -> None:
    assert published_at_from_url(url) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("Sat, 18 Jul 2026 07:01:00 GMT", datetime(2026, 7, 18, 7, 1, tzinfo=timezone.utc)),
        ("2022-11-11T00:00:00Z", datetime(2022, 11, 11, tzinfo=timezone.utc)),
        ("2022-11-11", datetime(2022, 11, 11, tzinfo=timezone.utc)),
        ("not a date", None),
        ("", None),
        (None, None),
        (True, None),
    ],
)
def test_parse_published_at(value: object, expected: datetime | None) -> None:
    assert parse_published_at(value) == expected


def test_naive_timestamps_are_assumed_utc() -> None:
    parsed = parse_published_at(datetime(2022, 11, 11))
    assert parsed == datetime(2022, 11, 11, tzinfo=timezone.utc)


def test_cleaned_content_overrides_provider_text() -> None:
    """Callers strip chrome before extraction; raw_content starts with navigation."""

    doc = tavily_result_to_source_document(
        GENERAL_RESULT, content="FTX was a cryptocurrency exchange founded in 2019."
    )

    assert not doc.content.startswith("[Jump to content]")
    assert doc.snippet == GENERAL_RESULT["content"]


def test_response_mapping_skips_empty_bodies() -> None:
    response = {
        "results": [
            NEWS_RESULT,
            {"url": "https://x.com/empty", "title": "Empty", "content": "", "raw_content": ""},
        ]
    }

    documents = tavily_response_to_source_documents(response, topic="news")

    assert [d.url for d in documents] == [NEWS_RESULT["url"]]


def test_document_id_falls_back_to_url() -> None:
    doc = tavily_result_to_source_document(GENERAL_RESULT)
    assert doc.document_id == GENERAL_RESULT["url"]
