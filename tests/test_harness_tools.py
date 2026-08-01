import asyncio

import pytest

from open_deep_research.graphrag.adapters.content import clean_text
from open_deep_research.graphrag.adapters.tavily import TAVILY_QUERY_MAX_CHARS
from open_deep_research.harness.ledger import SourceLinkCaptureStatus
from open_deep_research.harness.tools import (
    SourceReadError,
    read,
    read_with_links,
    search,
)


class FakeTavily:
    def __init__(
        self,
        *,
        search_response=None,
        extract_response=None,
        extract_responses_by_format=None,
    ):
        self.search_response = search_response or {"results": []}
        self.extract_response = extract_response or {"results": []}
        self.extract_responses_by_format = extract_responses_by_format
        self.search_calls = []
        self.extract_calls = []

    async def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return self.search_response

    async def extract(self, urls, **kwargs):
        self.extract_calls.append((urls, kwargs))
        if self.extract_responses_by_format is not None:
            response = self.extract_responses_by_format[kwargs["format"]]
            if isinstance(response, Exception):
                raise response
            return response
        return self.extract_response


def test_search_uses_injected_client_and_bounded_tavily_query():
    client = FakeTavily(
        search_response={
            "results": [
                {
                    "title": "Result",
                    "url": "https://example.com/article",
                    "content": "A result snippet.",
                    "score": 0.9,
                }
            ]
        }
    )
    long_query = " ".join(f"neutralword{index}" for index in range(100))

    results = asyncio.run(
        search(long_query, tavily_client=client, max_results=3)
    )

    provider_query, kwargs = client.search_calls[0]
    assert len(provider_query) <= TAVILY_QUERY_MAX_CHARS
    assert provider_query == long_query[: len(provider_query)].rstrip()
    assert kwargs == {"max_results": 3, "topic": "general"}
    assert results[0].model_dump() == {
        "title": "Result",
        "url": "https://example.com/article",
        "snippet": "A result snippet.",
        "score": 0.9,
    }


def test_read_reuses_clean_text_and_does_not_chunk_or_truncate():
    middle = "Body paragraph.\n\n" * 1000
    tail = "A tail marker beyond any plausible head limit."
    raw_text = (
        "Jump to content\n\n"
        + middle
        + tail
        + "\n\nReferences\nA citation entry that should be removed."
    )
    client = FakeTavily(
        extract_response={
            "results": [
                {
                    "url": "https://example.com/article",
                    "raw_content": raw_text,
                }
            ]
        }
    )

    cleaned = asyncio.run(
        read("https://example.com/article", tavily_client=client)
    )

    assert tail in cleaned
    assert not cleaned.startswith("Jump to content")
    assert "citation entry" not in cleaned
    urls, kwargs = client.extract_calls[0]
    assert urls == ["https://example.com/article"]
    assert kwargs == {"extract_depth": "advanced", "format": "text"}


def test_read_raises_clear_error_when_injected_client_returns_no_text():
    client = FakeTavily(extract_response={"results": []})

    with pytest.raises(SourceReadError, match="no extraction result"):
        asyncio.run(read("https://example.com/missing", tavily_client=client))


def test_read_with_links_keeps_text_bytes_and_captures_markdown_sidecar():
    url = "https://example.com/reports/page"
    raw_text = "Jump to content\n\nCanonical body.\n\nFinal source wording."
    markdown = (
        "Canonical body.\n\n"
        "[Original filing](https://records.example/filing.pdf)\n"
        "[Appendix](../files/appendix.pdf)\n"
        "[Duplicate](https://records.example/filing.pdf)\n"
        "[Email](mailto:desk@example.com)"
    )
    client = FakeTavily(
        extract_responses_by_format={
            "text": {
                "results": [{"url": url, "raw_content": raw_text}]
            },
            "markdown": {
                "results": [{"url": url, "raw_content": markdown}]
            },
        }
    )

    result = asyncio.run(read_with_links(url, tavily_client=client))

    assert result.cleaned_text == clean_text(raw_text)
    assert [record.model_dump() for record in result.source_links] == [
        {
            "target_url": "https://records.example/filing.pdf",
            "label": "Original filing",
        },
        {
            "target_url": "https://example.com/files/appendix.pdf",
            "label": "Appendix",
        },
    ]
    assert result.link_capture.status is SourceLinkCaptureStatus.CAPTURED
    assert result.link_capture.captured_link_count == 2
    assert result.link_capture.completeness_guaranteed is False
    assert client.extract_calls == [
        (
            [url],
            {"extract_depth": "advanced", "format": "text"},
        ),
        (
            [url],
            {"extract_depth": "advanced", "format": "markdown"},
        ),
    ]


def test_markdown_sidecar_failure_does_not_discard_canonical_text():
    url = "https://example.com/article"
    raw_text = "Canonical source wording."
    client = FakeTavily(
        extract_responses_by_format={
            "text": {
                "results": [{"url": url, "raw_content": raw_text}]
            },
            "markdown": RuntimeError("markdown unavailable"),
        }
    )

    result = asyncio.run(read_with_links(url, tavily_client=client))

    assert result.cleaned_text == clean_text(raw_text)
    assert result.source_links == ()
    assert result.link_capture.status is SourceLinkCaptureStatus.PROVIDER_ERROR
    assert result.link_capture.error == "RuntimeError: markdown unavailable"
