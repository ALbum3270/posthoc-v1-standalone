import asyncio

import pytest

from open_deep_research.graphrag.adapters.tavily import TAVILY_QUERY_MAX_CHARS
from open_deep_research.harness.tools import SourceReadError, read, search


class FakeTavily:
    def __init__(self, *, search_response=None, extract_response=None):
        self.search_response = search_response or {"results": []}
        self.extract_response = extract_response or {"results": []}
        self.search_calls = []
        self.extract_calls = []

    async def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return self.search_response

    async def extract(self, urls, **kwargs):
        self.extract_calls.append((urls, kwargs))
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
