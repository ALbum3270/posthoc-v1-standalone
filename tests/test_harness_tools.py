import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from open_deep_research.graphrag.adapters.content import clean_text
from open_deep_research.graphrag.adapters.tavily import TAVILY_QUERY_MAX_CHARS
from open_deep_research.harness.ledger import SourceLinkCaptureStatus
from open_deep_research.harness.tools import (
    ProviderCallTimeoutError,
    SearchResultError,
    _with_provider_deadline,
    SourceReadError,
    extract_markdown_links,
    read,
    read_with_links,
    search,
)


def test_markdown_links_preserve_document_order_across_syntaxes() -> None:
    records = extract_markdown_links(
        "<https://first.example/a>\n"
        "[second]: https://second.example/b\n"
        "[third](https://third.example/c)\n",
        source_url="https://source.example/page",
    )

    assert [record.target_url for record in records] == [
        "https://first.example/a",
        "https://second.example/b",
        "https://third.example/c",
    ]


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


def test_search_rejects_finance_v2_relative_redirect_results_as_recoverable():
    client = FakeTavily(
        search_response={
            "results": [
                {
                    "title": f"Redirect {index}",
                    "url": f"/goto?url=https%3A%2F%2Fexample.com%2F{index}",
                    "content": "Provider redirect rather than a source URL.",
                }
                for index in range(5)
            ]
        }
    )

    with pytest.raises(
        SearchResultError,
        match=r"5 result\(s\).*none had an absolute HTTP\(S\) URL",
    ):
        asyncio.run(search("customer fund transfer", tavily_client=client))


@pytest.mark.parametrize(
    "results",
    (
        {"url": "https://example.com/not-an-array"},
        ["not-an-object", None, {"url": ""}],
    ),
)
def test_search_rejects_malformed_provider_result_shapes(results):
    client = FakeTavily(search_response={"results": results})

    with pytest.raises(SearchResultError):
        asyncio.run(search("bounded query", tavily_client=client))


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
    assert kwargs == {
        "extract_depth": "advanced",
        "format": "text",
        "timeout": 60.0,
    }


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
    assert result.link_capture.ordering_basis == (
        "provider_markdown_document_order_first_occurrence"
    )
    assert client.extract_calls == [
        (
            [url],
            {
                "extract_depth": "advanced",
                "format": "text",
                "timeout": 60.0,
            },
        ),
        (
            [url],
            {
                "extract_depth": "advanced",
                "format": "markdown",
                "timeout": 60.0,
            },
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


def test_extract_result_for_another_url_is_never_bound_to_requested_url():
    client = FakeTavily(
        extract_response={
            "results": [
                {
                    "url": "https://other.example/page",
                    "raw_content": "Bytes from another page.",
                }
            ]
        }
    )

    with pytest.raises(SourceReadError, match="no extraction result"):
        asyncio.run(
            read("https://requested.example/page", tavily_client=client)
        )


def test_extract_result_accepts_same_url_after_http_to_https_upgrade():
    client = FakeTavily(
        extract_response={
            "results": [
                {
                    "url": "https://example.com/page/",
                    "raw_content": "Canonical redirected bytes.",
                }
            ]
        }
    )

    result = asyncio.run(
        read("http://EXAMPLE.com/page", tavily_client=client)
    )

    assert result == "Canonical redirected bytes."


def test_local_search_deadline_has_a_named_degradable_error():
    class HangingSearch(FakeTavily):
        async def search(self, query, **kwargs):
            await asyncio.Event().wait()

    with pytest.raises(RuntimeError, match="search timed out after 0.001 seconds"):
        asyncio.run(
            search(
                "bounded query",
                tavily_client=HangingSearch(),
                timeout_seconds=0.001,
            )
        )


def test_a_hanging_provider_call_becomes_a_typed_timeout_not_a_hang():
    """A provider that never returns must not stall the whole run.

    Round and cost caps bound how many calls happen, not how long one call
    takes. Without a local deadline a single hung request keeps the run alive
    indefinitely, spending nothing and producing nothing.
    """

    async def scenario() -> None:
        async def never_returns() -> dict[str, object]:
            await asyncio.Event().wait()
            return {}

        with pytest.raises(ProviderCallTimeoutError) as caught:
            await _with_provider_deadline(
                never_returns(),
                operation="search",
                timeout_seconds=0.01,
            )
        assert "search" in str(caught.value)

    asyncio.run(scenario())


def test_the_timeout_is_catchable_by_the_broad_guards_that_wrap_reads():
    """The collection loop guards provider calls with ``except Exception``.

    A timeout raised outside that hierarchy would escape those guards and lose
    the whole run, which is how two earlier runs were lost.
    """

    assert issubclass(ProviderCallTimeoutError, Exception)


def test_a_non_positive_deadline_is_rejected_before_any_call_is_made():
    async def scenario() -> None:
        async def unused() -> dict[str, object]:
            raise AssertionError("the awaitable must not be awaited")

        coro = unused()
        with pytest.raises(ValueError, match="positive"):
            await _with_provider_deadline(
                coro, operation="search", timeout_seconds=0
            )
        coro.close()

    asyncio.run(scenario())
