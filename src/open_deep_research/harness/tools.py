"""Injected retrieval tools for the research harness."""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from open_deep_research.graphrag.adapters.content import clean_text
from open_deep_research.graphrag.adapters.tavily import bounded_tavily_query


class TavilyClient(Protocol):
    """The asynchronous Tavily operations required by this module."""

    def search(self, query: str, **kwargs: Any) -> Awaitable[Mapping[str, Any]]:
        """Search the web."""

    def extract(
        self, urls: list[str] | str, **kwargs: Any
    ) -> Awaitable[Mapping[str, Any]]:
        """Extract full text for one or more URLs."""


class SearchResult(BaseModel):
    """The provider fields needed to decide whether a URL is worth reading."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = ""
    url: str = Field(min_length=1)
    snippet: str = ""
    score: float | None = None


class SourceReadError(RuntimeError):
    """Raised when the retrieval provider returns no text for a URL."""


async def search(
    query: str,
    *,
    tavily_client: TavilyClient,
    max_results: int = 5,
) -> list[SearchResult]:
    """Search with a provider-bounded query and return neutral result metadata."""

    provider_query = bounded_tavily_query(query)
    if not provider_query:
        raise ValueError("search query must not be blank")
    if max_results < 1:
        raise ValueError("max_results must be positive")

    response = await tavily_client.search(
        provider_query,
        max_results=max_results,
        topic="general",
    )
    results: list[SearchResult] = []
    for item in response.get("results", ()) or ():
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        results.append(
            SearchResult(
                title=str(item.get("title") or ""),
                url=url,
                snippet=str(item.get("content") or ""),
                score=item.get("score"),
            )
        )
    return results


async def read(
    url: str,
    *,
    tavily_client: TavilyClient,
) -> str:
    """Fetch and clean a complete page without chunking or truncation."""

    normalized_url = url.strip()
    if not normalized_url:
        raise ValueError("source URL must not be blank")

    response = await tavily_client.extract(
        urls=[normalized_url],
        extract_depth="advanced",
        format="text",
    )
    items = response.get("results", ()) or ()
    matching = next(
        (
            item
            for item in items
            if str(item.get("url") or "").rstrip("/")
            == normalized_url.rstrip("/")
        ),
        items[0] if items else None,
    )
    if matching is None:
        raise SourceReadError(f"no extraction result for URL: {normalized_url}")

    raw_text = matching.get("raw_content") or matching.get("content") or ""
    cleaned = clean_text(str(raw_text))
    if not cleaned:
        raise SourceReadError(f"no readable text for URL: {normalized_url}")
    return cleaned
