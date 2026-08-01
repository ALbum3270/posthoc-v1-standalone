"""Injected retrieval tools for the research harness."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Mapping
from typing import Any, Protocol
from urllib.parse import urljoin, urlsplit

from pydantic import BaseModel, ConfigDict, Field

from open_deep_research.graphrag.adapters.content import clean_text
from open_deep_research.graphrag.adapters.tavily import bounded_tavily_query
from open_deep_research.harness.ledger import (
    SourceLinkCaptureAudit,
    SourceLinkCaptureStatus,
    SourceLinkRecord,
)


_INLINE_MARKDOWN_LINK = re.compile(
    r"(?<!!)\[(?P<label>[^\]]*)\]\(\s*"
    r"(?P<target><[^>]+>|[^\s)]+)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)"
)
_REFERENCE_MARKDOWN_LINK = re.compile(
    r"(?m)^\s{0,3}\[(?P<label>[^\]]+)\]:\s*"
    r"(?P<target><[^>]+>|\S+)"
)
_AUTOLINK = re.compile(r"<(?P<target>https?://[^<>\s]+)>", re.IGNORECASE)


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


class SourceReadResult(BaseModel):
    """Canonical cleaned text plus an independent best-effort link sidecar."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cleaned_text: str = Field(min_length=1)
    source_links: tuple[SourceLinkRecord, ...] = ()
    link_capture: SourceLinkCaptureAudit


def _matching_extract_result(
    response: Mapping[str, Any],
    normalized_url: str,
) -> Mapping[str, Any] | None:
    items = response.get("results", ()) or ()
    return next(
        (
            item
            for item in items
            if str(item.get("url") or "").rstrip("/")
            == normalized_url.rstrip("/")
        ),
        items[0] if items else None,
    )


def _extract_content(item: Mapping[str, Any]) -> str:
    return str(item.get("raw_content") or item.get("content") or "")


def _http_link(target: str, *, source_url: str) -> str | None:
    stripped = target.strip()
    if stripped.startswith("<") and stripped.endswith(">"):
        stripped = stripped[1:-1].strip()
    if not stripped:
        return None
    absolute = urljoin(source_url, stripped)
    parsed = urlsplit(absolute)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return None
    return absolute


def extract_markdown_links(
    markdown: str,
    *,
    source_url: str,
) -> tuple[SourceLinkRecord, ...]:
    """Mechanically parse common Markdown links without semantic ranking.

    The provider does not guarantee that its Markdown preserves every link.
    This function therefore inventories only what was returned; an empty tuple
    is not evidence that the original page contained no links.
    """

    candidates: list[tuple[str, str]] = []
    for pattern in (_INLINE_MARKDOWN_LINK, _REFERENCE_MARKDOWN_LINK):
        for match in pattern.finditer(markdown):
            candidates.append(
                (match.group("label").strip(), match.group("target"))
            )
    for match in _AUTOLINK.finditer(markdown):
        candidates.append(("", match.group("target")))

    records: list[SourceLinkRecord] = []
    seen_targets: set[str] = set()
    for label, raw_target in candidates:
        target_url = _http_link(raw_target, source_url=source_url)
        if target_url is None or target_url in seen_targets:
            continue
        seen_targets.add(target_url)
        records.append(SourceLinkRecord(target_url=target_url, label=label))
    return tuple(records)


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
    matching = _matching_extract_result(response, normalized_url)
    if matching is None:
        raise SourceReadError(f"no extraction result for URL: {normalized_url}")

    cleaned = clean_text(_extract_content(matching))
    if not cleaned:
        raise SourceReadError(f"no readable text for URL: {normalized_url}")
    return cleaned


async def read_with_links(
    url: str,
    *,
    tavily_client: TavilyClient,
) -> SourceReadResult:
    """Read canonical text, then capture links through a Markdown sidecar.

    The first request deliberately preserves the historical ``format=text``
    and ``clean_text`` path byte-for-byte.  Markdown is never used as source
    text; it is a second, best-effort metadata channel.  A sidecar failure does
    not discard a successfully fetched canonical page.
    """

    normalized_url = url.strip()
    cleaned_text = await read(normalized_url, tavily_client=tavily_client)
    try:
        response = await tavily_client.extract(
            urls=[normalized_url],
            extract_depth="advanced",
            format="markdown",
        )
        matching = _matching_extract_result(response, normalized_url)
        if matching is None:
            capture = SourceLinkCaptureAudit(
                status=SourceLinkCaptureStatus.NO_MARKDOWN_CONTENT,
            )
            links: tuple[SourceLinkRecord, ...] = ()
        else:
            markdown = _extract_content(matching)
            if not markdown:
                capture = SourceLinkCaptureAudit(
                    status=SourceLinkCaptureStatus.NO_MARKDOWN_CONTENT,
                )
                links = ()
            else:
                links = extract_markdown_links(
                    markdown,
                    source_url=normalized_url,
                )
                capture = SourceLinkCaptureAudit(
                    status=(
                        SourceLinkCaptureStatus.CAPTURED
                        if links
                        else SourceLinkCaptureStatus.NO_LINKS_CAPTURED
                    ),
                    captured_link_count=len(links),
                )
    except Exception as exc:  # noqa: BLE001 - sidecar failure is audited data
        links = ()
        capture = SourceLinkCaptureAudit(
            status=SourceLinkCaptureStatus.PROVIDER_ERROR,
            error=f"{type(exc).__name__}: {exc}",
        )
    return SourceReadResult(
        cleaned_text=cleaned_text,
        source_links=links,
        link_capture=capture,
    )
