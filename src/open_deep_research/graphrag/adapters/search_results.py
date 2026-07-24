"""Normalize search-provider payloads into ``SourceDocument``.

The job that matters here is ``published_at``. Graphiti resolves relative time
expressions against ``reference_time`` (``prompts/extract_edges.py:78``), so a
2022 article carrying today's date as its reference produced "mid-November 2026"
in the baseline run (SESSION_HANDOFF §3.12 fix 1).

**Measured 2026-07-24, and it constrains the design:** Tavily returns a
``published_date`` only for ``topic="news"``. On ``topic="general"`` -- the mode
the research loop actually uses -- the field is absent from the response
entirely. So publication dates are frequently unavailable, and this module must
say "unknown" rather than invent one.

That is why the deterministic date guard (``validation.dates``, §3.12 fix 3) is
the load-bearing protection and this is the supplementary one: the guard needs no
external metadata, and it holds even when ``published_at`` is None.

``published_at=None`` means *unknown*. It must never be silently swapped for
``now()`` -- that substitution is the original bug.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from open_deep_research.graphrag.schemas import SourceDocument, SourceType

# Matches dates embedded in article paths: /2026/07/24/slug, /2026-07-24-slug.
_URL_DATE = re.compile(r"/(?P<y>19\d{2}|20\d{2})[/-](?P<m>0[1-9]|1[0-2])(?:[/-](?P<d>[0-3]\d))?(?=[/-]|$)")

_TOPIC_SOURCE_TYPES = {
    "news": SourceType.NEWS,
    "general": SourceType.WEB,
    "finance": SourceType.REPORT,
}


def parse_published_at(value: Any) -> datetime | None:
    """Parse a provider-supplied publication timestamp, or return None.

    Handles the two shapes seen in practice: RFC 2822 as Tavily's news topic
    emits it ("Sat, 18 Jul 2026 07:01:00 GMT") and ISO 8601. Anything
    unrecognized yields None rather than a guess.
    """

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        parsed = None
    if parsed is not None:
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    iso = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def published_at_from_url(url: str | None) -> datetime | None:
    """Recover a publication date from a dated article path.

    Many publishers put it there (``/politics/2026/07/slug``). This is a
    deterministic read of the URL, not an inference from the page text -- an
    article *about* November 2022 is full of 2022 dates that say nothing about
    when it was written, so content is deliberately not consulted.
    """

    if not url:
        return None
    match = _URL_DATE.search(url)
    if not match:
        return None
    try:
        return datetime(
            int(match["y"]),
            int(match["m"]),
            int(match["d"] or 1),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None


def tavily_result_to_source_document(
    item: dict[str, Any],
    *,
    topic: str = "general",
    retrieved_at: datetime | None = None,
    document_id: str | None = None,
    content: str | None = None,
) -> SourceDocument:
    """Map one Tavily result to a ``SourceDocument``.

    ``content`` overrides the provider text, so a caller that has already run
    boilerplate stripping and passage selection can pass the cleaned version --
    raw ``raw_content`` starts with navigation chrome, which is what starved
    extraction in the baseline run (§3.11 constraint 2).
    """

    url = item.get("url") or None
    published_at = parse_published_at(item.get("published_date"))
    if published_at is None:
        published_at = published_at_from_url(url)

    body = content if content is not None else (
        item.get("raw_content") or item.get("content") or ""
    )

    metadata: dict[str, Any] = {"provider": "tavily", "topic": topic}
    if item.get("score") is not None:
        metadata["score"] = item["score"]
    if published_at is not None and not item.get("published_date"):
        metadata["published_at_source"] = "url_path"
    elif published_at is not None:
        metadata["published_at_source"] = "provider"

    return SourceDocument(
        document_id=document_id or url or (item.get("title") or "unknown"),
        title=item.get("title") or "",
        url=url,
        source_type=_TOPIC_SOURCE_TYPES.get(topic, SourceType.WEB),
        content=body,
        snippet=item.get("content") or None,
        published_at=published_at,
        retrieved_at=retrieved_at or datetime.now(timezone.utc),
        metadata=metadata,
    )


def tavily_response_to_source_documents(
    response: dict[str, Any],
    *,
    topic: str = "general",
    retrieved_at: datetime | None = None,
) -> list[SourceDocument]:
    """Map a full Tavily response, skipping results with no usable body."""

    stamp = retrieved_at or datetime.now(timezone.utc)
    documents: list[SourceDocument] = []
    for item in response.get("results", []) or []:
        document = tavily_result_to_source_document(
            item, topic=topic, retrieved_at=stamp
        )
        if document.content.strip():
            documents.append(document)
    return documents
