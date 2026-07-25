"""Tavily-specific request normalization."""

from __future__ import annotations

import re

TAVILY_QUERY_MAX_CHARS = 400

_QUERY_PART = re.compile(r'"[^"]*"|\S+')


def bounded_tavily_query(
    query: str,
    *,
    max_chars: int = TAVILY_QUERY_MAX_CHARS,
) -> str:
    """Fit a query to Tavily's limit without splitting words or quoted phrases."""

    compact = " ".join((query or "").split())
    if len(compact) <= max_chars:
        return compact

    kept: list[str] = []
    current_length = 0
    for part in _QUERY_PART.findall(compact):
        added_length = len(part) + (1 if kept else 0)
        if current_length + added_length > max_chars:
            break
        kept.append(part)
        current_length += added_length

    if kept:
        return " ".join(kept)

    # A single over-limit token has no word boundary at which to truncate.
    # Keep the provider request bounded even for this malformed edge case.
    return compact[:max_chars]
