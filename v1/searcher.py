"""Tavily 搜索封装与面向调查问题的正文选段。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import urldefrag


@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str
    raw_text: str


_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]*\)")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)]\([^)]*\)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,}|[\u4e00-\u9fff]{2,}", re.IGNORECASE)
_SPACE_RE = re.compile(r"[ \t]+")
_PARAGRAPH_RE = re.compile(r"\n\s*\n+")
_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "answer",
    "event",
    "find",
    "from",
    "given",
    "information",
    "investigation",
    "into",
    "main",
    "most",
    "question",
    "research",
    "specific",
    "that",
    "the",
    "this",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _normalize_url(url: str) -> str:
    """Normalize URLs enough to de-duplicate repeated search results."""

    return urldefrag(url.strip())[0].rstrip("/")


def _clean_text(text: str) -> str:
    """Remove common HTML/Markdown chrome while preserving readable paragraphs."""

    if not text:
        return ""

    cleaned = _MARKDOWN_IMAGE_RE.sub(" ", text)
    cleaned = _MARKDOWN_LINK_RE.sub(r"\1", cleaned)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

    lines: list[str] = []
    for line in cleaned.splitlines():
        normalized = _SPACE_RE.sub(" ", line).strip()
        if not normalized:
            lines.append("")
            continue

        lowered = normalized.casefold()
        if lowered in {"jump to content", "main menu", "navigation menu"}:
            continue
        lines.append(normalized)

    return _PARAGRAPH_RE.sub("\n\n", "\n".join(lines)).strip()


def _query_terms(*parts: str) -> set[str]:
    """Extract useful English or Chinese terms for passage scoring."""

    terms: set[str] = set()
    for part in parts:
        for match in _TOKEN_RE.findall(part.casefold()):
            if match not in _STOPWORDS:
                terms.add(match)
    return terms


def _split_chunks(text: str, *, target_chars: int = 900) -> list[str]:
    """Split cleaned text into paragraph-aware chunks."""

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_size = 0

    for paragraph in _PARAGRAPH_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        parts = [
            paragraph[index:index + target_chars]
            for index in range(0, len(paragraph), target_chars)
        ]
        for part in parts:
            additional_size = len(part) + (2 if buffer else 0)
            if buffer and buffer_size + additional_size > target_chars:
                chunks.append("\n\n".join(buffer))
                buffer = []
                buffer_size = 0
            buffer.append(part)
            buffer_size += len(part) + (2 if len(buffer) > 1 else 0)

    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks


def _score_chunk(chunk: str, terms: set[str]) -> tuple[float, int]:
    """Score a chunk by focus-term matches and basic content quality."""

    lowered = chunk.casefold()
    matched_terms = sum(min(lowered.count(term), 3) for term in terms)
    number_count = len(re.findall(r"\b\d[\d,.%$€£¥-]*", chunk))
    prose_chars = len(re.findall(r"[A-Za-z\u4e00-\u9fff]", chunk))
    link_noise = lowered.count("http") + lowered.count(".svg")

    score = matched_terms * 4.0
    score += min(number_count, 8) * 0.15
    score += min(prose_chars / max(len(chunk), 1), 1.0)
    score -= link_noise * 0.5
    return score, prose_chars


def select_relevant_text(
    raw_text: str,
    *,
    snippet: str = "",
    query: str = "",
    focus: str = "",
    max_chars: int = 2000,
) -> str:
    """Select relevant passages instead of blindly taking the document prefix.

    Tavily's short ``content`` field is already query-focused, so it is retained
    first. Remaining space is filled with the highest-scoring passages from the
    cleaned full document.
    """

    if max_chars <= 0:
        return ""

    cleaned_snippet = _clean_text(snippet)
    cleaned_raw = _clean_text(raw_text)
    terms = _query_terms(query, focus)

    selected: list[str] = []
    remaining = max_chars

    if cleaned_snippet:
        snippet_budget = remaining
        if cleaned_raw:
            snippet_budget = max(1, int(max_chars * 0.6))
        snippet_part = cleaned_snippet[:snippet_budget]
        selected.append(snippet_part)
        remaining -= len(snippet_part)

    if cleaned_raw and remaining > 2:
        ranked_chunks = sorted(
            enumerate(_split_chunks(cleaned_raw)),
            key=lambda item: (
                -_score_chunk(item[1], terms)[0],
                -_score_chunk(item[1], terms)[1],
                item[0],
            ),
        )
        normalized_existing = " ".join(selected).casefold()

        for _, chunk in ranked_chunks:
            if remaining <= 2:
                break

            comparable = _SPACE_RE.sub(" ", chunk).casefold()
            if comparable and comparable[:160] in normalized_existing:
                continue

            separator = "\n\n" if selected else ""
            available = remaining - len(separator)
            if available <= 0:
                break
            selected.append(separator + chunk[:available])
            remaining -= len(separator) + min(len(chunk), available)
            normalized_existing += " " + comparable

    return "".join(selected).strip()


async def search(
    query: str,
    *,
    max_results: int = 5,
    max_chars: int = 2000,
    focus: str = "",
    exclude_urls: set[str] | None = None,
) -> list[SearchResult]:
    """Execute Tavily search and return focused, de-duplicated page text."""

    from tavily import AsyncTavilyClient

    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set in environment.")

    excluded = {_normalize_url(url) for url in (exclude_urls or set())}
    client = AsyncTavilyClient(api_key=api_key)
    response = await client.search(
        query,
        max_results=max_results,
        include_raw_content="text",
        topic="general",
    )

    results: list[SearchResult] = []
    for item in response.get("results", []):
        url = item.get("url", "")
        normalized_url = _normalize_url(url)
        if not normalized_url or normalized_url in excluded:
            continue
        excluded.add(normalized_url)

        text = select_relevant_text(
            item.get("raw_content") or "",
            snippet=item.get("content") or "",
            query=query,
            focus=focus,
            max_chars=max_chars,
        )
        results.append(
            SearchResult(
                url=url,
                title=item.get("title", ""),
                raw_text=text,
            )
        )
    return results
