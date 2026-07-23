"""
Tavily 搜索封装。
返回 list[SearchResult]，每条包含 url / title / raw_text（截断至 max_chars）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class SearchResult:
    url: str
    title: str
    raw_text: str  # 已截断


async def search(
    query: str,
    *,
    max_results: int = 5,
    max_chars: int = 2000,
) -> list[SearchResult]:
    """执行 Tavily 搜索，返回最多 max_results 条结果，每条正文截断至 max_chars。"""
    from tavily import AsyncTavilyClient

    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set in environment.")

    client = AsyncTavilyClient(api_key=api_key)
    response = await client.search(
        query,
        max_results=max_results,
        include_raw_content=True,
        topic="general",
    )

    results: list[SearchResult] = []
    for item in response.get("results", []):
        raw = item.get("raw_content") or item.get("content") or ""
        results.append(
            SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                raw_text=raw[:max_chars],
            )
        )
    return results
