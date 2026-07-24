"""
Supervisor：
1. 检测当前哪个槽位优先级最高且未填充
2. 调用 LLM 生成专门针对该槽位的搜索词（而非直接用 question 搜）

返回 (slot, search_query)，其中 slot 是 SLOTS 中的一条记录。
"""

from __future__ import annotations

import os

from v1.ontology import get_empty_slots

QUERY_GEN_PROMPT = """\
You are a research planner. Given a research topic and a specific investigation question, \
generate ONE precise web search query that will most likely find the answer.

The query should be:
- Specific and factual (not generic)
- Suitable for Tavily/Google web search
- 5-15 words maximum

Return ONLY the search query as a plain string, no explanation.
"""

_QUERY_RETRY_HINTS = (
    "official report primary source",
    "court filing regulator statement",
    "statistics amount timeline",
    "independent analysis alternative source",
)


def select_next_slot(
    filled_ids: set[str],
    failure_counts: dict[str, int] | None = None,
) -> dict:
    """Choose an open slot while backing off from repeatedly failing slots."""

    empties = get_empty_slots(filled_ids)
    if not empties:
        raise StopIteration("All slots are filled.")

    failures = failure_counts or {}
    return min(
        empties,
        key=lambda slot: (
            failures.get(slot["slot_id"], 0),
            -slot["priority"],
        ),
    )


def make_query_novel(
    query: str,
    *,
    topic: str,
    slot: dict,
    previous_queries: list[str],
) -> str:
    """Return a non-empty query that is not an exact repeat."""

    fallback = f"{topic} {slot['label']} {slot['question'][:60]}".strip()
    candidate = " ".join((query or fallback).strip().strip('"').strip("'").split())
    normalized_previous = {
        " ".join(previous.casefold().split())
        for previous in previous_queries
    }

    if candidate.casefold() not in normalized_previous:
        return candidate

    for offset in range(len(_QUERY_RETRY_HINTS)):
        hint_index = (len(previous_queries) + offset) % len(_QUERY_RETRY_HINTS)
        alternative = f"{candidate} {_QUERY_RETRY_HINTS[hint_index]}"
        if alternative.casefold() not in normalized_previous:
            return alternative

    return f"{candidate} alternative source attempt {len(previous_queries) + 1}"


async def pick_next_slot_and_query(
    topic: str,
    filled_ids: set[str],
    *,
    failure_counts: dict[str, int] | None = None,
    previous_queries: dict[str, list[str]] | None = None,
    model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
) -> tuple[dict, str]:
    """
    선택 highest-priority unfilled slot and generate a targeted search query.

    Returns:
      (slot_dict, search_query)
    Raises StopIteration if all slots are already filled.
    """
    slot = select_next_slot(filled_ids, failure_counts)
    slot_queries = (previous_queries or {}).get(slot["slot_id"], [])
    query = await _generate_search_query(
        topic,
        slot,
        model=model,
        previous_queries=slot_queries,
        failure_count=(failure_counts or {}).get(slot["slot_id"], 0),
    )
    query = make_query_novel(
        query,
        topic=topic,
        slot=slot,
        previous_queries=slot_queries,
    )
    return slot, query


async def _generate_search_query(
    topic: str,
    slot: dict,
    *,
    model: str,
    previous_queries: list[str],
    failure_count: int,
) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )

    retry_context = ""
    if previous_queries:
        failed_queries = "\n".join(f"- {query}" for query in previous_queries[-3:])
        retry_context = (
            "\n\nPrevious queries returned no usable evidence:\n"
            f"{failed_queries}\n"
            "Use different terminology, a different source type, or a more "
            "specific named document. Do not repeat a previous query."
        )

    user_msg = (
        f"Research topic: {topic}\n\n"
        f"Investigation dimension: {slot['dimension']} — {slot['label']}\n"
        f"Specific question to answer: {slot['question']}\n\n"
        f"Failed attempts for this slot: {failure_count}"
        f"{retry_context}\n\n"
        "Generate a web search query to find this information."
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": QUERY_GEN_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0,
        max_tokens=64,
    )

    query = (response.choices[0].message.content or "").strip().strip('"').strip("'")
    return query
