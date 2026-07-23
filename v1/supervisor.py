"""
Supervisor：
1. 检测当前哪个槽位优先级最高且未填充
2. 调用 LLM 生成专门针对该槽位的搜索词（而非直接用 question 搜）

返回 (slot, search_query)，其中 slot 是 SLOTS 中的一条记录。
"""

from __future__ import annotations

import json
import os

from v1.ontology import SLOTS_SORTED, get_empty_slots

QUERY_GEN_PROMPT = """\
You are a research planner. Given a research topic and a specific investigation question, \
generate ONE precise web search query that will most likely find the answer.

The query should be:
- Specific and factual (not generic)
- Suitable for Tavily/Google web search
- 5-15 words maximum

Return ONLY the search query as a plain string, no explanation.
"""


async def pick_next_slot_and_query(
    topic: str,
    filled_ids: set[str],
    *,
    model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
) -> tuple[dict, str]:
    """
    선택 highest-priority unfilled slot and generate a targeted search query.

    Returns:
      (slot_dict, search_query)
    Raises StopIteration if all slots are already filled.
    """
    empties = get_empty_slots(filled_ids)
    if not empties:
        raise StopIteration("All slots are filled.")

    # Always take the highest-priority empty slot (already sorted by ontology.py)
    slot = empties[0]

    query = await _generate_search_query(topic, slot, model=model)
    return slot, query


async def _generate_search_query(topic: str, slot: dict, *, model: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )

    user_msg = (
        f"Research topic: {topic}\n\n"
        f"Investigation dimension: {slot['dimension']} — {slot['label']}\n"
        f"Specific question to answer: {slot['question']}\n\n"
        f"Generate a web search query to find this information."
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
    # Fallback: use topic + question if LLM fails
    return query or f"{topic} {slot['question'][:60]}"
