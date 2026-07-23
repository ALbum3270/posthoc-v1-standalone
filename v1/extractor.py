"""
LLM 三元组抽取（使用小模型 gpt-4o-mini）。

给定一段截断网页文本和一个调查问题，返回与该问题相关的事实列表：
  [{"subject": ..., "predicate": ..., "object": ..., "quote": ...}]

无相关内容则返回 []。
"""

from __future__ import annotations

import json
import os

SYSTEM_PROMPT = """\
You are a precise information extractor for investigative research. \
Your job is to read a text snippet and extract facts that directly answer the given investigation question.

Output ONLY a JSON array. Each element must have exactly these keys:
  "subject"   – the entity doing or being described (string)
  "predicate" – the relation or action (verb phrase, string)
  "object"    – the target entity or value (string)
  "quote"     – the exact 1-2 sentence quote from the text that supports this fact (string)

Rules:
- Extract ONLY facts that directly answer the investigation question.
- Do NOT invent facts. If nothing in the text answers the question, return [].
- Maximum 5 triples per call.
- Keep subject, predicate, object as concise noun phrases / verb phrases.
"""

USER_TEMPLATE = """\
## Investigation question
{question}

## Text snippet (may be truncated)
{text}

Extract relevant facts as a JSON array. Return [] if nothing applies.
"""


async def extract_triples(
    text: str,
    question: str,
    *,
    model: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
) -> list[dict]:
    """
    Returns list of dicts: [{subject, predicate, object, quote}]
    Uses openai directly (no langchain) for maximum simplicity.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            question=question,
            text=text[:2000],
        )},
    ]

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1024,
        response_format={"type": "json_object"},  # json mode
    )

    raw = response.choices[0].message.content or "[]"

    # The model may return {"triples": [...]} or just [...]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, list):
        triples = parsed
    elif isinstance(parsed, dict):
        # Accept any top-level array value
        for v in parsed.values():
            if isinstance(v, list):
                triples = v
                break
        else:
            return []
    else:
        return []

    # Normalize: ensure required keys present
    result = []
    for t in triples:
        if isinstance(t, dict) and "subject" in t and "predicate" in t and "object" in t:
            result.append({
                "subject":   str(t.get("subject", "")),
                "predicate": str(t.get("predicate", "")),
                "object":    str(t.get("object", "")),
                "quote":     str(t.get("quote", "")),
            })
    return result
