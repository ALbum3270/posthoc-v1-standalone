"""
Graphiti 写入封装（V1 极简版）。

将抽取到的三元组序列化为 episode_body 文本，写入 Graphiti，
metadata 中携带 ontology_slot 和 source_url，用于报告阶段 groupby。
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any


async def write_slot_episode(
    graphiti: Any,
    *,
    group_id: str,
    slot_id: str,
    slot_label: str,
    triples: list[dict],
    source_url: str,
    source_title: str,
) -> str:
    """
    将一批三元组作为一个 Episode 写入 Graphiti。
    返回 episode uuid（字符串），失败时返回空串。

    episode_body 格式：
        [slot_label] source: <url>
        - subject | predicate | object
        ...
    """
    from graphiti_core.nodes import EpisodeType

    lines = [f"[{slot_label}] source: {source_url}"]
    for t in triples:
        lines.append(f"- {t['subject']} | {t['predicate']} | {t['object']}")

    episode_body = "\n".join(lines)

    result = await graphiti.add_episode(
        name=f"{slot_id}::{source_title[:60]}",
        episode_body=episode_body,
        source_description=f"Web research — {source_url}",
        reference_time=datetime.now(timezone.utc),
        source=EpisodeType.text,
        group_id=group_id,
    )

    uuid = getattr(result, "episode", None)
    if uuid is not None:
        uuid = getattr(uuid, "uuid", "") or ""
    return str(uuid)
