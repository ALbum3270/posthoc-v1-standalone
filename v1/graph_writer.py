"""
Graphiti 写入封装（V1 极简版）。

将抽取到的三元组序列化为 episode_body 文本，写入 Graphiti。
研究 ID、槽位 ID 与 URL 保存在 Episode 名称/来源描述中；报告阶段使用
返回的 Episode UUID 精确回溯，不再依赖模糊语义检索分组。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


async def write_slot_episode(
    graphiti: Any,
    *,
    group_id: str,
    research_id: str,
    slot_id: str,
    slot_label: str,
    triples: list[dict],
    source_url: str,
    source_title: str,
    previous_episode_uuids: list[str] | None = None,
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
        name=f"{research_id}::{slot_id}::{source_title[:60]}",
        episode_body=episode_body,
        source_description=(
            "Web research"
            f" | research_id={research_id}"
            f" | slot_id={slot_id}"
            f" | url={source_url}"
        ),
        reference_time=datetime.now(timezone.utc),
        source=EpisodeType.text,
        group_id=group_id,
        previous_episode_uuids=previous_episode_uuids,
    )

    uuid = getattr(result, "episode", None)
    if uuid is not None:
        uuid = getattr(uuid, "uuid", "") or ""
    return str(uuid)
