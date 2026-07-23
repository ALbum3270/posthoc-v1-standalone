"""
报告合成器（V1 极简版）。

从 Graphiti 中检索当前 research_id (group_id) 下的知识，
按 5W1H 维度组织成 Markdown 报告。没有冲突消解，直接输出。
"""

from __future__ import annotations

from typing import Any

from v1.ontology import SLOTS_SORTED, coverage_pct


async def build_report(
    graphiti: Any,
    *,
    group_id: str,
    topic: str,
    filled_ids: set[str],
) -> str:
    """
    从图谱检索证据，生成 Markdown 报告。

    策略：
    - 对每个已填充的槽位，用 slot.question 做混合检索（group_ids=[group_id]）
    - 对每个未填充的槽位，标记为 "⚠️ 未查到相关证据"
    """
    sections: dict[str, list[str]] = {}  # dimension → list of bullet lines

    for slot in SLOTS_SORTED:
        sid = slot["slot_id"]
        dim = slot["dimension"]
        label = slot["label"]

        if sid in filled_ids:
            facts = await graphiti.search(
                slot["question"],
                group_ids=[group_id],
                num_results=3,
            )
            bullet_lines = []
            for edge in facts:
                fact_text = getattr(edge, "fact", None) or str(edge)
                bullet_lines.append(f"  - {fact_text}")

            if bullet_lines:
                sections.setdefault(dim, []).append(
                    f"**{label}**\n" + "\n".join(bullet_lines)
                )
            else:
                sections.setdefault(dim, []).append(
                    f"**{label}** — *已写入图谱，但检索无返回，可能仍在索引中*"
                )
        else:
            sections.setdefault(dim, []).append(
                f"**{label}** — ⚠️ 未查到相关证据"
            )

    pct = coverage_pct(filled_ids)
    filled_count = len(filled_ids)
    total_count = len(SLOTS_SORTED)

    lines = [
        f"# 调查报告：{topic}",
        "",
        f"> **图谱覆盖率**：{filled_count}/{total_count} 槽位已填充（{pct:.0f}%）",
        "",
    ]

    for dim in ["WHO", "WHAT", "WHEN", "WHERE", "WHY", "HOW"]:
        if dim in sections:
            lines.append(f"## {dim}")
            lines.append("")
            for block in sections[dim]:
                lines.append(block)
                lines.append("")

    lines.append("---")
    lines.append("*本报告由 V1 最小闭环系统生成，数据直接来自 Graphiti 图谱，无冲突消解。*")

    return "\n".join(lines)
