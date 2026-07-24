"""V1 报告合成器：按写入 Episode 精确回溯每个槽位的证据。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from v1.ontology import SLOTS_SORTED, coverage_pct


@dataclass(frozen=True)
class EvidenceRef:
    """One source Episode written for an ontology slot."""

    episode_uuid: str
    title: str
    url: str


def _source_line(evidence: EvidenceRef) -> str:
    """Render a source link without letting titles break Markdown syntax."""

    title = evidence.title.replace("[", "").replace("]", "").strip() or evidence.url
    if evidence.url:
        return f"  - [{title}]({evidence.url}) (`{evidence.episode_uuid}`)"
    return f"  - {title} (`{evidence.episode_uuid}`)"


async def build_report(
    graphiti: Any,
    *,
    topic: str,
    evidence_by_slot: dict[str, list[EvidenceRef]],
) -> str:
    """Read exact Episode subgraphs and generate a slot-isolated report."""

    sections: dict[str, list[str]] = {}  # dimension → list of bullet lines
    filled_ids = {
        slot_id
        for slot_id, evidence in evidence_by_slot.items()
        if evidence
    }

    for slot in SLOTS_SORTED:
        sid = slot["slot_id"]
        dim = slot["dimension"]
        label = slot["label"]
        evidence_refs = evidence_by_slot.get(sid, [])

        if evidence_refs:
            subgraph = await graphiti.get_nodes_and_edges_by_episode(
                [evidence.episode_uuid for evidence in evidence_refs]
            )
            facts = getattr(subgraph, "edges", [])
            bullet_lines = []
            seen_facts: set[str] = set()
            for edge in facts:
                fact_text = getattr(edge, "fact", None) or str(edge)
                normalized_fact = fact_text.strip().casefold()
                if not normalized_fact or normalized_fact in seen_facts:
                    continue
                seen_facts.add(normalized_fact)
                bullet_lines.append(f"  - {fact_text}")

            if bullet_lines:
                block_lines = [f"**{label}**", *bullet_lines[:8], "", "来源："]
                block_lines.extend(_source_line(evidence) for evidence in evidence_refs)
                sections.setdefault(dim, []).append("\n".join(block_lines))
            else:
                sections.setdefault(dim, []).append(
                    f"**{label}** — *Episode 已写入，但没有可回溯的事实边*"
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
    lines.append(
        "*本报告由 V1 最小闭环系统生成；事实按 Episode UUID 精确回溯，"
        "未进行冲突消解。*"
    )

    return "\n".join(lines)
