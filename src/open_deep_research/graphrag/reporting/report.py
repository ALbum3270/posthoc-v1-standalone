"""Render a report from an evidence pack, and from nothing else.

The report has no access to search results, notes or raw text -- only the pack.
That is the point: if a sentence appears here, some episode in the graph backs
it, and the episode id is printed next to it. There is no path by which the
writer can add a claim of its own.

Unfilled slots are stated as unfilled. An investigation that answered nine of
seventeen questions should read as one, not as nine answers with eight silences.
"""

from __future__ import annotations

from typing import Iterable

from open_deep_research.graphrag.ontology import INVESTIGATION_SCHEMA, OntologySlot
from open_deep_research.graphrag.reporting.evidence_pack import FactRecord
from open_deep_research.graphrag.schemas import EvidencePack

_DIMENSION_ORDER = ("WHO", "WHAT", "WHEN", "WHERE", "WHY", "HOW")


def build_source_index(facts: Iterable[FactRecord]) -> dict[str, str]:
    """Map episode uuid -> a human-readable citation."""

    index: dict[str, str] = {}
    for fact in facts:
        label = fact.source_title or fact.source_url or ""
        for episode in fact.episodes:
            if episode not in index and (fact.source_url or label):
                index[episode] = (
                    f"[{label}]({fact.source_url})" if fact.source_url else label
                )
    return index


def render_report(
    pack: EvidencePack,
    *,
    schema: dict[str, tuple[OntologySlot, ...]] | None = None,
    sources: dict[str, str] | None = None,
    show_episode_ids: bool = True,
) -> str:
    """Render the pack as Markdown.

    Items lacking provenance are skipped even though ``build_evidence_pack``
    already filters them: this is the last gate before text reaches a reader, and
    it should not depend on an upstream invariant holding.
    """

    active_schema = schema or INVESTIGATION_SCHEMA
    source_index = sources or {}

    by_slot: dict[str, list] = {}
    for item in pack.items:
        if not item.provenance_episode_ids:
            continue
        by_slot.setdefault(item.slot_id, []).append(item)

    all_slots = [slot for slots in active_schema.values() for slot in slots]
    filled_count = sum(1 for slot in all_slots if by_slot.get(slot.slot_id))

    lines = [
        f"# 调查报告：{pack.topic}",
        "",
        f"> **覆盖率** {filled_count}/{len(all_slots)} 槽位（{pack.coverage_ratio:.0%}）　"
        f"**事实** {len(pack.items)} 条　**来源 episode** {len(pack.provenance)} 个",
        "",
        "> 本报告仅消费图谱证据包。每条结论都标注了来源 episode；"
        "无证据的槽位如实标记为未查到，不做推断补全。",
        "",
    ]

    for dimension in _DIMENSION_ORDER:
        dimension_slots = [s for s in all_slots if s.dimension == dimension]
        if not dimension_slots:
            continue
        lines += [f"## {dimension}", ""]
        for slot in dimension_slots:
            items = by_slot.get(slot.slot_id, [])
            if not items:
                lines += [f"**{slot.label}** — ⚠️ 未查到相关证据", ""]
                continue

            lines.append(f"**{slot.label}**")
            for item in items:
                citation = ""
                if show_episode_ids and item.provenance_episode_ids:
                    citation = f" `{item.provenance_episode_ids[0][:8]}`"
                lines.append(f"  - {item.conclusion}{citation}")

            cited = {
                source_index[episode]
                for item in items
                for episode in item.provenance_episode_ids
                if episode in source_index
            }
            if cited:
                lines.append("  来源：" + "、".join(sorted(cited)))
            caveats = sorted({c for item in items for c in item.caveats})
            if caveats:
                lines.append("  ⚠️ " + "；".join(caveats))
            lines.append("")

    if pack.unresolved_conflicts:
        lines += ["## ⚔️ 未消解的冲突", "",
                  "以下事实互相矛盾，双方都保留在图谱中，未做裁决（§2.3）：", ""]
        for conflict in pack.unresolved_conflicts:
            lines.append(f"- **{conflict.slot_id}** — {conflict.summary}")
            if conflict.active_episode_ids:
                shown = ", ".join(e[:8] for e in conflict.active_episode_ids[:4])
                lines.append(f"  涉及 episode：`{shown}`")
        lines.append("")

    lines += [
        "---",
        f"*由证据包生成。冲突 {len(pack.unresolved_conflicts)} 处未消解。"
        "无明确日期的事实不带 valid_at（§3.12）。*",
    ]
    return "\n".join(lines)
