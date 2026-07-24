"""Assemble a report-ready evidence pack from the graph.

Two rules define this module:

* **Nothing without provenance.** Every item carries the episode uuids it came
  from. A conclusion that cannot name its evidence is not emitted, so the report
  has no path to assert something the graph does not hold.
* **Conflicts are surfaced, never silently resolved.** §2.3: both sides stay
  active (``expired_at IS NULL``) and the disagreement is reported.

Conflict detection is deliberately narrow, and the reason is empirical. The
obvious rule -- same subject and predicate, different objects -- was measured
against the 2026-07-24 run and every single group it flagged was a false
positive:

    ftx -acquired->     [Bitvo, Liquid, 一家破产的加密矿企]
    ftx -has_investor-> [Multicoin Capital, 安大略教师退休基金, 软银集团]
    11 月 11 日 -发生-> [three separate events]

All simultaneously true. A conflict requires the predicate to be *functional* --
single-valued, like "was founded in" or "is headquartered in" -- and nothing in
the data distinguishes those from "acquired" without semantics.

So detection is restricted to what can be decided mechanically: **contradictory
years and contradictory magnitudes** for the same subject and predicate. That is
the range §2.8's reversal authorised, and it is where the observed corruption
actually occurred (2023 vs 2026 bankruptcy years, §3.12).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from open_deep_research.graphrag.graph.queries import (
    coverage_ratio_from_gaps,
    get_gap_status,
)
from open_deep_research.graphrag.ontology import INVESTIGATION_SCHEMA, OntologySlot
from open_deep_research.graphrag.schemas import (
    ConflictRecord,
    EvidencePack,
    EvidencePackItem,
)

_FACT_QUERY = """
MATCH (subject:Entity)-[e:RELATES_TO {research_id: $research_id}]->(object:Entity)
WHERE e.expired_at IS NULL
RETURN e.uuid AS uuid,
       e.slot_id AS slot_id,
       e.fact AS fact,
       e.valid_at AS valid_at,
       e.source_url AS source_url,
       e.source_title AS source_title,
       coalesce(e.extraction_confidence, 0.0) AS confidence,
       e.episodes AS episodes,
       e.name AS predicate,
       subject.name AS subject,
       object.name AS object
ORDER BY e.slot_id, e.fact
"""

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w一-鿿]+", re.UNICODE)
# A magnitude plus the unit token that follows it: "8 billion USD", "2.75亿美元".
_MAGNITUDE = re.compile(
    r"(?P<value>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<scale>billion|million|trillion|thousand|亿|万|兆|千)?\s*"
    r"(?P<unit>USD|美元|dollars?|NT\$|新台币|人民币|EUR|欧元|BTC|%)",
    re.IGNORECASE,
)
_SCALES = {
    "": 1.0, "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12,
    "千": 1e3, "万": 1e4, "亿": 1e8, "兆": 1e12,
}


@dataclass
class FactRecord:
    """One active fact edge, with everything needed to cite it."""

    uuid: str
    slot_id: str
    fact: str
    subject: str
    predicate: str
    object: str
    confidence: float
    episodes: list[str] = field(default_factory=list)
    valid_at: Any | None = None
    source_url: str | None = None
    source_title: str | None = None

    @property
    def has_provenance(self) -> bool:
        return bool(self.episodes)


def _normalize(text: str) -> str:
    return _WS.sub(" ", _NON_WORD.sub(" ", (text or "")).strip().lower())


def _magnitudes(text: str) -> dict[str, set[float]]:
    """Numeric magnitudes in the text, keyed by unit, normalized to a base scale."""

    found: dict[str, set[float]] = {}
    for match in _MAGNITUDE.finditer(text or ""):
        try:
            value = float(match["value"].replace(",", ""))
        except ValueError:
            continue
        scale = _SCALES.get((match["scale"] or "").lower(), 1.0)
        unit = (match["unit"] or "").upper()
        # Fold currency aliases so "USD" and "美元" compare.
        if unit in {"美元", "DOLLAR", "DOLLARS"}:
            unit = "USD"
        found.setdefault(unit, set()).add(value * scale)
    return found


async def fetch_facts(graphiti: Any, *, research_id: str) -> list[FactRecord]:
    """Read all active fact edges for one research session."""

    records, _, _ = await graphiti.driver.execute_query(
        _FACT_QUERY, research_id=research_id
    )
    facts: list[FactRecord] = []
    for record in records:
        facts.append(
            FactRecord(
                uuid=record["uuid"],
                slot_id=record["slot_id"] or "",
                fact=record["fact"] or "",
                subject=record["subject"] or "",
                predicate=record["predicate"] or "",
                object=record["object"] or "",
                confidence=float(record["confidence"] or 0.0),
                episodes=[str(x) for x in (record["episodes"] or [])],
                valid_at=record["valid_at"],
                source_url=record["source_url"],
                source_title=record["source_title"],
            )
        )
    return facts


def detect_conflicts(facts: list[FactRecord]) -> list[ConflictRecord]:
    """Find mechanically decidable disagreements.

    Only years and magnitudes. Differing objects alone are not a conflict --
    see the module docstring for the measurement that rules that heuristic out.
    """

    grouped: dict[tuple[str, str], list[FactRecord]] = {}
    for fact in facts:
        key = (_normalize(fact.subject), _normalize(fact.predicate))
        if not key[0] or not key[1]:
            continue
        grouped.setdefault(key, []).append(fact)

    conflicts: list[ConflictRecord] = []
    for (subject, predicate), group in sorted(grouped.items()):
        if len(group) < 2:
            continue

        years = {f.valid_at.year for f in group if f.valid_at is not None}
        if len(years) > 1:
            conflicts.append(
                ConflictRecord(
                    conflict_id=f"date::{subject}::{predicate}",
                    slot_id=group[0].slot_id,
                    active_episode_ids=sorted(
                        {ep for f in group for ep in f.episodes}
                    ),
                    summary=(
                        f"'{group[0].subject} {group[0].predicate}' is dated to "
                        f"{sorted(years)} by different sources"
                    ),
                    leading_confidence=max(f.confidence for f in group),
                    metadata={
                        "kind": "date",
                        "years": sorted(years),
                        "edge_uuids": [f.uuid for f in group if f.valid_at],
                    },
                )
            )

        # Magnitudes are compared BETWEEN facts, never pooled across the group.
        # Pooling conflates "one fact cites a range" with "two facts disagree":
        # in the live run a single fact read "负债约为 80 亿美元，前负责人 Zane
        # Tackett 证实负债为 88 亿美元", which pooling reported as a conflict
        # although only one claim existed. Two facts disagree when their stated
        # values for a unit have nothing in common.
        per_unit: dict[str, dict[str, set[float]]] = {}
        for fact in group:
            for unit, values in _magnitudes(fact.object).items():
                per_unit.setdefault(unit, {})[fact.uuid] = values

        for unit, by_fact in sorted(per_unit.items()):
            if len(by_fact) < 2:
                continue
            disagreeing = [
                (a, b)
                for i, a in enumerate(sorted(by_fact))
                for b in sorted(by_fact)[i + 1 :]
                if not by_fact[a] & by_fact[b]
            ]
            if not disagreeing:
                continue
            involved = sorted({uuid for pair in disagreeing for uuid in pair})
            values = sorted({v for uuid in involved for v in by_fact[uuid]})
            conflicts.append(
                ConflictRecord(
                    conflict_id=f"value::{subject}::{predicate}::{unit}",
                    slot_id=group[0].slot_id,
                    active_episode_ids=sorted(
                        {ep for f in group if f.uuid in involved for ep in f.episodes}
                    ),
                    summary=(
                        f"'{group[0].subject} {group[0].predicate}' is given as "
                        f"{values} {unit} by different sources"
                    ),
                    leading_confidence=max(
                        f.confidence for f in group if f.uuid in involved
                    ),
                    metadata={
                        "kind": "magnitude",
                        "unit": unit,
                        "values": values,
                        "edge_uuids": involved,
                    },
                )
            )
    return conflicts


def _caveats(group: list[FactRecord]) -> list[str]:
    """Qualifications a reader needs in order to weigh the evidence."""

    notes: list[str] = []
    sources = {f.source_url for f in group if f.source_url}
    if len(sources) == 1:
        notes.append("single source; no cross-corroboration")
    if not any(f.valid_at for f in group):
        notes.append("no verified date: the sources did not state one explicitly")
    low = [f for f in group if f.confidence < 0.5]
    if low:
        notes.append(f"{len(low)} fact(s) below 0.5 extraction confidence")
    return notes


async def build_evidence_pack(
    graphiti: Any,
    *,
    research_id: str,
    topic: str,
    schema: dict[str, tuple[OntologySlot, ...]] | None = None,
) -> EvidencePack:
    """Assemble the pack the report is allowed to consume.

    Facts with no episode backlink are dropped rather than reported: an
    unciteable fact is exactly what §3.2's silent-provenance trap produces, and
    letting it through would put an unverifiable claim in the output.
    """

    active_schema = schema or INVESTIGATION_SCHEMA
    facts = await fetch_facts(graphiti, research_id=research_id)
    citable = [fact for fact in facts if fact.has_provenance]

    statuses = await get_gap_status(
        graphiti, research_id=research_id, schema=active_schema
    )
    coverage = coverage_ratio_from_gaps(statuses)

    by_slot: dict[str, list[FactRecord]] = {}
    for fact in citable:
        by_slot.setdefault(fact.slot_id, []).append(fact)

    items: list[EvidencePackItem] = []
    for slots in active_schema.values():
        for slot in slots:
            group = by_slot.get(slot.slot_id, [])
            if not group:
                continue
            caveats = _caveats(group)
            for fact in group:
                items.append(
                    EvidencePackItem(
                        slot_id=slot.slot_id,
                        conclusion=fact.fact,
                        confidence=fact.confidence,
                        provenance_episode_ids=list(fact.episodes),
                        caveats=caveats,
                    )
                )

    return EvidencePack(
        topic=topic,
        coverage_ratio=coverage,
        items=items,
        unresolved_conflicts=detect_conflicts(citable),
        provenance=sorted({ep for fact in citable for ep in fact.episodes}),
    )
