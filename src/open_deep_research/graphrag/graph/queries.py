"""Read coverage and gaps back out of the graph.

This is M0, and it is the judge of the project's one load-bearing assumption:
that ontology holes read *from the graph* can steer search to convergence
(SESSION_HANDOFF §5).

The distinction that makes it worth building is easy to lose. V1 tracked coverage
in an in-memory ``filled_ids`` set and marked a slot filled the moment an episode
was written for it -- regardless of whether a single fact came out the other
side. Coverage computed here is derived from the facts that actually landed, so
it differs from ``filled_ids`` exactly where it matters:

* an episode that produced no edges leaves the slot **open**, not filled;
* facts that failed verification never counted in the first place;
* the count survives a restart, because it was never in process memory.

Requires ``slot_id`` / ``research_id`` to be queryable on edges, which is why M1
had to land first: while slot attribution lived only inside an episode-name
string, the only answerable question was "did we write an episode for this slot",
which is what ``filled_ids`` already said.

No LLM is involved. Coverage is a count over persisted state, so it needs no
threshold calibration (§2.7).
"""

from __future__ import annotations

from typing import Any

from open_deep_research.graphrag.ontology import (
    INVESTIGATION_SCHEMA,
    OntologySlot,
    iter_slots,
)
from open_deep_research.graphrag.schemas import GapStatus
from open_deep_research.graphrag.schemas import (
    SlotApplicability,
    SlotApplicabilityStatus,
)

# Active facts only. §2.3: `expired_at IS NULL` is what "active" means -- there is
# deliberately no separate status field, and a contested fact is still active,
# marked by a non-empty `conflicts_with`.
_GAP_QUERY = """
MATCH ()-[e:RELATES_TO {research_id: $research_id}]->()
WHERE e.expired_at IS NULL AND e.slot_id IS NOT NULL
RETURN e.slot_id AS slot_id,
       count(e) AS fact_count,
       avg(coalesce(e.extraction_confidence, 0.0)) AS confidence,
       collect(e.episodes) AS episode_lists,
       sum(CASE WHEN size(coalesce(e.conflicts_with, [])) > 0 THEN 1 ELSE 0 END)
           AS contested_count
"""


def _flatten_episode_uuids(episode_lists: Any) -> list[str]:
    """Flatten the per-edge episode lists into a de-duplicated, ordered list."""

    seen: dict[str, None] = {}
    for episodes in episode_lists or []:
        for uuid in episodes or []:
            if uuid:
                seen.setdefault(str(uuid), None)
    return list(seen)


async def fetch_slot_evidence(
    graphiti: Any,
    *,
    research_id: str,
) -> dict[str, dict[str, Any]]:
    """Aggregate active fact edges for one research session, keyed by slot id."""

    records, _, _ = await graphiti.driver.execute_query(
        _GAP_QUERY, research_id=research_id
    )

    evidence: dict[str, dict[str, Any]] = {}
    for record in records:
        slot_id = record["slot_id"]
        if not slot_id:
            continue
        evidence[slot_id] = {
            "fact_count": int(record["fact_count"] or 0),
            "confidence": float(record["confidence"] or 0.0),
            "episode_uuids": _flatten_episode_uuids(record["episode_lists"]),
            "contested_count": int(record["contested_count"] or 0),
        }
    return evidence


async def get_gap_status(
    graphiti: Any,
    *,
    research_id: str,
    schema: dict[str, tuple[OntologySlot, ...]] | None = None,
    min_facts: int = 1,
    applicability: dict[str, SlotApplicability] | None = None,
) -> list[GapStatus]:
    """Report, per ontology slot, whether the graph actually answers it.

    ``min_facts`` is how many active fact edges a slot needs to count as filled.
    It defaults to 1 -- deliberately the weakest possible bar, so that coverage
    measures presence of evidence and nothing more. Judging *sufficiency* is a
    separate question that belongs to stopping control, not to this function.
    """

    evidence = await fetch_slot_evidence(graphiti, research_id=research_id)
    slots = iter_slots(schema or INVESTIGATION_SCHEMA)

    statuses: list[GapStatus] = []
    for slot in slots:
        decision = (applicability or {}).get(slot.slot_id)
        applicability_status = (
            decision.status
            if decision is not None
            else (
                SlotApplicabilityStatus.REQUIRED
                if slot.applicability == "always"
                else SlotApplicabilityStatus.OPTIONAL
            )
        )
        if applicability_status is SlotApplicabilityStatus.NOT_APPLICABLE:
            statuses.append(
                GapStatus(
                    slot_id=slot.slot_id,
                    dimension=slot.dimension,
                    question=slot.question,
                    filled=False,
                    applicability=applicability_status,
                    confidence=decision.confidence if decision is not None else 1.0,
                    notes=(
                        f"not applicable: {decision.reason}"
                        if decision is not None and decision.reason
                        else "not applicable to this investigation"
                    ),
                )
            )
            continue

        found = evidence.get(slot.slot_id)
        if found is None:
            statuses.append(
                GapStatus(
                    slot_id=slot.slot_id,
                    dimension=slot.dimension,
                    question=slot.question,
                    filled=False,
                    applicability=applicability_status,
                    confidence=0.0,
                    notes="no active facts in graph",
                )
            )
            continue

        fact_count = found["fact_count"]
        notes = f"{fact_count} active fact(s)"
        if found["contested_count"]:
            notes += f", {found['contested_count']} contested"

        statuses.append(
            GapStatus(
                slot_id=slot.slot_id,
                dimension=slot.dimension,
                question=slot.question,
                filled=fact_count >= min_facts,
                applicability=applicability_status,
                confidence=min(max(found["confidence"], 0.0), 1.0),
                supporting_episode_ids=found["episode_uuids"],
                notes=notes,
            )
        )

    # Slots written by an earlier, wider ontology still hold real evidence.
    # Dropping them silently would understate what the graph knows, so surface
    # them rather than let the schema decide what counts as known.
    known_slot_ids = {slot.slot_id for slot in slots}
    for slot_id, found in evidence.items():
        if slot_id in known_slot_ids:
            continue
        statuses.append(
            GapStatus(
                slot_id=slot_id,
                dimension="WHAT",
                question=f"(slot '{slot_id}' is not in the active ontology)",
                filled=found["fact_count"] >= min_facts,
                applicability=SlotApplicabilityStatus.OPTIONAL,
                confidence=min(max(found["confidence"], 0.0), 1.0),
                supporting_episode_ids=found["episode_uuids"],
                notes="orphan slot: present in graph, absent from schema",
            )
        )

    return statuses


def coverage_ratio_from_gaps(statuses: list[GapStatus]) -> float:
    """Fraction of in-schema slots the graph answers.

    Orphan slots are excluded from the denominator *and* the numerator: they are
    not part of the current investigation, so counting them would let a stale
    ontology inflate the score.
    """

    in_schema = [
        s
        for s in statuses
        if "orphan slot" not in (s.notes or "")
        and s.applicability is not SlotApplicabilityStatus.NOT_APPLICABLE
    ]
    if not in_schema:
        return 0.0
    return sum(1 for s in in_schema if s.filled) / len(in_schema)


async def get_open_slots_from_graph(
    graphiti: Any,
    *,
    research_id: str,
    schema: dict[str, tuple[OntologySlot, ...]] | None = None,
    min_facts: int = 1,
    applicability: dict[str, SlotApplicability] | None = None,
) -> list[OntologySlot]:
    """Unfilled slots, highest priority first -- the supervisor's work queue."""

    statuses = await get_gap_status(
        graphiti,
        research_id=research_id,
        schema=schema,
        min_facts=min_facts,
        applicability=applicability,
    )
    filled = {status.slot_id for status in statuses if status.filled}
    excluded = {
        status.slot_id
        for status in statuses
        if status.applicability is SlotApplicabilityStatus.NOT_APPLICABLE
    }
    open_slots = [
        slot for slot in iter_slots(schema or INVESTIGATION_SCHEMA)
        if slot.slot_id not in filled and slot.slot_id not in excluded
    ]
    return sorted(open_slots, key=lambda slot: (-slot.priority, slot.slot_id))
