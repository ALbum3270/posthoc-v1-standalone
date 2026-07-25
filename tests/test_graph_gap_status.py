"""Tests for graph-derived coverage (M0).

The property under test is not "does it count rows" but "does it differ from the
in-memory filled_ids it replaces". V1 marked a slot filled when an episode was
written for it; these tests pin the cases where that answer and the graph's
answer come apart.
"""

from __future__ import annotations

import asyncio

import pytest

from open_deep_research.graphrag.control.stopping import (
    StopReason,
    evaluate_stop,
)
from open_deep_research.graphrag.graph.queries import (
    coverage_ratio_from_gaps,
    get_gap_status,
    get_open_slots_from_graph,
)
from open_deep_research.graphrag.ontology import INVESTIGATION_SCHEMA, OntologySlot
from open_deep_research.graphrag.schemas import (
    SlotApplicability,
    SlotApplicabilityStatus,
)

SCHEMA: dict[str, tuple[OntologySlot, ...]] = {
    "WHO": (
        OntologySlot(
            slot_id="who.primary_actor",
            dimension="WHO",
            label="Primary Actor",
            question="Who is the main actor?",
            priority=100,
        ),
        OntologySlot(
            slot_id="who.affected_parties",
            dimension="WHO",
            label="Affected Parties",
            question="Who is affected?",
            priority=90,
        ),
    ),
    "WHY": (
        OntologySlot(
            slot_id="why.motive",
            dimension="WHY",
            label="Motive",
            question="Why did it happen?",
            priority=80,
        ),
    ),
}


class FakeDriver:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.queries: list[tuple[str, dict]] = []

    async def execute_query(self, query, **kwargs):
        self.queries.append((str(query), kwargs))
        return self.rows, None, None


class FakeGraphiti:
    def __init__(self, rows: list[dict]) -> None:
        self.driver = FakeDriver(rows)


def row(
    slot_id: str,
    fact_count: int = 1,
    confidence: float = 0.8,
    episode_lists: list | None = None,
    contested_count: int = 0,
) -> dict:
    return {
        "slot_id": slot_id,
        "fact_count": fact_count,
        "confidence": confidence,
        "episode_lists": episode_lists if episode_lists is not None else [["ep-1"]],
        "contested_count": contested_count,
    }


def gaps(rows: list[dict], **kwargs):
    return asyncio.run(
        get_gap_status(FakeGraphiti(rows), research_id="r-1", schema=SCHEMA, **kwargs)
    )


def by_id(statuses) -> dict:
    return {s.slot_id: s for s in statuses}


# --------------------------------------------------------------------------


def test_slot_with_facts_is_filled() -> None:
    status = by_id(gaps([row("who.primary_actor", fact_count=3, confidence=0.9)]))

    assert status["who.primary_actor"].filled is True
    assert status["who.primary_actor"].confidence == pytest.approx(0.9)
    assert status["who.primary_actor"].supporting_episode_ids == ["ep-1"]
    assert "3 active fact(s)" in status["who.primary_actor"].notes


def test_slot_with_no_facts_is_open() -> None:
    status = by_id(gaps([row("who.primary_actor")]))

    assert status["why.motive"].filled is False
    assert status["why.motive"].confidence == 0.0
    assert status["why.motive"].notes == "no active facts in graph"


def test_episode_without_facts_leaves_the_slot_open() -> None:
    """The case that separates this from V1's filled_ids.

    V1 added the slot to filled_ids as soon as an episode was written, even when
    extraction yielded nothing usable. The graph only knows about facts, so a
    barren episode contributes no coverage at all -- there is no row for it.
    """

    status = by_id(gaps([]))

    assert all(not s.filled for s in status.values())
    assert coverage_ratio_from_gaps(list(status.values())) == 0.0


def test_coverage_ratio_counts_in_schema_slots() -> None:
    statuses = gaps([row("who.primary_actor"), row("why.motive")])
    assert coverage_ratio_from_gaps(statuses) == pytest.approx(2 / 3)


def test_min_facts_raises_the_bar() -> None:
    statuses = gaps([row("who.primary_actor", fact_count=1)], min_facts=2)
    assert by_id(statuses)["who.primary_actor"].filled is False


def test_contested_facts_still_count_but_are_flagged() -> None:
    """§2.3: contested is expressed by conflicts_with, and is still active."""

    status = by_id(gaps([row("who.primary_actor", fact_count=2, contested_count=1)]))

    assert status["who.primary_actor"].filled is True
    assert "1 contested" in status["who.primary_actor"].notes


def test_episode_ids_are_flattened_and_deduplicated() -> None:
    status = by_id(
        gaps([row("who.primary_actor", episode_lists=[["ep-1", "ep-2"], ["ep-1"], []])])
    )

    assert status["who.primary_actor"].supporting_episode_ids == ["ep-1", "ep-2"]


def test_orphan_slots_are_surfaced_but_excluded_from_the_ratio() -> None:
    """Evidence written under a wider ontology is real; the score is not inflated."""

    statuses = gaps([row("who.primary_actor"), row("legacy.slot")])
    orphan = by_id(statuses)["legacy.slot"]

    assert orphan.filled is True
    assert "orphan slot" in orphan.notes
    # 1 of 3 in-schema slots filled; the orphan touches neither side of it.
    assert coverage_ratio_from_gaps(statuses) == pytest.approx(1 / 3)


def test_query_filters_to_active_facts_and_the_research_session() -> None:
    graphiti = FakeGraphiti([])
    asyncio.run(get_gap_status(graphiti, research_id="r-42", schema=SCHEMA))

    query, kwargs = graphiti.driver.queries[0]
    assert kwargs == {"research_id": "r-42"}
    assert "expired_at IS NULL" in query
    assert "slot_id IS NOT NULL" in query


def test_open_slots_are_ordered_by_priority() -> None:
    slots = asyncio.run(
        get_open_slots_from_graph(
            FakeGraphiti([row("who.primary_actor")]), research_id="r-1", schema=SCHEMA
        )
    )

    assert [s.slot_id for s in slots] == ["who.affected_parties", "why.motive"]


def test_defaults_to_the_real_investigation_schema() -> None:
    statuses = asyncio.run(get_gap_status(FakeGraphiti([]), research_id="r-1"))

    expected = sum(len(v) for v in INVESTIGATION_SCHEMA.values())
    assert len(statuses) == expected
    assert coverage_ratio_from_gaps(statuses) == 0.0


def test_not_applicable_slot_is_excluded_from_coverage_and_work_queue() -> None:
    applicability = {
        "why.motive": SlotApplicability(
            slot_id="why.motive",
            status=SlotApplicabilityStatus.NOT_APPLICABLE,
            confidence=0.95,
            reason="accidental event has no intent",
        )
    }
    graphiti = FakeGraphiti([row("who.primary_actor")])
    statuses = asyncio.run(
        get_gap_status(
            graphiti,
            research_id="r-1",
            schema=SCHEMA,
            applicability=applicability,
        )
    )
    open_slots = asyncio.run(
        get_open_slots_from_graph(
            graphiti,
            research_id="r-1",
            schema=SCHEMA,
            applicability=applicability,
        )
    )

    assert coverage_ratio_from_gaps(statuses) == pytest.approx(1 / 2)
    assert by_id(statuses)["why.motive"].applicability is (
        SlotApplicabilityStatus.NOT_APPLICABLE
    )
    assert [slot.slot_id for slot in open_slots] == ["who.affected_parties"]


def test_empty_optional_slot_is_bonus_and_does_not_block_coverage() -> None:
    applicability = {
        "why.motive": SlotApplicability(
            slot_id="why.motive",
            status=SlotApplicabilityStatus.OPTIONAL,
            confidence=0.8,
            reason="meaningful but not required",
        )
    }
    graphiti = FakeGraphiti(
        [row("who.primary_actor"), row("who.affected_parties")]
    )
    statuses = asyncio.run(
        get_gap_status(
            graphiti,
            research_id="r-1",
            schema=SCHEMA,
            applicability=applicability,
        )
    )
    open_slots = asyncio.run(
        get_open_slots_from_graph(
            graphiti,
            research_id="r-1",
            schema=SCHEMA,
            applicability=applicability,
        )
    )

    coverage = coverage_ratio_from_gaps(statuses)
    decision = evaluate_stop(
        round_number=1,
        coverage_ratio=coverage,
        rounds_without_improvement=0,
        open_slot_count=len(open_slots),
    )

    assert by_id(statuses)["why.motive"].filled is False
    assert coverage == 1.0
    assert open_slots == []
    assert decision.reason is StopReason.COVERAGE_REACHED
