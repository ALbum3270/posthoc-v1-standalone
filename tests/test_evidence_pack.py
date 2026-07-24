"""Tests for evidence assembly and report rendering (M3).

The conflict fixtures are drawn from the 2026-07-24 live run, including the
groups that a naive "same subject+predicate, different object" rule flags and
which are in fact all simultaneously true.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from open_deep_research.graphrag.ontology import OntologySlot
from open_deep_research.graphrag.reporting.evidence_pack import (
    FactRecord,
    build_evidence_pack,
    detect_conflicts,
    fetch_facts,
)
from open_deep_research.graphrag.reporting.report import build_source_index, render_report
from open_deep_research.graphrag.schemas import EvidencePack, EvidencePackItem

SCHEMA: dict[str, tuple[OntologySlot, ...]] = {
    "WHO": (
        OntologySlot(slot_id="who.primary_actor", dimension="WHO", label="主要行为人",
                     question="Who?", priority=100),
        OntologySlot(slot_id="who.affected", dimension="WHO", label="受影响方",
                     question="Affected?", priority=90),
    ),
    "WHEN": (
        OntologySlot(slot_id="when.event_time", dimension="WHEN", label="事发时间",
                     question="When?", priority=95),
    ),
}


def fact(
    subject: str,
    predicate: str,
    obj: str,
    *,
    slot_id: str = "who.primary_actor",
    uuid: str = "e-1",
    episodes: list[str] | None = None,
    valid_at: datetime | None = None,
    confidence: float = 0.8,
    url: str | None = "https://a.example",
    title: str | None = "Source A",
) -> FactRecord:
    return FactRecord(
        uuid=uuid,
        slot_id=slot_id,
        fact=f"{subject} {predicate} {obj}",
        subject=subject,
        predicate=predicate,
        object=obj,
        confidence=confidence,
        episodes=episodes if episodes is not None else ["ep-1"],
        valid_at=valid_at,
        source_url=url,
        source_title=title,
    )


class FakeDriver:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    async def execute_query(self, query, **kwargs):
        self.queries.append((str(query), kwargs))
        if "slot_id IS NOT NULL" in str(query):  # the gap-status query
            return [{"slot_id": "who.primary_actor", "fact_count": 1, "confidence": 0.8,
                     "episode_lists": [["ep-1"]], "contested_count": 0}], None, None
        return self.rows, None, None


class FakeGraphiti:
    def __init__(self, rows):
        self.driver = FakeDriver(rows)


def row(**overrides):
    base = {
        "uuid": "e-1", "slot_id": "who.primary_actor", "fact": "FTX filed for bankruptcy",
        "valid_at": None, "source_url": "https://a.example", "source_title": "Source A",
        "confidence": 0.8, "episodes": ["ep-1"], "predicate": "FILED_FOR",
        "subject": "FTX", "object": "bankruptcy",
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# conflict detection
# --------------------------------------------------------------------------


def test_multi_valued_predicates_are_not_conflicts() -> None:
    """Measured against the live run: every one of these is simultaneously true.

    A rule based on object multiplicity alone would have flagged all of them.
    """

    facts = [
        fact("FTX", "acquired", "Bitvo", uuid="a"),
        fact("FTX", "acquired", "Liquid", uuid="b"),
        fact("FTX", "acquired", "一家破产的加密矿企", uuid="c"),
        fact("FTX", "has_investor", "Multicoin Capital", uuid="d"),
        fact("FTX", "has_investor", "软银集团", uuid="e"),
        fact("11 月 11 日", "发生", "FTX 申请破产保护", uuid="f"),
        fact("11 月 11 日", "发生", "SBF 辞职", uuid="g"),
    ]

    assert detect_conflicts(facts) == []


def test_contradictory_years_are_a_conflict() -> None:
    """The actual v1 corruption: one event dated to two different years."""

    facts = [
        fact("FTX", "announced bankruptcy", "in mid-November 2023", uuid="a",
             episodes=["ep-1"], valid_at=datetime(2023, 11, 15, tzinfo=timezone.utc)),
        fact("FTX", "announced bankruptcy", "in mid-November 2026", uuid="b",
             episodes=["ep-2"], valid_at=datetime(2026, 11, 15, tzinfo=timezone.utc)),
    ]

    conflicts = detect_conflicts(facts)

    assert len(conflicts) == 1
    assert conflicts[0].metadata["kind"] == "date"
    assert conflicts[0].metadata["years"] == [2023, 2026]
    assert sorted(conflicts[0].active_episode_ids) == ["ep-1", "ep-2"]


def test_same_year_from_two_sources_is_agreement_not_conflict() -> None:
    facts = [
        fact("FTX", "filed", "on 11 November 2022", uuid="a",
             valid_at=datetime(2022, 11, 11, tzinfo=timezone.utc)),
        fact("FTX", "filed", "on November 2022", uuid="b",
             valid_at=datetime(2022, 11, 1, tzinfo=timezone.utc)),
    ]

    assert detect_conflicts(facts) == []


def test_contradictory_magnitudes_are_a_conflict() -> None:
    facts = [
        fact("Taiwan losses", "estimated at", "20 billion USD", uuid="a"),
        fact("Taiwan losses", "estimated at", "2 billion USD", uuid="b"),
    ]

    conflicts = detect_conflicts(facts)

    assert len(conflicts) == 1
    assert conflicts[0].metadata["kind"] == "magnitude"
    assert conflicts[0].metadata["values"] == [2e9, 2e10]


def test_currency_aliases_are_compared_together() -> None:
    facts = [
        fact("淡马锡", "投资", "2.75亿美元", uuid="a"),
        fact("淡马锡", "投资", "500000000 USD", uuid="b"),
    ]

    conflicts = detect_conflicts(facts)
    assert len(conflicts) == 1
    assert conflicts[0].metadata["unit"] == "USD"


def test_identical_magnitudes_agree() -> None:
    facts = [
        fact("红杉", "投资", "2.14亿美元", uuid="a"),
        fact("红杉", "投资", "2.14亿美元", uuid="b", url="https://b.example"),
    ]
    assert detect_conflicts(facts) == []


def test_a_single_fact_cannot_conflict_with_itself() -> None:
    assert detect_conflicts([fact("FTX", "is", "an exchange")]) == []


# --------------------------------------------------------------------------
# pack assembly
# --------------------------------------------------------------------------


def test_facts_are_fetched_with_endpoints_and_provenance() -> None:
    facts = asyncio.run(fetch_facts(FakeGraphiti([row()]), research_id="r-1"))

    assert len(facts) == 1
    assert facts[0].subject == "FTX"
    assert facts[0].episodes == ["ep-1"]
    assert facts[0].has_provenance is True


def test_facts_without_provenance_never_enter_the_pack() -> None:
    """An uncitable fact is what §3.2's silent-provenance trap produces."""

    graphiti = FakeGraphiti([row(uuid="good"), row(uuid="orphan", episodes=[])])

    pack = asyncio.run(
        build_evidence_pack(graphiti, research_id="r-1", topic="FTX", schema=SCHEMA)
    )

    assert len(pack.items) == 1
    assert all(item.provenance_episode_ids for item in pack.items)


def test_single_source_is_flagged_as_a_caveat() -> None:
    graphiti = FakeGraphiti([row(uuid="a"), row(uuid="b")])

    pack = asyncio.run(
        build_evidence_pack(graphiti, research_id="r-1", topic="FTX", schema=SCHEMA)
    )

    assert any("single source" in c for c in pack.items[0].caveats)


def test_undated_evidence_is_flagged() -> None:
    pack = asyncio.run(
        build_evidence_pack(
            FakeGraphiti([row()]), research_id="r-1", topic="FTX", schema=SCHEMA
        )
    )

    assert any("no verified date" in c for c in pack.items[0].caveats)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def make_pack(items, conflicts=None) -> EvidencePack:
    return EvidencePack(
        topic="FTX 暴雷事件",
        coverage_ratio=1 / 3,
        items=items,
        unresolved_conflicts=conflicts or [],
        provenance=["ep-1"],
    )


def test_unfilled_slots_are_stated_not_omitted() -> None:
    """Nine answers and eight silences must not read as a complete report."""

    pack = make_pack([
        EvidencePackItem(slot_id="who.primary_actor", conclusion="SBF founded FTX",
                         confidence=0.8, provenance_episode_ids=["ep-1"])
    ])

    report = render_report(pack, schema=SCHEMA)

    assert "SBF founded FTX" in report
    assert report.count("未查到相关证据") == 2  # who.affected and when.event_time


def test_items_without_provenance_are_refused_at_render_time() -> None:
    """Defence in depth: the last gate must not rely on an upstream invariant."""

    pack = make_pack([
        EvidencePackItem(slot_id="who.primary_actor", conclusion="unbacked claim",
                         confidence=0.9, provenance_episode_ids=[])
    ])

    report = render_report(pack, schema=SCHEMA)

    assert "unbacked claim" not in report
    assert report.count("未查到相关证据") == 3


def test_conclusions_carry_their_episode_id() -> None:
    pack = make_pack([
        EvidencePackItem(slot_id="who.primary_actor", conclusion="SBF founded FTX",
                         confidence=0.8, provenance_episode_ids=["abcdef123456"])
    ])

    assert "`abcdef12`" in render_report(pack, schema=SCHEMA)


def test_sources_are_cited_when_an_index_is_supplied() -> None:
    pack = make_pack([
        EvidencePackItem(slot_id="who.primary_actor", conclusion="SBF founded FTX",
                         confidence=0.8, provenance_episode_ids=["ep-1"])
    ])
    sources = build_source_index([fact("FTX", "is", "an exchange")])

    report = render_report(pack, schema=SCHEMA, sources=sources)

    assert "[Source A](https://a.example)" in report


def test_conflicts_are_reported_not_resolved() -> None:
    conflicts = detect_conflicts([
        fact("FTX", "announced bankruptcy", "2023", uuid="a", episodes=["ep-1"],
             valid_at=datetime(2023, 11, 15, tzinfo=timezone.utc)),
        fact("FTX", "announced bankruptcy", "2026", uuid="b", episodes=["ep-2"],
             valid_at=datetime(2026, 11, 15, tzinfo=timezone.utc)),
    ])
    pack = make_pack([
        EvidencePackItem(slot_id="who.primary_actor", conclusion="x",
                         confidence=0.8, provenance_episode_ids=["ep-1"])
    ], conflicts=conflicts)

    report = render_report(pack, schema=SCHEMA)

    assert "未消解的冲突" in report
    assert "[2023, 2026]" in report
    assert "1 处未消解" in report


def test_caveats_surface_in_the_report() -> None:
    pack = make_pack([
        EvidencePackItem(slot_id="who.primary_actor", conclusion="SBF founded FTX",
                         confidence=0.8, provenance_episode_ids=["ep-1"],
                         caveats=["single source; no cross-corroboration"])
    ])

    assert "single source" in render_report(pack, schema=SCHEMA)


def test_empty_pack_renders_an_honest_empty_report() -> None:
    report = render_report(make_pack([]), schema=SCHEMA)

    assert report.count("未查到相关证据") == 3
    assert "调查报告" in report


def test_one_fact_citing_two_figures_is_not_a_conflict() -> None:
    """Observed live: a single source stated both 80亿 and 88亿 for FTX's debt.

    Pooling magnitudes across a group reported that as a disagreement even though
    only one claim existed. A range inside one fact is a qualification, not a
    contradiction.
    """

    facts = [
        fact("11 月 11 日", "发生", "FTX US 暂停提款", uuid="a"),
        fact("11 月 11 日", "发生",
             "FTX 负债约为 80 亿美元，前负责人 Zane Tackett 证实负债为 88 亿美元", uuid="b"),
    ]

    assert detect_conflicts(facts) == []


def test_two_facts_sharing_a_value_are_not_in_conflict() -> None:
    """A range that overlaps a point estimate is compatible, not contradictory."""

    facts = [
        fact("FTX debt", "was", "between 8 billion USD and 8.8 billion USD", uuid="a"),
        fact("FTX debt", "was", "8 billion USD", uuid="b", url="https://b.example"),
    ]

    assert detect_conflicts(facts) == []


def test_two_facts_with_disjoint_values_do_conflict() -> None:
    facts = [
        fact("Taiwan losses", "estimated at", "20 billion USD", uuid="a"),
        fact("Taiwan losses", "estimated at", "2 billion USD", uuid="b",
             url="https://b.example"),
    ]

    conflicts = detect_conflicts(facts)
    assert len(conflicts) == 1
    assert conflicts[0].metadata["edge_uuids"] == ["a", "b"]
