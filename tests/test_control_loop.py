"""Tests for slot selection and stopping (M2 control flow).

The central case is the V1 livelock: seven consecutive rounds on one slot with a
byte-identical query. ``test_failure_breaks_the_livelock`` replays that scenario
and asserts it cannot recur.
"""

from __future__ import annotations

import asyncio

import pytest

from open_deep_research.graphrag.control.stopping import (
    StopReason,
    StoppingConfig,
    count_improvement,
    evaluate_stop,
)
from open_deep_research.graphrag.control.supervisor import (
    SupervisorMemory,
    plan_next_round,
    select_next_slot,
)
from open_deep_research.graphrag.ontology import OntologySlot


def slot(slot_id: str, priority: int) -> OntologySlot:
    return OntologySlot(
        slot_id=slot_id,
        dimension="WHAT",
        label=slot_id,
        question=f"question for {slot_id}?",
        priority=priority,
    )


SLOTS = [slot("a.high", 100), slot("b.mid", 90), slot("c.low", 80)]


# --------------------------------------------------------------------------
# slot selection
# --------------------------------------------------------------------------


def test_highest_priority_wins_when_nothing_has_failed() -> None:
    assert select_next_slot(SLOTS, SupervisorMemory()).slot_id == "a.high"


def test_failure_breaks_the_livelock() -> None:
    """The V1 failure mode, replayed.

    V1 re-picked the failing slot every round because its input never changed.
    Here one failure is enough to let an untried slot through.
    """

    memory = SupervisorMemory()
    memory.record_attempt("a.high", query="q1")
    memory.record_failure("a.high")

    assert select_next_slot(SLOTS, memory).slot_id == "b.mid"


def test_a_failing_slot_is_retried_only_after_the_others() -> None:
    memory = SupervisorMemory()
    chosen: list[str] = []

    for _ in range(4):
        picked = select_next_slot(SLOTS, memory)
        if picked is None:
            break
        chosen.append(picked.slot_id)
        memory.record_attempt(picked.slot_id, query=f"q-{picked.slot_id}")
        memory.record_failure(picked.slot_id)

    # Every slot is tried once before any is tried twice.
    assert chosen[:3] == ["a.high", "b.mid", "c.low"]
    assert chosen[3] == "a.high"


def test_exhausted_slots_are_skipped() -> None:
    memory = SupervisorMemory()
    for _ in range(3):
        memory.record_attempt("a.high", query="q")

    assert select_next_slot(SLOTS, memory, max_attempts_per_slot=3).slot_id == "b.mid"


def test_all_exhausted_returns_none() -> None:
    memory = SupervisorMemory()
    for candidate in SLOTS:
        for _ in range(3):
            memory.record_attempt(candidate.slot_id, query="q")

    assert select_next_slot(SLOTS, memory, max_attempts_per_slot=3) is None


def test_success_clears_the_failure_streak() -> None:
    memory = SupervisorMemory()
    memory.record_attempt("a.high", query="q")
    memory.record_failure("a.high")
    memory.record_success("a.high")

    assert select_next_slot(SLOTS, memory).slot_id == "a.high"


def test_seen_urls_accumulate_without_duplicates() -> None:
    memory = SupervisorMemory()
    memory.record_attempt("a.high", query="q1", urls=["u1", "u2"])
    memory.record_attempt("a.high", query="q2", urls=["u2", "u3"])

    assert memory.for_slot("a.high").seen_urls == ["u1", "u2", "u3"]
    assert memory.for_slot("a.high").queries == ["q1", "q2"]


# --------------------------------------------------------------------------
# round planning
# --------------------------------------------------------------------------


def test_plan_passes_previous_queries_to_the_generator() -> None:
    memory = SupervisorMemory()
    memory.record_attempt("a.high", query="old query", urls=["seen-url"])
    memory.record_failure("a.high")
    memory.record_success("a.high")  # keep a.high top-ranked
    seen: dict = {}

    async def generator(*, topic, slot, previous_queries):
        seen["previous"] = previous_queries
        return "a fresh query"

    result = asyncio.run(plan_next_round("FTX", SLOTS, memory, generator))

    assert result is not None
    chosen, query, exclude_urls = result
    assert chosen.slot_id == "a.high"
    assert query == "a fresh query"
    assert seen["previous"] == ["old query"]
    assert exclude_urls == ["seen-url"]


def test_plan_falls_back_to_the_slot_question_on_an_empty_query() -> None:
    async def generator(*, topic, slot, previous_queries):
        return "   "

    result = asyncio.run(plan_next_round("FTX", SLOTS, SupervisorMemory(), generator))

    assert result is not None
    assert result[1] == "question for a.high?"


def test_plan_returns_none_when_everything_is_exhausted() -> None:
    memory = SupervisorMemory()
    for candidate in SLOTS:
        for _ in range(3):
            memory.record_attempt(candidate.slot_id, query="q")

    async def generator(*, topic, slot, previous_queries):
        raise AssertionError("must not be called")

    assert asyncio.run(plan_next_round("FTX", SLOTS, memory, generator)) is None


# --------------------------------------------------------------------------
# stopping
# --------------------------------------------------------------------------


def test_continue_while_work_remains() -> None:
    decision = evaluate_stop(
        round_number=2, coverage_ratio=0.3, rounds_without_improvement=0, open_slot_count=5
    )
    assert decision.should_stop is False
    assert decision.reason is StopReason.CONTINUE


def test_coverage_target_stops_and_counts_as_success() -> None:
    decision = evaluate_stop(
        round_number=5,
        coverage_ratio=1.0,
        rounds_without_improvement=0,
        open_slot_count=0,
    )
    assert decision.should_stop is True
    assert decision.reason is StopReason.COVERAGE_REACHED
    assert decision.is_success is True


def test_no_improvement_stops_but_is_not_success() -> None:
    decision = evaluate_stop(
        round_number=6,
        coverage_ratio=0.4,
        rounds_without_improvement=3,
        open_slot_count=4,
    )
    assert decision.should_stop is True
    assert decision.reason is StopReason.NO_IMPROVEMENT
    assert decision.is_success is False


def test_all_slots_exhausted_stops() -> None:
    """The terminal condition V1 lacked: nothing left worth trying."""

    decision = evaluate_stop(
        round_number=8,
        coverage_ratio=0.4,
        rounds_without_improvement=1,
        open_slot_count=3,
        exhausted_slot_count=3,
    )
    assert decision.should_stop is True
    assert decision.reason is StopReason.ALL_SLOTS_EXHAUSTED


def test_max_rounds_is_the_last_resort() -> None:
    decision = evaluate_stop(
        round_number=25,
        coverage_ratio=0.4,
        rounds_without_improvement=0,
        open_slot_count=4,
        config=StoppingConfig(max_rounds=24),
    )
    assert decision.reason is StopReason.MAX_ROUNDS


def test_finishing_outranks_running_out() -> None:
    """Reason ordering matters: a completed run must not report as capped."""

    decision = evaluate_stop(
        round_number=999,
        coverage_ratio=1.0,
        rounds_without_improvement=99,
        open_slot_count=0,
        config=StoppingConfig(max_rounds=1),
    )
    assert decision.reason is StopReason.COVERAGE_REACHED


@pytest.mark.parametrize(
    "previous,current,counter,expected",
    [
        (0.2, 0.4, 2, 0),  # improved -> reset
        (0.4, 0.4, 2, 3),  # flat -> no progress
        (0.4, 0.3, 0, 1),  # regressed (edge expired) -> also no progress
    ],
)
def test_improvement_counter(previous, current, counter, expected) -> None:
    assert count_improvement(previous, current, counter) == expected


def test_partial_coverage_target_can_stop_early() -> None:
    decision = evaluate_stop(
        round_number=4,
        coverage_ratio=0.75,
        rounds_without_improvement=0,
        open_slot_count=4,
        config=StoppingConfig(coverage_target=0.75),
    )
    assert decision.reason is StopReason.COVERAGE_REACHED


def test_no_improvement_is_suppressed_while_slots_are_untried() -> None:
    """"Out of ideas" cannot be true while a slot has never been attempted.

    Without this guard the patience counter races the ontology: with 16 slots and
    a 3-round patience, three unlucky slots at the front would end the run before
    the other thirteen were touched even once.
    """

    decision = evaluate_stop(
        round_number=4,
        coverage_ratio=0.0,
        rounds_without_improvement=3,
        open_slot_count=16,
        untried_slot_count=13,
    )

    assert decision.should_stop is False
    assert decision.reason is StopReason.CONTINUE


def test_no_improvement_fires_once_everything_has_been_tried() -> None:
    decision = evaluate_stop(
        round_number=20,
        coverage_ratio=0.4,
        rounds_without_improvement=3,
        open_slot_count=5,
        untried_slot_count=0,
    )

    assert decision.reason is StopReason.NO_IMPROVEMENT
