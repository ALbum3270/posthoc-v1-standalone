"""Does the graph-driven loop converge, and can it still livelock?

The project's one load-bearing assumption is that ontology holes read from the
graph can steer search to convergence (SESSION_HANDOFF §5). Convergence has two
halves, and only one of them needs the network:

* whether real sources answer real questions -- needs a live run;
* whether the control loop makes progress and terminates given some sources
  answer and others never do -- is pure logic, and is what these tests pin.

The second half is where V1 actually failed. It never ran out of rounds because
the questions were unanswerable; it burned seven rounds re-issuing one identical
query while fifteen other slots sat untouched (§3.11).

The harness wires the real supervisor, researcher and stopping modules together.
Only search and extraction are simulated, so the loop logic under test is the
shipped code.
"""

from __future__ import annotations

import asyncio

from graphiti_core.driver.driver import GraphProvider

from open_deep_research.graphrag.control.researcher import run_research_round
from open_deep_research.graphrag.control.stopping import (
    StopReason,
    StoppingConfig,
    count_improvement,
    evaluate_stop,
)
from open_deep_research.graphrag.control.supervisor import (
    SupervisorMemory,
    plan_next_round,
)
from open_deep_research.graphrag.ontology import OntologySlot
from open_deep_research.graphrag.schemas import (
    EntityRef,
    ExtractedTriple,
    SourceDocument,
)

SLOTS = [
    OntologySlot(
        slot_id=f"s{i}.slot", dimension="WHAT", label=f"slot {i}",
        question=f"question {i}?", priority=100 - i,
    )
    for i in range(4)
]


class FakeDriver:
    provider = GraphProvider.NEO4J
    graph_operations_interface = None

    async def execute_query(self, query, **kwargs):
        return [], None, None


class FakeGraphiti:
    def __init__(self) -> None:
        self.driver = FakeDriver()
        self.embedder = self

    async def create_batch(self, texts):
        return [[0.1] * 8 for _ in texts]


class LoopRun:
    """Records what a simulated run did, for assertions."""

    def __init__(self) -> None:
        self.rounds: list[str] = []
        self.queries: list[str] = []
        self.coverage: list[float] = []
        self.decision = None


def simulate(
    *,
    answerable: set[str],
    config: StoppingConfig | None = None,
    hard_round_cap: int = 40,
) -> LoopRun:
    """Run the loop until it stops. Slots outside ``answerable`` never yield facts."""

    settings = config or StoppingConfig(max_rounds=24, max_no_improvement_rounds=3)
    memory = SupervisorMemory()
    filled: set[str] = set()
    run = LoopRun()
    tried: set[str] = set()

    async def generate_query(*, topic, slot, previous_queries):
        # A real generator is asked to vary; simulate that faithfully.
        return f"{slot.slot_id} query #{len(previous_queries) + 1}"

    async def search(*, query, exclude_urls):
        seen = set(exclude_urls)
        return [
            SourceDocument(
                document_id=f"{query}-doc{i}",
                title="doc",
                url=f"https://example.com/{query.replace(' ', '-')}-{i}",
                content="body",
            )
            for i in range(2)
            if f"https://example.com/{query.replace(' ', '-')}-{i}" not in seen
        ]

    async def extract(*, document, slot):
        if slot.slot_id not in answerable:
            return []
        return [
            ExtractedTriple(
                slot_id=slot.slot_id,
                subject=EntityRef(name="FTX"),
                predicate="did",
                object="something",
                confidence=0.8,
                source_document_id=document.document_id,
            )
        ]

    async def loop() -> None:
        graphiti = FakeGraphiti()
        previous_coverage = 0.0
        no_improvement = 0

        for round_number in range(1, hard_round_cap + 1):
            open_slots = [s for s in SLOTS if s.slot_id not in filled]
            coverage = len(filled) / len(SLOTS)
            run.coverage.append(coverage)

            decision = evaluate_stop(
                round_number=round_number,
                coverage_ratio=coverage,
                rounds_without_improvement=no_improvement,
                open_slot_count=len(open_slots),
                exhausted_slot_count=memory.exhausted_count(
                    [s.slot_id for s in open_slots], settings.max_attempts_per_slot
                ),
                untried_slot_count=sum(
                    1 for s in open_slots if s.slot_id not in tried
                ),
                config=settings,
            )
            if decision.should_stop:
                run.decision = decision
                return

            plan = await plan_next_round(
                "FTX", open_slots, memory, generate_query,
                max_attempts_per_slot=settings.max_attempts_per_slot,
            )
            if plan is None:
                run.decision = decision
                return

            slot, query, exclude_urls = plan
            run.rounds.append(slot.slot_id)
            tried.add(slot.slot_id)
            run.queries.append(query)

            result = await run_research_round(
                graphiti,
                topic="FTX",
                research_id="r-1",
                slot=slot,
                query=query,
                search=search,
                extract=extract,
                exclude_urls=exclude_urls,
            )
            memory.record_attempt(
                slot.slot_id, query=query, urls=result.documents_seen
            )
            if result.succeeded:
                filled.add(slot.slot_id)
                memory.record_success(slot.slot_id)
            else:
                memory.record_failure(slot.slot_id)

            no_improvement = count_improvement(
                previous_coverage, len(filled) / len(SLOTS), no_improvement
            )
            previous_coverage = len(filled) / len(SLOTS)

        raise AssertionError("loop did not terminate within the hard cap")

    asyncio.run(loop())
    return run


# --------------------------------------------------------------------------


def test_all_answerable_converges_to_full_coverage() -> None:
    run = simulate(answerable={s.slot_id for s in SLOTS})

    assert run.decision.reason is StopReason.COVERAGE_REACHED
    assert run.decision.is_success is True
    assert run.coverage[-1] == 1.0
    assert len(run.rounds) == len(SLOTS), "no wasted rounds when everything answers"


def test_one_unanswerable_slot_does_not_block_the_others() -> None:
    """The V1 livelock, replayed against the real control modules.

    s0 never yields facts. V1 would have spent every remaining round on it,
    because it was the highest-priority open slot and nothing recorded failure.
    """

    answerable = {s.slot_id for s in SLOTS} - {"s0.slot"}
    run = simulate(answerable=answerable)

    assert run.coverage[-1] == 0.75, "the three answerable slots all got filled"
    # The unanswerable slot is tried, bounded, and abandoned.
    assert run.rounds.count("s0.slot") <= 3
    # Crucially it never pre-empts a slot nobody has tried yet: every slot gets
    # its first attempt before s0 gets its second. V1 failed exactly here.
    assert sorted(run.rounds[:4]) == sorted(s.slot_id for s in SLOTS)


def test_repeated_attempts_never_reuse_a_query() -> None:
    """V1's queries were byte-identical across seven rounds."""

    run = simulate(answerable=set())

    assert len(run.queries) == len(set(run.queries)), "a query was issued twice"


def test_nothing_answerable_terminates_instead_of_spinning() -> None:
    run = simulate(answerable=set())

    assert run.decision.should_stop is True
    assert run.decision.reason in {
        StopReason.ALL_SLOTS_EXHAUSTED,
        StopReason.NO_IMPROVEMENT,
    }
    assert run.decision.is_success is False
    # 4 slots x 3 attempts is the ceiling; V1 would have run to its round cap.
    assert len(run.rounds) <= 12


def test_every_slot_is_tried_before_any_is_retried() -> None:
    run = simulate(answerable=set())

    first_four = run.rounds[:4]
    assert sorted(first_four) == sorted(s.slot_id for s in SLOTS)


def test_coverage_never_regresses() -> None:
    run = simulate(answerable={"s0.slot", "s2.slot"})

    assert run.coverage == sorted(run.coverage)


def test_attempt_budget_is_enforced_per_slot() -> None:
    run = simulate(
        answerable=set(),
        config=StoppingConfig(max_attempts_per_slot=2, max_no_improvement_rounds=99),
    )

    for slot in SLOTS:
        assert run.rounds.count(slot.slot_id) <= 2
    assert run.decision.reason is StopReason.ALL_SLOTS_EXHAUSTED
