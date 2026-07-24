"""Slot selection with failure memory.

V1 deadlocked here. ``pick_next_slot_and_query`` took only the filled set, so a
failed round changed nothing about the next round's input: same slot, a
byte-identical query, the same two pages, the same empty extraction, seven times
running (SESSION_HANDOFF §3.11 constraint 1). Four deterministic stages in a row
with no state carried between them -- a closed loop with no entropy, which no
single component could break.

The fix is to make failure change the input. A slot that just failed loses to any
slot that has not been tried, its already-issued queries are handed to the query
generator as exclusions, and its seen URLs are filtered out of the next search.

Priority still decides among equals; it just no longer outranks the fact that
something is not working.

Query text generation needs a model and therefore lives at the edge: pass a
generator in. Selection itself is pure, so the loop's control logic is testable
without a network.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Protocol

from pydantic import BaseModel, ConfigDict, Field

from open_deep_research.graphrag.ontology import OntologySlot


class SlotAttempt(BaseModel):
    """What has already been tried for one slot."""

    model_config = ConfigDict(extra="forbid")

    attempts: int = 0
    failures: int = 0
    queries: list[str] = Field(default_factory=list)
    seen_urls: list[str] = Field(default_factory=list)


class SupervisorMemory(BaseModel):
    """Per-slot attempt history for one research session.

    Deliberately not in the LangGraph state: it is loop bookkeeping, not
    knowledge, and state holds ids, statuses and scores only (§5).
    """

    model_config = ConfigDict(extra="forbid")

    attempts: dict[str, SlotAttempt] = Field(default_factory=dict)

    def for_slot(self, slot_id: str) -> SlotAttempt:
        return self.attempts.setdefault(slot_id, SlotAttempt())

    def record_attempt(self, slot_id: str, *, query: str, urls: list[str] | None = None) -> None:
        """Note that a query was issued, so the next one has to differ."""

        record = self.for_slot(slot_id)
        record.attempts += 1
        if query and query not in record.queries:
            record.queries.append(query)
        for url in urls or []:
            if url and url not in record.seen_urls:
                record.seen_urls.append(url)

    def record_failure(self, slot_id: str) -> None:
        """Note that an attempt produced nothing usable."""

        self.for_slot(slot_id).failures += 1

    def record_success(self, slot_id: str) -> None:
        """Clear the failure streak; the slot is answerable after all."""

        self.for_slot(slot_id).failures = 0

    def is_exhausted(self, slot_id: str, max_attempts: int) -> bool:
        return self.for_slot(slot_id).attempts >= max_attempts

    def exhausted_count(self, slot_ids: list[str], max_attempts: int) -> int:
        return sum(1 for slot_id in slot_ids if self.is_exhausted(slot_id, max_attempts))


class QueryGenerator(Protocol):
    """Produces a search query for a slot, avoiding previously issued ones."""

    def __call__(
        self, *, topic: str, slot: OntologySlot, previous_queries: list[str]
    ) -> Awaitable[str]: ...


def select_next_slot(
    open_slots: list[OntologySlot],
    memory: SupervisorMemory,
    *,
    max_attempts_per_slot: int = 3,
) -> OntologySlot | None:
    """Pick the slot to investigate next, or None when all are exhausted.

    Ordering, in sequence:

    1. skip slots that have used their attempt budget;
    2. fewest failures first -- an untried slot outranks one that just failed,
       which is the rule that breaks the V1 livelock;
    3. highest ontology priority;
    4. slot id, so the choice is stable across runs.
    """

    candidates = [
        slot
        for slot in open_slots
        if not memory.is_exhausted(slot.slot_id, max_attempts_per_slot)
    ]
    if not candidates:
        return None

    return min(
        candidates,
        key=lambda slot: (
            memory.for_slot(slot.slot_id).failures,
            -slot.priority,
            slot.slot_id,
        ),
    )


async def plan_next_round(
    topic: str,
    open_slots: list[OntologySlot],
    memory: SupervisorMemory,
    generate_query: QueryGenerator | Callable[..., Awaitable[str]],
    *,
    max_attempts_per_slot: int = 3,
) -> tuple[OntologySlot, str, list[str]] | None:
    """Choose the next slot and a query for it.

    Returns the slot, the query, and the URLs already seen for that slot so the
    caller can exclude them. Returns None when every open slot is exhausted --
    a terminal condition the caller should treat as a stop, not an error.
    """

    slot = select_next_slot(
        open_slots, memory, max_attempts_per_slot=max_attempts_per_slot
    )
    if slot is None:
        return None

    record = memory.for_slot(slot.slot_id)
    query = await generate_query(
        topic=topic, slot=slot, previous_queries=list(record.queries)
    )
    query = (query or "").strip() or slot.question

    return slot, query, list(record.seen_urls)
