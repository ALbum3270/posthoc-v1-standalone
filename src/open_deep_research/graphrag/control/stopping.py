"""Deterministic stopping rules for the research loop.

Every input here is a number already known from graph state. No model is called
(§2.7): LLM answer-quality scoring may be recorded for analysis, but it does not
participate in control flow. A generated score that decides when to stop is a
threshold needing per-run calibration, and it interacts with the verification
thresholds -- two coupled knobs, neither observable in isolation.

The rules exist to terminate; they are not a quality judgement. "Coverage reached
target" and "nothing left worth trying" are both legitimate stops, and the
distinction is preserved in the decision so callers can tell a finished
investigation from an exhausted one.

V1 had none of this and simply ran to its round cap, spending seven of ten rounds
re-issuing an identical failing query (§3.11).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class StopReason(str, Enum):
    """Why the loop stopped, or that it did not."""

    CONTINUE = "continue"
    COVERAGE_REACHED = "coverage_reached"
    NO_IMPROVEMENT = "no_improvement"
    MAX_ROUNDS = "max_rounds"
    NO_SLOTS_LEFT = "no_slots_left"
    ALL_SLOTS_EXHAUSTED = "all_slots_exhausted"


class StoppingConfig(BaseModel):
    """Thresholds for the deterministic stopping rules."""

    model_config = ConfigDict(extra="forbid")

    coverage_target: float = Field(default=1.0, ge=0.0, le=1.0)
    max_rounds: int = Field(default=24, ge=1)
    max_no_improvement_rounds: int = Field(default=3, ge=1)
    # A slot is abandoned after this many failed attempts. Bounded retry is what
    # separates "this source did not have it" from an unbreakable loop.
    max_attempts_per_slot: int = Field(default=3, ge=1)


class StopDecision(BaseModel):
    """Outcome of one stopping evaluation."""

    model_config = ConfigDict(extra="forbid")

    should_stop: bool
    reason: StopReason
    detail: str = ""

    @property
    def is_success(self) -> bool:
        """Whether the run ended by answering enough, rather than by giving up."""

        return self.reason is StopReason.COVERAGE_REACHED


def evaluate_stop(
    *,
    round_number: int,
    coverage_ratio: float,
    rounds_without_improvement: int,
    open_slot_count: int,
    exhausted_slot_count: int = 0,
    untried_slot_count: int = 0,
    config: StoppingConfig | None = None,
) -> StopDecision:
    """Decide whether the research loop should stop before the next round.

    ``round_number`` is 1-based and refers to the round about to be run.
    ``exhausted_slot_count`` counts open slots that have used up their attempt
    budget; ``untried_slot_count`` counts open slots never attempted at all.

    Rules are checked most-informative first, so the reported reason is the one
    worth acting on: finishing beats running out of rounds.
    """

    settings = config or StoppingConfig()

    if coverage_ratio >= settings.coverage_target:
        return StopDecision(
            should_stop=True,
            reason=StopReason.COVERAGE_REACHED,
            detail=f"coverage {coverage_ratio:.0%} >= target {settings.coverage_target:.0%}",
        )

    if open_slot_count <= 0:
        return StopDecision(
            should_stop=True,
            reason=StopReason.NO_SLOTS_LEFT,
            detail="no open slots remain",
        )

    if exhausted_slot_count >= open_slot_count:
        return StopDecision(
            should_stop=True,
            reason=StopReason.ALL_SLOTS_EXHAUSTED,
            detail=(
                f"all {open_slot_count} open slot(s) hit the "
                f"{settings.max_attempts_per_slot}-attempt limit"
            ),
        )

    # "No improvement" is meant to mean "we are out of ideas". A slot nobody has
    # attempted is still an idea, so the rule is suppressed while any remain.
    # Without this guard the counter is a race against the ontology: with 16 slots
    # and a 3-round patience, three unlucky slots at the front end the run before
    # the other thirteen are touched even once.
    if (
        rounds_without_improvement >= settings.max_no_improvement_rounds
        and untried_slot_count <= 0
    ):
        return StopDecision(
            should_stop=True,
            reason=StopReason.NO_IMPROVEMENT,
            detail=(
                f"{rounds_without_improvement} round(s) without new coverage "
                f">= {settings.max_no_improvement_rounds}"
            ),
        )

    if round_number > settings.max_rounds:
        return StopDecision(
            should_stop=True,
            reason=StopReason.MAX_ROUNDS,
            detail=f"round {round_number} exceeds cap {settings.max_rounds}",
        )

    return StopDecision(
        should_stop=False,
        reason=StopReason.CONTINUE,
        detail=f"coverage {coverage_ratio:.0%}, {open_slot_count} slot(s) open",
    )


def count_improvement(
    previous_coverage: float,
    current_coverage: float,
    rounds_without_improvement: int,
) -> int:
    """Update the no-improvement counter from a coverage delta.

    Improvement means coverage strictly rose. Anything else -- flat or, if an
    edge expired, lower -- counts as no progress, because both mean the last
    round bought nothing.
    """

    if current_coverage > previous_coverage:
        return 0
    return rounds_without_improvement + 1
