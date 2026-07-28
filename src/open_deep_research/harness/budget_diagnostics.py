"""Run-level stop diagnosis: why work stopped, and whether money would help.

A single ``budget_exhausted`` state cannot distinguish "a runaway was caught"
from "ordinary work was cut off". Those call for opposite responses -- fix the
mechanism versus raise the ceiling -- so this module keeps three things apart
that used to be fused:

* the **resource reason** work stopped (a mechanical fact),
* the **completion status** of the work itself (an independent fact), and
* a **decision signal** about whether more budget would plausibly buy anything.

The signal is deliberately not a score. A "budget efficiency" number would
immediately become something to optimise, and optimising it would mean
selecting for runs that look thrifty rather than runs that gathered evidence.
It is an enum with an explicit ``indeterminate`` value, and the evidence behind
it is reported alongside so a reader can disagree with it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# How many recent acquisition rounds the decision signal looks at. Kept small
# because the question is "was the run still productive when it was cut off",
# not "was the run productive overall" -- a run can be productive for twenty
# rounds and then spin uselessly for the last five, and it is the last five
# that determine whether more budget buys anything.
RECENT_ROUND_WINDOW = 5

# Actions that attempt to acquire new external material. Only these can be
# judged productive or wasteful; bookkeeping actions are excluded rather than
# counted as zero-output, which would bias the signal toward "fix mechanism".
ACQUISITION_ACTIONS = frozenset({"search", "read"})


class ResourceStopReason(str, Enum):
    """Which ceiling or guard stopped the run. Orthogonal to completion."""

    NOT_RESOURCE_LIMITED = "not_resource_limited"
    RUN_COST_CAP_REACHED = "run_cost_cap_reached"
    COLLECTION_COST_LIMIT_REACHED = "collection_cost_limit_reached"
    COLLECTION_TOKEN_LIMIT_REACHED = "collection_token_limit_reached"
    COLLECTION_ROUND_LIMIT_REACHED = "collection_round_limit_reached"
    POSTHOC_RETRIEVAL_LIMIT_REACHED = "posthoc_retrieval_limit_reached"
    VERIFICATION_ADMISSION_DENIED = "verification_admission_denied"
    LOOP_GUARD_TRIGGERED = "loop_guard_triggered"


class CompletionStatus(str, Enum):
    """How much of the protocol the run actually finished.

    Recorded independently of :class:`ResourceStopReason` so that a run can
    honestly report "stopped at the absolute cost cap; report written, but 18
    of 90 claims never verified" instead of one unexplained word.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_STARTED = "not_started"
    FAILED = "failed"


class BudgetDecisionSignal(str, Enum):
    """Whether raising the ceiling is plausibly worth money.

    ``INDETERMINATE`` is a first-class outcome, not a failure to decide. Mixed
    evidence is the common case, and collapsing it into a recommendation would
    manufacture confidence the evidence does not carry.
    """

    MORE_BUDGET_MAY_HELP = "more_budget_may_help"
    FIX_MECHANISM_FIRST = "fix_mechanism_first"
    INDETERMINATE = "indeterminate"
    NOT_APPLICABLE = "not_applicable"


class BlockedOperationQuality(str, Enum):
    """What the ceiling actually stopped, judged separately from whether it did.

    ``cap_was_binding`` answers only "did the ceiling reject a call that would
    otherwise have been issued". It does not evaluate that call. A tenth reread
    of a dead URL is still binding; this enum is where the fact that it was
    waste gets recorded, so that one field never silently stands in for the
    other.
    """

    NOTHING_BLOCKED = "nothing_blocked"
    UNKNOWN = "unknown"
    KNOWN_INVALID_OR_REPEATED = "known_invalid_or_repeated"
    APPARENTLY_PRODUCTIVE = "apparently_productive"


class StopBoundary(BaseModel):
    """The exact ceiling that was hit, with the numbers that prove it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope: str = Field(min_length=1)
    resource: str = Field(min_length=1)
    used: float
    limit: float

    @property
    def headroom(self) -> float:
        return self.limit - self.used


class BlockedCall(BaseModel):
    """The next call the admission layer refused, and what it would have cost."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str = Field(min_length=1)
    prompt_chars: int = Field(ge=0)
    estimated_cost_usd: float | None = None
    estimator_available: bool = False
    available_before_reserve_usd: float | None = None
    reason: str = ""


class OutstandingWork(BaseModel):
    """What was still owed when the run stopped.

    Every field is a count of something a reader could go and finish; together
    they answer "is there anything left for more budget to buy at all".
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    open_checklist_items: int = Field(default=0, ge=0)
    unread_candidates: int = Field(default=0, ge=0)
    unattributed_claims: int = Field(default=0, ge=0)
    unverified_relations: int = Field(default=0, ge=0)
    evidence_gap_plan_unexecuted: bool = False
    disagreement_plan_unexecuted: bool = False

    @property
    def has_outstanding_work(self) -> bool:
        return bool(
            self.open_checklist_items
            or self.unread_candidates
            or self.unattributed_claims
            or self.unverified_relations
            or self.evidence_gap_plan_unexecuted
            or self.disagreement_plan_unexecuted
        )


class RoundProductivity(BaseModel):
    """Mechanical output of one acquisition round, with no quality judgement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    round_number: int = Field(ge=1)
    action: str = Field(min_length=1)
    new_sources: int = Field(default=0, ge=0)
    notes_created: int = Field(default=0, ge=0)
    repeated_action: bool = False
    tool_error: bool = False

    @property
    def produced_material(self) -> bool:
        return bool(self.new_sources or self.notes_created)

    @property
    def was_wasteful(self) -> bool:
        """Repeat, error, or zero output -- the shapes budget cannot fix."""

        return bool(
            self.repeated_action or self.tool_error or not self.produced_material
        )


class RunStopDiagnostic(BaseModel):
    """Everything needed to answer "should I add money to this run?"."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resource_stop_reason: ResourceStopReason
    completion_status: CompletionStatus
    boundary: StopBoundary | None = None
    blocked_call: BlockedCall | None = None
    cap_was_binding: bool = False
    blocked_operation_quality: BlockedOperationQuality = (
        BlockedOperationQuality.NOTHING_BLOCKED
    )
    stage_cost_usd: dict[str, float] = Field(default_factory=dict)
    observed_total_cost_usd: float = Field(default=0.0, ge=0.0)
    absolute_run_cost_cap_usd: float | None = None
    cost_objective_usd: float | None = None
    cost_objective_exceeded: bool = False
    verification_reserve_usd: float = Field(default=0.0, ge=0.0)
    outstanding: OutstandingWork = Field(default_factory=OutstandingWork)
    recent_rounds: tuple[RoundProductivity, ...] = ()
    budget_decision_signal: BudgetDecisionSignal = (
        BudgetDecisionSignal.NOT_APPLICABLE
    )
    signal_evidence: tuple[str, ...] = ()
    limitations: tuple[str, ...] = (
        "the decision signal is an enum, not a score; it is deliberately not "
        "combined into a single budget-efficiency number, because such a "
        "number would become a target and reward thrift over evidence",
        "cap_was_binding reports only that a call was refused, never whether "
        "that call was worth making; blocked_operation_quality carries that",
        "recent-round productivity is measured over the last "
        f"{RECENT_ROUND_WINDOW} acquisition rounds, so a run that stalls "
        "briefly and then recovers can be read as unproductive",
        "cost_objective_usd never blocks work; exceeding it is an audit event",
    )


def _parse_summary(raw: str) -> Mapping[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def summarise_round_productivity(
    rounds: Sequence[Any],
    *,
    window: int = RECENT_ROUND_WINDOW,
) -> tuple[RoundProductivity, ...]:
    """Read mechanical output off the ledger's own round records.

    Nothing here consults the model or re-judges relevance: a round produced
    material or it did not, and it repeated an earlier query/URL or it did not.
    """

    seen_targets: set[tuple[str, str]] = set()
    seen_urls: set[str] = set()
    productivity: list[RoundProductivity] = []
    for record in rounds:
        action = getattr(record, "action", "") or ""
        if action not in ACQUISITION_ACTIONS:
            continue
        summary = _parse_summary(getattr(record, "result_summary", "") or "")
        query = getattr(record, "query", None)
        url = getattr(record, "url", None)
        target = (action, str(query or url or ""))
        repeated = target in seen_targets
        seen_targets.add(target)

        new_sources = 0
        if action == "read" and url:
            if url not in seen_urls:
                seen_urls.add(url)
                new_sources = 1
        elif action == "search":
            candidates = summary.get("candidate_urls")
            if isinstance(candidates, list):
                fresh = [
                    str(item)
                    for item in candidates
                    if str(item) and str(item) not in seen_urls
                ]
                new_sources = len(fresh)

        notes_created = summary.get("notes_created", 0)
        notes_created = (
            int(notes_created) if isinstance(notes_created, (int, float)) else 0
        )
        tool_error = bool(summary.get("error"))

        productivity.append(
            RoundProductivity(
                round_number=int(getattr(record, "round_number", 1) or 1),
                action=action,
                new_sources=max(0, new_sources),
                notes_created=max(0, notes_created),
                repeated_action=repeated,
                tool_error=tool_error,
            )
        )
    return tuple(productivity[-window:]) if window > 0 else tuple(productivity)


def judge_budget_decision(
    *,
    cap_was_binding: bool,
    outstanding: OutstandingWork,
    recent_rounds: Iterable[RoundProductivity],
) -> tuple[BudgetDecisionSignal, tuple[str, ...]]:
    """Decide whether more budget is plausibly worth buying.

    The rule is mechanical and states its evidence. It refuses to recommend
    spending on mixed evidence, because the failure mode we are guarding
    against -- topping up a run that was spinning rather than working -- looks
    exactly like mixed evidence from the inside.
    """

    rounds = tuple(recent_rounds)
    evidence: list[str] = []

    if not cap_was_binding:
        return (
            BudgetDecisionSignal.NOT_APPLICABLE,
            ("no ceiling refused a call, so no work was withheld for cost",),
        )

    if not outstanding.has_outstanding_work:
        return (
            BudgetDecisionSignal.NOT_APPLICABLE,
            (
                "a ceiling was binding but no outstanding work remained, so "
                "more budget has nothing to buy",
            ),
        )

    if not rounds:
        return (
            BudgetDecisionSignal.INDETERMINATE,
            ("no acquisition rounds ran, so productivity is unmeasured",),
        )

    productive = [item for item in rounds if item.produced_material]
    wasteful = [item for item in rounds if item.was_wasteful]
    evidence.append(
        f"{len(productive)} of {len(rounds)} recent acquisition rounds "
        "produced new sources or notes"
    )
    if wasteful:
        evidence.append(
            f"{len(wasteful)} of {len(rounds)} were repeats, errors, or "
            "produced nothing"
        )

    if not productive:
        evidence.append(
            "no recent round produced material; more budget would repeat the "
            "same unproductive calls"
        )
        return BudgetDecisionSignal.FIX_MECHANISM_FIRST, tuple(evidence)

    if not wasteful:
        evidence.append(
            "every recent round produced material and work remains outstanding"
        )
        return BudgetDecisionSignal.MORE_BUDGET_MAY_HELP, tuple(evidence)

    evidence.append(
        "recent rounds mix production and waste; this is not enough to "
        "recommend spending, and no recommendation is made"
    )
    return BudgetDecisionSignal.INDETERMINATE, tuple(evidence)


def classify_blocked_operation(
    recent_rounds: Sequence[RoundProductivity],
    *,
    cap_was_binding: bool,
) -> BlockedOperationQuality:
    """Describe what the ceiling stopped, without changing whether it bound."""

    if not cap_was_binding:
        return BlockedOperationQuality.NOTHING_BLOCKED
    if not recent_rounds:
        return BlockedOperationQuality.UNKNOWN
    last = recent_rounds[-1]
    if last.repeated_action or last.tool_error:
        return BlockedOperationQuality.KNOWN_INVALID_OR_REPEATED
    if last.produced_material:
        return BlockedOperationQuality.APPARENTLY_PRODUCTIVE
    return BlockedOperationQuality.UNKNOWN


def _collection_stop_reason(loop_stop_reason: Any) -> ResourceStopReason:
    """Map the collection loop's own vocabulary onto the run-level one."""

    value = getattr(loop_stop_reason, "value", loop_stop_reason)
    mapping = {
        "collection_round_limit_reached": (
            ResourceStopReason.COLLECTION_ROUND_LIMIT_REACHED
        ),
        "collection_token_limit_reached": (
            ResourceStopReason.COLLECTION_TOKEN_LIMIT_REACHED
        ),
        "collection_cost_limit_reached": (
            ResourceStopReason.COLLECTION_COST_LIMIT_REACHED
        ),
        "malformed_action_limit": ResourceStopReason.LOOP_GUARD_TRIGGERED,
    }
    return mapping.get(str(value), ResourceStopReason.NOT_RESOURCE_LIMITED)


def build_run_stop_diagnostic(
    *,
    loop_result: Any,
    run_cost_audit: Any,
    unattributed_claims: int = 0,
    unverified_relations: int = 0,
    evidence_gap_plan_unexecuted: bool = False,
    disagreement_plan_unexecuted: bool = False,
    posthoc_retrieval_limited: bool = False,
    report_written: bool = True,
) -> RunStopDiagnostic:
    """Assemble the stop diagnosis from records that already exist.

    Nothing here re-runs work or asks a model anything; every field is read off
    the loop result, the cost ledger, and the verification registry.
    """

    cap_was_binding = bool(getattr(run_cost_audit, "cap_was_binding", False))
    recent = summarise_round_productivity(
        getattr(getattr(loop_result, "ledger", None), "rounds", []) or []
    )

    resource_stop = _collection_stop_reason(
        getattr(loop_result, "stop_reason", None)
    )
    # A run-level refusal outranks a collection-local ceiling in the report,
    # because it is the one a reader can act on by changing a single number.
    if cap_was_binding:
        resource_stop = ResourceStopReason.RUN_COST_CAP_REACHED
    elif posthoc_retrieval_limited and (
        resource_stop is ResourceStopReason.NOT_RESOURCE_LIMITED
    ):
        resource_stop = ResourceStopReason.POSTHOC_RETRIEVAL_LIMIT_REACHED

    boundary: StopBoundary | None = None
    limit_resource = getattr(loop_result, "limit_resource", None)
    if cap_was_binding and getattr(run_cost_audit, "max_cost_usd", None):
        boundary = StopBoundary(
            scope="run",
            resource="cost_usd",
            used=float(run_cost_audit.observed_total_cost_usd),
            limit=float(run_cost_audit.max_cost_usd),
        )
    elif limit_resource:
        boundary = StopBoundary(
            scope="collection",
            resource=str(limit_resource),
            used=float(getattr(loop_result, "limit_used", 0.0) or 0.0),
            limit=float(getattr(loop_result, "limit_value", 0.0) or 0.0),
        )

    blocked_call: BlockedCall | None = None
    for record in getattr(run_cost_audit, "admissions", ()) or ():
        if not getattr(record, "admitted", True):
            blocked_call = BlockedCall(
                stage=record.stage,
                prompt_chars=record.prompt_chars,
                estimated_cost_usd=record.estimated_cost_usd,
                estimator_available=record.estimator_available,
                available_before_reserve_usd=None,
                reason=record.reason,
            )
            break

    outstanding = OutstandingWork(
        open_checklist_items=len(getattr(loop_result, "open_item_ids", ()) or ()),
        unread_candidates=0,
        unattributed_claims=max(0, int(unattributed_claims)),
        unverified_relations=max(0, int(unverified_relations)),
        evidence_gap_plan_unexecuted=bool(evidence_gap_plan_unexecuted),
        disagreement_plan_unexecuted=bool(disagreement_plan_unexecuted),
    )

    if not report_written:
        completion = CompletionStatus.FAILED
    elif not getattr(loop_result, "rounds_executed", 0):
        completion = CompletionStatus.NOT_STARTED
    elif getattr(loop_result, "is_success", False) and not (
        outstanding.has_outstanding_work
    ):
        completion = CompletionStatus.COMPLETE
    else:
        completion = CompletionStatus.PARTIAL

    signal, evidence = judge_budget_decision(
        cap_was_binding=cap_was_binding,
        outstanding=outstanding,
        recent_rounds=recent,
    )

    return RunStopDiagnostic(
        resource_stop_reason=resource_stop,
        completion_status=completion,
        boundary=boundary,
        blocked_call=blocked_call,
        cap_was_binding=cap_was_binding,
        blocked_operation_quality=classify_blocked_operation(
            recent, cap_was_binding=cap_was_binding
        ),
        stage_cost_usd=dict(getattr(run_cost_audit, "stage_cost_usd", {}) or {}),
        observed_total_cost_usd=float(
            getattr(run_cost_audit, "observed_total_cost_usd", 0.0) or 0.0
        ),
        absolute_run_cost_cap_usd=getattr(run_cost_audit, "max_cost_usd", None),
        cost_objective_usd=getattr(run_cost_audit, "cost_objective_usd", None),
        cost_objective_exceeded=bool(
            getattr(run_cost_audit, "cost_objective_exceeded", False)
        ),
        verification_reserve_usd=float(
            getattr(run_cost_audit, "verification_reserve_usd", 0.0) or 0.0
        ),
        outstanding=outstanding,
        recent_rounds=recent,
        budget_decision_signal=signal,
        signal_evidence=evidence,
    )
