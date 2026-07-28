"""Run-level cost admission with explicit future-stage reserves."""

from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunCostBudget(BaseModel):
    """A run-wide admission ceiling, not a provider-side billing cap.

    Providers report actual cost only after a call. The controller therefore
    rejects calls whose calibrated estimate exceeds the remaining allowance,
    then records actual usage. One admitted call can still overshoot; that
    limitation is exposed in every audit.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_cost_usd: float | None = Field(default=None, ge=0.0)
    verification_reserve_usd: float = Field(default=0.0, ge=0.0)
    cost_objective_usd: float | None = Field(default=None, ge=0.0)
    """A product target, deliberately not a control-flow input.

    This never blocks a call and never becomes a stop reason. It exists so an
    audit can distinguish "this run cost $0.42" from "this run cost $0.42
    against a target of $0.30", which is otherwise unrecoverable after the
    fact. There is no default: no measurement so far establishes that any
    particular figure is a defensible target, and inventing one would give a
    guess the authority of a setting.
    """

    @model_validator(mode="after")
    def _reserve_requires_and_fits_limit(self) -> RunCostBudget:
        if self.max_cost_usd is None:
            if self.verification_reserve_usd:
                raise ValueError(
                    "verification_reserve_usd requires max_cost_usd"
                )
        elif self.verification_reserve_usd > self.max_cost_usd:
            raise ValueError(
                "verification_reserve_usd must not exceed max_cost_usd"
            )
        return self


class RunCostAdmissionRecord(BaseModel):
    """One attempted model-call admission and its measured outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_number: int = Field(ge=1)
    stage: str = Field(min_length=1)
    prompt_chars: int = Field(ge=0)
    protected_reserve_usd: float = Field(ge=0.0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    estimator_available: bool
    admitted: bool
    actual_cost_usd: float = Field(default=0.0, ge=0.0)
    reason: str


class RunCostBudgetAudit(BaseModel):
    """Reader-facing truth about configured and observed run cost control."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    configured: bool
    max_cost_usd: float | None = Field(default=None, ge=0.0)
    verification_reserve_usd: float = Field(ge=0.0)
    enforcement: str
    is_exact_provider_billing_cap: bool = False
    observed_total_cost_usd: float = Field(ge=0.0)
    remaining_cost_usd: float | None = None
    observed_overshoot_usd: float = Field(ge=0.0)
    cost_objective_usd: float | None = None
    cost_objective_exceeded: bool = False
    cap_was_binding: bool = False
    """Whether a ceiling refused a call that would otherwise have been issued.

    This answers only that. It says nothing about whether the refused call was
    worth making -- a tenth reread of a dead URL still sets it true. Judging
    the blocked call is a separate field elsewhere, kept separate so that a
    record of fact never quietly doubles as a verdict on call quality.
    """
    admitted_call_count: int = Field(ge=0)
    rejected_call_count: int = Field(ge=0)
    unestimated_admitted_call_count: int = Field(ge=0)
    stage_cost_usd: dict[str, float] = Field(default_factory=dict)
    admissions: tuple[RunCostAdmissionRecord, ...] = ()
    limitations: tuple[str, ...] = (
        "provider cost is known only after a model call completes",
        "an admitted call can exceed its estimate and overshoot the ceiling",
        "an unavailable role-specific estimate permits one bootstrap call",
        "the ceiling excludes provider charges not present in usage envelopes",
        "the configured ceiling is an absolute run cost cap, not a "
        "runaway-only guard: it sits close enough to normal run cost that it "
        "can and does interrupt legitimate work",
        "cost_objective_usd is advisory; exceeding it changes nothing at run "
        "time and is recorded only for later reading",
    )


class RunCostAdmissionDenied(RuntimeError):
    """Raised before a model call that cannot fit the run allowance."""


class RunCostCapReached(RunCostAdmissionDenied):
    """A run that stopped at its cost cap before it could produce a bundle.

    Carries the diagnosis rather than only the message. A bare admission
    failure tells an operator that something ran out of money but not which
    ceiling bound, what had already been paid for, or how far the run got --
    which is exactly what they need in order to decide whether to raise the
    ceiling or fix something first.

    This is deliberately not degraded into an empty result. Several downstream
    registries (block coverage, checklist coverage) require explicit counts,
    and synthesising zeros for them would make the audit claim it assessed
    everything it never looked at.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        audit: RunCostBudgetAudit,
        completed_stages: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.audit = audit
        self.completed_stages = completed_stages

    def report(self) -> str:
        """Render the operator-facing diagnosis of where the money went."""

        lines = [
            f"run stopped at its cost cap during stage {self.stage!r}",
            (
                f"  spent ${self.audit.observed_total_cost_usd:.4f} of "
                f"${self.audit.max_cost_usd:.4f}"
                if self.audit.max_cost_usd is not None
                else f"  spent ${self.audit.observed_total_cost_usd:.4f}"
            ),
        ]
        if self.audit.verification_reserve_usd:
            lines.append(
                f"  verification reserve held back: "
                f"${self.audit.verification_reserve_usd:.4f}"
            )
        for stage, cost in sorted(
            self.audit.stage_cost_usd.items(), key=lambda item: -item[1]
        ):
            lines.append(f"  {stage}: ${cost:.4f}")
        if self.completed_stages:
            lines.append(
                "  stages that had already been paid for: "
                + ", ".join(self.completed_stages)
            )
        lines.append(
            "  no artifact bundle was written: the remaining stages populate "
            "coverage registries that cannot be filled with zeros without "
            "claiming work that never happened"
        )
        return "\n".join(lines)


def _usage_cost(response: Any, model_client: Any) -> float:
    raw: Any
    if isinstance(response, Mapping):
        raw = response.get("cost_usd")
        if raw is not None:
            return max(0.0, float(raw))
    raw = getattr(model_client, "last_usage", None)
    if callable(raw):
        raw = raw()
    if isinstance(raw, Mapping):
        return max(0.0, float(raw.get("cost_usd", 0.0) or 0.0))
    return max(0.0, float(getattr(raw, "cost_usd", 0.0) or 0.0))


class RunCostController:
    """Share one observed-cost ledger across every model role in a run."""

    def __init__(self, budget: RunCostBudget | None = None) -> None:
        self.budget = budget or RunCostBudget()
        self._observed_cost = 0.0
        self._stage_cost: dict[str, float] = defaultdict(float)
        self._admissions: list[RunCostAdmissionRecord] = []

    @property
    def configured(self) -> bool:
        return self.budget.max_cost_usd is not None

    @property
    def observed_cost_usd(self) -> float:
        return self._observed_cost

    @property
    def remaining_cost_usd(self) -> float | None:
        if self.budget.max_cost_usd is None:
            return None
        return max(0.0, self.budget.max_cost_usd - self._observed_cost)

    def available_before_reserve(self, reserve_usd: float) -> float | None:
        remaining = self.remaining_cost_usd
        if remaining is None:
            return None
        return max(0.0, remaining - reserve_usd)

    def record_external_usage(self, stage: str, cost_usd: float) -> None:
        """Record measured work governed by an existing stage-local budget."""

        measured = max(0.0, float(cost_usd))
        self._observed_cost += measured
        self._stage_cost[stage] += measured

    def wrap(
        self,
        model_client: Any,
        *,
        stage: str,
        protected_reserve_usd: float = 0.0,
    ) -> _BudgetedModelClient:
        return _BudgetedModelClient(
            model_client,
            controller=self,
            stage=stage,
            protected_reserve_usd=protected_reserve_usd,
        )

    def _estimate(self, model_client: Any, prompt: str) -> float | None:
        estimator = getattr(model_client, "estimate_cost_usd", None)
        if not callable(estimator):
            return None
        try:
            return max(0.0, float(estimator(prompt)))
        except (RuntimeError, TypeError, ValueError):
            return None

    def _admit(
        self,
        *,
        model_client: Any,
        stage: str,
        prompt: str,
        protected_reserve_usd: float,
    ) -> tuple[int, float | None]:
        estimate = self._estimate(model_client, prompt)
        call_number = len(self._admissions) + 1
        allowed = self.available_before_reserve(protected_reserve_usd)
        reason = "run cost limit is not configured"
        admitted = True
        if allowed is not None:
            if allowed <= 0.0:
                admitted = False
                reason = "no run cost allowance remains before reserve"
            elif estimate is not None and estimate > allowed:
                admitted = False
                reason = (
                    f"estimated call cost {estimate:.8f} exceeds available "
                    f"{allowed:.8f} before reserve"
                )
            elif estimate is None:
                reason = (
                    "admitted as the role estimator has no observation yet"
                )
            else:
                reason = "estimated call fits the run cost allowance"
        self._admissions.append(
            RunCostAdmissionRecord(
                call_number=call_number,
                stage=stage,
                prompt_chars=len(prompt),
                protected_reserve_usd=protected_reserve_usd,
                estimated_cost_usd=estimate,
                estimator_available=estimate is not None,
                admitted=admitted,
                reason=reason,
            )
        )
        if not admitted:
            # Raise the diagnosing subclass so that whatever catches this --
            # including a bare top-level handler -- has the boundary, the
            # per-stage spend, and how far the run got, rather than one line
            # saying money ran out.
            raise RunCostCapReached(
                reason,
                stage=stage,
                audit=self.audit(),
                completed_stages=tuple(
                    name
                    for name, cost in sorted(self._stage_cost.items())
                    if cost > 0.0
                ),
            )
        return call_number, estimate

    def _complete(self, call_number: int, stage: str, cost_usd: float) -> None:
        measured = max(0.0, float(cost_usd))
        self._observed_cost += measured
        self._stage_cost[stage] += measured
        index = call_number - 1
        self._admissions[index] = self._admissions[index].model_copy(
            update={"actual_cost_usd": measured}
        )

    def audit(self) -> RunCostBudgetAudit:
        limit = self.budget.max_cost_usd
        overshoot = (
            max(0.0, self._observed_cost - limit)
            if limit is not None
            else 0.0
        )
        objective = self.budget.cost_objective_usd
        return RunCostBudgetAudit(
            configured=limit is not None,
            cost_objective_usd=objective,
            cost_objective_exceeded=(
                objective is not None and self._observed_cost > objective
            ),
            cap_was_binding=any(
                not record.admitted for record in self._admissions
            ),
            max_cost_usd=limit,
            verification_reserve_usd=(
                self.budget.verification_reserve_usd
            ),
            enforcement=(
                "pre_call_estimate_admission_plus_observed_usage"
                if limit is not None
                else "no_run_level_cost_limit"
            ),
            observed_total_cost_usd=self._observed_cost,
            remaining_cost_usd=self.remaining_cost_usd,
            observed_overshoot_usd=overshoot,
            admitted_call_count=sum(
                record.admitted for record in self._admissions
            ),
            rejected_call_count=sum(
                not record.admitted for record in self._admissions
            ),
            unestimated_admitted_call_count=sum(
                record.admitted and not record.estimator_available
                for record in self._admissions
            ),
            stage_cost_usd={
                stage: cost
                for stage, cost in sorted(self._stage_cost.items())
            },
            admissions=tuple(self._admissions),
        )


class _BudgetedModelClient:
    """Protocol-transparent model proxy governed by a RunCostController."""

    def __init__(
        self,
        model_client: Any,
        *,
        controller: RunCostController,
        stage: str,
        protected_reserve_usd: float,
    ) -> None:
        self._model_client = model_client
        self._controller = controller
        self._stage = stage
        self._protected_reserve_usd = protected_reserve_usd
        self.last_usage: Any = {"token_count": 0, "cost_usd": 0.0}

    async def generate(self, prompt: str) -> Any:
        call_number, _ = self._controller._admit(
            model_client=self._model_client,
            stage=self._stage,
            prompt=prompt,
            protected_reserve_usd=self._protected_reserve_usd,
        )
        response = self._model_client.generate(prompt)
        if inspect.isawaitable(response):
            response = await response
        cost = _usage_cost(response, self._model_client)
        raw_usage = getattr(self._model_client, "last_usage", None)
        if callable(raw_usage):
            raw_usage = raw_usage()
        if raw_usage is not None:
            self.last_usage = raw_usage
        elif isinstance(response, Mapping):
            self.last_usage = {
                "token_count": response.get("token_count", 0),
                "cost_usd": response.get("cost_usd", 0.0),
            }
        self._controller._complete(call_number, self._stage, cost)
        return response

    def estimate_cost_usd(self, prompt: str) -> float:
        estimate = self._controller._estimate(self._model_client, prompt)
        return estimate if estimate is not None else 0.0

    def estimate_tokens(self, prompt: str) -> int:
        estimator = getattr(self._model_client, "estimate_tokens", None)
        if not callable(estimator):
            return 0
        try:
            return max(0, int(estimator(prompt)))
        except (RuntimeError, TypeError, ValueError):
            return 0
