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
    )


class RunCostAdmissionDenied(RuntimeError):
    """Raised before a model call that cannot fit the run allowance."""


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
            raise RunCostAdmissionDenied(reason)
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
        return RunCostBudgetAudit(
            configured=limit is not None,
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
