from __future__ import annotations

import asyncio

import pytest

from open_deep_research.harness.budget import (
    RunCostAdmissionDenied,
    RunCostBudget,
    RunCostController,
)


class EstimatedEnvelopeModel:
    def __init__(self, *, estimate: float, actual: float) -> None:
        self.estimate = estimate
        self.actual = actual
        self.calls = 0

    def estimate_cost_usd(self, prompt: str) -> float:
        return self.estimate

    async def generate(self, prompt: str):
        self.calls += 1
        return {
            "content": {"ok": True},
            "token_count": 1,
            "cost_usd": self.actual,
        }


class UnestimatedEnvelopeModel:
    async def generate(self, prompt: str):
        return {
            "content": {"ok": True},
            "token_count": 1,
            "cost_usd": 0.03,
        }


def test_verification_reserve_mechanically_rejects_an_early_call() -> None:
    controller = RunCostController(
        RunCostBudget(
            max_cost_usd=0.30,
            verification_reserve_usd=0.10,
        )
    )
    model = EstimatedEnvelopeModel(estimate=0.21, actual=0.15)
    early = controller.wrap(
        model,
        stage="decomposition_attribution",
        protected_reserve_usd=0.10,
    )

    with pytest.raises(RunCostAdmissionDenied, match="exceeds available"):
        asyncio.run(early.generate("large structured prompt"))

    assert model.calls == 0
    audit = controller.audit()
    assert audit.rejected_call_count == 1
    assert audit.admitted_call_count == 0
    assert audit.admissions[0].protected_reserve_usd == 0.10
    assert audit.observed_total_cost_usd == 0.0

    verification = controller.wrap(model, stage="verification")
    asyncio.run(verification.generate("same estimated call"))
    assert model.calls == 1
    assert controller.audit().stage_cost_usd == {"verification": 0.15}


def test_admitted_single_call_overshoot_is_exposed_and_stops_future_calls() -> None:
    controller = RunCostController(RunCostBudget(max_cost_usd=0.10))
    first_model = EstimatedEnvelopeModel(estimate=0.09, actual=0.12)
    first = controller.wrap(first_model, stage="writing")

    asyncio.run(first.generate("prompt"))

    audit = controller.audit()
    assert audit.observed_total_cost_usd == 0.12
    assert audit.observed_overshoot_usd == pytest.approx(0.02)
    assert audit.is_exact_provider_billing_cap is False
    second_model = EstimatedEnvelopeModel(estimate=0.001, actual=0.001)
    second = controller.wrap(second_model, stage="verification")
    with pytest.raises(RunCostAdmissionDenied, match="no run cost allowance"):
        asyncio.run(second.generate("another prompt"))
    assert second_model.calls == 0


def test_unestimated_bootstrap_call_is_counted_not_hidden() -> None:
    controller = RunCostController(RunCostBudget(max_cost_usd=0.10))
    wrapped = controller.wrap(
        UnestimatedEnvelopeModel(),
        stage="checklist",
    )

    asyncio.run(wrapped.generate("first role-specific prompt"))

    audit = controller.audit()
    assert audit.unestimated_admitted_call_count == 1
    assert audit.admissions[0].estimator_available is False
    assert "no observation yet" in audit.admissions[0].reason


def test_absent_run_limit_is_explicit_in_audit() -> None:
    controller = RunCostController()
    controller.record_external_usage("collection", 0.08)

    audit = controller.audit()

    assert audit.configured is False
    assert audit.max_cost_usd is None
    assert audit.enforcement == "no_run_level_cost_limit"
    assert audit.remaining_cost_usd is None
    assert audit.observed_total_cost_usd == 0.08
