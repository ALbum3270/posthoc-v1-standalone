"""Enhancement passes must never eat the money the evidence tail needs.

Reserving only for verification looked sufficient until it wasn't: a verifier
with money but a registry that was already truncated upstream reports "fully
verified" over a shrunken denominator. So the protected unit is the whole
mandatory tail, and enhancement passes may only spend what is left after it.
"""

import pytest

from open_deep_research.harness.tail_budget import (
    EvidenceTailReserveController,
    TailCheckpointName,
    TailWorkUnit,
)


def controller(initial: float = 0.10) -> EvidenceTailReserveController:
    return EvidenceTailReserveController(initial)


def test_an_enhancement_pass_can_never_touch_the_tail_reserve():
    """evidence_gap and disagreement are extras, not part of the tail."""

    tail = controller(0.10)

    for stage in ("evidence_gap", "disagreement", "evaluative_diagnostics"):
        assert tail.reserve_for_call(stage, 0.02) == pytest.approx(0.10), stage
        # Even a call that would fit must still see the full reserve withheld.
        assert tail.reserve_for_call(stage, None) == pytest.approx(0.10), stage


def test_a_mandatory_stage_may_consume_the_reserve_it_belongs_to():
    """The reserve exists to be spent on exactly this work."""

    tail = controller(0.10)

    protected = tail.reserve_for_call("attribution", 0.04)

    # 0.04 of the 0.10 reserve is released for this call; 0.06 stays protected.
    assert protected == pytest.approx(0.06)


def test_an_uncalibrated_mandatory_call_is_not_starved_by_its_own_reserve():
    """Withholding money from the work it was saved for defeats the reserve.

    A role with no observation yet cannot be estimated. Refusing the call would
    protect the money and prevent the very stage it was protecting.
    """

    tail = controller(0.10)

    assert tail.reserve_for_call("initial_verification", None) == 0.0


def test_spending_draws_the_reserve_down_only_for_mandatory_stages():
    tail = controller(0.10)

    tail.record_call_cost("attribution", 0.03)
    assert tail.current_reserve_usd == pytest.approx(0.07)

    # An enhancement pass spends real money but never out of the reserve.
    tail.record_call_cost("evidence_gap", 0.05)
    assert tail.current_reserve_usd == pytest.approx(0.07)


def test_the_reserve_never_goes_negative():
    tail = controller(0.02)

    tail.record_call_cost("attribution", 0.09)

    assert tail.current_reserve_usd == 0.0


def test_an_incomplete_estimate_can_raise_the_reserve_but_never_lower_it():
    """A partial estimate is a lower bound and must not be read as the total.

    Before selection runs, only some downstream prompts are constructible. If
    that partial figure replaced a larger standing reserve, the run would free
    money it still needs and discover the shortfall only at the cap.
    """

    tail = controller(0.10)

    tail.checkpoint(
        TailCheckpointName.DRAFT_AVAILABLE,
        work_units=(TailWorkUnit(stage="claim_decomposition", unit="markdown_block", count=16),),
        estimated_remaining_cost_usd=0.04,
        estimate_complete=False,
        limitations=("only selection prompts are constructible",),
    )

    assert tail.current_reserve_usd == pytest.approx(0.10)

    # A larger lower bound does raise it.
    tail.checkpoint(
        TailCheckpointName.DRAFT_AVAILABLE,
        work_units=(),
        estimated_remaining_cost_usd=0.15,
        estimate_complete=False,
    )
    assert tail.current_reserve_usd == pytest.approx(0.15)


def test_a_complete_estimate_replaces_the_reserve_in_both_directions():
    tail = controller(0.10)

    tail.checkpoint(
        TailCheckpointName.DRAFT_AVAILABLE,
        work_units=(),
        estimated_remaining_cost_usd=0.04,
        estimate_complete=True,
    )

    assert tail.current_reserve_usd == pytest.approx(0.04)


def test_the_reserve_is_released_once_the_tail_is_done():
    tail = controller(0.10)

    tail.checkpoint(
        TailCheckpointName.MANDATORY_TAIL_COMPLETE,
        work_units=(),
        estimated_remaining_cost_usd=None,
        estimate_complete=True,
    )

    # Enhancement passes may now use everything that is left.
    assert tail.current_reserve_usd == 0.0
    assert tail.reserve_for_call("evidence_gap", 0.02) == 0.0


def test_the_run_start_estimate_is_recorded_as_an_estimate_not_a_guarantee():
    audit = controller(0.10).audit()

    first = audit.checkpoints[0]
    assert first.checkpoint is TailCheckpointName.RUN_START
    assert first.estimate_complete is False
    assert first.limitations == (
        "downstream work units do not exist before the draft",
    )


def test_an_unconfigured_reserve_records_null_rather_than_zero():
    """No estimate is not an estimate of zero."""

    audit = EvidenceTailReserveController().audit()

    assert audit.checkpoints[0].estimated_remaining_cost_usd is None


def test_per_unit_cost_is_null_when_no_units_were_observed():
    """Calibration data must not gain a fabricated denominator.

    These observations exist so a future reserve can be calibrated from real
    work. A zero-unit stage that recorded 0.0 per unit would silently drag any
    future average toward zero.
    """

    tail = controller()
    tail.observe_stage(
        "attribution", work_units=(), token_count=900, cost_usd=0.02
    )
    tail.observe_stage(
        "attribution",
        work_units=(TailWorkUnit(stage="attribution", unit="external_claim", count=4),),
        token_count=800,
        cost_usd=0.02,
    )

    empty, measured = tail.audit().stage_observations
    assert empty.cost_per_unit_usd is None
    assert empty.tokens_per_unit is None
    assert measured.cost_per_unit_usd == pytest.approx(0.005)
    assert measured.tokens_per_unit == pytest.approx(200.0)


def test_observations_accumulate_for_later_calibration():
    """The point of recording these is that no defensible number exists yet."""

    tail = controller()
    tail.observe_stage(
        "claim_decomposition",
        work_units=(TailWorkUnit(stage="claim_decomposition", unit="markdown_block", count=16),),
        token_count=3200,
        cost_usd=0.008,
    )

    audit = tail.audit()
    assert len(audit.stage_observations) == 1
    assert audit.stage_observations[0].stage == "claim_decomposition"
    assert audit.stage_observations[0].cost_per_unit_usd == pytest.approx(
        0.0005
    )
