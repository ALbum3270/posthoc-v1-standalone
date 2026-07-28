"""The stop diagnosis must stay a record of facts, not a recommendation engine."""

import json

import pytest

from open_deep_research.harness.budget import RunCostBudget, RunCostController
from open_deep_research.harness.budget_diagnostics import (
    BlockedOperationQuality,
    BudgetDecisionSignal,
    CompletionStatus,
    OutstandingWork,
    RoundProductivity,
    classify_blocked_operation,
    judge_budget_decision,
    summarise_round_productivity,
)
from open_deep_research.harness.ledger import ResearchLedger


def productive(number: int) -> RoundProductivity:
    return RoundProductivity(
        round_number=number, action="read", new_sources=1, notes_created=2
    )


def wasteful(number: int, **kwargs: object) -> RoundProductivity:
    return RoundProductivity(round_number=number, action="read", **kwargs)


def outstanding() -> OutstandingWork:
    return OutstandingWork(open_checklist_items=3, unverified_relations=11)


def test_a_binding_cap_on_productive_work_says_budget_may_help():
    signal, evidence = judge_budget_decision(
        cap_was_binding=True,
        outstanding=outstanding(),
        recent_rounds=[productive(1), productive(2), productive(3)],
    )

    assert signal is BudgetDecisionSignal.MORE_BUDGET_MAY_HELP
    assert any("3 of 3" in line for line in evidence)


def test_a_binding_cap_on_unproductive_work_says_fix_the_mechanism_first():
    """Money cannot buy anything from calls that were producing nothing."""

    signal, evidence = judge_budget_decision(
        cap_was_binding=True,
        outstanding=outstanding(),
        recent_rounds=[
            wasteful(1, repeated_action=True),
            wasteful(2, tool_error=True),
            wasteful(3),
        ],
    )

    assert signal is BudgetDecisionSignal.FIX_MECHANISM_FIRST
    assert any("would repeat the same unproductive calls" in x for x in evidence)


def test_mixed_evidence_returns_indeterminate_and_recommends_nothing():
    """The failure we guard against looks exactly like mixed evidence.

    Topping up a run that was half spinning is the expensive mistake, so a
    mixture must not be rounded up into a recommendation to spend.
    """

    signal, evidence = judge_budget_decision(
        cap_was_binding=True,
        outstanding=outstanding(),
        recent_rounds=[productive(1), wasteful(2, repeated_action=True)],
    )

    assert signal is BudgetDecisionSignal.INDETERMINATE
    assert any("no recommendation is made" in line for line in evidence)


def test_a_binding_cap_with_no_outstanding_work_buys_nothing():
    signal, evidence = judge_budget_decision(
        cap_was_binding=True,
        outstanding=OutstandingWork(),
        recent_rounds=[productive(1)],
    )

    assert signal is BudgetDecisionSignal.NOT_APPLICABLE
    assert any("nothing to buy" in line for line in evidence)


def test_no_binding_cap_makes_the_question_moot():
    signal, evidence = judge_budget_decision(
        cap_was_binding=False,
        outstanding=outstanding(),
        recent_rounds=[wasteful(1)],
    )

    assert signal is BudgetDecisionSignal.NOT_APPLICABLE
    assert any("no work was withheld for cost" in line for line in evidence)


def test_zero_measured_rounds_is_indeterminate_not_a_verdict():
    signal, evidence = judge_budget_decision(
        cap_was_binding=True,
        outstanding=outstanding(),
        recent_rounds=[],
    )

    assert signal is BudgetDecisionSignal.INDETERMINATE
    assert evidence == ("no acquisition rounds ran, so productivity is unmeasured",)


def test_blocking_a_worthless_call_still_counts_as_binding():
    """cap_was_binding reports that a call was refused, never its worth.

    A tenth reread of a dead URL is a real refusal. Letting its worthlessness
    flip the flag to false would make a factual record double as a verdict on
    call quality, and would hide that the ceiling was active at all.
    """

    quality = classify_blocked_operation(
        [productive(1), wasteful(2, repeated_action=True)],
        cap_was_binding=True,
    )

    assert quality is BlockedOperationQuality.KNOWN_INVALID_OR_REPEATED


def test_blocking_productive_work_is_labelled_separately():
    quality = classify_blocked_operation(
        [wasteful(1), productive(2)], cap_was_binding=True
    )

    assert quality is BlockedOperationQuality.APPARENTLY_PRODUCTIVE


def test_nothing_blocked_when_no_ceiling_bound():
    assert (
        classify_blocked_operation([productive(1)], cap_was_binding=False)
        is BlockedOperationQuality.NOTHING_BLOCKED
    )


def test_productivity_is_read_off_the_ledger_without_asking_a_model():
    ledger = ResearchLedger(research_id="r", topic="t")
    ledger.record_round(
        round_number=1,
        action="search",
        query="q",
        result_summary=json.dumps({"candidate_urls": ["https://a", "https://b"]}),
    )
    ledger.record_round(
        round_number=2,
        action="read",
        url="https://a",
        result_summary=json.dumps({"notes_created": 3}),
    )
    ledger.record_round(
        round_number=3,
        action="search",
        query="q",
        result_summary=json.dumps({"candidate_urls": []}),
    )
    ledger.record_round(
        round_number=4,
        action="reanalyze",
        result_summary=json.dumps({"notes_created": 9}),
    )

    rounds = summarise_round_productivity(ledger.rounds)

    # reanalyze is bookkeeping, not acquisition, and is excluded rather than
    # counted as zero output -- counting it would bias the signal.
    assert [item.round_number for item in rounds] == [1, 2, 3]
    assert rounds[0].new_sources == 2
    assert rounds[1].notes_created == 3
    assert rounds[2].repeated_action is True
    assert rounds[2].was_wasteful is True
    assert rounds[1].was_wasteful is False


def test_recent_window_is_bounded_so_a_late_stall_is_visible():
    ledger = ResearchLedger(research_id="r", topic="t")
    for number in range(1, 12):
        ledger.record_round(
            round_number=number,
            action="read",
            url=f"https://{number}",
            result_summary=json.dumps({"notes_created": 1}),
        )

    rounds = summarise_round_productivity(ledger.rounds)

    assert len(rounds) == 5
    assert [item.round_number for item in rounds] == [7, 8, 9, 10, 11]


def test_cost_objective_is_recorded_and_never_blocks():
    """The objective must reach the audit without reaching control flow."""

    controller = RunCostController(
        RunCostBudget(max_cost_usd=0.5, cost_objective_usd=0.3)
    )
    controller.record_external_usage("collection", 0.42)
    audit = controller.audit()

    assert audit.cost_objective_usd == pytest.approx(0.3)
    assert audit.cost_objective_exceeded is True
    # Exceeding the objective did not stop anything: no call was ever refused.
    assert audit.cap_was_binding is False
    assert audit.remaining_cost_usd == pytest.approx(0.08)


def test_the_cap_admits_that_it_is_not_a_runaway_only_guard():
    """A cap this close to normal run cost must not be described as a net."""

    audit = RunCostController(RunCostBudget(max_cost_usd=0.5)).audit()

    assert any(
        "not a runaway-only guard" in limitation
        for limitation in audit.limitations
    )


def test_completion_status_has_a_value_for_work_that_never_began():
    assert CompletionStatus.NOT_STARTED.value == "not_started"
    assert CompletionStatus.COMPLETE.value == "complete"
