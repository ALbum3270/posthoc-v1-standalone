"""A stage that never ran must never be readable as a zero-valued result."""

import pytest
from pydantic import ValidationError

from open_deep_research.harness.stages import (
    MANDATORY_PUBLICATION_STAGES,
    PostDraftExecutionAudit,
    StageExecutionRecord,
    StageExecutionStatus,
    StageScope,
    publication_audit,
)


def complete(unit: str, count: int) -> StageExecutionRecord:
    return StageExecutionRecord(
        status=StageExecutionStatus.COMPLETE,
        reason="every unit received a disposition",
        expected_scope=StageScope(unit=unit, count=count),
        evaluated_scope=StageScope(unit=unit, count=count),
    )


def all_mandatory_complete() -> dict[str, StageExecutionRecord]:
    return {name: complete("claim", 3) for name in MANDATORY_PUBLICATION_STAGES}


def test_a_stage_that_never_ran_keeps_its_denominator():
    """not_run with a known scope is the whole point: 0/87, never 0/0.

    A cost cutoff during attribution leaves 87 claims unattributed. Reporting
    that as an evaluated zero would let a reader conclude the claims were
    checked and found sourceless, which is a domain conclusion the run never
    reached.
    """

    record = StageExecutionRecord(
        status=StageExecutionStatus.NOT_RUN,
        reason="no run cost allowance remains before reserve",
        expected_scope=StageScope(unit="external_claim", count=87),
        evaluated_scope=StageScope(unit="external_claim", count=0),
        unevaluated_ids=tuple(f"claim-{n}" for n in range(87)),
    )

    assert record.expected_scope.count == 87
    assert record.evaluated_scope.count == 0
    assert len(record.unevaluated_ids) == 87


def test_not_run_cannot_claim_evaluated_work():
    with pytest.raises(ValidationError):
        StageExecutionRecord(
            status=StageExecutionStatus.NOT_RUN,
            reason="cap reached",
            expected_scope=StageScope(unit="claim", count=5),
            evaluated_scope=StageScope(unit="claim", count=2),
        )


def test_an_unestablished_denominator_stays_absent_rather_than_zero():
    """No scope at all is honest; a zero scope would invent a denominator."""

    record = StageExecutionRecord(
        status=StageExecutionStatus.NOT_RUN,
        reason="not run because an earlier stage stopped",
    )

    assert record.expected_scope is None
    assert record.evaluated_scope is None

    # Claiming evaluated work without an established denominator is refused.
    with pytest.raises(ValidationError):
        StageExecutionRecord(
            status=StageExecutionStatus.NOT_RUN,
            reason="cap reached",
            evaluated_scope=StageScope(unit="claim", count=0),
        )


def test_complete_requires_the_whole_expected_scope():
    with pytest.raises(ValidationError):
        StageExecutionRecord(
            status=StageExecutionStatus.COMPLETE,
            reason="claims to be done",
            expected_scope=StageScope(unit="claim", count=10),
            evaluated_scope=StageScope(unit="claim", count=7),
        )


def test_complete_cannot_retain_unevaluated_ids():
    with pytest.raises(ValidationError):
        StageExecutionRecord(
            status=StageExecutionStatus.COMPLETE,
            reason="claims to be done",
            expected_scope=StageScope(unit="claim", count=3),
            evaluated_scope=StageScope(unit="claim", count=3),
            unevaluated_ids=("claim-1",),
        )


def test_evaluated_scope_cannot_exceed_expected():
    with pytest.raises(ValidationError):
        StageExecutionRecord(
            status=StageExecutionStatus.PARTIAL,
            reason="over-counted",
            expected_scope=StageScope(unit="claim", count=3),
            evaluated_scope=StageScope(unit="claim", count=4),
        )


def test_scope_units_must_match():
    with pytest.raises(ValidationError):
        StageExecutionRecord(
            status=StageExecutionStatus.PARTIAL,
            reason="mismatched units",
            expected_scope=StageScope(unit="claim", count=3),
            evaluated_scope=StageScope(unit="markdown_block", count=1),
        )


def test_publication_eligibility_is_derived_not_asserted():
    """A caller must not be able to declare its own output publishable.

    The banner and the filename are both easy to get wrong or to ignore. The
    eligibility flag is the mechanical gate, so it has to be impossible to set
    it to a value the stage ledger does not support.
    """

    stages = all_mandatory_complete()
    stages["attribution"] = StageExecutionRecord(
        status=StageExecutionStatus.NOT_RUN,
        reason="cap reached",
        expected_scope=StageScope(unit="claim", count=87),
        evaluated_scope=StageScope(unit="claim", count=0),
    )

    with pytest.raises(ValidationError):
        PostDraftExecutionAudit(
            stages=stages,
            publication_eligible=True,
            publication_reason="looks fine to me",
        )


def test_one_incomplete_mandatory_stage_blocks_publication():
    stages = all_mandatory_complete()
    stages["initial_verification"] = StageExecutionRecord(
        status=StageExecutionStatus.PARTIAL,
        reason="cap reached mid-pass",
        expected_scope=StageScope(unit="relation", count=66),
        evaluated_scope=StageScope(unit="relation", count=20),
    )

    audit = publication_audit(stages)

    assert audit.publication_eligible is False
    assert "initial_verification" in audit.publication_reason


def test_a_fully_complete_tail_is_publishable():
    audit = publication_audit(all_mandatory_complete())

    assert audit.publication_eligible is True
    assert audit.publication_reason == (
        "all mandatory post-draft stages completed"
    )


def test_a_missing_mandatory_stage_is_refused_rather_than_assumed():
    """Absence must not read as success; the audit refuses to be constructed."""

    stages = all_mandatory_complete()
    del stages["checklist_reconciliation"]

    with pytest.raises(ValidationError):
        PostDraftExecutionAudit(
            stages=stages,
            publication_eligible=False,
            publication_reason="missing a stage",
        )


def test_optional_enhancement_passes_do_not_gate_publication():
    """Gap and disagreement are enhancements, not part of the evidence tail."""

    stages = all_mandatory_complete()
    stages["evidence_gap"] = StageExecutionRecord(
        status=StageExecutionStatus.NOT_RUN, reason="budget exhausted"
    )
    stages["disagreement"] = StageExecutionRecord(
        status=StageExecutionStatus.NOT_RUN, reason="budget exhausted"
    )

    assert publication_audit(stages).publication_eligible is True


def test_an_ineligible_bundle_carries_no_footnote_definitions():
    """An incomplete run must not ship artifacts that read as verified.

    A measured cap-hit run produced a bundle whose sources file said, in place
    of footnotes, that it contains no fabricated empty evidence records. That
    is the property under test: incompleteness has to be visible in the
    artifacts themselves, not only in a flag that a reader may never open.
    """

    from pathlib import Path

    bundle = (
        Path(__file__).resolve().parents[1]
        / "harness_runs"
        / "_prebugfix"
        / "partial-04"
    )
    if not bundle.is_dir():
        pytest.skip("measured partial bundle is not present in this checkout")

    import json

    audit = json.loads((bundle / "audit.json").read_text(encoding="utf-8"))
    report = (bundle / "report.md").read_text(encoding="utf-8")
    sources = (bundle / "sources.md").read_text(encoding="utf-8")

    assert audit["publication_eligible"] is False
    assert audit["completion_status"] == "partial"
    # No footnote markers and no footnote definitions anywhere.
    assert "[^" not in report
    assert "[^" not in sources
    # The banner is the first thing a reader sees.
    assert report.lstrip().startswith("> **不完整运行产物")

    stages = audit["posthoc_evidence"]["stage_execution"]["stages"]
    attribution = stages["attribution"]
    assert attribution["status"] == "not_run"
    # The denominator survived: 0 of 87, never 0 of 0.
    assert attribution["expected_scope"]["count"] == 87
    assert attribution["evaluated_scope"]["count"] == 0
    assert len(attribution["unevaluated_ids"]) == 87


def test_the_invariant_catches_a_stage_that_completed_without_output():
    """This is the check that would have caught the evaluative bug for free."""

    from open_deep_research.harness.stages import (
        stages_claiming_completion_without_output,
    )

    audit = {
        "posthoc_evidence": {
            "claim_decomposition": {"blocks": []},
            "evaluative_claim_diagnostics": None,
            "verification": None,
            "stage_execution": {
                "stages": {
                    "claim_decomposition": {"status": "complete"},
                    "evaluative_diagnostics": {"status": "complete"},
                    "initial_verification": {"status": "not_run"},
                }
            },
        }
    }

    # Only the stage that claims completion with nothing to show is named. A
    # stage that honestly reports not_run has no payload and no case to answer.
    assert stages_claiming_completion_without_output(audit) == (
        "evaluative_diagnostics",
    )


def test_the_invariant_is_quiet_when_every_completed_stage_produced_output():
    from open_deep_research.harness.stages import (
        stages_claiming_completion_without_output,
    )

    audit = {
        "posthoc_evidence": {
            "claim_decomposition": {"blocks": []},
            "attribution": {"attributions": []},
            "stage_execution": {
                "stages": {
                    "claim_decomposition": {"status": "complete"},
                    "attribution": {"status": "complete"},
                }
            },
        }
    }

    assert stages_claiming_completion_without_output(audit) == ()


def test_every_completed_stage_in_a_measured_bundle_left_output(tmp_path):
    """Applied to whatever real bundles this checkout has on disk."""

    import json
    from pathlib import Path

    from open_deep_research.harness.stages import (
        stages_claiming_completion_without_output,
    )

    runs = Path(__file__).resolve().parents[1] / "harness_runs"
    bundles = sorted(runs.glob("*/audit.json")) if runs.is_dir() else []
    checked = 0
    for path in bundles:
        audit = json.loads(path.read_text(encoding="utf-8"))
        if "posthoc_evidence" not in audit:
            continue  # predates the stage ledger
        checked += 1
        offenders = stages_claiming_completion_without_output(audit)
        assert not offenders, f"{path.parent.name}: {offenders}"
    if not checked:
        pytest.skip("no stage-ledger bundles present in this checkout")
