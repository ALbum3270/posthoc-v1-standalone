"""A stage that never ran must never be readable as a zero-valued result."""

import pytest
from pydantic import ValidationError

from open_deep_research.harness.stages import (
    MANDATORY_PIPELINE_STAGES,
    PostDraftExecutionAudit,
    QualityReviewStatus,
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
    return {name: complete("claim", 3) for name in MANDATORY_PIPELINE_STAGES}


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


def test_pipeline_completion_is_derived_not_asserted():
    """A caller cannot declare execution complete against its stage ledger.

    This is only a workflow fact.  It deliberately says nothing about report
    correctness or publication quality, which require a separate review.
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
            pipeline_complete=True,
            pipeline_completion_reason="looks complete to me",
        )


def test_one_incomplete_mandatory_stage_blocks_pipeline_completion():
    stages = all_mandatory_complete()
    stages["initial_verification"] = StageExecutionRecord(
        status=StageExecutionStatus.PARTIAL,
        reason="cap reached mid-pass",
        expected_scope=StageScope(unit="relation", count=66),
        evaluated_scope=StageScope(unit="relation", count=20),
    )

    audit = publication_audit(stages)

    assert audit.pipeline_complete is False
    assert audit.quality_review_passed is None
    assert audit.publication_eligible is False
    assert "initial_verification" in audit.pipeline_completion_reason


def test_a_fully_complete_tail_is_not_an_independent_quality_review():
    audit = publication_audit(all_mandatory_complete())

    assert audit.pipeline_complete is True
    assert audit.pipeline_completion_reason == (
        "all mandatory post-draft stages completed"
    )
    assert audit.quality_review_status is QualityReviewStatus.NOT_REVIEWED
    assert audit.quality_review_passed is None
    assert audit.publication_eligible is False
    assert "independent" in audit.quality_review_reason


def test_a_missing_mandatory_stage_is_refused_rather_than_assumed():
    """Absence must not read as success; the audit refuses to be constructed."""

    stages = all_mandatory_complete()
    del stages["checklist_reconciliation"]

    with pytest.raises(ValidationError):
        PostDraftExecutionAudit(
            stages=stages,
            pipeline_complete=False,
            pipeline_completion_reason="missing a stage",
        )


def test_optional_enhancement_passes_do_not_gate_pipeline_completion():
    """Gap and disagreement are enhancements, not part of the evidence tail."""

    stages = all_mandatory_complete()
    stages["evidence_gap"] = StageExecutionRecord(
        status=StageExecutionStatus.NOT_RUN, reason="budget exhausted"
    )
    stages["disagreement"] = StageExecutionRecord(
        status=StageExecutionStatus.NOT_RUN, reason="budget exhausted"
    )

    audit = publication_audit(stages)
    assert audit.pipeline_complete is True
    assert audit.quality_review_passed is None


def test_historical_publication_shape_is_read_but_not_reemitted():
    """Old audits remain readable without reviving their over-strong claim."""

    audit = PostDraftExecutionAudit.model_validate(
        {
            "stages": {
                name: record.model_dump(mode="json")
                for name, record in all_mandatory_complete().items()
            },
            "mandatory_publication_stages": list(MANDATORY_PIPELINE_STAGES),
            "publication_eligible": True,
            "publication_reason": "all old mandatory stages completed",
        }
    )

    assert audit.pipeline_complete is True
    assert audit.quality_review_passed is None
    assert audit.publication_eligible is False
    emitted = audit.model_dump(mode="json")
    assert "publication_eligible" not in emitted
    assert "publication_reason" not in emitted
    assert "mandatory_publication_stages" not in emitted


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


def test_invariant_catches_gap_payload_whose_routes_do_not_match_completion():
    """A non-null payload is not proof that 58 target claims were evaluated."""

    from open_deep_research.harness.stages import (
        stages_claiming_completion_without_output,
    )

    target_ids = [f"claim-{index:04d}" for index in range(1, 59)]
    audit = {
        "posthoc_evidence": {
            "evidence_gap": {
                "target_claim_ids": target_ids,
                "routed_target_claim_ids": target_ids[:2],
                "unrouted_target_claim_ids": target_ids[2:],
                "cached_candidate_hints": [],
                "searches": [
                    {"query": {"claim_ids": target_ids[:2]}}
                ],
            },
            "stage_execution": {
                "stages": {
                    "evidence_gap": {
                        "status": "complete",
                        "expected_scope": {
                            "unit": "target_claim",
                            "count": 58,
                        },
                        "evaluated_scope": {
                            "unit": "target_claim",
                            "count": 58,
                        },
                        "unevaluated_ids": [],
                    }
                }
            },
        }
    }

    assert stages_claiming_completion_without_output(audit) == (
        "evidence_gap",
    )


def test_invariant_accepts_honest_partial_gap_route_coverage():
    from open_deep_research.harness.stages import (
        stages_claiming_completion_without_output,
    )

    target_ids = [f"claim-{index:04d}" for index in range(1, 59)]
    audit = {
        "posthoc_evidence": {
            "evidence_gap": {
                "target_claim_ids": target_ids,
                "routed_target_claim_ids": target_ids[:2],
                "unrouted_target_claim_ids": target_ids[2:],
                "cached_candidate_hints": [],
                "searches": [
                    {"query": {"claim_ids": target_ids[:2]}}
                ],
            },
            "stage_execution": {
                "stages": {
                    "evidence_gap": {
                        "status": "partial",
                        "expected_scope": {
                            "unit": "target_claim",
                            "count": 58,
                        },
                        "evaluated_scope": {
                            "unit": "target_claim",
                            "count": 2,
                        },
                        "unevaluated_ids": target_ids[2:],
                    }
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
        gap_payload = audit["posthoc_evidence"].get("evidence_gap") or {}
        if (
            "evidence_gap" in offenders
            and "routed_target_claim_ids" not in gap_payload
        ):
            # Historical bundles predate explicit route coverage.  The
            # upgraded invariant deliberately exposes their old 58/58-style
            # completion claim, but immutable run evidence cannot acquire a
            # field retroactively.  Only that diagnosed legacy offender is
            # acknowledged here; current-schema bundles get no exemption.
            offenders = tuple(
                stage for stage in offenders if stage != "evidence_gap"
            )
        assert not offenders, f"{path.parent.name}: {offenders}"
    if not checked:
        pytest.skip("no stage-ledger bundles present in this checkout")


def test_an_empty_scope_from_a_finished_upstream_is_still_complete():
    """A report with no external claims genuinely leaves attribution nothing.

    The demotion must not punish that case: there the emptiness is a finding,
    not an artefact of truncation.
    """

    from open_deep_research.harness.stages import demote_vacuous_completions

    stages = {
        "claim_decomposition": complete("markdown_block", 4),
        "attribution": StageExecutionRecord(
            status=StageExecutionStatus.COMPLETE,
            reason="no external claims to attribute",
            expected_scope=StageScope(unit="external_claim", count=0),
            evaluated_scope=StageScope(unit="external_claim", count=0),
        ),
    }

    adjusted = demote_vacuous_completions(stages)

    assert adjusted["attribution"].status is StageExecutionStatus.COMPLETE


def test_an_empty_scope_from_a_truncated_upstream_is_demoted():
    from open_deep_research.harness.stages import demote_vacuous_completions

    stages = {
        "claim_decomposition": StageExecutionRecord(
            status=StageExecutionStatus.PARTIAL,
            reason="cap reached before any block was assessed",
            expected_scope=StageScope(unit="markdown_block", count=4),
            evaluated_scope=StageScope(unit="markdown_block", count=0),
        ),
        "attribution": StageExecutionRecord(
            status=StageExecutionStatus.COMPLETE,
            reason="no external claims to attribute",
            expected_scope=StageScope(unit="external_claim", count=0),
            evaluated_scope=StageScope(unit="external_claim", count=0),
        ),
    }

    adjusted = demote_vacuous_completions(stages)

    assert adjusted["attribution"].status is StageExecutionStatus.NOT_RUN
    assert "claim_decomposition did not complete" in (
        adjusted["attribution"].reason
    )
    # The fabricated denominator is dropped rather than kept at zero.
    assert adjusted["attribution"].expected_scope is None


def test_a_real_scope_is_never_demoted_even_if_upstream_was_cut():
    """Work actually done upstream must keep its record."""

    from open_deep_research.harness.stages import demote_vacuous_completions

    stages = {
        "claim_decomposition": StageExecutionRecord(
            status=StageExecutionStatus.PARTIAL,
            reason="cap reached partway",
            expected_scope=StageScope(unit="markdown_block", count=9),
            evaluated_scope=StageScope(unit="markdown_block", count=5),
        ),
        "attribution": complete("external_claim", 12),
    }

    adjusted = demote_vacuous_completions(stages)

    assert adjusted["attribution"].status is StageExecutionStatus.COMPLETE
