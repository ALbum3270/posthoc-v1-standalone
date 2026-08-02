from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from open_deep_research.harness.claim_quality import (
    ClaimAmbiguityLabel,
    ClaimEntailmentLabel,
    ClaimExtractionGold,
    ClaimExtractionSystemJudgement,
    ClaimReviewStatus,
    ExtractedClaimJudgement,
    FrozenClaimExtractionCase,
    FrozenSurfaceSpan,
    compare_claim_extraction,
    evaluate_claim_extraction,
)


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "harness_claim_extraction_finance19_review.json"
)
CASES_SHA256 = "56d873274cc58c39ebc31115ca222136c59f77f5e68566ee9fa5cff050a125e9"


def _load_fixture() -> tuple[
    tuple[FrozenClaimExtractionCase, ...],
    tuple[ClaimExtractionGold, ...],
]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return (
        tuple(FrozenClaimExtractionCase.model_validate(x) for x in payload["cases"]),
        tuple(ClaimExtractionGold.model_validate(x) for x in payload["gold"]),
    )


def test_finance19_claim_extraction_inputs_are_frozen_but_gold_is_pending() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    canonical_cases = json.dumps(
        payload["cases"],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert hashlib.sha256(canonical_cases).hexdigest() == CASES_SHA256
    assert payload["cases_sha256"] == CASES_SHA256
    cases, gold = _load_fixture()

    assert len(cases) == 16
    assert len(gold) == 16
    assert {case.original_claim_id for case in cases} == {
        "claim-0009",
        "claim-0012",
        "claim-0018",
        "claim-0026",
        "claim-0030",
        "claim-0036",
        "claim-0039",
        "claim-0050",
        "claim-0072",
        "claim-0073",
        "claim-0076",
        "claim-0077",
        "claim-0081",
        "claim-0082",
        "claim-0092",
        "claim-0095",
    }
    assert all(item.review_status is ClaimReviewStatus.PENDING_REVIEW for item in gold)
    assert all(
        case.block_text[span.start_char : span.end_char] == span.text
        for case in cases
        for span in case.addressable_spans
    )


def test_pending_human_review_cannot_be_reported_as_a_semantic_score() -> None:
    cases, gold = _load_fixture()
    judgements = tuple(
        ClaimExtractionSystemJudgement(
            case_id=case.case_id,
            system_id="layered-v2",
            review_status=ClaimReviewStatus.PENDING_REVIEW,
        )
        for case in cases
    )

    metrics = evaluate_claim_extraction(cases, gold, judgements)

    assert metrics.reviewed_cases == 0
    assert len(metrics.pending_case_ids) == 16
    assert metrics.surface_binding_accuracy is None
    assert metrics.entailment_rate is None
    assert metrics.element_f1 is None


def _reviewed_inputs(
    *,
    system_id: str,
    entailed: bool,
) -> tuple[
    tuple[FrozenClaimExtractionCase, ...],
    tuple[ClaimExtractionGold, ...],
    tuple[ClaimExtractionSystemJudgement, ...],
]:
    block = "It expanded in 2022."
    surface = FrozenSurfaceSpan(
        text=block,
        start_char=0,
        end_char=len(block),
    )
    case = FrozenClaimExtractionCase(
        case_id="case-1",
        source_run_id="fixture",
        audit_view="post_edit",
        original_claim_id="claim-0001",
        block_id="block-0001",
        report_text_sha256="0" * 64,
        block_text=block,
        observed_selected_text=block,
        observed_claim_text="The facility expanded in 2022.",
        observed_failure="none",
        addressable_spans=(surface,),
    )
    gold = ClaimExtractionGold(
        case_id=case.case_id,
        review_status=ClaimReviewStatus.REVIEWED,
        acceptable_surface_spans=(surface,),
        ambiguity=ClaimAmbiguityLabel.RESOLVED_FROM_CONTEXT,
        verifiable_elements=("the facility expanded", "in 2022"),
        necessary_context=("A facility was opened.",),
        preferred_atomic_claims=("The facility expanded in 2022.",),
        reviewer="human-1",
        rationale="The preceding sentence uniquely resolves 'it'.",
    )
    judgement = ClaimExtractionSystemJudgement(
        case_id=case.case_id,
        system_id=system_id,
        review_status=ClaimReviewStatus.REVIEWED,
        selected_surface_spans=(surface,),
        claims=(
            ExtractedClaimJudgement(
                claim_text="The facility expanded in 2022.",
                entailment=(
                    ClaimEntailmentLabel.ENTAILED
                    if entailed
                    else ClaimEntailmentLabel.NOT_ENTAILED
                ),
                atomic=True,
                decontextualized=True,
                covered_element_ids=(0, 1) if entailed else (0,),
                extraneous_element_count=0 if entailed else 1,
                rationale="Human-reviewed synthetic control.",
            ),
        ),
        reviewer="human-1",
    )
    return (case,), (gold,), (judgement,)


def test_paired_metrics_use_identical_cases_and_separate_quality_axes() -> None:
    cases, gold, baseline_judgements = _reviewed_inputs(
        system_id="baseline",
        entailed=False,
    )
    _, _, candidate_judgements = _reviewed_inputs(
        system_id="candidate",
        entailed=True,
    )
    baseline = evaluate_claim_extraction(cases, gold, baseline_judgements)
    candidate = evaluate_claim_extraction(cases, gold, candidate_judgements)

    comparison = compare_claim_extraction(
        baseline,
        candidate,
    )

    assert baseline.entailment_rate == 0.0
    assert candidate.entailment_rate == 1.0
    assert candidate.surface_binding_accuracy == 1.0
    assert comparison.metric_deltas["entailment_rate"] == 1.0
    assert comparison.metric_deltas["element_recall"] == 0.5
    assert "success_score" not in comparison.model_dump(mode="json")

    moving_candidate_payload = candidate.model_dump(mode="json")
    moving_candidate_payload["reviewed_case_ids"] = ["different-case"]
    moving_candidate = type(candidate).model_validate(moving_candidate_payload)
    with pytest.raises(ValueError, match="identical ordered case IDs"):
        compare_claim_extraction(
            baseline,
            moving_candidate,
        )


def test_pending_records_cannot_smuggle_in_guessed_gold() -> None:
    with pytest.raises(ValueError, match="pending gold"):
        ClaimExtractionGold(
            case_id="case-1",
            review_status=ClaimReviewStatus.PENDING_REVIEW,
            preferred_atomic_claims=("A guessed claim.",),
        )
