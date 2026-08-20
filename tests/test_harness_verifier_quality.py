from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from open_deep_research.harness.verifier_quality import (
    FrozenVerifierCase,
    VerifierGold,
    VerifierGoldStatus,
    VerifierPrediction,
    evaluate_verifier_challenge,
    original_verifier_predictions,
)
from open_deep_research.harness.verify import VerificationVerdict


FIXTURE = (
    Path(__file__).parent / "fixtures" / "harness_verifier_finance18_review.json"
)
CASES_SHA256 = "b15bc254aaa7d704440e48175761d004650cb81e898b39291ee0fa5a19ca761d"
FINANCE24_FIXTURE = (
    Path(__file__).parent / "fixtures" / "harness_verifier_finance24_review.json"
)
FINANCE24_CASES_SHA256 = (
    "7fb0161938bd8e532ca4509c435d5a012709ca4e47f96e05764fb1e6875e6382"
)


def _load_fixture() -> tuple[tuple[FrozenVerifierCase, ...], tuple[VerifierGold, ...]]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    canonical_cases = json.dumps(
        payload["cases"],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert hashlib.sha256(canonical_cases).hexdigest() == CASES_SHA256
    assert payload["cases_sha256"] == CASES_SHA256
    return (
        tuple(FrozenVerifierCase.model_validate(item) for item in payload["cases"]),
        tuple(VerifierGold.model_validate(item) for item in payload["gold"]),
    )


def test_finance18_verifier_challenge_freezes_the_tenfold_error() -> None:
    cases, gold = _load_fixture()

    assert len(cases) == 81
    assert len(gold) == 81
    known = [case for case in cases if "9000 万美元" in case.claim_text]
    assert len(known) == 1
    assert known[0].original_verdict is VerificationVerdict.SUPPORTS
    assert "$900 million" in known[0].evidence_quote
    assert all(item.review_status is VerifierGoldStatus.PENDING_REVIEW for item in gold)


def test_finance24_frozen_44_case_denominator_is_hash_and_id_closed() -> None:
    payload = json.loads(FINANCE24_FIXTURE.read_text(encoding="utf-8"))
    canonical_cases = json.dumps(
        payload["cases"],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    cases = tuple(
        FrozenVerifierCase.model_validate(item) for item in payload["cases"]
    )
    gold = tuple(VerifierGold.model_validate(item) for item in payload["gold"])

    assert hashlib.sha256(canonical_cases).hexdigest() == FINANCE24_CASES_SHA256
    assert payload["cases_sha256"] == FINANCE24_CASES_SHA256
    assert len(cases) == len(gold) == 44
    assert tuple(item.case_id for item in cases) == tuple(
        item.case_id for item in gold
    )
    assert all(
        item.review_status is not VerifierGoldStatus.PENDING_REVIEW
        for item in gold
    )


def test_pending_verifier_gold_produces_no_accuracy_or_hidden_negatives() -> None:
    cases, gold = _load_fixture()

    metrics = evaluate_verifier_challenge(
        cases,
        gold,
        original_verifier_predictions(cases),
        system_id="finance-18",
    )

    assert metrics.reviewed_case_ids == ()
    assert len(metrics.pending_case_ids) == 81
    assert metrics.accuracy is None
    assert all(
        count == 0
        for row in metrics.confusion_matrix.values()
        for count in row.values()
    )


def test_confusion_matrix_keeps_false_support_separate_from_other_errors() -> None:
    cases = (
        FrozenVerifierCase(
            case_id="false-support",
            source_run_id="fixture",
            audit_view="post_edit",
            claim_id="claim-1",
            claim_text="The company held $90 million.",
            source_id="source-1",
            url="https://example.test/one",
            source_text_sha256="0" * 64,
            evidence_quote="The company held $900 million.",
            original_verdict=VerificationVerdict.SUPPORTS,
            original_explanation="The model overlooked the magnitude.",
        ),
        FrozenVerifierCase(
            case_id="true-contradiction",
            source_run_id="fixture",
            audit_view="post_edit",
            claim_id="claim-2",
            claim_text="The filing occurred on Monday.",
            source_id="source-2",
            url="https://example.test/two",
            source_text_sha256="1" * 64,
            evidence_quote="The filing did not occur on Monday.",
            original_verdict=VerificationVerdict.CONTRADICTS,
            original_explanation="Explicit negation.",
        ),
    )
    gold = (
        VerifierGold(
            case_id="false-support",
            review_status=VerifierGoldStatus.REVIEWED,
            verdict=VerificationVerdict.DOES_NOT_SUPPORT,
            reviewer="human-1",
            rationale="The amount differs by a factor of ten.",
        ),
        VerifierGold(
            case_id="true-contradiction",
            review_status=VerifierGoldStatus.REVIEWED,
            verdict=VerificationVerdict.CONTRADICTS,
            reviewer="human-1",
            rationale="The evidence explicitly negates the claim.",
        ),
    )

    predictions = (
        VerifierPrediction(
            case_id="false-support",
            verdict=VerificationVerdict.SUPPORTS,
            explanation="Candidate prediction.",
        ),
        VerifierPrediction(
            case_id="true-contradiction",
            verdict=VerificationVerdict.CONTRADICTS,
            explanation="Candidate prediction.",
        ),
    )
    metrics = evaluate_verifier_challenge(
        cases,
        gold,
        predictions,
        system_id="candidate",
    )

    assert metrics.accuracy == 0.5
    assert metrics.confusion_matrix["does_not_support"]["supports"] == 1
    assert metrics.confusion_matrix["contradicts"]["contradicts"] == 1
    supports = next(
        item for item in metrics.per_class if item.label is VerificationVerdict.SUPPORTS
    )
    assert supports.predicted == 1
    assert supports.true_positive == 0
    assert supports.precision == 0.0


def test_pending_verifier_gold_cannot_smuggle_in_a_verdict() -> None:
    with pytest.raises(ValueError, match="pending verifier gold"):
        VerifierGold(
            case_id="case-1",
            review_status=VerifierGoldStatus.PENDING_REVIEW,
            verdict=VerificationVerdict.SUPPORTS,
        )

    with pytest.raises(ValueError, match="pending verifier gold"):
        VerifierGold(
            case_id="case-1",
            review_status=VerifierGoldStatus.PENDING_REVIEW,
            reviewer="",
        )


def test_candidate_predictions_cannot_skip_hard_cases() -> None:
    cases, gold = _load_fixture()

    with pytest.raises(ValueError, match="must cover the frozen challenge"):
        evaluate_verifier_challenge(
            cases,
            gold,
            original_verifier_predictions(cases[:-1]),
            system_id="incomplete-candidate",
        )


def test_a_provisional_model_label_is_never_scored_as_ground_truth():
    """A model-drafted verdict must carry its work without claiming authority.

    The two-state enum forced an annotator to choose between deleting the
    annotation and calling a model label human-reviewed. The third state exists
    so the label can be recorded and still stay out of the numerator.
    """

    case = FrozenVerifierCase(
        case_id="c1",
        source_run_id="finance-24",
        audit_view="post_edit",
        claim_id="claim-0018",
        claim_text="Alameda owns 90% of FTX.",
        source_id="source-1",
        url="https://example.com/a",
        source_text_sha256="0" * 64,
        evidence_quote="Bankman-Fried still owned 90% of Alameda.",
        original_verdict=VerificationVerdict.SUPPORTS,
        original_explanation="matched 90% and Alameda",
    )
    gold = VerifierGold(
        case_id="c1",
        review_status=VerifierGoldStatus.PROVISIONAL_MODEL_REVIEW,
        verdict=VerificationVerdict.DOES_NOT_SUPPORT,
        reviewer="claude-pm",
        rationale="the quote reverses subject and object",
    )

    metrics = evaluate_verifier_challenge(
        cases=(case,),
        gold=(gold,),
        predictions=(
            VerifierPrediction(
                case_id="c1",
                verdict=VerificationVerdict.DOES_NOT_SUPPORT,
                explanation="reversed",
            ),
        ),
        system_id="probe",
    )

    assert metrics.reviewed_case_ids == ()
    assert metrics.pending_case_ids == ("c1",)
    # A perfect prediction against a provisional label earns no accuracy.
    assert metrics.accuracy is None


def test_a_provisional_label_without_its_reasoning_is_rejected():
    with pytest.raises(ValidationError, match="rationale"):
        VerifierGold(
            case_id="c1",
            review_status=VerifierGoldStatus.PROVISIONAL_MODEL_REVIEW,
            verdict=VerificationVerdict.DOES_NOT_SUPPORT,
            reviewer="claude-pm",
        )
