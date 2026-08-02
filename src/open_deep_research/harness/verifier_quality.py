"""Human-gold evaluation of the semantic verifier.

The verifier is part of the system under test, not an oracle.  This module
keeps its frozen inputs and predictions separate from human gold, then reports
an explicit confusion matrix over reviewed cases only.  Pending cases remain
visible and never become true negatives, failures, or passes.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from open_deep_research.harness.verify import VerificationVerdict


class VerifierGoldStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    REVIEWED = "reviewed"


class FrozenVerifierCase(BaseModel):
    """One real claim/evidence-span pair and the system's original verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    source_run_id: str = Field(min_length=1)
    audit_view: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_quote: str = Field(min_length=1)
    original_verdict: VerificationVerdict
    original_explanation: str


class VerifierGold(BaseModel):
    """Human-approved verdict; no automatic judge may populate it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    review_status: VerifierGoldStatus
    verdict: VerificationVerdict | None = None
    reviewer: str | None = None
    rationale: str | None = None

    @model_validator(mode="after")
    def _gold_is_human_substantive(self) -> VerifierGold:
        if self.review_status is VerifierGoldStatus.PENDING_REVIEW:
            if any(
                value is not None
                for value in (self.verdict, self.reviewer, self.rationale)
            ):
                raise ValueError("pending verifier gold cannot contain guessed labels")
            return self
        if (
            self.verdict is None
            or not self.reviewer
            or not self.reviewer.strip()
            or not self.rationale
            or not self.rationale.strip()
        ):
            raise ValueError(
                "reviewed verifier gold requires verdict, reviewer, and rationale"
            )
        return self


class VerifierPrediction(BaseModel):
    """One system prediction kept separate from both cases and human gold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    verdict: VerificationVerdict
    explanation: str = ""


def original_verifier_predictions(
    cases: tuple[FrozenVerifierCase, ...],
) -> tuple[VerifierPrediction, ...]:
    """Expose the measured run as an explicit baseline prediction set."""

    return tuple(
        VerifierPrediction(
            case_id=case.case_id,
            verdict=case.original_verdict,
            explanation=case.original_explanation,
        )
        for case in cases
    )


class VerifierClassMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: VerificationVerdict
    support: int = Field(ge=0)
    predicted: int = Field(ge=0)
    true_positive: int = Field(ge=0)
    precision: float | None = Field(default=None, ge=0, le=1)
    recall: float | None = Field(default=None, ge=0, le=1)
    f1: float | None = Field(default=None, ge=0, le=1)


class VerifierChallengeMetrics(BaseModel):
    """No quality claim beyond the explicitly reviewed case subset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    system_id: str = Field(min_length=1)
    total_cases: int = Field(ge=0)
    reviewed_case_ids: tuple[str, ...]
    pending_case_ids: tuple[str, ...]
    confusion_matrix: dict[str, dict[str, int]]
    per_class: tuple[VerifierClassMetrics, ...]
    accuracy: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _case_partition_is_explicit(self) -> VerifierChallengeMetrics:
        all_ids = self.reviewed_case_ids + self.pending_case_ids
        if len(all_ids) != self.total_cases or len(set(all_ids)) != len(all_ids):
            raise ValueError("reviewed and pending IDs must partition total cases")
        return self


def evaluate_verifier_challenge(
    cases: tuple[FrozenVerifierCase, ...],
    gold: tuple[VerifierGold, ...],
    predictions: tuple[VerifierPrediction, ...],
    *,
    system_id: str,
) -> VerifierChallengeMetrics:
    """Build a confusion matrix without modifying frozen cases or gold."""

    case_by_id = {case.case_id: case for case in cases}
    gold_by_id = {item.case_id: item for item in gold}
    prediction_by_id = {item.case_id: item for item in predictions}
    if (
        len(case_by_id) != len(cases)
        or len(gold_by_id) != len(gold)
        or len(prediction_by_id) != len(predictions)
    ):
        raise ValueError("verifier case, gold, and prediction IDs must be unique")
    unknown = set(gold_by_id) - set(case_by_id)
    if unknown:
        raise ValueError(f"unknown verifier gold case IDs: {sorted(unknown)}")
    missing_predictions = set(case_by_id) - set(prediction_by_id)
    unknown_predictions = set(prediction_by_id) - set(case_by_id)
    if missing_predictions or unknown_predictions:
        raise ValueError(
            "verifier predictions must cover the frozen challenge exactly; "
            f"missing={sorted(missing_predictions)}, "
            f"unknown={sorted(unknown_predictions)}"
        )
    reviewed = tuple(
        case.case_id
        for case in cases
        if (
            (item := gold_by_id.get(case.case_id)) is not None
            and item.review_status is VerifierGoldStatus.REVIEWED
        )
    )
    pending = tuple(case.case_id for case in cases if case.case_id not in reviewed)
    labels = tuple(VerificationVerdict)
    matrix = {
        gold_label.value: {predicted.value: 0 for predicted in labels}
        for gold_label in labels
    }
    for case_id in reviewed:
        gold_item = gold_by_id[case_id]
        assert gold_item.verdict is not None
        predicted = prediction_by_id[case_id].verdict
        matrix[gold_item.verdict.value][predicted.value] += 1

    class_metrics: list[VerifierClassMetrics] = []
    correct = 0
    for label in labels:
        support = sum(matrix[label.value].values())
        predicted = sum(row[label.value] for row in matrix.values())
        true_positive = matrix[label.value][label.value]
        correct += true_positive
        precision = true_positive / predicted if predicted else None
        recall = true_positive / support if support else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None
            and recall is not None
            and precision + recall
            else None
        )
        class_metrics.append(
            VerifierClassMetrics(
                label=label,
                support=support,
                predicted=predicted,
                true_positive=true_positive,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return VerifierChallengeMetrics(
        system_id=system_id,
        total_cases=len(cases),
        reviewed_case_ids=reviewed,
        pending_case_ids=pending,
        confusion_matrix=matrix,
        per_class=tuple(class_metrics),
        accuracy=correct / len(reviewed) if reviewed else None,
    )
