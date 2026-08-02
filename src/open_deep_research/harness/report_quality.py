"""Frozen, human-auditable evaluation of report-answer preservation.

The production editor may reason about usefulness, but it cannot certify that
removing an unsupported statement preserved the answer to the user's question.
This module keeps that semantic judgement outside the write path.  It binds
human or independently reviewed coverage findings to exact report bytes and
compares the same frozen keypoints before and after an editorial revision.

No composite score is emitted: evidence attribution and answer preservation are
separate axes, and pending review is never converted into a zero or a pass.
"""

from __future__ import annotations

import hashlib
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReportReviewStatus(str, Enum):
    """Whether a human or independent reviewer supplied a semantic review."""

    PENDING_REVIEW = "pending_review"
    REVIEWED = "reviewed"


class ReportKeypointImportance(str, Enum):
    """Human-supplied task relevance; it is not a runtime editor threshold."""

    CORE = "core"
    SUPPORTING = "supporting"


class KeypointCoverageStatus(str, Enum):
    """A reviewer's semantic coverage judgement for one answer keypoint."""

    COVERED = "covered"
    PARTIALLY_COVERED = "partially_covered"
    NOT_COVERED = "not_covered"


class ReportReviewAnchor(BaseModel):
    """One exact report-side location cited by a coverage reviewer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)

    @model_validator(mode="after")
    def _bounds_match_text(self) -> ReportReviewAnchor:
        if self.end_char <= self.start_char:
            raise ValueError("report review anchor end must exceed start")
        if self.end_char - self.start_char != len(self.text):
            raise ValueError("report review anchor bounds must match text")
        return self


class FrozenChecklistQuestion(BaseModel):
    """One original research question retained for human rubric construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1)
    question: str = Field(min_length=1)


class FrozenReportRubricCase(BaseModel):
    """Immutable report snapshot and task context for a preservation review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    source_run_id: str = Field(min_length=1)
    source_audit_path: str = Field(min_length=1)
    source_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    topic: str = Field(min_length=1)
    checklist_questions: tuple[FrozenChecklistQuestion, ...] = ()
    baseline_report_text: str = Field(min_length=1)
    baseline_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _frozen_input_is_hash_bound(self) -> FrozenReportRubricCase:
        digest = hashlib.sha256(
            self.baseline_report_text.encode("utf-8")
        ).hexdigest()
        if self.baseline_report_sha256 != digest:
            raise ValueError("baseline report hash does not match report bytes")
        identifiers = [item.item_id for item in self.checklist_questions]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("frozen checklist question IDs must be unique")
        return self


class ReportRubricKeypoint(BaseModel):
    """A human-authored answer point for one frozen research task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    keypoint_id: str = Field(min_length=1)
    requirement: str = Field(min_length=1)
    importance: ReportKeypointImportance
    rationale: str = Field(min_length=1)


class ReportRubricGold(BaseModel):
    """Human report rubric; pending packets deliberately contain no keypoints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    review_status: ReportReviewStatus
    keypoints: tuple[ReportRubricKeypoint, ...] = ()
    reviewer: str | None = None
    rationale: str | None = None

    @model_validator(mode="after")
    def _gold_is_not_guessed(self) -> ReportRubricGold:
        if self.review_status is ReportReviewStatus.PENDING_REVIEW:
            if self.keypoints or self.reviewer is not None or self.rationale is not None:
                raise ValueError("pending report rubric cannot contain guessed gold")
            return self
        if not self.keypoints:
            raise ValueError("reviewed report rubric requires keypoints")
        if not self.reviewer or not self.rationale:
            raise ValueError("reviewed report rubric requires reviewer and rationale")
        identifiers = [item.keypoint_id for item in self.keypoints]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("report rubric keypoint IDs must be unique")
        return self


class ReportKeypointAssessment(BaseModel):
    """One independent coverage assessment tied to exact report locations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    keypoint_id: str = Field(min_length=1)
    coverage: KeypointCoverageStatus
    anchors: tuple[ReportReviewAnchor, ...] = ()
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _coverage_has_the_right_kind_of_evidence(self) -> ReportKeypointAssessment:
        if self.coverage is KeypointCoverageStatus.NOT_COVERED and self.anchors:
            raise ValueError("not-covered keypoints cannot invent report anchors")
        if self.coverage is not KeypointCoverageStatus.NOT_COVERED and not self.anchors:
            raise ValueError("covered keypoints require exact report anchors")
        return self


class ReportQualitySystemJudgement(BaseModel):
    """Independent review of one concrete report version against a rubric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    system_id: str = Field(min_length=1)
    report_version_id: str = Field(min_length=1)
    report_text: str = Field(min_length=1)
    report_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_status: ReportReviewStatus
    keypoint_assessments: tuple[ReportKeypointAssessment, ...] = ()
    reviewer: str | None = None
    rationale: str | None = None

    @model_validator(mode="after")
    def _judgement_is_hash_bound_and_not_self_filled(self) -> ReportQualitySystemJudgement:
        digest = hashlib.sha256(self.report_text.encode("utf-8")).hexdigest()
        if self.report_text_sha256 != digest:
            raise ValueError("reviewed report hash does not match report bytes")
        if self.review_status is ReportReviewStatus.PENDING_REVIEW:
            if (
                self.keypoint_assessments
                or self.reviewer is not None
                or self.rationale is not None
            ):
                raise ValueError(
                    "pending report judgement cannot contain semantic labels"
                )
            return self
        if not self.reviewer or not self.rationale:
            raise ValueError(
                "reviewed report judgement requires reviewer and rationale"
            )
        identifiers = [item.keypoint_id for item in self.keypoint_assessments]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("report keypoint assessments must be unique")
        for assessment in self.keypoint_assessments:
            for anchor in assessment.anchors:
                if (
                    self.report_text[anchor.start_char : anchor.end_char]
                    != anchor.text
                ):
                    raise ValueError(
                        "report keypoint anchor must round-trip to report bytes"
                    )
        return self


class KeypointCoverageDistribution(BaseModel):
    """Counts, not a composite score, for a reviewed report version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    covered: int = Field(ge=0)
    partially_covered: int = Field(ge=0)
    not_covered: int = Field(ge=0)

    @property
    def total(self) -> int:
        return self.covered + self.partially_covered + self.not_covered


class ReportPreservationMetrics(BaseModel):
    """Paired human review results without a synthetic quality score."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    baseline_report_version_id: str
    candidate_report_version_id: str
    total_keypoints: int | None = Field(default=None, ge=0)
    reviewed_keypoint_ids: tuple[str, ...] = ()
    pending_keypoint_ids: tuple[str, ...] = ()
    baseline_distribution: KeypointCoverageDistribution | None = None
    candidate_distribution: KeypointCoverageDistribution | None = None
    preserved_or_improved_keypoint_ids: tuple[str, ...] = ()
    degraded_keypoint_ids: tuple[str, ...] = ()
    improved_keypoint_ids: tuple[str, ...] = ()
    unchanged_keypoint_ids: tuple[str, ...] = ()
    core_degraded_keypoint_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _paired_review_is_explicit(self) -> ReportPreservationMetrics:
        reviewed = set(self.reviewed_keypoint_ids)
        pending = set(self.pending_keypoint_ids)
        named_id_groups = {
            "reviewed": self.reviewed_keypoint_ids,
            "pending": self.pending_keypoint_ids,
            "preserved_or_improved": self.preserved_or_improved_keypoint_ids,
            "degraded": self.degraded_keypoint_ids,
            "improved": self.improved_keypoint_ids,
            "unchanged": self.unchanged_keypoint_ids,
            "core_degraded": self.core_degraded_keypoint_ids,
        }
        for label, identifiers in named_id_groups.items():
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"{label} keypoint IDs must be unique")
        if reviewed & pending:
            raise ValueError("reviewed and pending keypoints must not overlap")
        if self.total_keypoints is None:
            if reviewed or pending or self.baseline_distribution or self.candidate_distribution:
                raise ValueError("unreviewed rubric cannot manufacture keypoint metrics")
            return self
        if self.total_keypoints != len(reviewed) + len(pending):
            raise ValueError("reviewed and pending keypoints must cover rubric")
        if (self.baseline_distribution is None) != (self.candidate_distribution is None):
            raise ValueError("paired distributions must both be present or absent")
        if self.baseline_distribution is not None:
            if self.baseline_distribution.total != len(reviewed):
                raise ValueError("baseline distribution must cover reviewed keypoints")
            if self.candidate_distribution is None or self.candidate_distribution.total != len(reviewed):
                raise ValueError("candidate distribution must cover reviewed keypoints")
        classified = (
            set(self.degraded_keypoint_ids)
            | set(self.improved_keypoint_ids)
            | set(self.unchanged_keypoint_ids)
        )
        if self.total_keypoints is not None and classified != reviewed:
            raise ValueError("paired coverage changes must partition reviewed keypoints")
        if set(self.preserved_or_improved_keypoint_ids) - reviewed:
            raise ValueError("preservation IDs must be reviewed keypoints")
        if set(self.core_degraded_keypoint_ids) - set(self.degraded_keypoint_ids):
            raise ValueError("core degraded IDs must be a subset of degraded IDs")
        return self


_COVERAGE_SCORE = {
    KeypointCoverageStatus.NOT_COVERED: 0,
    KeypointCoverageStatus.PARTIALLY_COVERED: 1,
    KeypointCoverageStatus.COVERED: 2,
}


def _distribution(
    assessments: dict[str, ReportKeypointAssessment],
) -> KeypointCoverageDistribution:
    return KeypointCoverageDistribution(
        covered=sum(
            assessment.coverage is KeypointCoverageStatus.COVERED
            for assessment in assessments.values()
        ),
        partially_covered=sum(
            assessment.coverage is KeypointCoverageStatus.PARTIALLY_COVERED
            for assessment in assessments.values()
        ),
        not_covered=sum(
            assessment.coverage is KeypointCoverageStatus.NOT_COVERED
            for assessment in assessments.values()
        ),
    )


def evaluate_report_preservation(
    case: FrozenReportRubricCase,
    gold: ReportRubricGold,
    baseline: ReportQualitySystemJudgement,
    candidate: ReportQualitySystemJudgement,
) -> ReportPreservationMetrics:
    """Compare human-reviewed baseline/candidate coverage on fixed keypoints."""

    for value in (gold, baseline, candidate):
        if value.case_id != case.case_id:
            raise ValueError("report rubric inputs must share one frozen case")

    # The pre-edit side is not another arbitrary report version.  It is the
    # frozen snapshot from which the human rubric was authored.  Without this
    # byte-level binding, a comparison could silently move its baseline after
    # seeing the editor's revision and defeat the paired-evaluation contract.
    if (
        baseline.report_text != case.baseline_report_text
        or baseline.report_text_sha256 != case.baseline_report_sha256
    ):
        raise ValueError("baseline judgement must review frozen baseline report")

    if gold.review_status is not ReportReviewStatus.REVIEWED:
        return ReportPreservationMetrics(
            case_id=case.case_id,
            baseline_report_version_id=baseline.report_version_id,
            candidate_report_version_id=candidate.report_version_id,
        )

    keypoints = {item.keypoint_id: item for item in gold.keypoints}
    if (
        baseline.review_status is not ReportReviewStatus.REVIEWED
        or candidate.review_status is not ReportReviewStatus.REVIEWED
    ):
        return ReportPreservationMetrics(
            case_id=case.case_id,
            baseline_report_version_id=baseline.report_version_id,
            candidate_report_version_id=candidate.report_version_id,
            total_keypoints=len(keypoints),
            pending_keypoint_ids=tuple(keypoints),
        )

    baseline_by_id = {
        item.keypoint_id: item for item in baseline.keypoint_assessments
    }
    candidate_by_id = {
        item.keypoint_id: item for item in candidate.keypoint_assessments
    }
    if set(baseline_by_id) != set(keypoints) or set(candidate_by_id) != set(keypoints):
        raise ValueError(
            "reviewed baseline and candidate must both assess every gold keypoint"
        )

    degraded: list[str] = []
    improved: list[str] = []
    unchanged: list[str] = []
    preserved_or_improved: list[str] = []
    core_degraded: list[str] = []
    for keypoint_id, keypoint in keypoints.items():
        before = _COVERAGE_SCORE[baseline_by_id[keypoint_id].coverage]
        after = _COVERAGE_SCORE[candidate_by_id[keypoint_id].coverage]
        if after < before:
            degraded.append(keypoint_id)
            if keypoint.importance is ReportKeypointImportance.CORE:
                core_degraded.append(keypoint_id)
        elif after > before:
            improved.append(keypoint_id)
        else:
            unchanged.append(keypoint_id)
        if before > 0 and after >= before:
            preserved_or_improved.append(keypoint_id)

    reviewed_ids = tuple(keypoints)
    return ReportPreservationMetrics(
        case_id=case.case_id,
        baseline_report_version_id=baseline.report_version_id,
        candidate_report_version_id=candidate.report_version_id,
        total_keypoints=len(keypoints),
        reviewed_keypoint_ids=reviewed_ids,
        baseline_distribution=_distribution(baseline_by_id),
        candidate_distribution=_distribution(candidate_by_id),
        preserved_or_improved_keypoint_ids=tuple(preserved_or_improved),
        degraded_keypoint_ids=tuple(degraded),
        improved_keypoint_ids=tuple(improved),
        unchanged_keypoint_ids=tuple(unchanged),
        core_degraded_keypoint_ids=tuple(core_degraded),
    )


__all__ = [
    "FrozenChecklistQuestion",
    "FrozenReportRubricCase",
    "KeypointCoverageDistribution",
    "KeypointCoverageStatus",
    "ReportKeypointAssessment",
    "ReportKeypointImportance",
    "ReportPreservationMetrics",
    "ReportQualitySystemJudgement",
    "ReportReviewAnchor",
    "ReportReviewStatus",
    "ReportRubricGold",
    "ReportRubricKeypoint",
    "evaluate_report_preservation",
]
