from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from open_deep_research.harness.report_quality import (
    FrozenChecklistQuestion,
    FrozenReportRubricCase,
    KeypointCoverageStatus,
    ReportKeypointAssessment,
    ReportKeypointImportance,
    ReportQualitySystemJudgement,
    ReportReviewAnchor,
    ReportReviewStatus,
    ReportRubricGold,
    ReportRubricKeypoint,
    evaluate_report_preservation,
)
from scripts.export_report_rubric_review import export_report_rubric_packet


def _digest(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _case() -> FrozenReportRubricCase:
    report = "Cause: misuse.\n\nRecovery: 90 percent was returned."
    return FrozenReportRubricCase(
        case_id="finance-11:original-report",
        source_run_id="finance-11",
        source_audit_path="harness_runs/finance-11/audit.json",
        source_audit_sha256="a" * 64,
        topic="What happened and what was recovered?",
        checklist_questions=(
            FrozenChecklistQuestion(
                item_id="why-01",
                question="What caused the failure?",
            ),
            FrozenChecklistQuestion(
                item_id="where-01",
                question="What happened to customer funds?",
            ),
        ),
        baseline_report_text=report,
        baseline_report_sha256=_digest(report),
    )


def _gold(case: FrozenReportRubricCase) -> ReportRubricGold:
    return ReportRubricGold(
        case_id=case.case_id,
        review_status=ReportReviewStatus.REVIEWED,
        keypoints=(
            ReportRubricKeypoint(
                keypoint_id="cause",
                requirement="Explain the documented cause.",
                importance=ReportKeypointImportance.CORE,
                rationale="The user explicitly asked what happened.",
            ),
            ReportRubricKeypoint(
                keypoint_id="recovery",
                requirement="Answer what happened to customer funds.",
                importance=ReportKeypointImportance.CORE,
                rationale="The user explicitly asked about funds.",
            ),
            ReportRubricKeypoint(
                keypoint_id="timing",
                requirement="Give an event timeline.",
                importance=ReportKeypointImportance.SUPPORTING,
                rationale="Timeline provides supporting orientation.",
            ),
        ),
        reviewer="human-reviewer",
        rationale="A human froze task-relevant answer points before comparison.",
    )


def _judgement(
    case: FrozenReportRubricCase,
    *,
    version_id: str,
    report: str,
    cause: KeypointCoverageStatus,
    recovery: KeypointCoverageStatus,
    timing: KeypointCoverageStatus,
) -> ReportQualitySystemJudgement:
    coverage = {
        "cause": (cause, "Cause: misuse."),
        "recovery": (recovery, "Recovery: 90 percent was returned."),
        "timing": (timing, ""),
    }
    assessments = []
    for keypoint_id, (status, text) in coverage.items():
        anchors = ()
        if status is not KeypointCoverageStatus.NOT_COVERED:
            start = report.index(text)
            anchors = (
                ReportReviewAnchor(
                    text=text,
                    start_char=start,
                    end_char=start + len(text),
                ),
            )
        assessments.append(
            ReportKeypointAssessment(
                keypoint_id=keypoint_id,
                coverage=status,
                anchors=anchors,
                rationale="Independent human coverage assessment.",
            )
        )
    return ReportQualitySystemJudgement(
        case_id=case.case_id,
        system_id="editorial-candidate",
        report_version_id=version_id,
        report_text=report,
        report_text_sha256=_digest(report),
        review_status=ReportReviewStatus.REVIEWED,
        keypoint_assessments=tuple(assessments),
        reviewer="independent-reviewer",
        rationale="Review is independent of the editor that wrote the report.",
    )


def test_pending_report_rubric_never_becomes_zero_coverage() -> None:
    case = _case()
    gold = ReportRubricGold(
        case_id=case.case_id,
        review_status=ReportReviewStatus.PENDING_REVIEW,
    )
    pending = ReportQualitySystemJudgement(
        case_id=case.case_id,
        system_id="candidate",
        report_version_id="baseline",
        report_text=case.baseline_report_text,
        report_text_sha256=case.baseline_report_sha256,
        review_status=ReportReviewStatus.PENDING_REVIEW,
    )

    metrics = evaluate_report_preservation(case, gold, pending, pending)

    assert metrics.total_keypoints is None
    assert metrics.baseline_distribution is None
    assert metrics.candidate_distribution is None
    assert metrics.degraded_keypoint_ids == ()


def test_paired_report_review_exposes_lost_core_answer_without_composite_score() -> None:
    case = _case()
    gold = _gold(case)
    baseline = _judgement(
        case,
        version_id="before-edit",
        report=case.baseline_report_text,
        cause=KeypointCoverageStatus.COVERED,
        recovery=KeypointCoverageStatus.COVERED,
        timing=KeypointCoverageStatus.NOT_COVERED,
    )
    candidate_report = "Cause: misuse."
    candidate = _judgement(
        case,
        version_id="after-edit",
        report=candidate_report,
        cause=KeypointCoverageStatus.COVERED,
        recovery=KeypointCoverageStatus.NOT_COVERED,
        timing=KeypointCoverageStatus.NOT_COVERED,
    )

    metrics = evaluate_report_preservation(case, gold, baseline, candidate)

    assert metrics.total_keypoints == 3
    assert metrics.baseline_distribution.covered == 2
    assert metrics.candidate_distribution.covered == 1
    assert metrics.preserved_or_improved_keypoint_ids == ("cause",)
    assert metrics.degraded_keypoint_ids == ("recovery",)
    assert metrics.core_degraded_keypoint_ids == ("recovery",)
    assert "quality_score" not in metrics.model_dump(mode="json")


def test_reviewed_report_requires_exact_anchor_and_full_gold_coverage() -> None:
    case = _case()
    gold = _gold(case)
    with pytest.raises(ValidationError, match="round-trip"):
        ReportQualitySystemJudgement(
            case_id=case.case_id,
            system_id="candidate",
            report_version_id="candidate",
            report_text=case.baseline_report_text,
            report_text_sha256=case.baseline_report_sha256,
            review_status=ReportReviewStatus.REVIEWED,
            keypoint_assessments=(
                ReportKeypointAssessment(
                    keypoint_id="cause",
                    coverage=KeypointCoverageStatus.COVERED,
                    anchors=(
                        ReportReviewAnchor(
                            text="invented location",
                            start_char=0,
                            end_char=len("invented location"),
                        ),
                    ),
                    rationale="Bad anchor.",
                ),
            ),
            reviewer="independent-reviewer",
            rationale="Bad test control.",
        )

    reviewed = _judgement(
        case,
        version_id="baseline",
        report=case.baseline_report_text,
        cause=KeypointCoverageStatus.COVERED,
        recovery=KeypointCoverageStatus.COVERED,
        timing=KeypointCoverageStatus.NOT_COVERED,
    )
    incomplete = ReportQualitySystemJudgement(
        case_id=case.case_id,
        system_id="candidate",
        report_version_id="candidate",
        report_text=case.baseline_report_text,
        report_text_sha256=case.baseline_report_sha256,
        review_status=ReportReviewStatus.REVIEWED,
        keypoint_assessments=reviewed.keypoint_assessments[:1],
        reviewer="independent-reviewer",
        rationale="This intentionally omits keypoints.",
    )
    with pytest.raises(ValueError, match="every gold keypoint"):
        evaluate_report_preservation(case, gold, reviewed, incomplete)


def test_paired_review_cannot_move_its_frozen_baseline() -> None:
    case = _case()
    gold = _gold(case)
    moved_baseline = _judgement(
        case,
        version_id="not-the-frozen-baseline",
        report="Cause: misuse.",
        cause=KeypointCoverageStatus.COVERED,
        recovery=KeypointCoverageStatus.NOT_COVERED,
        timing=KeypointCoverageStatus.NOT_COVERED,
    )
    candidate = _judgement(
        case,
        version_id="candidate",
        report="Cause: misuse.",
        cause=KeypointCoverageStatus.COVERED,
        recovery=KeypointCoverageStatus.NOT_COVERED,
        timing=KeypointCoverageStatus.NOT_COVERED,
    )

    with pytest.raises(ValueError, match="frozen baseline"):
        evaluate_report_preservation(case, gold, moved_baseline, candidate)


def test_exporter_freezes_real_shape_without_inventing_report_keypoints(tmp_path) -> None:
    audit = {
        "run_id": "finance-11",
        "topic": "What happened to customer funds?",
        "canonical_draft": "A report awaiting human rubric review.",
        "checklist": {
            "items": [
                {"item_id": "where-01", "question": "Where did funds go?"}
            ]
        },
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")

    packet = export_report_rubric_packet(path)

    case = FrozenReportRubricCase.model_validate(packet["case"])
    gold = ReportRubricGold.model_validate(packet["gold"])
    assert case.source_run_id == "finance-11"
    assert case.checklist_questions[0].item_id == "where-01"
    assert gold.review_status is ReportReviewStatus.PENDING_REVIEW
    assert gold.keypoints == ()
