#!/usr/bin/env python3
"""Export a real run as a pending human report-preservation review packet.

The exporter is intentionally read-only: it freezes the original task,
checklist questions, report bytes, and source-audit hash, but it never invents
keypoints or coverage judgements.  A reviewer supplies those semantic labels
after export; pending output is therefore not a zero-valued rubric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from open_deep_research.harness.report_quality import (
    FrozenChecklistQuestion,
    FrozenReportRubricCase,
    ReportReviewStatus,
    ReportRubricGold,
)


def export_report_rubric_packet(audit_path: Path) -> dict[str, Any]:
    """Freeze one audit's task/report inputs and leave rubric gold pending."""

    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    checklist = audit.get("checklist")
    if not isinstance(checklist, dict):
        raise ValueError("audit has no checklist object")
    raw_items = checklist.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("audit checklist has no items list")
    questions = tuple(
        FrozenChecklistQuestion(
            item_id=str(item["item_id"]),
            question=str(item["question"]),
        )
        for item in raw_items
        if isinstance(item, dict)
        and isinstance(item.get("item_id"), str)
        and isinstance(item.get("question"), str)
    )
    if len(questions) != len(raw_items):
        raise ValueError("every checklist item needs an ID and question")
    report = audit.get("canonical_draft")
    if not isinstance(report, str) or not report:
        raise ValueError("audit has no canonical draft")
    run_id = audit.get("run_id")
    topic = audit.get("topic")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("audit has no run_id")
    if not isinstance(topic, str) or not topic:
        raise ValueError("audit has no topic")
    case = FrozenReportRubricCase(
        case_id=f"{run_id}:original-report",
        source_run_id=run_id,
        source_audit_path=f"harness_runs/{run_id}/audit.json",
        source_audit_sha256=hashlib.sha256(audit_bytes).hexdigest(),
        topic=topic,
        checklist_questions=questions,
        baseline_report_text=report,
        baseline_report_sha256=hashlib.sha256(report.encode("utf-8")).hexdigest(),
    )
    gold = ReportRubricGold(
        case_id=case.case_id,
        review_status=ReportReviewStatus.PENDING_REVIEW,
    )
    return {
        "schema_version": "report-preservation-review-v1",
        "case": case.model_dump(mode="json"),
        "gold": gold.model_dump(mode="json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = export_report_rubric_packet(args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
