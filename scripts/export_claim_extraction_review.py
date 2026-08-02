#!/usr/bin/env python3
"""Export a real audit slice as a human claim-extraction review packet.

This is a read-only replay/export tool.  It never calls a model and never
modifies the source run.  The output freezes inputs and leaves every semantic
annotation pending rather than deriving gold labels from the system under
evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from open_deep_research.harness.claim_quality import (
    ClaimExtractionGold,
    ClaimReviewStatus,
    FrozenClaimExtractionCase,
    FrozenSurfaceSpan,
)
from open_deep_research.harness.source_spans import build_source_span_registry


def _claim_decomposition(audit: dict[str, Any], view: str) -> dict[str, Any]:
    posthoc = audit["posthoc_evidence"]
    if view == "post_edit":
        decomposition = posthoc.get("claim_decomposition")
    else:
        pre_edit = posthoc.get("pre_edit_evidence") or {}
        decomposition = pre_edit.get("claim_decomposition")
    if not isinstance(decomposition, dict):
        raise ValueError(f"audit has no {view} claim decomposition")
    return decomposition


def export_review_packet(
    audit_path: Path,
    *,
    view: str,
    failure_reason: str,
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    decomposition = _claim_decomposition(audit, view)
    report = audit["canonical_draft"]
    report_sha = hashlib.sha256(report.encode("utf-8")).hexdigest()
    block_by_id = {
        str(block["block_id"]): block for block in decomposition["blocks"]
    }
    cases: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    for claim in decomposition["claims"]:
        if claim.get("normalization_failure") != failure_reason:
            continue
        block = block_by_id[str(claim["block_id"])]
        block_text = str(block["text"])
        registry = build_source_span_registry(block_text)
        case_id = (
            f"{audit['run_id']}:{view}:{claim['claim_id']}"
        )
        case = FrozenClaimExtractionCase(
            case_id=case_id,
            source_run_id=str(audit["run_id"]),
            audit_view=view,
            original_claim_id=str(claim["claim_id"]),
            block_id=str(claim["block_id"]),
            report_text_sha256=report_sha,
            block_text=block_text,
            observed_selected_text=str(claim["selected_text"]),
            observed_claim_text=str(claim["claim_text"]),
            observed_failure=failure_reason,
            addressable_spans=tuple(
                FrozenSurfaceSpan(
                    text=segment.text,
                    start_char=segment.start_char,
                    end_char=segment.end_char,
                )
                for segment in registry.segments
            ),
        )
        cases.append(case.model_dump(mode="json"))
        gold.append(
            ClaimExtractionGold(
                case_id=case_id,
                review_status=ClaimReviewStatus.PENDING_REVIEW,
            ).model_dump(mode="json")
        )
    if not cases:
        raise ValueError(f"no claims matched failure reason {failure_reason}")
    canonical_cases = json.dumps(
        cases,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "schema_version": "claim-extraction-review-v1",
        "source_audit": f"harness_runs/{audit['run_id']}/audit.json",
        "source_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(canonical_cases).hexdigest(),
        "cases": cases,
        "gold": gold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--view",
        choices=("post_edit", "pre_edit"),
        default="post_edit",
    )
    parser.add_argument(
        "--failure-reason",
        default="selected_assertion_not_verbatim_in_block",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = export_review_packet(
        args.audit,
        view=args.view,
        failure_reason=args.failure_reason,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
