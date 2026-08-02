#!/usr/bin/env python3
"""Freeze real verifier claim/evidence pairs without inventing gold labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from open_deep_research.harness.verifier_quality import (
    FrozenVerifierCase,
    VerifierGold,
    VerifierGoldStatus,
)


def _verification(audit: dict[str, Any], view: str) -> dict[str, Any]:
    posthoc = audit["posthoc_evidence"]
    if view == "post_edit":
        value = posthoc.get("verification")
    else:
        value = (posthoc.get("pre_edit_evidence") or {}).get("verification")
    if not isinstance(value, dict):
        raise ValueError(f"audit has no {view} verification payload")
    return value


def export_verifier_packet(audit_path: Path, *, view: str) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    verification = _verification(audit, view)
    cases: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    for claim_result in verification["claims"]:
        claim = claim_result["claim"]
        for relation_index, relation in enumerate(claim_result["relations"]):
            if relation.get("status") != "completed":
                continue
            quote = relation.get("source_quote")
            source_sha = relation.get("source_text_sha256")
            verdict = relation.get("semantic_verdict")
            if not quote or not source_sha or not verdict:
                continue
            case_id = (
                f"{audit['run_id']}:{view}:{claim['claim_id']}:"
                f"{relation['source_id']}:{relation_index}"
            )
            case = FrozenVerifierCase(
                case_id=case_id,
                source_run_id=str(audit["run_id"]),
                audit_view=view,
                claim_id=str(claim["claim_id"]),
                claim_text=str(claim["claim_text"]),
                source_id=str(relation["source_id"]),
                url=str(relation["url"]),
                source_text_sha256=str(source_sha),
                evidence_quote=str(quote),
                original_verdict=verdict,
                original_explanation=str(relation.get("explanation") or ""),
            )
            cases.append(case.model_dump(mode="json"))
            gold.append(
                VerifierGold(
                    case_id=case_id,
                    review_status=VerifierGoldStatus.PENDING_REVIEW,
                ).model_dump(mode="json")
            )
    if not cases:
        raise ValueError("audit has no completed, located verifier relations")
    canonical_cases = json.dumps(
        cases,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "schema_version": "verifier-challenge-review-v1",
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
        "--view", choices=("post_edit", "pre_edit"), default="post_edit"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = export_verifier_packet(args.audit, view=args.view)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
