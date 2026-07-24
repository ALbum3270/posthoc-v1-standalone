"""Recompute deterministic regression metrics from saved live-run audits.

No network calls are made.  The previous model-judged relevance score is carried
forward from its evaluation JSON; all fixed checks and structural metrics are
recomputed from the saved :class:`GraphResearchResult`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE / "src"))

from open_deep_research.graphrag.evaluation.regression import (  # noqa: E402
    DEFAULT_REGRESSION_CASES,
    CaseEvaluation,
    RegressionSuiteResult,
    evaluate_case,
    render_regression_summary,
)
from open_deep_research.graphrag.runtime import GraphResearchResult  # noqa: E402


def main(directory: Path) -> int:
    """Re-evaluate every known case audit in ``directory``."""

    case_by_id = {case.case_id: case for case in DEFAULT_REGRESSION_CASES}
    evaluations = []
    for result_path in sorted(directory.glob("*_run-*.json")):
        if result_path.name.endswith("_evaluation.json"):
            continue
        case_id = next(
            (case_id for case_id in case_by_id if result_path.name.startswith(case_id)),
            None,
        )
        if case_id is None:
            continue

        result = GraphResearchResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        evaluation_path = result_path.with_name(
            result_path.stem + "_evaluation.json"
        )
        old_evaluation = None
        if evaluation_path.exists():
            old_evaluation = CaseEvaluation.model_validate_json(
                evaluation_path.read_text(encoding="utf-8")
            )
        evaluation = evaluate_case(
            result,
            case_by_id[case_id],
            slot_relevance=(
                old_evaluation.model_judged_slot_relevance
                if old_evaluation is not None
                else None
            ),
            judged_item_count=(
                old_evaluation.judged_item_count if old_evaluation is not None else 0
            ),
        )
        evaluation_path.write_text(
            evaluation.model_dump_json(indent=2),
            encoding="utf-8",
        )
        evaluations.append(evaluation)

    suite = RegressionSuiteResult(cases=evaluations)
    (directory / "summary.json").write_text(
        suite.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (directory / "summary.md").write_text(
        render_regression_summary(suite),
        encoding="utf-8",
    )
    print(f"Re-evaluated {len(evaluations)} case(s) in {directory}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    raise SystemExit(main(args.directory))
