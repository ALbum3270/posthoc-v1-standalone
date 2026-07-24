"""Run the fixed three-topic live GraphRAG regression suite.

The suite costs real Tavily, chat-model, and embedding calls.  Results are
written under ``regression_results/<timestamp>/`` as individual reports, full
JSON audits, and a compact cross-case summary.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE / "src"))
load_dotenv(_BASE / ".env")

from open_deep_research.graphrag.evaluation.regression import (  # noqa: E402
    DEFAULT_REGRESSION_CASES,
    RegressionSuiteResult,
    evaluate_case,
    judge_slot_relevance,
    render_regression_summary,
)
from open_deep_research.graphrag.runtime import (  # noqa: E402
    GraphResearchRunner,
    GraphResearchSettings,
    LiveServiceConfig,
)


async def main(
    *,
    case_ids: list[str],
    max_rounds: int,
    coverage_target: float,
    output_dir: Path | None,
    strict: bool,
) -> int:
    """Execute selected cases sequentially against shared live clients."""

    from graphiti_core import Graphiti
    from openai import AsyncOpenAI
    from tavily import AsyncTavilyClient

    selected = [
        case
        for case in DEFAULT_REGRESSION_CASES
        if not case_ids or case.case_id in set(case_ids)
    ]
    if not selected:
        raise ValueError(f"No regression cases matched: {case_ids}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = output_dir or (_BASE / "regression_results" / timestamp)
    destination.mkdir(parents=True, exist_ok=True)

    services = LiveServiceConfig.from_environment()
    settings = GraphResearchSettings(
        model=os.environ.get("OPENAI_MODEL", "openai/gpt-4.1-mini"),
        max_rounds=max_rounds,
        coverage_target=coverage_target,
    )
    llm = AsyncOpenAI(
        api_key=services.openai_api_key,
        base_url=services.openai_base_url,
    )
    tavily = AsyncTavilyClient(api_key=services.tavily_api_key)
    graphiti = Graphiti(
        uri=services.neo4j_uri,
        user=services.neo4j_user,
        password=services.neo4j_password,
    )

    evaluations = []
    try:
        for case in selected:
            print(f"\n{'=' * 72}\nCASE {case.case_id}: {case.topic}\n{'=' * 72}")
            runner = GraphResearchRunner(
                graphiti=graphiti,
                llm=llm,
                tavily=tavily,
                settings=settings,
                progress=lambda message, case_id=case.case_id: print(
                    f"[{case_id}] {message}"
                ),
            )
            result = await runner.run(case.topic)

            judge_started = time.perf_counter()
            relevance, judged_count = await judge_slot_relevance(
                result,
                llm=llm,
                model=settings.model,
            )
            result.usage.elapsed_seconds += time.perf_counter() - judge_started
            evaluation = evaluate_case(
                result,
                case,
                slot_relevance=relevance,
                judged_item_count=judged_count,
            )
            evaluations.append(evaluation)

            stem = f"{case.case_id}_{result.research_id}"
            (destination / f"{stem}.md").write_text(result.report, encoding="utf-8")
            (destination / f"{stem}.json").write_text(
                result.model_dump_json(indent=2),
                encoding="utf-8",
            )
            (destination / f"{stem}_evaluation.json").write_text(
                evaluation.model_dump_json(indent=2),
                encoding="utf-8",
            )
            print(
                f"[{case.case_id}] coverage={evaluation.coverage_ratio:.0%} "
                f"checks={evaluation.factual_check_pass_rate:.0%} "
                f"citations={evaluation.citation_coverage:.0%} "
                f"relevance={evaluation.model_judged_slot_relevance!r}"
            )
    finally:
        await graphiti.close()
        await llm.close()

    suite = RegressionSuiteResult(cases=evaluations)
    summary = render_regression_summary(suite)
    (destination / "summary.md").write_text(summary, encoding="utf-8")
    (destination / "summary.json").write_text(
        suite.model_dump_json(indent=2),
        encoding="utf-8",
    )
    print(f"\n回归结果：{destination}")
    return 2 if strict and not suite.all_checks_passed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        choices=[case.case_id for case in DEFAULT_REGRESSION_CASES],
        help="Run only this case; repeat to select multiple cases.",
    )
    parser.add_argument("--max-rounds", type=int, default=24)
    parser.add_argument("--coverage-target", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            main(
                case_ids=args.case,
                max_rounds=args.max_rounds,
                coverage_target=args.coverage_target,
                output_dir=args.output_dir,
                strict=args.strict,
            )
        )
    )
