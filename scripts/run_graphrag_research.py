"""Run one live investigation through the shared GraphRAG runtime.

This is intentionally a thin command-line adapter.  The LangGraph application
and multi-topic regression suite import the same runtime, so fixes cannot land
in a script while production keeps executing a stale copy.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE / "src"))
load_dotenv(_BASE / ".env")

from open_deep_research.graphrag.runtime import (  # noqa: E402
    GraphResearchSettings,
    run_live_graph_research,
)


async def main(topic: str, *, max_rounds: int, coverage_target: float) -> int:
    """Execute one run and persist its report plus machine-readable audit."""

    settings = GraphResearchSettings(
        model=os.environ.get("OPENAI_MODEL", "openai/gpt-4.1-mini"),
        max_rounds=max_rounds,
        coverage_target=coverage_target,
    )
    result = await run_live_graph_research(
        topic,
        settings=settings,
        progress=print,
    )

    report_path = _BASE / f"graphrag_report_{result.research_id}.md"
    audit_path = _BASE / f"graphrag_run_{result.research_id}.json"
    report_path.write_text(result.report, encoding="utf-8")
    audit_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    cost = (
        f"${result.usage.chat_provider_cost_usd:.4f}"
        if result.usage.provider_cost_reported
        else "provider did not report chat cost"
    )
    print(
        f"\n完成：coverage={result.coverage_ratio:.0%}, "
        f"facts={result.fact_count}, sources={result.source_count}, "
        f"rounds={len(result.rounds)}, tokens={result.usage.total_tokens}, "
        f"chat_cost={cost}, elapsed={result.usage.elapsed_seconds:.1f}s"
    )
    print(f"报告：{report_path}")
    print(f"审计：{audit_path}")
    return 0 if result.coverage_ratio >= coverage_target else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("topic")
    parser.add_argument("--max-rounds", type=int, default=24)
    parser.add_argument("--coverage-target", type=float, default=1.0)
    arguments = parser.parse_args()
    raise SystemExit(
        asyncio.run(
            main(
                arguments.topic,
                max_rounds=arguments.max_rounds,
                coverage_target=arguments.coverage_target,
            )
        )
    )
