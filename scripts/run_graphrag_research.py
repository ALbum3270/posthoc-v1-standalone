"""End-to-end graph-driven research run.

Wires the shipped modules together against live services -- the last half of the
core assumption that simulation cannot reach: whether real sources answer real
ontology questions (SESSION_HANDOFF §3.16).

    supervisor.plan_next_round        pick a slot, vary the query
    Tavily + adapters.content         search, strip chrome, select passages
    TargetedExtractor                 slot-directed triples
    graph.add_verified_episode        verbatim write, gated dates
    graph.queries.get_gap_status      coverage read back FROM THE GRAPH
    control.stopping.evaluate_stop    deterministic termination

Coverage is never tracked in a Python set: every round re-reads it from Neo4j,
which is the property M0 exists to provide.

Costs real Tavily and LLM calls. Requires the OpenRouter proxy (§3.7).

    python scripts/run_graphrag_research.py "FTX 暴雷事件" --max-rounds 24
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE / "src"))
load_dotenv(_BASE / ".env")

from open_deep_research.graphrag.adapters.content import (  # noqa: E402
    select_relevant_text,
)
from open_deep_research.graphrag.adapters.search_results import (  # noqa: E402
    tavily_result_to_source_document,
)
from open_deep_research.graphrag.control.researcher import (  # noqa: E402
    run_research_round,
)
from open_deep_research.graphrag.extraction.triple_payload import (  # noqa: E402
    parse_triple_payload,
)
from open_deep_research.graphrag.control.stopping import (  # noqa: E402
    StoppingConfig,
    count_improvement,
    evaluate_stop,
)
from open_deep_research.graphrag.control.supervisor import (  # noqa: E402
    SupervisorMemory,
    plan_next_round,
)
from open_deep_research.graphrag.graph.queries import (  # noqa: E402
    coverage_ratio_from_gaps,
    get_gap_status,
)
from open_deep_research.graphrag.reporting.evidence_pack import (  # noqa: E402
    build_evidence_pack,
    fetch_facts,
)
from open_deep_research.graphrag.reporting.report import (  # noqa: E402
    build_source_index,
    render_report,
)
from open_deep_research.graphrag.ontology import (  # noqa: E402
    INVESTIGATION_SCHEMA,
    iter_slots,
)
from open_deep_research.graphrag.schemas import (  # noqa: E402
    EntityRef,
    ExtractedTriple,
    SourceSpan,
)

GROUP_ID = "neo4j"  # must equal driver._database (§3.4)
MAX_CHARS = 2000
SEARCH_RESULTS = 5

QUERY_SYSTEM = (
    "You write one web search query. Output the query text only -- no quotes, no "
    "explanation. Prefer concrete nouns, names, numbers and dates over generic "
    "phrasing."
)
# Must agree with response_format={"type": "json_object"}: asking for a bare
# array while the API forces a top-level object is what made the model emit one
# unwrapped triple per call, which the first run then read as "no facts".
EXTRACT_SYSTEM = (
    "Extract factual triples from the passage that answer the given question.\n"
    "Reply with a JSON object of exactly this form:\n"
    '{"triples": [{"subject": "...", "predicate": "...", "object": "..."}]}\n'
    "Copy names, numbers and dates exactly as the passage writes them. Never "
    "infer or complete a date the passage does not state.\n"
    'If the passage does not answer the question, reply {"triples": []}.'
)


def openai_client():
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )


async def main(topic: str, max_rounds: int) -> int:
    from graphiti_core import Graphiti
    from tavily import AsyncTavilyClient

    llm = openai_client()
    model = os.environ.get("OPENAI_MODEL", "openai/gpt-4.1-mini")
    tavily = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    graphiti = Graphiti(
        uri=os.environ["NEO4J_URI"],
        user=os.environ.get("NEO4J_USER", "neo4j"),
        password=os.environ["NEO4J_PASSWORD"],
    )

    research_id = f"run-{uuid4().hex[:8]}"
    settings = StoppingConfig(max_rounds=max_rounds, max_no_improvement_rounds=4)
    memory = SupervisorMemory()
    all_slots = iter_slots(INVESTIGATION_SCHEMA)
    tried: set[str] = set()

    print(f"\n{'=' * 66}")
    print(f"课题     : {topic}")
    print(f"research : {research_id}")
    print(f"槽位     : {len(all_slots)}   轮次上限: {max_rounds}")
    print(f"{'=' * 66}\n")

    # ---- injected edges -------------------------------------------------
    async def generate_query(*, topic, slot, previous_queries):
        avoid = ""
        if previous_queries:
            listed = "; ".join(previous_queries)
            avoid = (
                f"\nThese queries were already tried and failed, so write a "
                f"materially different one: {listed}"
            )
        response = await llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": QUERY_SYSTEM},
                {
                    "role": "user",
                    "content": f"Topic: {topic}\nQuestion to answer: {slot.question}{avoid}",
                },
            ],
            temperature=0.3 if previous_queries else 0.0,
        )
        return (response.choices[0].message.content or "").strip().strip('"')

    async def search(*, query, exclude_urls):
        seen = set(exclude_urls)
        raw = await tavily.search(
            query, max_results=SEARCH_RESULTS, include_raw_content=True, topic="general"
        )
        documents = []
        for item in raw.get("results", []) or []:
            if item.get("url") in seen:
                continue
            body = select_relevant_text(
                item.get("raw_content") or item.get("content") or "",
                focus=query,
                max_chars=MAX_CHARS,
            )
            if not body.strip():
                continue
            documents.append(
                tavily_result_to_source_document(
                    item, topic="general", content=body,
                    retrieved_at=datetime.now(timezone.utc),
                )
            )
        return documents

    async def extract(*, document, slot):
        response = await llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question: {slot.question}\n\nPassage:\n{document.content}"
                    ),
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        rows = parse_triple_payload(response.choices[0].message.content)
        triples = []
        for row in rows:
            subject, predicate, obj = row["subject"], row["predicate"], row["object"]
            quote = f"{subject} {predicate} {obj}"
            triples.append(
                ExtractedTriple(
                    slot_id=slot.slot_id,
                    subject=EntityRef(name=subject),
                    predicate=predicate,
                    object=obj,
                    confidence=0.8,
                    source_document_id=document.document_id,
                    source_span=SourceSpan(start_char=0, end_char=len(quote), quote=quote),
                )
            )
        return triples

    # ---- loop -----------------------------------------------------------
    previous_coverage = 0.0
    no_improvement = 0
    history: list[tuple[int, str, bool, float]] = []

    for round_number in range(1, max_rounds + 1):
        statuses = await get_gap_status(
            graphiti, research_id=research_id, schema=INVESTIGATION_SCHEMA
        )
        coverage = coverage_ratio_from_gaps(statuses)
        filled = {s.slot_id for s in statuses if s.filled}
        open_slots = [s for s in all_slots if s.slot_id not in filled]

        decision = evaluate_stop(
            round_number=round_number,
            coverage_ratio=coverage,
            rounds_without_improvement=no_improvement,
            open_slot_count=len(open_slots),
            exhausted_slot_count=memory.exhausted_count(
                [s.slot_id for s in open_slots], settings.max_attempts_per_slot
            ),
            untried_slot_count=sum(1 for s in open_slots if s.slot_id not in tried),
            config=settings,
        )
        if decision.should_stop:
            print(f"\n■ 停机: {decision.reason.value} — {decision.detail}\n")
            break

        plan = await plan_next_round(
            topic, open_slots, memory, generate_query,
            max_attempts_per_slot=settings.max_attempts_per_slot,
        )
        if plan is None:
            print("\n■ 停机: 所有开放槽位都已用尽尝试预算\n")
            break

        slot, query, exclude_urls = plan
        tried.add(slot.slot_id)
        print(f"── 第 {round_number} 轮 ── 覆盖 {coverage:.0%} ({len(filled)}/{len(all_slots)})"
              f"  开放 {len(open_slots)}")
        print(f"   🎯 [{slot.dimension}] {slot.label}")
        print(f"   🔎 {query}")

        result = await run_research_round(
            graphiti,
            topic=topic,
            research_id=research_id,
            slot=slot,
            query=query,
            search=search,
            extract=extract,
            exclude_urls=exclude_urls,
            group_id=GROUP_ID,
        )
        memory.record_attempt(slot.slot_id, query=query, urls=result.documents_seen)
        if result.succeeded:
            memory.record_success(slot.slot_id)
            print(f"   ✅ {result.facts_written} 条事实入图"
                  f"（{len(result.documents_seen)} 篇文档）")
        else:
            memory.record_failure(slot.slot_id)
            print(f"   ⚠️  {result.note}")

        new_statuses = await get_gap_status(
            graphiti, research_id=research_id, schema=INVESTIGATION_SCHEMA
        )
        new_coverage = coverage_ratio_from_gaps(new_statuses)
        no_improvement = count_improvement(previous_coverage, new_coverage, no_improvement)
        previous_coverage = new_coverage
        history.append((round_number, slot.slot_id, result.succeeded, new_coverage))
        print()

    # ---- report ---------------------------------------------------------
    statuses = await get_gap_status(
        graphiti, research_id=research_id, schema=INVESTIGATION_SCHEMA
    )
    coverage = coverage_ratio_from_gaps(statuses)
    filled = [s for s in statuses if s.filled]

    print(f"{'=' * 66}")
    print(f"结束 — 覆盖率 {coverage:.0%} ({len(filled)}/{len(all_slots)} 槽位)")
    print(f"轮次 {len(history)}，成功 {sum(1 for r in history if r[2])}")
    print(f"{'=' * 66}\n")

    facts = await fetch_facts(graphiti, research_id=research_id)
    pack = await build_evidence_pack(
        graphiti, research_id=research_id, topic=topic, schema=INVESTIGATION_SCHEMA
    )
    report = render_report(
        pack, schema=INVESTIGATION_SCHEMA, sources=build_source_index(facts)
    )
    print(f"证据包：{len(pack.items)} 条事实，"
          f"{len(pack.provenance)} 个来源 episode，"
          f"{len(pack.unresolved_conflicts)} 处未消解冲突")

    path = _BASE / f"graphrag_report_{research_id}.md"
    path.write_text(report, encoding="utf-8")
    print(f"报告已保存: {path}")

    await graphiti.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("topic")
    parser.add_argument("--max-rounds", type=int, default=24)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.topic, args.max_rounds)))
