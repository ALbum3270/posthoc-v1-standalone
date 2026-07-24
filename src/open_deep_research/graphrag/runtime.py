"""Reusable live runtime for graph-driven research.

The original end-to-end wiring lived in ``scripts/run_graphrag_research.py``.
That proved the architecture, but leaving orchestration in a script would make
the LangGraph entry point and regression runner grow separate implementations.
This module is the single runtime used by both.

Network clients are injected into :class:`GraphResearchRunner`, keeping the
control loop unit-testable.  :func:`run_live_graph_research` is the convenience
boundary that creates and closes the real OpenAI, Tavily, and Graphiti clients.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from open_deep_research.graphrag.adapters.content import select_relevant_text
from open_deep_research.graphrag.adapters.search_results import (
    tavily_result_to_source_document,
)
from open_deep_research.graphrag.control.researcher import run_research_round
from open_deep_research.graphrag.control.stopping import (
    StopDecision,
    StopReason,
    StoppingConfig,
    count_improvement,
    evaluate_stop,
)
from open_deep_research.graphrag.control.supervisor import (
    SupervisorMemory,
    plan_next_round,
)
from open_deep_research.graphrag.extraction.triple_payload import parse_triple_payload
from open_deep_research.graphrag.graph.queries import (
    coverage_ratio_from_gaps,
    get_gap_status,
)
from open_deep_research.graphrag.ontology import (
    INVESTIGATION_SCHEMA,
    OntologySlot,
    iter_slots,
)
from open_deep_research.graphrag.reporting.evidence_pack import (
    build_evidence_pack,
    fetch_facts,
)
from open_deep_research.graphrag.reporting.report import (
    build_source_index,
    render_report,
)
from open_deep_research.graphrag.schemas import EvidencePack, GapStatus
from open_deep_research.graphrag.validation.grounding import ground_extracted_row

ProgressFn = Callable[[str], None]

QUERY_SYSTEM_PROMPT = (
    "You write one web search query. Output the query text only -- no quotes, no "
    "explanation. Prefer concrete nouns, names, numbers, dates, official records, "
    "and primary sources over generic phrasing."
)

EXTRACTION_SYSTEM_PROMPT = (
    "Extract factual triples from the passage that directly answer the question.\n"
    "Reply with a JSON object of exactly this form:\n"
    '{"triples": [{"subject": "...", "predicate": "...", "object": "...", '
    '"quote": "an exact contiguous quote copied from the passage"}]}\n'
    "The quote is mandatory and must occur verbatim in the passage. Copy names, "
    "numbers and dates exactly. Never infer, translate, summarize, or complete a "
    "date the passage does not state. If no exact supporting quote exists, return "
    '{"triples": []}.'
)


class GraphResearchSettings(BaseModel):
    """Runtime knobs that are safe to serialize and report."""

    model_config = ConfigDict(extra="forbid")

    model: str = "openai/gpt-4.1-mini"
    group_id: str = "neo4j"
    max_rounds: int = Field(default=24, ge=1)
    coverage_target: float = Field(default=1.0, ge=0.0, le=1.0)
    max_no_improvement_rounds: int = Field(default=4, ge=1)
    max_attempts_per_slot: int = Field(default=3, ge=1)
    max_chars_per_document: int = Field(default=2400, ge=500)
    search_results: int = Field(default=5, ge=1, le=20)
    max_documents_per_round: int = Field(default=3, ge=1, le=10)


class GraphResearchUsage(BaseModel):
    """Observable API and rejection counters for one run.

    ``chat_provider_cost_usd`` is populated only when the OpenAI-compatible
    provider includes a cost field in its response.  Embedding calls happen
    inside Graphiti and are therefore reported separately as unmetered here,
    rather than folded into a misleading estimated total.
    """

    model_config = ConfigDict(extra="forbid")

    llm_calls: int = 0
    search_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    chat_provider_cost_usd: float = 0.0
    provider_cost_reported: bool = False
    extraction_rows: int = 0
    grounding_rejections: int = 0
    elapsed_seconds: float = 0.0

    def observe_llm_response(self, response: Any) -> None:
        """Accumulate token and optional provider-cost metadata."""

        self.llm_calls += 1
        usage = getattr(response, "usage", None)
        if usage is None:
            return

        if hasattr(usage, "model_dump"):
            data = usage.model_dump()
        elif isinstance(usage, dict):
            data = dict(usage)
        else:
            data = {
                key: getattr(usage, key, None)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
            }

        prompt = data.get("prompt_tokens") or data.get("input_tokens") or 0
        completion = data.get("completion_tokens") or data.get("output_tokens") or 0
        total = data.get("total_tokens") or (prompt + completion)
        self.prompt_tokens += int(prompt or 0)
        self.completion_tokens += int(completion or 0)
        self.total_tokens += int(total or 0)

        cost = data.get("cost")
        if cost is None:
            details = data.get("cost_details") or {}
            cost = details.get("total_cost") or details.get("upstream_inference_cost")
        if cost is not None:
            self.chat_provider_cost_usd += float(cost)
            self.provider_cost_reported = True


class ResearchRoundTrace(BaseModel):
    """Compact, serializable audit record for one attempted slot."""

    model_config = ConfigDict(extra="forbid")

    round_number: int
    slot_id: str
    query: str
    documents_seen: list[str] = Field(default_factory=list)
    succeeded: bool
    facts_written: int
    coverage_after: float


class GraphResearchResult(BaseModel):
    """Complete result returned to the CLI, evaluator, and LangGraph node."""

    model_config = ConfigDict(extra="forbid")

    topic: str
    research_id: str
    stop_reason: str
    stop_detail: str
    coverage_ratio: float
    gap_status: list[GapStatus] = Field(default_factory=list)
    evidence_pack: EvidencePack
    report: str
    rounds: list[ResearchRoundTrace] = Field(default_factory=list)
    fact_count: int = 0
    source_count: int = 0
    dated_fact_count: int = 0
    usage: GraphResearchUsage = Field(default_factory=GraphResearchUsage)

    @property
    def successful_rounds(self) -> int:
        return sum(1 for round_trace in self.rounds if round_trace.succeeded)


@dataclass(frozen=True)
class LiveServiceConfig:
    """Credentials and endpoints kept out of serialized settings/results."""

    openai_api_key: str
    tavily_api_key: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    openai_base_url: str | None = None

    @classmethod
    def from_environment(cls) -> "LiveServiceConfig":
        """Load live-service configuration without logging secret values."""

        required = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "TAVILY_API_KEY": os.environ.get("TAVILY_API_KEY"),
            "NEO4J_URI": os.environ.get("NEO4J_URI"),
            "NEO4J_PASSWORD": os.environ.get("NEO4J_PASSWORD"),
        }
        missing = sorted(key for key, value in required.items() if not value)
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
        return cls(
            openai_api_key=required["OPENAI_API_KEY"] or "",
            tavily_api_key=required["TAVILY_API_KEY"] or "",
            neo4j_uri=required["NEO4J_URI"] or "",
            neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
            neo4j_password=required["NEO4J_PASSWORD"] or "",
            openai_base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )


class GraphResearchRunner:
    """Run the graph-first research loop against injected clients."""

    def __init__(
        self,
        *,
        graphiti: Any,
        llm: Any,
        tavily: Any,
        settings: GraphResearchSettings | None = None,
        progress: ProgressFn | None = None,
    ) -> None:
        self.graphiti = graphiti
        self.llm = llm
        self.tavily = tavily
        self.settings = settings or GraphResearchSettings()
        self.progress = progress
        self.usage = GraphResearchUsage()

    def emit(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    async def generate_query(
        self,
        *,
        topic: str,
        slot: OntologySlot,
        previous_queries: list[str],
    ) -> str:
        """Generate and deterministically de-duplicate one search query."""

        avoid = ""
        if previous_queries:
            avoid = (
                "\nAlready tried; produce a materially different query: "
                + "; ".join(previous_queries)
            )
        response = await self.llm.chat.completions.create(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Topic: {topic}\nQuestion: {slot.question}{avoid}",
                },
            ],
            temperature=0.3 if previous_queries else 0.0,
        )
        self.usage.observe_llm_response(response)
        query = (response.choices[0].message.content or "").strip().strip('"')
        if not query:
            query = f"{topic} {slot.question}"

        previous = {item.strip().casefold() for item in previous_queries}
        if query.casefold() in previous:
            query = f"{query} primary source evidence attempt {len(previous_queries) + 1}"
        return query

    async def search(self, *, query: str, exclude_urls: list[str]) -> list[Any]:
        """Search and return focused, normalized source documents."""

        self.usage.search_calls += 1
        response = await self.tavily.search(
            query,
            max_results=self.settings.search_results,
            include_raw_content="text",
            topic="general",
        )
        excluded = {url.rstrip("/") for url in exclude_urls}
        documents = []
        retrieved_at = datetime.now(timezone.utc)
        for item in response.get("results", []) or []:
            url = str(item.get("url") or "")
            if not url or url.rstrip("/") in excluded:
                continue
            body = select_relevant_text(
                item.get("raw_content") or item.get("content") or "",
                focus=query,
                max_chars=self.settings.max_chars_per_document,
            )
            if not body.strip():
                continue
            documents.append(
                tavily_result_to_source_document(
                    item,
                    topic="general",
                    content=body,
                    retrieved_at=retrieved_at,
                )
            )
            excluded.add(url.rstrip("/"))
        return documents

    async def extract(self, *, document: Any, slot: OntologySlot) -> list[Any]:
        """Extract only triples carrying a quote found in the source document."""

        response = await self.llm.chat.completions.create(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {slot.question}\n\nPassage:\n{document.content}",
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        self.usage.observe_llm_response(response)
        rows = parse_triple_payload(response.choices[0].message.content)
        self.usage.extraction_rows += len(rows)

        triples = []
        for row in rows:
            triple = ground_extracted_row(document=document, slot=slot, row=row)
            if triple is None:
                self.usage.grounding_rejections += 1
                continue
            triples.append(triple)
        return triples

    async def run(
        self,
        topic: str,
        *,
        research_id: str | None = None,
    ) -> GraphResearchResult:
        """Execute the loop and build a report from its persisted evidence pack."""

        started = time.perf_counter()
        run_id = research_id or f"run-{uuid4().hex[:8]}"
        settings = StoppingConfig(
            coverage_target=self.settings.coverage_target,
            max_rounds=self.settings.max_rounds,
            max_no_improvement_rounds=self.settings.max_no_improvement_rounds,
            max_attempts_per_slot=self.settings.max_attempts_per_slot,
        )
        memory = SupervisorMemory()
        slots = iter_slots(INVESTIGATION_SCHEMA)
        tried: set[str] = set()
        previous_coverage = 0.0
        rounds_without_improvement = 0
        traces: list[ResearchRoundTrace] = []
        stop_decision: StopDecision | None = None

        self.emit(
            f"research={run_id} topic={topic!r} slots={len(slots)} "
            f"max_rounds={settings.max_rounds}"
        )

        for round_number in range(1, settings.max_rounds + 1):
            statuses = await get_gap_status(
                self.graphiti,
                research_id=run_id,
                schema=INVESTIGATION_SCHEMA,
            )
            coverage = coverage_ratio_from_gaps(statuses)
            filled = {status.slot_id for status in statuses if status.filled}
            open_slots = [slot for slot in slots if slot.slot_id not in filled]

            decision = evaluate_stop(
                round_number=round_number,
                coverage_ratio=coverage,
                rounds_without_improvement=rounds_without_improvement,
                open_slot_count=len(open_slots),
                exhausted_slot_count=memory.exhausted_count(
                    [slot.slot_id for slot in open_slots],
                    settings.max_attempts_per_slot,
                ),
                untried_slot_count=sum(
                    1 for slot in open_slots if slot.slot_id not in tried
                ),
                config=settings,
            )
            if decision.should_stop:
                stop_decision = decision
                break

            plan = await plan_next_round(
                topic,
                open_slots,
                memory,
                self.generate_query,
                max_attempts_per_slot=settings.max_attempts_per_slot,
            )
            if plan is None:
                stop_decision = StopDecision(
                    should_stop=True,
                    reason=StopReason.ALL_SLOTS_EXHAUSTED,
                    detail="all open slots exhausted their attempt budget",
                )
                break

            slot, query, exclude_urls = plan
            tried.add(slot.slot_id)
            self.emit(
                f"round={round_number} coverage={coverage:.0%} "
                f"slot={slot.slot_id} query={query!r}"
            )
            round_result = await run_research_round(
                self.graphiti,
                topic=topic,
                research_id=run_id,
                slot=slot,
                query=query,
                search=self.search,
                extract=self.extract,
                exclude_urls=exclude_urls,
                max_documents=self.settings.max_documents_per_round,
                group_id=self.settings.group_id,
            )
            memory.record_attempt(
                slot.slot_id,
                query=query,
                urls=round_result.documents_seen,
            )
            if round_result.succeeded:
                memory.record_success(slot.slot_id)
            else:
                memory.record_failure(slot.slot_id)

            new_statuses = await get_gap_status(
                self.graphiti,
                research_id=run_id,
                schema=INVESTIGATION_SCHEMA,
            )
            new_coverage = coverage_ratio_from_gaps(new_statuses)
            rounds_without_improvement = count_improvement(
                previous_coverage,
                new_coverage,
                rounds_without_improvement,
            )
            previous_coverage = new_coverage
            traces.append(
                ResearchRoundTrace(
                    round_number=round_number,
                    slot_id=slot.slot_id,
                    query=query,
                    documents_seen=round_result.documents_seen,
                    succeeded=round_result.succeeded,
                    facts_written=round_result.facts_written,
                    coverage_after=new_coverage,
                )
            )
            self.emit(
                f"round={round_number} success={round_result.succeeded} "
                f"facts={round_result.facts_written} coverage_after={new_coverage:.0%}"
            )

        if stop_decision is None:
            stop_decision = StopDecision(
                should_stop=True,
                reason=StopReason.MAX_ROUNDS,
                detail=f"reached cap of {settings.max_rounds} round(s)",
            )

        statuses = await get_gap_status(
            self.graphiti,
            research_id=run_id,
            schema=INVESTIGATION_SCHEMA,
        )
        coverage = coverage_ratio_from_gaps(statuses)
        facts = await fetch_facts(self.graphiti, research_id=run_id)
        evidence_pack = await build_evidence_pack(
            self.graphiti,
            research_id=run_id,
            topic=topic,
            schema=INVESTIGATION_SCHEMA,
        )
        report = render_report(
            evidence_pack,
            schema=INVESTIGATION_SCHEMA,
            sources=build_source_index(facts),
        )
        self.usage.elapsed_seconds = time.perf_counter() - started
        source_count = len({fact.source_url for fact in facts if fact.source_url})
        dated_fact_count = sum(1 for fact in facts if fact.valid_at is not None)

        self.emit(
            f"stop={stop_decision.reason.value} coverage={coverage:.0%} "
            f"facts={len(facts)} sources={source_count}"
        )
        return GraphResearchResult(
            topic=topic,
            research_id=run_id,
            stop_reason=stop_decision.reason.value,
            stop_detail=stop_decision.detail,
            coverage_ratio=coverage,
            gap_status=statuses,
            evidence_pack=evidence_pack,
            report=report,
            rounds=traces,
            fact_count=len(facts),
            source_count=source_count,
            dated_fact_count=dated_fact_count,
            usage=self.usage,
        )


async def run_live_graph_research(
    topic: str,
    *,
    settings: GraphResearchSettings | None = None,
    services: LiveServiceConfig | None = None,
    research_id: str | None = None,
    progress: ProgressFn | None = None,
) -> GraphResearchResult:
    """Create real clients, run one investigation, and close every owned client."""

    from graphiti_core import Graphiti
    from openai import AsyncOpenAI
    from tavily import AsyncTavilyClient

    active_services = services or LiveServiceConfig.from_environment()
    active_settings = settings or GraphResearchSettings(
        model=os.environ.get("OPENAI_MODEL", "openai/gpt-4.1-mini")
    )
    llm = AsyncOpenAI(
        api_key=active_services.openai_api_key,
        base_url=active_services.openai_base_url,
    )
    tavily = AsyncTavilyClient(api_key=active_services.tavily_api_key)
    graphiti = Graphiti(
        uri=active_services.neo4j_uri,
        user=active_services.neo4j_user,
        password=active_services.neo4j_password,
    )
    runner = GraphResearchRunner(
        graphiti=graphiti,
        llm=llm,
        tavily=tavily,
        settings=active_settings,
        progress=progress,
    )
    try:
        return await runner.run(topic, research_id=research_id)
    finally:
        await graphiti.close()
        close = getattr(llm, "close", None)
        if close is not None:
            await close()
