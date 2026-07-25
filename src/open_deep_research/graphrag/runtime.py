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

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from open_deep_research.graphrag.adapters.content import (
    query_terms,
    select_relevant_text,
)
from open_deep_research.graphrag.adapters.search_results import (
    tavily_result_to_source_document,
)
from open_deep_research.graphrag.adapters.tavily import bounded_tavily_query
from open_deep_research.graphrag.control.researcher import (
    run_research_round,
    run_support_round,
)
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
from open_deep_research.graphrag.schemas import (
    ClaimMatchAuditRecord,
    EntityRef,
    EvidencePack,
    ExtractedTriple,
    GapStatus,
    RelevanceStatus,
    SlotApplicability,
    SlotApplicabilityStatus,
)
from open_deep_research.graphrag.validation.grounding import ground_extracted_row
from open_deep_research.graphrag.validation.sources import publisher_identity

ProgressFn = Callable[[str], None]

QUERY_SYSTEM_PROMPT = (
    "You write one web search query. Output the query text only, with no "
    "explanation. Keep full proper names intact and put quotation marks around "
    "ambiguous multi-word names. Put distinct people and organization names in "
    "separate quoted phrases; never concatenate them into one name. Include the "
    "organization or event anchor. "
    "Prefer concrete nouns, names, numbers, dates, official records, and primary "
    "sources over generic phrasing."
)

SUPPORT_QUERY_SYSTEM_PROMPT = (
    "Write one web search query for an independent source supporting the exact "
    "structured claim supplied by the user. Output query text only, with no "
    "explanation. Keep full entity names and distinctive numbers or dates. Use "
    "quotation marks around ambiguous multi-word names and include the topic "
    "anchor."
)

EXTRACTION_SYSTEM_PROMPT = (
    "Extract factual triples from the passage that directly answer the question.\n"
    "Reply with a JSON object of exactly this form:\n"
    '{"triples": [{"subject": "...", "predicate": "...", "object": "...", '
    '"quote": "an exact contiguous quote copied from the passage"}]}\n'
    "The quote is mandatory and must occur verbatim in the passage. Use a "
    "complete, self-contained sentence or clause and never end mid-word. Copy "
    "names, numbers and dates exactly. Never infer, translate, summarize, or "
    "complete a date the passage does not state. If no exact supporting quote "
    "exists, return "
    '{"triples": []}.'
)

SUPPORT_EXTRACTION_SYSTEM_PROMPT = (
    "Find independent source text that supports one or more target claims.\n"
    "Return only claims that this passage directly supports. Copy subject, "
    "predicate, and object EXACTLY from a target claim; do not introduce a new "
    "claim merely because it answers the same broad question. Every row must "
    "include an exact contiguous quote from this passage that is a complete, "
    "self-contained sentence or clause and never ends mid-word.\n"
    'Reply as a JSON object: {"triples":[{"subject":"...","predicate":"...",'
    '"object":"...",'
    '"quote":"..."}]}. If none of the target claims is supported, return '
    '{"triples":[]}.'
)

RELEVANCE_SYSTEM_PROMPT = (
    "Act as a conservative pre-write evidence gate. For each candidate, judge "
    "whether its exact quote directly answers the assigned question in the "
    "context of the research topic. A real quote can still be irrelevant. "
    "Reject entity-name accidents (for example the chemical element Silicon in "
    "a Silicon Valley Bank investigation), bibliography metadata, navigation, "
    "and facts that merely mention topic words. A primary actor must be the "
    "event's central subject or responsible actor; reject later buyers, "
    "responders, regulators, and merely affected parties in that slot. "
    "Motivation requires intentional purpose; asset flow requires an actual "
    "movement of money/assets/activity. "
    "For support candidates, also require the quote to support the exact target "
    "claim. Use uncertain when the relationship is genuinely ambiguous. Return "
    'JSON exactly as {"decisions":[{"index":0,"status":"accepted|rejected|'
    'uncertain","confidence":0.0,"reason":"short explanation"}]}.'
)

APPLICABILITY_SYSTEM_PROMPT = (
    "Decide whether each conditional research question has a meaningful answer "
    "for this topic. Use not_applicable only when the concept itself does not "
    "apply, not merely because evidence may be hard to find. An accidental IT "
    "outage normally has no actor motivation or asset flow; a financial collapse "
    "can. Otherwise use optional. Return JSON exactly as "
    '{"slots":[{"slot_id":"...","status":"optional|not_applicable",'
    '"confidence":0.0,"reason":"short explanation"}]}.'
)

_CAPITALIZED_QUERY_RUN = re.compile(
    r"\b[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*)+\b"
)
_QUOTED_QUERY_PHRASE = re.compile(r'"([^"]+)"')
_SEARCH_TEXT_PUNCTUATION = re.compile(r"[^\w]+")


def _normalize_search_text(text: str) -> str:
    """Normalize punctuation while retaining letters from every script."""

    return _SEARCH_TEXT_PUNCTUATION.sub(
        " ",
        (text or "").casefold().replace("_", " "),
    ).strip()


def _looks_like_quoted_entity(phrase: str) -> bool:
    """Recognize explicitly quoted names without treating quoted claims as names."""

    has_latin_proper_name = re.search(r"\b[A-Z][A-Za-z'-]*", phrase) is not None
    has_non_latin_letter = any(
        character.isalpha() and not character.isascii()
        for character in phrase
    )
    return has_latin_proper_name or has_non_latin_letter


def _query_entity_anchors(query: str) -> list[str]:
    """Extract multi-token proper names that search results must preserve."""

    anchors: list[str] = []
    for match in _QUOTED_QUERY_PHRASE.finditer(query or ""):
        phrase = match.group(1)
        if not _looks_like_quoted_entity(phrase):
            continue
        normalized = _normalize_search_text(phrase)
        if normalized and normalized not in anchors:
            anchors.append(normalized)

    # Keep a conservative fallback for older, unquoted queries. A longer
    # title-cased run may be multiple adjacent entities; inventing a prefix or
    # sliding-window anchor would reject valid results or weaken a long name.
    unquoted_query = _QUOTED_QUERY_PHRASE.sub(" ", query or "")
    for match in _CAPITALIZED_QUERY_RUN.finditer(unquoted_query):
        words = match.group(0).split()
        if len(words) > 3:
            continue
        normalized = _normalize_search_text(match.group(0))
        if normalized and normalized not in anchors:
            anchors.append(normalized)
    return anchors


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
    min_sources_per_claim: int = Field(default=2, ge=1, le=5)
    enable_relevance_gate: bool = True
    relevance_reject_threshold: float = Field(default=0.8, ge=0.5, le=1.0)
    enable_slot_applicability: bool = True
    applicability_not_applicable_threshold: float = Field(
        default=0.8,
        ge=0.5,
        le=1.0,
    )
    claim_match_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
    )
    claim_match_top_k: int = Field(default=3, ge=1, le=3)


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
    search_results_rejected: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    chat_provider_cost_usd: float = 0.0
    provider_cost_reported: bool = False
    extraction_rows: int = 0
    grounding_rejections: int = 0
    relevance_accepted: int = 0
    relevance_uncertain: int = 0
    relevance_rejected: int = 0
    support_rows_rejected: int = 0
    not_applicable_slots: int = 0
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
    purpose: str = "coverage"
    query: str
    documents_seen: list[str] = Field(default_factory=list)
    succeeded: bool
    facts_written: int
    supports_added: int = 0
    contributing_sources: list[str] = Field(default_factory=list)
    contributing_source_identities: list[str] = Field(default_factory=list)
    target_edge_uuids: list[str] = Field(default_factory=list)
    corroborated_edge_uuids: list[str] = Field(default_factory=list)
    claim_match_audit: list[ClaimMatchAuditRecord] = Field(default_factory=list)
    coverage_after: float


class RelevanceAuditRecord(BaseModel):
    """One serialized pre-write relevance decision."""

    model_config = ConfigDict(extra="forbid")

    slot_id: str
    document_id: str
    source_url: str | None = None
    quote: str
    structured_claim: str
    purpose: str
    status: RelevanceStatus
    confidence: float
    reason: str


class GraphResearchResult(BaseModel):
    """Complete result returned to the CLI, evaluator, and LangGraph node."""

    model_config = ConfigDict(extra="forbid")

    topic: str
    research_id: str
    stop_reason: str
    stop_detail: str
    coverage_ratio: float
    gap_status: list[GapStatus] = Field(default_factory=list)
    slot_applicability: list[SlotApplicability] = Field(default_factory=list)
    relevance_audit: list[RelevanceAuditRecord] = Field(default_factory=list)
    claim_match_audit: list[ClaimMatchAuditRecord] = Field(default_factory=list)
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


def _finalize_loop_stop(
    decision: StopDecision | None,
    *,
    coverage_ratio: float,
    settings: StoppingConfig,
) -> StopDecision:
    """Label a loop that consumed its final allowed round truthfully."""

    if decision is not None:
        return decision
    if coverage_ratio >= settings.coverage_target:
        return StopDecision(
            should_stop=True,
            reason=StopReason.COVERAGE_REACHED,
            detail=(
                f"coverage {coverage_ratio:.0%} >= target "
                f"{settings.coverage_target:.0%}"
            ),
        )
    return StopDecision(
        should_stop=True,
        reason=StopReason.MAX_ROUNDS,
        detail=f"reached cap of {settings.max_rounds} round(s)",
    )


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
        self.relevance_audit: list[RelevanceAuditRecord] = []

    def emit(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    async def classify_slot_applicability(
        self,
        *,
        topic: str,
        slots: list[OntologySlot],
    ) -> dict[str, SlotApplicability]:
        """Classify only conditional slots; ambiguous decisions stay applicable."""

        decisions = {
            slot.slot_id: SlotApplicability(
                slot_id=slot.slot_id,
                status=(
                    SlotApplicabilityStatus.REQUIRED
                    if slot.applicability == "always"
                    else SlotApplicabilityStatus.OPTIONAL
                ),
                confidence=1.0 if slot.applicability == "always" else 0.0,
                reason=(
                    "core ontology slot"
                    if slot.applicability == "always"
                    else "conditional slot; no classifier decision"
                ),
            )
            for slot in slots
        }
        conditional = [slot for slot in slots if slot.applicability == "conditional"]
        if not self.settings.enable_slot_applicability or not conditional:
            return decisions

        response = await self.llm.chat.completions.create(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": APPLICABILITY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "topic": topic,
                            "conditional_slots": [
                                {
                                    "slot_id": slot.slot_id,
                                    "question": slot.question,
                                }
                                for slot in conditional
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        self.usage.observe_llm_response(response)
        try:
            payload = json.loads(response.choices[0].message.content or "{}")
        except (json.JSONDecodeError, TypeError):
            payload = {}
        allowed = {slot.slot_id for slot in conditional}
        for row in payload.get("slots", []) or []:
            slot_id = str(row.get("slot_id") or "")
            if slot_id not in allowed:
                continue
            try:
                confidence = min(max(float(row.get("confidence", 0.0)), 0.0), 1.0)
            except (TypeError, ValueError):
                confidence = 0.0
            requested = str(row.get("status") or "").casefold()
            if (
                requested == SlotApplicabilityStatus.NOT_APPLICABLE.value
                and confidence
                >= self.settings.applicability_not_applicable_threshold
            ):
                status = SlotApplicabilityStatus.NOT_APPLICABLE
            else:
                status = SlotApplicabilityStatus.OPTIONAL
            decisions[slot_id] = SlotApplicability(
                slot_id=slot_id,
                status=status,
                confidence=confidence,
                reason=str(row.get("reason") or "").strip(),
            )

        self.usage.not_applicable_slots = sum(
            decision.status is SlotApplicabilityStatus.NOT_APPLICABLE
            for decision in decisions.values()
        )
        return decisions

    @staticmethod
    def _triple_key(triple: ExtractedTriple) -> tuple[str, str, str]:
        obj = (
            triple.object
            if isinstance(triple.object, str)
            else triple.object.name
        )
        return tuple(
            " ".join(str(value).casefold().split())
            for value in (triple.subject.name, triple.predicate, obj)
        )

    @staticmethod
    def _triple_text(triple: ExtractedTriple) -> str:
        obj = (
            triple.object
            if isinstance(triple.object, str)
            else triple.object.name
        )
        return f"{triple.subject.name} | {triple.predicate} | {obj}"

    async def assess_relevance(
        self,
        *,
        topic: str,
        document: Any,
        slot: OntologySlot,
        triples: list[ExtractedTriple],
        purpose: str,
    ) -> list[ExtractedTriple]:
        """Apply a conservative, auditable semantic gate to grounded triples."""

        if not triples or not self.settings.enable_relevance_gate:
            return triples

        candidates = []
        for index, triple in enumerate(triples):
            quote = (
                triple.source_span.quote
                if triple.source_span is not None
                else ""
            ) or ""
            candidates.append(
                {
                    "index": index,
                    "structured_claim": self._triple_text(triple),
                    "quote": quote,
                    "purpose": purpose,
                }
            )
        response = await self.llm.chat.completions.create(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": RELEVANCE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "topic": topic,
                            "slot_id": slot.slot_id,
                            "question": slot.question,
                            "candidates": candidates,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        self.usage.observe_llm_response(response)
        try:
            payload = json.loads(response.choices[0].message.content or "{}")
        except (json.JSONDecodeError, TypeError):
            payload = {}

        raw_decisions: dict[int, dict[str, Any]] = {}
        for row in payload.get("decisions", []) or []:
            try:
                index = int(row.get("index"))
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(triples):
                raw_decisions[index] = row

        accepted: list[ExtractedTriple] = []
        for index, triple in enumerate(triples):
            row = raw_decisions.get(index, {})
            requested = str(row.get("status") or "uncertain").casefold()
            aliases = {
                "accept": RelevanceStatus.ACCEPTED,
                "accepted": RelevanceStatus.ACCEPTED,
                "reject": RelevanceStatus.REJECTED,
                "rejected": RelevanceStatus.REJECTED,
                "uncertain": RelevanceStatus.UNCERTAIN,
            }
            status = aliases.get(requested, RelevanceStatus.UNCERTAIN)
            try:
                confidence = min(max(float(row.get("confidence", 0.0)), 0.0), 1.0)
            except (TypeError, ValueError):
                confidence = 0.0
            reason = str(
                row.get("reason")
                or "relevance verifier returned no usable decision"
            ).strip()

            should_reject = (
                status is RelevanceStatus.REJECTED
                and confidence >= self.settings.relevance_reject_threshold
            )
            if status is RelevanceStatus.REJECTED and not should_reject:
                status = RelevanceStatus.UNCERTAIN
                reason = f"low-confidence rejection retained for review: {reason}"

            checked = triple.model_copy(
                update={
                    "relevance_status": status,
                    "relevance_confidence": confidence,
                    "relevance_reason": reason,
                }
            )
            quote = (
                checked.source_span.quote
                if checked.source_span is not None
                else ""
            ) or ""
            self.relevance_audit.append(
                RelevanceAuditRecord(
                    slot_id=slot.slot_id,
                    document_id=document.document_id,
                    source_url=document.url,
                    quote=quote,
                    structured_claim=self._triple_text(checked),
                    purpose=purpose,
                    status=status,
                    confidence=confidence,
                    reason=reason,
                )
            )

            if should_reject:
                self.usage.relevance_rejected += 1
                if purpose == "support":
                    self.usage.support_rows_rejected += 1
                continue
            if status is RelevanceStatus.ACCEPTED:
                self.usage.relevance_accepted += 1
            else:
                self.usage.relevance_uncertain += 1
            accepted.append(checked)
        return accepted

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
        search_guidance = ""
        if slot.slot_id == "who.primary_actor":
            search_guidance = (
                "\nFocus on identifying the event's central subject, not a "
                "later buyer, responder, regulator, or affected party."
            )
        elif slot.slot_id in {"why.trigger", "how.mechanism"}:
            search_guidance = (
                "\nSeek concrete evidence for both the proximate trigger and "
                "the longer-term underlying causal mechanism. Use specific "
                "candidate mechanism terms, not the generic phrase "
                "'underlying conditions'."
            )
        response = await self.llm.chat.completions.create(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Topic: {topic}\nQuestion: {slot.question}"
                        f"{search_guidance}{avoid}"
                    ),
                },
            ],
            temperature=0.3 if previous_queries else 0.0,
        )
        self.usage.observe_llm_response(response)
        query = (response.choices[0].message.content or "").strip()
        if not query:
            query = f"{topic} {slot.question}"

        previous = {item.strip().casefold() for item in previous_queries}
        if query.casefold() in previous:
            query = f"{query} primary source evidence attempt {len(previous_queries) + 1}"
        return query

    async def generate_support_query(
        self,
        *,
        topic: str,
        slot: OntologySlot,
        target: ExtractedTriple,
        previous_queries: list[str],
    ) -> str:
        """Generate a query aimed at one exact, already-persisted claim."""

        claim = self._triple_text(target)
        avoid = ""
        if previous_queries:
            avoid = (
                "\nAlready tried; produce a materially different query: "
                + "; ".join(previous_queries)
            )
        response = await self.llm.chat.completions.create(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": SUPPORT_QUERY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Topic: {topic}\nQuestion: {slot.question}\n"
                        f"Exact structured claim: {claim}{avoid}"
                    ),
                },
            ],
            temperature=0.2 if previous_queries else 0.0,
        )
        self.usage.observe_llm_response(response)
        query = (response.choices[0].message.content or "").strip()
        if not query:
            query = f"{topic} {claim}"
        previous = {item.strip().casefold() for item in previous_queries}
        if query.casefold() in previous:
            query = f"{query} independent confirmation"
        return query

    async def search(self, *, query: str, exclude_urls: list[str]) -> list[Any]:
        """Search and return focused, normalized source documents."""

        self.usage.search_calls += 1
        provider_query = bounded_tavily_query(query)
        response = await self.tavily.search(
            provider_query,
            max_results=self.settings.search_results,
            include_raw_content="text",
            topic="general",
        )
        excluded = {url.rstrip("/") for url in exclude_urls}
        documents = []
        retrieved_at = datetime.now(timezone.utc)
        terms = query_terms(provider_query)
        entity_anchors = _query_entity_anchors(provider_query)
        for item in response.get("results", []) or []:
            url = str(item.get("url") or "")
            if not url or url.rstrip("/") in excluded:
                continue
            raw_searchable = " ".join(
                str(item.get(field) or "")
                for field in ("title", "content", "raw_content")
            )
            searchable = raw_searchable.casefold()
            normalized_searchable = _normalize_search_text(raw_searchable)
            matched_terms = sum(term in searchable for term in terms)
            required_matches = 2 if len(terms) >= 3 else 1
            misses_topic_terms = terms and matched_terms < required_matches
            misses_entity_anchor = entity_anchors and not any(
                anchor in normalized_searchable for anchor in entity_anchors
            )
            if misses_topic_terms or misses_entity_anchor:
                # Tavily can occasionally resolve an ambiguous first name to a
                # dictionary entry (for example "Sam") even when the query is
                # about a full named person, or "Silicon Valley Bank" to the
                # chemical element Silicon. Do not spend extraction calls on a
                # result that misses the query's entity/topic anchors.
                self.usage.search_results_rejected += 1
                continue
            body = select_relevant_text(
                item.get("raw_content") or item.get("content") or "",
                focus=provider_query,
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

    async def extract(
        self,
        *,
        document: Any,
        slot: OntologySlot,
        topic: str = "",
    ) -> list[ExtractedTriple]:
        """Extract only triples carrying a quote found in the source document."""

        claim_limit = slot.max_initial_claims
        if claim_limit is None and slot.required_source_count > 1:
            claim_limit = 1
        claim_instruction = ""
        if claim_limit is not None:
            if slot.slot_id == "why.trigger" and claim_limit >= 2:
                claim_instruction = (
                    f"Return at most {claim_limit} central, atomic claims: one "
                    "for an underlying condition and one for the immediate "
                    "trigger when the passage states both.\n"
                )
            else:
                claim_instruction = (
                    f"Return at most {claim_limit} central, atomic claim"
                    f"{'s' if claim_limit > 1 else ''}. Choose the claim"
                    f"{'s' if claim_limit > 1 else ''} that most directly "
                    "answer the question.\n"
                )
        response = await self.llm.chat.completions.create(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question: {slot.question}\n"
                        + claim_instruction
                        + f"\nPassage:\n{document.content}"
                    ),
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
        accepted = await self.assess_relevance(
            topic=topic,
            document=document,
            slot=slot,
            triples=triples,
            purpose="initial",
        )
        # The prompt is the first line of defence; this deterministic cap keeps
        # a verbose provider response from turning one critical slot into six
        # different claims that each require a separate independent source.
        if claim_limit is not None:
            return accepted[:claim_limit]
        return accepted

    async def extract_support(
        self,
        *,
        document: Any,
        slot: OntologySlot,
        targets: list[ExtractedTriple],
        topic: str = "",
    ) -> list[ExtractedTriple]:
        """Extract evidence only for the exact structured claims already written."""

        target_rows = [
            {
                "subject": target.subject.name,
                "predicate": target.predicate,
                "object": (
                    target.object
                    if isinstance(target.object, str)
                    else target.object.name
                ),
            }
            for target in targets
        ]
        response = await self.llm.chat.completions.create(
            model=self.settings.model,
            messages=[
                {"role": "system", "content": SUPPORT_EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Topic: {topic}\nQuestion: {slot.question}\n"
                        f"Target claims: {json.dumps(target_rows, ensure_ascii=False)}"
                        f"\n\nPassage:\n{document.content}"
                    ),
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        self.usage.observe_llm_response(response)
        rows = parse_triple_payload(response.choices[0].message.content)
        self.usage.extraction_rows += len(rows)

        target_keys = {self._triple_key(target) for target in targets}
        grounded: list[ExtractedTriple] = []
        for row in rows:
            row_key = tuple(
                " ".join(str(row.get(key) or "").casefold().split())
                for key in ("subject", "predicate", "object")
            )
            if row_key not in target_keys:
                self.usage.support_rows_rejected += 1
                continue
            triple = ground_extracted_row(document=document, slot=slot, row=row)
            if triple is None:
                self.usage.grounding_rejections += 1
                continue
            grounded.append(triple)

        return await self.assess_relevance(
            topic=topic,
            document=document,
            slot=slot,
            triples=grounded,
            purpose="support",
        )

    async def run(
        self,
        topic: str,
        *,
        research_id: str | None = None,
    ) -> GraphResearchResult:
        """Execute the loop and build a report from its persisted evidence pack."""

        started = time.perf_counter()
        self.usage = GraphResearchUsage()
        self.relevance_audit = []
        run_id = research_id or f"run-{uuid4().hex[:8]}"
        settings = StoppingConfig(
            coverage_target=self.settings.coverage_target,
            max_rounds=self.settings.max_rounds,
            max_no_improvement_rounds=self.settings.max_no_improvement_rounds,
            max_attempts_per_slot=self.settings.max_attempts_per_slot,
        )
        memory = SupervisorMemory()
        slots = iter_slots(INVESTIGATION_SCHEMA)
        applicability = await self.classify_slot_applicability(
            topic=topic,
            slots=slots,
        )
        applicable_slots = [
            slot
            for slot in slots
            if applicability[slot.slot_id].status
            is not SlotApplicabilityStatus.NOT_APPLICABLE
        ]
        required_slots = [
            slot
            for slot in slots
            if applicability[slot.slot_id].status
            is SlotApplicabilityStatus.REQUIRED
        ]
        tried: set[str] = set()
        previous_coverage = 0.0
        rounds_without_improvement = 0
        traces: list[ResearchRoundTrace] = []
        stop_decision: StopDecision | None = None

        self.emit(
            f"research={run_id} topic={topic!r} slots={len(applicable_slots)} "
            f"required={len(required_slots)} "
            f"not_applicable={len(slots) - len(applicable_slots)} "
            f"max_rounds={settings.max_rounds}"
        )

        async def extract_initial(
            *,
            document: Any,
            slot: OntologySlot,
        ) -> list[ExtractedTriple]:
            return await self.extract(
                document=document,
                slot=slot,
                topic=topic,
            )

        async def extract_claim_support(
            *,
            document: Any,
            slot: OntologySlot,
            targets: list[ExtractedTriple],
        ) -> list[ExtractedTriple]:
            return await self.extract_support(
                document=document,
                slot=slot,
                targets=targets,
                topic=topic,
            )

        for round_number in range(1, settings.max_rounds + 1):
            statuses = await get_gap_status(
                self.graphiti,
                research_id=run_id,
                schema=INVESTIGATION_SCHEMA,
                applicability=applicability,
            )
            coverage = coverage_ratio_from_gaps(statuses)
            filled = {status.slot_id for status in statuses if status.filled}
            open_slots = [
                slot for slot in required_slots if slot.slot_id not in filled
            ]

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
                extract=extract_initial,
                extract_support=extract_claim_support,
                exclude_urls=exclude_urls,
                max_documents=self.settings.max_documents_per_round,
                min_sources=min(
                    self.settings.min_sources_per_claim,
                    slot.required_source_count,
                ),
                group_id=self.settings.group_id,
                claim_match_similarity_threshold=(
                    self.settings.claim_match_similarity_threshold
                ),
                claim_match_top_k=self.settings.claim_match_top_k,
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
                applicability=applicability,
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
                    supports_added=round_result.supports_added,
                    contributing_sources=round_result.contributing_sources,
                    contributing_source_identities=(
                        round_result.contributing_source_identities
                    ),
                    target_edge_uuids=round_result.target_edge_uuids,
                    corroborated_edge_uuids=(
                        round_result.corroborated_edge_uuids
                    ),
                    claim_match_audit=round_result.claim_match_audit,
                    coverage_after=new_coverage,
                )
            )
            self.emit(
                f"round={round_number} success={round_result.succeeded} "
                f"facts={round_result.facts_written} "
                f"supports={round_result.supports_added} "
                f"coverage_after={new_coverage:.0%}"
            )

        # Coverage and support are deliberately separate. Once the required,
        # applicable slots are filled, use any remaining round budget to target
        # critical claims that still have only one publisher. A support round is
        # unable to create a new fact, so it cannot inflate coverage.
        pre_support_statuses = await get_gap_status(
            self.graphiti,
            research_id=run_id,
            schema=INVESTIGATION_SCHEMA,
            applicability=applicability,
        )
        pre_support_coverage = coverage_ratio_from_gaps(pre_support_statuses)
        stop_decision = _finalize_loop_stop(
            stop_decision,
            coverage_ratio=pre_support_coverage,
            settings=settings,
        )
        support_candidates: list[tuple[Any, OntologySlot, int]] = []
        if (
            pre_support_coverage >= settings.coverage_target
            and self.settings.min_sources_per_claim > 1
            and len(traces) < settings.max_rounds
        ):
            slot_by_id = {slot.slot_id: slot for slot in applicable_slots}
            current_facts = await fetch_facts(self.graphiti, research_id=run_id)
            for fact in current_facts:
                slot = slot_by_id.get(fact.slot_id)
                if slot is None:
                    continue
                required = min(
                    self.settings.min_sources_per_claim,
                    slot.required_source_count,
                )
                if (
                    required > 1
                    and len(fact.distinct_source_identities) < required
                ):
                    support_candidates.append((fact, slot, required))
            support_candidates.sort(
                key=lambda row: (-row[1].priority, row[1].slot_id, row[0].uuid)
            )

        support_successes = 0
        support_queue = list(support_candidates)
        support_attempts: dict[str, int] = {}
        support_queries: dict[str, list[str]] = {}
        support_seen_urls = {
            fact.uuid: set(fact.distinct_source_urls)
            for fact, _, _ in support_candidates
        }
        support_seen_identities = {
            fact.uuid: set(fact.distinct_source_identities)
            for fact, _, _ in support_candidates
        }
        max_targeted_support_attempts = max(
            settings.max_attempts_per_slot - 1,
            1,
        )
        while support_queue and len(traces) < settings.max_rounds:
            fact, slot, required = support_queue.pop(0)
            support_attempts[fact.uuid] = support_attempts.get(fact.uuid, 0) + 1
            target = ExtractedTriple(
                slot_id=fact.slot_id,
                subject=EntityRef(name=fact.subject),
                predicate=fact.predicate,
                object=EntityRef(name=fact.object),
                source_document_id=fact.source_url or fact.uuid,
            )
            previous_queries = [
                *memory.for_slot(slot.slot_id).queries,
                *support_queries.get(fact.uuid, []),
            ]
            query = await self.generate_support_query(
                topic=topic,
                slot=slot,
                target=target,
                previous_queries=list(previous_queries),
            )
            support_queries.setdefault(fact.uuid, []).append(query)
            round_number = len(traces) + 1
            self.emit(
                f"round={round_number} purpose=support slot={slot.slot_id} "
                f"sources={len(fact.distinct_source_identities)}/{required} "
                f"query={query!r}"
            )
            support_result = await run_support_round(
                self.graphiti,
                research_id=run_id,
                slot=slot,
                target=target,
                target_edge_uuid=fact.uuid,
                query=query,
                search=self.search,
                extract_support=extract_claim_support,
                exclude_urls=sorted(support_seen_urls[fact.uuid]),
                exclude_source_identities=sorted(
                    support_seen_identities[fact.uuid]
                ),
                max_documents=self.settings.max_documents_per_round,
                group_id=self.settings.group_id,
                claim_match_similarity_threshold=(
                    self.settings.claim_match_similarity_threshold
                ),
                claim_match_top_k=self.settings.claim_match_top_k,
            )
            if support_result.succeeded:
                support_successes += 1
            else:
                for url in support_result.documents_seen:
                    support_seen_urls[fact.uuid].add(url)
                    identity = publisher_identity(url)
                    if identity:
                        support_seen_identities[fact.uuid].add(identity)
                if (
                    support_attempts[fact.uuid]
                    < max_targeted_support_attempts
                ):
                    support_queue.append((fact, slot, required))
            traces.append(
                ResearchRoundTrace(
                    round_number=round_number,
                    slot_id=slot.slot_id,
                    purpose="support",
                    query=query,
                    documents_seen=support_result.documents_seen,
                    succeeded=support_result.succeeded,
                    facts_written=0,
                    supports_added=support_result.supports_added,
                    contributing_sources=support_result.contributing_sources,
                    contributing_source_identities=(
                        support_result.contributing_source_identities
                    ),
                    target_edge_uuids=support_result.target_edge_uuids,
                    corroborated_edge_uuids=(
                        support_result.corroborated_edge_uuids
                    ),
                    claim_match_audit=support_result.claim_match_audit,
                    coverage_after=pre_support_coverage,
                )
            )
            self.emit(
                f"round={round_number} purpose=support "
                f"success={support_result.succeeded} "
                f"supports={support_result.supports_added}"
            )

        if support_candidates:
            # The exact attempted count is visible in traces; this summary is
            # reader-facing and must also state claims left outside the budget.
            attempted_supports = sum(
                1 for trace in traces if trace.purpose == "support"
            )
            stop_decision = stop_decision.model_copy(
                update={
                    "detail": (
                        f"{stop_decision.detail}; critical claim support "
                        f"{support_successes}/{attempted_supports} targeted "
                        f"follow-up(s), {len(support_candidates)} gap(s) found"
                    )
                }
            )

        statuses = await get_gap_status(
            self.graphiti,
            research_id=run_id,
            schema=INVESTIGATION_SCHEMA,
            applicability=applicability,
        )
        coverage = coverage_ratio_from_gaps(statuses)
        facts = await fetch_facts(self.graphiti, research_id=run_id)
        evidence_pack = await build_evidence_pack(
            self.graphiti,
            research_id=run_id,
            topic=topic,
            schema=INVESTIGATION_SCHEMA,
            applicability=applicability,
            max_required_source_count=self.settings.min_sources_per_claim,
        )
        report = render_report(
            evidence_pack,
            schema=INVESTIGATION_SCHEMA,
            sources=build_source_index(facts),
        )
        self.usage.elapsed_seconds = time.perf_counter() - started
        source_count = len(
            {
                url
                for fact in facts
                for url in fact.distinct_source_urls
                if url
            }
        )
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
            slot_applicability=list(applicability.values()),
            relevance_audit=self.relevance_audit,
            claim_match_audit=[
                audit
                for trace in traces
                for audit in trace.claim_match_audit
            ],
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
