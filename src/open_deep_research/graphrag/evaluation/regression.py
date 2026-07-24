"""Fixed-case regression metrics for the GraphRAG research runtime.

Coverage alone is not a quality metric: the first 17/17 live run showed that one
loosely related fact is enough to fill a slot.  This evaluator therefore reports
separate, named measurements:

* fixed factual checks (expected anchors present, known false claims absent);
* citation coverage;
* source diversity, slot-level source breadth, and claim-level corroboration;
* model-judged slot relevance, clearly labelled as such;
* loop behaviour, tokens, provider-reported chat cost, and elapsed time.

The fixed checks are intentionally small and auditable.  They do not pretend to
measure universal truth; they catch regressions on stable, well-known facts for
three historical incidents.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from open_deep_research.graphrag.ontology import INVESTIGATION_SCHEMA, iter_slots
from open_deep_research.graphrag.runtime import GraphResearchResult
from open_deep_research.graphrag.validation.sources import publisher_identity


class FactCheck(BaseModel):
    """One expected or forbidden fact pattern."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    description: str
    patterns: list[str] = Field(min_length=1)
    should_exist: bool = True
    slot_ids: list[str] = Field(default_factory=list)
    patterns_may_span_facts: bool = False


class RegressionCase(BaseModel):
    """A stable research topic and its minimal factual audit set."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    topic: str
    checks: list[FactCheck] = Field(default_factory=list)


class FactCheckResult(BaseModel):
    """Result of applying one deterministic check to report facts."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    description: str
    passed: bool
    should_exist: bool
    matching_facts: list[str] = Field(default_factory=list)


class CaseEvaluation(BaseModel):
    """All metrics for one live regression case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    topic: str
    research_id: str
    stop_reason: str
    coverage_ratio: float
    factual_check_pass_rate: float
    citation_coverage: float
    model_judged_slot_relevance: float | None = None
    judged_item_count: int = 0
    claim_corroboration_rate: float = 0.0
    high_impact_support_rate: float = 0.0
    multi_source_slot_rate: float = 0.0
    # Deprecated compatibility field.  Before M5 this mislabeled slot-level
    # source breadth as claim corroboration.
    cross_corroborated_slot_rate: float = 0.0
    round_success_rate: float
    duplicate_query_count: int
    grounding_rejection_rate: float
    dated_fact_rate: float
    fact_count: int
    source_count: int
    episode_count: int
    rounds: int
    llm_calls: int
    search_calls: int
    search_results_rejected: int = 0
    relevance_accepted: int = 0
    relevance_uncertain: int = 0
    relevance_rejected: int = 0
    support_rounds: int = 0
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    chat_provider_cost_usd: float | None = None
    elapsed_seconds: float
    fact_checks: list[FactCheckResult] = Field(default_factory=list)


class RegressionSuiteResult(BaseModel):
    """Serializable result for a multi-topic run."""

    model_config = ConfigDict(extra="forbid")

    cases: list[CaseEvaluation] = Field(default_factory=list)

    @property
    def all_checks_passed(self) -> bool:
        return all(case.factual_check_pass_rate == 1.0 for case in self.cases)


DEFAULT_REGRESSION_CASES = (
    RegressionCase(
        case_id="ftx_2022",
        topic="FTX cryptocurrency exchange collapse and bankruptcy in 2022",
        checks=[
            FactCheck(
                check_id="ftx_bankruptcy_2022",
                description="FTX bankruptcy is anchored to 2022",
                patterns=[r"\bFTX\b", r"bankrupt|Chapter\s+11", r"\b2022\b"],
            ),
            FactCheck(
                check_id="ftx_sbf",
                description="Sam Bankman-Fried is identified",
                patterns=[r"Sam Bankman-Fried|\bSBF\b"],
                slot_ids=["who.primary_actor"],
            ),
            FactCheck(
                check_id="ftx_alameda",
                description="Alameda Research is connected to the event",
                patterns=[r"Alameda Research"],
            ),
            FactCheck(
                check_id="ftx_bahamas",
                description="The Bahamas appears as FTX's operating jurisdiction",
                patterns=[r"Bahamas|巴哈马|巴哈馬"],
                slot_ids=["where.jurisdiction", "where.event_location"],
            ),
            FactCheck(
                check_id="ftx_no_2023_bankruptcy",
                description="The report does not date FTX's bankruptcy to 2023",
                patterns=[r"\bFTX\b", r"bankrupt|Chapter\s+11", r"\b2023\b"],
                should_exist=False,
            ),
            FactCheck(
                check_id="ftx_no_2026_bankruptcy",
                description="The report does not date FTX's bankruptcy to 2026",
                patterns=[r"\bFTX\b", r"bankrupt|Chapter\s+11", r"\b2026\b"],
                should_exist=False,
            ),
        ],
    ),
    RegressionCase(
        case_id="crowdstrike_2024",
        topic="CrowdStrike global IT outage on July 19, 2024",
        checks=[
            FactCheck(
                check_id="crowdstrike_date",
                description="The outage is anchored to July 19, 2024",
                patterns=[
                    r"July\s+19,?\s+2024|19\s+July\s+2024|2024[-年/]\s*0?7[-月/]\s*19"
                ],
                slot_ids=["when.event_time", "when.discovery_time"],
            ),
            FactCheck(
                check_id="crowdstrike_falcon",
                description="CrowdStrike Falcon is identified as the affected product",
                patterns=[r"CrowdStrike", r"Falcon"],
            ),
            FactCheck(
                check_id="crowdstrike_windows",
                description="Windows systems are identified as affected",
                patterns=[r"Windows|Microsoft"],
            ),
            FactCheck(
                check_id="crowdstrike_update",
                description="A faulty content/software update is identified as the trigger",
                patterns=[r"update|更新", r"fault|defect|bug|错误|故障|缺陷"],
                slot_ids=["why.trigger", "how.mechanism", "what.core_event"],
            ),
            FactCheck(
                check_id="crowdstrike_no_2025_event",
                description="The 2024 outage is not dated to 2025",
                patterns=[r"CrowdStrike", r"outage|中断|故障", r"\b2025\b"],
                should_exist=False,
            ),
            FactCheck(
                check_id="crowdstrike_no_2026_event",
                description="The 2024 outage is not dated to 2026",
                patterns=[r"CrowdStrike", r"outage|中断|故障", r"\b2026\b"],
                should_exist=False,
            ),
        ],
    ),
    RegressionCase(
        case_id="svb_2023",
        topic="Silicon Valley Bank collapse and FDIC closure in March 2023",
        checks=[
            FactCheck(
                check_id="svb_closure_date",
                description="SVB's closure is anchored to March 10, 2023",
                patterns=[
                    r"March\s+10,?\s+2023|10\s+March\s+2023|2023[-年/]\s*0?3[-月/]\s*10"
                ],
                slot_ids=["when.event_time", "when.intervention_time"],
            ),
            FactCheck(
                check_id="svb_fdic",
                description="The FDIC is identified as an intervening authority",
                patterns=[r"\bFDIC\b|Federal Deposit Insurance Corporation"],
                slot_ids=["who.regulators", "when.intervention_time"],
            ),
            FactCheck(
                check_id="svb_bank_run",
                description="Depositor withdrawals or a bank run appear in the mechanism",
                patterns=[r"bank run|withdraw(?:al|als|ing)?|withdrew|提款|挤兑|擠兌"],
                slot_ids=["why.trigger", "how.mechanism", "how.sequence"],
            ),
            FactCheck(
                check_id="svb_rates_bonds",
                description="Interest rates and securities losses appear in the explanation",
                patterns=[
                    r"interest rate|利率",
                    r"bond|securit|Treasur|债券|債券|证券|證券",
                ],
                slot_ids=["why.motivation", "why.trigger", "how.mechanism"],
                patterns_may_span_facts=True,
            ),
            FactCheck(
                check_id="svb_no_2022_closure",
                description="SVB's closure is not dated to 2022",
                patterns=[r"Silicon Valley Bank|\bSVB\b", r"clos|fail|collapse", r"\b2022\b"],
                should_exist=False,
            ),
            FactCheck(
                check_id="svb_no_2024_closure",
                description="SVB's closure is not dated to 2024",
                patterns=[r"Silicon Valley Bank|\bSVB\b", r"clos|fail|collapse", r"\b2024\b"],
                should_exist=False,
            ),
        ],
    ),
)


def _fact_matches(fact: str, patterns: list[str]) -> bool:
    return all(re.search(pattern, fact, re.IGNORECASE) for pattern in patterns)


def _evaluate_fact_checks(
    result: GraphResearchResult,
    case: RegressionCase,
) -> list[FactCheckResult]:
    items = result.evidence_pack.items
    outcomes: list[FactCheckResult] = []
    for check in case.checks:
        allowed_slots = set(check.slot_ids)
        eligible = [
            item
            for item in items
            if not allowed_slots or item.slot_id in allowed_slots
        ]
        matches = [
            item.conclusion
            for item in eligible
            if _fact_matches(item.conclusion, check.patterns)
        ]
        if check.patterns_may_span_facts:
            spanning_matches = [
                item.conclusion
                for item in eligible
                if any(
                    re.search(pattern, item.conclusion, re.IGNORECASE)
                    for pattern in check.patterns
                )
            ]
            patterns_present = all(
                any(
                    re.search(pattern, item.conclusion, re.IGNORECASE)
                    for item in eligible
                )
                for pattern in check.patterns
            )
            matches = list(dict.fromkeys([*matches, *spanning_matches]))
            passed = patterns_present if check.should_exist else not patterns_present
        else:
            passed = bool(matches) if check.should_exist else not matches
        outcomes.append(
            FactCheckResult(
                check_id=check.check_id,
                description=check.description,
                passed=passed,
                should_exist=check.should_exist,
                matching_facts=matches[:5],
            )
        )
    return outcomes


def evaluate_case(
    result: GraphResearchResult,
    case: RegressionCase,
    *,
    slot_relevance: float | None = None,
    judged_item_count: int = 0,
) -> CaseEvaluation:
    """Calculate deterministic metrics for one completed run."""

    checks = _evaluate_fact_checks(result, case)
    check_rate = (
        sum(1 for check in checks if check.passed) / len(checks) if checks else 1.0
    )
    items = result.evidence_pack.items
    citation_coverage = (
        sum(1 for item in items if item.provenance_episode_ids) / len(items)
        if items
        else 1.0
    )

    claim_corroboration = (
        sum(
            1
            for item in items
            if item.claim_corroborated or item.source_count >= 2
        )
        / len(items)
        if items
        else 0.0
    )
    high_impact = [item for item in items if item.required_source_count >= 2]
    high_impact_support = (
        sum(
            1
            for item in high_impact
            if item.support_requirement_met
            or item.source_count >= item.required_source_count
        )
        / len(high_impact)
        if high_impact
        else 1.0
    )

    by_slot: dict[str, list[Any]] = {}
    for item in items:
        by_slot.setdefault(item.slot_id, []).append(item)
    filled_slot_groups = list(by_slot.values())
    multi_source_slots = sum(
        1
        for group in filled_slot_groups
        if len(
            {
                publisher_identity(url)
                for item in group
                for url in item.source_urls
                if publisher_identity(url)
            }
        )
        >= 2
    )
    multi_source_slot_rate = (
        multi_source_slots / len(filled_slot_groups)
        if filled_slot_groups
        else 0.0
    )

    queries = [round_trace.query.strip().casefold() for round_trace in result.rounds]
    duplicate_queries = len(queries) - len(set(queries))
    success_rate = (
        result.successful_rounds / len(result.rounds) if result.rounds else 0.0
    )
    extraction_rows = result.usage.extraction_rows
    rejection_rate = (
        result.usage.grounding_rejections / extraction_rows
        if extraction_rows
        else 0.0
    )
    dated_rate = result.dated_fact_count / result.fact_count if result.fact_count else 0.0

    return CaseEvaluation(
        case_id=case.case_id,
        topic=case.topic,
        research_id=result.research_id,
        stop_reason=result.stop_reason,
        coverage_ratio=result.coverage_ratio,
        factual_check_pass_rate=check_rate,
        citation_coverage=citation_coverage,
        model_judged_slot_relevance=slot_relevance,
        judged_item_count=judged_item_count,
        claim_corroboration_rate=claim_corroboration,
        high_impact_support_rate=high_impact_support,
        multi_source_slot_rate=multi_source_slot_rate,
        cross_corroborated_slot_rate=claim_corroboration,
        round_success_rate=success_rate,
        duplicate_query_count=duplicate_queries,
        grounding_rejection_rate=rejection_rate,
        dated_fact_rate=dated_rate,
        fact_count=result.fact_count,
        source_count=result.source_count,
        episode_count=len(result.evidence_pack.provenance),
        rounds=len(result.rounds),
        llm_calls=result.usage.llm_calls,
        search_calls=result.usage.search_calls,
        search_results_rejected=result.usage.search_results_rejected,
        relevance_accepted=result.usage.relevance_accepted,
        relevance_uncertain=result.usage.relevance_uncertain,
        relevance_rejected=result.usage.relevance_rejected,
        support_rounds=sum(
            1 for trace in result.rounds if trace.purpose == "support"
        ),
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        total_tokens=result.usage.total_tokens,
        chat_provider_cost_usd=(
            result.usage.chat_provider_cost_usd
            if result.usage.provider_cost_reported
            else None
        ),
        elapsed_seconds=result.usage.elapsed_seconds,
        fact_checks=checks,
    )


async def judge_slot_relevance(
    result: GraphResearchResult,
    *,
    llm: Any,
    model: str,
    batch_size: int = 40,
) -> tuple[float | None, int]:
    """Ask a model only whether each fact answers its assigned slot.

    This is intentionally labelled model-judged and kept out of stopping or
    verification.  It is an evaluation signal, not an authority over graph state.
    """

    question_by_slot = {
        slot.slot_id: slot.question for slot in iter_slots(INVESTIGATION_SCHEMA)
    }
    rows = [
        {
            "index": index,
            "slot_id": item.slot_id,
            "question": question_by_slot.get(item.slot_id, ""),
            "fact": item.conclusion,
        }
        for index, item in enumerate(result.evidence_pack.items)
    ]
    if not rows:
        return None, 0

    judged: dict[int, bool] = {}
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        response = await llm.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Judge only whether each fact directly helps answer its "
                        "assigned question. Do not judge general truth. For the "
                        "primary-actor slot, a later buyer, responder, regulator, "
                        "or affected party is irrelevant unless the fact also "
                        "identifies the event's central subject or responsible "
                        "actor. Return JSON "
                        'exactly as {"judgements":[{"index":0,"relevant":true}]}.'
                    ),
                },
                {"role": "user", "content": json.dumps(batch, ensure_ascii=False)},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result.usage.observe_llm_response(response)
        try:
            payload = json.loads(response.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            continue
        for item in payload.get("judgements", []) or []:
            try:
                index = int(item["index"])
            except (KeyError, TypeError, ValueError):
                continue
            if index in {row["index"] for row in batch}:
                judged[index] = bool(item.get("relevant"))

    if not judged:
        return None, 0
    return sum(judged.values()) / len(judged), len(judged)


def render_regression_summary(suite: RegressionSuiteResult) -> str:
    """Render a compact Markdown audit report."""

    lines = [
        "# GraphRAG Regression Summary",
        "",
        "| Case | Coverage | Fixed checks | Citations | Slot relevance* | "
        "Claim corroboration | Critical support | Sources | Rounds | Tokens | "
        "Chat cost | Time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in suite.cases:
        relevance = (
            f"{case.model_judged_slot_relevance:.0%}"
            if case.model_judged_slot_relevance is not None
            else "n/a"
        )
        cost = (
            f"${case.chat_provider_cost_usd:.4f}"
            if case.chat_provider_cost_usd is not None
            else "not reported"
        )
        lines.append(
            f"| {case.case_id} | {case.coverage_ratio:.0%} | "
            f"{case.factual_check_pass_rate:.0%} | {case.citation_coverage:.0%} | "
            f"{relevance} | {case.claim_corroboration_rate:.0%} | "
            f"{case.high_impact_support_rate:.0%} | {case.source_count} | "
            f"{case.rounds} | "
            f"{case.total_tokens:,} | {cost} | {case.elapsed_seconds:.1f}s |"
        )

    lines += [
        "",
        "\\* Slot relevance is model-judged and is not used for graph writes or stopping.",
        "",
    ]
    for case in suite.cases:
        lines += [f"## {case.case_id}", ""]
        lines.append(
            f"- Stop: `{case.stop_reason}`; facts {case.fact_count}; "
            f"episodes {case.episode_count}; duplicate queries {case.duplicate_query_count}"
        )
        lines.append(
            f"- Grounding rejections: {case.grounding_rejection_rate:.0%}; "
            f"dated facts: {case.dated_fact_rate:.0%}; "
            f"multi-source slots: {case.multi_source_slot_rate:.0%}; "
            f"claim corroboration: {case.claim_corroboration_rate:.0%}"
        )
        lines.append(
            f"- Pre-write relevance: {case.relevance_accepted} accepted, "
            f"{case.relevance_uncertain} uncertain, "
            f"{case.relevance_rejected} rejected; "
            f"search results rejected: {case.search_results_rejected}; "
            f"targeted support rounds: {case.support_rounds}"
        )
        for check in case.fact_checks:
            marker = "✅" if check.passed else "❌"
            expectation = "present" if check.should_exist else "absent"
            lines.append(
                f"- {marker} `{check.check_id}` ({expectation}) — {check.description}"
            )
            for fact in check.matching_facts[:2]:
                lines.append(f"  - `{fact}`")
        lines.append("")
    return "\n".join(lines)
