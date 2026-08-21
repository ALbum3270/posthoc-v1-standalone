import asyncio
import json

import pytest

import open_deep_research.harness.evidence_gap as evidence_gap_module
from open_deep_research.harness import EvidenceGapPlanningCapacityAudit
from open_deep_research.harness.attribution import (
    AttributionResult,
    AttributionStatus,
    AttributionStopReason,
    CandidateSource,
    ClaimAttribution,
    _note_reference,
)
from open_deep_research.harness.budget import (
    RunCostBudget,
    RunCostCapReached,
    RunCostController,
)
from open_deep_research.harness.checklist import (
    ChecklistDimension,
    ChecklistItem,
    ChecklistStatus,
    ResearchChecklist,
)
from open_deep_research.harness.claims import (
    AtomicClaim,
    CitationRequirement,
    ClaimNormalizationStatus,
    SourceResolution,
    parse_markdown_blocks,
)
from open_deep_research.harness.evidence_gap import (
    EvidenceGapBudget,
    EvidenceGapResult,
    GapSearchQuery,
    GapSearchRecord,
    EvidenceGapStopReason,
    _merge_verifications,
    build_evidence_gap_plan_prompt,
    run_evidence_gap_round,
)
from open_deep_research.harness.ledger import ResearchLedger
from open_deep_research.harness.notes import (
    NoteLocationStatus,
    QuoteSpan,
    create_note,
)
from open_deep_research.harness.tools import SearchResult
from open_deep_research.harness.truth_conditions import (
    ElementAssessmentExecutionStatus,
    ElementVerificationVerdict,
    ElementizationProposal,
    ElementizationReview,
    ElementizationSemanticStatus,
    ExecutionCompleteness,
    aggregate_truth_condition_claim,
    build_truth_condition_registry,
    select_truth_condition_registry,
    truth_condition_registry_sha256,
)
from open_deep_research.harness.verify import (
    ClaimEvidenceState,
    ClaimVerification,
    VerificationRecordStatus,
    VerificationResult,
    VerificationVerdict,
    VerifiedElementRelation,
    VerifiedSourceRelation,
    build_claim_verification,
)


def _checklist() -> ResearchChecklist:
    return ResearchChecklist(
        topic="A neutral topic",
        items=(
            ChecklistItem(
                item_id="what-1",
                dimension=ChecklistDimension.WHAT,
                question="What happened?",
                priority=1,
                required_source_count=2,
            ),
        ),
    )


def _claim(
    report: str,
    *,
    citation_requirement: CitationRequirement = CitationRequirement.EXTERNAL,
    claim_id: str = "claim-0001",
    text: str = "The event occurred.",
) -> AtomicClaim:
    start = report.index(text)
    return AtomicClaim(
        claim_id=claim_id,
        block_id=parse_markdown_blocks(report)[1].block_id,
        selected_text=text,
        claim_text=text,
        anchor_text=text,
        start_char=start,
        end_char=start + len(text),
        citation_requirement=citation_requirement,
        normalization_status=ClaimNormalizationStatus.LOCATED,
    )


def _note(ledger: ResearchLedger, url: str, text: str):
    ledger.cache_source(url, text)
    return ledger.add_note(
        create_note(
            item_id="what-1",
            finding="The source describes the event.",
            quote=text,
            url=url,
            source_text=text,
        )
    )


def _candidate(note) -> CandidateSource:
    return CandidateSource(
        note_id=note.note_id,
        source_id=note.source_id,
        item_id=note.item_id,
        publisher=note.publisher,
        url=note.url,
        location_status=note.location_status,
        resolution=SourceResolution.DIRECT,
    )


def _initial(
    claim: AtomicClaim,
    *,
    candidate: CandidateSource | None,
    state: ClaimEvidenceState,
) -> tuple[AttributionResult, VerificationResult]:
    attribution = ClaimAttribution(
        claim=claim,
        status=(
            AttributionStatus.CANDIDATE_SOURCES
            if candidate is not None
            else AttributionStatus.NO_CANDIDATE_SOURCE
        ),
        candidates=(candidate,) if candidate is not None else (),
    )
    relations = ()
    publishers = ()
    if candidate is not None:
        source_quote = "The event occurred."
        relations = (
            VerifiedSourceRelation(
                claim_id=claim.claim_id,
                source_id=candidate.source_id,
                url=candidate.url,
                publisher_domain_proxy=candidate.publisher,
                candidate_note_ids=(candidate.note_id,),
                candidate_source_ids=(candidate.source_id,),
                status=VerificationRecordStatus.COMPLETED,
                semantic_verdict=VerificationVerdict.SUPPORTS,
                model_quote=source_quote,
                source_quote=source_quote,
                span=QuoteSpan(start_char=0, end_char=len(source_quote)),
                location_status=NoteLocationStatus.LOCATABLE,
                is_formal_supporting_evidence=True,
            ),
        )
        publishers = (candidate.publisher,)
    return (
        AttributionResult(
            attributions=(attribution,),
            stop_reason=AttributionStopReason.COMPLETED,
        ),
        VerificationResult(
            claims=(
                ClaimVerification(
                    claim=claim,
                    state=state,
                    required_independent_sources=2,
                    relations=relations,
                    formal_supporting_evidence_count=len(relations),
                    publisher_domain_proxy_count=len(publishers),
                    publisher_domain_proxies=publishers,
                ),
            )
        ),
    )


class ScriptedModel:
    def __init__(self, *contents):
        self.contents = list(contents)
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        if not self.contents:
            raise AssertionError("unexpected model call")
        return {
            "content": json.dumps(self.contents.pop(0)),
            "token_count": 5,
            "cost_usd": 0.005,
        }


class NoNetwork:
    async def search(self, query, **kwargs):
        raise AssertionError("network should not be used")

    async def extract(self, urls, **kwargs):
        raise AssertionError("network should not be used")


class EmptySearchNetwork:
    def __init__(self):
        self.queries = []

    async def search(self, query, **kwargs):
        self.queries.append(query)
        return {"results": []}

    async def extract(self, urls, **kwargs):
        raise AssertionError("no empty search result should be read")


class FailedSearchNetwork:
    def __init__(self):
        self.queries = []

    async def search(self, query, **kwargs):
        self.queries.append(query)
        raise RuntimeError("provider unavailable")

    async def extract(self, urls, **kwargs):
        raise AssertionError("a failed search must not lead to a read")


def _estimate_tokens(client, prompt):
    return 1


def _estimate_cost(client, prompt):
    return 0.001


def test_non_external_gap_state_is_not_an_eligible_target() -> None:
    report = "# Report\n\nThe event occurred."
    claim = _claim(
        report,
        citation_requirement=CitationRequirement.INTERNAL,
    )
    ledger = ResearchLedger(topic="A neutral topic")
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=None,
        state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    )
    gap_model = ScriptedModel()

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=NoNetwork(),
            budget=EvidenceGapBudget(),
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert result.stop_reason == EvidenceGapStopReason.NO_TARGETS
    assert result.target_claim_ids == ()
    assert gap_model.prompts == []
    assert result.final_attribution == initial_attribution
    assert result.final_verification == initial_verification


def test_planner_uses_one_ordered_query_for_multiple_claims_within_hard_cap():
    report = "# Report\n\nThe event occurred. A later review confirmed it."
    first_claim = _claim(report)
    second_claim = _claim(
        report,
        claim_id="claim-0002",
        text="A later review confirmed it.",
    )
    first_attribution, first_verification = _initial(
        first_claim,
        candidate=None,
        state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    )
    second_attribution, second_verification = _initial(
        second_claim,
        candidate=None,
        state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    )
    initial_attribution = AttributionResult(
        attributions=(
            first_attribution.attributions[0],
            second_attribution.attributions[0],
        ),
        stop_reason=AttributionStopReason.COMPLETED,
    )
    initial_verification = VerificationResult(
        claims=(
            first_verification.claims[0],
            second_verification.claims[0],
        )
    )
    gap_model = ScriptedModel(
        {
            "cached_candidates": [],
            "queries": [
                {
                    # Model priority order deliberately differs from claim ID
                    # order. Code must preserve it rather than sort/truncate
                    # per claim.
                    "claim_ids": [
                        second_claim.claim_id,
                        first_claim.claim_id,
                    ],
                    "item_id": "what-1",
                    "query": "one focused account covering both events",
                },
                {
                    "claim_ids": [first_claim.claim_id],
                    "item_id": "what-1",
                    "query": "lower-priority overflow query",
                },
            ],
        }
    )
    network = EmptySearchNetwork()

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ResearchLedger(topic="A neutral topic"),
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=network,
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=1,
                max_reads=1,
            ),
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert network.queries == ["one focused account covering both events"]
    assert len(result.searches) == 1
    assert result.searches[0].query.claim_ids == (
        second_claim.claim_id,
        first_claim.claim_id,
    )
    assert any(
        entry.get("error") == "gap search query cap reached"
        for entry in result.rejected_entries
    )
    prompt = gap_model.prompts[0]
    assert "hard budget of at most 1 web search queries" in prompt
    assert "Return only semantic routes" in prompt
    assert "Do not return deferred_targets" in prompt
    assert "Code records every unrouted target" in prompt
    assert '"corroboration_target":2' in prompt
    assert result.verification_reserve is not None
    assert result.verification_reserve.planned_query_count == 1
    assert result.verification_reserve.planned_query_claim_count == 2
    assert result.information_yield.pass_completed_within_budget is True
    assert result.information_yield.new_completed_relation_count == 0
    assert "new completed claim-source relations=0" in result.stop_detail
    assert "not found" not in result.stop_detail


def _many_gap_targets(count: int):
    sentences = tuple(f"Fact number {index} occurred." for index in range(1, count + 1))
    report = "# Report\n\n" + " ".join(sentences)
    claims = tuple(
        _claim(
            report,
            claim_id=f"claim-{index:04d}",
            text=sentence,
        )
        for index, sentence in enumerate(sentences, start=1)
    )
    initial_pairs = tuple(
        _initial(
            claim,
            candidate=None,
            state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
        )
        for claim in claims
    )
    attribution = AttributionResult(
        attributions=tuple(pair[0].attributions[0] for pair in initial_pairs),
        stop_reason=AttributionStopReason.COMPLETED,
    )
    verification = VerificationResult(
        claims=tuple(pair[1].claims[0] for pair in initial_pairs)
    )
    return report, claims, attribution, verification


def test_finance_14_shape_executes_a_model_selected_partial_plan():
    """A sparse plan is auditable work, not a reason to force a retry."""

    report, claims, initial_attribution, initial_verification = (
        _many_gap_targets(43)
    )
    first_sparse_plan = {
        "cached_candidates": [],
        "queries": [
            {
                "claim_ids": [claims[0].claim_id],
                "item_id": "what-1",
                "query": "only the first route",
            }
        ],
    }
    gap_model = ScriptedModel(first_sparse_plan)
    network = EmptySearchNetwork()

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ResearchLedger(topic="A neutral topic"),
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=network,
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_reads=3,
            ),
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    from open_deep_research.harness.runner import _evidence_gap_execution_record

    stage = _evidence_gap_execution_record(result)
    assert EvidenceGapBudget().max_search_queries == 6
    assert len(gap_model.prompts) == 1
    assert network.queries == ["only the first route"]
    assert result.routed_target_claim_ids == (claims[0].claim_id,)
    assert result.unrouted_target_claim_ids == tuple(
        claim.claim_id for claim in claims[1:]
    )
    assert tuple(target.claim_id for target in result.deferred_targets) == (
        result.unrouted_target_claim_ids
    )
    assert {target.reason for target in result.deferred_targets} == {
        "query_capacity_not_allocated"
    }
    assert {target.allocation_source for target in result.deferred_targets} == {
        "code_derived"
    }
    assert result.planning_attempt_count == 1
    assert result.selected_planning_attempt == 1
    assert result.unused_query_slots == 5
    assert stage.status.value == "partial"
    assert stage.expected_scope.count == 43
    assert stage.evaluated_scope.count == 1
    assert "issued query slots=1/6" in result.stop_detail


def test_model_cannot_manufacture_a_semantic_no_query_escape():
    report, claims, initial_attribution, initial_verification = (
        _many_gap_targets(1)
    )
    gap_model = ScriptedModel(
        {
            "cached_candidates": [],
            "queries": [],
            "deferred_targets": [
                {
                    "claim_id": claims[0].claim_id,
                    "reason": "unsearchable",
                    "priority_rationale": "no likely source",
                }
            ],
        },
    )

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ResearchLedger(topic="A neutral topic"),
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=NoNetwork(),
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=0,
                max_reads=0,
            ),
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    # With no cache or search capacity there is no useful action to authorize.
    # Code records the deferral without paying a planner that could only
    # manufacture a semantic excuse for doing nothing.
    assert result.stop_reason is EvidenceGapStopReason.BUDGET_EXHAUSTED
    assert result.routed_target_claim_ids == ()
    assert result.unrouted_target_claim_ids == (claims[0].claim_id,)
    assert result.deferred_targets[0].reason == "query_capacity_not_allocated"
    assert result.deferred_targets[0].allocation_source == "code_derived"
    assert len(gap_model.prompts) == 0


def test_incomplete_plan_executes_its_audited_partial_route_once():
    report, claims, initial_attribution, initial_verification = (
        _many_gap_targets(43)
    )
    sparse_plan = {
        "cached_candidates": [],
        "queries": [
            {
                "claim_ids": [claims[0].claim_id],
                "item_id": "what-1",
                "query": "one valid partial route",
            }
        ],
    }
    gap_model = ScriptedModel(sparse_plan)
    network = EmptySearchNetwork()

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ResearchLedger(topic="A neutral topic"),
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=network,
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=6,
                max_reads=3,
            ),
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert result.stop_reason is EvidenceGapStopReason.COMPLETED
    assert result.planning_attempt_count == 1
    assert result.selected_planning_attempt == 1
    assert result.unused_query_slots == 5
    assert network.queries == ["one valid partial route"]
    assert len(result.searches) == 1
    assert result.routed_target_claim_ids == (claims[0].claim_id,)
    assert result.unrouted_target_claim_ids == tuple(
        claim.claim_id for claim in claims[1:]
    )
    assert tuple(target.claim_id for target in result.deferred_targets) == (
        result.unrouted_target_claim_ids
    )
    assert "issued query slots=1/6" in result.stop_detail


def test_failed_search_route_remains_unrouted_and_partial():
    report, claims, initial_attribution, initial_verification = (
        _many_gap_targets(1)
    )
    gap_model = ScriptedModel(
        {
            "cached_candidates": [],
            "queries": [
                {
                    "claim_ids": [claims[0].claim_id],
                    "item_id": "what-1",
                    "query": "a route whose provider fails",
                }
            ],
        }
    )
    network = FailedSearchNetwork()

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ResearchLedger(topic="A neutral topic"),
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=network,
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=1,
                max_reads=0,
            ),
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    from open_deep_research.harness.runner import _evidence_gap_execution_record

    stage = _evidence_gap_execution_record(result)
    assert result.searches[0].error == "RuntimeError: provider unavailable"
    assert result.routed_target_claim_ids == ()
    assert result.unrouted_target_claim_ids == (claims[0].claim_id,)
    assert result.deferred_targets[0].reason == "search_route_failed"
    assert stage.status.value == "partial"
    assert stage.evaluated_scope.count == 0


def test_zero_route_plan_returns_an_audited_partial_result():
    """No route is disclosed, never recast as a semantic source conclusion."""

    report, claims, initial_attribution, initial_verification = (
        _many_gap_targets(3)
    )
    gap_model = ScriptedModel(
        {"cached_candidates": [], "queries": []},
    )

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ResearchLedger(topic="A neutral topic"),
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=NoNetwork(),
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=3,
                max_reads=0,
            ),
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    from open_deep_research.harness.runner import _evidence_gap_execution_record

    stage = _evidence_gap_execution_record(result)
    assert result.stop_reason is EvidenceGapStopReason.COMPLETED
    assert result.planning_attempt_count == 1
    assert result.selected_planning_attempt == 1
    assert result.routed_target_claim_ids == ()
    assert result.unrouted_target_claim_ids == tuple(
        claim.claim_id for claim in claims
    )
    assert len(result.deferred_targets) == 3
    assert "issued query slots=0/3" in result.stop_detail
    assert stage.status.value == "partial"
    assert stage.evaluated_scope.count == 0


def test_finance_15_shape_executes_six_model_selected_routes_once():
    """Six focused routes run without a second slot-filling planning call."""

    report, claims, initial_attribution, initial_verification = (
        _many_gap_targets(45)
    )
    plan = {
        "cached_candidates": [],
        "queries": [
            {
                "claim_ids": [claim.claim_id],
                "item_id": "what-1",
                "query": f"bounded route {index}",
            }
            for index, claim in enumerate(claims[:6], start=1)
        ],
    }
    gap_model = ScriptedModel(plan)
    network = EmptySearchNetwork()

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ResearchLedger(topic="A neutral topic"),
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=network,
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=6,
                max_reads=3,
            ),
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    from open_deep_research.harness.runner import _evidence_gap_execution_record

    stage = _evidence_gap_execution_record(result)
    assert result.stop_reason is EvidenceGapStopReason.COMPLETED
    assert result.planning_attempt_count == 1
    assert result.selected_planning_attempt == 1
    assert result.unused_query_slots == 0
    assert network.queries == [f"bounded route {index}" for index in range(1, 7)]
    assert result.routed_target_claim_ids == tuple(
        claim.claim_id for claim in claims[:6]
    )
    assert len(result.deferred_targets) == 39
    assert all(
        target.allocation_source == "code_derived"
        for target in result.deferred_targets
    )
    assert stage.status.value == "partial"
    assert stage.expected_scope.count == 45
    assert stage.evaluated_scope.count == 6


def test_partial_plan_preserves_model_selected_merged_route_coverage():
    """Unused query capacity does not displace focused merged routes."""

    report, claims, initial_attribution, initial_verification = (
        _many_gap_targets(45)
    )
    first_plan = {
        "cached_candidates": [],
        "queries": [
            {
                "claim_ids": [claim.claim_id for claim in claims[:7]],
                "item_id": "what-1",
                "query": "focused route covering related facts one",
            },
            {
                "claim_ids": [claim.claim_id for claim in claims[7:14]],
                "item_id": "what-1",
                "query": "focused route covering related facts two",
            },
            {
                "claim_ids": [claim.claim_id for claim in claims[14:20]],
                "item_id": "what-1",
                "query": "focused route covering related facts three",
            },
        ],
    }
    gap_model = ScriptedModel(first_plan)
    network = EmptySearchNetwork()

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ResearchLedger(topic="A neutral topic"),
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=network,
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=6,
                max_reads=0,
            ),
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    from open_deep_research.harness.runner import _evidence_gap_execution_record

    stage = _evidence_gap_execution_record(result)
    assert result.stop_reason is EvidenceGapStopReason.COMPLETED
    assert result.planning_attempt_count == 1
    assert result.selected_planning_attempt == 1
    assert result.unused_query_slots == 3
    assert network.queries == [
        "focused route covering related facts one",
        "focused route covering related facts two",
        "focused route covering related facts three",
    ]
    assert result.routed_target_claim_ids == tuple(
        claim.claim_id for claim in claims[:20]
    )
    assert len(result.deferred_targets) == 25
    assert stage.status.value == "partial"
    assert stage.expected_scope.count == 45
    assert stage.evaluated_scope.count == 20


def test_single_partial_plan_does_not_spend_budget_on_a_phantom_retry():
    """The routing model's one plan does not incur a slot-filling retry."""

    report, claims, initial_attribution, initial_verification = (
        _many_gap_targets(4)
    )
    first_plan = {
        "cached_candidates": [],
        "queries": [
            {
                "claim_ids": [claims[0].claim_id],
                "item_id": "what-1",
                "query": "preserved route",
            }
        ],
    }
    gap_model = ScriptedModel(first_plan)
    network = EmptySearchNetwork()

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ResearchLedger(topic="A neutral topic"),
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=network,
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=4,
                max_reads=0,
            ),
            estimate_input_tokens=lambda _client, _prompt: 1,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert len(gap_model.prompts) == 1
    assert result.stop_reason is EvidenceGapStopReason.COMPLETED
    assert result.planning_attempt_count == 1
    assert result.selected_planning_attempt == 1
    assert result.unused_query_slots == 3
    assert network.queries == ["preserved route"]
    assert not result.rejected_entries


@pytest.mark.parametrize(
    ("target_count", "routed_count"),
    ((58, 2), (56, 6)),
)
def test_measured_sparse_gap_plans_report_actual_route_coverage(
    target_count,
    routed_count,
):
    """Regression for finance-13 (2/58) and finance-11 (6/56).

    Both measured passes returned normally, but the old runner equated that
    control-flow outcome with every target having received a route.  These are
    the measured cardinalities, reduced only to IDs and one accepted query.
    """

    from open_deep_research.harness.runner import (
        _evidence_gap_execution_record,
    )

    target_ids = tuple(
        f"claim-{index:04d}" for index in range(1, target_count + 1)
    )
    result = EvidenceGapResult(
        target_claim_ids=target_ids,
        searches=(
            GapSearchRecord(
                query=GapSearchQuery(
                    claim_ids=target_ids[:routed_count],
                    item_id="what-1",
                    query="one accepted merged query",
                ),
                results=(),
            ),
        ),
        stop_reason=EvidenceGapStopReason.COMPLETED,
        stop_detail="bounded pass returned normally",
        final_attribution=AttributionResult(
            attributions=(),
            stop_reason=AttributionStopReason.COMPLETED,
        ),
        final_verification=VerificationResult(claims=()),
    )

    stage = _evidence_gap_execution_record(result)

    assert result.routed_target_claim_ids == target_ids[:routed_count]
    assert result.unrouted_target_claim_ids == target_ids[routed_count:]
    assert stage.status.value == "partial"
    assert stage.expected_scope.count == target_count
    assert stage.evaluated_scope.count == routed_count
    assert stage.unevaluated_ids == target_ids[routed_count:]


def test_single_publisher_target_one_does_not_enter_gap_round() -> None:
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    ledger = ResearchLedger(topic="A neutral topic")
    note = _note(
        ledger,
        "https://source.example/article",
        "The event occurred.",
    )
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=_candidate(note),
        state=ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
    )
    initial_verification = VerificationResult(
        claims=(
            initial_verification.claims[0].model_copy(
                update={"corroboration_target": 1}
            ),
        )
    )
    gap_model = ScriptedModel()

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=NoNetwork(),
            budget=EvidenceGapBudget(),
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert result.stop_reason == EvidenceGapStopReason.NO_TARGETS
    assert gap_model.prompts == []


def test_distributed_element_support_enters_whole_claim_corroboration_gap() -> None:
    report = "# Report\n\nThe event occurred in 2024."
    claim = _claim(report, text="The event occurred in 2024.")
    ledger = ResearchLedger(topic="A neutral topic")
    notes = (
        _note(
            ledger,
            "https://first.example/article",
            "The event occurred.",
        ),
        _note(
            ledger,
            "https://second.example/article",
            "The date was 2024.",
        ),
    )
    same_element_publisher_note = _note(
        ledger,
        "https://first.example/alternative",
        "Another article from the already used publisher.",
    )
    candidates = tuple(_candidate(note) for note in notes)
    registry = build_truth_condition_registry(
        {claim.claim_id: claim.claim_text},
        proposals=(
            ElementizationProposal(
                claim_id=claim.claim_id,
                elements=("The event occurred.", "The date was 2024."),
                rationale="proposal",
            ),
        ),
        reviews=(
            ElementizationReview(
                claim_id=claim.claim_id,
                semantic_status=ElementizationSemanticStatus.COMPLETE,
                elements=("The event occurred.", "The date was 2024."),
                rationale="independent review",
            ),
        ),
    )
    entry = registry.entries[0]
    first_element, second_element = entry.elements

    def element_relation(
        candidate: CandidateSource,
        element,
        *,
        supports: bool,
    ) -> VerifiedElementRelation:
        quote = element.text if supports else None
        return VerifiedElementRelation(
            claim_id=claim.claim_id,
            element_id=element.element_id,
            element_text=element.text,
            source_id=candidate.source_id,
            status=ElementAssessmentExecutionStatus.COMPLETE,
            semantic_verdict=(
                ElementVerificationVerdict.SUPPORTS
                if supports
                else ElementVerificationVerdict.DOES_NOT_SUPPORT
            ),
            source_quote=quote,
            span=(
                QuoteSpan(start_char=0, end_char=len(quote))
                if quote is not None
                else None
            ),
            location_status=(
                NoteLocationStatus.LOCATABLE if supports else None
            ),
            is_formal_supporting_evidence=supports,
        )

    def partial_relation(
        candidate: CandidateSource,
        *,
        supports_first: bool,
    ) -> VerifiedSourceRelation:
        return VerifiedSourceRelation(
            claim_id=claim.claim_id,
            source_id=candidate.source_id,
            url=candidate.url,
            publisher_domain_proxy=candidate.publisher,
            candidate_note_ids=(candidate.note_id,),
            candidate_source_ids=(candidate.source_id,),
            status=VerificationRecordStatus.COMPLETED,
            semantic_verdict=VerificationVerdict.NOT_ENOUGH_INFORMATION,
            element_relations=(
                element_relation(
                    candidate,
                    first_element,
                    supports=supports_first,
                ),
                element_relation(
                    candidate,
                    second_element,
                    supports=not supports_first,
                ),
            ),
        )

    relations = (
        partial_relation(candidates[0], supports_first=True),
        partial_relation(candidates[1], supports_first=False),
    )
    aggregate = aggregate_truth_condition_claim(
        entry,
        tuple(
            element.as_assessment()
            for relation in relations
            for element in relation.element_relations
        ),
        expected_source_ids=tuple(candidate.source_id for candidate in candidates),
    )
    first_source_aggregate = aggregate_truth_condition_claim(
        entry,
        tuple(
            element.as_assessment()
            for element in relations[0].element_relations
        ),
        expected_source_ids=(candidates[0].source_id,),
    )
    focused_plan = build_evidence_gap_plan_prompt(
        targets=(
            build_claim_verification(
                claim,
                (relations[0],),
                required_sources=2,
                truth_condition_aggregate=first_source_aggregate,
            ),
        ),
        notes=(),
        checklist=_checklist(),
        max_queries=1,
        truth_condition_registry=registry,
    )
    assert '"truth_condition":"The event occurred."' in focused_plan
    assert '"truth_condition":"The date was 2024."' in focused_plan
    assert '"semantic_state":"supported"' in focused_plan
    assert '"semantic_state":"not_supported"' in focused_plan
    assert "supporting_source_ids" not in focused_plan
    assert "not_supporting_source_ids" not in focused_plan
    initial_attribution = AttributionResult(
        attributions=(
            ClaimAttribution(
                claim=claim,
                status=AttributionStatus.CANDIDATE_SOURCES,
                candidates=candidates,
            ),
        ),
        stop_reason=AttributionStopReason.COMPLETED,
    )
    initial_verification = VerificationResult(
        claims=(
            build_claim_verification(
                claim,
                relations,
                required_sources=2,
                truth_condition_aggregate=aggregate,
            ),
        ),
        truth_condition_registry_sha256=truth_condition_registry_sha256(
            registry
        ),
    )
    gap_model = ScriptedModel(
        {
            "cached_candidates": [
                {
                    "claim_id": claim.claim_id,
                    "note_id": same_element_publisher_note.note_id,
                    "source_id": same_element_publisher_note.source_id,
                    "independent_from_existing_publishers": True,
                    "publisher_identity": "first",
                    "independence_rationale": "claimed alternative",
                }
            ],
            "queries": [],
        },
    )

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=NoNetwork(),
            truth_condition_registry=registry,
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=1,
                max_reads=0,
            ),
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert result.target_claim_ids == (claim.claim_id,)
    assert gap_model.prompts
    assert "supported_distributed_element_evidence" in gap_model.prompts[0]
    assert "whole_claim_supporting_publisher_domain_proxies" in (
        gap_model.prompts[0]
    )
    assert "element_supporting_publisher_domain_proxies" in (
        gap_model.prompts[0]
    )
    assert "used_supporting_publisher_domain_proxies" in gap_model.prompts[0]
    assert any(
        entry.get("error")
        == "publisher domain proxy already supports this claim"
        for entry in result.rejected_entries
    )


def test_cached_unused_source_adds_multi_domain_support_without_corroborating():
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    ledger = ResearchLedger(topic="A neutral topic")
    first = _note(
        ledger,
        "https://first.example/article",
        "The event occurred.",
    )
    second = _note(
        ledger,
        "https://second.example/article",
        "The event occurred.",
    )
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=_candidate(first),
        state=ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
    )
    gap_model = ScriptedModel(
        {
            "cached_candidates": [
                {
                    "claim_id": claim.claim_id,
                    "note_id": second.note_id,
                    "source_id": second.source_id,
                    "independent_from_existing_publishers": True,
                    "publisher_identity": "second",
                    "independence_rationale": "different publisher",
                }
            ],
            "queries": [],
        }
    )
    attribution_model = ScriptedModel(
        {
            "action": "attribute",
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "candidates": [
                        {
                            "note_id": second.note_id,
                            "source_id": second.source_id,
                            "inherited_from_claim_id": None,
                        }
                    ],
                }
            ],
        }
    )
    verifier = ScriptedModel(
        {
            "results": [
                {
                    "claim_id": claim.claim_id,
                    "verdict": "supports",
                    "start_segment_id": "S000001",
                    "end_segment_id": "S000001",
                    "explanation": "independent statement",
                }
            ]
        },
    )

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=attribution_model,
            verification_model=verifier,
            tavily_client=NoNetwork(),
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=2,
                max_reads=2,
            ),
            required_independent_sources={claim.claim_id: 2},
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert result.stop_reason == EvidenceGapStopReason.COMPLETED
    assert result.searches == ()
    assert len(result.cached_candidate_hints) == 1
    assert result.final_verification.claims[0].state == (
        ClaimEvidenceState.SUPPORTED_MULTIPLE_DOMAIN_PROXIES
    )
    assert result.final_verification.claims[0].publisher_domain_proxy_count == 2
    assert len(verifier.prompts) == 1
    assert "https://second.example/article" in verifier.prompts[0]
    assert "https://first.example/article" not in verifier.prompts[0]
    assert result.verification_merge is not None
    assert result.verification_merge.initial_completed_relation_count == 1
    assert result.verification_merge.incremental_completed_relation_count == 1
    assert result.verification_merge.final_completed_relation_count == 2
    assert result.verification_merge.preserved_initial_completed_relation_count == 1
    assert result.verification_merge.completed_relation_count_non_decreasing is True
    assert result.information_yield.new_completed_relation_count == 1
    assert result.information_yield.new_completed_verdict_counts["supports"] == 1
    assert result.information_yield.claims_newly_corroborated == 0
    assert (
        result.information_yield
        .claims_newly_supported_by_multiple_domain_proxies
        == 1
    )
    assert result.information_yield.claims_newly_conflicting == 0
    assert result.verification_reserve is not None
    assert result.verification_reserve.cached_hint_batch_count == 1
    assert result.verification_reserve.reserved_tokens == 1
    assert [call.stage for call in result.usage] == [
        "cache_review_and_search_plan",
        "reverification",
    ]
    assert result.claim_registry_unchanged is True
    assert ledger.source_cache.keys() == {
        "https://first.example/article",
        "https://second.example/article",
    }
    assert ledger.evidence_gap_history[0].event == "cache_review"


def test_explicit_incremental_subset_projects_registry_in_attribution_order(
    monkeypatch,
) -> None:
    report = "# Report\n\nThe first event occurred. The second event occurred."
    first_claim = _claim(
        report,
        claim_id="claim-0001",
        text="The first event occurred.",
    )
    second_claim = _claim(
        report,
        claim_id="claim-0002",
        text="The second event occurred.",
    )
    claims = (first_claim, second_claim)
    registry = build_truth_condition_registry(
        {claim.claim_id: claim.claim_text for claim in claims},
        proposals=tuple(
            ElementizationProposal(
                claim_id=claim.claim_id,
                elements=(claim.claim_text,),
            )
            for claim in claims
        ),
        reviews=tuple(
            ElementizationReview(
                claim_id=claim.claim_id,
                semantic_status=ElementizationSemanticStatus.COMPLETE,
                elements=(claim.claim_text,),
                rationale="The claim has one complete truth condition.",
            )
            for claim in claims
        ),
    )
    initial_attribution = AttributionResult(
        attributions=tuple(
            ClaimAttribution(
                claim=claim,
                status=AttributionStatus.NO_CANDIDATE_SOURCE,
            )
            for claim in claims
        ),
        stop_reason=AttributionStopReason.COMPLETED,
    )
    initial_verification = VerificationResult(
        claims=tuple(
            build_claim_verification(
                claim,
                (),
                required_sources=2,
                attribution_status=AttributionStatus.NO_CANDIDATE_SOURCE,
                truth_condition_aggregate=aggregate_truth_condition_claim(
                    registry.entry_for(claim.claim_id),
                    (),
                    expected_source_ids=(),
                ),
            )
            for claim in claims
        ),
        truth_condition_registry_sha256=truth_condition_registry_sha256(
            registry
        ),
    )
    ledger = ResearchLedger(topic="A neutral topic")
    notes = {
        first_claim.claim_id: _note(
            ledger,
            "https://first.example/record",
            first_claim.claim_text,
        ),
        second_claim.claim_id: _note(
            ledger,
            "https://second.example/record",
            second_claim.claim_text,
        ),
    }
    selected_ids = (second_claim.claim_id, first_claim.claim_id)
    gap_model = ScriptedModel(
        {
            "cached_candidates": [
                {
                    "claim_id": claim_id,
                    "note_id": notes[claim_id].note_id,
                    "source_id": notes[claim_id].source_id,
                    "independent_from_existing_publishers": True,
                    "publisher_identity": notes[claim_id].publisher,
                    "independence_rationale": "No existing publisher exists.",
                }
                for claim_id in selected_ids
            ],
            "queries": [],
        }
    )
    observed_registry_orders: list[tuple[str, ...]] = []
    real_select_registry = evidence_gap_module.select_truth_condition_registry

    def select_exact_order(registry_arg, claim_ids):
        # Capacity preflight estimates complete one-claim routes without
        # mutating execution scope. The actual incremental verifier call must
        # still preserve the model-routed order below.
        if len(claim_ids) == 1:
            return real_select_registry(registry_arg, claim_ids)
        # A set may happen to iterate in the requested order under one hash
        # seed, so also lock the caller contract to an ordered sequence.
        assert isinstance(claim_ids, tuple)
        assert claim_ids == selected_ids
        return real_select_registry(registry_arg, claim_ids)

    async def verify_exact_subset(attributions, *, registry, **_kwargs):
        assert registry is not None
        attribution_ids = tuple(
            attribution.claim.claim_id for attribution in attributions
        )
        assert registry.denominator.selected_claim_ids == attribution_ids
        observed_registry_orders.append(
            registry.denominator.selected_claim_ids
        )
        return VerificationResult(
            claims=tuple(
                build_claim_verification(
                    attribution.claim,
                    (),
                    required_sources=2,
                    attribution_status=attribution.status,
                    truth_condition_aggregate=aggregate_truth_condition_claim(
                        registry.entry_for(attribution.claim.claim_id),
                        (),
                        expected_source_ids=tuple(
                            candidate.source_id
                            for candidate in attribution.candidates
                        ),
                    ),
                )
                for attribution in attributions
            ),
            truth_condition_registry_sha256=(
                truth_condition_registry_sha256(registry)
            ),
        )

    monkeypatch.setattr(
        evidence_gap_module,
        "verify_attributions",
        verify_exact_subset,
    )
    monkeypatch.setattr(
        evidence_gap_module,
        "select_truth_condition_registry",
        select_exact_order,
    )

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=NoNetwork(),
            budget=EvidenceGapBudget(
                max_tokens=500,
                max_cost_usd=1,
                max_search_queries=1,
                max_reads=0,
            ),
            truth_condition_registry=registry,
            explicit_target_claim_ids=selected_ids,
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert result.stop_reason is EvidenceGapStopReason.COMPLETED
    assert observed_registry_orders == [selected_ids]


def test_budget_failure_for_new_relation_preserves_completed_initial_verdict():
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    ledger = ResearchLedger(topic="A neutral topic")
    first = _note(
        ledger,
        "https://first.example/article",
        "The event occurred.",
    )
    second = _note(
        ledger,
        "https://second.example/article",
        "The event occurred.",
    )
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=_candidate(first),
        state=ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
    )
    gap_model = ScriptedModel(
        {
            "cached_candidates": [
                {
                    "claim_id": claim.claim_id,
                    "note_id": second.note_id,
                    "source_id": second.source_id,
                    "independent_from_existing_publishers": True,
                    "publisher_identity": "second",
                    "independence_rationale": "different publisher",
                }
            ],
            "queries": [],
        }
    )
    attribution_model = ScriptedModel(
        {
            "action": "attribute",
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "candidates": [
                        {
                            "note_id": second.note_id,
                            "source_id": second.source_id,
                            "inherited_from_claim_id": None,
                        }
                    ],
                }
            ],
        }
    )
    verifier = ScriptedModel()

    def reserve_exceeds_remaining(client, prompt):
        return 6 if client is verifier else 1

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=attribution_model,
            verification_model=verifier,
            tavily_client=NoNetwork(),
            budget=EvidenceGapBudget(
                max_tokens=10,
                max_cost_usd=1,
                max_search_queries=0,
                max_reads=0,
            ),
            required_independent_sources={claim.claim_id: 2},
            estimate_input_tokens=reserve_exceeds_remaining,
            estimate_cost_usd=_estimate_cost,
        )
    )

    relations = result.final_verification.claims[0].relations
    assert result.stop_reason == EvidenceGapStopReason.BUDGET_EXHAUSTED
    assert result.information_yield.pass_completed_within_budget is False
    assert verifier.prompts == []
    assert [relation.status for relation in relations] == [
        VerificationRecordStatus.COMPLETED,
        VerificationRecordStatus.VERIFICATION_NOT_RUN_BUDGET,
    ]
    assert relations[0].semantic_verdict == VerificationVerdict.SUPPORTS
    assert result.final_verification.claims[0].state == (
        ClaimEvidenceState.VERIFICATION_INCOMPLETE
    )
    assert result.verification_merge is not None
    assert result.verification_merge.initial_completed_relation_count == 1
    assert result.verification_merge.final_completed_relation_count == 1
    assert result.verification_merge.completed_relation_count_non_decreasing is True


def test_same_relation_budget_failure_cannot_replace_completed_verdict():
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    ledger = ResearchLedger(topic="A neutral topic")
    note = _note(
        ledger,
        "https://source.example/article",
        "The event occurred.",
    )
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=_candidate(note),
        state=ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
    )
    failed_relation = VerifiedSourceRelation(
        claim_id=claim.claim_id,
        source_id=note.source_id,
        url=note.url,
        publisher_domain_proxy=note.publisher,
        candidate_note_ids=(note.note_id,),
        candidate_source_ids=(note.source_id,),
        status=VerificationRecordStatus.VERIFICATION_NOT_RUN_BUDGET,
        error="budget exhausted",
    )
    refreshed = VerificationResult(
        claims=(
            ClaimVerification(
                claim=claim,
                state=ClaimEvidenceState.VERIFICATION_NOT_RUN,
                required_independent_sources=2,
                relations=(failed_relation,),
                formal_supporting_evidence_count=0,
                publisher_domain_proxy_count=0,
            ),
        )
    )

    merged, audit = _merge_verifications(
        initial_verification,
        refreshed,
        merged_attribution=initial_attribution,
    )

    relation = merged.claims[0].relations[0]
    assert relation.status == VerificationRecordStatus.COMPLETED
    assert relation.semantic_verdict == VerificationVerdict.SUPPORTS
    assert audit.initial_completed_relation_count == 1
    assert audit.final_completed_relation_count == 1
    assert len(audit.protected_completed_relations) == 1
    assert audit.protected_completed_relations[0].attempted_status == (
        VerificationRecordStatus.VERIFICATION_NOT_RUN_BUDGET
    )


def test_element_merge_keeps_attribution_candidate_denominator_without_relation():
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    ledger = ResearchLedger(topic="A neutral topic")
    first_note = _note(
        ledger,
        "https://first.example/record",
        "The event occurred.",
    )
    second_note = _note(
        ledger,
        "https://second.example/record",
        "A second record exists.",
    )
    first_candidate = _candidate(first_note)
    second_candidate = _candidate(second_note)
    registry = build_truth_condition_registry(
        {claim.claim_id: claim.selected_text},
        proposals=(
            ElementizationProposal(
                claim_id=claim.claim_id,
                elements=(claim.selected_text,),
            ),
        ),
        reviews=(
            ElementizationReview(
                claim_id=claim.claim_id,
                semantic_status=ElementizationSemanticStatus.COMPLETE,
                elements=(claim.selected_text,),
                rationale="One condition closes the claim denominator.",
            ),
        ),
    )
    element = registry.entries[0].elements[0]
    element_relation = VerifiedElementRelation(
        claim_id=claim.claim_id,
        element_id=element.element_id,
        element_text=element.text,
        source_id=first_note.source_id,
        status=ElementAssessmentExecutionStatus.COMPLETE,
        semantic_verdict=ElementVerificationVerdict.SUPPORTS,
        source_quote="The event occurred.",
        span=QuoteSpan(start_char=0, end_char=len("The event occurred.")),
        location_status=NoteLocationStatus.LOCATABLE,
        is_formal_supporting_evidence=True,
    )
    relation = VerifiedSourceRelation(
        claim_id=claim.claim_id,
        source_id=first_note.source_id,
        url=first_note.url,
        publisher_domain_proxy="first.example",
        candidate_note_ids=(first_note.note_id,),
        candidate_source_ids=(first_note.source_id,),
        status=VerificationRecordStatus.COMPLETED,
        semantic_verdict=VerificationVerdict.SUPPORTS,
        explanation="All registered elements are supported.",
        is_formal_supporting_evidence=True,
        element_relations=(element_relation,),
    )
    initial_aggregate = aggregate_truth_condition_claim(
        registry.entries[0],
        (element_relation.as_assessment(),),
        expected_source_ids=(first_note.source_id,),
    )
    initial = VerificationResult(
        claims=(
            build_claim_verification(
                claim,
                (relation,),
                required_sources=2,
                attribution_status=AttributionStatus.CANDIDATE_SOURCES,
                truth_condition_aggregate=initial_aggregate,
            ),
        ),
        truth_condition_registry_sha256=truth_condition_registry_sha256(
            registry
        ),
    )
    merged_attribution = AttributionResult(
        attributions=(
            ClaimAttribution(
                claim=claim,
                status=AttributionStatus.CANDIDATE_SOURCES,
                candidates=(first_candidate, second_candidate),
            ),
        ),
        stop_reason=AttributionStopReason.COMPLETED,
    )

    empty_refreshed = VerificationResult(
        claims=(),
        truth_condition_registry_sha256=(
            truth_condition_registry_sha256(
                select_truth_condition_registry(registry, ())
            )
        ),
    )
    merged, _ = _merge_verifications(
        initial,
        empty_refreshed,
        merged_attribution=merged_attribution,
        truth_condition_registry=registry,
    )

    aggregate = merged.claims[0].truth_condition_aggregate
    assert aggregate is not None
    assert aggregate.elements[0].expected_source_ids == (
        first_note.source_id,
        second_note.source_id,
    )
    assert aggregate.elements[0].execution_completeness is (
        ExecutionCompleteness.PARTIAL
    )
    assert aggregate.elements[0].unresolved_source_ids == (
        second_note.source_id,
    )
    # The merged audit is derived from the final relation denominator.  It
    # must not retain the pre-gap zero-count snapshot after a relation exists.
    assert merged.independence.unresolved_relation_count == 1

    with pytest.raises(ValueError, match="initial verification uses a different"):
        _merge_verifications(
            initial.model_copy(
                update={"truth_condition_registry_sha256": None}
            ),
            empty_refreshed,
            merged_attribution=merged_attribution,
            truth_condition_registry=registry,
        )


class SearchAndReadNetwork:
    def __init__(self, url):
        self.url = url
        self.search_calls = 0
        self.extract_calls = 0

    async def search(self, query, **kwargs):
        self.search_calls += 1
        return {
            "results": [
                {
                    "title": "Candidate",
                    "url": self.url,
                    "content": "A candidate snippet.",
                }
            ]
        }

    async def extract(self, urls, **kwargs):
        self.extract_calls += 1
        return {
            "results": [
                {
                    "url": self.url,
                    "raw_content": "The event occurred.",
                }
            ]
        }


def test_cache_failure_for_one_gap_source_is_recorded_and_does_not_escape(
    monkeypatch,
):
    """A ledger invariant failure is one failed acquisition, not a lost run."""

    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    ledger = ResearchLedger(topic="A neutral topic")
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=None,
        state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    )
    selected_url = "https://new.example/article"
    gap_model = ScriptedModel(
        {
            "cached_candidates": [],
            "queries": [
                {
                    "claim_ids": [claim.claim_id],
                    "item_id": "what-1",
                    "query": "one candidate",
                }
            ],
        },
        {
            "reads": [
                {
                    "url": selected_url,
                    "item_id": "what-1",
                    "claim_ids": [claim.claim_id],
                    "independent_from_existing_publishers": True,
                    "publisher_identity": "New Example",
                    "independence_rationale": "A new publisher candidate.",
                }
            ]
        },
    )

    original_cache_source = ResearchLedger.cache_source

    def fail_selected_cache(self, url, cleaned_text, **kwargs):
        if url == selected_url:
            raise ValueError("synthetic cache invariant failure")
        return original_cache_source(self, url, cleaned_text, **kwargs)

    monkeypatch.setattr(ResearchLedger, "cache_source", fail_selected_cache)

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=SearchAndReadNetwork(selected_url),
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=1,
                max_reads=1,
            ),
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert result.stop_reason is EvidenceGapStopReason.COMPLETED
    assert result.acquisitions[0].outcome == "read_error"
    assert "ValueError: synthetic cache invariant failure" in (
        result.acquisitions[0].error or ""
    )
    assert ledger.get_source(selected_url) is None
    assert any(
        event.event == "source_read_error"
        and "synthetic cache invariant failure" in event.result_summary
        for event in ledger.evidence_gap_history
    )


def test_grouped_read_rejects_only_claim_with_existing_publisher():
    """A cached URL is re-analysed only for the claim it has not checked."""

    report = "# Report\n\nThe first event occurred. The second event occurred."
    first_claim = _claim(
        report,
        claim_id="claim-0001",
        text="The first event occurred.",
    )
    duplicate_claim = _claim(
        report,
        claim_id="claim-0048",
        text="The second event occurred.",
    )
    ledger = ResearchLedger(topic="A neutral topic")
    first_note = _note(
        ledger,
        "https://first.example/article",
        "The event occurred.",
    )
    selected_url = "https://sciencedirect.com/article"
    duplicate_note = _note(
        ledger,
        selected_url,
        "The event occurred.\nCACHE-ONLY-MARKER",
    )
    first_attribution, first_verification = _initial(
        first_claim,
        candidate=_candidate(first_note),
        state=ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
    )
    duplicate_attribution, duplicate_verification = _initial(
        duplicate_claim,
        candidate=_candidate(duplicate_note),
        state=ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
    )
    initial_attribution = AttributionResult(
        attributions=(
            first_attribution.attributions[0],
            duplicate_attribution.attributions[0],
        ),
        stop_reason=AttributionStopReason.COMPLETED,
    )
    initial_verification = VerificationResult(
        claims=(
            first_verification.claims[0],
            duplicate_verification.claims[0],
        )
    )
    gap_model = ScriptedModel(
        {
            "cached_candidates": [],
            "queries": [
                {
                    "claim_ids": [first_claim.claim_id, duplicate_claim.claim_id],
                    "item_id": "what-1",
                    "query": "one result routed to two claims",
                }
            ],
        },
        {
            "reads": [
                {
                    "url": selected_url,
                    "item_id": "what-1",
                    "claim_ids": [first_claim.claim_id, duplicate_claim.claim_id],
                    "independent_from_existing_publishers": True,
                    "publisher_identity": "ScienceDirect",
                    "independence_rationale": (
                        "A different publisher for the first claim."
                    ),
                }
            ]
        },
    )
    note_model = ScriptedModel(
        {
            "notes": [
                {
                    "item_id": "what-1",
                    "finding": "The source is relevant to the first event.",
                    "quote": "The event occurred.",
                }
            ]
        }
    )

    class NewClaimAttribution:
        def __init__(self):
            self.prompts = []

        async def generate(self, _prompt):
            self.prompts.append(_prompt)
            new_note = ledger.notes[-1]
            return {
                "content": {
                    "action": "attribute",
                    "claims": [
                        {
                            "claim_id": first_claim.claim_id,
                            "candidates": [
                                {
                                    "note_ref": _note_reference(new_note),
                                    "inherited_from_claim_id": None,
                                }
                            ],
                        }
                    ],
                },
                "token_count": 5,
                "cost_usd": 0.005,
            }

    attribution_model = NewClaimAttribution()
    verifier = ScriptedModel(
        {
            "results": [
                {
                    "claim_id": first_claim.claim_id,
                    "verdict": "supports",
                    "start_segment_id": "S000001",
                    "end_segment_id": "S000001",
                    "explanation": "The cached source supports the claim.",
                }
            ]
        }
    )
    network = SearchAndReadNetwork(selected_url)

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=note_model,
            attribution_model=attribution_model,
            verification_model=verifier,
            tavily_client=network,
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=1,
                max_reads=1,
            ),
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert len(result.read_selections) == 1
    assert result.read_selections[0].claim_ids == (first_claim.claim_id,)
    assert result.acquisitions[0].claim_ids == (first_claim.claim_id,)
    assert result.acquisitions[0].cache_hit is True
    assert result.acquisitions[0].outcome == "notes_created"
    assert len(result.acquisitions[0].note_ids) == 1
    assert result.information_yield.new_completed_relation_count == 1
    assert result.added_source_urls == ()
    assert network.extract_calls == 0
    assert len(note_model.prompts) == 1
    assert "CACHE-ONLY-MARKER" in note_model.prompts[0]
    assert first_claim.claim_text in note_model.prompts[0]
    assert duplicate_claim.claim_text not in note_model.prompts[0]
    assert len(attribution_model.prompts) == 1
    assert len(verifier.prompts) == 1
    final_by_id = {
        entry.claim.claim_id: entry for entry in result.final_verification.claims
    }
    assert len(final_by_id[duplicate_claim.claim_id].relations) == 1
    assert final_by_id[duplicate_claim.claim_id].relations[0] == (
        duplicate_verification.claims[0].relations[0]
    )
    assert any(
        entry.get("stage") == "read_selection_claim"
        and entry.get("claim_id") == duplicate_claim.claim_id
        and entry.get("error")
        == "source was already checked for this claim"
        for entry in result.rejected_entries
    )


def test_cached_read_note_model_error_preserves_cache_provenance():
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    ledger = ResearchLedger(topic="A neutral topic")
    selected_url = "https://cached.example/record"
    _note(ledger, selected_url, "The event occurred.\nCACHE-ONLY-MARKER")
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=None,
        state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    )
    gap_model = ScriptedModel(
        {
            "cached_candidates": [],
            "queries": [
                {
                    "claim_ids": [claim.claim_id],
                    "item_id": "what-1",
                    "query": "primary event record",
                }
            ],
        },
        {
            "reads": [
                {
                    "url": selected_url,
                    "item_id": "what-1",
                    "claim_ids": [claim.claim_id],
                    "independent_from_existing_publishers": True,
                    "publisher_identity": "cached.example",
                    "independence_rationale": "No source has checked this claim.",
                }
            ]
        },
    )

    class FailingNoteModel:
        def __init__(self):
            self.prompts = []

        async def generate(self, prompt):
            self.prompts.append(prompt)
            raise ValueError("synthetic note failure")

    note_model = FailingNoteModel()
    network = SearchAndReadNetwork(selected_url)

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=note_model,
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=network,
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=1,
                max_reads=1,
            ),
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert len(result.acquisitions) == 1
    assert result.acquisitions[0].outcome == "note_model_error"
    assert result.acquisitions[0].cache_hit is True
    assert "ValueError: synthetic note failure" in (
        result.acquisitions[0].error or ""
    )
    assert len(note_model.prompts) == 1
    assert "CACHE-ONLY-MARKER" in note_model.prompts[0]
    assert network.extract_calls == 0


def test_failed_cached_relation_can_be_retried_and_replaced():
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    ledger = ResearchLedger(topic="A neutral topic")
    note = _note(
        ledger,
        "https://cached.example/record",
        claim.claim_text,
    )
    candidate = _candidate(note)
    attribution = AttributionResult(
        attributions=(
            ClaimAttribution(
                claim=claim,
                status=AttributionStatus.CANDIDATE_SOURCES,
                candidates=(candidate,),
            ),
        ),
        stop_reason=AttributionStopReason.COMPLETED,
    )
    failed_relation = VerifiedSourceRelation(
        claim_id=claim.claim_id,
        source_id=note.source_id,
        url=note.url,
        publisher_domain_proxy=note.publisher,
        candidate_note_ids=(note.note_id,),
        candidate_source_ids=(note.source_id,),
        status=VerificationRecordStatus.VERIFICATION_MODEL_ERROR,
        error="temporary verifier failure",
    )
    failed_verification = VerificationResult(
        claims=(
            build_claim_verification(
                claim,
                (failed_relation,),
                required_sources=2,
                attribution_status=AttributionStatus.CANDIDATE_SOURCES,
            ),
        )
    )
    gap_model = ScriptedModel(
        {
            "cached_candidates": [
                {
                    "claim_id": claim.claim_id,
                    "note_id": note.note_id,
                    "source_id": note.source_id,
                    "independent_from_existing_publishers": True,
                    "publisher_identity": note.publisher,
                    "independence_rationale": (
                        "Retrying a source whose prior verifier call failed."
                    ),
                }
            ],
            "queries": [],
        }
    )
    verifier = ScriptedModel(
        {
            "results": [
                {
                    "claim_id": claim.claim_id,
                    "verdict": "supports",
                    "start_segment_id": "S000001",
                    "end_segment_id": "S000001",
                    "explanation": "The cached record states the claim.",
                }
            ]
        }
    )

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=attribution,
            initial_verification=failed_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=verifier,
            tavily_client=NoNetwork(),
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=0,
                max_reads=0,
            ),
            explicit_target_claim_ids=(claim.claim_id,),
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert len(verifier.prompts) == 1
    assert len(result.final_verification.claims[0].relations) == 1
    repaired = result.final_verification.claims[0].relations[0]
    assert repaired.status is VerificationRecordStatus.COMPLETED
    assert repaired.semantic_verdict is VerificationVerdict.SUPPORTS
    assert result.information_yield.new_completed_relation_count == 1


def test_read_selection_is_not_reserved_against_unknown_source_length():
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    ledger = ResearchLedger(topic="A neutral topic")
    initial_note = _note(
        ledger,
        "https://initial.example/article",
        "The event occurred.",
    )
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=_candidate(initial_note),
        state=ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
    )
    new_url = "https://new.example/article"
    gap_model = ScriptedModel(
        {
            "cached_candidates": [],
            "queries": [
                {
                    "claim_ids": [claim.claim_id],
                    "item_id": "what-1",
                    "query": "a second independent account",
                }
            ],
        },
        {"reads": []},
    )
    verifier = ScriptedModel()
    network = SearchAndReadNetwork(new_url)

    def stage_sensitive_estimate(client, prompt):
        return 6 if client is verifier else 5

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=verifier,
            tavily_client=network,
            budget=EvidenceGapBudget(
                # Planning must now preserve one bounded read -> note ->
                # verification route.  The stage-sensitive estimates below
                # require planning/read-selection headroom plus 5 + 5 + 6
                # tokens for note, incremental attribution, and verification,
                # so
                # the historical 15-token envelope is intentionally no
                # longer a valid admission shape.
                max_tokens=30,
                max_cost_usd=1,
                max_search_queries=1,
                max_reads=1,
            ),
            estimate_input_tokens=stage_sensitive_estimate,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert result.verification_reserve is not None
    # Before a selected URL has been read, source length is unknown.  An
    # unrelated cached page must not become a synthetic future-source reserve.
    pre_read_reserve = result.verification_reserve_history[0]
    assert pre_read_reserve.estimated_tokens == 0
    assert pre_read_reserve.prerequisite_stage == "read_selection"
    assert pre_read_reserve.prerequisite_estimated_tokens == 6
    # Unknown source length is still never extrapolated from an unrelated
    # cache entry.  The separate minimum-action probe now protects one
    # bounded note+verification route until the model has selected a URL.
    assert pre_read_reserve.minimum_action_estimated_tokens == 18
    assert pre_read_reserve.reserved_tokens == 18
    assert result.verification_reserve.estimated_tokens == 0
    assert result.verification_reserve.admitted_read_source_urls == ()
    assert result.stop_reason == EvidenceGapStopReason.COMPLETED
    assert network.search_calls == 1
    assert network.extract_calls == 0
    assert [call.stage for call in result.usage] == [
        "cache_review_and_search_plan",
        "read_selection",
    ]
    assert not any(
        entry.get("stage") == "read_selection"
        and "preserving the verification reserve" in entry.get("error", "")
        for entry in result.rejected_entries
    )


def test_large_selected_source_releases_reserve_so_later_small_source_can_finish():
    """Finance-18 shape: an oversized read cannot starve a later real source."""

    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    ledger = ResearchLedger(topic="A neutral topic")
    # This cache entry has no candidate relation.  The legacy implementation
    # used it as the synthetic maximum for every future web result.
    ledger.cache_source("https://unrelated.example/large", "x" * 20_000)
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=None,
        state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    )
    large_url = "https://large.example/report"
    small_url = "https://small.example/report"
    gap_model = ScriptedModel(
        {
            "cached_candidates": [],
            "queries": [
                {
                    "claim_ids": [claim.claim_id],
                    "item_id": "what-1",
                    "query": "independent account",
                }
            ],
        },
        {
            "reads": [
                {
                    "url": large_url,
                    "item_id": "what-1",
                    "claim_ids": [claim.claim_id],
                    "independent_from_existing_publishers": True,
                    "publisher_identity": "large",
                    "independence_rationale": "different publisher",
                },
                {
                    "url": small_url,
                    "item_id": "what-1",
                    "claim_ids": [claim.claim_id],
                    "independent_from_existing_publishers": True,
                    "publisher_identity": "small",
                    "independence_rationale": "different publisher",
                },
            ]
        },
    )

    class TwoReadNetwork:
        async def search(self, query, **kwargs):
            return {
                "results": [
                    {"title": "Large", "url": large_url, "content": "a"},
                    {"title": "Small", "url": small_url, "content": "b"},
                ]
            }

        async def extract(self, urls, **kwargs):
            return {
                "results": [
                    {
                        "url": url,
                        "raw_content": (
                            "large source " * 2_000
                            if url == large_url
                            else "The event occurred."
                        ),
                    }
                    for url in urls
                ]
            }

    note_model = ScriptedModel(
        {
            "notes": [
                {
                    "item_id": "what-1",
                    "finding": "The event occurred.",
                    "quote": "The event occurred.",
                }
            ]
        }
    )

    class AttributeNewNote:
        async def generate(self, prompt):
            note = ledger.notes[-1]
            return {
                "content": {
                    "action": "attribute",
                    "claims": [
                        {
                            "claim_id": claim.claim_id,
                            "candidates": [
                                {
                                    "note_ref": _note_reference(note),
                                    "inherited_from_claim_id": None,
                                }
                            ],
                        }
                    ],
                },
                "token_count": 5,
                "cost_usd": 0.005,
            }

    verifier = ScriptedModel(
        {
            "results": [
                {
                    "claim_id": claim.claim_id,
                    "verdict": "supports",
                    "start_segment_id": "S000001",
                    "end_segment_id": "S000001",
                    "explanation": "the source states the event",
                }
            ]
        }
    )

    def dynamic_estimate(client, prompt):
        if client is verifier:
            return 30 if large_url in prompt else 5
        if client is note_model:
            return 35 if large_url in prompt else 3
        return 1

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=note_model,
            attribution_model=AttributeNewNote(),
            verification_model=verifier,
            tavily_client=TwoReadNetwork(),
            budget=EvidenceGapBudget(
                max_tokens=40,
                max_cost_usd=1,
                max_search_queries=1,
                max_reads=2,
            ),
            estimate_input_tokens=dynamic_estimate,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert result.stop_reason is EvidenceGapStopReason.BUDGET_EXHAUSTED
    assert [entry.outcome for entry in result.acquisitions] == [
        "note_extraction_not_run_budget_after_actual_read",
        "notes_created",
    ]
    assert large_url not in note_model.prompts[0]
    assert small_url in note_model.prompts[0]
    assert result.information_yield.new_completed_relation_count == 1
    assert result.final_verification.claims[0].state is (
        ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER
    )
    assert result.verification_reserve is not None
    assert result.verification_reserve.admitted_read_source_urls == (small_url,)
    large_pre_note_reserve = result.verification_reserve_history[1]
    assert large_pre_note_reserve.prerequisite_stage == "note_extraction"
    assert large_pre_note_reserve.prerequisite_estimated_tokens == 42
    assert (
        large_pre_note_reserve.incremental_reattribution_estimated_tokens
        > 0
    )
    assert large_pre_note_reserve.required_downstream_tokens == (
        large_pre_note_reserve.estimated_tokens
        + large_pre_note_reserve.incremental_reattribution_estimated_tokens
    )
    assert large_pre_note_reserve.reserve_fully_funded is False
    assert all(
        large_url not in reserve.admitted_read_source_urls
        for reserve in result.verification_reserve_history[2:]
    )
    large_acquisition = result.acquisitions[0]
    assert large_acquisition.source_chars > 20_000
    assert any(
        entry.get("stage") == "note_extraction"
        and entry.get("url") == large_url
        # The read helper normalizes trailing whitespace.  Audit the length
        # of that canonical cached text, not the pre-cleaning fixture input.
        and entry.get("source_chars") == large_acquisition.source_chars
        and "actual source not admitted" in entry.get("error", "")
        and "cannot preserve incremental reattribution" in entry.get(
            "error", ""
        )
        for entry in result.rejected_entries
    )


def test_new_source_and_notes_enter_gap_history_without_collection_rounds():
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    ledger = ResearchLedger(topic="A neutral topic")
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=None,
        state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    )
    new_url = "https://new.example/article"
    gap_model = ScriptedModel(
        {
            "cached_candidates": [],
            "queries": [
                {
                    "claim_ids": [claim.claim_id],
                    "item_id": "what-1",
                    "query": "independent account of the event",
                }
            ],
        },
        {
            "reads": [
                {
                    "url": new_url,
                    "item_id": "what-1",
                    "claim_ids": [claim.claim_id],
                    "independent_from_existing_publishers": True,
                    "publisher_identity": "new",
                    "independence_rationale": "new publishing organization",
                }
            ]
        },
    )
    note_model = ScriptedModel(
        {
            "notes": [
                {
                    "item_id": "what-1",
                    "finding": "The source states that the event occurred.",
                    "quote": "The event occurred.",
                }
            ]
        }
    )

    class NewNoteAttribution:
        async def generate(self, prompt):
            note = ledger.notes[-1]
            return {
                "content": {
                    "action": "attribute",
                    "claims": [
                        {
                                "claim_id": claim.claim_id,
                                "candidates": [
                                    {
                                        "note_ref": _note_reference(note),
                                        "inherited_from_claim_id": None,
                                    }
                            ],
                        }
                    ],
                },
                "token_count": 5,
                "cost_usd": 0.005,
            }

    verifier = ScriptedModel(
        {
            "results": [
                {
                    "claim_id": claim.claim_id,
                    "verdict": "contradicts",
                    "start_segment_id": "S000001",
                    "end_segment_id": "S000001",
                    "explanation": "the source relation was checked",
                }
            ]
        }
    )
    network = SearchAndReadNetwork(new_url)

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=note_model,
            attribution_model=NewNoteAttribution(),
            verification_model=verifier,
            tavily_client=network,
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=1,
                max_reads=1,
            ),
            required_independent_sources={claim.claim_id: 2},
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert result.stop_reason == EvidenceGapStopReason.COMPLETED
    assert result.added_source_urls == (new_url,)
    assert result.added_note_ids == ("note-000001",)
    assert ledger.get_source(new_url) == "The event occurred."
    assert ledger.rounds == []
    assert [event.event for event in ledger.evidence_gap_history] == [
        "cache_review",
        "source_acquired",
        "gap_stop",
    ]
    assert result.final_verification.claims[0].state == (
        ClaimEvidenceState.REFUTED
    )
    assert result.final_verification.claims[0].relations[0].semantic_verdict == (
        VerificationVerdict.CONTRADICTS
    )
    assert result.information_yield.pass_completed_within_budget is True
    assert result.information_yield.new_completed_relation_count == 1
    assert result.information_yield.new_completed_verdict_counts == {
        "supports": 0,
        "does_not_support": 0,
        "contradicts": 1,
        "not_enough_information": 0,
    }
    assert result.information_yield.new_publisher_domain_proxies == (
        "new.example",
    )
    assert result.information_yield.new_claim_publisher_relation_count == 1
    assert "new completed claim-source relations=1" in result.stop_detail


def test_gap_budget_exhaustion_is_not_reported_as_sources_exhausted():
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    ledger = ResearchLedger(topic="A neutral topic")
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=None,
        state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    )

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=ScriptedModel(),
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=NoNetwork(),
            budget=EvidenceGapBudget(max_tokens=0, max_cost_usd=0),
            estimate_input_tokens=lambda client, prompt: 1,
            estimate_cost_usd=lambda client, prompt: 0,
        )
    )

    assert result.stop_reason == EvidenceGapStopReason.BUDGET_EXHAUSTED
    assert "budget" in result.stop_detail
    assert "not found" not in result.stop_detail
    assert result.final_verification == initial_verification


def test_same_brand_on_another_domain_is_rejected_before_read():
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    ledger = ResearchLedger(topic="A neutral topic")
    first = _note(
        ledger,
        "https://bbc.com/article",
        "The event occurred.",
    )
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=_candidate(first),
        state=ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
    )
    other_domain = "https://bbc.co.uk/other"
    gap_model = ScriptedModel(
        {
            "cached_candidates": [],
            "queries": [
                {
                    "claim_ids": [claim.claim_id],
                    "item_id": "what-1",
                    "query": "another account",
                }
            ],
        },
        {
            "reads": [
                {
                    "url": other_domain,
                    "item_id": "what-1",
                    "claim_ids": [claim.claim_id],
                    "independent_from_existing_publishers": True,
                    "publisher_identity": "BBC",
                    "independence_rationale": "different domain",
                }
            ]
        },
    )
    network = SearchAndReadNetwork(other_domain)
    attribution_model = ScriptedModel(
        {
            "action": "attribute",
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "candidates": [
                        {
                            "note_id": first.note_id,
                            "source_id": first.source_id,
                            "inherited_from_claim_id": None,
                        }
                    ],
                }
            ],
        }
    )
    verifier = ScriptedModel(
        {
            "results": [
                {
                    "claim_id": claim.claim_id,
                    "verdict": "supports",
                    "start_segment_id": "S000001",
                    "end_segment_id": "S000001",
                    "explanation": "same original source",
                }
            ]
        }
    )

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=attribution_model,
            verification_model=verifier,
            tavily_client=network,
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=1,
                max_reads=1,
            ),
            required_independent_sources={claim.claim_id: 2},
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert result.read_selections == ()
    assert network.extract_calls == 0
    assert any(
        entry["error"] == "publisher identity matches an existing domain label"
        for entry in result.rejected_entries
    )
    assert result.final_verification.claims[0].state == (
        ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER
    )


def test_recovery_element_intent_survives_query_read_and_acquisition():
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    initial_attribution, legacy_verification = _initial(
        claim,
        candidate=None,
        state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    )
    registry = build_truth_condition_registry(
        {claim.claim_id: claim.claim_text},
        proposals=(
            ElementizationProposal(
                claim_id=claim.claim_id,
                elements=("The event occurred.",),
            ),
        ),
        reviews=(
            ElementizationReview(
                claim_id=claim.claim_id,
                semantic_status=ElementizationSemanticStatus.COMPLETE,
                elements=("The event occurred.",),
                rationale="One atomic event is the complete denominator.",
            ),
        ),
    )
    entry = registry.entries[0]
    target_element_id = entry.elements[0].element_id
    aggregate = aggregate_truth_condition_claim(
        entry,
        (),
        expected_source_ids=(),
    )
    element_verification = build_claim_verification(
        claim,
        legacy_verification.claims[0].relations,
        required_sources=(
            legacy_verification.claims[0].corroboration_target
        ),
        truth_condition_aggregate=aggregate,
    )
    initial_verification = VerificationResult(
        claims=(element_verification,),
        truth_condition_registry_sha256=truth_condition_registry_sha256(
            registry
        ),
    )
    selected_url = "https://new.example/element-record"
    gap_model = ScriptedModel(
        {
            "cached_candidates": [],
            "queries": [
                {
                    "claim_ids": [claim.claim_id],
                    "item_id": "what-1",
                    "query": "dated record of the event",
                }
            ],
        },
        {
            "reads": [
                {
                    "url": selected_url,
                    "item_id": "what-1",
                    "claim_ids": [claim.claim_id],
                    "independent_from_existing_publishers": True,
                    "publisher_identity": "new",
                    "independence_rationale": "new publishing organization",
                }
            ]
        },
    )
    note_model = ScriptedModel({"notes": []})
    estimated_prompts: list[str] = []

    def estimate_tokens(_client, prompt):
        estimated_prompts.append(prompt)
        return 1

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ResearchLedger(topic="A neutral topic"),
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=note_model,
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=SearchAndReadNetwork(selected_url),
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=1,
                max_reads=1,
            ),
            truth_condition_registry=registry,
            explicit_target_claim_ids=(claim.claim_id,),
            explicit_target_element_ids={
                claim.claim_id: (target_element_id,)
            },
            estimate_input_tokens=estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert result.searches[0].query.target_element_ids == (
        target_element_id,
    )
    assert result.read_selections[0].target_element_ids == (
        target_element_id,
    )
    assert result.acquisitions[0].target_element_ids == (
        target_element_id,
    )
    assert target_element_id in note_model.prompts[0]
    assert "The event occurred." in note_model.prompts[0]
    assert any(
        "https://capacity.invalid/evidence" in prompt
        and target_element_id in prompt
        and "The event occurred." in prompt
        and "registered truth-condition element" in prompt
        for prompt in estimated_prompts
    )
    assert any(
        target_element_id in prompt
        and "registered truth-condition element" in prompt
        for prompt in estimated_prompts
    )


def test_live_shape_compact_plan_preserves_one_complete_evidence_action():
    """The finance-26 planning shape must not spend the whole 60k pass."""

    report, claims, initial_attribution, initial_verification = (
        _many_gap_targets(27)
    )
    ledger = ResearchLedger(topic="A neutral topic")
    for index in range(176):
        url = f"https://archive-{index:03d}.example/record"
        # The finding size approximates the real 176-note registry without
        # depending on a mutable paid-run artifact.
        finding = (
            f"Archived finding {index:03d}: "
            + "contextual evidence remains available for model review. " * 5
        )
        source_text = f"Archived quotation {index:03d}. " + "context " * 25
        ledger.cache_source(url, source_text)
        ledger.add_note(
            create_note(
                item_id="what-1",
                finding=finding,
                quote=f"Archived quotation {index:03d}.",
                url=url,
                source_text=source_text,
            )
        )

    selected_claim = claims[0]
    selected_url = "https://new.example/live-shape-record"

    class RatioModel:
        def __init__(self, *contents):
            self.contents = list(contents)
            self.prompts = []

        async def generate(self, prompt):
            self.prompts.append(prompt)
            if not self.contents:
                raise AssertionError("unexpected model call")
            # Reproduce the paid run's actual total-token/prompt-char ratio.
            actual_tokens = max(
                1,
                (len(prompt) * 68_234 + 205_484) // 205_485,
            )
            return {
                "content": json.dumps(self.contents.pop(0)),
                "token_count": actual_tokens,
                "cost_usd": 0.001,
            }

    gap_model = RatioModel(
        {
            "cached_candidates": [],
            "queries": [
                {
                    "claim_ids": [selected_claim.claim_id],
                    "item_id": "what-1",
                    "query": "primary record for the first claim",
                }
            ],
        },
        {
            "reads": [
                {
                    "url": selected_url,
                    "item_id": "what-1",
                    "claim_ids": [selected_claim.claim_id],
                    "independent_from_existing_publishers": True,
                    "publisher_identity": "new.example",
                    "independence_rationale": "A distinct primary record.",
                }
            ]
        },
    )
    note_model = RatioModel(
        {
            "notes": [
                {
                    "item_id": "what-1",
                    "finding": selected_claim.claim_text,
                    "quote": selected_claim.claim_text,
                }
            ]
        }
    )

    class RoutedAttributionModel:
        def __init__(self):
            self.prompts = []

        async def generate(self, prompt):
            self.prompts.append(prompt)
            note = ledger.notes[-1]
            return {
                "content": {
                    "action": "attribute",
                    "claims": [
                        {
                            "claim_id": selected_claim.claim_id,
                            "candidates": [
                                {
                                    "note_ref": _note_reference(note),
                                    "inherited_from_claim_id": None,
                                }
                            ],
                        }
                    ],
                },
                "token_count": max(
                    1,
                    (len(prompt) * 68_234 + 205_484) // 205_485,
                ),
                "cost_usd": 0.001,
            }

    attribution_model = RoutedAttributionModel()
    verifier = RatioModel(
        {
            "results": [
                {
                    "claim_id": selected_claim.claim_id,
                    "verdict": "supports",
                    "start_segment_id": "S000001",
                    "end_segment_id": "S000001",
                    "explanation": "The source states the selected claim.",
                }
            ]
        }
    )

    class LiveShapeNetwork(SearchAndReadNetwork):
        async def extract(self, urls, **kwargs):
            self.extract_calls += 1
            return {
                "results": [
                    {
                        "url": self.url,
                        "raw_content": selected_claim.claim_text,
                    }
                ]
            }

    def live_input_estimate(_client, prompt):
        # Reproduce the paid run's input-estimate/prompt-char ratio.
        return max(1, (len(prompt) * 59_673 + 205_484) // 205_485)

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=note_model,
            attribution_model=attribution_model,
            verification_model=verifier,
            tavily_client=LiveShapeNetwork(selected_url),
            budget=EvidenceGapBudget(
                max_tokens=60_000,
                max_cost_usd=1,
                max_search_queries=1,
                max_reads=1,
            ),
            estimate_input_tokens=live_input_estimate,
            estimate_cost_usd=lambda _client, _prompt: 0.001,
        )
    )

    capacity = result.planning_capacity
    assert isinstance(capacity, EvidenceGapPlanningCapacityAudit)
    assert capacity.target_count == 27
    assert capacity.cached_note_count == 176
    assert capacity.prompt_chars < 120_000
    assert capacity.estimated_planning_input_tokens <= 36_000
    assert capacity.reserve_fully_funded is True
    plan_prompt = gap_model.prompts[0]
    assert "unused_by_target_claim_ids" not in plan_prompt
    assert plan_prompt.count('"checked_for_target_claim_ids":') == 176
    assert '"note_id":"note-000001"' in plan_prompt
    assert '"note_id":"note-000176"' in plan_prompt
    assert result.acquisitions[0].outcome == "notes_created"
    assert result.information_yield.new_completed_relation_count == 1
    assert [entry.stage for entry in result.usage] == [
        "cache_review_and_search_plan",
        "read_selection",
        "note_extraction",
        "reattribution",
        "reverification",
    ]


def test_new_note_reattribution_is_limited_to_model_routed_claims():
    report = "# Report\n\nClaim A was reported. Claim B was reported."
    claim_a = _claim(
        report,
        claim_id="claim-0001",
        text="Claim A was reported.",
    )
    claim_b = _claim(
        report,
        claim_id="claim-0002",
        text="Claim B was reported.",
    )
    initial_pairs = tuple(
        _initial(
            claim,
            candidate=None,
            state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
        )
        for claim in (claim_a, claim_b)
    )
    initial_attribution = AttributionResult(
        attributions=tuple(pair[0].attributions[0] for pair in initial_pairs),
        stop_reason=AttributionStopReason.COMPLETED,
    )
    initial_verification = VerificationResult(
        claims=tuple(pair[1].claims[0] for pair in initial_pairs)
    )
    ledger = ResearchLedger(topic="A neutral topic")
    selected_url = "https://new.example/claim-b"
    gap_model = ScriptedModel(
        {
            "cached_candidates": [],
            "queries": [
                {
                    "claim_ids": [claim_b.claim_id],
                    "item_id": "what-1",
                    "query": "record for claim B",
                }
            ],
        },
        {
            "reads": [
                {
                    "url": selected_url,
                    "item_id": "what-1",
                    "claim_ids": [claim_b.claim_id],
                    "independent_from_existing_publishers": True,
                    "publisher_identity": "new.example",
                    "independence_rationale": "A distinct source.",
                }
            ]
        },
    )
    note_model = ScriptedModel(
        {
            "notes": [
                {
                    "item_id": "what-1",
                    "finding": claim_b.claim_text,
                    "quote": claim_b.claim_text,
                }
            ]
        }
    )

    class CaptureRoutedAttribution:
        def __init__(self):
            self.prompts = []

        async def generate(self, prompt):
            self.prompts.append(prompt)
            note = ledger.notes[-1]
            return {
                "content": {
                    "action": "attribute",
                    "claims": [
                        {
                            "claim_id": claim_b.claim_id,
                            "candidates": [
                                {
                                    "note_ref": _note_reference(note),
                                    "inherited_from_claim_id": None,
                                }
                            ],
                        }
                    ],
                },
                "token_count": 5,
                "cost_usd": 0.005,
            }

    attribution_model = CaptureRoutedAttribution()
    verifier = ScriptedModel(
        {
            "results": [
                {
                    "claim_id": claim_b.claim_id,
                    "verdict": "supports",
                    "start_segment_id": "S000001",
                    "end_segment_id": "S000001",
                    "explanation": "The source states claim B.",
                }
            ]
        }
    )

    class ClaimBNetwork(SearchAndReadNetwork):
        async def extract(self, urls, **kwargs):
            self.extract_calls += 1
            return {
                "results": [
                    {"url": self.url, "raw_content": claim_b.claim_text}
                ]
            }

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=note_model,
            attribution_model=attribution_model,
            verification_model=verifier,
            tavily_client=ClaimBNetwork(selected_url),
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=1,
                max_reads=1,
            ),
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert result.acquisitions[0].claim_ids == (claim_b.claim_id,)
    assert len(attribution_model.prompts) == 1
    reattribution_prompt = attribution_model.prompts[0]
    assert claim_b.claim_id in reattribution_prompt
    assert claim_b.claim_text in reattribution_prompt
    assert claim_a.claim_text not in reattribution_prompt
    assert [entry.stage for entry in result.usage][-2:] == [
        "reattribution",
        "reverification",
    ]


def test_cached_only_plan_reserves_one_useful_verification_after_overrun():
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    ledger = ResearchLedger(topic="A neutral topic")
    cached_note = _note(
        ledger,
        "https://cached.example/record",
        claim.claim_text,
    )
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=None,
        state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    )

    class PlanningOverrunModel(ScriptedModel):
        async def generate(self, prompt):
            self.prompts.append(prompt)
            if not self.contents:
                raise AssertionError("unexpected model call")
            return {
                "content": json.dumps(self.contents.pop(0)),
                # Estimate is 5; the explicit 20% headroom covers this sixth
                # provider output/reasoning token without starving verify.
                "token_count": 6,
                "cost_usd": 0.0,
            }

    gap_model = PlanningOverrunModel(
        {
            "cached_candidates": [
                {
                    "claim_id": claim.claim_id,
                    "note_id": cached_note.note_id,
                    "source_id": cached_note.source_id,
                    "independent_from_existing_publishers": True,
                    "publisher_identity": cached_note.publisher,
                    "independence_rationale": "No prior publisher exists.",
                }
            ],
            "queries": [],
        }
    )
    verifier = ScriptedModel(
        {
            "results": [
                {
                    "claim_id": claim.claim_id,
                    "verdict": "supports",
                    "start_segment_id": "S000001",
                    "end_segment_id": "S000001",
                    "explanation": "The cached source states the claim.",
                }
            ]
        }
    )
    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=verifier,
            tavily_client=NoNetwork(),
            budget=EvidenceGapBudget(
                max_tokens=11,
                max_cost_usd=1,
                max_search_queries=0,
                max_reads=0,
            ),
            estimate_input_tokens=lambda _client, _prompt: 5,
            estimate_cost_usd=lambda _client, _prompt: 0.0,
        )
    )

    assert result.planning_capacity is not None
    assert result.planning_capacity.read_selection_estimated_tokens == 0
    assert result.planning_capacity.downstream_action_estimated_tokens == 5
    assert result.planning_capacity.reserve_fully_funded is True
    assert [entry.stage for entry in result.usage] == [
        "cache_review_and_search_plan",
        "reverification",
    ]
    assert result.information_yield.new_completed_relation_count == 1


def test_tight_budget_rebuilds_unpaid_plan_as_cache_only_before_model_call():
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    ledger = ResearchLedger(topic="A neutral topic")
    cached_note = _note(
        ledger,
        "https://cached.example/record",
        claim.claim_text,
    )
    checklist = ResearchChecklist(
        topic="A neutral topic",
        items=(
            ChecklistItem(
                item_id="what-2",
                dimension=ChecklistDimension.WHAT,
                question="What evidence resolves the active claim?",
                priority=1,
                required_source_count=1,
            ),
            ChecklistItem(
                item_id="what-1",
                dimension=ChecklistDimension.WHAT,
                question="Which discarded framing first found the note?",
                priority=2,
                required_source_count=1,
                status=ChecklistStatus.OUT_OF_SCOPE,
            ),
        ),
    )
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=None,
        state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    )
    gap_model = ScriptedModel(
        {
            "cached_candidates": [
                {
                    "claim_id": claim.claim_id,
                    "note_id": cached_note.note_id,
                    "source_id": cached_note.source_id,
                    "independent_from_existing_publishers": True,
                    "publisher_identity": cached_note.publisher,
                    "independence_rationale": "No prior publisher exists.",
                }
            ],
            "queries": [],
        }
    )
    verifier = ScriptedModel(
        {
            "results": [
                {
                    "claim_id": claim.claim_id,
                    "verdict": "supports",
                    "start_segment_id": "S000001",
                    "end_segment_id": "S000001",
                    "explanation": "The cached source states the claim.",
                }
            ]
        }
    )

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=checklist,
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=verifier,
            tavily_client=NoNetwork(),
            budget=EvidenceGapBudget(
                max_tokens=12,
                max_cost_usd=1,
                max_search_queries=1,
                max_reads=1,
            ),
            estimate_input_tokens=lambda _client, _prompt: 5,
            estimate_cost_usd=lambda _client, _prompt: 0.0,
        )
    )

    assert result.stop_reason is EvidenceGapStopReason.COMPLETED
    assert len(gap_model.prompts) == 1
    assert "hard budget of at most 0 web search queries" in gap_model.prompts[0]
    assert cached_note.note_id in gap_model.prompts[0]
    assert 'Checklist item IDs:\n["what-2"]' in gap_model.prompts[0]
    assert result.planning_capacity is not None
    assert result.planning_capacity.advertised_max_search_queries == 0
    assert result.planning_capacity.reserve_fully_funded is True
    assert any(
        entry.get("outcome") == "downgraded_to_cache_only"
        for entry in result.rejected_entries
    )
    assert [entry.stage for entry in result.usage] == [
        "cache_review_and_search_plan",
        "reverification",
    ]


def test_pre_read_reserve_adds_cached_verification_and_one_web_action():
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    ledger = ResearchLedger(topic="A neutral topic")
    cached_note = _note(
        ledger,
        "https://cached.example/record",
        claim.claim_text,
    )
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=None,
        state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    )
    selected_url = "https://new.example/candidate"
    gap_model = ScriptedModel(
        {
            "cached_candidates": [
                {
                    "claim_id": claim.claim_id,
                    "note_id": cached_note.note_id,
                    "source_id": cached_note.source_id,
                    "independent_from_existing_publishers": True,
                    "publisher_identity": cached_note.publisher,
                    "independence_rationale": "No prior publisher exists.",
                }
            ],
            "queries": [
                {
                    "claim_ids": [claim.claim_id],
                    "item_id": "what-1",
                    "query": "another independent account",
                }
            ],
        },
        {"reads": []},
    )
    verifier = ScriptedModel(
        {
            "results": [
                {
                    "claim_id": claim.claim_id,
                    "verdict": "supports",
                    "start_segment_id": "S000001",
                    "end_segment_id": "S000001",
                    "explanation": "The cached source states the claim.",
                }
            ]
        }
    )
    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=verifier,
            tavily_client=SearchAndReadNetwork(selected_url),
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=1,
                max_reads=1,
            ),
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    pre_read = result.verification_reserve_history[0]
    assert pre_read.estimated_tokens == 1
    assert pre_read.minimum_action_estimated_tokens == 5
    assert pre_read.required_downstream_tokens == 6
    assert pre_read.reserved_tokens == 6
    assert pre_read.reserve_fully_funded is True
    assert result.information_yield.new_completed_relation_count == 1


def test_oversized_compact_plan_fails_explicitly_without_model_call():
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=None,
        state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    )
    gap_model = ScriptedModel()
    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ResearchLedger(topic="A neutral topic"),
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=NoNetwork(),
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_planning_prompt_chars=100,
            ),
            plan_prompt_builder=lambda **_kwargs: "x" * 101,
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert result.stop_reason is EvidenceGapStopReason.BUDGET_EXHAUSTED
    assert "character bound" in result.stop_detail
    assert result.planning_capacity is not None
    assert result.planning_capacity.prompt_chars == 101
    assert gap_model.prompts == []


def test_legacy_plan_prompt_builder_without_registry_keyword_still_runs():
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=None,
        state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    )
    builder_calls = []

    def legacy_builder(*, targets, notes, checklist, max_queries):
        builder_calls.append(max_queries)
        return build_evidence_gap_plan_prompt(
            targets=targets,
            notes=notes,
            checklist=checklist,
            max_queries=max_queries,
        )

    gap_model = ScriptedModel({"cached_candidates": [], "queries": []})
    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ResearchLedger(topic="A neutral topic"),
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=NoNetwork(),
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=1,
                max_reads=0,
            ),
            plan_prompt_builder=legacy_builder,
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert result.stop_reason is EvidenceGapStopReason.COMPLETED
    assert builder_calls == [1]
    assert len(gap_model.prompts) == 1


def test_note_protocol_rejects_overflow_beyond_two_notes():
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    ledger = ResearchLedger(topic="A neutral topic")
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=None,
        state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    )
    selected_url = "https://new.example/record"
    gap_model = ScriptedModel(
        {
            "cached_candidates": [],
            "queries": [
                {
                    "claim_ids": [claim.claim_id],
                    "item_id": "what-1",
                    "query": "primary record for the event",
                }
            ],
        },
        {
            "reads": [
                {
                    "url": selected_url,
                    "item_id": "what-1",
                    "claim_ids": [claim.claim_id],
                    "independent_from_existing_publishers": True,
                    "publisher_identity": "new.example",
                    "independence_rationale": "No prior publisher exists.",
                }
            ]
        },
    )
    note_model = ScriptedModel(
        {
            "notes": [
                {
                    "item_id": "what-1",
                    "finding": f"Evidence candidate {index}.",
                    "quote": claim.claim_text,
                }
                for index in range(1, 4)
            ]
        }
    )

    class FirstNewNoteAttribution:
        async def generate(self, _prompt):
            return {
                "content": {
                    "action": "attribute",
                    "claims": [
                        {
                            "claim_id": claim.claim_id,
                            "candidates": [
                                {
                                    "note_ref": _note_reference(
                                        ledger.notes[0]
                                    ),
                                    "inherited_from_claim_id": None,
                                }
                            ],
                        }
                    ],
                },
                "token_count": 5,
                "cost_usd": 0.005,
            }

    verifier = ScriptedModel(
        {
            "results": [
                {
                    "claim_id": claim.claim_id,
                    "verdict": "supports",
                    "start_segment_id": "S000001",
                    "end_segment_id": "S000001",
                    "explanation": "The source states the claim.",
                }
            ]
        }
    )
    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=gap_model,
            note_model=note_model,
            attribution_model=FirstNewNoteAttribution(),
            verification_model=verifier,
            tavily_client=SearchAndReadNetwork(selected_url),
            budget=EvidenceGapBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_search_queries=1,
                max_reads=1,
            ),
            estimate_input_tokens=_estimate_tokens,
            estimate_cost_usd=_estimate_cost,
        )
    )

    assert len(result.added_note_ids) == 2
    assert len(ledger.notes) == 2
    assert any(
        entry.get("stage") == "note_extraction"
        and "rejected 1 overflow entries" in entry.get("error", "")
        for entry in result.rejected_entries
    )


def test_run_cost_cap_exception_propagates_out_of_gap_round():
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=None,
        state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    )
    controller = RunCostController(RunCostBudget(max_cost_usd=1))
    cap_error = RunCostCapReached(
        "synthetic run cap",
        stage="evidence_gap_plan",
        audit=controller.audit(),
    )

    class CappedModel:
        async def generate(self, _prompt):
            raise cap_error

    with pytest.raises(RunCostCapReached) as caught:
        asyncio.run(
            run_evidence_gap_round(
                canonical_draft=report,
                checklist=_checklist(),
                blocks=parse_markdown_blocks(report),
                ledger=ResearchLedger(topic="A neutral topic"),
                initial_attribution=initial_attribution,
                initial_verification=initial_verification,
                gap_model=CappedModel(),
                note_model=ScriptedModel(),
                attribution_model=ScriptedModel(),
                verification_model=ScriptedModel(),
                tavily_client=NoNetwork(),
                budget=EvidenceGapBudget(
                    max_tokens=100,
                    max_cost_usd=1,
                    max_search_queries=1,
                    max_reads=0,
                ),
                estimate_input_tokens=_estimate_tokens,
                estimate_cost_usd=_estimate_cost,
            )
        )

    assert caught.value is cap_error


def test_run_cost_cap_from_note_stage_is_not_downgraded_to_model_error():
    report = "# Report\n\nThe event occurred."
    claim = _claim(report)
    initial_attribution, initial_verification = _initial(
        claim,
        candidate=None,
        state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    )
    selected_url = "https://new.example/record"
    gap_model = ScriptedModel(
        {
            "cached_candidates": [],
            "queries": [
                {
                    "claim_ids": [claim.claim_id],
                    "item_id": "what-1",
                    "query": "primary event record",
                }
            ],
        },
        {
            "reads": [
                {
                    "url": selected_url,
                    "item_id": "what-1",
                    "claim_ids": [claim.claim_id],
                    "independent_from_existing_publishers": True,
                    "publisher_identity": "new.example",
                    "independence_rationale": "No prior publisher exists.",
                }
            ]
        },
    )
    controller = RunCostController(RunCostBudget(max_cost_usd=1))
    cap_error = RunCostCapReached(
        "synthetic note-stage cap",
        stage="evidence_gap_note",
        audit=controller.audit(),
    )

    class CappedNoteModel:
        async def generate(self, _prompt):
            raise cap_error

    with pytest.raises(RunCostCapReached) as caught:
        asyncio.run(
            run_evidence_gap_round(
                canonical_draft=report,
                checklist=_checklist(),
                blocks=parse_markdown_blocks(report),
                ledger=ResearchLedger(topic="A neutral topic"),
                initial_attribution=initial_attribution,
                initial_verification=initial_verification,
                gap_model=gap_model,
                note_model=CappedNoteModel(),
                attribution_model=ScriptedModel(),
                verification_model=ScriptedModel(),
                tavily_client=SearchAndReadNetwork(selected_url),
                budget=EvidenceGapBudget(
                    max_tokens=100,
                    max_cost_usd=1,
                    max_search_queries=1,
                    max_reads=1,
                ),
                estimate_input_tokens=_estimate_tokens,
                estimate_cost_usd=_estimate_cost,
            )
        )

    assert caught.value is cap_error


def _already_checked_gap_state(report: str):
    claim = _claim(report)
    ledger = ResearchLedger(topic="A neutral topic")
    note = _note(
        ledger,
        "https://checked.example/record",
        claim.claim_text,
    )
    candidate = _candidate(note)
    relation = VerifiedSourceRelation(
        claim_id=claim.claim_id,
        source_id=note.source_id,
        url=note.url,
        publisher_domain_proxy=note.publisher,
        candidate_note_ids=(note.note_id,),
        candidate_source_ids=(note.source_id,),
        status=VerificationRecordStatus.COMPLETED,
        semantic_verdict=VerificationVerdict.NOT_ENOUGH_INFORMATION,
        explanation="The source was checked but did not resolve the claim.",
    )
    attribution = AttributionResult(
        attributions=(
            ClaimAttribution(
                claim=claim,
                status=AttributionStatus.CANDIDATE_SOURCES,
                candidates=(candidate,),
            ),
        ),
        stop_reason=AttributionStopReason.COMPLETED,
    )
    verification = VerificationResult(
        claims=(
            ClaimVerification(
                claim=claim,
                state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
                corroboration_target=2,
                relations=(relation,),
                formal_supporting_evidence_count=0,
                publisher_domain_proxy_count=0,
            ),
        )
    )
    return claim, ledger, note, attribution, verification


def test_checked_cached_source_cannot_fund_or_enter_a_cache_only_plan():
    report = "# Report\n\nThe event occurred."
    claim, ledger, _note_record, attribution, verification = (
        _already_checked_gap_state(report)
    )
    gap_model = ScriptedModel()

    result = asyncio.run(
        run_evidence_gap_round(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ledger,
            initial_attribution=attribution,
            initial_verification=verification,
            gap_model=gap_model,
            note_model=ScriptedModel(),
            attribution_model=ScriptedModel(),
            verification_model=ScriptedModel(),
            tavily_client=NoNetwork(),
            budget=EvidenceGapBudget(
                max_tokens=12,
                max_cost_usd=1,
                max_search_queries=1,
                max_reads=1,
            ),
            explicit_target_claim_ids=(claim.claim_id,),
            estimate_input_tokens=lambda _client, _prompt: 5,
            estimate_cost_usd=lambda _client, _prompt: 0.0,
        )
    )

    assert result.stop_reason is EvidenceGapStopReason.BUDGET_EXHAUSTED
    assert gap_model.prompts == []
    assert not any(
        entry.get("outcome") == "downgraded_to_cache_only"
        for entry in result.rejected_entries
    )


def test_checked_cached_hint_and_read_url_are_rejected_as_no_ops():
    report = "# Report\n\nThe event occurred."
    claim, ledger, note, _attribution, verification = (
        _already_checked_gap_state(report)
    )
    target = verification.claims[0]
    hints, _queries, _deferred, plan_rejected, _valid = (
        evidence_gap_module._parse_plan(
            {
                "cached_candidates": [
                    {
                        "claim_id": claim.claim_id,
                        "note_id": note.note_id,
                        "source_id": note.source_id,
                        "independent_from_existing_publishers": True,
                        "publisher_identity": note.publisher,
                        "independence_rationale": "A proposed cached route.",
                    }
                ],
                "queries": [],
            },
            targets=(target,),
            notes=ledger.notes,
            checklist=_checklist(),
            max_queries=1,
        )
    )
    assert hints == ()
    assert any(
        entry.get("error") == "source was already checked for this claim"
        for entry in plan_rejected
    )

    search = GapSearchRecord(
        query=GapSearchQuery(
            claim_ids=(claim.claim_id,),
            item_id="what-1",
            query="another record",
        ),
        results=(
            SearchResult(
                title="Already checked",
                url=note.url,
                snippet="The same source.",
            ),
        ),
    )
    selections, read_rejected = evidence_gap_module._parse_reads(
        {
            "reads": [
                {
                    "url": note.url,
                    "item_id": "what-1",
                    "claim_ids": [claim.claim_id],
                    "independent_from_existing_publishers": True,
                    "publisher_identity": note.publisher,
                    "independence_rationale": "A proposed read route.",
                }
            ]
        },
        targets=(target,),
        searches=(search,),
        cached_hints=(),
        checklist=_checklist(),
        max_reads=1,
    )
    assert selections == ()
    assert any(
        entry.get("error") == "source was already checked for this claim"
        for entry in read_rejected
    )
