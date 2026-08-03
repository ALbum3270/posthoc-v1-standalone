import asyncio
import json

import pytest

from open_deep_research.harness.attribution import (
    AttributionResult,
    AttributionStatus,
    AttributionStopReason,
    CandidateSource,
    ClaimAttribution,
    _note_reference,
)
from open_deep_research.harness.checklist import (
    ChecklistDimension,
    ChecklistItem,
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
    run_evidence_gap_round,
)
from open_deep_research.harness.ledger import ResearchLedger
from open_deep_research.harness.notes import (
    NoteLocationStatus,
    QuoteSpan,
    create_note,
)
from open_deep_research.harness.verify import (
    ClaimEvidenceState,
    ClaimVerification,
    VerificationRecordStatus,
    VerificationResult,
    VerificationVerdict,
    VerifiedSourceRelation,
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
    assert '"corroboration_target": 2' in prompt
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

    # With no search capacity, code—not the planner's semantic assertion—
    # records the only legal deferral.  This is a completed bounded pass with
    # zero executable capacity, not a budget admission failure.
    assert result.stop_reason is EvidenceGapStopReason.COMPLETED
    assert result.routed_target_claim_ids == ()
    assert result.unrouted_target_claim_ids == (claims[0].claim_id,)
    assert result.deferred_targets[0].reason == "query_capacity_not_allocated"
    assert any(
        entry.get("error", "").startswith("planner_supplied_deferred_target")
        for entry in result.rejected_entries
    )
    assert result.deferred_targets[0].allocation_source == "code_derived"
    assert len(gap_model.prompts) == 1


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


def test_grouped_read_rejects_only_claim_with_existing_publisher():
    """Finance-13 grouped one useful and one duplicate claim on one URL."""

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
    duplicate_note = _note(ledger, selected_url, "The event occurred.")
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

    assert len(result.read_selections) == 1
    assert result.read_selections[0].claim_ids == (first_claim.claim_id,)
    assert result.acquisitions[0].claim_ids == (first_claim.claim_id,)
    assert result.acquisitions[0].outcome == "cache_hit_no_reanalysis"
    assert network.extract_calls == 0
    assert any(
        entry.get("stage") == "read_selection_claim"
        and entry.get("claim_id") == duplicate_claim.claim_id
        and entry.get("error")
        == "publisher domain proxy already supports this claim"
        for entry in result.rejected_entries
    )


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
                max_tokens=15,
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
    assert pre_read_reserve.prerequisite_estimated_tokens == 5
    assert pre_read_reserve.reserved_tokens == 0
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
        and "actual source could not be admitted" in entry.get("error", "")
        and "preserving the verification reserve" in entry.get("error", "")
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
