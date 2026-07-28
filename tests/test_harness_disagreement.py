import asyncio
import json

from open_deep_research.harness.attribution import (
    AttributionResult,
    AttributionStatus,
    AttributionStopReason,
    ClaimAttribution,
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
    parse_markdown_blocks,
)
from open_deep_research.harness.disagreement import (
    DisagreementBudget,
    DisagreementSelection,
    PosthocRetrievalBudget,
    _attempts,
    build_disagreement_selection_prompt,
    run_disagreement_detection,
    shared_posthoc_budget_audit,
)
from open_deep_research.harness.evidence_gap import (
    CachedCandidateHint,
    EvidenceGapResult,
    EvidenceGapStopReason,
)
from open_deep_research.harness.ledger import ResearchLedger
from open_deep_research.harness.notes import (
    NoteLocationStatus,
    QuoteSpan,
)
from open_deep_research.harness.render import render_verified_report
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
    claim_id: str = "claim-0001",
    citation_requirement: CitationRequirement = CitationRequirement.EXTERNAL,
) -> AtomicClaim:
    text = "A measured value was reported."
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


def _initial(
    claims: tuple[AtomicClaim, ...],
) -> tuple[AttributionResult, VerificationResult]:
    attributions = tuple(
        ClaimAttribution(
            claim=claim,
            status=AttributionStatus.NO_CANDIDATE_SOURCE,
        )
        for claim in claims
    )
    verification = tuple(
        ClaimVerification(
            claim=claim,
            state=ClaimEvidenceState.NO_CANDIDATE_SOURCE,
            required_independent_sources=2,
            formal_supporting_evidence_count=0,
            publisher_domain_proxy_count=0,
        )
        for claim in claims
    )
    return (
        AttributionResult(
            attributions=attributions,
            stop_reason=AttributionStopReason.COMPLETED,
        ),
        VerificationResult(claims=verification),
    )


class ScriptedModel:
    def __init__(self, *contents):
        self.contents = list(contents)
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return {
            "content": json.dumps(self.contents.pop(0)),
            "token_count": 5,
            "cost_usd": 0.005,
        }


class EmptySearch:
    def __init__(self):
        self.queries = []

    async def search(self, query, **kwargs):
        self.queries.append(query)
        return {"results": []}

    async def extract(self, urls, **kwargs):
        raise AssertionError("empty results must not be read")


class UnusedModel:
    async def generate(self, prompt):
        raise AssertionError("model must not be called")


def _tokens(client, prompt):
    return 1


def _cost(client, prompt):
    return 0.001


def test_selection_is_neutral_and_excludes_non_external_claims() -> None:
    report = "# Report\n\nA measured value was reported."
    external = _claim(report)
    internal = _claim(
        report,
        claim_id="claim-0002",
        citation_requirement=CitationRequirement.INTERNAL,
    )
    _, verification = _initial((external, internal))

    prompt = build_disagreement_selection_prompt(
        verification.claims,
        max_claims=4,
    )

    assert external.claim_id in prompt
    assert internal.claim_id not in prompt
    assert "never\nbecause you predict" in prompt
    assert "Do not optimize the\nnumber of conflicts" in prompt
    assert all(
        verdict.value in prompt for verdict in VerificationVerdict
    )
    assert "Zero conflicts\nis a normal result" in prompt


def test_zero_conflicts_is_normal_and_each_attempt_is_audited() -> None:
    report = "# Report\n\nA measured value was reported."
    claim = _claim(report)
    attribution, verification = _initial((claim,))
    model = ScriptedModel(
        {
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "reason": "An alternative measurement is informative.",
                }
            ]
        },
        {
            "cached_candidates": [],
            "queries": [
                {
                    "claim_ids": [claim.claim_id],
                    "item_id": "what-1",
                    "query": "alternative measurement account",
                }
            ],
        },
    )
    network = EmptySearch()

    result = asyncio.run(
        run_disagreement_detection(
            canonical_draft=report,
            checklist=_checklist(),
            blocks=parse_markdown_blocks(report),
            ledger=ResearchLedger(topic="A neutral topic"),
            initial_attribution=attribution,
            initial_verification=verification,
            selection_model=model,
            note_model=UnusedModel(),
            attribution_model=UnusedModel(),
            verification_model=UnusedModel(),
            tavily_client=network,
            budget=DisagreementBudget(
                max_tokens=100,
                max_cost_usd=1,
                max_selected_claims=2,
                max_search_queries=2,
                max_reads=0,
            ),
            estimate_input_tokens=_tokens,
            estimate_cost_usd=_cost,
        )
    )

    assert network.queries == ["alternative measurement account"]
    assert len(result.disagreement_search_attempted) == 1
    attempt = result.disagreement_search_attempted[0]
    assert attempt.claim_id == claim.claim_id
    assert attempt.methods == ("web_search",)
    assert attempt.completed_verdict_counts == {
        verdict.value: 0 for verdict in VerificationVerdict
    }
    assert result.conflict_count_is_success_metric is False
    assert "does not establish absence" in result.stop_detail
    assert result.claim_registry_unchanged
    assert result.canonical_draft_unchanged


def test_all_four_verdicts_are_information_not_a_conflict_score() -> None:
    report = "# Report\n\nA measured value was reported."
    claim = _claim(report)
    attribution, initial = _initial((claim,))
    relations = tuple(
        VerifiedSourceRelation(
            claim_id=claim.claim_id,
            source_id=f"source-{index}",
            url=f"https://source{index}.example/page",
            publisher_domain_proxy=f"source{index}.example",
            candidate_note_ids=(f"note-{index}",),
            candidate_source_ids=(f"source-{index}",),
            status=VerificationRecordStatus.COMPLETED,
            semantic_verdict=verdict,
            model_quote="Exact source text.",
            source_quote="Exact source text.",
            span=QuoteSpan(start_char=0, end_char=18),
            location_status=NoteLocationStatus.LOCATABLE,
            is_formal_supporting_evidence=(
                verdict is VerificationVerdict.SUPPORTS
            ),
        )
        for index, verdict in enumerate(VerificationVerdict, start=1)
    )
    final = VerificationResult(
        claims=(
            ClaimVerification(
                claim=claim,
                state=ClaimEvidenceState.CONFLICTING_EVIDENCE,
                required_independent_sources=2,
                relations=relations,
                formal_supporting_evidence_count=1,
                publisher_domain_proxy_count=1,
                publisher_domain_proxies=("source1.example",),
            ),
        )
    )
    acquisition = EvidenceGapResult(
        cached_candidate_hints=(
            CachedCandidateHint(
                claim_id=claim.claim_id,
                note_id="note-000001",
                source_id="source-1",
                publisher_identity="Publisher",
                independence_rationale="Alternative publication.",
            ),
        ),
        stop_reason=EvidenceGapStopReason.COMPLETED,
        stop_detail="bounded pass completed",
        final_attribution=attribution,
        final_verification=final,
    )

    attempts = _attempts(
        (
            DisagreementSelection(
                claim_id=claim.claim_id,
                reason="Alternative check.",
            ),
        ),
        acquisition,
        initial,
    )

    assert len(attempts) == 1
    assert attempts[0].new_completed_relation_count == 4
    assert attempts[0].completed_verdict_counts == {
        verdict.value: 1 for verdict in VerificationVerdict
    }


def test_render_distinguishes_no_attempt_from_attempt_with_no_conflict() -> None:
    empty = VerificationResult(claims=())

    not_attempted = render_verified_report(
        "# Report\n",
        empty,
        disagreement_attempted_count=0,
    )
    attempted = render_verified_report(
        "# Report\n",
        empty,
        disagreement_attempted_count=3,
    )

    assert "仅表示现有候选中未发现；未执行分歧探测" in (
        not_attempted.evidence_summary_line
    )
    assert "分歧探测已尝试 3 条，未发现冲突" in (
        attempted.evidence_summary_line
    )
    assert "不表示已确认无争议" in attempted.evidence_summary_line


def test_shared_budget_caps_the_sum_without_combining_success_metrics() -> None:
    audit = shared_posthoc_budget_audit(
        budget=PosthocRetrievalBudget(
            max_tokens=60_000,
            max_cost_usd=0.10,
        ),
        evidence_gap_tokens=52_000,
        evidence_gap_cost_usd=0.08,
        disagreement_tokens=8_000,
        disagreement_cost_usd=0.02,
    )

    assert audit.within_shared_budget
    assert audit.remaining_tokens == 0
    assert audit.remaining_cost_usd == 0
    assert audit.evidence_gap_tokens + audit.disagreement_tokens == 60_000
