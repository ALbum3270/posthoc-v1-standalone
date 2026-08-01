import asyncio
import hashlib
import json

from open_deep_research.harness.attribution import (
    AttributionResult,
    AttributionStopReason,
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
)
from open_deep_research.harness.evidence_gap import (
    EvidenceGapInformationAudit,
    EvidenceGapResult,
    EvidenceGapStopReason,
    GapReadSelection,
    GapSearchQuery,
    GapSearchRecord,
    GapSourceAcquisition,
)
from open_deep_research.harness.notes import NoteLocationStatus, QuoteSpan
from open_deep_research.harness.recovery import (
    EvidenceRecoveryStopReason,
    RecoveryImportance,
    RecoveryTriageAction,
    RecoveryTriageDecision,
    RecoveryTriageResult,
    RecoveryTriageStatus,
    summarize_evidence_recovery,
    triage_evidence_recovery,
)
from open_deep_research.harness.tools import SearchResult
from open_deep_research.harness.verify import (
    ClaimEvidenceState,
    ClaimVerification,
    VerificationRecordStatus,
    VerificationResult,
    VerificationVerdict,
    VerifiedSourceRelation,
)


def _checklist():
    return ResearchChecklist(
        topic="A question with a sequence and recovery outcomes",
        items=(
            ChecklistItem(
                item_id="what-1",
                dimension=ChecklistDimension.WHAT,
                question="What happened and what followed?",
                priority=1,
                corroboration_target=2,
            ),
        ),
    )


def _claim(claim_id):
    return AtomicClaim(
        claim_id=claim_id,
        block_id=f"block-{claim_id[-4:]}",
        selected_text=f"Specific assertion {claim_id}.",
        claim_text=f"Specific assertion {claim_id}.",
        anchor_text=f"Specific assertion {claim_id}.",
        start_char=0,
        end_char=len(f"Specific assertion {claim_id}."),
        citation_requirement=CitationRequirement.EXTERNAL,
        normalization_status=ClaimNormalizationStatus.LOCATED,
    )


def _verification(claim_id, *, refuted=False):
    claim = _claim(claim_id)
    relation = VerifiedSourceRelation(
        claim_id=claim_id,
        source_id="source-current",
        url="https://current.example/report",
        publisher_domain_proxy="current.example",
        candidate_note_ids=("note-current",),
        candidate_source_ids=("source-current",),
        status=VerificationRecordStatus.COMPLETED,
        semantic_verdict=(
            VerificationVerdict.CONTRADICTS
            if refuted
            else VerificationVerdict.DOES_NOT_SUPPORT
        ),
        explanation="The inspected batch does not establish the assertion.",
        source_quote="A contradictory statement." if refuted else None,
        span=(
            QuoteSpan(start_char=0, end_char=26) if refuted else None
        ),
        location_status=(
            NoteLocationStatus.LOCATABLE if refuted else None
        ),
        is_formal_supporting_evidence=False,
    )
    return ClaimVerification(
        claim=claim,
        state=(
            ClaimEvidenceState.REFUTED
            if refuted
            else ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT
        ),
        corroboration_target=2,
        relations=(relation,),
        formal_supporting_evidence_count=0,
        publisher_domain_proxy_count=0,
    )


class ScriptedTriageModel:
    def __init__(self, decisions):
        self.decisions = decisions
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return {
            "content": json.dumps({"decisions": self.decisions}),
            "token_count": 31,
            "cost_usd": 0.004,
        }


def _decision(claim_id, action):
    research = action == "research_more"
    return {
        "claim_id": claim_id,
        "action": action,
        "importance": "central" if research else "supporting",
        "importance_reason": "It materially affects the requested sequence.",
        "evidence_need": "A dated record of the specific event" if research else None,
        "preferred_source_role": "underlying record" if research else None,
        "query": f"{claim_id} dated record" if research else None,
        "source_document_hint": None,
    }


def test_finance_11_shape_routes_material_facts_to_research_before_editing():
    """Freeze the observed nine editorial targets and required split.

    Recovery triage is a new interface, so there is no old callable whose
    output can form a literal red/green assertion. The regression instead uses
    all nine real claim IDs from finance-11 and fixes the five user-answering
    facts that must reach research before any editor sees them.
    """

    claim_ids = (
        "claim-0033",
        "claim-0035",
        "claim-0036",
        "claim-0040",
        "claim-0047",
        "claim-0053",
        "claim-0055",
        "claim-0056",
        "claim-0068",
    )
    research_ids = {
        "claim-0035",
        "claim-0036",
        "claim-0040",
        "claim-0047",
        "claim-0056",
    }
    records = tuple(
        _verification(claim_id, refuted=claim_id == "claim-0053")
        for claim_id in claim_ids
    )
    decisions = [
        _decision(
            claim_id,
            "research_more"
            if claim_id in research_ids
            else "edit_directly",
        )
        for claim_id in claim_ids
    ]
    model = ScriptedTriageModel(decisions)
    draft = "# Report\n\nNine audited assertions."

    result = asyncio.run(
        triage_evidence_recovery(
            draft,
            checklist=_checklist(),
            verification=VerificationResult(claims=records),
            model_client=model,
        )
    )

    assert result.status is RecoveryTriageStatus.COMPLETE
    assert result.target_claim_ids == claim_ids
    assert set(result.research_target_claim_ids) == research_ids
    assert {
        decision.claim_id: decision.action
        for decision in result.decisions
        if decision.claim_id in {"claim-0053", "claim-0068"}
    } == {
        "claim-0053": RecoveryTriageAction.EDIT_DIRECTLY,
        "claim-0068": RecoveryTriageAction.EDIT_DIRECTLY,
    }
    assert result.failed_claim_ids == ()
    assert result.canonical_draft_sha256 == hashlib.sha256(
        draft.encode()
    ).hexdigest()
    assert result.canonical_draft_unchanged is True
    assert result.claim_registry_unchanged is True
    assert "Do not edit, delete, qualify, or rewrite" in model.prompts[0]
    assert "Research is not a search for\nagreement" in model.prompts[0]


def test_refuted_claim_cannot_be_routed_to_support_seeking():
    model = ScriptedTriageModel([_decision("claim-0053", "research_more")])
    result = asyncio.run(
        triage_evidence_recovery(
            "A draft.",
            checklist=_checklist(),
            verification=VerificationResult(
                claims=(_verification("claim-0053", refuted=True),)
            ),
            model_client=model,
        )
    )

    assert result.status is RecoveryTriageStatus.FAILED
    assert result.decisions == ()
    assert result.failed_claim_ids == ("claim-0053",)
    assert any(
        "refuted_research_rejected" in diagnostic
        for diagnostic in result.diagnostics
    )


def test_zero_yield_56_target_shape_is_not_silently_called_complete():
    """Reproduce 56 targets, one search/read, and zero information yield."""

    claim_ids = tuple(f"claim-{index:04d}" for index in range(1, 57))
    verification = VerificationResult(
        claims=tuple(_verification(claim_id) for claim_id in claim_ids)
    )
    draft_hash = hashlib.sha256(b"draft").hexdigest()
    registry_hash = hashlib.sha256(b"registry").hexdigest()
    triage = RecoveryTriageResult(
        status=RecoveryTriageStatus.COMPLETE,
        target_claim_ids=claim_ids,
        decisions=tuple(
            RecoveryTriageDecision(
                claim_id=claim_id,
                action=RecoveryTriageAction.RESEARCH_MORE,
                importance=RecoveryImportance.CENTRAL,
                importance_reason="The assertion is answer-bearing.",
                evidence_need="A record addressing the assertion",
                preferred_source_role="underlying record",
                query=f"focused query {claim_id}",
            )
            for claim_id in claim_ids
        ),
        canonical_draft_sha256=draft_hash,
        claim_registry_sha256=registry_hash,
    )
    cached_url = "https://cached.example/page"
    query = GapSearchQuery(
        claim_ids=(claim_ids[0],),
        item_id="what-1",
        query="one executed query",
    )
    pass_result = EvidenceGapResult(
        target_claim_ids=claim_ids,
        searches=(
            GapSearchRecord(
                query=query,
                results=(
                    SearchResult(
                        title="Cached result",
                        url=cached_url,
                        snippet="A routing snippet.",
                    ),
                ),
            ),
        ),
        read_selections=(
            GapReadSelection(
                url=cached_url,
                item_id="what-1",
                claim_ids=(claim_ids[0],),
                publisher_identity="Cached publisher",
                independence_rationale="A separate candidate.",
            ),
        ),
        acquisitions=(
            GapSourceAcquisition(
                url=cached_url,
                claim_ids=(claim_ids[0],),
                publisher_identity="Cached publisher",
                cache_hit=True,
                source_chars=100,
                outcome="cache_hit_no_reanalysis",
            ),
        ),
        information_yield=EvidenceGapInformationAudit(
            pass_completed_within_budget=True,
            new_completed_relation_count=0,
        ),
        stop_reason=EvidenceGapStopReason.COMPLETED,
        stop_detail="single evidence-gap pass completed",
        final_attribution=AttributionResult(
            attributions=(),
            stop_reason=AttributionStopReason.COMPLETED,
        ),
        final_verification=verification,
    )

    result = summarize_evidence_recovery(
        triage=triage,
        pass_result=pass_result,
        initial_verification=verification,
        cached_source_urls=(cached_url,),
    )

    assert result.stop_reason is EvidenceRecoveryStopReason.NO_INFORMATION_YIELD
    assert "no new source" in result.stop_detail
    assert "attempted=1/56" in result.stop_detail
    assert result.attempted_claim_ids == (claim_ids[0],)
    assert len(result.unattempted_claim_ids) == 55
    assert result.unread_candidate_urls == ()
    assert result.attempts[0].new_completed_relation_count == 0
    assert all(
        not attempt.attempted for attempt in result.attempts[1:]
    )
