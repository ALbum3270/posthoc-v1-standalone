import asyncio
import hashlib
import json

import pytest

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
    RecoveryQueryRoute,
    RecoverySourceChainAccess,
    RecoveryTriageAction,
    RecoveryTriageDecision,
    RecoveryTriageResult,
    RecoveryTriageStatus,
    summarize_evidence_recovery,
    triage_evidence_recovery,
)
from open_deep_research.harness.source_leads import (
    SourceLeadKind,
    inventory_source_lead_candidates,
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
        "selected_source_lead_id": None,
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


def test_registered_lead_selection_structurally_routes_chain_or_fallback():
    source_cache = {
        "https://secondary.example/report": (
            '1. Records Office. "Dated Filing." Docket #314.\n'
            "A secondary discussion."
        )
    }
    leads = inventory_source_lead_candidates(source_cache)
    filing = next(
        lead
        for lead in leads
        if lead.kind is SourceLeadKind.BIBLIOGRAPHIC_ENTRY
    )
    model = ScriptedTriageModel(
        [
            {
                **_decision("claim-0001", "research_more"),
                "selected_source_lead_id": filing.lead_id,
            },
            _decision("claim-0002", "research_more"),
        ]
    )

    result = asyncio.run(
        triage_evidence_recovery(
            "A draft.",
            checklist=_checklist(),
            verification=VerificationResult(
                claims=(
                    _verification("claim-0001"),
                    _verification("claim-0002"),
                )
            ),
            model_client=model,
            source_cache=source_cache,
        )
    )

    by_id = {decision.claim_id: decision for decision in result.decisions}
    assert by_id["claim-0001"].query_route is RecoveryQueryRoute.SOURCE_CHAIN
    assert by_id["claim-0001"].source_document_hint == (
        'Records Office. "Dated Filing." Docket #314.'
    )
    assert by_id["claim-0002"].query_route is (
        RecoveryQueryRoute.DIRECT_SEARCH_FALLBACK
    )
    assert by_id["claim-0002"].source_document_hint is None
    assert result.source_leads == leads
    assert filing.lead_id in model.prompts[0]
    assert "the current Tavily text extraction boundary can omit" in " ".join(
        result.source_lead_inventory_limitations
    )


def test_unknown_lead_id_is_audited_and_uses_direct_fallback_query():
    decision = _decision("claim-0001", "research_more")
    decision["selected_source_lead_id"] = "lead-ffffffffffffffff"
    result = asyncio.run(
        triage_evidence_recovery(
            "A draft.",
            checklist=_checklist(),
            verification=VerificationResult(
                claims=(_verification("claim-0001"),)
            ),
            model_client=ScriptedTriageModel([decision]),
            source_cache={},
        )
    )

    parsed = result.decisions[0]
    assert parsed.query_route is RecoveryQueryRoute.DIRECT_SEARCH_FALLBACK
    assert parsed.selected_source_lead_id is None
    assert parsed.rejected_source_lead_id == "lead-ffffffffffffffff"
    assert any(
        "unknown_source_lead_fell_back_direct" in diagnostic
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
                query_route=RecoveryQueryRoute.DIRECT_SEARCH_FALLBACK,
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
    assert result.attempts[0].query_route is (
        RecoveryQueryRoute.DIRECT_SEARCH_FALLBACK
    )
    assert result.attempts[0].source_chain_access is (
        RecoverySourceChainAccess.NOT_APPLICABLE_DIRECT_SEARCH
    )
    assert all(
        not attempt.attempted for attempt in result.attempts[1:]
    )


def test_recovery_rejects_target_drift_between_triage_and_gap_executor():
    """A future caller cannot silently execute a different frozen scope."""

    claim_id = "claim-0001"
    verification = VerificationResult(claims=(_verification(claim_id),))
    triage = RecoveryTriageResult(
        status=RecoveryTriageStatus.COMPLETE,
        target_claim_ids=(claim_id,),
        decisions=(
            RecoveryTriageDecision(
                claim_id=claim_id,
                action=RecoveryTriageAction.RESEARCH_MORE,
                importance=RecoveryImportance.CENTRAL,
                importance_reason="The assertion is answer-bearing.",
                evidence_need="A record addressing the assertion",
                preferred_source_role="underlying record",
                query="focused query",
                query_route=RecoveryQueryRoute.DIRECT_SEARCH_FALLBACK,
            ),
        ),
        canonical_draft_sha256=hashlib.sha256(b"draft").hexdigest(),
        claim_registry_sha256=hashlib.sha256(b"registry").hexdigest(),
    )
    drifted_pass = EvidenceGapResult(
        target_claim_ids=("claim-9999",),
        stop_reason=EvidenceGapStopReason.COMPLETED,
        stop_detail="wrong target was executed",
        final_attribution=AttributionResult(
            attributions=(),
            stop_reason=AttributionStopReason.COMPLETED,
        ),
        final_verification=verification,
    )

    with pytest.raises(
        ValueError,
        match="gap executor targets must equal frozen recovery targets",
    ):
        summarize_evidence_recovery(
            triage=triage,
            pass_result=drifted_pass,
            initial_verification=verification,
            cached_source_urls=(),
        )


@pytest.mark.parametrize(
    ("acquisition_outcome", "expected_access"),
    [
        (
            "notes_extracted",
            RecoverySourceChainAccess.LEAD_FOUND_AND_READABLE,
        ),
        (
            "read_error",
            RecoverySourceChainAccess.LEAD_FOUND_BUT_UNREADABLE,
        ),
    ],
)
def test_source_chain_audit_distinguishes_readable_from_unreadable(
    acquisition_outcome,
    expected_access,
):
    claim_id = "claim-0001"
    verification = VerificationResult(claims=(_verification(claim_id),))
    triage = RecoveryTriageResult(
        status=RecoveryTriageStatus.COMPLETE,
        target_claim_ids=(claim_id,),
        decisions=(
            RecoveryTriageDecision(
                claim_id=claim_id,
                action=RecoveryTriageAction.RESEARCH_MORE,
                importance=RecoveryImportance.CENTRAL,
                importance_reason="The assertion is answer-bearing.",
                evidence_need="A dated filing",
                preferred_source_role="underlying record",
                query="Dated Filing docket 314",
                selected_source_lead_id="lead-1111111111111111",
                source_document_hint='Records Office. "Dated Filing."',
                query_route=RecoveryQueryRoute.SOURCE_CHAIN,
            ),
        ),
        canonical_draft_sha256=hashlib.sha256(b"draft").hexdigest(),
        claim_registry_sha256=hashlib.sha256(b"registry").hexdigest(),
    )
    result_url = "https://records.example/filing"
    query = GapSearchQuery(
        claim_ids=(claim_id,),
        item_id="what-1",
        query="Dated Filing docket 314",
    )
    pass_result = EvidenceGapResult(
        target_claim_ids=(claim_id,),
        searches=(
            GapSearchRecord(
                query=query,
                results=(
                    SearchResult(
                        title="Dated Filing",
                        url=result_url,
                        snippet="A filing result.",
                    ),
                ),
            ),
        ),
        read_selections=(
            GapReadSelection(
                url=result_url,
                item_id="what-1",
                claim_ids=(claim_id,),
                publisher_identity="Records Office",
                independence_rationale="The selected record merits reading.",
            ),
        ),
        acquisitions=(
            GapSourceAcquisition(
                url=result_url,
                claim_ids=(claim_id,),
                publisher_identity="Records Office",
                cache_hit=False,
                outcome=acquisition_outcome,
                error=(
                    "unreadable document"
                    if acquisition_outcome == "read_error"
                    else None
                ),
            ),
        ),
        information_yield=EvidenceGapInformationAudit(
            pass_completed_within_budget=True
        ),
        stop_reason=EvidenceGapStopReason.COMPLETED,
        stop_detail="single pass completed",
        final_attribution=AttributionResult(
            attributions=(),
            stop_reason=AttributionStopReason.COMPLETED,
        ),
        final_verification=verification,
    )

    recovery = summarize_evidence_recovery(
        triage=triage,
        pass_result=pass_result,
        initial_verification=verification,
        cached_source_urls=(),
    )

    assert recovery.attempts[0].source_chain_access is expected_access
    assert recovery.attempts[0].source_document_hint == (
        'Records Office. "Dated Filing."'
    )
