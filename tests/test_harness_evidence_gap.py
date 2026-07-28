import asyncio
import json

from open_deep_research.harness.attribution import (
    AttributionResult,
    AttributionStatus,
    AttributionStopReason,
    CandidateSource,
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
    SourceResolution,
    parse_markdown_blocks,
)
from open_deep_research.harness.evidence_gap import (
    EvidenceGapBudget,
    EvidenceGapStopReason,
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


def _claim(report: str) -> AtomicClaim:
    text = "The event occurred."
    start = report.index(text)
    return AtomicClaim(
        claim_id="claim-0001",
        block_id=parse_markdown_blocks(report)[1].block_id,
        selected_text=text,
        claim_text=text,
        anchor_text=text,
        start_char=start,
        end_char=start + len(text),
        citation_requirement=CitationRequirement.EXTERNAL,
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


def _estimate_tokens(client, prompt):
    return 1


def _estimate_cost(client, prompt):
    return 0.001


def test_cached_unused_source_is_checked_before_network_and_can_corroborate():
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
        state=ClaimEvidenceState.SUPPORTED_BELOW_REQUIREMENT,
    )
    gap_model = ScriptedModel(
        {
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "cached_candidates": [
                        {
                            "note_id": second.note_id,
                            "source_id": second.source_id,
                            "independent_from_existing_publishers": True,
                            "publisher_identity": "second",
                            "independence_rationale": "different publisher",
                        }
                    ],
                    "needs_web_search": False,
                    "queries": [],
                }
            ]
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
                    "quote": "The event occurred.",
                    "explanation": "direct statement",
                }
            ]
        },
        {
            "results": [
                {
                    "claim_id": claim.claim_id,
                    "verdict": "supports",
                    "quote": "The event occurred.",
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
        ClaimEvidenceState.CORROBORATED
    )
    assert result.final_verification.claims[0].publisher_domain_proxy_count == 2
    assert result.claim_registry_unchanged is True
    assert ledger.source_cache.keys() == {
        "https://first.example/article",
        "https://second.example/article",
    }
    assert ledger.evidence_gap_history[0].event == "cache_review"


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
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "cached_candidates": [],
                    "needs_web_search": True,
                    "queries": [
                        {
                            "item_id": "what-1",
                            "query": "independent account of the event",
                        }
                    ],
                }
            ]
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
                                    "note_id": note.note_id,
                                    "source_id": note.source_id,
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
                    "quote": "The event occurred.",
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
        state=ClaimEvidenceState.SUPPORTED_BELOW_REQUIREMENT,
    )
    other_domain = "https://bbc.co.uk/other"
    gap_model = ScriptedModel(
        {
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "cached_candidates": [],
                    "needs_web_search": True,
                    "queries": [
                        {"item_id": "what-1", "query": "another account"}
                    ],
                }
            ]
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
                    "quote": "The event occurred.",
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
        ClaimEvidenceState.SUPPORTED_BELOW_REQUIREMENT
    )
