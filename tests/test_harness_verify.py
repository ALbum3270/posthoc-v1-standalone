from __future__ import annotations

import asyncio
import hashlib
import json
import re

import pytest
from pydantic import ValidationError

from open_deep_research.harness.attribution import (
    AttributionError,
    AttributionStatus,
    CandidateSource,
    ClaimAttribution,
)
from open_deep_research.harness.budget import (
    RunCostBudget,
    RunCostCapReached,
    RunCostController,
)
from open_deep_research.harness.claims import (
    AtomicClaim,
    CitationRequirement,
    ClaimDerivation,
    ClaimNormalizationStatus,
    ClaimRepresentationVersion,
    ContextSpan,
    ReportSurfaceSpan,
    SourceResolution,
)
from open_deep_research.harness.notes import (
    NoteLocationStatus,
    QuoteSpan,
    create_note,
)
from open_deep_research.harness.numeric_consistency import (
    NumericConsistencyStatus,
)
from open_deep_research.harness.source_provenance import (
    SourceLineageAssessment,
    SourceLineageStatus,
    SourceRole,
)
from open_deep_research.harness.truth_conditions import (
    ClaimCoverageState,
    ElementAssessmentExecutionStatus,
    ElementizationProposal,
    ElementizationReview,
    ElementizationSemanticStatus,
    ExecutionCompleteness,
    build_truth_condition_registry,
)
from open_deep_research.harness.verify import (
    ClaimEvidenceState,
    ClaimVerification,
    VerificationBudget,
    VerificationRecordStatus,
    VerificationResult,
    VerificationSettings,
    VerificationVerdict,
    VerifiedSourceRelation,
    build_verification_prompt,
    verify_attributions,
)


class ScriptedVerificationModel:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> object:
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, dict) and "content" in response:
            return response
        return {
            "content": json.dumps(response),
            "token_count": 11,
            "cost_usd": 0.01,
        }


class EchoSupportModel:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.claim_batches: list[tuple[str, ...]] = []

    async def generate(self, prompt: str) -> dict[str, object]:
        self.prompts.append(prompt)
        match = re.search(
            r"Claims:\n(.*?)\n\nBEGIN COMPLETE CACHED SOURCE WITH",
            prompt,
            flags=re.DOTALL,
        )
        assert match is not None
        claims = json.loads(match.group(1))
        claim_ids = tuple(entry["claim_id"] for entry in claims)
        self.claim_batches.append(claim_ids)
        return {
            "content": json.dumps(
                {
                    "results": [
                        {
                            "claim_id": claim_id,
                            "verdict": "supports",
                            "start_segment_id": "S000002",
                            "end_segment_id": "S000002",
                            "explanation": "The complete source states it.",
                        }
                        for claim_id in claim_ids
                    ]
                }
            ),
            "token_count": 5,
            "cost_usd": 0.005,
        }


def _claim(claim_id: str, text: str | None = None) -> AtomicClaim:
    claim_text = text or f"Self-contained assertion {claim_id}."
    return AtomicClaim(
        claim_id=claim_id,
        block_id=f"block-{claim_id}",
        selected_text=claim_text,
        claim_text=claim_text,
        anchor_text=claim_text,
        start_char=0,
        end_char=len(claim_text),
        citation_requirement=CitationRequirement.EXTERNAL,
        source_resolution=SourceResolution.DIRECT,
        normalization_status=ClaimNormalizationStatus.LOCATED,
    )


def _layered_claim(
    claim_id: str,
    *,
    report_surface_text: str,
    retrieval_gloss: str,
    necessary_context: tuple[str, ...] = (),
) -> AtomicClaim:
    context_spans = tuple(
        ContextSpan(
            text=text,
            start_char=index * 100,
            end_char=index * 100 + len(text),
        )
        for index, text in enumerate(necessary_context, start=1)
    )
    return AtomicClaim(
        claim_id=claim_id,
        block_id=f"block-{claim_id}",
        representation_version=ClaimRepresentationVersion.LAYERED_V2,
        report_surface=ReportSurfaceSpan(
            block_id=f"block-{claim_id}",
            text=report_surface_text,
            start_char=0,
            end_char=len(report_surface_text),
            start_segment_id="S000001",
            end_segment_id="S000001",
            span_registry_id="report-registry",
            report_text_sha256="0" * 64,
            segmentation_version="test-v1",
        ),
        selected_text=report_surface_text,
        claim_text=retrieval_gloss,
        derivation=ClaimDerivation(),
        anchor_text=report_surface_text,
        start_char=0,
        end_char=len(report_surface_text),
        context_spans=context_spans,
        citation_requirement=CitationRequirement.EXTERNAL,
        source_resolution=SourceResolution.DIRECT,
        normalization_status=ClaimNormalizationStatus.LOCATED,
    )


def _candidate(
    *,
    claim: AtomicClaim,
    url: str,
    note_id: str,
    location_status: NoteLocationStatus = NoteLocationStatus.LOCATABLE,
) -> CandidateSource:
    from open_deep_research.harness.notes import source_id_for_url

    return CandidateSource(
        note_id=note_id,
        source_id=source_id_for_url(url),
        item_id="item-from-any-checklist-dimension",
        publisher=url.split("/")[2].removeprefix("www."),
        url=url,
        location_status=location_status,
        resolution=SourceResolution.DIRECT,
    )


def _lineage(
    *,
    source_id: str,
    url: str,
    source_text: str,
    lineage_id: str,
    status: SourceLineageStatus = SourceLineageStatus.CONFIRMED,
    independence_eligible: bool = True,
) -> SourceLineageAssessment:
    return SourceLineageAssessment(
        source_id=source_id,
        url=url,
        status=status,
        source_role=SourceRole.INDEPENDENT_REPORTING,
        originating_organization=lineage_id,
        lineage_id=lineage_id,
        independence_eligible=independence_eligible,
        evaluator="independent-reviewer",
        rationale="The exact page identifies its originating newsroom.",
        source_text_sha256=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        basis_quote=source_text,
        basis_start_char=0,
        basis_end_char=len(source_text),
    )


def test_verifier_judges_exact_report_surface_not_model_retrieval_gloss() -> None:
    claim = _layered_claim(
        "claim-qualified",
        report_surface_text="两人被指出参与推广，并被纳入集体诉讼。",
        retrieval_gloss="两人参与推广并被纳入集体诉讼。",
        necessary_context=("报道讨论了两位代言人。",),
    )

    prompt = build_verification_prompt(
        url="https://example.test/source",
        source_text="A cached source.",
        claims=(claim,),
    )

    match = re.search(
        r"Claims:\n(.*?)\n\nBEGIN COMPLETE CACHED SOURCE WITH",
        prompt,
        flags=re.DOTALL,
    )
    assert match is not None
    payload = json.loads(match.group(1))
    assert payload == [
        {
            "claim_id": "claim-qualified",
            "necessary_context": ["报道讨论了两位代言人。"],
            "report_surface_text": "两人被指出参与推广，并被纳入集体诉讼。",
            "retrieval_gloss": "两人参与推广并被纳入集体诉讼。",
        }
    ]
    assert "report_surface_text" in prompt
    assert "authoritative statement" in prompt
    assert "must not strengthen, weaken, or replace" in prompt


def test_numeric_gate_uses_report_surface_instead_of_lossy_retrieval_gloss() -> None:
    claim = _layered_claim(
        "claim-layered-numeric",
        report_surface_text="崩溃时，FTX 报告的资产约为 9000 万美元。",
        retrieval_gloss="FTX held about 900 million dollars in assets.",
    )
    url = "https://www.congress.gov/example"
    source = "The company held $900 million in liquid assets."
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(
                    claim.claim_id,
                    "supports",
                    ("S000001", "S000001"),
                )
            ]
        }
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    claim,
                    _candidate(
                        claim=claim,
                        url=url,
                        note_id="note-layered-numeric",
                    ),
                )
            ],
            source_cache={url: source},
            model_client=model,
        )
    )

    relation = result.claims[0].relations[0]
    assert relation.semantic_verdict is VerificationVerdict.SUPPORTS
    assert relation.numeric_consistency_status is NumericConsistencyStatus.MISMATCH
    assert "90000000" in (relation.numeric_consistency_detail or "")
    assert "900000000" in (relation.numeric_consistency_detail or "")
    assert relation.is_formal_supporting_evidence is False


def _attribution(
    claim: AtomicClaim,
    *candidates: CandidateSource,
) -> ClaimAttribution:
    if candidates:
        return ClaimAttribution(
            claim=claim,
            status=AttributionStatus.CANDIDATE_SOURCES,
            candidates=tuple(candidates),
        )
    unresolved = claim.model_copy(
        update={"source_resolution": SourceResolution.UNRESOLVED}
    )
    return ClaimAttribution(
        claim=unresolved,
        status=AttributionStatus.NO_CANDIDATE_SOURCE,
    )


def _result(
    claim_id: str,
    verdict: str,
    segment_range: tuple[str, str] | None,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "verdict": verdict,
        "start_segment_id": (
            segment_range[0] if segment_range is not None else None
        ),
        "end_segment_id": (
            segment_range[1] if segment_range is not None else None
        ),
        "explanation": "Auditable semantic judgement.",
    }


def _capacity_result(
    claim_id: str,
    disposition: str,
    verdict: str | None = None,
    segment_range: tuple[str, str] | None = None,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "disposition": disposition,
        "verdict": verdict,
        "start_segment_id": (
            segment_range[0] if segment_range is not None else None
        ),
        "end_segment_id": (
            segment_range[1] if segment_range is not None else None
        ),
        "explanation": "Auditable capacity retry outcome.",
    }


def _truth_registry(
    *claims_and_elements: tuple[AtomicClaim, tuple[str, ...]],
    semantic_status: ElementizationSemanticStatus = (
        ElementizationSemanticStatus.COMPLETE
    ),
):
    surfaces = {
        claim.claim_id: (
            claim.report_surface.text
            if claim.report_surface is not None
            else claim.selected_text
        )
        for claim, _ in claims_and_elements
    }
    proposals = tuple(
        ElementizationProposal(
            claim_id=claim.claim_id,
            elements=elements,
            rationale="proposal",
        )
        for claim, elements in claims_and_elements
    )
    reviews = tuple(
        ElementizationReview(
            claim_id=claim.claim_id,
            semantic_status=semantic_status,
            elements=elements,
            missing_conditions=(
                ()
                if semantic_status is ElementizationSemanticStatus.COMPLETE
                else ("The registered denominator may omit a condition.",)
            ),
            rationale="independent review",
        )
        for claim, elements in claims_and_elements
    )
    return build_truth_condition_registry(
        surfaces,
        proposals=proposals,
        reviews=reviews,
    )


def _element_result(
    element_id: str,
    verdict: str,
    segment_range: tuple[str, str] | None = None,
) -> dict[str, object]:
    return {
        "element_id": element_id,
        "verdict": verdict,
        "start_segment_id": (
            segment_range[0] if segment_range is not None else None
        ),
        "end_segment_id": (
            segment_range[1] if segment_range is not None else None
        ),
        "explanation": "Element-level semantic judgement.",
    }


def _element_claim_result(
    claim_id: str,
    *elements: dict[str, object],
) -> dict[str, object]:
    return {"claim_id": claim_id, "elements": list(elements)}


def test_groups_by_url_sorts_claim_ids_and_never_exceeds_twenty() -> None:
    url = "https://one.example/full"
    source = (
        "Beginning of the complete source. Exact supporting sentence. "
        "TAIL-SENTINEL-THAT-MUST-NOT-BE-TRUNCATED"
    )
    claims = [_claim(f"claim-{index:02d}") for index in range(21)]
    attributions = [
        _attribution(
            claim,
            _candidate(
                claim=claim,
                url=url,
                note_id=f"note-{index:03d}",
            ),
        )
        for index, claim in enumerate(reversed(claims))
    ]
    model = EchoSupportModel()

    result = asyncio.run(
        verify_attributions(
            attributions,
            source_cache={url: source},
            model_client=model,
        )
    )

    assert tuple(map(len, model.claim_batches)) == (20, 1)
    assert model.claim_batches[0] == tuple(
        f"claim-{index:02d}" for index in range(20)
    )
    assert model.claim_batches[1] == ("claim-20",)
    assert all("<S000001>Beginning" in prompt for prompt in model.prompts)
    assert all("<S000002>Exact supporting" in prompt for prompt in model.prompts)
    assert all(
        "TAIL-SENTINEL-THAT-MUST-NOT-BE-TRUNCATED" in prompt
        for prompt in model.prompts
    )
    assert len(result.claims) == 21
    legacy_payload = result.model_dump(mode="json")
    assert "truth_condition_registry_sha256" not in legacy_payload
    assert all(
        "truth_condition_aggregate" not in item
        for item in legacy_payload["claims"]
    )
    assert all(
        "element_supporting_domain_proxy_count" not in item
        and "element_supporting_domain_proxies" not in item
        for item in legacy_payload["claims"]
    )
    assert all(
        "element_relations" not in relation
        for item in legacy_payload["claims"]
        for relation in item["relations"]
    )
    assert all(
        claim.state == ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER
        for claim in result.claims
    )
    assert all(
        claim.model_dump(mode="json")["state"]
        == "supported_single_domain_proxy"
        for claim in result.claims
    )
    assert (
        ClaimEvidenceState("supported_below_requirement")
        is ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER
    )
    assert all(
        claim.publisher_domain_proxy_count == 1
        for claim in result.claims
    )
    restored = VerificationResult.model_validate(legacy_payload)
    assert restored.model_dump(mode="json") == legacy_payload
    with pytest.raises(ValidationError):
        VerificationSettings(batch_size=21)


def test_malformed_sibling_is_retried_without_discarding_valid_result() -> None:
    url = "https://retry.example/article"
    source = "First exact passage. Second exact passage."
    first = _claim("claim-1")
    second = _claim("claim-2")
    attributions = [
        _attribution(
            first,
            _candidate(claim=first, url=url, note_id="note-1"),
        ),
        _attribution(
            second,
            _candidate(claim=second, url=url, note_id="note-2"),
        ),
    ]
    model = ScriptedVerificationModel(
        {
            "content": json.dumps(
                {
                    "results": [
                        _result(
                            "claim-1",
                            "supports",
                            ("S000001", "S000001"),
                        ),
                        _result("claim-2", "supports", None),
                    ]
                }
            ),
            "token_count": 17,
            "cost_usd": 0.017,
        },
        {
            "content": json.dumps(
                {
                    "results": [
                        _result(
                            "claim-2",
                            "supports",
                            ("S000002", "S000002"),
                        )
                    ]
                }
            ),
            "token_count": 13,
            "cost_usd": 0.013,
        },
    )

    result = asyncio.run(
        verify_attributions(
            attributions,
            source_cache={url: source},
            model_client=model,
        )
    )

    assert len(model.prompts) == 2
    assert '"claim_id": "claim-1"' in model.prompts[0]
    assert '"claim_id": "claim-1"' not in model.prompts[1]
    assert '"claim_id": "claim-2"' in model.prompts[1]
    assert result.usage[0].outcome == "partial_malformed"
    assert result.usage[1].retry is True
    assert result.total_tokens == 30
    assert result.total_cost_usd == pytest.approx(0.03)
    assert all(
        claim.relations[0].is_formal_supporting_evidence
        for claim in result.claims
    )


def test_pointer_gate_and_unlocatable_note_history() -> None:
    claim = _claim("claim-evidence")
    strict_url = "https://strict.example/a"
    repair_url = "https://repair.example/b"
    rescued_url = "https://rescued.example/c"
    failed_url = "https://unlocated.example/d"
    historical_note = create_note(
        item_id="other-item",
        finding="Potentially useful context.",
        quote="A model paraphrase absent from the source.",
        url=rescued_url,
        source_text="Original wording is materially different.",
    )
    historical_snapshot = historical_note.model_dump(mode="json")
    attributions = [
        _attribution(
            claim,
            _candidate(claim=claim, url=strict_url, note_id="note-strict"),
            _candidate(claim=claim, url=repair_url, note_id="note-repair"),
            CandidateSource(
                note_id="note-historical-unlocatable",
                source_id=historical_note.source_id,
                item_id=historical_note.item_id,
                publisher=historical_note.publisher,
                url=historical_note.url,
                location_status=historical_note.location_status,
                resolution=SourceResolution.DIRECT,
            ),
            _candidate(
                claim=claim,
                url=failed_url,
                note_id="note-still-unlocated",
            ),
        )
    ]
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(
                    claim.claim_id,
                    "supports",
                    ("S000001", "S000001"),
                )
            ]
        },
        {
            "results": [
                _result(
                    claim.claim_id,
                    "supports",
                    ("S000001", "S000001"),
                )
            ]
        },
        {
            "results": [
                _result(
                    claim.claim_id,
                    "supports",
                    ("S000001", "S000001"),
                )
            ]
        },
        {
            "results": [
                _result(
                    claim.claim_id,
                    "supports",
                    ("S999999", "S999999"),
                )
            ]
        },
    )

    result = asyncio.run(
        verify_attributions(
            attributions,
            source_cache={
                strict_url: "Exact source-authored passage.",
                repair_url: "AlphaBeta 2026.",
                rescued_url: "Original wording is materially different.",
                failed_url: "Original wording is materially different.",
            },
            model_client=model,
            required_independent_sources={claim.claim_id: 3},
        )
    )

    verified = result.claims[0]
    by_url = {relation.url: relation for relation in verified.relations}
    assert by_url[strict_url].location_status == NoteLocationStatus.LOCATABLE
    assert by_url[repair_url].location_status == NoteLocationStatus.LOCATABLE
    assert by_url[repair_url].model_quote is None
    assert by_url[repair_url].source_quote == "AlphaBeta 2026."
    assert by_url[repair_url].start_segment_id == "S000001"
    assert by_url[repair_url].span_registry_id is not None
    assert by_url[rescued_url].is_formal_supporting_evidence is True
    assert by_url[rescued_url].source_quote == (
        "Original wording is materially different."
    )
    assert by_url[failed_url].status == (
        VerificationRecordStatus.QUOTE_UNLOCATABLE
    )
    assert by_url[failed_url].semantic_verdict == VerificationVerdict.SUPPORTS
    assert by_url[failed_url].is_formal_supporting_evidence is False
    assert verified.formal_supporting_evidence_count == 3
    assert verified.publisher_domain_proxy_count == 3
    assert verified.state == ClaimEvidenceState.SUPPORTED_MULTIPLE_DOMAIN_PROXIES
    assert historical_note.model_dump(mode="json") == historical_snapshot


def test_support_and_contradiction_aggregate_as_conflicting_evidence() -> None:
    claim = _claim("claim-conflict")
    support_url = "https://support.example/article"
    contradict_url = "https://contradict.example/article"
    attributions = [
        _attribution(
            claim,
            _candidate(claim=claim, url=support_url, note_id="note-support"),
            _candidate(
                claim=claim,
                url=contradict_url,
                note_id="note-contradict",
            ),
        )
    ]
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(
                    claim.claim_id,
                    "contradicts",
                    ("S000001", "S000001"),
                )
            ]
        },
        {
            "results": [
                _result(
                    claim.claim_id,
                    "supports",
                    ("S000001", "S000001"),
                )
            ]
        },
    )

    result = asyncio.run(
        verify_attributions(
            attributions,
            source_cache={
                support_url: "The source explicitly supports the assertion.",
                contradict_url: (
                    "The source explicitly contradicts the assertion."
                ),
            },
            model_client=model,
        )
    )

    verified = result.claims[0]
    assert verified.state == ClaimEvidenceState.CONFLICTING_EVIDENCE
    assert {
        relation.semantic_verdict for relation in verified.relations
    } == {
        VerificationVerdict.SUPPORTS,
        VerificationVerdict.CONTRADICTS,
    }
    assert verified.formal_supporting_evidence_count == 1


def test_invalid_segment_pointer_cannot_become_formal_evidence() -> None:
    claim = _claim("claim-composite")
    url = "https://composite.example/article"
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(
                    claim.claim_id,
                    "supports",
                    ("S999998", "S999999"),
                )
            ]
        }
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    claim,
                    _candidate(
                        claim=claim,
                        url=url,
                        note_id="note-composite",
                    ),
                )
            ],
            source_cache={url: "AlphaBeta"},
            model_client=model,
        )
    )

    relation = result.claims[0].relations[0]
    assert relation.status == VerificationRecordStatus.QUOTE_UNLOCATABLE
    assert relation.error is not None
    assert "unknown start_segment_id" in relation.error
    assert relation.start_segment_id == "S999998"
    assert relation.source_quote is None
    assert relation.span is None
    assert relation.is_formal_supporting_evidence is False


def test_finance07_style_cleaned_source_artifact_is_copied_by_code() -> None:
    """Reproduce the free-copy failure shape from finance-07.

    The naturalized model quote omitted conversion artifacts and therefore
    could not be located. The model now selects the segment; code retains the
    exact cached bytes instead of asking it to reproduce them.
    """

    claim = _claim("claim-artifact")
    url = "https://artifact.example/article"
    source = (
        "The record listed $22currency-dollar22\\$22$ 22 units each."
    )
    naturalized_free_copy = "The record listed $22 units each."
    assert naturalized_free_copy not in source
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(
                    claim.claim_id,
                    "supports",
                    ("S000001", "S000001"),
                )
            ]
        }
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    claim,
                    _candidate(claim=claim, url=url, note_id="note-artifact"),
                )
            ],
            source_cache={url: source},
            model_client=model,
        )
    )

    relation = result.claims[0].relations[0]
    assert '"quote"' not in model.prompts[0]
    assert '"start_segment_id"' in model.prompts[0]
    assert relation.status is VerificationRecordStatus.COMPLETED
    assert relation.model_quote is None
    assert relation.source_quote == source
    assert source[relation.span.start_char : relation.span.end_char] == source
    assert relation.is_formal_supporting_evidence is True


def test_finance18_tenfold_currency_mismatch_cannot_become_formal_support() -> None:
    """Regression for a real false-positive verifier result from finance-18.

    The model called the relation supports and selected a real Congressional
    Research Service passage, but the report claimed 90 million dollars while
    the source says 900 million.  Quote location therefore remains successful;
    only formal evidence admission must fail.
    """

    claim = _claim(
        "claim-0040",
        "崩溃时，FTX 报告的资产约为 9000 万美元。",
    )
    url = "https://www.congress.gov/crs_external_products/IN/PDF/IN12047/IN12047.1.pdf"
    source = (
        "The company held $900 million in easily sellable assets compared "
        "to $9 billion in liabilities."
    )
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(
                    claim.claim_id,
                    "supports",
                    ("S000001", "S000001"),
                )
            ]
        }
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    claim,
                    _candidate(
                        claim=claim,
                        url=url,
                        note_id="note-congress-0001",
                    ),
                )
            ],
            source_cache={url: source},
            model_client=model,
        )
    )

    verified = result.claims[0]
    relation = verified.relations[0]
    assert relation.status is VerificationRecordStatus.COMPLETED
    assert relation.semantic_verdict is VerificationVerdict.SUPPORTS
    assert relation.source_quote == source
    assert relation.numeric_consistency_status is NumericConsistencyStatus.MISMATCH
    assert "90000000" in (relation.numeric_consistency_detail or "")
    assert "900000000" in (relation.numeric_consistency_detail or "")
    assert relation.is_formal_supporting_evidence is False
    assert verified.formal_supporting_evidence_count == 0
    assert verified.state is ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT


def test_matching_currency_range_can_remain_formal_evidence() -> None:
    claim = _claim(
        "claim-numeric-range",
        "负债接近 90-100 亿美元。",
    )
    url = "https://numeric.example/report"
    source = "The report listed approximately $9 billion in liabilities."
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(
                    claim.claim_id,
                    "supports",
                    ("S000001", "S000001"),
                )
            ]
        }
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    claim,
                    _candidate(
                        claim=claim,
                        url=url,
                        note_id="note-numeric-range",
                    ),
                )
            ],
            source_cache={url: source},
            model_client=model,
        )
    )

    relation = result.claims[0].relations[0]
    assert relation.numeric_consistency_status is NumericConsistencyStatus.ALIGNED
    assert relation.is_formal_supporting_evidence is True
    assert result.claims[0].state is ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER


def test_numeric_mismatch_cannot_be_constructed_as_formal_evidence() -> None:
    """The formal-support invariant holds even outside verifier orchestration."""

    with pytest.raises(ValidationError, match="non-mismatching numeric"):
        VerifiedSourceRelation(
            claim_id="claim-0040",
            source_id="source-congress",
            url="https://www.congress.gov/example",
            publisher_domain_proxy="congress.gov",
            candidate_note_ids=("note-congress-0001",),
            candidate_source_ids=("source-congress",),
            status=VerificationRecordStatus.COMPLETED,
            semantic_verdict=VerificationVerdict.SUPPORTS,
            source_quote="$900 million in liquid assets.",
            span=QuoteSpan(start_char=0, end_char=32),
            location_status=NoteLocationStatus.LOCATABLE,
            numeric_consistency_status=NumericConsistencyStatus.MISMATCH,
            is_formal_supporting_evidence=True,
        )


def test_verifier_range_crosses_adjacent_units_but_stays_contiguous() -> None:
    claim = _claim("claim-adjacent")
    url = "https://adjacent.example/article"
    source = "The first fact holds.\n\nTherefore the result follows."
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(
                    claim.claim_id,
                    "supports",
                    ("S000001", "S000002"),
                )
            ]
        }
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    claim,
                    _candidate(claim=claim, url=url, note_id="note-adjacent"),
                )
            ],
            source_cache={url: source},
            model_client=model,
        )
    )

    relation = result.claims[0].relations[0]
    assert relation.source_quote == source
    assert relation.start_segment_id == "S000001"
    assert relation.end_segment_id == "S000002"


def test_oversized_verifier_range_is_rejected_whole_without_truncation() -> None:
    claim = _claim("claim-oversized")
    url = "https://oversized.example/article"
    source = " ".join(f"Sentence {index}." for index in range(1, 14))
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(
                    claim.claim_id,
                    "supports",
                    ("S000001", "S000013"),
                )
            ]
        },
        {"results": [_capacity_result(claim.claim_id, "cannot_narrow")]},
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    claim,
                    _candidate(claim=claim, url=url, note_id="note-oversized"),
                )
            ],
            source_cache={url: source},
            model_client=model,
        )
    )

    relation = result.claims[0].relations[0]
    assert relation.status is VerificationRecordStatus.QUOTE_UNLOCATABLE
    assert relation.source_quote is None
    assert relation.span is None
    assert relation.error is not None
    assert "span_too_many_segments" in relation.error
    assert relation.start_segment_id == "S000001"
    assert relation.end_segment_id == "S000013"
    assert result.claims[0].state is ClaimEvidenceState.SUPPORT_QUOTE_UNLOCATABLE
    assert len(model.prompts) == 2
    assert result.usage[1].outcome == "capacity_retry_cannot_narrow"
    assert any("capacity_retry_attempted" in item for item in result.diagnostics)
    assert any('"segment_count": 13' in item for item in result.diagnostics)


def test_oversized_verifier_range_gets_one_semantic_compact_retry() -> None:
    claim = _claim("claim-capacity-retry")
    url = "https://capacity-retry.example/article"
    source = " ".join(f"Sentence {index}." for index in range(1, 14))
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(claim.claim_id, "supports", ("S000001", "S000013"))
            ]
        },
        {
            "results": [
                _capacity_result(
                    claim.claim_id,
                    "replacement",
                    "supports",
                    ("S000006", "S000006"),
                )
            ]
        },
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    claim,
                    _candidate(claim=claim, url=url, note_id="note-capacity-retry"),
                )
            ],
            source_cache={url: source},
            model_client=model,
        )
    )

    relation = result.claims[0].relations[0]
    assert relation.status is VerificationRecordStatus.COMPLETED
    assert relation.start_segment_id == "S000006"
    assert relation.end_segment_id == "S000006"
    assert relation.source_quote == "Sentence 6."
    assert len(result.usage) == 2
    assert result.usage[1].retry is True
    assert result.usage[1].outcome == "capacity_retry_replacement"
    assert '"disposition":"replacement|cannot_narrow"' in model.prompts[1]
    assert any("capacity_retry_replacement" in item for item in result.diagnostics)


def test_capacity_retry_may_revise_semantic_verdict() -> None:
    claim = _claim("claim-capacity-revise")
    url = "https://capacity-revise.example/article"
    source = " ".join(f"Sentence {index}." for index in range(1, 14))
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(claim.claim_id, "supports", ("S000001", "S000013"))
            ]
        },
        {
            "results": [
                _capacity_result(
                    claim.claim_id,
                    "replacement",
                    "does_not_support",
                )
            ]
        },
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    claim,
                    _candidate(claim=claim, url=url, note_id="note-revise"),
                )
            ],
            source_cache={url: source},
            model_client=model,
        )
    )

    relation = result.claims[0].relations[0]
    assert relation.status is VerificationRecordStatus.COMPLETED
    assert relation.semantic_verdict is VerificationVerdict.DOES_NOT_SUPPORT
    assert relation.start_segment_id is None
    assert result.claims[0].state is ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT
    assert result.usage[1].outcome == "capacity_retry_replacement"


def test_capacity_retry_may_return_compact_located_contradiction() -> None:
    claim = _claim("claim-capacity-contradiction")
    url = "https://capacity-contradiction.example/article"
    source = " ".join(f"Sentence {index}." for index in range(1, 14))
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(claim.claim_id, "supports", ("S000001", "S000013"))
            ]
        },
        {
            "results": [
                _capacity_result(
                    claim.claim_id,
                    "replacement",
                    "contradicts",
                    ("S000006", "S000006"),
                )
            ]
        },
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    claim,
                    _candidate(
                        claim=claim,
                        url=url,
                        note_id="note-contradiction",
                    ),
                )
            ],
            source_cache={url: source},
            model_client=model,
        )
    )

    relation = result.claims[0].relations[0]
    assert relation.status is VerificationRecordStatus.COMPLETED
    assert relation.semantic_verdict is VerificationVerdict.CONTRADICTS
    assert relation.source_quote == "Sentence 6."
    assert result.claims[0].state is ClaimEvidenceState.REFUTED


def test_capacity_retry_second_oversized_range_is_not_retried_again() -> None:
    claim = _claim("claim-capacity-exhausted")
    url = "https://capacity-exhausted.example/article"
    source = " ".join(f"Sentence {index}." for index in range(1, 14))
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(claim.claim_id, "supports", ("S000001", "S000013"))
            ]
        },
        {
            "results": [
                _capacity_result(
                    claim.claim_id,
                    "replacement",
                    "supports",
                    ("S000001", "S000013"),
                )
            ]
        },
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    claim,
                    _candidate(claim=claim, url=url, note_id="note-exhausted"),
                )
            ],
            source_cache={url: source},
            model_client=model,
        )
    )

    relation = result.claims[0].relations[0]
    assert relation.status is VerificationRecordStatus.QUOTE_UNLOCATABLE
    assert len(model.prompts) == 2
    assert len(result.usage) == 2
    assert result.usage[1].outcome == "capacity_retry_exhausted"
    assert any("capacity_retry_exhausted" in item for item in result.diagnostics)


def test_capacity_retry_provider_error_preserves_original_relation() -> None:
    claim = _claim("claim-capacity-provider-error")
    url = "https://capacity-provider-error.example/article"
    source = " ".join(f"Sentence {index}." for index in range(1, 14))
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(claim.claim_id, "supports", ("S000001", "S000013"))
            ]
        },
        RuntimeError("retry provider unavailable"),
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    claim,
                    _candidate(claim=claim, url=url, note_id="note-provider"),
                )
            ],
            source_cache={url: source},
            model_client=model,
        )
    )

    relation = result.claims[0].relations[0]
    assert relation.status is VerificationRecordStatus.QUOTE_UNLOCATABLE
    assert relation.semantic_verdict is VerificationVerdict.SUPPORTS
    assert len(model.prompts) == 2
    assert result.usage[1].outcome == "capacity_retry_model_error"
    assert any("retry provider unavailable" in item for item in result.diagnostics)


def test_capacity_retry_budget_denial_preserves_original_relation() -> None:
    claim = _claim("claim-capacity-budget")
    url = "https://capacity-budget.example/article"
    source = " ".join(f"Sentence {index}." for index in range(1, 14))
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(claim.claim_id, "supports", ("S000001", "S000013"))
            ]
        }
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    claim,
                    _candidate(claim=claim, url=url, note_id="note-budget"),
                )
            ],
            source_cache={url: source},
            model_client=model,
            budget=VerificationBudget(max_tokens=11),
            estimate_input_tokens=lambda prompt: 1,
        )
    )

    relation = result.claims[0].relations[0]
    assert relation.status is VerificationRecordStatus.QUOTE_UNLOCATABLE
    assert relation.semantic_verdict is VerificationVerdict.SUPPORTS
    assert len(model.prompts) == 1
    assert len(result.usage) == 1
    assert any("capacity_retry_not_run" in item for item in result.diagnostics)


def test_capacity_retry_only_retries_oversized_claim_from_batch() -> None:
    oversized = _claim("claim-batch-oversized")
    compact = _claim("claim-batch-compact")
    url = "https://capacity-batch.example/article"
    source = " ".join(f"Sentence {index}." for index in range(1, 14))
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(
                    oversized.claim_id,
                    "supports",
                    ("S000001", "S000013"),
                ),
                _result(
                    compact.claim_id,
                    "supports",
                    ("S000002", "S000002"),
                ),
            ]
        },
        {
            "results": [
                _capacity_result(
                    oversized.claim_id,
                    "replacement",
                    "supports",
                    ("S000006", "S000006"),
                )
            ]
        },
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    oversized,
                    _candidate(claim=oversized, url=url, note_id="note-over"),
                ),
                _attribution(
                    compact,
                    _candidate(claim=compact, url=url, note_id="note-compact"),
                ),
            ],
            source_cache={url: source},
            model_client=model,
        )
    )

    by_id = {entry.claim.claim_id: entry for entry in result.claims}
    assert by_id[oversized.claim_id].relations[0].source_quote == "Sentence 6."
    assert by_id[compact.claim_id].relations[0].source_quote == "Sentence 2."
    assert len(model.prompts) == 2
    assert oversized.claim_id in model.prompts[1]
    assert compact.claim_id not in model.prompts[1]


def test_capacity_retry_handles_single_segment_character_limit() -> None:
    claim = _claim("claim-capacity-characters")
    url = "https://capacity-characters.example/article"
    source = "x" * 2_001
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(claim.claim_id, "supports", ("S000001", "S000001"))
            ]
        },
        {"results": [_capacity_result(claim.claim_id, "cannot_narrow")]},
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    claim,
                    _candidate(claim=claim, url=url, note_id="note-characters"),
                )
            ],
            source_cache={url: source},
            model_client=model,
        )
    )

    relation = result.claims[0].relations[0]
    assert relation.status is VerificationRecordStatus.QUOTE_UNLOCATABLE
    assert relation.error is not None
    assert "span_too_many_chars" in relation.error
    assert any('"char_count": 2001' in item for item in result.diagnostics)


def test_malformed_retry_then_oversized_range_gets_capacity_retry() -> None:
    claim = _claim("claim-malformed-then-capacity")
    url = "https://malformed-capacity.example/article"
    source = " ".join(f"Sentence {index}." for index in range(1, 14))
    model = ScriptedVerificationModel(
        {
            "results": [
                {
                    "claim_id": claim.claim_id,
                    "verdict": "supports",
                    "start_segment_id": None,
                    "end_segment_id": None,
                    "explanation": "Malformed evidentiary result.",
                }
            ]
        },
        {
            "results": [
                _result(claim.claim_id, "supports", ("S000001", "S000013"))
            ]
        },
        {"results": [_capacity_result(claim.claim_id, "cannot_narrow")]},
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    claim,
                    _candidate(claim=claim, url=url, note_id="note-malformed"),
                )
            ],
            source_cache={url: source},
            model_client=model,
        )
    )

    relation = result.claims[0].relations[0]
    assert relation.status is VerificationRecordStatus.QUOTE_UNLOCATABLE
    assert len(model.prompts) == 3
    assert [entry.retry for entry in result.usage] == [False, True, True]
    assert result.usage[2].outcome == "capacity_retry_cannot_narrow"


def test_unlocatable_contradiction_does_not_refute_claim() -> None:
    claim = _claim("claim-unlocatable-contradiction")
    url = "https://unlocatable-contradiction.example/article"
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(
                    claim.claim_id,
                    "contradicts",
                    ("S999998", "S999999"),
                )
            ]
        }
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    claim,
                    _candidate(claim=claim, url=url, note_id="note-unlocatable"),
                )
            ],
            source_cache={url: "One cached sentence."},
            model_client=model,
        )
    )

    relation = result.claims[0].relations[0]
    assert relation.status is VerificationRecordStatus.QUOTE_UNLOCATABLE
    assert relation.semantic_verdict is VerificationVerdict.CONTRADICTS
    assert result.claims[0].state is ClaimEvidenceState.VERIFICATION_INCOMPLETE


def test_no_candidate_and_all_admission_or_model_failures_remain_distinct() -> None:
    no_candidate = _claim("claim-none")
    budget_claim = _claim("claim-budget")
    large_claim = _claim("claim-large")
    error_claim = _claim("claim-error")
    budget_url = "https://budget.example/a"
    large_url = "https://large.example/b"
    error_url = "https://error.example/c"

    empty_result = asyncio.run(
        verify_attributions(
            [_attribution(no_candidate)],
            source_cache={},
            model_client=ScriptedVerificationModel(),
        )
    )
    assert empty_result.claims[0].state == (
        ClaimEvidenceState.NO_CANDIDATE_SOURCE
    )
    assert empty_result.usage == ()

    budget_model = ScriptedVerificationModel()
    budget_result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    budget_claim,
                    _candidate(
                        claim=budget_claim,
                        url=budget_url,
                        note_id="note-budget",
                    ),
                )
            ],
            source_cache={budget_url: "Complete cached text."},
            model_client=budget_model,
            budget=VerificationBudget(max_tokens=0),
            estimate_input_tokens=len,
        )
    )
    budget_relation = budget_result.claims[0].relations[0]
    assert budget_relation.status == (
        VerificationRecordStatus.VERIFICATION_NOT_RUN_BUDGET
    )
    assert budget_model.prompts == []

    large_model = ScriptedVerificationModel()
    large_result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    large_claim,
                    _candidate(
                        claim=large_claim,
                        url=large_url,
                        note_id="note-large",
                    ),
                )
            ],
            source_cache={large_url: "SOURCE-IS-TOO-LARGE"},
            model_client=large_model,
            settings=VerificationSettings(max_source_chars=5),
        )
    )
    large_relation = large_result.claims[0].relations[0]
    assert large_relation.status == (
        VerificationRecordStatus.SOURCE_TOO_LARGE_FOR_ADMISSION
    )
    assert "not truncated" in (large_relation.error or "")
    assert large_model.prompts == []

    error_model = ScriptedVerificationModel(RuntimeError("provider failed"))
    error_result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    error_claim,
                    _candidate(
                        claim=error_claim,
                        url=error_url,
                        note_id="note-error",
                    ),
                )
            ],
            source_cache={error_url: "Complete cached source."},
            model_client=error_model,
        )
    )
    error_relation = error_result.claims[0].relations[0]
    assert error_relation.status == (
        VerificationRecordStatus.VERIFICATION_MODEL_ERROR
    )
    assert "provider failed" in (error_relation.error or "")


def test_attribution_error_is_not_laundered_into_no_candidate_source() -> None:
    claim = _claim("claim-attribution-error")
    attribution = ClaimAttribution(
        claim=claim.model_copy(
            update={"source_resolution": SourceResolution.UNRESOLVED}
        ),
        status=AttributionStatus.ATTRIBUTION_ERROR,
        errors=(
            AttributionError(
                claim_id=claim.claim_id,
                code="malformed_candidates",
                detail="candidate payload could not be parsed",
            ),
        ),
    )

    result = asyncio.run(
        verify_attributions(
            [attribution],
            source_cache={},
            model_client=ScriptedVerificationModel(),
        )
    )

    assert result.claims[0].state == ClaimEvidenceState.ATTRIBUTION_ERROR
    assert result.claims[0].state != (
        ClaimEvidenceState.NO_CANDIDATE_SOURCE
    )


def test_domain_count_is_disclosed_but_cannot_establish_independence() -> None:
    claim = _claim("claim-publishers")
    first_url = "https://bbc.com/a"
    second_url = "https://bbc.co.uk/b"
    attributions = [
        _attribution(
            claim,
            _candidate(claim=claim, url=first_url, note_id="note-a"),
            _candidate(claim=claim, url=second_url, note_id="note-b"),
        )
    ]
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(
                    claim.claim_id,
                    "supports",
                    ("S000001", "S000001"),
                )
            ]
        },
        {
            "results": [
                _result(
                    claim.claim_id,
                    "supports",
                    ("S000001", "S000001"),
                )
            ]
        },
    )

    result = asyncio.run(
        verify_attributions(
            attributions,
            source_cache={
                first_url: "Exact claim support.",
                second_url: "Exact claim support.",
            },
            model_client=model,
            required_independent_sources={claim.claim_id: 2},
        )
    )

    assert result.claims[0].publisher_domain_proxy_count == 2
    assert result.claims[0].state == (
        ClaimEvidenceState.SUPPORTED_MULTIPLE_DOMAIN_PROXIES
    )
    assert result.claims[0].independent_lineage_count == 0
    assert result.independence.method == "confirmed_source_lineage_v1"
    assert result.independence.is_strict_independence_determination is False
    assert "unresolved_or_model_proposed_lineage_never_establishes_independence" in (
        result.independence.limitations
    )
    assert result.independence.unresolved_relation_count == 2


def test_two_confirmed_source_lineages_can_establish_corroboration() -> None:
    claim = _claim("claim-confirmed-lineage")
    first_url = "https://host-one.example/a"
    second_url = "https://host-two.example/b"
    first_text = "Newsroom One independently reports the event."
    second_text = "Newsroom Two independently reports the event."
    first_source_id = _candidate(
        claim=claim,
        url=first_url,
        note_id="note-one",
    ).source_id
    second_source_id = _candidate(
        claim=claim,
        url=second_url,
        note_id="note-two",
    ).source_id
    model = ScriptedVerificationModel(
        {"results": [_result(claim.claim_id, "supports", ("S000001", "S000001"))]},
        {"results": [_result(claim.claim_id, "supports", ("S000001", "S000001"))]},
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    claim,
                    _candidate(claim=claim, url=first_url, note_id="note-one"),
                    _candidate(claim=claim, url=second_url, note_id="note-two"),
                )
            ],
            source_cache={first_url: first_text, second_url: second_text},
            model_client=model,
            source_lineage_assessments={
                first_url: _lineage(
                    source_id=first_source_id,
                    url=first_url,
                    source_text=first_text,
                    lineage_id="newsroom-one",
                ),
                second_url: _lineage(
                    source_id=second_source_id,
                    url=second_url,
                    source_text=second_text,
                    lineage_id="newsroom-two",
                ),
            },
        )
    )

    verified = result.claims[0]
    assert verified.state is ClaimEvidenceState.CORROBORATED
    assert verified.independent_lineage_ids == ("newsroom-one", "newsroom-two")
    assert verified.lineage_assessment_complete is True
    assert result.independence.confirmed_assessment_count == 2


def test_same_host_can_corroborate_when_confirmed_lineages_are_distinct() -> None:
    claim = _claim("claim-shared-host-distinct-lineages")
    first_url = "https://archive.example/newsroom-one"
    second_url = "https://archive.example/newsroom-two"
    first_text = "Newsroom One identifies itself as the originating publisher."
    second_text = "Newsroom Two identifies itself as the originating publisher."
    first = _candidate(claim=claim, url=first_url, note_id="note-one")
    second = _candidate(claim=claim, url=second_url, note_id="note-two")
    model = ScriptedVerificationModel(
        {"results": [_result(claim.claim_id, "supports", ("S000001", "S000001"))]},
        {"results": [_result(claim.claim_id, "supports", ("S000001", "S000001"))]},
    )

    result = asyncio.run(
        verify_attributions(
            [_attribution(claim, first, second)],
            source_cache={first_url: first_text, second_url: second_text},
            model_client=model,
            source_lineage_assessments={
                first_url: _lineage(
                    source_id=first.source_id,
                    url=first_url,
                    source_text=first_text,
                    lineage_id="newsroom-one",
                ),
                second_url: _lineage(
                    source_id=second.source_id,
                    url=second_url,
                    source_text=second_text,
                    lineage_id="newsroom-two",
                ),
            },
        )
    )

    verified = result.claims[0]
    assert verified.publisher_domain_proxy_count == 1
    assert verified.independent_lineage_count == 2
    assert verified.state is ClaimEvidenceState.CORROBORATED


def test_distinct_hosts_do_not_corroborate_when_lineage_is_shared() -> None:
    claim = _claim("claim-distinct-hosts-shared-lineage")
    first_url = "https://host-one.example/syndicated"
    second_url = "https://host-two.example/republished"
    first_text = "This page republishes the wire report."
    second_text = "This page also republishes the wire report."
    first = _candidate(claim=claim, url=first_url, note_id="note-one")
    second = _candidate(claim=claim, url=second_url, note_id="note-two")
    model = ScriptedVerificationModel(
        {"results": [_result(claim.claim_id, "supports", ("S000001", "S000001"))]},
        {"results": [_result(claim.claim_id, "supports", ("S000001", "S000001"))]},
    )

    result = asyncio.run(
        verify_attributions(
            [_attribution(claim, first, second)],
            source_cache={first_url: first_text, second_url: second_text},
            model_client=model,
            source_lineage_assessments={
                first_url: _lineage(
                    source_id=first.source_id,
                    url=first_url,
                    source_text=first_text,
                    lineage_id="shared-wire-service",
                ),
                second_url: _lineage(
                    source_id=second.source_id,
                    url=second_url,
                    source_text=second_text,
                    lineage_id="shared-wire-service",
                ),
            },
        )
    )

    verified = result.claims[0]
    assert verified.publisher_domain_proxy_count == 2
    assert verified.independent_lineage_count == 1
    assert verified.state is ClaimEvidenceState.SUPPORTED_MULTIPLE_DOMAIN_PROXIES


def test_model_proposed_lineage_is_retained_but_cannot_corroborate() -> None:
    claim = _claim("claim-proposed-lineage")
    url = "https://host.example/a"
    source_text = "A page identifies the originating newsroom."
    candidate = _candidate(claim=claim, url=url, note_id="note-proposed")
    model = ScriptedVerificationModel(
        {"results": [_result(claim.claim_id, "supports", ("S000001", "S000001"))]}
    )

    result = asyncio.run(
        verify_attributions(
            [_attribution(claim, candidate)],
            source_cache={url: source_text},
            model_client=model,
            source_lineage_assessments={
                url: _lineage(
                    source_id=candidate.source_id,
                    url=url,
                    source_text=source_text,
                    lineage_id="newsroom-proposed",
                    status=SourceLineageStatus.PROPOSED,
                )
            },
        )
    )

    relation = result.claims[0].relations[0]
    assert relation.source_lineage is not None
    assert relation.source_lineage.status is SourceLineageStatus.PROPOSED
    assert result.claims[0].independent_lineage_count == 0
    assert result.claims[0].state is ClaimEvidenceState.SUPPORTED_SINGLE_DOMAIN_PROXY
    assert result.independence.proposed_assessment_count == 1


def test_ungrounded_lineage_degrades_to_audited_unresolved_not_failure() -> None:
    claim = _claim("claim-bad-lineage-basis")
    url = "https://host.example/a"
    source_text = "The actual cached page text."
    candidate = _candidate(claim=claim, url=url, note_id="note-bad-basis")
    bad_assessment = _lineage(
        source_id=candidate.source_id,
        url=url,
        source_text="Different bytes used by the assessor.",
        lineage_id="newsroom-untrusted",
    )
    model = ScriptedVerificationModel(
        {"results": [_result(claim.claim_id, "supports", ("S000001", "S000001"))]}
    )

    result = asyncio.run(
        verify_attributions(
            [_attribution(claim, candidate)],
            source_cache={url: source_text},
            model_client=model,
            source_lineage_assessments={url: bad_assessment},
        )
    )

    relation = result.claims[0].relations[0]
    assert relation.is_formal_supporting_evidence is True
    assert relation.source_lineage is None
    assert relation.source_lineage_error == (
        "lineage source text hash does not match cache"
    )
    assert result.claims[0].state is ClaimEvidenceState.SUPPORTED_SINGLE_DOMAIN_PROXY
    assert result.independence.unresolved_relation_count == 1


def test_historical_domain_proxy_corroboration_is_explicitly_reclassified() -> None:
    claim = _claim("claim-historical")
    first = VerifiedSourceRelation(
        claim_id=claim.claim_id,
        source_id="source-one",
        url="https://one.example/a",
        publisher_domain_proxy="one.example",
        candidate_note_ids=("note-one",),
        candidate_source_ids=("source-one",),
        status=VerificationRecordStatus.COMPLETED,
        semantic_verdict=VerificationVerdict.SUPPORTS,
        source_quote="First support.",
        span=QuoteSpan(start_char=0, end_char=14),
        location_status=NoteLocationStatus.LOCATABLE,
        is_formal_supporting_evidence=True,
    )
    second = first.model_copy(
        update={
            "source_id": "source-two",
            "url": "https://two.example/b",
            "publisher_domain_proxy": "two.example",
            "candidate_note_ids": ("note-two",),
            "candidate_source_ids": ("source-two",),
        }
    )
    payload = {
        "claims": [
            {
                "claim": claim.model_dump(mode="json"),
                "state": "corroborated",
                "required_independent_sources": 2,
                "relations": [
                    first.model_dump(mode="json"),
                    second.model_dump(mode="json"),
                ],
                "formal_supporting_evidence_count": 2,
                "publisher_domain_proxy_count": 2,
                "publisher_domain_proxies": ["one.example", "two.example"],
            }
        ],
        "independence": {
            "method": "publisher_domain_proxy",
            "is_strict_independence_determination": False,
            "limitations": ["common_ownership_not_resolved"],
        },
    }

    migrated = VerificationResult.model_validate(payload)

    verified = migrated.claims[0]
    assert verified.state is ClaimEvidenceState.SUPPORTED_MULTIPLE_DOMAIN_PROXIES
    assert verified.historical_domain_proxy_corroboration_reclassified is True
    assert verified.independent_lineage_count == 0
    assert migrated.independence.method == "confirmed_source_lineage_v1"


def test_element_registry_verifies_all_elements_in_one_claim_source_call() -> None:
    claim = _claim("claim-elements", "Alpha acquired Beta for $2 billion.")
    registry = _truth_registry(
        (claim, ("Alpha acquired Beta.", "The price was $2 billion."))
    )
    element_ids = tuple(item.element_id for item in registry.entries[0].elements)
    url = "https://elements.example/report"
    model = ScriptedVerificationModel(
        {
            "results": [
                _element_claim_result(
                    claim.claim_id,
                    *(
                        _element_result(element_id, "supports", ("S000001", "S000001"))
                        for element_id in element_ids
                    ),
                )
            ]
        }
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    claim,
                    _candidate(claim=claim, url=url, note_id="note-elements"),
                )
            ],
            source_cache={url: "Alpha acquired Beta for $2 billion."},
            model_client=model,
            registry=registry,
        )
    )

    assert len(model.prompts) == 1
    assert all(element_id in model.prompts[0] for element_id in element_ids)
    relation = result.claims[0].relations[0]
    assert [item.element_id for item in relation.element_relations] == list(element_ids)
    assert all(item.is_formal_supporting_evidence for item in relation.element_relations)
    assert relation.is_formal_supporting_evidence is True
    aggregate = result.claims[0].truth_condition_aggregate
    assert aggregate is not None
    assert aggregate.coverage_state is ClaimCoverageState.FULLY_SUPPORTED
    assert aggregate.execution_completeness is ExecutionCompleteness.COMPLETE
    assert result.truth_condition_registry_sha256 is not None

    tampered = result.claims[0].model_dump(mode="json")
    tampered["state"] = ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT.value
    with pytest.raises(ValidationError, match="evidence state"):
        ClaimVerification.model_validate(tampered)


def test_element_registry_partial_support_is_not_full_claim_support() -> None:
    claim = _claim("claim-partial", "Alpha acquired Beta for $2 billion.")
    registry = _truth_registry(
        (claim, ("Alpha acquired Beta.", "The price was $2 billion."))
    )
    first, second = (item.element_id for item in registry.entries[0].elements)
    url = "https://partial.example/report"
    model = ScriptedVerificationModel(
        {
            "results": [
                _element_claim_result(
                    claim.claim_id,
                    _element_result(first, "supports", ("S000001", "S000001")),
                    _element_result(second, "does_not_support"),
                )
            ]
        }
    )

    result = asyncio.run(
        verify_attributions(
            [_attribution(claim, _candidate(claim=claim, url=url, note_id="n"))],
            source_cache={url: "Alpha acquired Beta."},
            model_client=model,
            registry=registry,
        )
    )

    assert result.claims[0].relations[0].is_formal_supporting_evidence is False
    aggregate = result.claims[0].truth_condition_aggregate
    assert aggregate is not None
    assert aggregate.coverage_state is ClaimCoverageState.PARTIALLY_SUPPORTED
    assert aggregate.execution_completeness is ExecutionCompleteness.COMPLETE


def test_element_support_can_close_claim_across_different_sources() -> None:
    claim = _claim("claim-split-elements", "Alpha acquired Beta for $2 billion.")
    registry = _truth_registry(
        (claim, ("Alpha acquired Beta.", "The price was $2 billion."))
    )
    first, second = (item.element_id for item in registry.entries[0].elements)
    first_url = "https://first-elements.example/report"
    second_url = "https://second-elements.example/report"
    model = ScriptedVerificationModel(
        {
            "results": [
                _element_claim_result(
                    claim.claim_id,
                    _element_result(first, "supports", ("S000001", "S000001")),
                    _element_result(second, "does_not_support"),
                )
            ]
        },
        {
            "results": [
                _element_claim_result(
                    claim.claim_id,
                    _element_result(first, "does_not_support"),
                    _element_result(second, "supports", ("S000001", "S000001")),
                )
            ]
        },
    )

    result = asyncio.run(
        verify_attributions(
            [
                _attribution(
                    claim,
                    _candidate(claim=claim, url=first_url, note_id="first"),
                    _candidate(claim=claim, url=second_url, note_id="second"),
                )
            ],
            source_cache={
                first_url: "Alpha acquired Beta.",
                second_url: "The price was $2 billion.",
            },
            model_client=model,
            registry=registry,
        )
    )

    verification = result.claims[0]
    assert not any(
        relation.is_formal_supporting_evidence for relation in verification.relations
    )
    assert verification.truth_condition_aggregate is not None
    assert (
        verification.truth_condition_aggregate.coverage_state
        is ClaimCoverageState.FULLY_SUPPORTED
    )
    assert verification.state is (
        ClaimEvidenceState.SUPPORTED_DISTRIBUTED_ELEMENT_EVIDENCE
    )
    # Neither source supports the whole claim. The nested evidence union is
    # retained separately instead of inflating whole-claim source counts.
    assert verification.formal_supporting_evidence_count == 0
    assert verification.publisher_domain_proxy_count == 0
    assert verification.publisher_domain_proxies == ()
    assert verification.element_supporting_domain_proxy_count == 2
    assert verification.element_supporting_domain_proxies == (
        "first-elements.example",
        "second-elements.example",
    )

    # The immediately preceding audit schema projected every source supporting
    # any element into the whole-claim publisher fields and had no projection
    # version or element-support fields.  Preserve that exact historical shape
    # as a real split-evidence compatibility fixture.
    historical_payload = verification.model_dump(mode="json")
    historical_payload.pop("element_support_projection_version")
    historical_payload.pop("element_supporting_domain_proxy_count")
    historical_payload.pop("element_supporting_domain_proxies")
    historical_payload["state"] = (
        ClaimEvidenceState.SUPPORTED_MULTIPLE_DOMAIN_PROXIES.value
    )
    historical_payload["publisher_domain_proxy_count"] = 2
    historical_payload["publisher_domain_proxies"] = [
        "first-elements.example",
        "second-elements.example",
    ]

    restored = ClaimVerification.model_validate(historical_payload)

    assert restored.state is (
        ClaimEvidenceState.SUPPORTED_DISTRIBUTED_ELEMENT_EVIDENCE
    )
    assert restored.publisher_domain_proxy_count == 0
    assert restored.element_supporting_domain_proxy_count == 2
    assert restored.historical_element_support_projection_reclassified is True
    assert restored.element_support_projection_version == (
        "whole-claim-element-support-v2"
    )

    # A versioned current payload never enters migration.  Counter tampering is
    # rejected instead of being silently recomputed from nested relations.
    tampered_current = verification.model_dump(mode="json")
    tampered_current["element_supporting_domain_proxy_count"] = 0
    tampered_current["element_supporting_domain_proxies"] = []
    with pytest.raises(ValidationError, match="element support|counters"):
        ClaimVerification.model_validate(tampered_current)


def test_element_denominator_error_recovers_only_the_bad_claim() -> None:
    first_claim = _claim("claim-element-good", "Alpha acquired Beta.")
    second_claim = _claim("claim-element-bad", "The price was $2 billion.")
    registry = _truth_registry(
        (first_claim, ("Alpha acquired Beta.",)),
        (second_claim, ("The price was $2 billion.", "It was paid in 2024.")),
    )
    first_id = registry.entries[0].elements[0].element_id
    second_ids = tuple(item.element_id for item in registry.entries[1].elements)
    url = "https://element-recovery.example/report"
    source = "Alpha acquired Beta. The price was $2 billion. It was paid in 2024."
    model = ScriptedVerificationModel(
        {
            "results": [
                _element_claim_result(
                    first_claim.claim_id,
                    _element_result(first_id, "supports", ("S000001", "S000001")),
                ),
                _element_claim_result(
                    second_claim.claim_id,
                    _element_result(second_ids[0], "supports", ("S000002", "S000002")),
                    _element_result(second_ids[0], "supports", ("S000002", "S000002")),
                ),
            ]
        },
        {
            "results": [
                _element_claim_result(
                    second_claim.claim_id,
                    _element_result(second_ids[0], "supports", ("S000002", "S000002")),
                    _element_result(second_ids[1], "supports", ("S000003", "S000003")),
                )
            ]
        },
    )
    attributions = [
        _attribution(
            claim,
            _candidate(claim=claim, url=url, note_id=f"note-{claim.claim_id}"),
        )
        for claim in (first_claim, second_claim)
    ]

    result = asyncio.run(
        verify_attributions(
            attributions,
            source_cache={url: source},
            model_client=model,
            registry=registry,
        )
    )

    assert len(model.prompts) == 2
    assert '"claim_id": "claim-element-good"' not in model.prompts[1]
    assert result.usage[0].outcome == "element_partial_malformed"
    assert result.usage[1].retry is True
    assert len(result.claims[0].relations) == 1
    assert len(result.claims[1].relations) == 1
    assert all(
        item.truth_condition_aggregate is not None
        and item.truth_condition_aggregate.coverage_state
        is ClaimCoverageState.FULLY_SUPPORTED
        for item in result.claims
    )


def test_element_numeric_value_missing_from_quote_is_not_formal_support() -> None:
    claim = _claim("claim-element-number", "The transfer was $10 billion.")
    registry = _truth_registry((claim, ("The transfer was $10 billion.",)))
    element_id = registry.entries[0].elements[0].element_id
    url = "https://numeric-element.example/report"
    model = ScriptedVerificationModel(
        {
            "results": [
                _element_claim_result(
                    claim.claim_id,
                    _element_result(element_id, "supports", ("S000001", "S000001")),
                )
            ]
        }
    )

    result = asyncio.run(
        verify_attributions(
            [_attribution(claim, _candidate(claim=claim, url=url, note_id="n"))],
            source_cache={url: "The source describes the transfer without a value."},
            model_client=model,
            registry=registry,
        )
    )

    element = result.claims[0].relations[0].element_relations[0]
    assert (
        element.numeric_consistency_status
        is NumericConsistencyStatus.SOURCE_VALUES_NOT_RECOGNIZED
    )
    assert element.is_formal_supporting_evidence is False
    aggregate = result.claims[0].truth_condition_aggregate
    assert aggregate is not None
    assert aggregate.coverage_state is ClaimCoverageState.UNRESOLVED


def test_element_capacity_retry_is_once_per_claim_source_not_per_element() -> None:
    claim = _claim("claim-element-capacity", "Two related conditions hold.")
    registry = _truth_registry((claim, ("Condition one holds.", "Condition two holds.")))
    element_ids = tuple(item.element_id for item in registry.entries[0].elements)
    url = "https://element-capacity.example/report"
    source = " ".join(f"Sentence {index}." for index in range(1, 14))
    model = ScriptedVerificationModel(
        {
            "results": [
                _element_claim_result(
                    claim.claim_id,
                    *(
                        _element_result(
                            element_id,
                            "supports",
                            ("S000001", "S000013"),
                        )
                        for element_id in element_ids
                    ),
                )
            ]
        },
        {
            "results": [
                {
                    "claim_id": claim.claim_id,
                    "disposition": "cannot_narrow",
                    "elements": [],
                    "explanation": "No sufficient compact ranges.",
                }
            ]
        },
    )

    result = asyncio.run(
        verify_attributions(
            [_attribution(claim, _candidate(claim=claim, url=url, note_id="n"))],
            source_cache={url: source},
            model_client=model,
            registry=registry,
        )
    )

    assert len(model.prompts) == 2
    assert result.usage[1].outcome == "element_capacity_retry_cannot_narrow"
    relation = result.claims[0].relations[0]
    assert len(relation.element_relations) == 2
    assert all(
        item.status is ElementAssessmentExecutionStatus.QUOTE_UNLOCATABLE
        for item in relation.element_relations
    )
    aggregate = result.claims[0].truth_condition_aggregate
    assert aggregate is not None
    assert aggregate.coverage_state is ClaimCoverageState.UNRESOLVED
    assert aggregate.execution_completeness is ExecutionCompleteness.FAILED


def test_element_malformed_recovery_still_enters_capacity_retry_path() -> None:
    claim = _claim("claim-element-malformed-capacity", "One condition holds.")
    registry = _truth_registry((claim, ("One condition holds.",)))
    element_id = registry.entries[0].elements[0].element_id
    url = "https://element-malformed-capacity.example/report"
    source = " ".join(f"Sentence {index}." for index in range(1, 14))
    model = ScriptedVerificationModel(
        {
            "results": [
                _element_claim_result(
                    claim.claim_id,
                    _element_result(element_id, "supports"),
                )
            ]
        },
        {
            "results": [
                _element_claim_result(
                    claim.claim_id,
                    _element_result(
                        element_id,
                        "supports",
                        ("S000001", "S000013"),
                    ),
                )
            ]
        },
        {
            "results": [
                {
                    "claim_id": claim.claim_id,
                    "disposition": "cannot_narrow",
                    "elements": [],
                    "explanation": "No sufficient compact range.",
                }
            ]
        },
    )

    result = asyncio.run(
        verify_attributions(
            [_attribution(claim, _candidate(claim=claim, url=url, note_id="n"))],
            source_cache={url: source},
            model_client=model,
            registry=registry,
        )
    )

    assert len(model.prompts) == 3
    assert [record.retry for record in result.usage] == [False, True, True]
    assert result.usage[-1].outcome == "element_capacity_retry_cannot_narrow"


def test_incomplete_elementization_cannot_become_fully_supported_in_verifier() -> None:
    claim = _claim("claim-element-incomplete", "Alpha acquired Beta for $2 billion.")
    registry = _truth_registry(
        (claim, ("Alpha acquired Beta.",)),
        semantic_status=ElementizationSemanticStatus.INCOMPLETE,
    )
    element_id = registry.entries[0].elements[0].element_id
    url = "https://element-incomplete.example/report"
    model = ScriptedVerificationModel(
        {
            "results": [
                _element_claim_result(
                    claim.claim_id,
                    _element_result(element_id, "supports", ("S000001", "S000001")),
                )
            ]
        }
    )

    result = asyncio.run(
        verify_attributions(
            [_attribution(claim, _candidate(claim=claim, url=url, note_id="n"))],
            source_cache={url: "Alpha acquired Beta."},
            model_client=model,
            registry=registry,
        )
    )

    aggregate = result.claims[0].truth_condition_aggregate
    assert aggregate is not None
    assert aggregate.coverage_state is ClaimCoverageState.PARTIALLY_SUPPORTED
    assert result.claims[0].relations[0].is_formal_supporting_evidence is False


def test_run_cost_cap_is_not_downgraded_to_verification_model_error() -> None:
    claim = _claim("claim-cap")
    url = "https://cap.example/report"
    controller = RunCostController(RunCostBudget(max_cost_usd=1))
    cap_error = RunCostCapReached(
        "synthetic verifier cap",
        stage="initial_verification",
        audit=controller.audit(),
    )
    model = ScriptedVerificationModel(cap_error)

    with pytest.raises(RunCostCapReached) as caught:
        asyncio.run(
            verify_attributions(
                [
                    _attribution(
                        claim,
                        _candidate(claim=claim, url=url, note_id="note-cap"),
                    )
                ],
                source_cache={url: claim.claim_text},
                model_client=model,
            )
        )

    assert caught.value is cap_error
