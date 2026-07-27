from __future__ import annotations

import asyncio
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
from open_deep_research.harness.claims import (
    AtomicClaim,
    CitationRequirement,
    ClaimNormalizationStatus,
    SourceResolution,
)
from open_deep_research.harness.notes import (
    NoteLocationStatus,
    create_note,
)
from open_deep_research.harness.verify import (
    ClaimEvidenceState,
    VerificationBudget,
    VerificationRecordStatus,
    VerificationSettings,
    VerificationVerdict,
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
    def __init__(self, quote: str) -> None:
        self.quote = quote
        self.prompts: list[str] = []
        self.claim_batches: list[tuple[str, ...]] = []

    async def generate(self, prompt: str) -> dict[str, object]:
        self.prompts.append(prompt)
        match = re.search(
            r"Claims:\n(.*?)\n\nBEGIN COMPLETE CACHED SOURCE",
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
                            "quote": self.quote,
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
    quote: str | None,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "verdict": verdict,
        "quote": quote,
        "explanation": "Auditable semantic judgement.",
    }


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
    model = EchoSupportModel("Exact supporting sentence.")

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
    assert all(source in prompt for prompt in model.prompts)
    assert all("TAIL-SENTINEL-THAT-MUST-NOT-BE-TRUNCATED" in prompt for prompt in model.prompts)
    assert len(result.claims) == 21
    assert all(
        claim.state == ClaimEvidenceState.CORROBORATED
        for claim in result.claims
    )
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
                        _result("claim-1", "supports", "First exact passage."),
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
                            "Second exact passage.",
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


def test_strict_then_repair_gate_and_unlocatable_note_history() -> None:
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
                _result(claim.claim_id, "supports", "alpha beta 2026.")
            ]
        },
        {
            "results": [
                _result(
                    claim.claim_id,
                    "supports",
                    "Original wording is materially different.",
                )
            ]
        },
        {
            "results": [
                _result(
                    claim.claim_id,
                    "supports",
                    "Exact source-authored passage.",
                )
            ]
        },
        {
            "results": [
                _result(
                    claim.claim_id,
                    "supports",
                    "A paraphrase with different words.",
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
    assert by_url[repair_url].location_status == (
        NoteLocationStatus.REPAIRED_LOCATABLE
    )
    assert by_url[repair_url].model_quote == "alpha beta 2026."
    assert by_url[repair_url].source_quote == "AlphaBeta 2026"
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
    assert verified.state == ClaimEvidenceState.CORROBORATED
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
                    "The source explicitly contradicts the assertion.",
                )
            ]
        },
        {
            "results": [
                _result(
                    claim.claim_id,
                    "supports",
                    "The source explicitly supports the assertion.",
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


def test_noncontiguous_composite_diagnostic_cannot_become_formal_evidence() -> None:
    claim = _claim("claim-composite")
    url = "https://composite.example/article"
    model = ScriptedVerificationModel(
        {
            "results": [
                _result(claim.claim_id, "supports", "Alpha...Beta")
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
    assert relation.quote_failure_reason.value == "noncontiguous_composite"
    assert relation.source_quote is None
    assert relation.span is None
    assert relation.is_formal_supporting_evidence is False


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


def test_publisher_count_is_disclosed_as_a_proxy_not_strict_independence() -> None:
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
                _result(claim.claim_id, "supports", "Exact claim support.")
            ]
        },
        {
            "results": [
                _result(claim.claim_id, "supports", "Exact claim support.")
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
    assert result.independence.method == "publisher_domain_proxy"
    assert result.independence.is_strict_independence_determination is False
    assert "common_ownership_not_resolved" in result.independence.limitations
    assert "cross_domain_brand_identity_not_resolved" in (
        result.independence.limitations
    )
