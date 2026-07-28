import asyncio
import json

import pytest
from pydantic import ValidationError

from open_deep_research.harness.claims import (
    AtomicClaim,
    CitationRequirement,
    ClaimNormalizationStatus,
)
from open_deep_research.harness.evaluative import (
    EvaluativeDiagnosticResult,
    EvaluativeDiagnosticStatus,
    EvaluativeUnderspecification,
    build_evaluative_diagnostic_prompt,
    diagnose_underspecified_evaluative_claims,
)


def _claim(
    claim_id: str,
    text: str,
    *,
    requirement: CitationRequirement = CitationRequirement.EXTERNAL,
    start: int = 0,
) -> AtomicClaim:
    return AtomicClaim(
        claim_id=claim_id,
        block_id=f"block-{claim_id}",
        selected_text=text,
        claim_text=text,
        anchor_text=text,
        start_char=start,
        end_char=start + len(text),
        citation_requirement=requirement,
        normalization_status=ClaimNormalizationStatus.LOCATED,
    )


class ScriptedModel:
    def __init__(self, *contents):
        self.contents = list(contents)
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return {
            "content": json.dumps(self.contents.pop(0)),
            "token_count": 7,
            "cost_usd": 0.007,
        }


def test_prompt_is_independent_non_gating_and_uses_coarse_categories() -> None:
    claim = _claim(
        "claim-0001",
        "It was one of the most consequential developments in recent years.",
    )

    prompt = build_evaluative_diagnostic_prompt((claim,))

    assert "independent, non-gating audit" in prompt
    assert "cannot delete a claim" in prompt
    assert "remove anything from a denominator" in prompt
    assert "A diagnosed claim remains an\nexternal claim" in prompt
    assert all(
        category.value in prompt for category in EvaluativeUnderspecification
    )


def test_diagnostics_are_multilabel_but_cannot_change_registry_or_denominator():
    claims = (
        _claim(
            "claim-0001",
            "It was one of the most consequential developments in recent years.",
        ),
        _claim(
            "claim-0002",
            "The measured value was 12 units.",
            start=80,
        ),
        _claim(
            "claim-0003",
            "This report has three sections.",
            requirement=CitationRequirement.INTERNAL,
            start=120,
        ),
    )
    before = [claim.model_dump(mode="json") for claim in claims]
    model = ScriptedModel(
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "status": "underspecified_evaluative_claim",
                    "categories": [
                        "comparison_scope_unspecified",
                        "temporal_scope_unspecified",
                    ],
                    "reason": "Comparison set and time interval are absent.",
                },
                {
                    "claim_id": "claim-0002",
                    "status": "not_underspecified",
                    "categories": [],
                    "reason": "The quantity has an explicit value.",
                },
            ]
        }
    )

    result = asyncio.run(
        diagnose_underspecified_evaluative_claims(
            claims,
            model_client=model,
        )
    )

    assert result.external_claim_ids == ("claim-0001", "claim-0002")
    assert result.external_denominator_before == 2
    assert result.external_denominator_after == 2
    assert result.underspecified_claim_count == 1
    assert result.diagnostic_failed_count == 0
    assert result.category_counts == {
        "comparison_scope_unspecified": 1,
        "evaluation_criterion_unspecified": 0,
        "temporal_scope_unspecified": 1,
    }
    assert result.claim_registry_unchanged
    assert result.citation_requirements_unchanged
    assert result.diagnostic_is_non_gating
    assert [claim.model_dump(mode="json") for claim in claims] == before
    assert "claims" not in result.model_dump(mode="json")
    assert "claim-0003" not in model.prompts[0]


def test_omission_becomes_diagnostic_failure_not_not_underspecified() -> None:
    claims = (
        _claim("claim-0001", "The effect was widespread."),
        _claim("claim-0002", "The measured value was 12 units.", start=40),
    )
    model = ScriptedModel(
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "status": "underspecified_evaluative_claim",
                    "categories": ["evaluation_criterion_unspecified"],
                    "reason": "No operational extent criterion is stated.",
                }
            ]
        }
    )

    result = asyncio.run(
        diagnose_underspecified_evaluative_claims(
            claims,
            model_client=model,
        )
    )

    assert tuple(
        assessment.status for assessment in result.assessments
    ) == (
        EvaluativeDiagnosticStatus.UNDERSPECIFIED,
        EvaluativeDiagnosticStatus.DIAGNOSTIC_FAILED,
    )
    assert result.diagnostic_failed_count == 1
    assert result.external_denominator_after == 2
    assert result.batches[0].outcome == "partial"
    assert result.batches[0].failed_claim_ids == ("claim-0002",)
    assert any(
        "diagnostic omitted this external claim" in diagnostic
        for diagnostic in result.diagnostics
    )


def test_result_schema_rejects_any_denominator_change() -> None:
    with pytest.raises(ValidationError, match="external denominator"):
        EvaluativeDiagnosticResult(
            registry_claim_count=1,
            external_denominator_before=1,
            external_denominator_after=0,
            external_claim_ids=(),
            underspecified_claim_count=0,
            diagnostic_failed_count=0,
            claim_registry_sha256_before="same",
            claim_registry_sha256_after="same",
            claim_registry_unchanged=True,
            citation_requirements_unchanged=True,
        )
