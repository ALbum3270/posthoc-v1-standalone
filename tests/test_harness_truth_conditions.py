from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from open_deep_research.harness.truth_conditions import (
    ClaimCoverageState,
    ClaimTruthConditionAggregate,
    ElementAssessmentExecutionStatus,
    ElementSemanticState,
    ElementSourceAssessment,
    ElementTruthConditionAggregate,
    ElementVerificationVerdict,
    ElementizationExecutionStatus,
    ElementizationFailure,
    ElementizationProposal,
    ElementizationReview,
    ElementizationSemanticStatus,
    ElementizationStage,
    ExecutionCompleteness,
    TruthConditionDenominatorAudit,
    TruthConditionProtocol,
    aggregate_truth_condition_claim,
    aggregate_truth_condition_registry,
    build_elementization_proposal_prompt,
    build_elementization_review_prompt,
    build_registry_from_parse_results,
    build_truth_condition_registry,
    make_truth_condition_element_id,
    parse_elementization_proposals,
    parse_elementization_reviews,
    select_truth_condition_registry,
    truth_condition_protocol,
    truth_condition_registry_sha256,
)


CLAIMS = {
    "claim-1": "Alpha acquired Beta for $2 billion in 2024.",
    "claim-2": "The regulator announced the settlement.",
}


def _proposal(claim_id: str, *elements: str) -> ElementizationProposal:
    return ElementizationProposal(
        claim_id=claim_id,
        elements=elements,
        rationale="proposal",
    )


def _review(
    claim_id: str,
    *elements: str,
    status: ElementizationSemanticStatus = ElementizationSemanticStatus.COMPLETE,
    missing: tuple[str, ...] = (),
) -> ElementizationReview:
    return ElementizationReview(
        claim_id=claim_id,
        semantic_status=status,
        elements=elements,
        missing_conditions=missing,
        rationale="independent review",
    )


def _complete_registry():
    return build_truth_condition_registry(
        CLAIMS,
        proposals=(
            _proposal("claim-1", "Alpha acquired Beta.", "The price was $2 billion."),
            _proposal("claim-2", "The regulator announced the settlement."),
        ),
        reviews=(
            _review(
                "claim-1",
                "Alpha acquired Beta.",
                "The price was $2 billion.",
                "The acquisition occurred in 2024.",
            ),
            _review("claim-2", "The regulator announced the settlement."),
        ),
    )


def _assessment(
    *,
    claim_id: str,
    element_id: str,
    source_id: str,
    verdict: ElementVerificationVerdict,
    located: bool = True,
    formal: bool = False,
) -> ElementSourceAssessment:
    return ElementSourceAssessment(
        claim_id=claim_id,
        element_id=element_id,
        source_id=source_id,
        execution_status=ElementAssessmentExecutionStatus.COMPLETE,
        verdict=verdict,
        evidence_located=located,
        formal_supporting_evidence=formal,
    )


def test_code_allocates_stable_element_ids_after_review() -> None:
    registry = _complete_registry()
    first = registry.entries[0]

    assert [item.element_id for item in first.elements] == [
        "claim-1::tc-0001",
        "claim-1::tc-0002",
        "claim-1::tc-0003",
    ]
    assert first.proposal_elements == (
        "Alpha acquired Beta.",
        "The price was $2 billion.",
    )
    assert first.elements[2].text == "The acquisition occurred in 2024."
    assert make_truth_condition_element_id("claim-1", 2) == "claim-1::tc-0003"
    assert len(truth_condition_registry_sha256(registry)) == 64


def test_subset_registry_closes_exact_incremental_denominator_in_requested_order() -> None:
    registry = _complete_registry()

    selected = select_truth_condition_registry(registry, ("claim-2",))

    assert tuple(entry.claim_id for entry in selected.entries) == ("claim-2",)
    assert selected.denominator.selected_claim_ids == ("claim-2",)
    assert selected.denominator.complete_claim_ids == ("claim-2",)
    assert selected.entries[0] == registry.entries[1]

    with pytest.raises(ValueError, match="unknown claims"):
        select_truth_condition_registry(registry, ("claim-missing",))


def test_review_failure_is_unresolved_and_retains_proposal_for_audit() -> None:
    registry = build_truth_condition_registry(
        {"claim-1": CLAIMS["claim-1"]},
        proposals=(_proposal("claim-1", "Alpha acquired Beta."),),
        failures=(
            ElementizationFailure(
                claim_id="claim-1",
                stage=ElementizationStage.REVIEW,
                execution_status=ElementizationExecutionStatus.MODEL_ERROR,
                diagnostic="provider timeout",
            ),
        ),
    )

    entry = registry.entries[0]
    assert entry.proposal_elements == ("Alpha acquired Beta.",)
    assert [item.text for item in entry.elements] == ["Alpha acquired Beta."]
    assert entry.semantic_status is None
    assert registry.denominator.unresolved_claim_ids == ("claim-1",)
    assert registry.denominator.silent_bypass_count == 0


def test_missing_outputs_close_denominator_as_unresolved() -> None:
    registry = build_truth_condition_registry(CLAIMS)

    assert registry.denominator.selected_claim_ids == ("claim-1", "claim-2")
    assert registry.denominator.unresolved_claim_ids == ("claim-1", "claim-2")
    assert registry.denominator.is_closed
    assert all(
        entry.execution_status is ElementizationExecutionStatus.INVALID_RESPONSE
        for entry in registry.entries
    )


def test_incomplete_and_uncertain_reviews_cannot_be_semantic_successes() -> None:
    registry = build_truth_condition_registry(
        CLAIMS,
        proposals=(
            _proposal("claim-1", "Alpha acquired Beta."),
            _proposal("claim-2", "The regulator made an announcement."),
        ),
        reviews=(
            _review(
                "claim-1",
                "Alpha acquired Beta.",
                status=ElementizationSemanticStatus.INCOMPLETE,
                missing=("The price or date may be missing.",),
            ),
            _review(
                "claim-2",
                status=ElementizationSemanticStatus.UNCERTAIN,
            ),
        ),
    )

    assert registry.denominator.incomplete_claim_ids == ("claim-1",)
    assert registry.denominator.unresolved_claim_ids == ("claim-2",)
    assert registry.denominator.complete_claim_ids == ()


def test_denominator_model_rejects_overlap_and_gaps() -> None:
    with pytest.raises(ValidationError):
        TruthConditionDenominatorAudit(
            selected_claim_ids=("a", "b"),
            complete_claim_ids=("a",),
            incomplete_claim_ids=("a",),
            unresolved_claim_ids=(),
            silent_bypass_claim_ids=(),
        )

    with pytest.raises(ValidationError):
        TruthConditionDenominatorAudit(
            selected_claim_ids=("a", "b"),
            complete_claim_ids=("a",),
            incomplete_claim_ids=(),
            unresolved_claim_ids=(),
            silent_bypass_claim_ids=(),
        )


def test_proposal_parser_recovers_sibling_and_closes_missing_claim() -> None:
    parsed = parse_elementization_proposals(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "claim-1",
                        "elements": ["Alpha acquired Beta."],
                        "rationale": "one condition",
                    }
                ]
            }
        ),
        ("claim-1", "claim-2"),
    )

    assert [item.claim_id for item in parsed.proposals] == ["claim-1"]
    assert [item.claim_id for item in parsed.failures] == ["claim-2"]
    assert parsed.failures[0].execution_status is ElementizationExecutionStatus.INVALID_RESPONSE


def test_duplicate_proposal_claim_invalidates_only_that_claim() -> None:
    parsed = parse_elementization_proposals(
        json.dumps(
            {
                "claims": [
                    {"claim_id": "claim-1", "elements": ["one"]},
                    {"claim_id": "claim-1", "elements": ["two"]},
                    {"claim_id": "claim-2", "elements": ["three"]},
                ]
            }
        ),
        ("claim-1", "claim-2"),
    )

    assert [item.claim_id for item in parsed.proposals] == ["claim-2"]
    assert [item.claim_id for item in parsed.failures] == ["claim-1"]


def test_invalid_review_json_marks_every_claim_unresolved() -> None:
    parsed = parse_elementization_reviews("not json", ("claim-1", "claim-2"))

    assert parsed.reviews == ()
    assert [item.claim_id for item in parsed.failures] == ["claim-1", "claim-2"]
    assert all(item.stage is ElementizationStage.REVIEW for item in parsed.failures)


def test_parse_results_build_registry_without_silent_bypass() -> None:
    proposals = parse_elementization_proposals(
        json.dumps(
            {
                "claims": [
                    {"claim_id": "claim-1", "elements": ["condition one"]},
                    {"claim_id": "claim-2", "elements": ["condition two"]},
                ]
            }
        ),
        tuple(CLAIMS),
    )
    reviews = parse_elementization_reviews(
        json.dumps(
            {
                "claims": [
                    {
                        "claim_id": "claim-1",
                        "semantic_status": "complete",
                        "elements": ["condition one"],
                        "missing_conditions": [],
                        "rationale": "complete",
                    }
                ]
            }
        ),
        tuple(CLAIMS),
    )

    registry = build_registry_from_parse_results(CLAIMS, proposals, reviews)

    assert registry.denominator.complete_claim_ids == ("claim-1",)
    assert registry.denominator.unresolved_claim_ids == ("claim-2",)
    assert registry.denominator.silent_bypass_count == 0


def test_legacy_none_is_explicit_and_does_not_create_empty_success() -> None:
    assert truth_condition_protocol(None) is TruthConditionProtocol.LEGACY_WHOLE_CLAIM
    assert truth_condition_protocol(_complete_registry()) is TruthConditionProtocol.ELEMENT_REGISTRY
    assert (
        aggregate_truth_condition_registry(
            None,
            (),
            expected_source_ids_by_claim={},
        )
        is None
    )


def test_prompts_assign_semantics_to_models_and_ids_to_code() -> None:
    proposals = tuple(
        _proposal(claim_id, f"condition for {claim_id}") for claim_id in CLAIMS
    )
    proposal_prompt = build_elementization_proposal_prompt(CLAIMS)
    review_prompt = build_elementization_review_prompt(
        CLAIMS,
        proposals,
        claim_contexts={"claim-1": ("Alpha and Beta are companies.",)},
        claim_glosses={"claim-1": "Alpha bought Beta in the stated deal."},
    )

    assert "do not invent element ids" in proposal_prompt.lower()
    assert "independent" in review_prompt.lower()
    assert "code will only validate the response shape and allocate IDs" in review_prompt
    assert '"report_surface_text"' in review_prompt
    assert '"necessary_context"' in review_prompt
    assert '"retrieval_gloss"' in review_prompt
    assert "model-derived aid only" in review_prompt
    assert "domain" not in proposal_prompt.lower()


def test_formal_support_requires_located_supporting_evidence() -> None:
    with pytest.raises(ValidationError):
        ElementSourceAssessment(
            claim_id="claim-1",
            element_id="claim-1::tc-0001",
            source_id="source-1",
            execution_status=ElementAssessmentExecutionStatus.COMPLETE,
            verdict=ElementVerificationVerdict.SUPPORTS,
            evidence_located=False,
            formal_supporting_evidence=True,
        )


def test_support_and_contradiction_are_conflicted_not_supported() -> None:
    entry = _complete_registry().entries[1]
    element = entry.elements[0]
    assessments = (
        _assessment(
            claim_id=entry.claim_id,
            element_id=element.element_id,
            source_id="source-1",
            verdict=ElementVerificationVerdict.SUPPORTS,
            formal=True,
        ),
        _assessment(
            claim_id=entry.claim_id,
            element_id=element.element_id,
            source_id="source-2",
            verdict=ElementVerificationVerdict.CONTRADICTS,
        ),
    )

    aggregate = aggregate_truth_condition_claim(
        entry,
        assessments,
        expected_source_ids=("source-1", "source-2"),
    )

    assert aggregate.coverage_state is ClaimCoverageState.CONFLICTED
    assert aggregate.elements[0].semantic_state is ElementSemanticState.CONFLICTED
    assert aggregate.execution_completeness is ExecutionCompleteness.COMPLETE


def test_element_aggregate_rejects_semantic_state_inconsistent_with_sources() -> None:
    entry = _complete_registry().entries[1]
    element = entry.elements[0]
    aggregate = aggregate_truth_condition_claim(
        entry,
        (
            _assessment(
                claim_id=entry.claim_id,
                element_id=element.element_id,
                source_id="source-1",
                verdict=ElementVerificationVerdict.NOT_ENOUGH_INFORMATION,
            ),
        ),
        expected_source_ids=("source-1",),
    ).elements[0]
    payload = aggregate.model_dump(mode="json")
    payload["semantic_state"] = ElementSemanticState.SUPPORTED.value

    with pytest.raises(ValidationError, match="semantic state"):
        ElementTruthConditionAggregate.model_validate(payload)


def test_claim_aggregate_rejects_coverage_inconsistent_with_child_states() -> None:
    entry = _complete_registry().entries[1]
    element = entry.elements[0]
    aggregate = aggregate_truth_condition_claim(
        entry,
        (
            _assessment(
                claim_id=entry.claim_id,
                element_id=element.element_id,
                source_id="source-1",
                verdict=ElementVerificationVerdict.NOT_ENOUGH_INFORMATION,
            ),
        ),
        expected_source_ids=("source-1",),
    )
    payload = aggregate.model_dump(mode="json")
    payload["coverage_state"] = ClaimCoverageState.FULLY_SUPPORTED.value

    with pytest.raises(ValidationError, match="coverage state"):
        ClaimTruthConditionAggregate.model_validate(payload)


def test_full_semantic_support_can_coexist_with_partial_execution() -> None:
    entry = _complete_registry().entries[1]
    element = entry.elements[0]
    assessments = (
        _assessment(
            claim_id=entry.claim_id,
            element_id=element.element_id,
            source_id="source-1",
            verdict=ElementVerificationVerdict.SUPPORTS,
            formal=True,
        ),
        ElementSourceAssessment(
            claim_id=entry.claim_id,
            element_id=element.element_id,
            source_id="source-2",
            execution_status=ElementAssessmentExecutionStatus.MODEL_ERROR,
            diagnostic="timeout",
        ),
    )

    aggregate = aggregate_truth_condition_claim(
        entry,
        assessments,
        expected_source_ids=("source-1", "source-2"),
    )

    assert aggregate.elements[0].semantic_state is ElementSemanticState.SUPPORTED
    assert aggregate.execution_completeness is ExecutionCompleteness.PARTIAL
    assert aggregate.coverage_state is ClaimCoverageState.FULLY_SUPPORTED


def test_all_completed_does_not_support_is_not_supported() -> None:
    entry = _complete_registry().entries[1]
    element = entry.elements[0]
    assessments = tuple(
        _assessment(
            claim_id=entry.claim_id,
            element_id=element.element_id,
            source_id=source_id,
            verdict=ElementVerificationVerdict.DOES_NOT_SUPPORT,
            located=False,
        )
        for source_id in ("source-1", "source-2")
    )

    aggregate = aggregate_truth_condition_claim(
        entry,
        assessments,
        expected_source_ids=("source-1", "source-2"),
    )

    assert aggregate.coverage_state is ClaimCoverageState.NOT_SUPPORTED
    assert aggregate.execution_completeness is ExecutionCompleteness.COMPLETE
    assert aggregate.elements[0].not_supporting_source_ids == (
        "source-1",
        "source-2",
    )
    assert aggregate.elements[0].unresolved_source_ids == ()


def test_insufficient_evidence_is_not_mislabeled_as_unexecuted_source() -> None:
    entry = _complete_registry().entries[1]
    element = entry.elements[0]
    assessment = _assessment(
        claim_id=entry.claim_id,
        element_id=element.element_id,
        source_id="source-1",
        verdict=ElementVerificationVerdict.NOT_ENOUGH_INFORMATION,
        located=False,
    )

    aggregate = aggregate_truth_condition_claim(
        entry,
        (assessment,),
        expected_source_ids=("source-1",),
    )

    assert aggregate.elements[0].insufficient_source_ids == ("source-1",)
    assert aggregate.elements[0].unresolved_source_ids == ()
    assert aggregate.elements[0].semantic_state is ElementSemanticState.UNRESOLVED


def test_unlocatable_support_is_semantically_unresolved_and_execution_failed() -> None:
    entry = _complete_registry().entries[1]
    element = entry.elements[0]
    assessment = ElementSourceAssessment(
        claim_id=entry.claim_id,
        element_id=element.element_id,
        source_id="source-1",
        execution_status=ElementAssessmentExecutionStatus.QUOTE_UNLOCATABLE,
        verdict=ElementVerificationVerdict.SUPPORTS,
        diagnostic="quote not found",
    )

    aggregate = aggregate_truth_condition_claim(
        entry,
        (assessment,),
        expected_source_ids=("source-1",),
    )

    assert aggregate.elements[0].semantic_state is ElementSemanticState.UNRESOLVED
    assert aggregate.execution_completeness is ExecutionCompleteness.FAILED
    assert aggregate.coverage_state is ClaimCoverageState.UNRESOLVED


def test_incomplete_elementization_cannot_aggregate_to_fully_supported() -> None:
    registry = build_truth_condition_registry(
        {"claim-1": CLAIMS["claim-1"]},
        proposals=(_proposal("claim-1", "Alpha acquired Beta."),),
        reviews=(
            _review(
                "claim-1",
                "Alpha acquired Beta.",
                status=ElementizationSemanticStatus.INCOMPLETE,
                missing=("The consideration is missing.",),
            ),
        ),
    )
    entry = registry.entries[0]
    assessment = _assessment(
        claim_id="claim-1",
        element_id=entry.elements[0].element_id,
        source_id="source-1",
        verdict=ElementVerificationVerdict.SUPPORTS,
        formal=True,
    )

    aggregate = aggregate_truth_condition_claim(
        entry,
        (assessment,),
        expected_source_ids=("source-1",),
    )

    assert aggregate.coverage_state is ClaimCoverageState.PARTIALLY_SUPPORTED


def test_review_failure_stays_unresolved_even_if_proposal_was_assessed() -> None:
    registry = build_truth_condition_registry(
        {"claim-1": CLAIMS["claim-1"]},
        proposals=(_proposal("claim-1", "Alpha acquired Beta."),),
        failures=(
            ElementizationFailure(
                claim_id="claim-1",
                stage=ElementizationStage.REVIEW,
                execution_status=ElementizationExecutionStatus.MODEL_ERROR,
                diagnostic="review failed",
            ),
        ),
    )
    entry = registry.entries[0]
    assessment = _assessment(
        claim_id="claim-1",
        element_id=entry.elements[0].element_id,
        source_id="source-1",
        verdict=ElementVerificationVerdict.SUPPORTS,
        formal=True,
    )

    aggregate = aggregate_truth_condition_claim(
        entry,
        (assessment,),
        expected_source_ids=("source-1",),
    )

    assert aggregate.coverage_state is ClaimCoverageState.UNRESOLVED
    assert aggregate.execution_completeness is ExecutionCompleteness.FAILED


def test_empty_selected_denominator_is_closed_not_a_successful_claim() -> None:
    registry = build_truth_condition_registry({})

    assert registry.entries == ()
    assert registry.denominator.selected_claim_ids == ()
    assert registry.denominator.is_closed


def test_registry_aggregation_rejects_unknown_elements_and_binds_hash() -> None:
    registry = _complete_registry()
    bad = _assessment(
        claim_id="claim-1",
        element_id="claim-1::tc-9999",
        source_id="source-1",
        verdict=ElementVerificationVerdict.SUPPORTS,
        formal=True,
    )
    with pytest.raises(ValueError, match="unregistered element"):
        aggregate_truth_condition_registry(
            registry,
            (bad,),
            expected_source_ids_by_claim={
                "claim-1": ("source-1",),
                "claim-2": ("source-1",),
            },
        )

    result = aggregate_truth_condition_registry(
        registry,
        (),
        expected_source_ids_by_claim={},
    )
    assert result is not None
    assert result.registry_sha256 == truth_condition_registry_sha256(registry)
    assert all(
        claim.execution_completeness is ExecutionCompleteness.NOT_RUN
        for claim in result.claims
    )
