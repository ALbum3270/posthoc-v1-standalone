import asyncio
import hashlib
import json

import pytest
from pydantic import ValidationError

from open_deep_research.harness.budget import (
    RunCostBudgetAudit,
    RunCostCapReached,
)
from open_deep_research.harness.claims import (
    AtomicClaim,
    CitationRequirement,
    ClaimNormalizationStatus,
    EvidenceObligation,
    EvidenceObligationStatus,
    MarkdownBlock,
    MarkdownBlockKind,
    parse_markdown_blocks,
)
from open_deep_research.harness.edit import (
    EditorialAction,
    EditorialDecision,
    EditorialPreservationContext,
    EditorialPreservationImpact,
    EditorialResearchQuestion,
    EditorialRevisionStatus,
    EditorialSettings,
    audit_editorial_admission,
    build_editorial_prompt,
    revise_audited_draft,
)
from open_deep_research.harness.notes import NoteLocationStatus, QuoteSpan
from open_deep_research.harness.verify import (
    ClaimEvidenceState,
    ClaimVerification,
    VerificationRecordStatus,
    VerificationResult,
    VerificationVerdict,
    VerifiedSourceRelation,
)


def _claim(claim_id, block, text, start, end):
    return AtomicClaim(
        claim_id=claim_id,
        block_id=block.block_id,
        selected_text=text,
        claim_text=text,
        anchor_text=text,
        start_char=start,
        end_char=end,
        citation_requirement=CitationRequirement.EXTERNAL,
        normalization_status=ClaimNormalizationStatus.LOCATED,
    )


def _relation(claim_id, verdict, *, formal=False):
    located = verdict in {
        VerificationVerdict.SUPPORTS,
        VerificationVerdict.CONTRADICTS,
    }
    return VerifiedSourceRelation(
        claim_id=claim_id,
        source_id="source-real",
        url="https://source.example/article",
        publisher_domain_proxy="source.example",
        candidate_note_ids=("note-000001",),
        candidate_source_ids=("source-real",),
        status=VerificationRecordStatus.COMPLETED,
        semantic_verdict=verdict,
        explanation="The inspected source does not contain the added detail.",
        source_quote="A narrower supported fact." if located else None,
        span=QuoteSpan(start_char=0, end_char=26) if located else None,
        location_status=NoteLocationStatus.LOCATABLE if located else None,
        is_formal_supporting_evidence=formal,
    )


def _quote_unlocatable_relation(claim_id, source_id):
    return VerifiedSourceRelation(
        claim_id=claim_id,
        source_id=source_id,
        url=f"https://source.example/{source_id}",
        publisher_domain_proxy="source.example",
        candidate_note_ids=("note-000001",),
        candidate_source_ids=(source_id,),
        status=VerificationRecordStatus.QUOTE_UNLOCATABLE,
        semantic_verdict=VerificationVerdict.SUPPORTS,
        explanation="The proposed supporting range did not pass the protocol.",
        is_formal_supporting_evidence=False,
    )


def _synthetic_block(block_id, ordinal):
    text = f"Statement {ordinal}."
    start = ordinal * 100
    return MarkdownBlock(
        block_id=block_id,
        ordinal=ordinal,
        kind=MarkdownBlockKind.PARAGRAPH,
        text=text,
        start_char=start,
        end_char=start + len(text),
    )


def _synthetic_claim(claim_id, block, *, failed=False):
    if failed:
        return AtomicClaim(
            claim_id=claim_id,
            block_id=block.block_id,
            selected_text=block.text,
            claim_text=block.text,
            anchor_text=None,
            citation_requirement=CitationRequirement.EXTERNAL,
            normalization_status=ClaimNormalizationStatus.NORMALIZATION_FAILED,
            normalization_failure="context_span_not_verbatim",
        )
    return _claim(
        claim_id,
        block,
        block.text,
        block.start_char,
        block.end_char,
    )


def _verification(claim, state, relations=()):
    formal = sum(relation.is_formal_supporting_evidence for relation in relations)
    publishers = tuple(
        sorted(
            {
                relation.publisher_domain_proxy
                for relation in relations
                if relation.is_formal_supporting_evidence
            }
        )
    )
    return ClaimVerification(
        claim=claim,
        state=state,
        corroboration_target=2,
        relations=tuple(relations),
        formal_supporting_evidence_count=formal,
        publisher_domain_proxy_count=len(publishers),
        publisher_domain_proxies=publishers,
    )


class ScriptedEditor:
    def __init__(self, content):
        self.content = content
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return {
            "content": json.dumps(self.content),
            "token_count": 17,
            "cost_usd": 0.02,
        }


class SequencedEditor:
    def __init__(self, *contents):
        self.contents = list(contents)
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return {
            "content": json.dumps(self.contents.pop(0)),
            "token_count": 17,
            "cost_usd": 0.02,
        }


class CapAfterFirstEditor:
    def __init__(self, first_content):
        self.first_content = first_content
        self.calls = 0

    async def generate(self, prompt):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": json.dumps(self.first_content),
                "token_count": 17,
                "cost_usd": 0.02,
            }
        raise RunCostCapReached(
            "estimated call cannot fit the remaining run allowance",
            stage="audit_editing",
            audit=RunCostBudgetAudit(
                configured=True,
                max_cost_usd=1.0,
                enforcement=(
                    "pre_call_estimate_admission_plus_observed_usage"
                ),
                observed_total_cost_usd=0.98,
                remaining_cost_usd=0.02,
                observed_overshoot_usd=0.0,
                cap_was_binding=True,
                admitted_call_count=1,
                rejected_call_count=1,
                unestimated_admitted_call_count=0,
            ),
            completed_stages=("audit_editing",),
        )


def test_editor_targets_unresolved_internal_evidence_obligations():
    draft = "# Report\n\nA report-internal conclusion."
    block = parse_markdown_blocks(draft)[1]
    claim = AtomicClaim(
        claim_id="claim-internal",
        block_id=block.block_id,
        selected_text=block.text,
        claim_text=block.text,
        anchor_text=block.text,
        start_char=block.start_char,
        end_char=block.end_char,
        citation_requirement=CitationRequirement.INTERNAL,
        proposed_citation_requirement=CitationRequirement.INTERNAL,
        evidence_obligation=EvidenceObligation(
            claim_id="claim-internal",
            proposed_requirement=CitationRequirement.INTERNAL,
            status=EvidenceObligationStatus.INTERNAL_NOT_SUPPORTED,
            rationale="No independent report-artifact span supports it.",
        ),
        normalization_status=ClaimNormalizationStatus.LOCATED,
    )
    result = VerificationResult(
        claims=(
            _verification(
                claim,
                ClaimEvidenceState.INTERNAL_NOT_SUPPORTED,
            ),
        )
    )

    audit = audit_editorial_admission(result, blocks=(block,))

    assert audit.target_claim_ids == ("claim-internal",)
    assert audit.eligible_target_claim_ids == ("claim-internal",)
    prompt = build_editorial_prompt(
        (block,),
        verification=result,
    )
    assert '"claim_id": "claim-internal"' in prompt
    assert '"evidence_state": "internal_not_supported"' in prompt
    assert '"status": "internal_not_supported"' in prompt
    assert "No independent report-artifact span supports it." in prompt


def test_noop_remove_gets_one_block_local_correction_attempt():
    """Regression for finance-13/16's claimed edit with unchanged bytes."""

    draft = "# Report\n\nA precise but unsupported amount was recovered."
    block = parse_markdown_blocks(draft)[1]
    claim = _claim(
        "claim-0040", block, block.text, block.start_char, block.end_char
    )
    verification = VerificationResult(
        claims=(
            _verification(
                claim,
                ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT,
                (_relation(claim.claim_id, VerificationVerdict.DOES_NOT_SUPPORT),),
            ),
        )
    )
    first_noop = {
        "blocks": [
            {
                "block_id": block.block_id,
                "replacement_text": block.text,
                "decisions": [
                    {
                        "claim_id": claim.claim_id,
                        "action": "remove",
                        "reason": "The inspected source did not support it.",
                    }
                ],
            }
        ]
    }
    corrected = {
        "blocks": [
            {
                "block_id": block.block_id,
                "replacement_text": "",
                "decisions": [
                    {
                        "claim_id": claim.claim_id,
                        "action": "remove",
                        "reason": "The inspected source did not support it.",
                    }
                ],
            }
        ]
    }
    model = SequencedEditor(first_noop, corrected)

    result = asyncio.run(
        revise_audited_draft(
            draft,
            blocks=parse_markdown_blocks(draft),
            verification=verification,
            model_client=model,
        )
    )

    assert len(model.prompts) == 2
    assert "MECHANICAL BLOCK VALIDATION REJECTED" in model.prompts[1]
    assert result.status is EditorialRevisionStatus.COMPLETE
    assert result.evaluated_claim_ids == (claim.claim_id,)
    assert result.edited_draft == "# Report\n\n"
    assert result.changes_applied is True
    assert [record.outcome for record in result.usage] == [
        "partial",
        "correction_completed",
    ]
    assert any(
        diagnostic == f"editorial_block_recovered: {block.block_id}"
        for diagnostic in result.diagnostics
    )


def test_editor_receives_answer_context_and_records_semantic_preservation_judgement():
    """P4: answer preservation is model-judged and audit-visible, not gated."""

    draft = "# Report\n\nA disputed recovery amount was reported."
    block = parse_markdown_blocks(draft)[1]
    claim = _claim(
        "claim-0040", block, block.text, block.start_char, block.end_char
    )
    verification = VerificationResult(
        claims=(
            _verification(
                claim,
                ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT,
                (_relation(claim.claim_id, VerificationVerdict.DOES_NOT_SUPPORT),),
            ),
        )
    )
    context = EditorialPreservationContext(
        topic="What happened to customer funds?",
        research_questions=(
            EditorialResearchQuestion(
                item_id="where-01",
                question="What funds were recovered and distributed?",
            ),
        ),
    )
    model = ScriptedEditor(
        {
            "blocks": [
                {
                    "block_id": block.block_id,
                    "replacement_text": (
                        "Recovery remained under review; the audited amount "
                        "is not stated here."
                    ),
                    "decisions": [
                        {
                            "claim_id": claim.claim_id,
                            "action": "qualify",
                            "reason": "The inspected source does not support the amount.",
                            "preservation_impact": "narrowed_with_evidence",
                            "preservation_rationale": (
                                "The revised sentence keeps the funds-recovery "
                                "answer while removing an unsupported amount."
                            ),
                        }
                    ],
                }
            ]
        }
    )

    result = asyncio.run(
        revise_audited_draft(
            draft,
            blocks=parse_markdown_blocks(draft),
            verification=verification,
            model_client=model,
            preservation_context=context,
        )
    )

    decision = result.block_edits[0].decisions[0]
    assert result.preservation_context == context
    assert decision.preservation_impact is (
        EditorialPreservationImpact.NARROWED_WITH_EVIDENCE
    )
    assert "keeps the funds-recovery answer" in decision.preservation_rationale
    assert "What happened to customer funds?" in model.prompts[0]
    assert "What funds were recovered and distributed?" in model.prompts[0]
    assert "may_reduce_answer_coverage" in model.prompts[0]


def test_editor_prompt_carries_the_draft_language_contract_without_gating():
    """Replacement language follows the topic by model instruction only."""

    draft = "# 报告\n\n这是一条待修订的中文断言。"
    block = parse_markdown_blocks(draft)[1]
    claim = _claim(
        "claim-language", block, block.text, block.start_char, block.end_char
    )
    verification = VerificationResult(
        claims=(
            _verification(
                claim,
                ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT,
                (_relation(claim.claim_id, VerificationVerdict.DOES_NOT_SUPPORT),),
            ),
        )
    )
    context = EditorialPreservationContext(topic="FTX 客户资金去了哪里？")

    prompt = build_editorial_prompt(
        (block,),
        verification=verification,
        preservation_context=context,
    )

    assert "same primary natural language as the\nresearch topic" in prompt
    assert "not a mechanical acceptance test" in prompt
    assert context.topic in prompt


def test_recorded_editorial_preservation_impact_needs_a_rationale():
    with pytest.raises(ValidationError, match="preservation rationale"):
        EditorialDecision(
            claim_id="claim-0040",
            action=EditorialAction.QUALIFY,
            reason="The evidence does not support the amount.",
            preservation_impact=(
                EditorialPreservationImpact.MAY_REDUCE_ANSWER_COVERAGE
            ),
        )


def test_finance_shape_is_edited_by_block_and_keeps_the_original_audit():
    """Reproduce finance-08's real shape: two claims share one long anchor.

    Before this stage existed, both completed does_not_support verdicts flowed
    directly into rendering.  This is an interface addition rather than a
    change to an old function, so the regression cannot be expressed as an
    old-call red/new-call green assertion; it instead fixes the observed
    overlapping-anchor shape that motivated the new stage.
    """

    draft = (
        "# Report\n\n"
        "The company served users worldwide and earned money from fees and "
        "a token."
    )
    block = parse_markdown_blocks(draft)[1]
    first = _claim(
        "claim-0021", block, block.text, block.start_char, block.end_char
    )
    second = _claim(
        "claim-0029", block, block.text, block.start_char, block.end_char
    )
    verification = VerificationResult(
        claims=(
            ClaimVerification(
                claim=first,
                state=ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT,
                corroboration_target=2,
                relations=(
                    _relation(
                        first.claim_id,
                        VerificationVerdict.DOES_NOT_SUPPORT,
                    ),
                ),
                formal_supporting_evidence_count=0,
                publisher_domain_proxy_count=0,
            ),
            ClaimVerification(
                claim=second,
                state=ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT,
                corroboration_target=2,
                relations=(
                    _relation(
                        second.claim_id,
                        VerificationVerdict.DOES_NOT_SUPPORT,
                    ),
                ),
                formal_supporting_evidence_count=0,
                publisher_domain_proxy_count=0,
            ),
        )
    )
    replacement = "The company operated a trading platform."
    model = ScriptedEditor(
        {
            "blocks": [
                {
                    "block_id": block.block_id,
                    "replacement_text": replacement,
                    "decisions": [
                        {
                            "claim_id": "claim-0021",
                            "action": "remove",
                            "reason": "The worldwide scope is dispensable and unsupported.",
                        },
                        {
                            "claim_id": "claim-0029",
                            "action": "qualify",
                            "reason": "Keep only the narrower supported description.",
                        },
                    ],
                }
            ]
        }
    )

    result = asyncio.run(
        revise_audited_draft(
            draft,
            blocks=parse_markdown_blocks(draft),
            verification=verification,
            model_client=model,
        )
    )

    assert result.status is EditorialRevisionStatus.COMPLETE
    assert result.original_draft == draft
    assert result.edited_draft == "# Report\n\n" + replacement
    assert result.original_draft_sha256 == hashlib.sha256(
        draft.encode()
    ).hexdigest()
    assert result.changes_applied is True
    assert result.requires_reaudit is True
    assert [decision.action for decision in result.block_edits[0].decisions] == [
        EditorialAction.REMOVE,
        EditorialAction.QUALIFY,
    ]
    assert "not a lower unsupported\nclaim count" in model.prompts[0]
    assert result.total_tokens == 17
    assert result.total_cost_usd == 0.02


def test_missing_claim_decision_never_applies_a_partial_rewrite():
    draft = "# Report\n\nUnsupported detail one and unsupported detail two."
    block = parse_markdown_blocks(draft)[1]
    claims = tuple(
        _claim(claim_id, block, block.text, block.start_char, block.end_char)
        for claim_id in ("claim-0001", "claim-0002")
    )
    verification = VerificationResult(
        claims=tuple(
            ClaimVerification(
                claim=claim,
                state=ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT,
                corroboration_target=2,
                relations=(
                    _relation(
                        claim.claim_id,
                        VerificationVerdict.DOES_NOT_SUPPORT,
                    ),
                ),
                formal_supporting_evidence_count=0,
                publisher_domain_proxy_count=0,
            )
            for claim in claims
        )
    )
    model = ScriptedEditor(
        {
            "blocks": [
                {
                    "block_id": block.block_id,
                    "replacement_text": "A partial rewrite.",
                    "decisions": [
                        {
                            "claim_id": "claim-0001",
                            "action": "remove",
                            "reason": "Unsupported.",
                        }
                    ],
                }
            ]
        }
    )

    result = asyncio.run(
        revise_audited_draft(
            draft,
            blocks=parse_markdown_blocks(draft),
            verification=verification,
            model_client=model,
        )
    )

    assert result.status is EditorialRevisionStatus.FAILED
    assert result.edited_draft == draft
    assert result.changes_applied is False
    assert result.evaluated_claim_ids == ()
    assert set(result.unevaluated_claim_ids) == {
        "claim-0001",
        "claim-0002",
    }
    assert any(
        "editorial_claim_coverage_error" in diagnostic
        for diagnostic in result.diagnostics
    )


def test_finance_13_accepts_four_valid_blocks_and_keeps_rejected_block() -> None:
    """One unchanged remove proposal cannot discard four valid block edits."""

    paragraphs = [f"Unsupported detail {index}." for index in range(1, 6)]
    draft = "# Report\n\n" + "\n\n".join(paragraphs)
    blocks = parse_markdown_blocks(draft)
    claims = tuple(
        _claim(
            f"claim-{index:04d}",
            block,
            block.text,
            block.start_char,
            block.end_char,
        )
        for index, block in enumerate(blocks[1:], start=1)
    )
    verification = VerificationResult(
        claims=tuple(
            _verification(
                claim,
                ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT,
                (
                    _relation(
                        claim.claim_id,
                        VerificationVerdict.DOES_NOT_SUPPORT,
                    ),
                ),
            )
            for claim in claims
        )
    )
    proposals = []
    for index, (block, claim) in enumerate(
        zip(blocks[1:], claims, strict=True),
        start=1,
    ):
        proposals.append(
            {
                "block_id": block.block_id,
                "replacement_text": (
                    f"Qualified detail {index}." if index < 5 else block.text
                ),
                "decisions": [
                    {
                        "claim_id": claim.claim_id,
                        "action": "qualify" if index < 5 else "remove",
                        "reason": "Use only the audited wording.",
                    }
                ],
            }
        )

    result = asyncio.run(
        revise_audited_draft(
            draft,
            blocks=blocks,
            verification=verification,
            model_client=ScriptedEditor({"blocks": proposals}),
        )
    )

    assert result.status is EditorialRevisionStatus.PARTIAL
    assert result.evaluated_claim_ids == tuple(
        claim.claim_id for claim in claims[:4]
    )
    assert result.unevaluated_claim_ids == (claims[4].claim_id,)
    assert len(result.block_edits) == 4
    assert all(
        f"Qualified detail {index}." in result.edited_draft
        for index in range(1, 5)
    )
    assert paragraphs[4] in result.edited_draft
    assert result.changes_applied is True
    assert result.requires_reaudit is True
    assert any(
        f"editorial_block_rejected: {blocks[5].block_id}" in diagnostic
        for diagnostic in result.diagnostics
    )


def test_cost_cap_after_first_batch_preserves_accepted_editorial_work() -> None:
    """A later admission denial is partial progress, not whole-pass loss."""

    paragraphs = ("Unsupported first detail.", "Unsupported second detail.")
    draft = "# Report\n\n" + "\n\n".join(paragraphs)
    blocks = parse_markdown_blocks(draft)
    claims = tuple(
        _claim(
            f"claim-{index:04d}",
            block,
            block.text,
            block.start_char,
            block.end_char,
        )
        for index, block in enumerate(blocks[1:], start=1)
    )
    verification = VerificationResult(
        claims=tuple(
            _verification(
                claim,
                ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT,
                (
                    _relation(
                        claim.claim_id,
                        VerificationVerdict.DOES_NOT_SUPPORT,
                    ),
                ),
            )
            for claim in claims
        )
    )
    first_proposal = {
        "blocks": [
            {
                "block_id": blocks[1].block_id,
                "replacement_text": "Qualified first detail.",
                "decisions": [
                    {
                        "claim_id": claims[0].claim_id,
                        "action": "qualify",
                        "reason": "Use a narrower statement.",
                    }
                ],
            }
        ]
    }
    model = CapAfterFirstEditor(first_proposal)

    result = asyncio.run(
        revise_audited_draft(
            draft,
            blocks=blocks,
            verification=verification,
            model_client=model,
            settings=EditorialSettings(block_batch_size=1),
        )
    )

    assert model.calls == 2
    assert result.status is EditorialRevisionStatus.PARTIAL
    assert result.evaluated_claim_ids == (claims[0].claim_id,)
    assert result.unevaluated_claim_ids == (claims[1].claim_id,)
    assert "Qualified first detail." in result.edited_draft
    assert paragraphs[1] in result.edited_draft
    assert result.requires_reaudit is True
    assert [record.outcome for record in result.usage] == [
        "completed",
        "budget_exhausted",
    ]
    assert any(
        "editorial_budget_exhausted[2]" in diagnostic
        and f"current_block_ids=('{blocks[2].block_id}',)" in diagnostic
        and f"deferred_block_ids=('{blocks[2].block_id}',)" in diagnostic
        for diagnostic in result.diagnostics
    )


def test_protocol_failure_is_not_recast_as_an_editorial_content_problem():
    draft = "# Report\n\nA claim whose verifier call did not run."
    block = parse_markdown_blocks(draft)[1]
    claim = _claim(
        "claim-0001", block, block.text, block.start_char, block.end_char
    )
    verification = VerificationResult(
        claims=(
            ClaimVerification(
                claim=claim,
                state=ClaimEvidenceState.VERIFICATION_NOT_RUN,
                corroboration_target=2,
                formal_supporting_evidence_count=0,
                publisher_domain_proxy_count=0,
            ),
        )
    )

    class MustNotRun:
        async def generate(self, prompt):
            raise AssertionError("protocol failures are not editorial targets")

    result = asyncio.run(
        revise_audited_draft(
            draft,
            blocks=parse_markdown_blocks(draft),
            verification=verification,
            model_client=MustNotRun(),
        )
    )

    assert result.status is EditorialRevisionStatus.COMPLETE
    assert result.target_claim_ids == ()
    assert result.edited_draft == draft
    assert result.usage == ()


def test_finance_09_shape_gates_only_incomplete_block_dependency_closures():
    """Freeze the observed 111/108/3 relation and 21/15/6 edit shape.

    Before block-local admission, the three incomplete relations caused the
    whole editorial pass to be skipped. The fixture uses the exact observed
    claim and block IDs for the two blocked dependency closures without copying
    any topic-specific prose into the test.
    """

    safe_layout = {
        "block-0001": 2,
        "block-0002": 2,
        "block-0003": 2,
        "block-0004": 2,
        "block-0005": 2,
        "block-0006": 1,
        "block-0007": 1,
        "block-0008": 1,
        "block-0009": 2,
    }
    blocks = [
        _synthetic_block(block_id, ordinal)
        for ordinal, block_id in enumerate(safe_layout, start=1)
    ]
    blocked_22 = _synthetic_block("block-0022", 22)
    blocked_29 = _synthetic_block("block-0029", 29)
    unrelated = _synthetic_block("block-9000", 90)
    blocks.extend((blocked_22, blocked_29, unrelated))

    records = []
    next_target = 1
    for block in blocks[:9]:
        for _ in range(safe_layout[block.block_id]):
            claim = _synthetic_claim(f"claim-{next_target:04d}", block)
            records.append(
                _verification(
                    claim,
                    ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT,
                    (_relation(claim.claim_id, VerificationVerdict.DOES_NOT_SUPPORT),),
                )
            )
            next_target += 1

    for claim_id in ("claim-0075", "claim-0077", "claim-0078"):
        claim = _synthetic_claim(claim_id, blocked_22)
        records.append(
            _verification(
                claim,
                ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT,
                (_relation(claim_id, VerificationVerdict.DOES_NOT_SUPPORT),),
            )
        )
    normalized_failure = _synthetic_claim(
        "claim-0076", blocked_22, failed=True
    )
    records.append(
        _verification(
            normalized_failure,
            ClaimEvidenceState.NORMALIZATION_FAILED,
        )
    )
    for claim_id in ("claim-0102", "claim-0103", "claim-0104"):
        claim = _synthetic_claim(claim_id, blocked_29)
        records.append(
            _verification(
                claim,
                ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT,
                (_relation(claim_id, VerificationVerdict.DOES_NOT_SUPPORT),),
            )
        )
    quote_failure = _synthetic_claim("claim-0105", blocked_29)
    records.append(
        _verification(
            quote_failure,
            ClaimEvidenceState.SUPPORT_QUOTE_UNLOCATABLE,
            (_quote_unlocatable_relation("claim-0105", "source-0105"),),
        )
    )

    # 87 completed non-target relations plus two unrelated incomplete
    # relations establish the real aggregate: 111 total, 108 completed, 3
    # quote_unlocatable. The normalization failure has no relation of its own.
    for index in range(87):
        claim_id = f"claim-filler-{index:04d}"
        claim = _synthetic_claim(claim_id, unrelated)
        records.append(
            _verification(
                claim,
                ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
                (_relation(claim_id, VerificationVerdict.SUPPORTS, formal=True),),
            )
        )
    for index in range(2):
        claim_id = f"claim-unrelated-incomplete-{index + 1}"
        claim = _synthetic_claim(claim_id, unrelated)
        records.append(
            _verification(
                claim,
                ClaimEvidenceState.SUPPORT_QUOTE_UNLOCATABLE,
                (_quote_unlocatable_relation(claim_id, f"source-u{index + 1}"),),
            )
        )

    result = VerificationResult(claims=tuple(records))
    relations = [
        relation for record in result.claims for relation in record.relations
    ]
    audit = audit_editorial_admission(result, blocks=blocks)

    assert len(relations) == 111
    assert sum(
        relation.status is VerificationRecordStatus.COMPLETED
        for relation in relations
    ) == 108
    assert sum(
        relation.status is VerificationRecordStatus.QUOTE_UNLOCATABLE
        for relation in relations
    ) == 3
    assert len(audit.target_claim_ids) == 21
    assert len(audit.eligible_target_claim_ids) == 15
    assert len(audit.blocked_target_claim_ids) == 6
    assert len(audit.eligible_block_ids) == 9
    assert audit.blocked_block_ids == ("block-0022", "block-0029")
    assert audit.gating_unit == "markdown_block"
    assert {
        failure.dependency_claim_id: (
            failure.dependency_role,
            failure.block_id,
            failure.blocked_target_claim_ids,
        )
        for failure in audit.blocked_dependencies
    } == {
        "claim-0076": (
            "co_located_non_target",
            "block-0022",
            ("claim-0075", "claim-0077", "claim-0078"),
        ),
        "claim-0105": (
            "co_located_non_target",
            "block-0029",
            ("claim-0102", "claim-0103", "claim-0104"),
        ),
    }
    failure_reasons = {
        failure.dependency_claim_id: failure.failure_reasons
        for failure in audit.blocked_dependencies
    }
    assert failure_reasons["claim-0076"] == (
        "normalization_failed:context_span_not_verbatim",
    )
    assert failure_reasons["claim-0105"] == (
        "claim_state:support_quote_unlocatable",
        "claim_source_relation_incomplete:source-0105:quote_unlocatable",
    )
    assert audit.unrelated_incomplete_claim_ids == (
        "claim-unrelated-incomplete-1",
        "claim-unrelated-incomplete-2",
    )


def test_same_block_incomplete_dependency_blocks_bytes_but_not_safe_sibling_block():
    draft = "# Report\n\nUnsupported safe detail.\n\nUnsupported blocked detail."
    heading, safe_block, blocked_block = parse_markdown_blocks(draft)
    safe_target = _synthetic_claim("claim-safe", safe_block)
    blocked_target = _synthetic_claim("claim-blocked", blocked_block)
    blocked_dependency = _synthetic_claim(
        "claim-blocked-dependency", blocked_block, failed=True
    )
    verification = VerificationResult(
        claims=(
            _verification(
                safe_target,
                ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT,
                (_relation("claim-safe", VerificationVerdict.DOES_NOT_SUPPORT),),
            ),
            _verification(
                blocked_target,
                ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT,
                (
                    _relation(
                        "claim-blocked", VerificationVerdict.DOES_NOT_SUPPORT
                    ),
                ),
            ),
            _verification(
                blocked_dependency,
                ClaimEvidenceState.NORMALIZATION_FAILED,
            ),
        )
    )
    replacement = "Qualified safe detail."
    model = ScriptedEditor(
        {
            "blocks": [
                {
                    "block_id": safe_block.block_id,
                    "replacement_text": replacement,
                    "decisions": [
                        {
                            "claim_id": "claim-safe",
                            "action": "qualify",
                            "reason": "Use the narrower audited wording.",
                        }
                    ],
                }
            ]
        }
    )

    result = asyncio.run(
        revise_audited_draft(
            draft,
            blocks=(heading, safe_block, blocked_block),
            verification=verification,
            model_client=model,
        )
    )

    assert result.status is EditorialRevisionStatus.COMPLETE
    assert result.eligible_target_claim_ids == ("claim-safe",)
    assert result.blocked_target_claim_ids == ("claim-blocked",)
    assert result.evaluated_claim_ids == ("claim-safe",)
    assert result.edited_draft == (
        "# Report\n\nQualified safe detail.\n\nUnsupported blocked detail."
    )
    assert "claim-blocked" not in model.prompts[0]
    assert result.blocked_dependencies[0].dependency_role == (
        "co_located_non_target"
    )
