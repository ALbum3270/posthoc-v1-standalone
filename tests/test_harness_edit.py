import asyncio
import hashlib
import json

from open_deep_research.harness.claims import (
    AtomicClaim,
    CitationRequirement,
    ClaimNormalizationStatus,
    parse_markdown_blocks,
)
from open_deep_research.harness.edit import (
    EditorialAction,
    EditorialRevisionStatus,
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
