from __future__ import annotations

import json
import re
from pathlib import Path

from open_deep_research.harness.claims import (
    AtomicClaim,
    CitationRequirement,
    ClaimNormalizationStatus,
    ClaimRegistryCoverage,
    SourceResolution,
)
from open_deep_research.harness.notes import NoteLocationStatus, QuoteSpan
from open_deep_research.harness.render import render_verified_report
from open_deep_research.harness.verify import (
    ClaimEvidenceState,
    ClaimVerification,
    VerificationRecordStatus,
    VerificationResult,
    VerificationVerdict,
    VerifiedSourceRelation,
)
from open_deep_research.harness.write import parse_report_citations

_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "harness_posthoc_b1407b.json"
)
_DEFINITION = re.compile(r"^\[\^(\d+)\]:", re.MULTILINE)


def _claim(
    draft: str,
    claim_id: str,
    anchor: str,
) -> AtomicClaim:
    start = draft.index(anchor)
    return AtomicClaim(
        claim_id=claim_id,
        block_id=f"block-{claim_id}",
        selected_text=anchor,
        claim_text=f"Self-contained {claim_id}.",
        anchor_text=anchor,
        start_char=start,
        end_char=start + len(anchor),
        citation_requirement=CitationRequirement.EXTERNAL,
        source_resolution=SourceResolution.DIRECT,
        normalization_status=ClaimNormalizationStatus.LOCATED,
    )


def _support(
    claim_id: str,
    *,
    source_id: str = "source-shared",
    source_quote: str = "Exact source-authored evidence.",
    model_quote: str = "MODEL WORDING MUST NEVER BE RENDERED",
    url: str = "https://source.example/article",
) -> VerifiedSourceRelation:
    return VerifiedSourceRelation(
        claim_id=claim_id,
        source_id=source_id,
        url=url,
        publisher_domain_proxy="source.example",
        candidate_note_ids=("note-1",),
        candidate_source_ids=(source_id,),
        status=VerificationRecordStatus.COMPLETED,
        semantic_verdict=VerificationVerdict.SUPPORTS,
        model_quote=model_quote,
        source_quote=source_quote,
        span=QuoteSpan(start_char=10, end_char=41),
        location_status=NoteLocationStatus.LOCATABLE,
        is_formal_supporting_evidence=True,
    )


def _verified(
    claim: AtomicClaim,
    state: ClaimEvidenceState,
    *relations: VerifiedSourceRelation,
    required: int = 2,
) -> ClaimVerification:
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
        required_independent_sources=required,
        relations=tuple(relations),
        formal_supporting_evidence_count=sum(
            relation.is_formal_supporting_evidence
            for relation in relations
        ),
        publisher_domain_proxy_count=len(publishers),
        publisher_domain_proxies=publishers,
    )


def test_code_assigns_one_global_footnote_per_evidence_span() -> None:
    draft = "First assertion. Second assertion."
    first = _claim(draft, "claim-1", "First assertion.")
    second = _claim(draft, "claim-2", "Second assertion.")
    evidence = _support("claim-1")
    repeated = evidence.model_copy(update={"claim_id": "claim-2"})
    verification = VerificationResult(
        claims=(
            _verified(
                first,
                ClaimEvidenceState.SUPPORTED_BELOW_REQUIREMENT,
                evidence,
            ),
            _verified(
                second,
                ClaimEvidenceState.SUPPORTED_BELOW_REQUIREMENT,
                repeated,
            ),
        )
    )

    rendered = render_verified_report(draft, verification)

    assert rendered.markdown.count("[^1]") == 3
    assert rendered.markdown.count("[^1]:") == 1
    assert "[^2]" not in rendered.markdown
    assert len(rendered.footnotes) == 1
    assert rendered.footnotes[0].key.model_dump() == {
        "source_id": "source-shared",
        "start_char": 10,
        "end_char": 41,
    }
    assert rendered.markdown.count("〔单一来源：1/2〕") == 2
    assert "Exact source-authored evidence." in rendered.markdown
    assert "MODEL WORDING MUST NEVER BE RENDERED" not in rendered.markdown

    parsed = parse_report_citations(rendered.markdown)
    assert {citation.quote for citation in parsed.citations} == {
        "Exact source-authored evidence."
    }


def test_same_sentence_claims_keep_distinct_evidence_states() -> None:
    draft = "Alpha happened, while Beta remained open."
    supported = _claim(draft, "claim-supported", "Alpha happened")
    no_candidate = _claim(
        draft,
        "claim-no-candidate",
        "Beta remained open",
    )
    verification = VerificationResult(
        claims=(
            _verified(
                supported,
                ClaimEvidenceState.CORROBORATED,
                _support("claim-supported"),
                required=1,
            ),
            _verified(
                no_candidate,
                ClaimEvidenceState.NO_CANDIDATE_SOURCE,
            ),
        )
    )

    rendered = render_verified_report(draft, verification)

    assert (
        "Alpha happened[^1], while "
        "Beta remained open〔未找到候选来源〕."
    ) in rendered.markdown
    assert [annotation.claim_id for annotation in rendered.annotations] == [
        "claim-supported",
        "claim-no-candidate",
    ]
    assert {
        annotation.evidence_state for annotation in rendered.annotations
    } == {
        ClaimEvidenceState.CORROBORATED,
        ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    }


def test_conflict_and_unverified_reasons_are_visible_at_claim_anchors() -> None:
    draft = "The disputed assertion. The pending assertion."
    conflict_claim = _claim(
        draft,
        "claim-conflict",
        "The disputed assertion.",
    )
    pending_claim = _claim(
        draft,
        "claim-pending",
        "The pending assertion.",
    )
    support = _support("claim-conflict")
    contradiction = _support(
        "claim-conflict",
        source_id="source-contradiction",
        source_quote="Exact source-authored contradiction.",
        url="https://other.example/article",
    ).model_copy(
        update={
            "publisher_domain_proxy": "other.example",
            "semantic_verdict": VerificationVerdict.CONTRADICTS,
            "is_formal_supporting_evidence": False,
        }
    )
    pending = VerifiedSourceRelation(
        claim_id="claim-pending",
        source_id="source-pending",
        url="https://pending.example/article",
        publisher_domain_proxy="pending.example",
        candidate_note_ids=("note-pending",),
        candidate_source_ids=("source-pending",),
        status=VerificationRecordStatus.VERIFICATION_NOT_RUN_BUDGET,
        error="estimated call exceeds remaining budget",
    )
    verification = VerificationResult(
        claims=(
            _verified(
                conflict_claim,
                ClaimEvidenceState.CONFLICTING_EVIDENCE,
                support,
                contradiction,
            ),
            _verified(
                pending_claim,
                ClaimEvidenceState.VERIFICATION_NOT_RUN,
                pending,
            ),
        )
    )

    rendered = render_verified_report(
        draft,
        verification,
        settled_without_located_evidence=1,
        settled_without_located_evidence_item_ids=("where-01",),
    )

    assert "〔来源冲突：支持[^1]；反驳[^2]〕" in rendered.markdown
    assert "〔未核验：预算耗尽〕" in rendered.markdown
    assert (
        "settled_without_located_evidence=1 (where-01)"
        in rendered.evidence_summary_line
    )
    assert rendered.summary.conflicting == 1
    assert rendered.summary.unverified == 1


def test_summary_splits_each_unverified_failure_mode() -> None:
    draft = "Partial. Unrun. Unlocatable. Normalization."
    partial_claim = _claim(draft, "claim-partial", "Partial.")
    unrun_claim = _claim(draft, "claim-unrun", "Unrun.")
    unlocatable_claim = _claim(
        draft,
        "claim-unlocatable",
        "Unlocatable.",
    )
    normalization_claim = AtomicClaim(
        claim_id="claim-normalization",
        block_id="block-normalization",
        selected_text="Normalization.",
        claim_text="Normalization failed.",
        anchor_text=None,
        start_char=None,
        end_char=None,
        citation_requirement=CitationRequirement.EXTERNAL,
        source_resolution=SourceResolution.UNRESOLVED,
        normalization_status=(
            ClaimNormalizationStatus.NORMALIZATION_FAILED
        ),
        normalization_failure="anchor_not_found",
    )
    budget_failure = VerifiedSourceRelation(
        claim_id=partial_claim.claim_id,
        source_id="source-pending",
        url="https://pending.example/article",
        publisher_domain_proxy="pending.example",
        candidate_note_ids=("note-pending",),
        candidate_source_ids=("source-pending",),
        status=VerificationRecordStatus.VERIFICATION_NOT_RUN_BUDGET,
        error="estimated call exceeds remaining budget",
    )
    unrun_failure = budget_failure.model_copy(
        update={"claim_id": unrun_claim.claim_id}
    )
    unlocatable_relation = VerifiedSourceRelation(
        claim_id=unlocatable_claim.claim_id,
        source_id="source-unlocatable",
        url="https://unlocatable.example/article",
        publisher_domain_proxy="unlocatable.example",
        candidate_note_ids=("note-unlocatable",),
        candidate_source_ids=("source-unlocatable",),
        status=VerificationRecordStatus.QUOTE_UNLOCATABLE,
        semantic_verdict=VerificationVerdict.SUPPORTS,
        model_quote="Model-proposed quote.",
        location_status=NoteLocationStatus.UNLOCATABLE,
        quote_failure_reason="quote_not_found",
        is_formal_supporting_evidence=False,
    )
    verification = VerificationResult(
        claims=(
            _verified(
                partial_claim,
                ClaimEvidenceState.VERIFICATION_INCOMPLETE,
                _support("claim-partial"),
                budget_failure,
            ),
            _verified(
                unrun_claim,
                ClaimEvidenceState.VERIFICATION_NOT_RUN,
                unrun_failure,
            ),
            _verified(
                unlocatable_claim,
                ClaimEvidenceState.SUPPORT_QUOTE_UNLOCATABLE,
                unlocatable_relation,
            ),
            _verified(
                normalization_claim,
                ClaimEvidenceState.NORMALIZATION_FAILED,
            ),
        )
    )

    rendered = render_verified_report(draft, verification)

    assert rendered.summary.verification_incomplete == 1
    assert rendered.summary.verification_not_run == 1
    assert rendered.summary.support_quote_unlocatable == 1
    assert rendered.summary.claim_normalization_failed == 1
    assert rendered.summary.attribution_error == 0
    assert rendered.summary.unverified == 4
    assert "核验不完整 1" in rendered.evidence_summary_line
    assert "完全未核验 1" in rendered.evidence_summary_line
    assert "支持性引文无法定位 1" in rendered.evidence_summary_line
    assert "claim 定位失败 1" in rendered.evidence_summary_line
    assert "归因错误 0" in rendered.evidence_summary_line
    assert "未核验 4" not in rendered.evidence_summary_line


def test_renderer_removes_legacy_model_footnotes_and_is_byte_deterministic() -> None:
    fixture = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    legacy = fixture["report_markdown"]
    empty_verification = VerificationResult(claims=())

    first = render_verified_report(legacy, empty_verification)
    second = render_verified_report(legacy, empty_verification)

    assert first.markdown == second.markdown
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.removed_model_footnote_definitions == (
        fixture["expectations"]["report_reference_definition_count"]
    )
    assert first.removed_model_footnote_markers > 0
    assert _DEFINITION.findall(first.markdown) == []
    assert "[^11]" not in first.markdown
    assert "[^13]" not in first.markdown


def test_summary_discloses_when_claim_registry_does_not_cover_the_report() -> None:
    coverage = ClaimRegistryCoverage(
        evaluated_blocks=3,
        total_blocks=27,
        unassessed_blocks=24,
        unassessed_block_ids=tuple(
            f"block-{index:04d}" for index in range(4, 28)
        ),
        is_complete=False,
    )

    rendered = render_verified_report(
        "# Report\n\nUnassessed narrative.",
        VerificationResult(claims=()),
        registry_coverage=coverage,
    )

    assert "正文块评估 3/27" in rendered.evidence_summary_line
    assert "未评估块 24" in rendered.evidence_summary_line
    assert "以下断言统计仅覆盖已评估块" in rendered.evidence_summary_line
    assert "已识别外部可核验断言 0" in rendered.evidence_summary_line
    assert "已识别断言中核验不完整 0" in rendered.evidence_summary_line
    assert "完全未核验 0" in rendered.evidence_summary_line
    assert "支持性引文无法定位 0" in rendered.evidence_summary_line
    assert "claim 定位失败 0" in rendered.evidence_summary_line
    assert rendered.summary.registry_coverage == coverage
