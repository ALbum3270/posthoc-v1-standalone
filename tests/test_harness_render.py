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
from open_deep_research.harness.render import (
    InitialCollectionSnapshot,
    render_verified_report,
)
from open_deep_research.harness.verify import (
    ClaimEvidenceState,
    ClaimVerification,
    VerificationRecordStatus,
    VerificationResult,
    VerificationVerdict,
    VerifiedSourceRelation,
)
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
                ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
                evidence,
            ),
            _verified(
                second,
                ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
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
    assert "〔单一发布方支持〕" not in rendered.markdown
    assert rendered.evidence_legend_line in rendered.markdown
    assert (
        rendered.evidence_legend_line
        == "> 图例：带脚注且无额外状态标签 = "
        "单一发布方提供了可定位支持引文"
    )
    assert "单一发布方支持 2" in rendered.evidence_summary_line
    assert "多发布方交叉支持 0" in rendered.evidence_summary_line
    assert "零发布方支持 0" in rendered.evidence_summary_line
    assert "1/2" not in rendered.markdown
    assert "充分支持" not in rendered.evidence_summary_line
    assert "已核实" not in rendered.evidence_summary_line
    assert rendered.summary.claims_with_located_support == 2
    assert rendered.summary.single_publisher_support == 2
    assert rendered.summary.multi_publisher_support == 0
    assert rendered.summary.zero_publisher_support == 0
    assert "Exact source-authored evidence." not in rendered.markdown
    assert "Exact source-authored evidence." in rendered.sources_markdown
    assert "MODEL WORDING MUST NEVER BE RENDERED" not in rendered.markdown
    assert (
        "MODEL WORDING MUST NEVER BE RENDERED"
        not in rendered.sources_markdown
    )
    assert (
        "[逐字证据](report.sources.md#evidence-1)"
        in rendered.markdown
    )
    assert rendered.footnote_format_line in rendered.markdown
    assert (
        "[^1]: `source.example` · 支持 · "
        "[逐字证据](report.sources.md#evidence-1) · "
        "[原文][source-1]"
    ) in rendered.markdown
    assert rendered.markdown.count(
        "[source-1]: <https://source.example/article>"
    ) == 1
    assert rendered.sources_markdown.count(
        "[source-1]: <https://source.example/article>"
    ) == 1
    assert '<a id="evidence-1"></a>' in rendered.sources_markdown
    assert rendered.bundle_validation.model_dump() == {
        "every_definition_has_unique_source_anchor": True,
        "every_definition_has_marker": True,
        "every_definition_links_to_source_anchor": True,
        "every_marker_has_local_definition": True,
        "every_source_anchor_has_definition": True,
        "every_source_entry_contains_full_quote": True,
        "footnote_count": 1,
        "local_definition_count": 1,
        "no_duplicate_definitions": True,
        "no_duplicate_source_anchors": True,
        "every_footnote_uses_expected_url_reference": True,
        "report_and_sources_url_references_match": True,
        "report_url_reference_definition_count": 1,
        "report_url_references_are_unique": True,
        "source_anchor_count": 1,
        "sources_url_reference_definition_count": 1,
        "sources_url_references_are_unique": True,
        "sources_sha256_matches": True,
        "unique_source_url_count": 1,
    }


def test_url_references_are_deduplicated_by_first_footnote_occurrence() -> None:
    draft = "First assertion. Second assertion."
    first = _claim(draft, "claim-1", "First assertion.")
    second = _claim(draft, "claim-2", "Second assertion.")
    shared_url = "https://source.example/shared"
    first_relation = _support(
        "claim-1",
        source_id="source-a",
        source_quote="First source quote.",
        url=shared_url,
    ).model_copy(
        update={"span": QuoteSpan(start_char=10, end_char=29)}
    )
    second_relation = _support(
        "claim-2",
        source_id="source-b",
        source_quote="Second source quote.",
        url=shared_url,
    ).model_copy(
        update={"span": QuoteSpan(start_char=40, end_char=60)}
    )
    verification = VerificationResult(
        claims=(
            _verified(
                first,
                ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
                first_relation,
            ),
            _verified(
                second,
                ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
                second_relation,
            ),
        )
    )

    rendered = render_verified_report(draft, verification)

    assert len(rendered.footnotes) == 2
    assert "[^1]:" in rendered.markdown
    assert "[^2]:" in rendered.markdown
    assert rendered.markdown.count("[原文][source-1]") == 2
    assert rendered.markdown.count(
        f"[source-1]: <{shared_url}>"
    ) == 1
    assert rendered.sources_markdown.count(
        f"[source-1]: <{shared_url}>"
    ) == 1
    assert "[source-2]:" not in rendered.markdown
    assert "[source-2]:" not in rendered.sources_markdown


def test_renderer_does_not_read_corroboration_target() -> None:
    draft = "One assertion."
    claim = _claim(draft, "claim-1", "One assertion.")
    evidence = _support("claim-1")
    target_one = VerificationResult(
        claims=(
            _verified(
                claim,
                ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
                evidence,
                required=1,
            ),
        )
    )
    target_two = VerificationResult(
        claims=(
            _verified(
                claim,
                ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
                evidence,
                required=2,
            ),
        )
    )

    first = render_verified_report(draft, target_one)
    second = render_verified_report(draft, target_two)

    assert first.markdown == second.markdown
    assert first.sources_markdown == second.sources_markdown
    assert first.evidence_summary_line == second.evidence_summary_line
    assert "〔单一发布方支持〕" not in first.markdown


def test_support_label_rule_does_not_adapt_to_single_publisher_prevalence() -> None:
    anchors = tuple(f"Claim {index:02d}." for index in range(20))
    draft = " ".join(anchors)
    claims = tuple(
        _claim(draft, f"claim-{index:02d}", anchor)
        for index, anchor in enumerate(anchors)
    )

    def render_with_single_publisher_count(count: int):
        entries = []
        for index, claim in enumerate(claims):
            first = _support(
                claim.claim_id,
                source_id=f"source-{index:02d}-a",
                url=f"https://first-{index:02d}.example/article",
            ).model_copy(
                update={
                    "publisher_domain_proxy": (
                        f"first-{index:02d}.example"
                    ),
                }
            )
            if index < count:
                entries.append(
                    _verified(
                        claim,
                        ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
                        first,
                    )
                )
                continue
            second = _support(
                claim.claim_id,
                source_id=f"source-{index:02d}-b",
                source_quote="Second exact source-authored evidence.",
                url=f"https://second-{index:02d}.example/article",
            ).model_copy(
                update={
                    "publisher_domain_proxy": (
                        f"second-{index:02d}.example"
                    ),
                    "span": QuoteSpan(start_char=50, end_char=88),
                }
            )
            entries.append(
                _verified(
                    claim,
                    ClaimEvidenceState.CORROBORATED,
                    first,
                    second,
                )
            )
        return render_verified_report(
            draft,
            VerificationResult(claims=tuple(entries)),
        )

    five_percent = render_with_single_publisher_count(1)
    ninety_five_percent = render_with_single_publisher_count(19)

    assert five_percent.evidence_legend_line == (
        ninety_five_percent.evidence_legend_line
    )
    for rendered, expected_single_count in (
        (five_percent, 1),
        (ninety_five_percent, 19),
    ):
        assert rendered.markdown.count(rendered.evidence_legend_line) == 1
        assert "〔单一发布方支持〕" not in rendered.markdown
        assert rendered.summary.single_publisher_support == (
            expected_single_count
        )
        assert rendered.summary.multi_publisher_support == (
            20 - expected_single_count
        )
        for annotation in rendered.annotations:
            if (
                annotation.evidence_state
                == ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER
            ):
                assert re.fullmatch(
                    r"\[\^\d+\]",
                    annotation.rendered_suffix,
                )
            else:
                assert annotation.evidence_state == (
                    ClaimEvidenceState.CORROBORATED
                )
                assert annotation.rendered_suffix.endswith(
                    "〔多发布方交叉支持〕"
                )


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
        "Alpha happened[^1]〔多发布方交叉支持〕, while "
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
        rejected_exhausted_without_collection_attempt=1,
        rejected_exhausted_without_collection_attempt_item_ids=("how-01",),
        accepted_exhausted_attempt_unknown_legacy=1,
        accepted_exhausted_attempt_unknown_legacy_item_ids=("when-01",),
        exhausted_with_unread_candidates=1,
        exhausted_with_unread_candidates_item_ids=("what-01",),
    )

    assert "〔来源冲突：支持[^1]；反驳[^2]〕" in rendered.markdown
    assert "〔未核验：预算耗尽〕" in rendered.markdown
    assert (
        "settled_without_located_evidence=1 (where-01)"
        in rendered.evidence_summary_line
    )
    assert (
        "拒绝无采集尝试的查遍未找到声明 1 (how-01)"
        in rendered.evidence_summary_line
    )
    assert (
        "历史查遍未找到声明缺少尝试快照 1 (when-01)"
        in rendered.evidence_summary_line
    )
    assert (
        "仍有未读候选时判为查遍未找到 1 (what-01)"
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
    assert "Unlocatable.〔未核验：支持性引文无法定位〕" in (
        rendered.markdown
    )
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


def test_zero_formal_support_explains_coverage_without_rewriting_counts() -> None:
    draft = "A claim without candidate evidence."
    claim = _claim(draft, "claim-zero", draft)
    verification = VerificationResult(
        claims=(
            _verified(
                claim,
                ClaimEvidenceState.NO_CANDIDATE_SOURCE,
            ),
        )
    )

    rendered = render_verified_report(
        draft,
        verification,
        initial_collection_snapshot=InitialCollectionSnapshot(
            cached_source_count=0,
            note_count=0,
            usable_note_count=0,
        ),
    )

    assert rendered.evidence_status_line is not None
    assert (
        "本报告没有任何可定位的正式支持关系"
        in rendered.evidence_status_line
    )
    assert "初次采集阶段未取得任何原文" in rendered.evidence_status_line
    assert rendered.summary.external_claims == 1
    assert rendered.summary.zero_publisher_support == 1
    status_at = rendered.markdown.index("证据状态：")
    summary_at = rendered.markdown.index("证据摘要：")
    checklist_at = rendered.markdown.index(
        "清单内容覆盖（不表示来源支持）："
    )
    assert status_at < summary_at < checklist_at
    assert "清单对账：" not in rendered.markdown


def test_zero_formal_support_does_not_relabel_collected_sources_as_absent() -> None:
    rendered = render_verified_report(
        "Narrative.",
        VerificationResult(claims=()),
        initial_collection_snapshot=InitialCollectionSnapshot(
            cached_source_count=2,
            note_count=3,
            usable_note_count=2,
        ),
    )

    assert rendered.evidence_status_line is not None
    assert "没有任何可定位的正式支持关系" in rendered.evidence_status_line
    assert "初次采集阶段未取得任何原文" not in rendered.markdown


def test_formal_support_suppresses_zero_evidence_warning() -> None:
    draft = "Supported assertion."
    claim = _claim(draft, "claim-supported", draft)
    relation = _support("claim-supported")
    verification = VerificationResult(
        claims=(
            _verified(
                claim,
                ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
                relation,
            ),
        )
    )

    rendered = render_verified_report(
        draft,
        verification,
        initial_collection_snapshot=InitialCollectionSnapshot(
            cached_source_count=0,
            note_count=0,
            usable_note_count=0,
        ),
    )

    assert rendered.evidence_status_line is None
    assert "证据状态：" not in rendered.markdown
    assert "初次采集阶段未取得任何原文" not in rendered.markdown


def test_report_discloses_that_a_cost_ceiling_cut_the_run_short() -> None:
    """The interruption has to be visible to whoever reads the report.

    The person deciding whether to pay for a longer run reads the report, not
    audit.json. Recording the cutoff only in the audit file hands them the
    decision while withholding the one fact it turns on.
    """

    from open_deep_research.harness.budget_diagnostics import (
        BudgetDecisionSignal,
        CompletionStatus,
        OutstandingWork,
        ResourceStopReason,
        RunStopDiagnostic,
        StopBoundary,
    )

    draft = "First assertion."
    verification = VerificationResult(
        claims=(
            _verified(
                _claim(draft, "claim-1", "First assertion."),
                ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
                _support("claim-1"),
            ),
        )
    )
    diagnostic = RunStopDiagnostic(
        resource_stop_reason=ResourceStopReason.RUN_COST_CAP_REACHED,
        completion_status=CompletionStatus.PARTIAL,
        boundary=StopBoundary(
            scope="run", resource="cost_usd", used=0.5, limit=0.5
        ),
        cap_was_binding=True,
        outstanding=OutstandingWork(
            open_checklist_items=2, unverified_relations=18
        ),
        budget_decision_signal=BudgetDecisionSignal.INDETERMINATE,
    )

    rendered = render_verified_report(
        draft, verification, stop_diagnostic=diagnostic
    )

    assert "本次运行被成本上限截断（run cost_usd 0.5/0.5）" in rendered.markdown
    assert "未结清单项 2" in rendered.markdown
    assert "因预算未核验关系 18" in rendered.markdown
    # Mixed evidence must not be rendered as advice to spend money.
    assert "证据不足以判断加预算是否有用" in rendered.markdown
    assert "提高上限可能有用" not in rendered.markdown


def test_a_run_that_was_never_cut_off_says_nothing_about_budget() -> None:
    draft = "First assertion."
    verification = VerificationResult(
        claims=(
            _verified(
                _claim(draft, "claim-1", "First assertion."),
                ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
                _support("claim-1"),
            ),
        )
    )

    rendered = render_verified_report(draft, verification)

    assert "成本上限截断" not in rendered.markdown


def test_report_discloses_a_stage_that_budget_skipped_before_it_ran() -> None:
    """Silently deleting a planned round is the failure this line prevents."""

    from open_deep_research.harness.budget_diagnostics import (
        BudgetDecisionSignal,
        CompletionStatus,
        OutstandingWork,
        ResourceStopReason,
        RunStopDiagnostic,
    )

    draft = "First assertion."
    verification = VerificationResult(
        claims=(
            _verified(
                _claim(draft, "claim-1", "First assertion."),
                ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
                _support("claim-1"),
            ),
        )
    )
    diagnostic = RunStopDiagnostic(
        resource_stop_reason=ResourceStopReason.NOT_RESOURCE_LIMITED,
        completion_status=CompletionStatus.PARTIAL,
        cap_was_binding=False,
        budget_curtailed_stages=("evidence_gap", "disagreement_detection"),
        outstanding=OutstandingWork(evidence_gap_plan_unexecuted=True),
        budget_decision_signal=BudgetDecisionSignal.INDETERMINATE,
    )

    rendered = render_verified_report(
        draft, verification, stop_diagnostic=diagnostic
    )

    assert "evidence_gap、disagreement_detection" in rendered.markdown
    assert "因预算不足提前停止" in rendered.markdown
    # These stages may have spent money before stopping; never call them skipped.
    assert "可能已完成部分工作后停下" in rendered.markdown
    assert "被跳过" not in rendered.markdown
    # This was not a cap hit, so it must not be described as one.
    assert "成本上限截断" not in rendered.markdown
