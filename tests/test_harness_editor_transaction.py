import json

import pytest
from pydantic import ValidationError

from open_deep_research.harness.editor_transaction import (
    AnswerPreservation,
    AuditUnitInput,
    CharacterEdit,
    EditorialIntent,
    EditorialTransactionResult,
    TextSpanSnapshot,
    TransactionDecision,
    TransactionExecutionStatus,
    TransactionReviewRequest,
    TransactionTargetRequirement,
    build_editorial_affected_scope,
    build_editorial_change_manifest,
    build_transaction_review_request,
    build_transaction_reviewer_prompt,
    parse_transaction_review,
)


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _span(draft: str, text: str) -> TextSpanSnapshot:
    start = draft.index(text)
    return TextSpanSnapshot(
        text=text,
        start_char=start,
        end_char=start + len(text),
        text_sha256=_sha(text),
    )


def _unit(
    draft: str,
    surface: str,
    *,
    context: tuple[str, ...] = (),
    locator: str | None = None,
    state: str = "supported",
) -> AuditUnitInput:
    return AuditUnitInput(
        report_surface=_span(draft, surface),
        semantic_text=surface,
        context_spans=tuple(_span(draft, item) for item in context),
        audit_payload={"state": state},
        registry_locator=locator,
    )


def _edit(draft: str, old: str, new: str) -> CharacterEdit:
    start = draft.index(old)
    return CharacterEdit(
        start_char=start,
        end_char=start + len(old),
        original_text=old,
        replacement_text=new,
    )


def _review_fixture():
    original = "# 标题\n\n旧事实。"
    edit = _edit(original, "旧事实。", "较窄的新事实。")
    proposed = original.replace("旧事实。", "较窄的新事实。")
    manifest = build_editorial_change_manifest(
        original,
        proposed,
        edits=(edit,),
    )
    scope = build_editorial_affected_scope(
        manifest,
        original_draft=original,
        proposed_draft=proposed,
        pre_units=(_unit(original, "旧事实。", locator="claim-91"),),
        post_units=(_unit(proposed, "较窄的新事实。", locator="claim-01"),),
    )
    target_ref = scope.affected_pre_units[0].unit_ref
    post_ref = scope.affected_post_units[0].unit_ref
    request = build_transaction_review_request(
        scope,
        target_intents={target_ref: EditorialIntent.QUALIFY},
    )
    return manifest, scope, request, target_ref, post_ref


def test_manifest_uses_character_ranges_and_reconstructs_unicode_edit():
    original = "# 时间线\n\n融资额为400万美元。\n\n结论保持不变。"
    edit = _edit(original, "400万美元", "4亿美元")
    proposed = original.replace("400万美元", "4亿美元")

    manifest = build_editorial_change_manifest(
        original,
        proposed,
        edits=(edit,),
    )

    recorded = manifest.edits[0]
    assert recorded.pre_range.end_char - recorded.pre_range.start_char == len(
        "400万美元"
    )
    assert recorded.post_range.end_char - recorded.post_range.start_char == len(
        "4亿美元"
    )
    assert recorded.original_text_sha256 == _sha("400万美元")
    assert recorded.replacement_text_sha256 == _sha("4亿美元")
    assert manifest.original_draft_sha256 == _sha(original)
    assert manifest.proposed_draft_sha256 == _sha(proposed)


def test_affected_scope_identity_ignores_registry_ids_and_input_order():
    original = "# 旧标题\n\n第一项事实。\n\n第二项事实。"
    edit = _edit(original, "# 旧标题", "# 新标题")
    proposed = original.replace("# 旧标题", "# 新标题")
    manifest = build_editorial_change_manifest(
        original,
        proposed,
        edits=(edit,),
    )
    pre_a = (
        _unit(original, "第一项事实。", locator="claim-0001"),
        _unit(original, "第二项事实。", locator="claim-0002"),
    )
    post_a = (
        _unit(proposed, "第一项事实。", locator="claim-0002"),
        _unit(proposed, "第二项事实。", locator="claim-0001"),
    )
    scope_a = build_editorial_affected_scope(
        manifest,
        original_draft=original,
        proposed_draft=proposed,
        pre_units=pre_a,
        post_units=post_a,
    )
    scope_b = build_editorial_affected_scope(
        manifest,
        original_draft=original,
        proposed_draft=proposed,
        pre_units=tuple(reversed(pre_a)),
        post_units=tuple(reversed(post_a)),
    )

    assert {item.unit_ref for item in scope_a.affected_pre_units} == {
        item.unit_ref for item in scope_b.affected_pre_units
    }
    assert {item.unit_ref for item in scope_a.affected_post_units} == {
        item.unit_ref for item in scope_b.affected_post_units
    }
    assert all("claim-" not in item.unit_ref for item in scope_a.affected_pre_units)


def test_deleting_block_does_not_make_renumbered_following_block_affected():
    original = "# 报告\n\n删除这一段。\n\n后续事实保持原样。"
    edit = _edit(original, "删除这一段。", "")
    proposed = original.replace("删除这一段。", "")

    manifest = build_editorial_change_manifest(
        original,
        proposed,
        edits=(edit,),
    )

    affected_pre_text = {
        block.span.text
        for block in manifest.pre_blocks
        if block.block_ref in manifest.affected_pre_block_refs
    }
    affected_post_text = {
        block.span.text
        for block in manifest.post_blocks
        if block.block_ref in manifest.affected_post_block_refs
    }
    stable_text_pairs = {
        (
            next(
                block.span.text
                for block in manifest.pre_blocks
                if block.block_ref == pair.pre_block_ref
            ),
            next(
                block.span.text
                for block in manifest.post_blocks
                if block.block_ref == pair.post_block_ref
            ),
        )
        for pair in manifest.stable_block_pairs
    }
    assert affected_pre_text == {"删除这一段。"}
    assert affected_post_text == set()
    assert ("后续事实保持原样。", "后续事实保持原样。") in stable_text_pairs


def test_heading_change_expands_scope_when_section_path_changes():
    original = "# 旧章节\n\n章节内事实。\n\n## 子节\n\n子节事实。"
    edit = _edit(original, "# 旧章节", "# 新章节")
    proposed = original.replace("# 旧章节", "# 新章节")

    manifest = build_editorial_change_manifest(
        original,
        proposed,
        edits=(edit,),
    )

    affected_pre_text = {
        block.span.text
        for block in manifest.pre_blocks
        if block.block_ref in manifest.affected_pre_block_refs
    }
    affected_post_text = {
        block.span.text
        for block in manifest.post_blocks
        if block.block_ref in manifest.affected_post_block_refs
    }
    assert affected_pre_text == {
        "# 旧章节",
        "章节内事实。",
        "## 子节",
        "子节事实。",
    }
    assert affected_post_text == {
        "# 新章节",
        "章节内事实。",
        "## 子节",
        "子节事实。",
    }


def test_context_span_dependency_closes_over_otherwise_stable_block():
    original = "# 报告\n\n旧背景。\n\n它导致了结果。"
    edit = _edit(original, "旧背景。", "新背景。")
    proposed = original.replace("旧背景。", "新背景。")
    manifest = build_editorial_change_manifest(
        original,
        proposed,
        edits=(edit,),
    )
    assert {
        block.span.text
        for block in manifest.pre_blocks
        if block.block_ref in manifest.affected_pre_block_refs
    } == {"旧背景。"}

    scope = build_editorial_affected_scope(
        manifest,
        original_draft=original,
        proposed_draft=proposed,
        pre_units=(
            _unit(
                original,
                "它导致了结果。",
                context=("旧背景。",),
                locator="claim-before",
            ),
        ),
        post_units=(
            _unit(
                proposed,
                "它导致了结果。",
                context=("新背景。",),
                locator="claim-after",
            ),
        ),
    )

    affected_pre_text = {
        block.span.text
        for block in manifest.pre_blocks
        if block.block_ref in scope.affected_pre_block_refs
    }
    affected_post_text = {
        block.span.text
        for block in manifest.post_blocks
        if block.block_ref in scope.affected_post_block_refs
    }
    assert affected_pre_text == {"旧背景。", "它导致了结果。"}
    assert affected_post_text == {"新背景。", "它导致了结果。"}
    assert len(scope.affected_pre_units) == 1
    assert len(scope.affected_post_units) == 1
    assert scope.closure_rounds >= 2


def test_closed_positive_reviewer_result_is_the_only_commit_shape():
    manifest, scope, request, target_ref, post_ref = _review_fixture()
    raw = {
        "manifest_sha256": manifest.manifest_sha256,
        "targets": [
            {
                "pre_ref": target_ref,
                "outcome": "resolved",
                "post_refs": [post_ref],
                "rationale": "The weaker replacement is supported.",
            }
        ],
        "preserved": [],
        "post_units": [
            {
                "post_ref": post_ref,
                "lineage": "derived_from_pre",
                "pre_refs": [target_ref],
                "assessment": "acceptable",
                "rationale": "It is the supported narrowing of the target.",
            }
        ],
        "answer_preservation": "narrowed_with_evidence",
        "answer_preservation_rationale": "The answer remains useful.",
    }

    result = parse_transaction_review(json.dumps(raw), request=request)

    assert result.execution_status is TransactionExecutionStatus.COMPLETE
    assert result.decision is TransactionDecision.ACCEPT
    assert result.may_commit is True
    prompt = build_transaction_reviewer_prompt(
        request,
        affected_scope=scope,
        audit_payload={"pre": "audited", "post": "re-audited"},
    )
    assert manifest.manifest_sha256 in prompt
    assert target_ref in prompt
    assert post_ref in prompt
    assert "claim-91" not in prompt
    assert "claim-01" not in prompt


def test_missing_reviewer_denominator_fails_closed_to_rollback():
    manifest, _scope, request, target_ref, _post_ref = _review_fixture()
    raw = {
        "manifest_sha256": manifest.manifest_sha256,
        "targets": [
            {
                "pre_ref": target_ref,
                "outcome": "resolved",
                "post_refs": [],
                "rationale": "Incomplete response.",
            }
        ],
        "preserved": [],
        "post_units": [],
        "answer_preservation": "preserved",
        "answer_preservation_rationale": "Incomplete response.",
    }

    result = parse_transaction_review(raw, request=request)

    assert result.execution_status is TransactionExecutionStatus.FAILED
    assert result.decision is TransactionDecision.ROLLBACK
    assert result.may_commit is False
    assert set(result.unreviewed_refs) == {
        target.pre_ref for target in request.targets
    } | set(request.post_refs)


def test_review_request_cannot_silently_omit_affected_neighbour():
    original = "# 报告\n\n待修事实。 同段保留事实。"
    proposed = original.replace("待修事实。", "限定后的事实。")
    manifest = build_editorial_change_manifest(
        original,
        proposed,
        edits=(_edit(original, "待修事实。", "限定后的事实。"),),
    )
    scope = build_editorial_affected_scope(
        manifest,
        original_draft=original,
        proposed_draft=proposed,
        pre_units=(
            _unit(original, "待修事实。"),
            _unit(original, "同段保留事实。"),
        ),
        post_units=(
            _unit(proposed, "限定后的事实。"),
            _unit(proposed, "同段保留事实。"),
        ),
    )
    target_ref = next(
        unit.unit_ref
        for unit in scope.affected_pre_units
        if unit.semantic_text == "待修事实。"
    )
    request = build_transaction_review_request(
        scope,
        target_intents={target_ref: EditorialIntent.QUALIFY},
    )

    assert set(request.preserved_pre_refs) == {
        unit.unit_ref
        for unit in scope.affected_pre_units
        if unit.unit_ref != target_ref
    }
    assert set(request.post_refs) == {
        unit.unit_ref for unit in scope.affected_post_units
    }

    omitted = TransactionReviewRequest(
        manifest_sha256=request.manifest_sha256,
        targets=request.targets,
        post_refs=request.post_refs,
    )
    with pytest.raises(ValueError, match="every affected pre-edit unit"):
        build_transaction_reviewer_prompt(
            omitted,
            affected_scope=scope,
            audit_payload={},
        )


def test_complete_negative_semantic_review_is_a_completed_rollback():
    manifest, _scope, request, target_ref, post_ref = _review_fixture()
    raw = {
        "manifest_sha256": manifest.manifest_sha256,
        "targets": [
            {
                "pre_ref": target_ref,
                "outcome": "not_resolved",
                "post_refs": [post_ref],
                "rationale": "The same unsupported assertion remains.",
            }
        ],
        "preserved": [],
        "post_units": [
            {
                "post_ref": post_ref,
                "lineage": "derived_from_pre",
                "pre_refs": [target_ref],
                "assessment": "degraded",
                "rationale": "The replacement remains unsupported.",
            }
        ],
        "answer_preservation": "uncertain",
        "answer_preservation_rationale": "The mutation is not established safe.",
    }

    result = parse_transaction_review(raw, request=request)

    assert result.execution_status is TransactionExecutionStatus.COMPLETE
    assert result.decision is TransactionDecision.ROLLBACK
    assert result.unreviewed_refs == ()
    assert result.may_commit is False


def test_incomplete_result_contract_cannot_claim_accept():
    with pytest.raises(ValidationError):
        EditorialTransactionResult(
            manifest_sha256="0" * 64,
            execution_status=TransactionExecutionStatus.FAILED,
            decision=TransactionDecision.ACCEPT,
            target_refs=("pre-a",),
            preserved_pre_refs=(),
            post_refs=(),
            unreviewed_refs=("pre-a",),
            diagnostics=("provider failure",),
        )


def test_review_request_rejects_overlapping_target_and_preservation_scope():
    with pytest.raises(ValidationError):
        TransactionReviewRequest(
            manifest_sha256="1" * 64,
            targets=(
                TransactionTargetRequirement(
                    pre_ref="pre-a",
                    intended_action=EditorialIntent.REMOVE,
                ),
            ),
            preserved_pre_refs=("pre-a",),
        )


def test_answer_preservation_enum_remains_semantic_not_a_numeric_score():
    assert AnswerPreservation.PRESERVED.value == "preserved"
    assert not hasattr(AnswerPreservation, "score")
