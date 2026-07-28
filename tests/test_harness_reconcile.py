from __future__ import annotations

import asyncio
import json
from pathlib import Path

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
from open_deep_research.harness.reconcile import (
    ChecklistCoverageDisposition,
    CoverageAssessmentStatus,
    reconcile_checklist_report,
)
from open_deep_research.harness.render import render_verified_report
from open_deep_research.harness.verify import VerificationResult

_HOLDOUT_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "harness_checklist_report_236eb115.json"
)


def _checklist() -> ResearchChecklist:
    return ResearchChecklist(
        topic="A topic",
        items=(
            ChecklistItem(
                item_id="what-1",
                dimension=ChecklistDimension.WHAT,
                question="What happened?",
                priority=1,
                corroboration_target=1,
            ),
            ChecklistItem(
                item_id="how-1",
                dimension=ChecklistDimension.HOW,
                question="How did it happen?",
                priority=2,
                corroboration_target=1,
            ),
            ChecklistItem(
                item_id="where-1",
                dimension=ChecklistDimension.WHERE,
                question="Where did it happen?",
                priority=3,
                corroboration_target=1,
            ),
        ),
    )


def _claim(
    draft: str,
    *,
    claim_id: str,
    block_id: str,
    anchor: str,
) -> AtomicClaim:
    start = draft.index(anchor)
    return AtomicClaim(
        claim_id=claim_id,
        block_id=block_id,
        selected_text=anchor,
        claim_text=anchor,
        anchor_text=anchor,
        start_char=start,
        end_char=start + len(anchor),
        citation_requirement=CitationRequirement.EXTERNAL,
        source_resolution=SourceResolution.UNRESOLVED,
        normalization_status=ClaimNormalizationStatus.LOCATED,
    )


class ScriptedReconciliationModel:
    def __init__(self, content):
        self.content = content
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return {
            "content": json.dumps(self.content),
            "token_count": 17,
            "cost_usd": 0.04,
        }


def test_semantic_dispositions_require_code_located_report_references() -> None:
    draft = "# Report\n\nAlpha answers the first item. Beta touches the second."
    blocks = parse_markdown_blocks(draft)
    paragraph = blocks[1]
    claims = (
        _claim(
            draft,
            claim_id="claim-1",
            block_id=paragraph.block_id,
            anchor="Alpha answers the first item.",
        ),
        _claim(
            draft,
            claim_id="claim-2",
            block_id=paragraph.block_id,
            anchor="Beta touches the second.",
        ),
    )
    model = ScriptedReconciliationModel(
        {
            "items": [
                {
                    "item_id": "what-1",
                    "disposition": "covered",
                    "reason": "The first claim answers it directly.",
                    "claim_ids": ["claim-1"],
                },
                {
                    "item_id": "how-1",
                    "disposition": "partially_covered",
                    "reason": "The second claim only touches one aspect.",
                    "claim_ids": ["claim-2"],
                },
                {
                    "item_id": "where-1",
                    "disposition": "not_covered",
                    "reason": "The report gives no location.",
                    "claim_ids": [],
                },
            ]
        }
    )

    result = asyncio.run(
        reconcile_checklist_report(
            draft,
            _checklist(),
            blocks=blocks,
            claims=claims,
            model_client=model,
        )
    )

    by_item = {record.item_id: record for record in result.records}
    assert by_item["what-1"].disposition == (
        ChecklistCoverageDisposition.COVERED
    )
    reference = by_item["what-1"].references[0]
    assert reference.claim_id == "claim-1"
    assert reference.block_id == paragraph.block_id
    assert draft[reference.start_char : reference.end_char] == (
        reference.anchor_text
    )
    assert by_item["how-1"].disposition == (
        ChecklistCoverageDisposition.PARTIALLY_COVERED
    )
    assert by_item["where-1"].disposition == (
        ChecklistCoverageDisposition.NOT_COVERED
    )
    assert by_item["where-1"].rationale == "The report gives no location."
    assert result.summary.model_dump() == {
        "assessed_items": 3,
        "assessment_failed_item_ids": (),
        "assessment_failed_items": 0,
        "covered_items": 1,
        "covered_rate": 1 / 3,
        "not_covered_item_ids": ("where-1",),
        "not_covered_items": 1,
        "partially_covered_item_ids": ("how-1",),
        "partially_covered_items": 1,
        "total_items": 3,
    }
    assert result.total_tokens == 17
    assert result.total_cost_usd == 0.04
    assert result.affects_report_content is False
    assert result.blocks_artifact_write is False
    assert '"item_id":"what-1"' in model.prompts[0]
    assert '"claim_id":"claim-1"' in model.prompts[0]
    assert "do not rewrite the report or checklist" in model.prompts[0]

    rendered = render_verified_report(
        draft,
        VerificationResult(claims=()),
        checklist_coverage=result.summary,
    )
    assert rendered.checklist_coverage_line == (
        "> 清单内容覆盖（不表示来源支持）："
        "已评估 3/3；完整覆盖 1/3（33.3%）；"
        "部分覆盖 1；未覆盖 1（where-1）；对账失败 0（无）。"
    )
    assert rendered.markdown.split("\n\n", maxsplit=1)[1] == draft


def test_unlocated_coverage_and_omissions_are_assessment_failures() -> None:
    draft = "# Report\n\nOnly one statement."
    blocks = parse_markdown_blocks(draft)
    model = ScriptedReconciliationModel(
        {
            "items": [
                {
                    "item_id": "what-1",
                    "disposition": "covered",
                    "reason": "The report covers it.",
                    "claim_ids": ["invented-claim"],
                },
                {
                    "item_id": "where-1",
                    "disposition": "not_covered",
                    "reason": "",
                    "claim_ids": [],
                },
                {
                    "item_id": "unknown-item",
                    "disposition": "not_covered",
                    "reason": "Unknown.",
                    "claim_ids": [],
                },
            ]
        }
    )

    result = asyncio.run(
        reconcile_checklist_report(
            draft,
            _checklist(),
            blocks=blocks,
            claims=(),
            model_client=model,
        )
    )

    by_item = {record.item_id: record for record in result.records}
    rejected = by_item["what-1"]
    assert rejected.proposed_disposition == (
        ChecklistCoverageDisposition.COVERED
    )
    assert rejected.disposition is None
    assert rejected.invalid_claim_ids == ("invented-claim",)
    assert rejected.assessment_status == (
        CoverageAssessmentStatus.ASSESSMENT_FAILED
    )
    assert "covered_without_valid_report_reference" in rejected.diagnostics

    assert by_item["how-1"].disposition is None
    assert by_item["how-1"].diagnostics == ("checklist_item_omitted",)
    assert by_item["where-1"].disposition is None
    assert any(
        diagnostic.startswith("entry_invalid[")
        for diagnostic in by_item["where-1"].diagnostics
    )
    assert "checklist_item_omitted" in by_item["where-1"].diagnostics
    assert result.summary.not_covered_items == 0
    assert result.summary.assessment_failed_items == 3
    assert result.summary.assessment_failed_item_ids == (
        "what-1",
        "how-1",
        "where-1",
    )
    assert any(
        diagnostic == "unknown_item_id[2]: unknown-item"
        for diagnostic in result.diagnostics
    )


def test_malformed_output_usage_is_retained_and_not_called_not_covered() -> None:
    class MalformedModel:
        async def generate(self, prompt):
            return {
                "content": "not json",
                "token_count": 23,
                "cost_usd": 0.06,
            }

    result = asyncio.run(
        reconcile_checklist_report(
            "# Report",
            _checklist(),
            blocks=parse_markdown_blocks("# Report"),
            claims=(),
            model_client=MalformedModel(),
        )
    )

    assert result.total_tokens == 23
    assert result.total_cost_usd == 0.06
    assert result.summary.not_covered_items == 0
    assert result.summary.assessment_failed_items == 3
    assert all(record.disposition is None for record in result.records)
    assert result.diagnostics[0].startswith(
        "reconciliation_output_invalid:"
    )


def test_reconciliation_model_failure_is_audited_without_blocking() -> None:
    class FailingModel:
        async def generate(self, prompt):
            raise RuntimeError("provider unavailable")

    result = asyncio.run(
        reconcile_checklist_report(
            "# Report",
            _checklist(),
            blocks=parse_markdown_blocks("# Report"),
            claims=(),
            model_client=FailingModel(),
        )
    )

    assert result.summary.assessed_items == 0
    assert result.summary.assessment_failed_items == 3
    assert result.summary.not_covered_items == 0
    assert result.blocks_artifact_write is False
    assert result.diagnostics[0].startswith("reconciliation_model_error:")


def test_real_holdout_fixture_keeps_adjacent_mentions_not_covered() -> None:
    fixture = json.loads(_HOLDOUT_FIXTURE.read_text(encoding="utf-8"))
    draft = fixture["canonical_draft"]
    items = tuple(
        ChecklistItem(
            item_id=item["item_id"],
            dimension=ChecklistDimension(item["dimension"]),
            question=item["question"],
            priority=index,
            corroboration_target=1,
        )
        for index, item in enumerate(fixture["checklist"], start=1)
    )
    checklist = ResearchChecklist(topic="Holdout", items=items)
    expected = tuple(fixture["expected_not_covered_item_ids"])
    model = ScriptedReconciliationModel(
        {
            "items": [
                {
                    "item_id": item_id,
                    "disposition": "not_covered",
                    "reason": (
                        "The report contains adjacent subject matter but does "
                        "not answer this checklist question."
                    ),
                    "claim_ids": [],
                }
                for item_id in expected
            ]
        }
    )

    result = asyncio.run(
        reconcile_checklist_report(
            draft,
            checklist,
            blocks=parse_markdown_blocks(draft),
            claims=(),
            model_client=model,
        )
    )

    # These words really occur in the frozen model-authored report. Their
    # presence must not mechanically upgrade adjacent prose into an answer.
    assert "shelter" in draft.casefold()
    assert "reconstruction" in draft.casefold()
    assert "disaster preparedness" in draft.casefold()
    assert result.summary.not_covered_item_ids == expected
    assert result.summary.covered_items == 0
    assert result.summary.assessment_failed_items == 0
