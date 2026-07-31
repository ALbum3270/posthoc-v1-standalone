import asyncio
import hashlib
import json
import os
import re
from types import SimpleNamespace

import pytest

import run_harness as harness_cli
from open_deep_research.harness.claims import parse_markdown_blocks
from open_deep_research.harness.loop import LoopBudget, StopReason
from open_deep_research.harness.runner import (
    _publish_artifact_bundle,
    run_harness,
)
from open_deep_research.harness.verify import ClaimEvidenceState


class ChecklistModel:
    last_usage = {"token_count": 3, "cost_usd": 0.03}

    def __init__(self, events):
        self.events = events

    async def generate(self, prompt):
        self.events.append("checklist")
        return {
            "items": [
                {
                    "item_id": "what-1",
                    "dimension": "what",
                    "question": "What happened?",
                    "priority": 1,
                    "required_source_count": 1,
                }
            ]
        }


class DecisionModel:
    def __init__(self, events):
        self.events = events

    async def generate(self, prompt):
        self.events.append("decision")
        return {
            "content": {"action": "settle", "item_id": "what-1"},
            "token_count": 2,
            "cost_usd": 0.02,
        }


class UnusedNoteModel:
    async def generate(self, prompt):
        raise AssertionError("note model should not be called")


class WriteModel:
    def __init__(self, events):
        self.events = events
        self.prompts = []

    async def generate(self, prompt):
        self.events.append("write")
        self.prompts.append(prompt)
        return {
            "content": "# Report\n\nThe model wrote this report.",
            "token_count": 5,
            "cost_usd": 0.05,
        }


class ClaimModel:
    def __init__(self, events, draft):
        self.events = events
        self.draft = draft
        self.blocks = parse_markdown_blocks(draft)
        self.call_number = 0

    async def generate(self, prompt):
        self.call_number += 1
        self.events.append(f"claim-{self.call_number}")
        paragraph = self.blocks[1]
        if self.call_number == 1:
            content = {
                "blocks": [
                    {
                        "block_id": self.blocks[0].block_id,
                        "disposition": "no_verifiable_claims",
                        "rationale": "heading",
                        "assertions": [],
                    },
                    {
                        "block_id": paragraph.block_id,
                        "disposition": "claims_selected",
                        "rationale": "one external assertion",
                        "assertions": [
                            {
                                "selected_text": paragraph.text,
                                "citation_requirement": "external",
                            }
                        ],
                    },
                ]
            }
        elif self.call_number == 2:
            content = {
                "claims": [
                    {
                        "claim_id": "claim-0001",
                        "claim_text": paragraph.text,
                        "context_spans": [],
                    }
                ]
            }
        elif self.call_number == 3:
            content = {
                "claims": [
                    {
                        "claim_id": "claim-0001",
                        "start_segment_id": "S000002",
                        "end_segment_id": "S000002",
                    }
                ]
            }
        else:
            content = {
                "claims": [
                    {
                        "claim_id": "claim-0001",
                        "status": "not_underspecified",
                        "categories": [],
                        "reason": "The assertion has explicit boundaries.",
                    }
                ]
            }
        return {
            "content": json.dumps(content),
            "token_count": 10,
            "cost_usd": 0.01,
        }


class CoverageModel:
    def __init__(self, events):
        self.events = events

    async def generate(self, prompt):
        self.events.append("reconciliation")
        return {
            "content": json.dumps(
                {
                    "items": [
                        {
                            "item_id": "what-1",
                            "disposition": "covered",
                            "reason": "The report answers the item.",
                            "claim_ids": ["claim-0001"],
                        }
                    ]
                }
            ),
            "token_count": 4,
            "cost_usd": 0.01,
        }


class AttributionModel:
    def __init__(self, events):
        self.events = events

    async def generate(self, prompt):
        self.events.append("attribution")
        return {
            "content": json.dumps(
                {
                    "action": "attribute",
                    "claims": [
                        {"claim_id": "claim-0001", "candidates": []}
                    ],
                }
            ),
            "token_count": 7,
            "cost_usd": 0.02,
        }


class UnusedVerificationModel:
    async def generate(self, prompt):
        raise AssertionError("no-candidate claim must not call verifier")


class ReadThenSettleDecisionModel:
    def __init__(self, events, url):
        self.events = events
        self.url = url
        self.call_number = 0

    async def generate(self, prompt):
        self.call_number += 1
        self.events.append(f"decision-{self.call_number}")
        content = (
            {"action": "read", "item_id": "what-1", "url": self.url}
            if self.call_number == 1
            else {"action": "settle", "item_id": "what-1"}
        )
        return {"content": content, "token_count": 2, "cost_usd": 0.01}


class OneNoteModel:
    def __init__(self, events):
        self.events = events

    async def generate(self, prompt):
        self.events.append("note")
        return {
            "content": {
                "active_notes": [
                    {
                        "item_id": "what-1",
                        "finding": "A finding used after drafting.",
                        "start_segment_id": "S000001",
                        "end_segment_id": "S000001",
                    }
                ],
                "cross_item_seeds": [],
            },
            "token_count": 3,
            "cost_usd": 0.01,
        }


class ReadingTavily:
    def __init__(self, url):
        self.url = url

    async def search(self, query, **kwargs):
        raise AssertionError("search should not be called")

    async def extract(self, urls, **kwargs):
        assert urls == [self.url]
        return {
            "results": [
                {
                    "url": self.url,
                    "raw_content": "ExactSourceEvidence 2026.",
                }
            ]
        }


class EvidenceAttributionModel:
    def __init__(self, events, url):
        self.events = events
        self.url = url

    async def generate(self, prompt):
        self.events.append("attribution")
        note_ref = re.search(r"nref-[0-9a-f]{16}", prompt)
        assert note_ref is not None
        return {
            "content": json.dumps(
                {
                    "action": "attribute",
                    "claims": [
                        {
                            "claim_id": "claim-0001",
                            "candidates": [
                                {
                                    "note_ref": note_ref.group(0),
                                    "inherited_from_claim_id": None,
                                }
                            ],
                        }
                    ],
                }
            ),
            "token_count": 7,
            "cost_usd": 0.02,
        }


class EvidenceVerificationModel:
    def __init__(self, events):
        self.events = events

    async def generate(self, prompt):
        self.events.append("verification")
        return {
            "content": json.dumps(
                {
                    "results": [
                        {
                            "claim_id": "claim-0001",
                            "verdict": "supports",
                            "start_segment_id": "S000001",
                            "end_segment_id": "S000001",
                            "explanation": "The full source supports it.",
                        }
                    ]
                }
            ),
            "token_count": 9,
            "cost_usd": 0.03,
        }


class AuditEditWriteModel:
    async def generate(self, prompt):
        return {
            "content": "# Report\n\nThe model added an unsupported detail.",
            "token_count": 5,
            "cost_usd": 0.01,
        }


class ReauditClaimModel:
    """Decomposes the first and edited drafts without reusing old anchors."""

    def __init__(self):
        self.call_number = 0

    async def generate(self, prompt):
        self.call_number += 1
        edited = "A narrower supported fact." in prompt
        draft = (
            "# Report\n\nA narrower supported fact."
            if edited
            else "# Report\n\nThe model added an unsupported detail."
        )
        blocks = parse_markdown_blocks(draft)
        paragraph = blocks[1]
        phase = self.call_number if self.call_number <= 4 else self.call_number - 4
        if phase == 1:
            content = {
                "blocks": [
                    {
                        "block_id": blocks[0].block_id,
                        "assertions": [],
                        "rationale": "heading",
                    },
                    {
                        "block_id": paragraph.block_id,
                        "assertions": [
                            {
                                "selected_text": paragraph.text,
                                "citation_requirement": "external",
                            }
                        ],
                        "rationale": "external assertion",
                    },
                ]
            }
        elif phase == 2:
            content = {
                "claims": [
                    {
                        "claim_id": "claim-0001",
                        "claim_text": paragraph.text,
                        "context_spans": [],
                    }
                ]
            }
        elif phase == 3:
            content = {
                "claims": [
                    {
                        "claim_id": "claim-0001",
                        "start_segment_id": "S000002",
                        "end_segment_id": "S000002",
                    }
                ]
            }
        else:
            content = {
                "claims": [
                    {
                        "claim_id": "claim-0001",
                        "status": "not_underspecified",
                        "categories": [],
                        "reason": "Bounded assertion.",
                    }
                ]
            }
        return {
            "content": json.dumps(content),
            "token_count": 5,
            "cost_usd": 0.01,
        }


class AuditEditVerificationModel:
    def __init__(self):
        self.calls = 0

    async def generate(self, prompt):
        self.calls += 1
        result = {
            "claim_id": "claim-0001",
            "verdict": (
                "does_not_support" if self.calls == 1 else "supports"
            ),
            "start_segment_id": None if self.calls == 1 else "S000001",
            "end_segment_id": None if self.calls == 1 else "S000001",
            "explanation": (
                "The source does not contain the added detail."
                if self.calls == 1
                else "The source directly states the narrower fact."
            ),
        }
        return {
            "content": json.dumps({"results": [result]}),
            "token_count": 8,
            "cost_usd": 0.01,
        }


class AuditEditorModel:
    def __init__(self):
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return {
            "content": json.dumps(
                {
                    "blocks": [
                        {
                            "block_id": "block-0002",
                            "replacement_text": "A narrower supported fact.",
                            "decisions": [
                                {
                                    "claim_id": "claim-0001",
                                    "action": "qualify",
                                    "reason": (
                                        "Replace the unsupported detail with "
                                        "the narrower fact in the source."
                                    ),
                                }
                            ],
                        }
                    ]
                }
            ),
            "token_count": 6,
            "cost_usd": 0.01,
        }


class UnusedTavily:
    async def search(self, query, **kwargs):
        raise AssertionError("search should not be called")

    async def extract(self, urls, **kwargs):
        raise AssertionError("read should not be called")


def test_runner_executes_pipeline_and_writes_report_and_complete_audit(tmp_path):
    events = []
    writer = WriteModel(events)
    draft = "# Report\n\nThe model wrote this report."

    result = asyncio.run(
        run_harness(
            "A topic",
            checklist_model=ChecklistModel(events),
            decision_model=DecisionModel(events),
            note_model=UnusedNoteModel(),
            write_model=writer,
            claim_model=ClaimModel(events, draft),
            reconciliation_model=CoverageModel(events),
            attribution_model=AttributionModel(events),
            verification_model=UnusedVerificationModel(),
            tavily_client=UnusedTavily(),
            budget=LoopBudget(max_rounds=2, max_tokens=100, max_cost_usd=1),
            output_dir=tmp_path,
            run_id="fixed-run",
            model_names={
                "decision": "cheap-decision",
                "note": "cheap-note",
                "reconciliation": "coverage-model",
                "verification": "strong-verifier",
            },
        )
    )

    assert events == [
        "checklist",
        "decision",
        "write",
        "claim-1",
        "claim-2",
        "claim-3",
        "claim-4",
        "reconciliation",
        "attribution",
    ]
    assert result.loop_result.stop_reason is StopReason.ALL_ITEMS_TERMINAL
    assert result.report_path == tmp_path / "fixed-run" / "report.md"
    assert result.sources_path == tmp_path / "fixed-run" / "sources.md"
    assert result.audit_path == tmp_path / "fixed-run" / "audit.json"
    final_markdown = result.report_path.read_text(encoding="utf-8")
    sources_markdown = result.sources_path.read_text(encoding="utf-8")
    assert final_markdown == result.rendered_report.markdown
    assert final_markdown.startswith("> 证据包：")
    assert (
        "缺失逐字证据、提交标记或摘要不符则证据包不完整"
        in final_markdown.splitlines()[0]
    )
    assert "正文块评估 2/2" in final_markdown
    assert (
        "证据状态：本报告没有任何可定位的正式支持关系"
        in final_markdown
    )
    assert "初次采集阶段未取得任何原文" in final_markdown
    assert "Run ID：`fixed-run`" in sources_markdown
    assert "[report.md](report.md)" in sources_markdown
    assert (
        "> 域名代理集中度：没有正式 claim–source 支持关系；"
        "域名仅作发布方代理。"
    ) in final_markdown
    assert (
        "> 清单内容覆盖（不表示来源支持）："
        "已评估 1/1；完整覆盖 1/1"
    ) in final_markdown
    assert (
        "The model wrote this report.〔未找到候选来源〕"
        in final_markdown
    )
    assert "- Status: settled" in writer.prompts[0]

    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["canonical_draft"] == (
        "# Report\n\nThe model wrote this report."
    )
    assert audit["canonical_draft"] == result.report.canonical_draft
    assert result.report_path.read_text(encoding="utf-8") != (
        audit["canonical_draft"]
    )
    assert audit["ledger"]["research_id"] == "fixed-run"
    assert audit["ledger"]["rounds"][0]["action"] == "settle"
    assert audit["checklist"]["items"][0]["status"] == "settled"
    diagnostic = audit["stop"].pop("diagnostic")
    assert audit["stop"] == {
        "detail": (
            "all checklist items reached a terminal state; "
            "settled_without_located_evidence=1 (what-1)"
        ),
        "is_success": True,
        "open_item_ids": [],
        "reason": "all_items_terminal",
    }
    # A completed run reports no resource stop and no binding ceiling, and it
    # makes no budget recommendation, because nothing was withheld for cost.
    assert diagnostic["resource_stop_reason"] == "not_resource_limited"
    assert diagnostic["completion_status"] == "complete"
    assert diagnostic["cap_was_binding"] is False
    assert diagnostic["blocked_operation_quality"] == "nothing_blocked"
    assert diagnostic["budget_decision_signal"] == "not_applicable"
    assert diagnostic["cost_objective_usd"] is None
    assert diagnostic["cost_objective_exceeded"] is False
    assert audit["budget_decision_signal"] == "not_applicable"
    assert audit["collection_summary"] == {
        "initial_collection_snapshot": {
            "cached_source_count": 0,
            "note_count": 0,
            "usable_note_count": 0,
        },
        "known_gaps": ["writing_input_budget_preflight_not_enforced"],
        "quote_quality": {
            "format_repair_rate": 0.0,
            "noncontiguous_composite_count": 0,
            "noncontiguous_composite_rate": 0.0,
            "note_count": 0,
            "repaired_locatable_count": 0,
            "strict_locatable_count": 0,
            "strict_locatable_rate": 0.0,
            "usable_source_span_count": 0,
            "usable_source_span_rate": 0.0,
        },
        "settled_without_located_evidence": 1,
        "settled_without_located_evidence_item_ids": ["what-1"],
        "rejected_exhausted_without_collection_attempt": 0,
        "rejected_exhausted_without_collection_attempt_item_ids": [],
        "accepted_exhausted_without_collection_attempt": 0,
        "accepted_exhausted_without_collection_attempt_item_ids": [],
        "accepted_exhausted_attempt_unknown_legacy": 0,
        "accepted_exhausted_attempt_unknown_legacy_item_ids": [],
        "exhausted_with_unread_candidates": 0,
        "exhausted_with_unread_candidates_item_ids": [],
        "writing_reserve": {"cost_usd": 0.0, "tokens": 0},
    }
    assert audit["usage"] == {
        "checklist": {"cost_usd": 0.03, "token_count": 3},
        "collection": {"cost_usd": 0.02, "token_count": 2},
            "decomposition_attribution": {
                "cost_usd": 0.06,
                "token_count": 47,
            },
            "disagreement": {"cost_usd": 0.0, "token_count": 0},
            "evidence_gap": {"cost_usd": 0.0, "token_count": 0},
        "reconciliation": {"cost_usd": 0.01, "token_count": 4},
        "total": {"cost_usd": 0.17, "token_count": 61},
        "verification": {"cost_usd": 0.0, "token_count": 0},
        "writing": {"cost_usd": 0.05, "token_count": 5},
    }
    assert audit["run_cost_limit_status"] == "no_run_level_cost_limit"
    assert audit["run_cost_budget"]["configured"] is False
    assert audit["run_cost_budget"]["max_cost_usd"] is None
    assert audit["run_cost_budget"]["enforcement"] == (
        "no_run_level_cost_limit"
    )
    assert audit["run_cost_budget"]["observed_total_cost_usd"] == 0.17
    assert result.run_cost_budget.model_dump(mode="json") == (
        audit["run_cost_budget"]
    )
    assert audit["posthoc_evidence"]["verification"]["claims"][0][
        "state"
    ] == "no_candidate_source"
    assert audit["posthoc_evidence"]["verification"]["claims"][0][
        "corroboration_target"
    ] == 2
    assert (
        "required_independent_sources"
        not in audit["posthoc_evidence"]["verification"]["claims"][0]
    )
    assert audit["posthoc_evidence"]["claim_decomposition"][
        "registry_coverage"
    ] == {
        "evaluated_blocks": 2,
        "is_complete": True,
        "total_blocks": 2,
        "unassessed_block_ids": [],
        "unassessed_blocks": 0,
    }
    assert audit["posthoc_evidence"]["claim_decomposition"][
        "anchor_copied_from_selection_rate"
    ] == 0.0
    diagnostic = audit["posthoc_evidence"][
        "evaluative_claim_diagnostics"
    ]
    assert diagnostic["external_denominator_before"] == 1
    assert diagnostic["external_denominator_after"] == 1
    assert diagnostic["claim_registry_unchanged"] is True
    assert diagnostic["citation_requirements_unchanged"] is True
    assert diagnostic["diagnostic_is_non_gating"] is True
    assert diagnostic["assessments"][0]["status"] == "not_underspecified"
    assert audit["posthoc_evidence"]["checklist_report_reconciliation"][
        "summary"
    ] == {
        "assessed_items": 1,
        "assessment_failed_item_ids": [],
        "assessment_failed_items": 0,
        "covered_items": 1,
        "covered_rate": 1.0,
        "not_covered_item_ids": [],
        "not_covered_items": 0,
        "partially_covered_item_ids": [],
        "partially_covered_items": 0,
        "total_items": 1,
    }
    assert audit["posthoc_evidence"]["checklist_report_reconciliation"][
        "affects_report_content"
    ] is False
    assert audit["posthoc_evidence"]["checklist_report_reconciliation"][
        "blocks_artifact_write"
    ] is False
    assert result.verification.claims[0].state == (
        ClaimEvidenceState.NO_CANDIDATE_SOURCE
    )
    assert audit["posthoc_evidence"][
        "corroboration_target_for_external_claims"
    ] == 2
    assert (
        "required_independent_sources_for_external_claims"
        not in audit["posthoc_evidence"]
    )
    assert "markdown" not in audit["posthoc_evidence"]["rendering"]
    assert "sources_markdown" not in audit["posthoc_evidence"]["rendering"]
    assert audit["posthoc_evidence"]["rendering"]["summary"][
        "settled_without_located_evidence"
    ] == 1
    assert audit["posthoc_evidence"]["domain_proxy_concentration"][
        "counting_unit"
    ] == "formal_claim_source_support_relation"
    assert audit["posthoc_evidence"]["domain_proxy_concentration"][
        "overall"
    ]["formal_support_relation_count"] == 0
    assert audit["posthoc_evidence"]["domain_proxy_concentration"][
        "is_organization_independence_determination"
    ] is False
    assert audit["posthoc_evidence"]["domain_proxy_concentration"][
        "is_viewpoint_diversity_determination"
    ] is False
    assert result.domain_proxy_concentration.model_dump(mode="json") == (
        audit["posthoc_evidence"]["domain_proxy_concentration"]
    )
    assert audit["models"]["verification"] == "strong-verifier"
    assert audit["models"]["reconciliation"] == "coverage-model"
    assert audit["artifacts"] == {
        "audit": "audit.json",
        "bundle_complete": True,
        "commit_marker": "audit.json",
        "directory": "fixed-run",
        "publication_order": ["directory"],
        "report": "report.md",
        "report_sha256": hashlib.sha256(
            final_markdown.encode("utf-8")
        ).hexdigest(),
        "sources": "sources.md",
        "sources_sha256": hashlib.sha256(
            sources_markdown.encode("utf-8")
        ).hexdigest(),
        "staging_write_order": ["sources", "report", "audit"],
    }


def test_runner_rejects_run_id_that_could_escape_output_directory(tmp_path):
    with pytest.raises(ValueError, match="run_id"):
        asyncio.run(
            run_harness(
                "A topic",
                checklist_model=ChecklistModel([]),
                decision_model=DecisionModel([]),
                note_model=UnusedNoteModel(),
                write_model=WriteModel([]),
                claim_model=ClaimModel(
                    [],
                    "# Report\n\nThe model wrote this report.",
                ),
                reconciliation_model=CoverageModel([]),
                attribution_model=AttributionModel([]),
                verification_model=UnusedVerificationModel(),
                tavily_client=UnusedTavily(),
                output_dir=tmp_path,
                run_id="../outside",
            )
        )


def test_runner_wires_verified_source_quote_into_code_owned_footnote(tmp_path):
    events = []
    url = "https://evidence.example/article"
    draft = "# Report\n\nThe model wrote this report."
    result = asyncio.run(
        run_harness(
            "A topic",
            checklist_model=ChecklistModel(events),
            decision_model=ReadThenSettleDecisionModel(events, url),
            note_model=OneNoteModel(events),
            write_model=WriteModel(events),
            claim_model=ClaimModel(events, draft),
            reconciliation_model=CoverageModel(events),
            attribution_model=EvidenceAttributionModel(events, url),
            verification_model=EvidenceVerificationModel(events),
            tavily_client=ReadingTavily(url),
            budget=LoopBudget(
                max_rounds=3,
                max_tokens=100,
                max_cost_usd=1,
            ),
            output_dir=tmp_path,
            run_id="verified-run",
        )
    )

    markdown = result.report_path.read_text(encoding="utf-8")
    sources_markdown = result.sources_path.read_text(encoding="utf-8")
    assert "The model wrote this report.[^1]" in markdown
    assert "〔单一发布方支持〕" not in markdown
    assert (
        "> 图例：带脚注且无额外状态标签 = "
        "单一发布方提供了可定位支持引文"
    ) in markdown
    assert markdown.count("[^1]:") == 1
    assert "ExactSourceEvidence 2026" not in markdown
    assert "ExactSourceEvidence 2026" in sources_markdown
    assert "exact source evidence 2026." not in markdown
    assert "exact source evidence 2026." not in sources_markdown
    assert (
        "[逐字证据]"
        "(sources.md#evidence-1)"
        in markdown
    )
    assert '<a id="evidence-1"></a>' in sources_markdown
    assert "source_id" not in sources_markdown
    assert "start_char" not in sources_markdown
    assert "end_char" not in sources_markdown
    assert result.verification.claims[0].relations[0].model_quote is None
    assert result.verification.claims[0].relations[0].source_quote == (
        "ExactSourceEvidence 2026."
    )
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["usage"]["verification"] == {
        "cost_usd": 0.03,
        "token_count": 9,
    }
    assert audit["posthoc_evidence"]["rendering"]["footnotes"][0][
        "source_quote"
    ] == "ExactSourceEvidence 2026."
    assert audit["artifacts"]["sources_sha256"] == hashlib.sha256(
        result.sources_path.read_bytes()
    ).hexdigest()


def test_runner_reaudits_changed_draft_before_committing_editorial_revision(
    tmp_path,
):
    """A finance-08-shaped unsupported claim cannot keep its old registry."""

    events = []
    url = "https://evidence.example/article"
    claim_model = ReauditClaimModel()
    verifier = AuditEditVerificationModel()
    editor = AuditEditorModel()

    result = asyncio.run(
        run_harness(
            "A topic",
            checklist_model=ChecklistModel(events),
            decision_model=ReadThenSettleDecisionModel(events, url),
            note_model=OneNoteModel(events),
            write_model=AuditEditWriteModel(),
            claim_model=claim_model,
            reconciliation_model=CoverageModel(events),
            attribution_model=EvidenceAttributionModel(events, url),
            verification_model=verifier,
            editor_model=editor,
            tavily_client=ReadingTavily(url),
            budget=LoopBudget(max_rounds=3, max_tokens=100, max_cost_usd=1),
            output_dir=tmp_path,
            run_id="audit-edit-run",
        )
    )

    assert result.publication_eligible is True
    assert result.editorial_revision is not None
    assert result.editorial_revision.committed_after_reaudit is True
    assert result.report.canonical_draft == (
        "# Report\n\nA narrower supported fact."
    )
    assert result.verification.claims[0].state is (
        ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER
    )
    assert verifier.calls == 2
    assert len(editor.prompts) == 1
    # Three claim calls plus one advisory call for the first draft, then a
    # completely new three-stage decomposition for the changed bytes.
    assert claim_model.call_number == 7
    assert "A narrower supported fact.[^1]" in result.rendered_report.markdown
    assert "unsupported detail" not in result.rendered_report.markdown

    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    posthoc = audit["posthoc_evidence"]
    assert audit["original_canonical_draft"] == (
        "# Report\n\nThe model added an unsupported detail."
    )
    assert audit["canonical_draft"] == (
        "# Report\n\nA narrower supported fact."
    )
    assert posthoc["pre_edit_evidence"]["verification"]["claims"][0][
        "state"
    ] == "cited_sources_do_not_support"
    assert posthoc["verification"]["claims"][0]["state"] == (
        "supported_single_publisher"
    )
    assert posthoc["editorial_revision"]["committed_after_reaudit"] is True
    stages = posthoc["stage_execution"]["stages"]
    for stage in (
        "audit_editing",
        "post_edit_claim_decomposition",
        "post_edit_attribution",
        "post_edit_initial_verification",
        "post_edit_checklist_reconciliation",
    ):
        assert stages[stage]["status"] == "complete", stage
    assert posthoc["stage_execution"]["mandatory_publication_stages"] == [
        "claim_decomposition",
        "attribution",
        "initial_verification",
        "checklist_reconciliation",
        "deterministic_rendering",
        "audit_editing",
        "post_edit_claim_decomposition",
        "post_edit_attribution",
        "post_edit_initial_verification",
        "post_edit_checklist_reconciliation",
    ]


def test_artifact_bundle_publishes_by_one_directory_rename_or_not_at_all(
    tmp_path,
    monkeypatch,
):
    run_directory = tmp_path / "run"
    report_path = run_directory / "report.md"
    sources_path = run_directory / "sources.md"
    audit_path = run_directory / "audit.json"
    calls = []

    def fail_publish(source, destination):
        calls.append((source, destination))
        raise OSError("simulated directory publish failure")

    monkeypatch.setattr(
        "open_deep_research.harness.runner.os.replace",
        fail_publish,
    )

    with pytest.raises(OSError, match="simulated directory publish failure"):
        _publish_artifact_bundle(
            destination=tmp_path,
            report_path=report_path,
            sources_path=sources_path,
            audit_path=audit_path,
            report_markdown="report",
            sources_markdown="sources",
            audit_json="audit",
        )

    assert len(calls) == 1
    assert calls[0][1] == run_directory
    assert not run_directory.exists()
    assert list(tmp_path.iterdir()) == []


def test_artifact_bundle_refuses_existing_directory_and_legacy_flat_run(
    tmp_path,
) -> None:
    run_directory = tmp_path / "run"
    report_path = run_directory / "report.md"
    sources_path = run_directory / "sources.md"
    audit_path = run_directory / "audit.json"
    run_directory.mkdir()

    with pytest.raises(FileExistsError, match="existing artifact bundle"):
        _publish_artifact_bundle(
            destination=tmp_path,
            report_path=report_path,
            sources_path=sources_path,
            audit_path=audit_path,
            report_markdown="new report",
            sources_markdown="new sources",
            audit_json="new audit",
        )
    assert list(run_directory.iterdir()) == []

    run_directory.rmdir()
    legacy_audit = tmp_path / "run.json"
    legacy_audit.write_text("historical", encoding="utf-8")
    with pytest.raises(FileExistsError, match="run.json"):
        _publish_artifact_bundle(
            destination=tmp_path,
            report_path=report_path,
            sources_path=sources_path,
            audit_path=audit_path,
            report_markdown="new report",
            sources_markdown="new sources",
            audit_json="new audit",
        )
    assert legacy_audit.read_text(encoding="utf-8") == "historical"
    assert not run_directory.exists()


def test_cli_configures_openrouter_proxy_without_touching_no_proxy(monkeypatch):
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.setenv("no_proxy", "leave-this-alone")

    harness_cli.configure_openrouter_proxy("https://openrouter.ai/api/v1")

    assert os.environ["https_proxy"] == "http://127.0.0.1:7890"
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7890"
    assert os.environ["no_proxy"] == "leave-this-alone"


def test_cli_separates_run_cost_limit_from_collection_subcap() -> None:
    args = harness_cli.build_parser().parse_args(
        [
            "A topic",
            "--max-cost-usd",
            "0.28",
            "--collection-max-cost-usd",
            "0.09",
            "--verification-cost-reserve-usd",
            "0.10",
        ]
    )

    assert args.max_cost_usd == 0.28
    assert args.collection_max_cost_usd == 0.09
    assert args.verification_cost_reserve_usd == 0.10
    # argparse rewraps help text to the terminal width, so compare on
    # whitespace-normalised text rather than pinning a line break.
    help_text = " ".join(harness_cli.build_parser().format_help().split())
    assert "absolute run cost cap" in help_text
    assert "collection-only sub-cap" in help_text
    # The cap must not be advertised as a safety net it cannot be. It sits
    # inside the normal cost range, so saying so is part of its description.
    assert "Not a runaway-only guard" in help_text
    # The cost objective is a product target and must stay out of control flow.
    assert args.cost_objective_usd is None
    assert "Never blocks a call and never becomes a stop reason" in help_text


def test_cli_constructs_a_separate_strong_verification_model(monkeypatch):
    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTavily:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(harness_cli, "AsyncOpenAI", FakeOpenAI)
    monkeypatch.setattr(harness_cli, "AsyncTavilyClient", FakeTavily)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setenv("OPENAI_MODEL", "cheap-default")
    monkeypatch.setenv("HARNESS_DECISION_MODEL", "cheap-decision")
    monkeypatch.setenv("HARNESS_NOTE_MODEL", "cheap-note")
    monkeypatch.setenv("HARNESS_CLAIM_MODEL", "cheap-claim")
    monkeypatch.setenv("HARNESS_RECONCILIATION_MODEL", "coverage-model")
    monkeypatch.setenv("HARNESS_ATTRIBUTION_MODEL", "cheap-attribution")
    monkeypatch.setenv("HARNESS_VERIFICATION_MODEL", "strong-verifier")

    clients = harness_cli.build_live_clients()

    assert clients.decision_model.model == "cheap-decision"
    assert clients.note_model.model == "cheap-note"
    assert clients.claim_model.model == "cheap-claim"
    assert clients.reconciliation_model.model == "coverage-model"
    assert clients.attribution_model.model == "cheap-attribution"
    assert clients.verification_model.model == "strong-verifier"
    assert clients.verification_model is not clients.decision_model


def test_cli_defaults_reconciliation_to_attribution_tier(monkeypatch):
    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTavily:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(harness_cli, "AsyncOpenAI", FakeOpenAI)
    monkeypatch.setattr(harness_cli, "AsyncTavilyClient", FakeTavily)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    monkeypatch.setenv("OPENAI_MODEL", "default-model")
    monkeypatch.setenv("HARNESS_ATTRIBUTION_MODEL", "attribution-tier")
    monkeypatch.delenv("HARNESS_RECONCILIATION_MODEL", raising=False)

    clients = harness_cli.build_live_clients()

    assert clients.attribution_model.model == "attribution-tier"
    assert clients.reconciliation_model.model == "attribution-tier"
    assert clients.reconciliation_model is not clients.attribution_model


def test_json_mode_adapter_supplies_provider_required_literal() -> None:
    class FakeCompletions:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"ok":true}')
                    )
                ],
                usage=None,
            )

    completions = FakeCompletions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    model = harness_cli.OpenAIEnvelopeModel(
        client,
        "test-model",
        json_mode=True,
    )

    result = asyncio.run(model.generate("Return one structured object."))

    sent_prompt = completions.calls[0]["messages"][0]["content"]
    assert "json" in sent_prompt.casefold()
    assert completions.calls[0]["response_format"] == {
        "type": "json_object"
    }
    assert result["content"] == '{"ok":true}'


class EmptyEvaluativeClaimModel(ClaimModel):
    """Decomposes normally, but its advisory pass assesses nothing.

    This is what an admission refusal looks like from the runner's side: the
    diagnostic function still returns a well-formed record, it just carries no
    assessments.
    """

    async def generate(self, prompt):
        response = await super().generate(prompt)
        if self.call_number >= 4:
            response = dict(response)
            response["content"] = json.dumps({"claims": []})
        return response


def test_a_diagnostic_pass_that_assessed_nothing_is_not_recorded_as_complete(
    tmp_path,
):
    """Returning a record is not doing the work.

    expected_count and evaluated_count were once the same expression -- both
    counted the external claims the pass was *asked* to assess. A measured run
    whose ten advisory calls were all refused by the admission layer therefore
    recorded "87 of 87 evaluated" while its diagnostics payload was null.
    """

    events = []
    draft = "# Report\n\nThe model wrote this report."

    result = asyncio.run(
        run_harness(
            "A topic",
            checklist_model=ChecklistModel(events),
            decision_model=DecisionModel(events),
            note_model=UnusedNoteModel(),
            write_model=WriteModel(events),
            claim_model=EmptyEvaluativeClaimModel(events, draft),
            reconciliation_model=CoverageModel(events),
            attribution_model=AttributionModel(events),
            verification_model=UnusedVerificationModel(),
            tavily_client=UnusedTavily(),
            budget=LoopBudget(max_rounds=2, max_tokens=100, max_cost_usd=1),
            output_dir=tmp_path,
            run_id="empty-evaluative",
        )
    )

    audit = json.loads(
        (tmp_path / "empty-evaluative" / "audit.json").read_text("utf-8")
    )
    record = audit["posthoc_evidence"]["stage_execution"]["stages"][
        "evaluative_diagnostics"
    ]

    assert record["status"] == "not_run"
    assert record["evaluated_scope"]["count"] == 0
    # The denominator survives so the gap is legible rather than invisible.
    assert record["expected_scope"]["count"] == 1
    assert record["unevaluated_ids"] == ["claim-0001"]
    # An advisory pass is not part of the spine, so it must not block release.
    assert result.publication_eligible is True


class OmittingAttributionModel:
    """Returns a well-formed response that covers none of the claims."""

    def __init__(self, events):
        self.events = events

    async def generate(self, prompt):
        self.events.append("attribution")
        return {
            "content": json.dumps({"action": "attribute", "claims": []}),
            "token_count": 7,
            "cost_usd": 0.02,
        }


def test_an_ineligible_run_still_writes_its_bundle(tmp_path):
    """The incomplete-run path must survive all the way to the audit file.

    This path replaced completion_status via model_copy, which does not
    validate: a raw "partial" string silently displaced the enum and only blew
    up later where the audit reads .value. Every stage had already run and been
    paid for, so the failure cost a full run and produced nothing -- the exact
    total-loss outcome the partial bundle exists to prevent.
    """

    events = []
    draft = "# Report\n\nThe model wrote this report."

    result = asyncio.run(
        run_harness(
            "A topic",
            checklist_model=ChecklistModel(events),
            decision_model=DecisionModel(events),
            note_model=UnusedNoteModel(),
            write_model=WriteModel(events),
            claim_model=ClaimModel(events, draft),
            reconciliation_model=CoverageModel(events),
            attribution_model=OmittingAttributionModel(events),
            verification_model=UnusedVerificationModel(),
            tavily_client=UnusedTavily(),
            budget=LoopBudget(max_rounds=2, max_tokens=100, max_cost_usd=1),
            output_dir=tmp_path,
            run_id="ineligible-run",
        )
    )

    assert result.publication_eligible is False

    audit_path = tmp_path / "ineligible-run" / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    # The value must be the enum's serialised form, not whatever a caller
    # happened to pass to model_copy.
    assert audit["completion_status"] == "partial"
    assert audit["publication_eligible"] is False
    # The bundle is complete even though the run is not.
    assert (tmp_path / "ineligible-run" / "report.md").is_file()
    assert (tmp_path / "ineligible-run" / "sources.md").is_file()
    # And the reader is told, in the report itself, that it is incomplete.
    report = (tmp_path / "ineligible-run" / "report.md").read_text("utf-8")
    assert "不完整运行产物" in report


def test_a_cap_that_bites_after_the_draft_still_writes_an_honest_bundle(
    tmp_path,
):
    """The early-exit branches had no test at all; a real run cost $0.43.

    Fakes spend 0.03 + 0.02 + 0.05 before decomposition, so a 0.10 run cap
    admits the draft and then refuses every post-draft stage. Each of those
    stages must record not_run with its denominator intact, and the bundle must
    still be written -- losing the paid-for draft is the outcome this whole
    mechanism exists to prevent.
    """

    from open_deep_research.harness.budget import RunCostBudget

    events = []
    draft = "# Report\n\nThe model wrote this report."

    result = asyncio.run(
        run_harness(
            "A topic",
            checklist_model=ChecklistModel(events),
            decision_model=DecisionModel(events),
            note_model=UnusedNoteModel(),
            write_model=WriteModel(events),
            claim_model=ClaimModel(events, draft),
            reconciliation_model=CoverageModel(events),
            attribution_model=AttributionModel(events),
            verification_model=UnusedVerificationModel(),
            tavily_client=UnusedTavily(),
            budget=LoopBudget(max_rounds=2, max_tokens=100, max_cost_usd=1),
            run_cost_budget=RunCostBudget(max_cost_usd=0.10),
            output_dir=tmp_path,
            run_id="capped-after-draft",
        )
    )

    assert result.publication_eligible is False

    bundle = tmp_path / "capped-after-draft"
    assert (bundle / "report.md").is_file()
    assert (bundle / "sources.md").is_file()
    audit = json.loads((bundle / "audit.json").read_text(encoding="utf-8"))

    assert audit["completion_status"] == "partial"
    stages = audit["posthoc_evidence"]["stage_execution"]["stages"]
    # Every mandatory post-draft stage was refused, and says so.
    assert stages["claim_decomposition"]["status"] == "partial"
    assert stages["claim_decomposition"]["evaluated_scope"]["count"] == 0
    # The denominator that *was* mechanically knowable survives.
    assert stages["claim_decomposition"]["expected_scope"]["count"] > 0

    # Downstream stages had an empty scope only because decomposition was cut
    # off. Reporting that as "complete over 0 of 0" would smuggle the zero
    # denominator back in through the door the design closed.
    for downstream in ("attribution", "initial_verification"):
        assert stages[downstream]["status"] == "not_run", downstream
        assert stages[downstream]["expected_scope"] is None, downstream

    # The draft the run already paid for is still in the bundle.
    assert "The model wrote this report." in (
        bundle / "report.md"
    ).read_text(encoding="utf-8")

    # And no stage claims completion without having produced anything.
    from open_deep_research.harness.stages import (
        stages_claiming_completion_without_output,
    )

    assert stages_claiming_completion_without_output(audit) == ()
