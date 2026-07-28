import asyncio
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

import run_harness as harness_cli
from open_deep_research.harness.claims import parse_markdown_blocks
from open_deep_research.harness.loop import LoopBudget, StopReason
from open_deep_research.harness.notes import source_id_for_url
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
                        "anchor_text": paragraph.text,
                        "start_char": paragraph.start_char,
                        "end_char": paragraph.end_char,
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
        return {
            "content": json.dumps(
                {
                    "action": "attribute",
                    "claims": [
                        {
                            "claim_id": "claim-0001",
                            "candidates": [
                                {
                                    "note_id": "note-000001",
                                    "source_id": source_id_for_url(self.url),
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
                            "quote": "exact source evidence 2026.",
                            "explanation": "The full source supports it.",
                        }
                    ]
                }
            ),
            "token_count": 9,
            "cost_usd": 0.03,
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
    assert result.report_path == tmp_path / "fixed-run.md"
    assert result.sources_path == tmp_path / "fixed-run.sources.md"
    assert result.audit_path == tmp_path / "fixed-run.json"
    final_markdown = result.report_path.read_text(encoding="utf-8")
    sources_markdown = result.sources_path.read_text(encoding="utf-8")
    assert final_markdown == result.rendered_report.markdown
    assert final_markdown.startswith("> 证据包：")
    assert (
        "缺失逐字证据、提交标记或摘要不符则证据包不完整"
        in final_markdown.splitlines()[0]
    )
    assert "正文块评估 2/2" in final_markdown.splitlines()[1]
    assert "Run ID：`fixed-run`" in sources_markdown
    assert "[fixed-run.md](fixed-run.md)" in sources_markdown
    assert (
        "> 域名代理集中度：没有正式 claim–source 支持关系；"
        "域名仅作发布方代理。"
    ) in final_markdown
    assert "> 清单对账：已评估 1/1；完整覆盖 1/1" in final_markdown
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
    assert audit["stop"] == {
        "detail": (
            "all checklist items reached a terminal state; "
            "settled_without_located_evidence=1 (what-1)"
        ),
        "is_success": True,
        "open_item_ids": [],
        "reason": "all_items_terminal",
    }
    assert audit["collection_summary"] == {
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
    ] == 1.0
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
        "audit": "fixed-run.json",
        "bundle_complete": True,
        "commit_marker": "fixed-run.json",
        "publication_order": ["sources", "report", "audit"],
        "report": "fixed-run.md",
        "report_sha256": hashlib.sha256(
            final_markdown.encode("utf-8")
        ).hexdigest(),
        "sources": "fixed-run.sources.md",
        "sources_sha256": hashlib.sha256(
            sources_markdown.encode("utf-8")
        ).hexdigest(),
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
        "[查看逐字证据]"
        "(verified-run.sources.md#evidence-1)"
        in markdown
    )
    assert '<a id="evidence-1"></a>' in sources_markdown
    assert "source_id" not in sources_markdown
    assert "start_char" not in sources_markdown
    assert "end_char" not in sources_markdown
    assert result.verification.claims[0].relations[0].model_quote == (
        "exact source evidence 2026."
    )
    assert result.verification.claims[0].relations[0].source_quote == (
        "ExactSourceEvidence 2026"
    )
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["usage"]["verification"] == {
        "cost_usd": 0.03,
        "token_count": 9,
    }
    assert audit["posthoc_evidence"]["rendering"]["footnotes"][0][
        "source_quote"
    ] == "ExactSourceEvidence 2026"
    assert audit["artifacts"]["sources_sha256"] == hashlib.sha256(
        result.sources_path.read_bytes()
    ).hexdigest()


def test_artifact_bundle_rolls_back_when_report_publish_fails(
    tmp_path,
    monkeypatch,
):
    report_path = tmp_path / "run.md"
    sources_path = tmp_path / "run.sources.md"
    audit_path = tmp_path / "run.json"
    real_replace = os.replace
    calls = 0

    def fail_on_report(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated report publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(
        "open_deep_research.harness.runner.os.replace",
        fail_on_report,
    )

    with pytest.raises(OSError, match="simulated report publish failure"):
        _publish_artifact_bundle(
            destination=tmp_path,
            report_path=report_path,
            sources_path=sources_path,
            audit_path=audit_path,
            report_markdown="report",
            sources_markdown="sources",
            audit_json="audit",
        )

    assert not sources_path.exists()
    assert not report_path.exists()
    assert not audit_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_cli_configures_openrouter_proxy_without_touching_no_proxy(monkeypatch):
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.setenv("no_proxy", "leave-this-alone")

    harness_cli.configure_openrouter_proxy("https://openrouter.ai/api/v1")

    assert os.environ["https_proxy"] == "http://127.0.0.1:7890"
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7890"
    assert os.environ["no_proxy"] == "leave-this-alone"


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
