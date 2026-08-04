import asyncio
import json

import pytest

from harness_pipeline_driver import (
    POST_DRAFT_SEAM_STAGES,
    FullPipelineSeamDriver,
)
from open_deep_research.harness.stages import MANDATORY_PIPELINE_STAGES
from open_deep_research.harness import runner as harness_runner
from open_deep_research.harness.stages import StageExecutionStatus
from open_deep_research.harness.budget import RunCostBudget
from open_deep_research.harness.claims import parse_markdown_blocks
from open_deep_research.harness.edit import EditorialSettings
from open_deep_research.harness.ledger import (
    ResearchLedger,
    SourceLinkCaptureAudit,
    SourceLinkCaptureStatus,
    SourceLinkRecord,
)
from open_deep_research.harness.recovery import RecoveryTriageSettings
from test_harness_edit import CapAfterFirstEditor
from test_harness_recovery import CapAfterFirstTriageModel, _decision


def _audit(result):
    return json.loads(result.audit_path.read_text(encoding="utf-8"))


def _assert_scope_records_are_truthful(stages):
    for stage, record in stages.items():
        if record["expected_scope"] is None:
            assert record["evaluated_scope"] is None, stage
            assert record["status"] != "complete", stage
            assert record["reason"], stage
            continue
        expected = record["expected_scope"]["count"]
        evaluated = record["evaluated_scope"]["count"]
        assert evaluated <= expected, stage
        if record["status"] == "complete":
            assert evaluated == expected, stage
            assert record["unevaluated_ids"] == [], stage


def test_seam_driver_reaches_the_complete_edit_and_reaudit_pipeline(tmp_path):
    result = asyncio.run(FullPipelineSeamDriver(tmp_path).run("seam-baseline"))
    audit = _audit(result)
    stages = audit["posthoc_evidence"]["stage_execution"]["stages"]

    assert result.pipeline_complete is True
    _assert_scope_records_are_truthful(stages)
    for stage in POST_DRAFT_SEAM_STAGES:
        assert stage in stages
        assert stages[stage]["status"] == "complete", stage


@pytest.mark.parametrize("rejected_stage", POST_DRAFT_SEAM_STAGES)
def test_cost_cap_at_each_post_draft_seam_is_recoverable(
    tmp_path,
    rejected_stage,
):
    result = asyncio.run(
        FullPipelineSeamDriver(
            tmp_path,
            rejected_stage=rejected_stage,
        ).run(f"cap-{rejected_stage}")
    )
    audit = _audit(result)
    stages = audit["posthoc_evidence"]["stage_execution"]["stages"]
    assert result.report_path.is_file()
    assert result.sources_path.is_file()
    assert result.audit_path.is_file()
    assert stages[rejected_stage]["status"] != "complete"
    _assert_scope_records_are_truthful(stages)

    incomplete_mandatory = {
        stage
        for stage in MANDATORY_PIPELINE_STAGES
        if stages[stage]["status"] != "complete"
    }
    if rejected_stage in MANDATORY_PIPELINE_STAGES:
        assert result.pipeline_complete is False
        assert rejected_stage in incomplete_mandatory
    elif result.pipeline_complete:
        assert incomplete_mandatory == set()
    else:
        # A non-mandatory rejection may become dynamically required after an
        # edit, cascade, or trip the absolute run cap. Its own scope record is
        # the durable explanation; the fixed five need not be incomplete.
        assert stages[rejected_stage]["status"] != "complete"


def test_unrouted_disagreement_selection_is_partial_and_bundle_survives(tmp_path):
    result = asyncio.run(
        FullPipelineSeamDriver(
            tmp_path,
            select_all_disagreements=True,
            both_claims_supported=True,
        ).run("historical-disagreement-scope")
    )
    audit = _audit(result)
    stage = audit["posthoc_evidence"]["stage_execution"]["stages"][
        "disagreement"
    ]

    assert result.report_path.is_file()
    assert stage["status"] == "partial"
    assert stage["expected_scope"]["count"] == 2
    assert stage["evaluated_scope"]["count"] == 1
    assert stage["unevaluated_ids"] == ["claim-0002"]
    disagreement = audit["posthoc_evidence"]["pre_edit_evidence"][
        "disagreement"
    ]
    assert (
        disagreement["stop_reason"]
        == "single_pass_ended_with_unattempted_selections"
    )


def test_impossible_scope_count_degrades_inside_full_run(tmp_path, monkeypatch):
    def impossible_disagreement_record(result):
        return harness_runner._scope_record(
            status=StageExecutionStatus.PARTIAL,
            reason="synthetic upstream count mismatch",
            unit="selected_claim",
            expected_count=1,
            evaluated_count=2,
        )

    monkeypatch.setattr(
        harness_runner,
        "_disagreement_execution_record",
        impossible_disagreement_record,
    )
    result = asyncio.run(
        FullPipelineSeamDriver(tmp_path).run("historical-impossible-scope")
    )
    audit = _audit(result)
    stage = audit["posthoc_evidence"]["stage_execution"]["stages"][
        "disagreement"
    ]

    assert result.audit_path.is_file()
    assert stage["status"] == "partial"
    assert stage["evaluated_scope"] is None
    assert "evaluated_scope_cannot_exceed_expected_scope" in stage["reason"]


def test_failed_gap_search_is_not_reported_as_evaluated(tmp_path):
    result = asyncio.run(
        FullPipelineSeamDriver(tmp_path, fail_gap_search=True).run(
            "historical-failed-gap-search"
        )
    )
    audit = _audit(result)
    stage = audit["posthoc_evidence"]["stage_execution"]["stages"][
        "evidence_gap"
    ]

    assert result.audit_path.is_file()
    assert stage["status"] == "partial"
    assert stage["expected_scope"]["count"] == 1
    assert stage["evaluated_scope"]["count"] == 0
    assert stage["unevaluated_ids"] == ["claim-0002"]


def test_one_gap_read_failure_is_audited_without_losing_the_pass(
    tmp_path,
    monkeypatch,
):
    original = ResearchLedger.cache_source
    selected_url = "https://alternative.example/record"

    def fail_selected(self, url, cleaned_text, **kwargs):
        if url == selected_url:
            raise ValueError("synthetic cache invariant failure")
        return original(self, url, cleaned_text, **kwargs)

    monkeypatch.setattr(ResearchLedger, "cache_source", fail_selected)
    result = asyncio.run(
        FullPipelineSeamDriver(tmp_path).run("historical-gap-read-failure")
    )
    audit = _audit(result)
    gap = audit["posthoc_evidence"]["pre_edit_evidence"]["evidence_gap"]

    assert result.audit_path.is_file()
    assert gap["acquisitions"][0]["outcome"] == "read_error"
    assert "synthetic cache invariant failure" in gap["acquisitions"][0]["error"]


def test_rejected_cache_sidecar_leaves_no_partial_source_in_full_run(
    tmp_path,
    monkeypatch,
):
    original = ResearchLedger.cache_source
    poisoned_url = "https://source.example/full-pipeline"
    links = (
        SourceLinkRecord(
            target_url="https://records.example/filing.pdf",
            label="record",
        ),
    )
    capture = SourceLinkCaptureAudit(
        status=SourceLinkCaptureStatus.NO_LINKS_CAPTURED,
    )

    def inject_invalid_sidecar(self, url, cleaned_text, **kwargs):
        if url == poisoned_url:
            return original(
                self,
                url,
                cleaned_text,
                source_links=links,
                link_capture=capture,
            )
        return original(self, url, cleaned_text, **kwargs)

    monkeypatch.setattr(ResearchLedger, "cache_source", inject_invalid_sidecar)
    result = asyncio.run(
        FullPipelineSeamDriver(tmp_path).run("historical-atomic-cache")
    )
    audit = _audit(result)

    assert result.audit_path.is_file()
    assert poisoned_url not in audit["ledger"]["source_cache"]
    assert poisoned_url not in audit["ledger"]["source_links"]
    assert poisoned_url not in audit["ledger"]["source_link_capture"]


def test_large_source_lead_inventory_is_truncated_before_triage_provider(
    tmp_path,
):
    result = asyncio.run(
        FullPipelineSeamDriver(
            tmp_path,
            source_lead_count=450,
            prompt_capacity_recovery=True,
        ).run("historical-triage-capacity")
    )
    audit = _audit(result)
    triage = audit["posthoc_evidence"]["pre_edit_evidence"]["recovery_triage"]

    assert result.audit_path.is_file()
    assert triage["status"] == "complete"
    assert triage["source_lead_prompt_inventory_count"] >= 450
    assert triage["source_lead_prompt_truncated"] is True
    assert triage["source_lead_prompt_serialized_chars"] <= 80_000


def test_mid_batch_caps_preserve_recovery_and_editor_progress(tmp_path):
    draft = (
        "# Report\n\nThe model added an unsupported detail.\n\n"
        "A separate audited assertion."
    )
    blocks = parse_markdown_blocks(draft)
    editor = CapAfterFirstEditor(
        {
            "blocks": [
                {
                    "block_id": blocks[1].block_id,
                    "replacement_text": "A narrower supported fact.",
                    "decisions": [
                        {
                            "claim_id": "claim-0001",
                            "action": "qualify",
                            "reason": "Use the narrower audited wording.",
                        }
                    ],
                }
            ]
        }
    )
    result = asyncio.run(
        FullPipelineSeamDriver(
            tmp_path,
            both_claims_unsupported=True,
            recovery_model=CapAfterFirstTriageModel(
                _decision("claim-0001", "edit_directly")
            ),
            editor_model=editor,
            recovery_triage_settings=RecoveryTriageSettings(batch_size=1),
            editorial_settings=EditorialSettings(block_batch_size=1),
        ).run("historical-mid-batch-caps")
    )
    audit = _audit(result)
    pre_edit = audit["posthoc_evidence"]["pre_edit_evidence"]
    triage = pre_edit["recovery_triage"]
    revision = audit["posthoc_evidence"]["editorial_revision"]

    assert result.audit_path.is_file()
    assert triage["status"] == "partial"
    assert [item["claim_id"] for item in triage["decisions"]] == ["claim-0001"]
    assert triage["failed_claim_ids"] == ["claim-0002"]
    assert revision["status"] == "partial"
    assert revision["evaluated_claim_ids"] == ["claim-0001"]
    assert revision["unevaluated_claim_ids"] == ["claim-0002"]
    assert "A narrower supported fact." in revision["edited_draft"]


def test_tail_reserve_never_blocks_the_first_canonical_draft(tmp_path):
    result = asyncio.run(
        FullPipelineSeamDriver(
            tmp_path,
            run_cost_budget=RunCostBudget(
                max_cost_usd=0.035,
                evidence_tail_reserve_usd=0.0055,
            ),
        ).run("historical-writing-admission")
    )
    audit = _audit(result)
    writing = next(
        item
        for item in audit["run_cost_budget"]["admissions"]
        if item["stage"] == "writing"
    )

    assert result.report_path.is_file()
    assert writing["admitted"] is True
    assert writing["protected_reserve_usd"] == 0.0
    assert "The model added an unsupported detail." in result.report.canonical_draft
