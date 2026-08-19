"""End-to-end orchestration and durable harness artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from open_deep_research.harness.assemble import assemble_notes
from open_deep_research.harness.attribution import (
    AttributionModelClient,
    AttributionResult,
    AttributionSettings,
    AttributionStatus,
    AttributionStopReason,
    attribute_claims,
)
from open_deep_research.harness.checklist import (
    ChecklistModelClient,
    generate_checklist,
)
from open_deep_research.harness.budget import (
    RunCostBudget,
    RunCostBudgetAudit,
    RunCostCapReached,
    RunCostController,
)
from open_deep_research.harness.budget_diagnostics import (
    CompletionStatus,
    RunStopDiagnostic,
    build_run_stop_diagnostic,
)
from open_deep_research.harness.claims import (
    AtomicClaim,
    CitationRequirement,
    ClaimDecompositionResult,
    ClaimDecompositionSettings,
    ClaimModelClient,
    build_selection_prompt,
    decompose_claims,
    parse_markdown_blocks,
)
from open_deep_research.harness.concentration import (
    DomainProxyConcentrationAudit,
    audit_domain_proxy_concentration,
)
from open_deep_research.harness.disagreement import (
    allocate_posthoc_retrieval_budget,
    DisagreementBudget,
    DisagreementResult,
    DisagreementStopReason,
    PosthocRetrievalBudget,
    PosthocRetrievalBudgetAudit,
    disabled_disagreement_result,
    run_disagreement_detection,
    shared_posthoc_budget_audit,
)
from open_deep_research.harness.ledger import ResearchLedger
from open_deep_research.harness.evidence_gap import (
    EvidenceGapBudget,
    EvidenceGapResult,
    EvidenceGapStopReason,
    run_evidence_gap_round,
)
from open_deep_research.harness.edit import (
    EditorialAdmissionAudit,
    EditorialModelClient,
    EditorialRevisionResult,
    EditorialRevisionStatus,
    EditorialSettings,
    audit_editorial_admission,
    editorial_preservation_context,
    revise_audited_draft,
)
from open_deep_research.harness.evaluative import (
    EvaluativeDiagnosticStatus,
    EvaluativeDiagnosticResult,
    EvaluativeDiagnosticSettings,
    diagnose_underspecified_evaluative_claims,
)
from open_deep_research.harness.loop import (
    LoopBudget,
    LoopModelClient,
    LoopResult,
    LoopSettings,
    quote_quality_metrics,
    run_research_loop,
)
from open_deep_research.harness.render import (
    InitialCollectionSnapshot,
    ReaderReportStyle,
    RenderedReport,
    render_verified_report,
)
from open_deep_research.harness.reconcile import (
    ChecklistReportReconciliation,
    ReconciliationModelClient,
    reconcile_checklist_report,
)
from open_deep_research.harness.recovery import (
    EvidenceRecoveryResult,
    EvidenceRecoveryStopReason,
    RecoveryTriageModelClient,
    RecoveryTriageResult,
    RecoveryTriageSettings,
    RecoveryTriageStatus,
    build_recovery_gap_plan_prompt,
    recovery_triage_targets,
    summarize_evidence_recovery,
    triage_evidence_recovery,
)
from open_deep_research.harness.source_spans import (
    build_source_span_registry,
)
from open_deep_research.harness.tools import TavilyClient
from open_deep_research.harness.verify import (
    VerificationBudget,
    VerificationModelClient,
    VerificationRecordStatus,
    VerificationResult,
    VerificationSettings,
    verify_attributions,
)
from open_deep_research.harness.write import (
    ReportDraft,
    WriteModelClient,
    write_report,
)
from open_deep_research.harness.stages import (
    MANDATORY_PIPELINE_STAGES,
    PostDraftExecutionAudit,
    StageExecutionRecord,
    StageExecutionStatus,
    demote_vacuous_completions,
    StageScope,
    publication_audit,
)
from open_deep_research.harness.tail_budget import (
    EvidenceTailReserveAudit,
    EvidenceTailReserveController,
    TailCheckpointName,
    TailWorkUnit,
)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class UsageRecord(BaseModel):
    """Measured usage for one harness stage."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    token_count: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)


class HarnessRunResult(BaseModel):
    """Completed artifact paths and the in-memory collection result."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    run_id: str
    report_path: Path
    sources_path: Path
    audit_path: Path
    report: ReportDraft
    rendered_report: RenderedReport | None
    loop_result: LoopResult
    claim_decomposition: ClaimDecompositionResult | None
    evaluative_diagnostics: EvaluativeDiagnosticResult | None
    checklist_report_reconciliation: ChecklistReportReconciliation | None
    domain_proxy_concentration: DomainProxyConcentrationAudit | None
    attribution: AttributionResult | None
    verification: VerificationResult | None
    evidence_gap: EvidenceGapResult | None
    disagreement: DisagreementResult | None
    recovery_triage: RecoveryTriageResult | None = None
    evidence_recovery: EvidenceRecoveryResult | None = None
    editorial_admission: EditorialAdmissionAudit | None
    editorial_revision: EditorialRevisionResult | None
    posthoc_retrieval_budget: PosthocRetrievalBudgetAudit | None
    run_cost_budget: RunCostBudgetAudit
    stop_diagnostic: RunStopDiagnostic
    post_draft_execution: PostDraftExecutionAudit
    evidence_tail_reserve: EvidenceTailReserveAudit
    pipeline_complete: bool
    quality_review_passed: bool | None
    usage: dict[str, UsageRecord]

    @property
    def publication_eligible(self) -> bool:
        """Compatibility accessor; completion alone is never publication."""

        return self.pipeline_complete and self.quality_review_passed is True


def _new_run_id() -> str:
    return uuid.uuid4().hex


def _normalize_run_id(run_id: str | None) -> str:
    candidate = run_id or _new_run_id()
    if _RUN_ID.fullmatch(candidate) is None:
        raise ValueError(
            "run_id must start with an alphanumeric character and contain only "
            "letters, numbers, underscores, or hyphens"
        )
    return candidate


def _publish_artifact_bundle(
    *,
    destination: Path,
    report_path: Path,
    sources_path: Path,
    audit_path: Path,
    report_markdown: str,
    sources_markdown: str,
    audit_json: str,
) -> None:
    """Stage a complete run directory and publish it with one rename."""

    destination.mkdir(parents=True, exist_ok=True)
    final_directory = report_path.parent
    if (
        final_directory.parent != destination
        or sources_path.parent != final_directory
        or audit_path.parent != final_directory
    ):
        raise ValueError("artifact paths must share one run directory")
    if (
        report_path.name != "report.md"
        or sources_path.name != "sources.md"
        or audit_path.name != "audit.json"
    ):
        raise ValueError("artifact paths must use the canonical filenames")
    legacy_paths = (
        destination / f"{final_directory.name}.md",
        destination / f"{final_directory.name}.sources.md",
        destination / f"{final_directory.name}.json",
    )
    existing = tuple(
        path
        for path in (final_directory, *legacy_paths)
        if path.exists()
    )
    if existing:
        raise FileExistsError(
            "refusing to overwrite an existing artifact bundle: "
            + ", ".join(path.name for path in existing)
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{final_directory.name}-staging-",
            dir=destination,
        )
    )
    staged_sources = staging / sources_path.name
    staged_report = staging / report_path.name
    staged_audit = staging / audit_path.name
    try:
        # Build and validate the whole bundle while it is still invisible.
        staged_sources.write_text(sources_markdown, encoding="utf-8")
        staged_report.write_text(report_markdown, encoding="utf-8")
        staged_audit.write_text(audit_json, encoding="utf-8")
        expected = {
            staged_sources: sources_markdown,
            staged_report: report_markdown,
            staged_audit: audit_json,
        }
        if any(
            path.read_text(encoding="utf-8") != content
            for path, content in expected.items()
        ):
            raise OSError("staged artifact validation failed")
        # The target is preflighted as absent. os.replace publishes either the
        # complete directory or nothing; an existing run is never merged.
        os.replace(staging, final_directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _usage_from_model(model_client: Any) -> UsageRecord:
    raw = getattr(model_client, "last_usage", None)
    if callable(raw):
        raw = raw()
    if isinstance(raw, UsageRecord):
        return raw
    if isinstance(raw, Mapping):
        token_count = raw.get("token_count", 0)
        cost_usd = raw.get("cost_usd", 0.0)
    else:
        token_count = getattr(raw, "token_count", 0)
        cost_usd = getattr(raw, "cost_usd", 0.0)
    try:
        return UsageRecord(token_count=token_count, cost_usd=cost_usd)
    except (TypeError, ValueError):
        return UsageRecord()


def _usage_payload(
    *,
    checklist_usage: UsageRecord,
    collection_usage: UsageRecord,
    writing_usage: UsageRecord,
    decomposition_attribution_usage: UsageRecord,
    reconciliation_usage: UsageRecord,
    verification_usage: UsageRecord,
    evidence_gap_usage: UsageRecord,
    disagreement_usage: UsageRecord,
    additional_stages: Mapping[str, UsageRecord] | None = None,
) -> tuple[dict[str, UsageRecord], dict[str, Any]]:
    stages = {
        "checklist": checklist_usage,
        "collection": collection_usage,
        "writing": writing_usage,
        "decomposition_attribution": decomposition_attribution_usage,
        "reconciliation": reconciliation_usage,
        "verification": verification_usage,
        "evidence_gap": evidence_gap_usage,
        "disagreement": disagreement_usage,
    }
    if additional_stages:
        stages.update(additional_stages)
    total = UsageRecord(
        token_count=sum(value.token_count for value in stages.values()),
        cost_usd=sum(value.cost_usd for value in stages.values()),
    )
    usage = {**stages, "total": total}
    return usage, {
        key: value.model_dump(mode="json")
        for key, value in usage.items()
    }


_MODEL_FOOTNOTE_DEFINITION = re.compile(
    r"^\s*\[\^[A-Za-z0-9_-]+\]:.*$",
    re.MULTILINE,
)
_MODEL_FOOTNOTE_MARKER = re.compile(r"\[\^[A-Za-z0-9_-]+\]")


def _citation_free_partial_draft(canonical_draft: str) -> str:
    """Defend the partial path against a writer violating the P-Cite contract."""

    without_definitions = _MODEL_FOOTNOTE_DEFINITION.sub("", canonical_draft)
    return _MODEL_FOOTNOTE_MARKER.sub("", without_definitions).strip()


def _partial_bundle_markdown(
    *,
    canonical_draft: str,
    run_id: str,
    failed_stage: str,
    audit_filename: str,
) -> tuple[str, str]:
    """Render an unmistakably non-publishable checkpoint without fake metrics."""

    warning = (
        "> **不完整运行产物：证据流程未完成；"
        "`pipeline_complete=false`。** "
        f"后置阶段 `{failed_stage}` 因运行成本上限停止。"
        "未执行的工作没有被解释为无来源、不支持、零覆盖或零候选；"
        f"精确阶段状态与成本诊断见 [{audit_filename}]({audit_filename})。"
    )
    body = _citation_free_partial_draft(canonical_draft)
    report = warning + "\n\n" + body + "\n"
    sources = (
        "# 不完整证据包\n\n"
        f"- Run ID：`{run_id}`\n"
        "- Pipeline complete：`false`\n"
        "- Quality review passed：`not reviewed`\n"
        "- 本文件不包含伪造的空证据记录。证据尾链未完成，"
        "因此没有生成可冒充完整核验结果的脚注定义。\n"
        "- [返回报告](report.md)\n"
    )
    return report, sources


def _scope_record(
    *,
    status: StageExecutionStatus,
    reason: str,
    unit: str | None = None,
    expected_count: int | None = None,
    evaluated_count: int | None = None,
    unevaluated_ids: tuple[str, ...] = (),
) -> StageExecutionRecord:
    """Construct an honest scope record, degrading a false complete state.

    ``StageExecutionRecord`` remains the final mechanical backstop.  At this
    runner boundary, however, a known scope mismatch must produce a durable
    partial bundle rather than let a validation exception erase work already
    paid for.  The reason and ``unevaluated_ids`` retain the exact invariant
    and affected IDs for audit.
    """

    expected = (
        StageScope(unit=unit, count=expected_count)
        if unit is not None and expected_count is not None
        else None
    )
    evaluated = (
        StageScope(unit=unit, count=evaluated_count)
        if expected is not None and evaluated_count is not None
        else None
    )
    if (
        expected is not None
        and evaluated is not None
        and evaluated.count > expected.count
    ):
        observed_count = evaluated.count
        evaluated = None
        status = StageExecutionStatus.PARTIAL
        reason = (
            "mechanical scope invariant "
            "evaluated_scope_cannot_exceed_expected_scope was not satisfied; "
            f"observed_evaluated_count={observed_count}; "
            f"expected_count={expected.count}; recorded partial with unknown "
            "evaluated scope instead. "
            + reason
        )
    if (
        status is StageExecutionStatus.COMPLETE
        and expected is not None
        and (
            evaluated is None
            or evaluated.count != expected.count
            or unevaluated_ids
        )
    ):
        status = StageExecutionStatus.PARTIAL
        reason = (
            "mechanical scope invariant "
            "complete_requires_full_expected_scope was not satisfied; "
            "recorded partial instead. "
            + reason
        )
    return StageExecutionRecord(
        status=status,
        reason=reason,
        expected_scope=expected,
        evaluated_scope=evaluated,
        unevaluated_ids=unevaluated_ids,
    )


def _evidence_gap_execution_record(
    result: EvidenceGapResult,
) -> StageExecutionRecord:
    """Report accepted target routes, never the requested target count.

    ``EvidenceGapStopReason.COMPLETED`` means the one bounded control-flow pass
    ended without a budget/model failure.  It does not mean every requested
    claim received a cache or search route.  Keeping these meanings separate
    prevents a sparse two-claim plan from being audited as 58/58 evaluated.
    """

    expected_count = len(result.target_claim_ids)
    evaluated_count = len(result.routed_target_claim_ids)
    if result.stop_reason is EvidenceGapStopReason.NO_TARGETS:
        status = StageExecutionStatus.COMPLETE
    elif result.stop_reason is EvidenceGapStopReason.COMPLETED:
        status = (
            StageExecutionStatus.COMPLETE
            if evaluated_count == expected_count
            else StageExecutionStatus.PARTIAL
        )
    elif result.stop_reason is EvidenceGapStopReason.BUDGET_EXHAUSTED:
        status = StageExecutionStatus.PARTIAL
    else:
        status = StageExecutionStatus.FAILED
    return _scope_record(
        status=status,
        reason=result.stop_detail,
        unit="target_claim",
        expected_count=expected_count,
        evaluated_count=evaluated_count,
        unevaluated_ids=result.unrouted_target_claim_ids,
    )


def _disagreement_execution_record(
    result: DisagreementResult,
) -> StageExecutionRecord:
    """Record checks actually attempted, never merely selected.

    A disagreement pass may complete its bounded control flow while its plan
    leaves selected claims without a cache or search route.  That is a partial
    scope, not a completed disagreement check and not evidence of agreement.
    """

    selected_ids = tuple(selection.claim_id for selection in result.selected_claims)
    attempted_ids = {
        attempt.claim_id
        for attempt in result.disagreement_search_attempted
        if attempt.methods
    }
    evaluated_ids = tuple(
        claim_id for claim_id in selected_ids if claim_id in attempted_ids
    )
    unevaluated_ids = tuple(
        claim_id for claim_id in selected_ids if claim_id not in attempted_ids
    )
    if result.stop_reason in {
        DisagreementStopReason.COMPLETED,
        DisagreementStopReason.NO_ELIGIBLE_CLAIMS,
        DisagreementStopReason.NO_SELECTION,
    }:
        status = StageExecutionStatus.COMPLETE
    elif result.stop_reason in {
        DisagreementStopReason.BUDGET_EXHAUSTED,
        DisagreementStopReason.SINGLE_PASS_ENDED_WITH_UNATTEMPTED_SELECTIONS,
    }:
        status = StageExecutionStatus.PARTIAL
    else:
        status = StageExecutionStatus.FAILED
    return _scope_record(
        status=status,
        reason=result.stop_detail,
        unit="selected_claim",
        expected_count=len(selected_ids),
        evaluated_count=len(evaluated_ids),
        unevaluated_ids=unevaluated_ids,
    )


def _evaluative_execution_record(
    claims: tuple[AtomicClaim, ...],
    result: EvaluativeDiagnosticResult,
    *,
    draft_label: str,
) -> StageExecutionRecord:
    """Count only usable diagnoses; failure placeholders are not work done."""

    external_ids = tuple(
        claim.claim_id
        for claim in claims
        if claim.citation_requirement is CitationRequirement.EXTERNAL
    )
    assessed_ids = {
        assessment.claim_id
        for assessment in result.assessments
        if assessment.status is not EvaluativeDiagnosticStatus.DIAGNOSTIC_FAILED
    }
    unassessed = tuple(
        claim_id for claim_id in external_ids if claim_id not in assessed_ids
    )
    return _scope_record(
        status=(
            StageExecutionStatus.COMPLETE
            if not unassessed
            else StageExecutionStatus.PARTIAL
            if assessed_ids
            else StageExecutionStatus.NOT_RUN
        ),
        reason=(
            f"advisory diagnostic assessed every {draft_label} external claim"
            if not unassessed
            else f"advisory diagnostic did not assess {len(unassessed)} of "
            f"{len(external_ids)} {draft_label} external claims"
        ),
        unit="external_claim",
        expected_count=len(external_ids),
        evaluated_count=len(assessed_ids),
        unevaluated_ids=unassessed,
    )
def _estimate_cost(model_client: Any, prompt: str) -> float | None:
    estimator = getattr(model_client, "estimate_cost_usd", None)
    if not callable(estimator):
        return None
    try:
        return max(0.0, float(estimator(prompt)))
    except (RuntimeError, TypeError, ValueError):
        return None


def _partial_usage(
    run_cost_audit: RunCostBudgetAudit,
    *,
    checklist_usage: UsageRecord,
    collection_usage: UsageRecord,
    report: ReportDraft,
) -> dict[str, UsageRecord]:
    """Expose measured cost even when a stage did not return its own envelope."""

    stage_cost = run_cost_audit.stage_cost_usd
    usage = {
        "checklist": checklist_usage,
        "collection": collection_usage,
        "writing": UsageRecord(
            token_count=report.token_count,
            cost_usd=report.cost_usd,
        ),
        "decomposition_attribution": UsageRecord(
            cost_usd=stage_cost.get("decomposition_attribution", 0.0)
        ),
        "reconciliation": UsageRecord(
            cost_usd=stage_cost.get("reconciliation", 0.0)
        ),
        "verification": UsageRecord(
            cost_usd=stage_cost.get("verification", 0.0)
        ),
        "evidence_gap": UsageRecord(
            cost_usd=stage_cost.get("evidence_gap", 0.0)
        ),
        "disagreement": UsageRecord(
            cost_usd=stage_cost.get("disagreement", 0.0)
        ),
    }
    usage["total"] = UsageRecord(
        token_count=sum(record.token_count for record in usage.values()),
        cost_usd=run_cost_audit.observed_total_cost_usd,
    )
    return usage


def _publish_partial_result(
    *,
    normalized_run_id: str,
    output_dir: str | Path,
    report: ReportDraft,
    failed_stage: str,
    error: RunCostCapReached,
    ledger: ResearchLedger,
    loop_result: LoopResult,
    checklist_usage: UsageRecord,
    collection_usage: UsageRecord,
    stages: dict[str, StageExecutionRecord],
    tail_reserve: EvidenceTailReserveController,
    initial_collection_snapshot: InitialCollectionSnapshot,
    model_names: Mapping[str, str] | None,
    claim_decomposition: ClaimDecompositionResult | None,
    evaluative_diagnostics: EvaluativeDiagnosticResult | None,
    checklist_reconciliation: ChecklistReportReconciliation | None,
    attribution: AttributionResult | None,
    verification: VerificationResult | None,
) -> HarnessRunResult:
    """Persist a controlled cutoff as a checkpoint, never as a final report."""

    for stage_name in (
        "claim_decomposition",
        "evaluative_diagnostics",
        "checklist_reconciliation",
        "attribution",
        "initial_verification",
        "evidence_gap",
        "disagreement",
        "deterministic_rendering",
    ):
        stages.setdefault(
            stage_name,
            _scope_record(
                status=StageExecutionStatus.NOT_RUN,
                reason=(
                    "not run because the absolute run cost cap stopped an "
                    f"earlier stage ({failed_stage})"
                ),
            ),
        )
    post_draft_execution = publication_audit(stages)
    if post_draft_execution.pipeline_complete:
        raise AssertionError("a cost-cutoff checkpoint cannot be pipeline-complete")

    run_cost_audit = error.audit
    stop_diagnostic = build_run_stop_diagnostic(
        loop_result=loop_result,
        run_cost_audit=run_cost_audit,
        unattributed_claims=(
            sum(
                claim.citation_requirement is CitationRequirement.EXTERNAL
                for claim in claim_decomposition.claims
            )
            if claim_decomposition is not None and attribution is None
            else 0
        ),
        unverified_relations=(
            sum(len(record.candidates) for record in attribution.attributions)
            if attribution is not None and verification is None
            else 0
        ),
        report_written=True,
    )
    if stop_diagnostic.completion_status.value != "partial":
        raise AssertionError("a post-draft cost cutoff must be partial")

    report_filename = "report.md"
    sources_filename = "sources.md"
    audit_filename = "audit.json"
    report_markdown, sources_markdown = _partial_bundle_markdown(
        canonical_draft=report.canonical_draft,
        run_id=normalized_run_id,
        failed_stage=failed_stage,
        audit_filename=audit_filename,
    )
    sources_sha256 = hashlib.sha256(
        sources_markdown.encode("utf-8")
    ).hexdigest()
    report_sha256 = hashlib.sha256(
        report_markdown.encode("utf-8")
    ).hexdigest()
    usage = _partial_usage(
        run_cost_audit,
        checklist_usage=checklist_usage,
        collection_usage=collection_usage,
        report=report,
    )
    usage_audit = {
        name: record.model_dump(mode="json")
        for name, record in usage.items()
    }
    tail_audit = tail_reserve.audit()

    destination = Path(output_dir)
    run_directory = destination / normalized_run_id
    report_path = run_directory / report_filename
    sources_path = run_directory / sources_filename
    audit_path = run_directory / audit_filename
    posthoc: dict[str, Any] = {
        "stage_execution": post_draft_execution.model_dump(mode="json"),
        "claim_decomposition": (
            claim_decomposition.model_dump(mode="json")
            if claim_decomposition is not None
            else None
        ),
        "evaluative_claim_diagnostics": (
            evaluative_diagnostics.model_dump(mode="json")
            if evaluative_diagnostics is not None
            else None
        ),
        "checklist_report_reconciliation": (
            checklist_reconciliation.model_dump(mode="json")
            if checklist_reconciliation is not None
            else None
        ),
        "attribution": (
            attribution.model_dump(mode="json")
            if attribution is not None
            else None
        ),
        "verification": (
            verification.model_dump(mode="json")
            if verification is not None
            else None
        ),
    }
    audit = {
        "run_id": normalized_run_id,
        "topic": loop_result.checklist.topic,
        "ledger": ledger.to_audit_dict(),
        "checklist": loop_result.checklist.model_dump(mode="json"),
        "canonical_draft": report.canonical_draft,
        "completion_status": "partial",
        "pipeline_complete": False,
        "quality_review_passed": None,
        "stop": {
            "reason": loop_result.stop_reason.value,
            "detail": loop_result.stop_detail,
            "open_item_ids": list(loop_result.open_item_ids),
            "is_success": loop_result.is_success,
            "diagnostic": stop_diagnostic.model_dump(mode="json"),
            "post_draft_cutoff_stage": failed_stage,
        },
        "collection_summary": {
            "initial_collection_snapshot": (
                initial_collection_snapshot.model_dump(mode="json")
            ),
        },
        "posthoc_evidence": posthoc,
        "usage": usage_audit,
        "run_cost_budget": run_cost_audit.model_dump(mode="json"),
        "evidence_tail_reserve": tail_audit.model_dump(mode="json"),
        "models": dict(model_names or {}),
        "artifacts": {
            "directory": normalized_run_id,
            "report": report_filename,
            "report_sha256": report_sha256,
            "sources": sources_filename,
            "sources_sha256": sources_sha256,
            "audit": audit_filename,
            "bundle_complete": True,
            "pipeline_complete": False,
            "quality_review_passed": None,
            "artifact_kind": "partial_checkpoint_bundle",
            "staging_write_order": ["sources", "report", "audit"],
            "publication_order": ["directory"],
        },
    }
    audit_json = (
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    _publish_artifact_bundle(
        destination=destination,
        report_path=report_path,
        sources_path=sources_path,
        audit_path=audit_path,
        report_markdown=report_markdown,
        sources_markdown=sources_markdown,
        audit_json=audit_json,
    )
    return HarnessRunResult(
        run_id=normalized_run_id,
        report_path=report_path,
        sources_path=sources_path,
        audit_path=audit_path,
        report=report,
        rendered_report=None,
        loop_result=loop_result,
        claim_decomposition=claim_decomposition,
        evaluative_diagnostics=evaluative_diagnostics,
        checklist_report_reconciliation=checklist_reconciliation,
        domain_proxy_concentration=None,
        attribution=attribution,
        verification=verification,
        evidence_gap=None,
        disagreement=None,
        editorial_admission=None,
        editorial_revision=None,
        posthoc_retrieval_budget=None,
        run_cost_budget=run_cost_audit,
        stop_diagnostic=stop_diagnostic,
        post_draft_execution=post_draft_execution,
        evidence_tail_reserve=tail_audit,
        pipeline_complete=False,
        quality_review_passed=None,
        usage=usage,
    )


async def run_harness(
    topic: str,
    *,
    checklist_model: ChecklistModelClient,
    decision_model: LoopModelClient,
    note_model: LoopModelClient,
    write_model: WriteModelClient,
    claim_model: ClaimModelClient,
    reconciliation_model: ReconciliationModelClient,
    attribution_model: AttributionModelClient,
    verification_model: VerificationModelClient,
    tavily_client: TavilyClient,
    editor_model: EditorialModelClient | None = None,
    recovery_model: RecoveryTriageModelClient | None = None,
    budget: LoopBudget | None = None,
    loop_settings: LoopSettings | None = None,
    claim_settings: ClaimDecompositionSettings | None = None,
    evaluative_diagnostic_settings: (
        EvaluativeDiagnosticSettings | None
    ) = None,
    attribution_settings: AttributionSettings | None = None,
    verification_settings: VerificationSettings | None = None,
    verification_budget: VerificationBudget | None = None,
    editorial_settings: EditorialSettings | None = None,
    recovery_triage_settings: RecoveryTriageSettings | None = None,
    evidence_gap_budget: EvidenceGapBudget | None = None,
    evidence_recovery_budget: EvidenceGapBudget | None = None,
    disagreement_budget: DisagreementBudget | None = None,
    posthoc_retrieval_budget: PosthocRetrievalBudget | None = None,
    run_cost_budget: RunCostBudget | None = None,
    corroboration_target_for_external_claims: int = 2,
    verification_required_independent_sources: int | None = None,
    verification_input_token_estimator: Callable[[str], int] | None = None,
    verification_cost_estimator: Callable[[str], float] | None = None,
    evidence_gap_input_token_estimator: (
        Callable[[Any, str], int] | None
    ) = None,
    evidence_gap_cost_estimator: (
        Callable[[Any, str], float] | None
    ) = None,
    output_dir: str | Path = Path("harness_runs"),
    run_id: str | None = None,
    model_names: Mapping[str, str] | None = None,
    reader_report_style: ReaderReportStyle = ReaderReportStyle.CLEAN,
) -> HarnessRunResult:
    """Run collection, drafting, post-hoc evidence, and artifact rendering."""

    normalized_run_id = _normalize_run_id(run_id)
    reader_report_style = ReaderReportStyle(reader_report_style)
    report_filename = "report.md"
    sources_filename = "sources.md"
    audit_filename = "audit.json"
    ledger = ResearchLedger(research_id=normalized_run_id, topic=topic.strip())
    active_budget = budget or LoopBudget()
    run_cost = RunCostController(run_cost_budget)
    evidence_tail_reserve_usd = (
        run_cost.budget.effective_evidence_tail_reserve_usd
    )
    tail_reserve = EvidenceTailReserveController(
        evidence_tail_reserve_usd
    )
    if (
        run_cost.budget.max_cost_usd is not None
        and (
            active_budget.writing_cost_reserve_usd
            + evidence_tail_reserve_usd
            > run_cost.budget.max_cost_usd
        )
    ):
        raise ValueError(
            "writing and evidence-tail cost reserves together must not "
            "exceed the run-level cost limit"
        )
    if verification_required_independent_sources is not None:
        if corroboration_target_for_external_claims != 2:
            raise ValueError(
                "use corroboration_target_for_external_claims or legacy "
                "verification_required_independent_sources, not both"
            )
        corroboration_target_for_external_claims = (
            verification_required_independent_sources
        )
    if corroboration_target_for_external_claims not in {1, 2}:
        raise ValueError(
            "corroboration_target_for_external_claims must be 1 or 2"
        )

    budgeted_checklist_model = run_cost.wrap(
        checklist_model,
        stage="checklist",
        protected_reserve_usd=(
            active_budget.writing_cost_reserve_usd
            + evidence_tail_reserve_usd
        ),
    )
    checklist = await generate_checklist(
        topic,
        model_client=budgeted_checklist_model,
    )
    checklist_usage = _usage_from_model(budgeted_checklist_model)
    collection_allowance = run_cost.available_before_reserve(
        active_budget.writing_cost_reserve_usd
        + evidence_tail_reserve_usd
    )
    if collection_allowance is not None:
        active_budget = active_budget.model_copy(
            update={
                "max_cost_usd": min(
                    active_budget.max_cost_usd,
                    collection_allowance
                    + active_budget.writing_cost_reserve_usd,
                )
            }
        )
    loop_result = await run_research_loop(
        checklist,
        ledger=ledger,
        decision_model=decision_model,
        note_model=note_model,
        tavily_client=tavily_client,
        budget=active_budget,
        settings=loop_settings,
    )
    collection_usage = UsageRecord(
        token_count=ledger.total_tokens,
        cost_usd=ledger.total_cost_usd,
    )
    run_cost.record_external_usage(
        "collection",
        collection_usage.cost_usd,
    )
    # Freeze this before any post-hoc retrieval can grow the ledger. Reader
    # messaging about the initial collection must not be inferred from the
    # final source cache.
    initial_collection_snapshot = InitialCollectionSnapshot(
        cached_source_count=len(ledger.source_cache),
        note_count=len(ledger.notes),
        usable_note_count=sum(
            note.has_usable_source_span for note in ledger.notes
        ),
    )
    collection_quote_quality = quote_quality_metrics(ledger.notes)
    settled_without_located_evidence = (
        ledger.settled_without_located_evidence
    )
    settled_without_located_evidence_item_ids = (
        ledger.settled_without_located_evidence_item_ids
    )
    rejected_exhausted_without_collection_attempt = (
        ledger.rejected_exhausted_without_collection_attempt
    )
    rejected_exhausted_without_collection_attempt_item_ids = (
        ledger.rejected_exhausted_without_collection_attempt_item_ids
    )
    accepted_exhausted_without_collection_attempt = (
        ledger.accepted_exhausted_without_collection_attempt
    )
    accepted_exhausted_without_collection_attempt_item_ids = (
        ledger.accepted_exhausted_without_collection_attempt_item_ids
    )
    accepted_exhausted_attempt_unknown_legacy = (
        ledger.accepted_exhausted_attempt_unknown_legacy
    )
    accepted_exhausted_attempt_unknown_legacy_item_ids = (
        ledger.accepted_exhausted_attempt_unknown_legacy_item_ids
    )
    exhausted_with_unread_candidates = (
        ledger.exhausted_with_unread_candidates
    )
    exhausted_with_unread_candidates_item_ids = (
        ledger.exhausted_with_unread_candidates_item_ids
    )
    assembled = assemble_notes(loop_result.checklist, ledger.notes)
    budgeted_write_model = run_cost.wrap(
        write_model,
        stage="writing",
        # Writing is the recoverability boundary: before it succeeds there is
        # no canonical draft from which an honest partial bundle can be
        # produced. Collection already protected both the writing allowance
        # and the evidence-tail estimate. Do not turn the latter into a second
        # write-admission gate; if writing leaves too little for the tail, the
        # post-draft cutoff path persists the draft and records the unrun scope.
        protected_reserve_usd=0.0,
    )
    report = await write_report(
        assembled,
        model_client=budgeted_write_model,
        topic=topic,
    )

    # From this point onward a canonical draft exists. A controlled cost cutoff
    # must persist it as a partial checkpoint instead of losing all paid work.
    stage_records: dict[str, StageExecutionRecord] = {}
    claim_decomposition: ClaimDecompositionResult | None = None
    evaluative_diagnostics: EvaluativeDiagnosticResult | None = None
    checklist_report_reconciliation: ChecklistReportReconciliation | None = (
        None
    )
    initial_attribution: AttributionResult | None = None
    initial_verification: VerificationResult | None = None
    recovery_triage: RecoveryTriageResult | None = None
    evidence_recovery: EvidenceRecoveryResult | None = None
    editorial_revision: EditorialRevisionResult | None = None
    editorial_admission: EditorialAdmissionAudit | None = None
    additional_usage: dict[str, UsageRecord] = {}

    parsed_blocks = parse_markdown_blocks(report.canonical_draft)
    claim_batch_size = (
        claim_settings.batch_size
        if claim_settings is not None
        else ClaimDecompositionSettings().batch_size
    )
    selection_span_registry = build_source_span_registry(
        report.canonical_draft
    )
    selection_prompts = tuple(
        build_selection_prompt(
            report.canonical_draft,
            parsed_blocks[index : index + claim_batch_size],
            span_registry=selection_span_registry,
        )
        for index in range(0, len(parsed_blocks), claim_batch_size)
    )
    selection_estimates = tuple(
        _estimate_cost(claim_model, prompt) for prompt in selection_prompts
    )
    known_selection_estimates = tuple(
        estimate for estimate in selection_estimates if estimate is not None
    )
    tail_reserve.checkpoint(
        TailCheckpointName.DRAFT_AVAILABLE,
        work_units=(
            TailWorkUnit(
                stage="claim_decomposition",
                unit="markdown_block",
                count=len(parsed_blocks),
            ),
            TailWorkUnit(
                stage="claim_decomposition",
                unit="selection_batch",
                count=len(selection_prompts),
            ),
        ),
        estimated_remaining_cost_usd=(
            sum(known_selection_estimates)
            if known_selection_estimates
            else None
        ),
        # At this point claim count, attribution paging, and verification
        # relations do not exist. The prompt estimate is therefore explicitly
        # a lower bound, never a complete tail forecast.
        estimate_complete=False,
        limitations=(
            "only selection prompts are constructible before selection",
            "decontextualization, extraction, attribution, reconciliation, "
            "and verification remain unestimated at this checkpoint",
        ),
    )

    budgeted_claim_model = run_cost.wrap(
        claim_model,
        stage="claim_decomposition",
        tail_reserve_controller=tail_reserve,
    )
    try:
        claim_decomposition = await decompose_claims(
            report.canonical_draft,
            model_client=budgeted_claim_model,
            settings=claim_settings,
        )
    except RunCostCapReached as error:
        stage_records["claim_decomposition"] = _scope_record(
            status=StageExecutionStatus.PARTIAL,
            reason=str(error),
            unit="markdown_block",
            expected_count=len(parsed_blocks),
            evaluated_count=0,
            unevaluated_ids=tuple(block.block_id for block in parsed_blocks),
        )
        return _publish_partial_result(
            normalized_run_id=normalized_run_id,
            output_dir=output_dir,
            report=report,
            failed_stage="claim_decomposition",
            error=error,
            ledger=ledger,
            loop_result=loop_result,
            checklist_usage=checklist_usage,
            collection_usage=collection_usage,
            stages=stage_records,
            tail_reserve=tail_reserve,
            initial_collection_snapshot=initial_collection_snapshot,
            model_names=model_names,
            claim_decomposition=None,
            evaluative_diagnostics=None,
            checklist_reconciliation=None,
            attribution=None,
            verification=None,
        )
    claim_coverage = claim_decomposition.registry_coverage
    stage_records["claim_decomposition"] = _scope_record(
        status=(
            StageExecutionStatus.COMPLETE
            if claim_coverage.is_complete
            else StageExecutionStatus.PARTIAL
        ),
        reason=(
            "every markdown block received a selection disposition"
            if claim_coverage.is_complete
            else "some markdown blocks were not successfully assessed"
        ),
        unit="markdown_block",
        expected_count=claim_coverage.total_blocks,
        evaluated_count=claim_coverage.evaluated_blocks,
        unevaluated_ids=claim_coverage.unassessed_block_ids,
    )
    tail_reserve.observe_stage(
        "claim_decomposition",
        work_units=(
            TailWorkUnit(
                stage="claim_decomposition",
                unit="markdown_block",
                count=claim_coverage.total_blocks,
            ),
            TailWorkUnit(
                stage="claim_decomposition",
                unit="atomic_claim",
                count=len(claim_decomposition.claims),
            ),
        ),
        token_count=claim_decomposition.total_tokens,
        cost_usd=claim_decomposition.total_cost_usd,
    )
    tail_reserve.checkpoint(
        TailCheckpointName.CLAIMS_AVAILABLE,
        work_units=(
            TailWorkUnit(
                stage="attribution",
                unit="external_claim",
                count=sum(
                    claim.citation_requirement
                    is CitationRequirement.EXTERNAL
                    for claim in claim_decomposition.claims
                ),
            ),
            TailWorkUnit(
                stage="checklist_reconciliation",
                unit="checklist_item",
                count=len(loop_result.checklist.items),
            ),
        ),
        estimated_remaining_cost_usd=None,
        estimate_complete=False,
        limitations=(
            "attribution paging and claim-source relations are model outputs "
            "that do not exist yet",
        ),
    )

    # This diagnostic is an enhancement, not part of the publication spine.
    # It may run here for prompt locality, but the dynamic controller protects
    # the complete evidence-tail reserve from it.
    budgeted_evaluative_model = run_cost.wrap(
        claim_model,
        stage="evaluative_diagnostics",
        tail_reserve_controller=tail_reserve,
    )
    try:
        evaluative_diagnostics = (
            await diagnose_underspecified_evaluative_claims(
                claim_decomposition.claims,
                model_client=budgeted_evaluative_model,
                settings=evaluative_diagnostic_settings,
            )
        )
        stage_records["evaluative_diagnostics"] = (
            _evaluative_execution_record(
                claim_decomposition.claims,
                evaluative_diagnostics,
                draft_label="initial-draft",
            )
        )
        tail_reserve.observe_stage(
            "evaluative_diagnostics",
            work_units=(
                TailWorkUnit(
                    stage="evaluative_diagnostics",
                    unit="external_claim",
                    count=sum(
                        claim.citation_requirement
                        is CitationRequirement.EXTERNAL
                        for claim in claim_decomposition.claims
                    ),
                ),
            ),
            token_count=evaluative_diagnostics.total_tokens,
            cost_usd=evaluative_diagnostics.total_cost_usd,
        )
    except RunCostCapReached:
        diagnostic_claim_ids = tuple(
            claim.claim_id
            for claim in claim_decomposition.claims
            if claim.citation_requirement is CitationRequirement.EXTERNAL
        )
        evaluative_diagnostics = None
        stage_records["evaluative_diagnostics"] = _scope_record(
            status=StageExecutionStatus.NOT_RUN,
            reason=(
                "enhancement was denied while preserving the mandatory "
                "evidence tail; claim registry and denominators remain unchanged"
            ),
            unit="external_claim",
            expected_count=len(diagnostic_claim_ids),
            evaluated_count=0,
            unevaluated_ids=diagnostic_claim_ids,
        )

    budgeted_reconciliation_model = run_cost.wrap(
        reconciliation_model,
        stage="checklist_reconciliation",
        tail_reserve_controller=tail_reserve,
    )
    try:
        checklist_report_reconciliation = await reconcile_checklist_report(
            report.canonical_draft,
            loop_result.checklist,
            blocks=claim_decomposition.blocks,
            claims=claim_decomposition.claims,
            model_client=budgeted_reconciliation_model,
        )
    except RunCostCapReached as error:
        item_ids = tuple(item.item_id for item in loop_result.checklist.items)
        stage_records["checklist_reconciliation"] = _scope_record(
            status=StageExecutionStatus.NOT_RUN,
            reason=str(error),
            unit="checklist_item",
            expected_count=len(item_ids),
            evaluated_count=0,
            unevaluated_ids=item_ids,
        )
        return _publish_partial_result(
            normalized_run_id=normalized_run_id,
            output_dir=output_dir,
            report=report,
            failed_stage="checklist_reconciliation",
            error=error,
            ledger=ledger,
            loop_result=loop_result,
            checklist_usage=checklist_usage,
            collection_usage=collection_usage,
            stages=stage_records,
            tail_reserve=tail_reserve,
            initial_collection_snapshot=initial_collection_snapshot,
            model_names=model_names,
            claim_decomposition=claim_decomposition,
            evaluative_diagnostics=None,
            checklist_reconciliation=None,
            attribution=None,
            verification=None,
        )
    reconciliation_summary = checklist_report_reconciliation.summary
    stage_records["checklist_reconciliation"] = _scope_record(
        status=(
            StageExecutionStatus.COMPLETE
            if reconciliation_summary.assessment_failed_items == 0
            else StageExecutionStatus.PARTIAL
        ),
        reason=(
            "every checklist item received an auditable coverage disposition"
            if reconciliation_summary.assessment_failed_items == 0
            else "some checklist items could not be assessed"
        ),
        unit="checklist_item",
        expected_count=reconciliation_summary.total_items,
        evaluated_count=reconciliation_summary.assessed_items,
        unevaluated_ids=reconciliation_summary.assessment_failed_item_ids,
    )
    tail_reserve.observe_stage(
        "checklist_reconciliation",
        work_units=(
            TailWorkUnit(
                stage="checklist_reconciliation",
                unit="checklist_item",
                count=reconciliation_summary.total_items,
            ),
        ),
        token_count=checklist_report_reconciliation.total_tokens,
        cost_usd=checklist_report_reconciliation.total_cost_usd,
    )

    budgeted_attribution_model = run_cost.wrap(
        attribution_model,
        stage="attribution",
        tail_reserve_controller=tail_reserve,
    )
    try:
        if claim_decomposition.claims:
            initial_attribution = await attribute_claims(
                claim_decomposition.claims,
                blocks=claim_decomposition.blocks,
                notes=ledger.notes,
                model_client=budgeted_attribution_model,
                settings=attribution_settings,
            )
        else:
            initial_attribution = AttributionResult(
                attributions=(),
                stop_reason=AttributionStopReason.COMPLETED,
            )
    except RunCostCapReached as error:
        external_ids = tuple(
            claim.claim_id
            for claim in claim_decomposition.claims
            if claim.citation_requirement is CitationRequirement.EXTERNAL
        )
        stage_records["attribution"] = _scope_record(
            status=StageExecutionStatus.NOT_RUN,
            reason=str(error),
            unit="external_claim",
            expected_count=len(external_ids),
            evaluated_count=0,
            unevaluated_ids=external_ids,
        )
        return _publish_partial_result(
            normalized_run_id=normalized_run_id,
            output_dir=output_dir,
            report=report,
            failed_stage="attribution",
            error=error,
            ledger=ledger,
            loop_result=loop_result,
            checklist_usage=checklist_usage,
            collection_usage=collection_usage,
            stages=stage_records,
            tail_reserve=tail_reserve,
            initial_collection_snapshot=initial_collection_snapshot,
            model_names=model_names,
            claim_decomposition=claim_decomposition,
            evaluative_diagnostics=None,
            checklist_reconciliation=checklist_report_reconciliation,
            attribution=None,
            verification=None,
        )
    external_claim_ids = tuple(
        claim.claim_id
        for claim in claim_decomposition.claims
        if claim.citation_requirement is CitationRequirement.EXTERNAL
    )
    # An ATTRIBUTION_ERROR record says the stage could not reach a conclusion
    # for this claim, so presence in the result set is not evidence the work
    # was done. NO_CANDIDATE_SOURCE is different and does count: attribution
    # ran and concluded there were no candidates, which is a finding.
    attributed_ids = {
        record.claim.claim_id
        for record in initial_attribution.attributions
        if record.status is not AttributionStatus.ATTRIBUTION_ERROR
    }
    missing_attribution_ids = tuple(
        claim_id
        for claim_id in external_claim_ids
        if claim_id not in attributed_ids
    )
    stage_records["attribution"] = _scope_record(
        status=(
            StageExecutionStatus.COMPLETE
            if not missing_attribution_ids
            else StageExecutionStatus.PARTIAL
        ),
        reason=(
            "every external claim received an attribution result"
            if not missing_attribution_ids
            else "some external claims received no attribution result"
        ),
        unit="external_claim",
        expected_count=len(external_claim_ids),
        evaluated_count=len(external_claim_ids) - len(missing_attribution_ids),
        unevaluated_ids=missing_attribution_ids,
    )
    tail_reserve.observe_stage(
        "attribution",
        work_units=(
            TailWorkUnit(
                stage="attribution",
                unit="external_claim",
                count=len(external_claim_ids),
            ),
        ),
        token_count=initial_attribution.total_tokens,
        cost_usd=initial_attribution.total_cost_usd,
    )
    corroboration_targets = {
        claim.claim_id: corroboration_target_for_external_claims
        for claim in claim_decomposition.claims
        if claim.citation_requirement == CitationRequirement.EXTERNAL
    }
    budgeted_verification_model = run_cost.wrap(
        verification_model,
        stage="initial_verification",
        tail_reserve_controller=tail_reserve,
    )
    run_verification_remaining = run_cost.remaining_cost_usd
    effective_verification_budget = verification_budget
    if run_verification_remaining is not None:
        if effective_verification_budget is None:
            effective_verification_budget = VerificationBudget(
                max_cost_usd=run_verification_remaining
            )
        else:
            configured_verification_cost = (
                effective_verification_budget.max_cost_usd
            )
            effective_verification_budget = (
                effective_verification_budget.model_copy(
                    update={
                        "max_cost_usd": min(
                            run_verification_remaining,
                            configured_verification_cost,
                        )
                        if configured_verification_cost is not None
                        else run_verification_remaining
                    }
                )
            )
    effective_verification_cost_estimator = verification_cost_estimator
    if (
        effective_verification_cost_estimator is None
        and run_cost.configured
    ):
        effective_verification_cost_estimator = (
            budgeted_verification_model.estimate_cost_usd
        )
    candidate_relation_ids = tuple(
        f"{record.claim.claim_id}|{candidate.source_id}"
        for record in initial_attribution.attributions
        for candidate in record.candidates
    )
    tail_reserve.checkpoint(
        TailCheckpointName.ATTRIBUTION_AVAILABLE,
        work_units=(
            TailWorkUnit(
                stage="initial_verification",
                unit="claim_source_relation",
                count=len(candidate_relation_ids),
            ),
        ),
        estimated_remaining_cost_usd=None,
        estimate_complete=False,
        limitations=(
            "verification prompts are URL-grouped and estimates are recorded "
            "at call admission rather than converted into a fixed ratio",
        ),
    )
    try:
        initial_verification = await verify_attributions(
            initial_attribution.attributions,
            source_cache=ledger.source_cache,
            model_client=budgeted_verification_model,
            settings=verification_settings,
            budget=effective_verification_budget,
            corroboration_targets=corroboration_targets,
            estimate_input_tokens=verification_input_token_estimator,
            estimate_cost_usd=effective_verification_cost_estimator,
        )
    except RunCostCapReached as error:
        stage_records["initial_verification"] = _scope_record(
            status=StageExecutionStatus.PARTIAL,
            reason=str(error),
            unit="claim_source_relation",
            expected_count=len(candidate_relation_ids),
            evaluated_count=0,
            unevaluated_ids=candidate_relation_ids,
        )
        return _publish_partial_result(
            normalized_run_id=normalized_run_id,
            output_dir=output_dir,
            report=report,
            failed_stage="initial_verification",
            error=error,
            ledger=ledger,
            loop_result=loop_result,
            checklist_usage=checklist_usage,
            collection_usage=collection_usage,
            stages=stage_records,
            tail_reserve=tail_reserve,
            initial_collection_snapshot=initial_collection_snapshot,
            model_names=model_names,
            claim_decomposition=claim_decomposition,
            evaluative_diagnostics=None,
            checklist_reconciliation=checklist_report_reconciliation,
            attribution=initial_attribution,
            verification=None,
        )
    completed_relation_ids = {
        f"{claim.claim.claim_id}|{relation.source_id}"
        for claim in initial_verification.claims
        for relation in claim.relations
        if relation.status is VerificationRecordStatus.COMPLETED
    }
    unevaluated_relation_ids = tuple(
        relation_id
        for relation_id in candidate_relation_ids
        if relation_id not in completed_relation_ids
    )
    stage_records["initial_verification"] = _scope_record(
        status=(
            StageExecutionStatus.COMPLETE
            if not unevaluated_relation_ids
            else StageExecutionStatus.PARTIAL
        ),
        reason=(
            "every candidate claim-source relation completed verification"
            if not unevaluated_relation_ids
            else "some candidate relations did not complete verification"
        ),
        unit="claim_source_relation",
        expected_count=len(candidate_relation_ids),
        evaluated_count=len(candidate_relation_ids)
        - len(unevaluated_relation_ids),
        unevaluated_ids=unevaluated_relation_ids,
    )
    tail_reserve.observe_stage(
        "initial_verification",
        work_units=(
            TailWorkUnit(
                stage="initial_verification",
                unit="claim_source_relation",
                count=len(candidate_relation_ids),
            ),
        ),
        token_count=initial_verification.total_tokens,
        cost_usd=initial_verification.total_cost_usd,
    )
    # With audit-after-edit enabled, the first verification is deliberately
    # not the end of the mandatory evidence tail. Enhancement passes may run
    # next, but they cannot borrow the still-frozen reserve needed to audit an
    # edited draft. The reserve is released only after the second registry has
    # either completed or the unchanged draft makes it unnecessary.
    if editor_model is None:
        tail_reserve.checkpoint(
            TailCheckpointName.MANDATORY_TAIL_COMPLETE,
            work_units=(),
            estimated_remaining_cost_usd=0.0,
            estimate_complete=True,
        )

    posthoc_allocation = allocate_posthoc_retrieval_budget(
        shared_budget=posthoc_retrieval_budget,
        evidence_gap_budget=evidence_gap_budget,
        disagreement_budget=disagreement_budget,
    )
    effective_evidence_gap_budget = posthoc_allocation.evidence_gap_budget
    run_gap_remaining = run_cost.remaining_cost_usd
    if (
        effective_evidence_gap_budget is not None
        and run_gap_remaining is not None
    ):
        effective_evidence_gap_budget = (
            effective_evidence_gap_budget.model_copy(
                update={
                    "max_cost_usd": min(
                        effective_evidence_gap_budget.max_cost_usd,
                        run_gap_remaining,
                    )
                }
            )
        )
    if effective_evidence_gap_budget is None:
        evidence_gap = EvidenceGapResult(
            stop_reason=EvidenceGapStopReason.DISABLED,
            stop_detail="no independent evidence-gap budget was configured",
            final_attribution=initial_attribution,
            final_verification=initial_verification,
        )
        stage_records["evidence_gap"] = _scope_record(
            status=StageExecutionStatus.NOT_RUN,
            reason="no independent evidence-gap budget was configured",
        )
    else:
        gap_decision_model = run_cost.wrap(
            decision_model,
            stage="evidence_gap",
        )
        gap_note_model = run_cost.wrap(
            note_model,
            stage="evidence_gap",
        )
        gap_attribution_model = run_cost.wrap(
            attribution_model,
            stage="evidence_gap",
        )
        gap_verification_model = run_cost.wrap(
            verification_model,
            stage="evidence_gap",
        )
        try:
            evidence_gap = await run_evidence_gap_round(
                canonical_draft=report.canonical_draft,
                checklist=loop_result.checklist,
                blocks=claim_decomposition.blocks,
                ledger=ledger,
                initial_attribution=initial_attribution,
                initial_verification=initial_verification,
                gap_model=gap_decision_model,
                note_model=gap_note_model,
                attribution_model=gap_attribution_model,
                verification_model=gap_verification_model,
                tavily_client=tavily_client,
                budget=effective_evidence_gap_budget,
                attribution_settings=attribution_settings,
                verification_settings=verification_settings,
                corroboration_targets=corroboration_targets,
                estimate_input_tokens=evidence_gap_input_token_estimator,
                estimate_cost_usd=evidence_gap_cost_estimator,
            )
        except RunCostCapReached:
            evidence_gap = EvidenceGapResult(
                stop_reason=EvidenceGapStopReason.BUDGET_EXHAUSTED,
                stop_detail=(
                    "absolute run cost admission stopped this bounded "
                    "enhancement pass; mandatory evidence registries are "
                    "preserved"
                ),
                final_attribution=initial_attribution,
                final_verification=initial_verification,
            )
        stage_records["evidence_gap"] = _evidence_gap_execution_record(
            evidence_gap
        )
    evidence_gap_attribution = evidence_gap.final_attribution
    evidence_gap_verification = evidence_gap.final_verification
    if disagreement_budget is None:
        disagreement = disabled_disagreement_result(
            evidence_gap_attribution,
            evidence_gap_verification,
        )
        stage_records["disagreement"] = _scope_record(
            status=StageExecutionStatus.NOT_RUN,
            reason="no disagreement-detection budget was configured",
        )
    else:
        if posthoc_retrieval_budget is None:
            effective_disagreement_budget = disagreement_budget
        else:
            effective_disagreement_budget = disagreement_budget.model_copy(
                update={
                    "max_tokens": min(
                        disagreement_budget.max_tokens,
                        max(
                            0,
                            posthoc_retrieval_budget.max_tokens
                            - evidence_gap.total_tokens,
                        ),
                    ),
                    "max_cost_usd": min(
                        disagreement_budget.max_cost_usd,
                        max(
                            0.0,
                            posthoc_retrieval_budget.max_cost_usd
                            - evidence_gap.total_cost_usd,
                        ),
                    ),
                }
            )
        run_disagreement_remaining = run_cost.remaining_cost_usd
        if run_disagreement_remaining is not None:
            effective_disagreement_budget = (
                effective_disagreement_budget.model_copy(
                    update={
                        "max_cost_usd": min(
                            effective_disagreement_budget.max_cost_usd,
                            run_disagreement_remaining,
                        )
                    }
                )
            )
        disagreement_selection_model = run_cost.wrap(
            decision_model,
            stage="disagreement",
        )
        disagreement_note_model = run_cost.wrap(
            note_model,
            stage="disagreement",
        )
        disagreement_attribution_model = run_cost.wrap(
            attribution_model,
            stage="disagreement",
        )
        disagreement_verification_model = run_cost.wrap(
            verification_model,
            stage="disagreement",
        )
        try:
            disagreement = await run_disagreement_detection(
                canonical_draft=report.canonical_draft,
                checklist=loop_result.checklist,
                blocks=claim_decomposition.blocks,
                ledger=ledger,
                initial_attribution=evidence_gap_attribution,
                initial_verification=evidence_gap_verification,
                selection_model=disagreement_selection_model,
                note_model=disagreement_note_model,
                attribution_model=disagreement_attribution_model,
                verification_model=disagreement_verification_model,
                tavily_client=tavily_client,
                budget=effective_disagreement_budget,
                attribution_settings=attribution_settings,
                verification_settings=verification_settings,
                corroboration_targets=corroboration_targets,
                estimate_input_tokens=evidence_gap_input_token_estimator,
                estimate_cost_usd=evidence_gap_cost_estimator,
            )
        except RunCostCapReached:
            disagreement = disabled_disagreement_result(
                evidence_gap_attribution,
                evidence_gap_verification,
                detail=(
                    "absolute run cost admission stopped this bounded "
                    "enhancement pass"
                ),
            ).model_copy(
                update={
                    "stop_reason": DisagreementStopReason.BUDGET_EXHAUSTED,
                }
            )
        stage_records["disagreement"] = _disagreement_execution_record(
            disagreement
        )
    attribution = disagreement.final_attribution
    verification = disagreement.final_verification
    posthoc_budget_audit = shared_posthoc_budget_audit(
        budget=posthoc_retrieval_budget,
        evidence_gap_tokens=evidence_gap.total_tokens,
        evidence_gap_cost_usd=evidence_gap.total_cost_usd,
        disagreement_tokens=disagreement.total_tokens,
        disagreement_cost_usd=disagreement.total_cost_usd,
        disagreement_reserved_tokens=(
            posthoc_allocation.disagreement_reserved_tokens
        ),
        disagreement_reserved_cost_usd=(
            posthoc_allocation.disagreement_reserved_cost_usd
        ),
        evidence_gap_admission_max_tokens=(
            effective_evidence_gap_budget.max_tokens
            if effective_evidence_gap_budget is not None
            else 0
        ),
        evidence_gap_admission_max_cost_usd=(
            effective_evidence_gap_budget.max_cost_usd
            if effective_evidence_gap_budget is not None
            else 0.0
        ),
    )

    # A recovery pass is deliberately separate from both the ordinary gap
    # enhancer and the mutating editor. Triage may identify a material fact as
    # worth one more bounded research attempt, but it cannot edit report bytes;
    # the frozen claim IDs then flow through the existing cache-first executor.
    recovery_expected_target_ids = tuple(
        target.claim.claim_id
        for target in recovery_triage_targets(verification)
    )
    if recovery_model is None or evidence_recovery_budget is None:
        disabled_reason = (
            "no recovery triage model was configured"
            if recovery_model is None
            else "no independent evidence-recovery budget was configured"
        )
        stage_records["recovery_triage"] = _scope_record(
            status=StageExecutionStatus.NOT_RUN,
            reason=disabled_reason,
            unit="evidence_exception_claim",
            expected_count=len(recovery_expected_target_ids),
            evaluated_count=0,
            unevaluated_ids=recovery_expected_target_ids,
        )
        stage_records["evidence_recovery"] = _scope_record(
            status=StageExecutionStatus.NOT_RUN,
            reason=disabled_reason,
        )
    else:
        budgeted_recovery_model = run_cost.wrap(
            recovery_model,
            stage="recovery_triage",
            tail_reserve_controller=tail_reserve,
        )
        try:
            recovery_triage = await triage_evidence_recovery(
                report.canonical_draft,
                checklist=loop_result.checklist,
                verification=verification,
                model_client=budgeted_recovery_model,
                settings=recovery_triage_settings,
                source_cache=ledger.source_cache,
                source_links=ledger.source_links,
            )
        except RunCostCapReached as error:
            stage_records["recovery_triage"] = _scope_record(
                status=StageExecutionStatus.NOT_RUN,
                reason=str(error),
                unit="evidence_exception_claim",
                expected_count=len(recovery_expected_target_ids),
                evaluated_count=0,
                unevaluated_ids=recovery_expected_target_ids,
            )
            stage_records["evidence_recovery"] = _scope_record(
                status=StageExecutionStatus.NOT_RUN,
                reason="triage was denied, so no recovery target set exists",
            )
        else:
            stage_records["recovery_triage"] = _scope_record(
                status={
                    RecoveryTriageStatus.NO_TARGETS: (
                        StageExecutionStatus.COMPLETE
                    ),
                    RecoveryTriageStatus.COMPLETE: (
                        StageExecutionStatus.COMPLETE
                    ),
                    RecoveryTriageStatus.PARTIAL: (
                        StageExecutionStatus.PARTIAL
                    ),
                    RecoveryTriageStatus.FAILED: StageExecutionStatus.FAILED,
                }[recovery_triage.status],
                reason=(
                    "every applicable external evidence exception received "
                    "a non-mutating recovery disposition; non-external "
                    "exceptions remain separately audited as inapplicable"
                    if recovery_triage.status
                    in {
                        RecoveryTriageStatus.NO_TARGETS,
                        RecoveryTriageStatus.COMPLETE,
                    }
                    else "some evidence exceptions did not receive a usable "
                    "recovery disposition"
                ),
                unit="evidence_exception_claim",
                expected_count=len(recovery_triage.target_claim_ids),
                evaluated_count=len(recovery_triage.decisions),
                unevaluated_ids=recovery_triage.failed_claim_ids,
            )
            additional_usage["recovery_triage"] = UsageRecord(
                token_count=recovery_triage.total_tokens,
                cost_usd=recovery_triage.total_cost_usd,
            )
            research_ids = recovery_triage.research_target_claim_ids
            if research_ids:
                effective_recovery_budget = evidence_recovery_budget
                recovery_remaining = run_cost.remaining_cost_usd
                if recovery_remaining is not None:
                    effective_recovery_budget = (
                        effective_recovery_budget.model_copy(
                            update={
                                "max_cost_usd": min(
                                    effective_recovery_budget.max_cost_usd,
                                    recovery_remaining,
                                )
                            }
                        )
                    )
                recovery_gap_model = run_cost.wrap(
                    decision_model,
                    stage="evidence_recovery",
                    tail_reserve_controller=tail_reserve,
                )
                recovery_note_model = run_cost.wrap(
                    note_model,
                    stage="evidence_recovery",
                    tail_reserve_controller=tail_reserve,
                )
                recovery_attribution_model = run_cost.wrap(
                    attribution_model,
                    stage="evidence_recovery",
                    tail_reserve_controller=tail_reserve,
                )
                recovery_verification_model = run_cost.wrap(
                    verification_model,
                    stage="evidence_recovery",
                    tail_reserve_controller=tail_reserve,
                )

                def recovery_plan_builder(**kwargs: Any) -> str:
                    return build_recovery_gap_plan_prompt(
                        **kwargs,
                        triage=recovery_triage,
                    )

                try:
                    recovery_pass = await run_evidence_gap_round(
                        canonical_draft=report.canonical_draft,
                        checklist=loop_result.checklist,
                        blocks=claim_decomposition.blocks,
                        ledger=ledger,
                        initial_attribution=attribution,
                        initial_verification=verification,
                        gap_model=recovery_gap_model,
                        note_model=recovery_note_model,
                        attribution_model=recovery_attribution_model,
                        verification_model=recovery_verification_model,
                        tavily_client=tavily_client,
                        budget=effective_recovery_budget,
                        attribution_settings=attribution_settings,
                        verification_settings=verification_settings,
                        corroboration_targets=corroboration_targets,
                        estimate_input_tokens=(
                            evidence_gap_input_token_estimator
                        ),
                        estimate_cost_usd=evidence_gap_cost_estimator,
                        explicit_target_claim_ids=research_ids,
                        plan_prompt_builder=recovery_plan_builder,
                        ledger_event_prefix="recovery",
                    )
                except RunCostCapReached:
                    recovery_pass = EvidenceGapResult(
                        target_claim_ids=research_ids,
                        stop_reason=EvidenceGapStopReason.BUDGET_EXHAUSTED,
                        stop_detail=(
                            "absolute run cost admission stopped the only "
                            "evidence-recovery pass"
                        ),
                        final_attribution=attribution,
                        final_verification=verification,
                    )
            else:
                recovery_pass = EvidenceGapResult(
                    stop_reason=EvidenceGapStopReason.NO_TARGETS,
                    stop_detail=(
                        "triage selected no claims for evidence recovery"
                    ),
                    final_attribution=attribution,
                    final_verification=verification,
                )
            evidence_recovery = summarize_evidence_recovery(
                triage=recovery_triage,
                pass_result=recovery_pass,
                initial_verification=verification,
                cached_source_urls=tuple(ledger.source_cache),
            )
            recovery_complete = (
                not evidence_recovery.unattempted_claim_ids
                and evidence_recovery.stop_reason
                not in {
                    EvidenceRecoveryStopReason.BUDGET_EXHAUSTED,
                    EvidenceRecoveryStopReason.MODEL_ERROR,
                }
            )
            recovery_failed = (
                evidence_recovery.stop_reason
                is EvidenceRecoveryStopReason.MODEL_ERROR
            )
            stage_records["evidence_recovery"] = _scope_record(
                status=(
                    StageExecutionStatus.COMPLETE
                    if recovery_complete
                    else StageExecutionStatus.FAILED
                    if recovery_failed
                    else StageExecutionStatus.PARTIAL
                ),
                reason=evidence_recovery.stop_detail,
                unit="frozen_recovery_target",
                expected_count=len(
                    evidence_recovery.frozen_target_claim_ids
                ),
                evaluated_count=len(evidence_recovery.attempted_claim_ids),
                unevaluated_ids=evidence_recovery.unattempted_claim_ids,
            )
            additional_usage["evidence_recovery"] = UsageRecord(
                token_count=recovery_pass.total_tokens,
                cost_usd=recovery_pass.total_cost_usd,
            )
            attribution = recovery_pass.final_attribution
            verification = recovery_pass.final_verification

    # Freeze the complete first audit before any editorial judgement.  These
    # records remain in the durable audit even when a revised draft later earns
    # a fresh registry; deleting a claim from prose must never delete the fact
    # that the first draft made it or how the sources assessed it.
    pre_edit_claim_decomposition = claim_decomposition
    pre_edit_evaluative_diagnostics = evaluative_diagnostics
    pre_edit_reconciliation = checklist_report_reconciliation
    pre_edit_attribution = attribution
    pre_edit_verification = verification
    pre_edit_draft = report.canonical_draft
    post_edit_required_stages: tuple[str, ...] = ()
    post_edit_evaluative_diagnostics: EvaluativeDiagnosticResult | None = None

    if editor_model is not None:
        editorial_admission = audit_editorial_admission(
            pre_edit_verification,
            blocks=pre_edit_claim_decomposition.blocks,
        )
        target_ids = editorial_admission.target_claim_ids
        eligible_target_ids = editorial_admission.eligible_target_claim_ids
        budgeted_editor_model = run_cost.wrap(
            editor_model,
            stage="audit_editing",
            tail_reserve_controller=tail_reserve,
        )
        try:
            editorial_revision = await revise_audited_draft(
                pre_edit_draft,
                blocks=pre_edit_claim_decomposition.blocks,
                verification=pre_edit_verification,
                model_client=budgeted_editor_model,
                settings=editorial_settings,
                preservation_context=editorial_preservation_context(
                    loop_result.checklist
                ),
            )
        except RunCostCapReached as error:
            stage_records["audit_editing"] = _scope_record(
                status=StageExecutionStatus.NOT_RUN,
                reason=str(error),
                unit="eligible_audited_claim",
                expected_count=len(eligible_target_ids),
                evaluated_count=0,
                unevaluated_ids=eligible_target_ids,
            )
        else:
            editorial_status = {
                EditorialRevisionStatus.COMPLETE: StageExecutionStatus.COMPLETE,
                EditorialRevisionStatus.PARTIAL: StageExecutionStatus.PARTIAL,
                EditorialRevisionStatus.FAILED: StageExecutionStatus.FAILED,
            }[editorial_revision.status]
            stage_records["audit_editing"] = _scope_record(
                status=editorial_status,
                reason=(
                    "every block-locally eligible evidence exception received "
                    "one auditable editorial decision; blocked blocks remained "
                    "byte-for-byte untouched"
                    if editorial_status is StageExecutionStatus.COMPLETE
                    else (
                        "the editorial pass did not assess every eligible "
                        "target; accepted blocks form a partial proposal, "
                        "rejected blocks remain byte-for-byte unchanged, and "
                        "any changed draft still requires full re-audit"
                    )
                ),
                unit="eligible_audited_claim",
                expected_count=len(
                    editorial_revision.eligible_target_claim_ids
                ),
                evaluated_count=len(editorial_revision.evaluated_claim_ids),
                unevaluated_ids=editorial_revision.unevaluated_claim_ids,
            )
            additional_usage["audit_editing"] = UsageRecord(
                token_count=editorial_revision.total_tokens,
                cost_usd=editorial_revision.total_cost_usd,
            )

        if (
            editorial_revision is not None
            and editorial_revision.requires_reaudit
        ):
            post_edit_required_stages = (
                "post_edit_claim_decomposition",
                "post_edit_attribution",
                "post_edit_initial_verification",
                "post_edit_checklist_reconciliation",
            )
            proposed_draft = editorial_revision.edited_draft
            proposed_blocks = parse_markdown_blocks(proposed_draft)
            post_claims: ClaimDecompositionResult | None = None
            post_attribution: AttributionResult | None = None
            post_verification: VerificationResult | None = None
            post_reconciliation: ChecklistReportReconciliation | None = None

            budgeted_post_claim_model = run_cost.wrap(
                claim_model,
                stage="post_edit_claim_decomposition",
                tail_reserve_controller=tail_reserve,
            )
            try:
                post_claims = await decompose_claims(
                    proposed_draft,
                    model_client=budgeted_post_claim_model,
                    settings=claim_settings,
                )
            except RunCostCapReached as error:
                stage_records["post_edit_claim_decomposition"] = _scope_record(
                    status=StageExecutionStatus.NOT_RUN,
                    reason=str(error),
                    unit="markdown_block",
                    expected_count=len(proposed_blocks),
                    evaluated_count=0,
                    unevaluated_ids=tuple(
                        block.block_id for block in proposed_blocks
                    ),
                )
            else:
                coverage = post_claims.registry_coverage
                stage_records["post_edit_claim_decomposition"] = _scope_record(
                    status=(
                        StageExecutionStatus.COMPLETE
                        if coverage.is_complete
                        else StageExecutionStatus.PARTIAL
                    ),
                    reason=(
                        "the edited draft received a complete new claim registry"
                        if coverage.is_complete
                        else "the edited draft claim registry is incomplete"
                    ),
                    unit="markdown_block",
                    expected_count=coverage.total_blocks,
                    evaluated_count=coverage.evaluated_blocks,
                    unevaluated_ids=coverage.unassessed_block_ids,
                )
                additional_usage["post_edit_claim_decomposition"] = UsageRecord(
                    token_count=post_claims.total_tokens,
                    cost_usd=post_claims.total_cost_usd,
                )
                tail_reserve.observe_stage(
                    "post_edit_claim_decomposition",
                    work_units=(
                        TailWorkUnit(
                            stage="post_edit_claim_decomposition",
                            unit="markdown_block",
                            count=coverage.total_blocks,
                        ),
                        TailWorkUnit(
                            stage="post_edit_claim_decomposition",
                            unit="atomic_claim",
                            count=len(post_claims.claims),
                        ),
                    ),
                    token_count=post_claims.total_tokens,
                    cost_usd=post_claims.total_cost_usd,
                )

            post_claim_stage_complete = (
                stage_records["post_edit_claim_decomposition"].status
                is StageExecutionStatus.COMPLETE
            )
            if post_claim_stage_complete and post_claims is not None:
                post_external_ids = tuple(
                    claim.claim_id
                    for claim in post_claims.claims
                    if claim.citation_requirement is CitationRequirement.EXTERNAL
                )
                budgeted_post_evaluative_model = run_cost.wrap(
                    claim_model,
                    stage="post_edit_evaluative_diagnostics",
                    tail_reserve_controller=tail_reserve,
                )
                try:
                    post_edit_evaluative_diagnostics = (
                        await diagnose_underspecified_evaluative_claims(
                            post_claims.claims,
                            model_client=budgeted_post_evaluative_model,
                            settings=evaluative_diagnostic_settings,
                        )
                    )
                except RunCostCapReached as error:
                    stage_records[
                        "post_edit_evaluative_diagnostics"
                    ] = _scope_record(
                        status=StageExecutionStatus.NOT_RUN,
                        reason=str(error),
                        unit="external_claim",
                        expected_count=len(post_external_ids),
                        evaluated_count=0,
                        unevaluated_ids=post_external_ids,
                    )
                else:
                    stage_records[
                        "post_edit_evaluative_diagnostics"
                    ] = _evaluative_execution_record(
                        post_claims.claims,
                        post_edit_evaluative_diagnostics,
                        draft_label="edited-draft",
                    )
                    additional_usage[
                        "post_edit_evaluative_diagnostics"
                    ] = UsageRecord(
                        token_count=(
                            post_edit_evaluative_diagnostics.total_tokens
                        ),
                        cost_usd=(
                            post_edit_evaluative_diagnostics.total_cost_usd
                        ),
                    )
                    tail_reserve.observe_stage(
                        "post_edit_evaluative_diagnostics",
                        work_units=(
                            TailWorkUnit(
                                stage="post_edit_evaluative_diagnostics",
                                unit="external_claim",
                                count=len(post_external_ids),
                            ),
                        ),
                        token_count=(
                            post_edit_evaluative_diagnostics.total_tokens
                        ),
                        cost_usd=(
                            post_edit_evaluative_diagnostics.total_cost_usd
                        ),
                    )
            else:
                stage_records[
                    "post_edit_evaluative_diagnostics"
                ] = _scope_record(
                    status=StageExecutionStatus.NOT_RUN,
                    reason=(
                        "not run because the edited claim registry did not "
                        "complete"
                    ),
                )
            if post_claim_stage_complete and post_claims is not None:
                post_external_ids = tuple(
                    claim.claim_id
                    for claim in post_claims.claims
                    if claim.citation_requirement is CitationRequirement.EXTERNAL
                )
                budgeted_post_attribution_model = run_cost.wrap(
                    attribution_model,
                    stage="post_edit_attribution",
                    tail_reserve_controller=tail_reserve,
                )
                try:
                    post_attribution = (
                        await attribute_claims(
                            post_claims.claims,
                            blocks=post_claims.blocks,
                            notes=ledger.notes,
                            model_client=budgeted_post_attribution_model,
                            settings=attribution_settings,
                        )
                        if post_claims.claims
                        else AttributionResult(
                            attributions=(),
                            stop_reason=AttributionStopReason.COMPLETED,
                        )
                    )
                except RunCostCapReached as error:
                    stage_records["post_edit_attribution"] = _scope_record(
                        status=StageExecutionStatus.NOT_RUN,
                        reason=str(error),
                        unit="external_claim",
                        expected_count=len(post_external_ids),
                        evaluated_count=0,
                        unevaluated_ids=post_external_ids,
                    )
                else:
                    post_attributed_ids = {
                        item.claim.claim_id
                        for item in post_attribution.attributions
                        if item.status is not AttributionStatus.ATTRIBUTION_ERROR
                    }
                    post_unattributed_ids = tuple(
                        claim_id
                        for claim_id in post_external_ids
                        if claim_id not in post_attributed_ids
                    )
                    stage_records["post_edit_attribution"] = _scope_record(
                        status=(
                            StageExecutionStatus.COMPLETE
                            if not post_unattributed_ids
                            else StageExecutionStatus.PARTIAL
                        ),
                        reason=(
                            "every edited external claim received an "
                            "attribution conclusion"
                            if not post_unattributed_ids
                            else "some edited claims received attribution "
                            "errors, which are not conclusions"
                        ),
                        unit="external_claim",
                        expected_count=len(post_external_ids),
                        evaluated_count=(
                            len(post_external_ids) - len(post_unattributed_ids)
                        ),
                        unevaluated_ids=post_unattributed_ids,
                    )
                    additional_usage["post_edit_attribution"] = UsageRecord(
                        token_count=post_attribution.total_tokens,
                        cost_usd=post_attribution.total_cost_usd,
                    )
                    tail_reserve.observe_stage(
                        "post_edit_attribution",
                        work_units=(
                            TailWorkUnit(
                                stage="post_edit_attribution",
                                unit="external_claim",
                                count=len(post_external_ids),
                            ),
                        ),
                        token_count=post_attribution.total_tokens,
                        cost_usd=post_attribution.total_cost_usd,
                    )
            else:
                stage_records["post_edit_attribution"] = _scope_record(
                    status=StageExecutionStatus.NOT_RUN,
                    reason=(
                        "not run because the edited claim registry did not "
                        "complete; an empty downstream scope is not completion"
                    ),
                )

            post_attr_complete = (
                stage_records["post_edit_attribution"].status
                is StageExecutionStatus.COMPLETE
            )
            if (
                post_attr_complete
                and post_attribution is not None
                and post_claims is not None
            ):
                post_relation_ids = tuple(
                    f"{item.claim.claim_id}|{candidate.source_id}"
                    for item in post_attribution.attributions
                    for candidate in item.candidates
                )
                post_targets = {
                    claim.claim_id: corroboration_target_for_external_claims
                    for claim in post_claims.claims
                    if claim.citation_requirement is CitationRequirement.EXTERNAL
                }
                budgeted_post_verification_model = run_cost.wrap(
                    verification_model,
                    stage="post_edit_initial_verification",
                    tail_reserve_controller=tail_reserve,
                )
                post_verification_budget = verification_budget
                remaining = run_cost.remaining_cost_usd
                if remaining is not None:
                    configured = (
                        post_verification_budget.max_cost_usd
                        if post_verification_budget is not None
                        else None
                    )
                    post_verification_budget = VerificationBudget(
                        max_tokens=(
                            post_verification_budget.max_tokens
                            if post_verification_budget is not None
                            else None
                        ),
                        max_cost_usd=(
                            min(remaining, configured)
                            if configured is not None
                            else remaining
                        ),
                    )
                post_cost_estimator = verification_cost_estimator
                if post_cost_estimator is None and run_cost.configured:
                    post_cost_estimator = (
                        budgeted_post_verification_model.estimate_cost_usd
                    )
                try:
                    post_verification = await verify_attributions(
                        post_attribution.attributions,
                        source_cache=ledger.source_cache,
                        model_client=budgeted_post_verification_model,
                        settings=verification_settings,
                        budget=post_verification_budget,
                        corroboration_targets=post_targets,
                        estimate_input_tokens=verification_input_token_estimator,
                        estimate_cost_usd=post_cost_estimator,
                    )
                except RunCostCapReached as error:
                    stage_records[
                        "post_edit_initial_verification"
                    ] = _scope_record(
                        status=StageExecutionStatus.NOT_RUN,
                        reason=str(error),
                        unit="claim_source_relation",
                        expected_count=len(post_relation_ids),
                        evaluated_count=0,
                        unevaluated_ids=post_relation_ids,
                    )
                else:
                    post_completed_relations = {
                        f"{claim.claim.claim_id}|{relation.source_id}"
                        for claim in post_verification.claims
                        for relation in claim.relations
                        if relation.status is VerificationRecordStatus.COMPLETED
                    }
                    post_unevaluated_relations = tuple(
                        relation_id
                        for relation_id in post_relation_ids
                        if relation_id not in post_completed_relations
                    )
                    stage_records[
                        "post_edit_initial_verification"
                    ] = _scope_record(
                        status=(
                            StageExecutionStatus.COMPLETE
                            if not post_unevaluated_relations
                            else StageExecutionStatus.PARTIAL
                        ),
                        reason=(
                            "every edited claim-source relation completed "
                            "verification"
                            if not post_unevaluated_relations
                            else "some edited claim-source relations did not "
                            "complete verification"
                        ),
                        unit="claim_source_relation",
                        expected_count=len(post_relation_ids),
                        evaluated_count=(
                            len(post_relation_ids)
                            - len(post_unevaluated_relations)
                        ),
                        unevaluated_ids=post_unevaluated_relations,
                    )
                    additional_usage[
                        "post_edit_initial_verification"
                    ] = UsageRecord(
                        token_count=post_verification.total_tokens,
                        cost_usd=post_verification.total_cost_usd,
                    )
                    tail_reserve.observe_stage(
                        "post_edit_initial_verification",
                        work_units=(
                            TailWorkUnit(
                                stage="post_edit_initial_verification",
                                unit="claim_source_relation",
                                count=len(post_relation_ids),
                            ),
                        ),
                        token_count=post_verification.total_tokens,
                        cost_usd=post_verification.total_cost_usd,
                    )
            else:
                stage_records["post_edit_initial_verification"] = _scope_record(
                    status=StageExecutionStatus.NOT_RUN,
                    reason=(
                        "not run because edited attribution did not complete; "
                        "no verification denominator was manufactured"
                    ),
                )

            # Checklist reconciliation is about the edited prose and its claim
            # anchors, not whether every source relation produced a verdict. A
            # partial verification registry must remain publishable as a partial
            # bundle instead of suppressing this independent audit.
            if post_claim_stage_complete and post_claims is not None:
                item_ids = tuple(
                    item.item_id for item in loop_result.checklist.items
                )
                budgeted_post_reconciliation_model = run_cost.wrap(
                    reconciliation_model,
                    stage="post_edit_checklist_reconciliation",
                    tail_reserve_controller=tail_reserve,
                )
                try:
                    post_reconciliation = await reconcile_checklist_report(
                        proposed_draft,
                        loop_result.checklist,
                        blocks=post_claims.blocks,
                        claims=post_claims.claims,
                        model_client=budgeted_post_reconciliation_model,
                    )
                except RunCostCapReached as error:
                    stage_records[
                        "post_edit_checklist_reconciliation"
                    ] = _scope_record(
                        status=StageExecutionStatus.NOT_RUN,
                        reason=str(error),
                        unit="checklist_item",
                        expected_count=len(item_ids),
                        evaluated_count=0,
                        unevaluated_ids=item_ids,
                    )
                else:
                    summary = post_reconciliation.summary
                    stage_records[
                        "post_edit_checklist_reconciliation"
                    ] = _scope_record(
                        status=(
                            StageExecutionStatus.COMPLETE
                            if summary.assessment_failed_items == 0
                            else StageExecutionStatus.PARTIAL
                        ),
                        reason=(
                            "every checklist item was reconciled against the "
                            "edited draft"
                            if summary.assessment_failed_items == 0
                            else "some checklist items could not be reconciled "
                            "against the edited draft"
                        ),
                        unit="checklist_item",
                        expected_count=summary.total_items,
                        evaluated_count=summary.assessed_items,
                        unevaluated_ids=summary.assessment_failed_item_ids,
                    )
                    additional_usage[
                        "post_edit_checklist_reconciliation"
                    ] = UsageRecord(
                        token_count=post_reconciliation.total_tokens,
                        cost_usd=post_reconciliation.total_cost_usd,
                    )
                    tail_reserve.observe_stage(
                        "post_edit_checklist_reconciliation",
                        work_units=(
                            TailWorkUnit(
                                stage="post_edit_checklist_reconciliation",
                                unit="checklist_item",
                                count=summary.total_items,
                            ),
                        ),
                        token_count=post_reconciliation.total_tokens,
                        cost_usd=post_reconciliation.total_cost_usd,
                    )
            else:
                stage_records[
                    "post_edit_checklist_reconciliation"
                ] = _scope_record(
                    status=StageExecutionStatus.NOT_RUN,
                    reason=(
                        "not run because the edited claim registry did not "
                        "complete"
                    ),
                )

            # Committing bytes and declaring them publication-eligible are two
            # separate gates. A complete claim registry plus complete attribution
            # and an actual (possibly partial) verification result bind every
            # rendered status to the edited draft. Global stage completeness is
            # still assessed below and keeps pipeline_complete false whenever
            # any original or re-audit work is incomplete.
            post_edit_registry_coherent = (
                post_claim_stage_complete
                and post_attr_complete
                and post_claims is not None
                and post_attribution is not None
                and post_verification is not None
                and post_reconciliation is not None
            )
            if post_edit_registry_coherent:
                report = report.model_copy(
                    update={"canonical_draft": proposed_draft}
                )
                editorial_revision = editorial_revision.model_copy(
                    update={"committed_after_reaudit": True}
                )
                claim_decomposition = post_claims
                checklist_report_reconciliation = post_reconciliation
                attribution = post_attribution
                verification = post_verification
                # The first diagnostic remains nested under pre_edit_evidence.
                # The top-level payload belongs only to the edited claim IDs
                # and is null when that independent rerun did not execute.
                evaluative_diagnostics = post_edit_evaluative_diagnostics

        if (
            editorial_revision is not None
            and not editorial_revision.requires_reaudit
        ):
            tail_reserve.checkpoint(
                TailCheckpointName.MANDATORY_TAIL_COMPLETE,
                work_units=(),
                estimated_remaining_cost_usd=0.0,
                estimate_complete=True,
            )
        elif (
            editorial_revision is not None
            and editorial_revision.committed_after_reaudit
        ):
            tail_reserve.checkpoint(
                TailCheckpointName.MANDATORY_TAIL_COMPLETE,
                work_units=(),
                estimated_remaining_cost_usd=0.0,
                estimate_complete=True,
            )

    domain_proxy_concentration = audit_domain_proxy_concentration(
        verification,
        blocks=claim_decomposition.blocks,
        reconciliation=checklist_report_reconciliation,
        source_cache=ledger.source_cache,
        notes=ledger.notes,
    )
    # Snapshot the cost ledger here rather than after rendering: the report
    # itself has to disclose that a ceiling cut the run short, so the
    # diagnostic must exist before the text is built. Nothing between this
    # point and the audit write calls a model, so the numbers are final.
    run_cost_audit = run_cost.audit()

    # Counts of work still owed, used only to decide whether more budget has
    # anything left to buy. NO_CANDIDATE_SOURCE is excluded on purpose: that is
    # attribution running to completion and finding nothing, which is a result,
    # not an unfinished task. Counting it would report a finished run as
    # partial and imply money could buy a conclusion it already reached.
    unattributed_claims = sum(
        1
        for record in attribution.attributions
        if record.status is AttributionStatus.ATTRIBUTION_ERROR
    )
    # Only budget-blocked relations count. A model error is also unverified,
    # but money does not fix it, and this number exists to answer whether money
    # would buy anything.
    unverified_relations = sum(
        1
        for claim in verification.claims
        for relation in claim.relations
        if relation.status
        is VerificationRecordStatus.VERIFICATION_NOT_RUN_BUDGET
    )
    stop_diagnostic = build_run_stop_diagnostic(
        loop_result=loop_result,
        run_cost_audit=run_cost_audit,
        unattributed_claims=unattributed_claims,
        unverified_relations=unverified_relations,
        evidence_gap_plan_unexecuted=(
            evidence_gap.stop_reason
            is EvidenceGapStopReason.BUDGET_EXHAUSTED
        ),
        disagreement_plan_unexecuted=(
            disagreement.stop_reason
            is DisagreementStopReason.BUDGET_EXHAUSTED
        ),
        report_written=bool(report.canonical_draft),
    )

    rendered_report = render_verified_report(
        report.canonical_draft,
        verification,
        settled_without_located_evidence=(
            settled_without_located_evidence
        ),
        settled_without_located_evidence_item_ids=(
            settled_without_located_evidence_item_ids
        ),
        rejected_exhausted_without_collection_attempt=(
            rejected_exhausted_without_collection_attempt
        ),
        rejected_exhausted_without_collection_attempt_item_ids=(
            rejected_exhausted_without_collection_attempt_item_ids
        ),
        accepted_exhausted_without_collection_attempt=(
            accepted_exhausted_without_collection_attempt
        ),
        accepted_exhausted_without_collection_attempt_item_ids=(
            accepted_exhausted_without_collection_attempt_item_ids
        ),
        accepted_exhausted_attempt_unknown_legacy=(
            accepted_exhausted_attempt_unknown_legacy
        ),
        accepted_exhausted_attempt_unknown_legacy_item_ids=(
            accepted_exhausted_attempt_unknown_legacy_item_ids
        ),
        exhausted_with_unread_candidates=exhausted_with_unread_candidates,
        exhausted_with_unread_candidates_item_ids=(
            exhausted_with_unread_candidates_item_ids
        ),
        registry_coverage=claim_decomposition.registry_coverage,
        checklist_coverage=checklist_report_reconciliation.summary,
        domain_proxy_concentration=domain_proxy_concentration,
        disagreement_attempted_count=len(
            disagreement.disagreement_search_attempted
        ),
        initial_collection_snapshot=initial_collection_snapshot,
        stop_diagnostic=stop_diagnostic,
        run_id=normalized_run_id,
        report_filename=report_filename,
        sources_filename=sources_filename,
        audit_filename=audit_filename,
        reader_report_style=reader_report_style,
    )
    stage_records["deterministic_rendering"] = _scope_record(
        status=StageExecutionStatus.COMPLETE,
        reason="deterministic report and evidence package rendering completed",
        unit="report_bundle",
        expected_count=1,
        evaluated_count=1,
    )
    # A stage whose scope is empty only because its upstream was cut off has
    # not completed anything; letting that stand would reintroduce 0/0.
    stage_records = demote_vacuous_completions(stage_records)
    mandatory_pipeline_stages = MANDATORY_PIPELINE_STAGES
    if editor_model is not None:
        mandatory_pipeline_stages = (
            *mandatory_pipeline_stages,
            "audit_editing",
            *post_edit_required_stages,
        )
    post_draft_execution = publication_audit(
        stage_records,
        mandatory_stages=mandatory_pipeline_stages,
    )
    if (
        post_draft_execution.quality_review_passed is None
        and reader_report_style is ReaderReportStyle.AUDIT_ANNOTATED
    ):
        quality_warning = (
            "> **质量审核状态：未完成独立质量审核。** "
            "`pipeline_complete` 只表示规定流程已执行，"
            "不表示事实正确、来源充分或适合公开发布。"
        )
        rendered_report = rendered_report.model_copy(
            update={"markdown": quality_warning + "\n" + rendered_report.markdown}
        )
    if not post_draft_execution.pipeline_complete:
        incomplete_warning = (
            "> **不完整运行产物：证据流程未完整覆盖；"
            "`pipeline_complete=false`。** "
            f"{post_draft_execution.pipeline_completion_reason}。"
            "未评估工作不得解释为无来源、不支持或零覆盖。"
        )
        rendered_report = rendered_report.model_copy(
            update={
                "markdown": incomplete_warning
                + "\n"
                + rendered_report.markdown,
            }
        )
        # model_copy does not validate, so a raw string here silently replaces
        # the enum and only fails much later when the audit reads .value --
        # after the whole pipeline has run and been paid for.
        stop_diagnostic = stop_diagnostic.model_copy(
            update={"completion_status": CompletionStatus.PARTIAL}
        )

    writing_usage = UsageRecord(
        token_count=report.token_count,
        cost_usd=report.cost_usd,
    )
    decomposition_attribution_usage = UsageRecord(
        token_count=(
            pre_edit_claim_decomposition.total_tokens
            + (
                pre_edit_evaluative_diagnostics.total_tokens
                if pre_edit_evaluative_diagnostics is not None
                else 0
            )
            + initial_attribution.total_tokens
        ),
        cost_usd=(
            pre_edit_claim_decomposition.total_cost_usd
            + (
                pre_edit_evaluative_diagnostics.total_cost_usd
                if pre_edit_evaluative_diagnostics is not None
                else 0.0
            )
            + initial_attribution.total_cost_usd
        ),
    )
    verification_usage = UsageRecord(
        token_count=initial_verification.total_tokens,
        cost_usd=initial_verification.total_cost_usd,
    )
    reconciliation_usage = UsageRecord(
        token_count=pre_edit_reconciliation.total_tokens,
        cost_usd=pre_edit_reconciliation.total_cost_usd,
    )
    evidence_gap_usage = UsageRecord(
        token_count=evidence_gap.total_tokens,
        cost_usd=evidence_gap.total_cost_usd,
    )
    disagreement_usage = UsageRecord(
        token_count=disagreement.total_tokens,
        cost_usd=disagreement.total_cost_usd,
    )
    usage, usage_audit = _usage_payload(
        checklist_usage=checklist_usage,
        collection_usage=collection_usage,
        writing_usage=writing_usage,
        decomposition_attribution_usage=(
            decomposition_attribution_usage
        ),
        reconciliation_usage=reconciliation_usage,
        verification_usage=verification_usage,
        evidence_gap_usage=evidence_gap_usage,
        disagreement_usage=disagreement_usage,
        additional_stages=additional_usage,
    )

    destination = Path(output_dir)
    run_directory = destination / normalized_run_id
    report_path = run_directory / report_filename
    sources_path = run_directory / sources_filename
    audit_path = run_directory / audit_filename
    report_sha256 = hashlib.sha256(
        rendered_report.markdown.encode("utf-8")
    ).hexdigest()
    audit = {
        "run_id": normalized_run_id,
        "topic": loop_result.checklist.topic,
        "ledger": ledger.to_audit_dict(),
        "checklist": loop_result.checklist.model_dump(mode="json"),
        "stop": {
            "reason": loop_result.stop_reason.value,
            "detail": loop_result.stop_detail,
            "open_item_ids": list(loop_result.open_item_ids),
            # Protocol completion only. It is not derived from how much of the
            # budget was spent, so a run that finished its work under a cap and
            # a run that was cut off by one are never conflated here.
            "is_success": loop_result.is_success,
            "diagnostic": stop_diagnostic.model_dump(mode="json"),
        },
        "collection_summary": {
            "initial_collection_snapshot": (
                initial_collection_snapshot.model_dump(mode="json")
            ),
            "settled_without_located_evidence": (
                settled_without_located_evidence
            ),
            "settled_without_located_evidence_item_ids": list(
                settled_without_located_evidence_item_ids
            ),
            "rejected_exhausted_without_collection_attempt": (
                rejected_exhausted_without_collection_attempt
            ),
            "rejected_exhausted_without_collection_attempt_item_ids": list(
                rejected_exhausted_without_collection_attempt_item_ids
            ),
            "accepted_exhausted_without_collection_attempt": (
                accepted_exhausted_without_collection_attempt
            ),
            "accepted_exhausted_without_collection_attempt_item_ids": list(
                accepted_exhausted_without_collection_attempt_item_ids
            ),
            "accepted_exhausted_attempt_unknown_legacy": (
                accepted_exhausted_attempt_unknown_legacy
            ),
            "accepted_exhausted_attempt_unknown_legacy_item_ids": list(
                accepted_exhausted_attempt_unknown_legacy_item_ids
            ),
            "exhausted_with_unread_candidates": (
                exhausted_with_unread_candidates
            ),
            "exhausted_with_unread_candidates_item_ids": list(
                exhausted_with_unread_candidates_item_ids
            ),
            "writing_reserve": {
                "tokens": active_budget.writing_token_reserve,
                "cost_usd": active_budget.writing_cost_reserve_usd,
            },
            "quote_quality": collection_quote_quality,
            # Collection protects this allocation, but the first version does
            # not yet estimate assembled-notes input before the writing call.
            # Keep the gap explicit until stage admission is implemented.
            "known_gaps": ["writing_input_budget_preflight_not_enforced"],
        },
        "posthoc_evidence": {
            "stage_execution": post_draft_execution.model_dump(mode="json"),
            "claim_decomposition": claim_decomposition.model_dump(mode="json"),
            "pre_edit_evidence": (
                {
                    "canonical_draft_sha256": hashlib.sha256(
                        pre_edit_draft.encode("utf-8")
                    ).hexdigest(),
                    "claim_decomposition": (
                        pre_edit_claim_decomposition.model_dump(mode="json")
                    ),
                    "evaluative_claim_diagnostics": (
                        pre_edit_evaluative_diagnostics.model_dump(mode="json")
                        if pre_edit_evaluative_diagnostics is not None
                        else None
                    ),
                    "checklist_report_reconciliation": (
                        pre_edit_reconciliation.model_dump(mode="json")
                    ),
                    "attribution": pre_edit_attribution.model_dump(mode="json"),
                    "verification": pre_edit_verification.model_dump(mode="json"),
                    "evidence_gap": evidence_gap.model_dump(mode="json"),
                    "disagreement": disagreement.model_dump(mode="json"),
                    "recovery_triage": (
                        recovery_triage.model_dump(mode="json")
                        if recovery_triage is not None
                        else None
                    ),
                    "evidence_recovery": (
                        evidence_recovery.model_dump(mode="json")
                        if evidence_recovery is not None
                        else None
                    ),
                }
                if editor_model is not None
                else None
            ),
            "editorial_revision": (
                editorial_revision.model_dump(mode="json")
                if editorial_revision is not None
                else None
            ),
            "editorial_admission": (
                editorial_admission.model_dump(mode="json")
                if editorial_admission is not None
                else None
            ),
            "recovery_triage": (
                recovery_triage.model_dump(mode="json")
                if recovery_triage is not None
                else None
            ),
            "evidence_recovery": (
                evidence_recovery.model_dump(mode="json")
                if evidence_recovery is not None
                else None
            ),
            "evaluative_claim_diagnostics": (
                evaluative_diagnostics.model_dump(mode="json")
                if evaluative_diagnostics is not None
                else None
            ),
            "post_edit_evaluative_claim_diagnostics": (
                post_edit_evaluative_diagnostics.model_dump(mode="json")
                if post_edit_evaluative_diagnostics is not None
                else None
            ),
            "checklist_report_reconciliation": (
                checklist_report_reconciliation.model_dump(mode="json")
            ),
            "initial_attribution": (
                attribution.model_dump(mode="json")
                if editorial_revision is not None
                and editorial_revision.committed_after_reaudit
                else initial_attribution.model_dump(mode="json")
            ),
            "initial_verification": (
                verification.model_dump(mode="json")
                if editorial_revision is not None
                and editorial_revision.committed_after_reaudit
                else initial_verification.model_dump(mode="json")
            ),
            "evidence_gap": (
                None
                if editorial_revision is not None
                and editorial_revision.committed_after_reaudit
                else evidence_gap.model_dump(mode="json")
            ),
            "disagreement": (
                None
                if editorial_revision is not None
                and editorial_revision.committed_after_reaudit
                else disagreement.model_dump(mode="json")
            ),
            "posthoc_retrieval_budget": posthoc_budget_audit.model_dump(
                mode="json"
            ),
            "attribution": attribution.model_dump(mode="json"),
            "verification": verification.model_dump(mode="json"),
            "domain_proxy_concentration": (
                domain_proxy_concentration.model_dump(mode="json")
            ),
            "rendering": rendered_report.model_dump(
                mode="json",
                exclude={"markdown", "sources_markdown"},
            ),
            "corroboration_target_for_external_claims": (
                corroboration_target_for_external_claims
            ),
        },
        "usage": usage_audit,
        "run_cost_budget": run_cost_audit.model_dump(mode="json"),
        "evidence_tail_reserve": tail_reserve.audit().model_dump(mode="json"),
        "completion_status": stop_diagnostic.completion_status.value,
        "pipeline_complete": post_draft_execution.pipeline_complete,
        "quality_review_passed": post_draft_execution.quality_review_passed,
        "budget_decision_signal": (
            stop_diagnostic.budget_decision_signal.value
        ),
        "run_cost_limit_status": (
            "configured_run_level_admission_limit"
            if run_cost_audit.configured
            else "no_run_level_cost_limit"
        ),
        "models": dict(model_names or {}),
        "canonical_draft": report.canonical_draft,
        "original_canonical_draft": pre_edit_draft,
        "artifacts": {
            "directory": normalized_run_id,
            "report": report_path.name,
            "report_sha256": report_sha256,
            "sources": sources_path.name,
            "sources_sha256": rendered_report.sources_sha256,
            "audit": audit_path.name,
            "commit_marker": audit_path.name,
            "bundle_complete": True,
            "pipeline_complete": post_draft_execution.pipeline_complete,
            "quality_review_passed": (
                post_draft_execution.quality_review_passed
            ),
            "staging_write_order": ["sources", "report", "audit"],
            "publication_order": ["directory"],
        },
    }

    audit_json = (
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    _publish_artifact_bundle(
        destination=destination,
        report_path=report_path,
        sources_path=sources_path,
        audit_path=audit_path,
        report_markdown=rendered_report.markdown,
        sources_markdown=rendered_report.sources_markdown,
        audit_json=audit_json,
    )
    return HarnessRunResult(
        run_id=normalized_run_id,
        report_path=report_path,
        sources_path=sources_path,
        audit_path=audit_path,
        report=report,
        rendered_report=rendered_report,
        loop_result=loop_result,
        claim_decomposition=claim_decomposition,
        evaluative_diagnostics=evaluative_diagnostics,
        checklist_report_reconciliation=(
            checklist_report_reconciliation
        ),
        domain_proxy_concentration=domain_proxy_concentration,
        attribution=attribution,
        verification=verification,
        evidence_gap=evidence_gap,
        disagreement=disagreement,
        recovery_triage=recovery_triage,
        evidence_recovery=evidence_recovery,
        editorial_admission=editorial_admission,
        editorial_revision=editorial_revision,
        posthoc_retrieval_budget=posthoc_budget_audit,
        run_cost_budget=run_cost_audit,
        stop_diagnostic=stop_diagnostic,
        post_draft_execution=post_draft_execution,
        evidence_tail_reserve=tail_reserve.audit(),
        pipeline_complete=post_draft_execution.pipeline_complete,
        quality_review_passed=post_draft_execution.quality_review_passed,
        usage=usage,
    )
