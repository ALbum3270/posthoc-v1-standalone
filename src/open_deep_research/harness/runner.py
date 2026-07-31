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
    RunStopDiagnostic,
    build_run_stop_diagnostic,
)
from open_deep_research.harness.claims import (
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
    RenderedReport,
    render_verified_report,
)
from open_deep_research.harness.reconcile import (
    ChecklistReportReconciliation,
    ReconciliationModelClient,
    reconcile_checklist_report,
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
    PostDraftExecutionAudit,
    StageExecutionRecord,
    StageExecutionStatus,
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
    posthoc_retrieval_budget: PosthocRetrievalBudgetAudit | None
    run_cost_budget: RunCostBudgetAudit
    stop_diagnostic: RunStopDiagnostic
    post_draft_execution: PostDraftExecutionAudit
    evidence_tail_reserve: EvidenceTailReserveAudit
    publication_eligible: bool
    usage: dict[str, UsageRecord]


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
        "`publication_eligible=false`。** "
        f"后置阶段 `{failed_stage}` 因运行成本上限停止。"
        "未执行的工作没有被解释为无来源、不支持、零覆盖或零候选；"
        f"精确阶段状态与成本诊断见 [{audit_filename}]({audit_filename})。"
    )
    body = _citation_free_partial_draft(canonical_draft)
    report = warning + "\n\n" + body + "\n"
    sources = (
        "# 不完整证据包\n\n"
        f"- Run ID：`{run_id}`\n"
        "- Publication eligible：`false`\n"
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
    return StageExecutionRecord(
        status=status,
        reason=reason,
        expected_scope=expected,
        evaluated_scope=evaluated,
        unevaluated_ids=unevaluated_ids,
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
    if post_draft_execution.publication_eligible:
        raise AssertionError("a cost-cutoff checkpoint cannot be publishable")

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
        "publication_eligible": False,
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
            "publication_eligible": False,
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
        posthoc_retrieval_budget=None,
        run_cost_budget=run_cost_audit,
        stop_diagnostic=stop_diagnostic,
        post_draft_execution=post_draft_execution,
        evidence_tail_reserve=tail_audit,
        publication_eligible=False,
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
    budget: LoopBudget | None = None,
    loop_settings: LoopSettings | None = None,
    claim_settings: ClaimDecompositionSettings | None = None,
    evaluative_diagnostic_settings: (
        EvaluativeDiagnosticSettings | None
    ) = None,
    attribution_settings: AttributionSettings | None = None,
    verification_settings: VerificationSettings | None = None,
    verification_budget: VerificationBudget | None = None,
    evidence_gap_budget: EvidenceGapBudget | None = None,
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
) -> HarnessRunResult:
    """Run collection, drafting, post-hoc evidence, and artifact rendering."""

    normalized_run_id = _normalize_run_id(run_id)
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
        protected_reserve_usd=evidence_tail_reserve_usd,
    )
    report = await write_report(
        assembled,
        model_client=budgeted_write_model,
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

    parsed_blocks = parse_markdown_blocks(report.canonical_draft)
    claim_batch_size = (
        claim_settings.batch_size
        if claim_settings is not None
        else ClaimDecompositionSettings().batch_size
    )
    selection_prompts = tuple(
        build_selection_prompt(parsed_blocks[index : index + claim_batch_size])
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
        # Count what the pass actually assessed, never what it was asked to
        # assess. These were once the same expression, so a pass whose calls
        # were all refused by the admission layer still returned a record and
        # was recorded as "87 of 87 evaluated". Returning a record is not
        # doing the work.
        evaluative_expected = sum(
            claim.citation_requirement is CitationRequirement.EXTERNAL
            for claim in claim_decomposition.claims
        )
        # A diagnostic_failed entry is a record that the pass could not assess
        # this claim -- code writes it when the model omitted, duplicated, or
        # malformed the entry. Counting it as assessed repeats the original
        # error one layer down: having a record is not having done the work.
        evaluative_assessed_ids = {
            assessment.claim_id
            for assessment in evaluative_diagnostics.assessments
            if assessment.status is not EvaluativeDiagnosticStatus.DIAGNOSTIC_FAILED
        }
        evaluative_unassessed = tuple(
            claim.claim_id
            for claim in claim_decomposition.claims
            if claim.citation_requirement is CitationRequirement.EXTERNAL
            and claim.claim_id not in evaluative_assessed_ids
        )
        stage_records["evaluative_diagnostics"] = _scope_record(
            status=(
                StageExecutionStatus.COMPLETE
                if not evaluative_unassessed
                else StageExecutionStatus.PARTIAL
                if evaluative_assessed_ids
                else StageExecutionStatus.NOT_RUN
            ),
            reason=(
                "advisory diagnostic assessed every external claim"
                if not evaluative_unassessed
                else "advisory diagnostic pass did not assess "
                f"{len(evaluative_unassessed)} of {evaluative_expected} "
                "external claims"
            ),
            unit="external_claim",
            expected_count=evaluative_expected,
            evaluated_count=len(evaluative_assessed_ids),
            unevaluated_ids=evaluative_unassessed,
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
    attributed_ids = {
        record.claim.claim_id for record in initial_attribution.attributions
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
    tail_reserve.checkpoint(
        TailCheckpointName.MANDATORY_TAIL_COMPLETE,
        work_units=(),
        estimated_remaining_cost_usd=0.0,
        estimate_complete=True,
    )

    effective_evidence_gap_budget = evidence_gap_budget
    if (
        evidence_gap_budget is not None
        and posthoc_retrieval_budget is not None
    ):
        effective_evidence_gap_budget = evidence_gap_budget.model_copy(
            update={
                "max_tokens": min(
                    evidence_gap_budget.max_tokens,
                    posthoc_retrieval_budget.max_tokens,
                ),
                "max_cost_usd": min(
                    evidence_gap_budget.max_cost_usd,
                    posthoc_retrieval_budget.max_cost_usd,
                ),
            }
        )
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
        stage_records["evidence_gap"] = _scope_record(
            status=(
                StageExecutionStatus.COMPLETE
                if evidence_gap.stop_reason
                in {
                    EvidenceGapStopReason.COMPLETED,
                    EvidenceGapStopReason.NO_TARGETS,
                }
                else (
                    StageExecutionStatus.PARTIAL
                    if evidence_gap.stop_reason
                    is EvidenceGapStopReason.BUDGET_EXHAUSTED
                    else StageExecutionStatus.FAILED
                )
            ),
            reason=evidence_gap.stop_detail,
            unit="target_claim",
            expected_count=len(evidence_gap.target_claim_ids),
            evaluated_count=(
                len(evidence_gap.target_claim_ids)
                if evidence_gap.stop_reason
                in {
                    EvidenceGapStopReason.COMPLETED,
                    EvidenceGapStopReason.NO_TARGETS,
                }
                else 0
            ),
            unevaluated_ids=(
                ()
                if evidence_gap.stop_reason
                in {
                    EvidenceGapStopReason.COMPLETED,
                    EvidenceGapStopReason.NO_TARGETS,
                }
                else evidence_gap.target_claim_ids
            ),
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
        stage_records["disagreement"] = _scope_record(
            status=(
                StageExecutionStatus.COMPLETE
                if disagreement.stop_reason
                in {
                    DisagreementStopReason.COMPLETED,
                    DisagreementStopReason.NO_ELIGIBLE_CLAIMS,
                    DisagreementStopReason.NO_SELECTION,
                }
                else (
                    StageExecutionStatus.PARTIAL
                    if disagreement.stop_reason
                    is DisagreementStopReason.BUDGET_EXHAUSTED
                    else StageExecutionStatus.FAILED
                )
            ),
            reason=disagreement.stop_detail,
            unit="selected_claim",
            expected_count=len(disagreement.selected_claims),
            evaluated_count=len(disagreement.disagreement_search_attempted),
            unevaluated_ids=tuple(
                selection.claim_id
                for selection in disagreement.selected_claims
                if selection.claim_id
                not in {
                    attempt.claim_id
                    for attempt in disagreement.disagreement_search_attempted
                }
            ),
        )
    attribution = disagreement.final_attribution
    verification = disagreement.final_verification
    posthoc_budget_audit = shared_posthoc_budget_audit(
        budget=posthoc_retrieval_budget,
        evidence_gap_tokens=evidence_gap.total_tokens,
        evidence_gap_cost_usd=evidence_gap.total_cost_usd,
        disagreement_tokens=disagreement.total_tokens,
        disagreement_cost_usd=disagreement.total_cost_usd,
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
    )
    stage_records["deterministic_rendering"] = _scope_record(
        status=StageExecutionStatus.COMPLETE,
        reason="deterministic report and evidence package rendering completed",
        unit="report_bundle",
        expected_count=1,
        evaluated_count=1,
    )
    post_draft_execution = publication_audit(stage_records)
    if not post_draft_execution.publication_eligible:
        incomplete_warning = (
            "> **不完整运行产物：证据流程未完整覆盖；"
            "`publication_eligible=false`。** "
            f"{post_draft_execution.publication_reason}。"
            "未评估工作不得解释为无来源、不支持或零覆盖。"
        )
        rendered_report = rendered_report.model_copy(
            update={
                "markdown": incomplete_warning
                + "\n"
                + rendered_report.markdown,
            }
        )
        stop_diagnostic = stop_diagnostic.model_copy(
            update={"completion_status": "partial"}
        )

    writing_usage = UsageRecord(
        token_count=report.token_count,
        cost_usd=report.cost_usd,
    )
    decomposition_attribution_usage = UsageRecord(
        token_count=(
            claim_decomposition.total_tokens
            + (
                evaluative_diagnostics.total_tokens
                if evaluative_diagnostics is not None
                else 0
            )
            + initial_attribution.total_tokens
        ),
        cost_usd=(
            claim_decomposition.total_cost_usd
            + (
                evaluative_diagnostics.total_cost_usd
                if evaluative_diagnostics is not None
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
        token_count=checklist_report_reconciliation.total_tokens,
        cost_usd=checklist_report_reconciliation.total_cost_usd,
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
            "evaluative_claim_diagnostics": (
                evaluative_diagnostics.model_dump(mode="json")
                if evaluative_diagnostics is not None
                else None
            ),
            "checklist_report_reconciliation": (
                checklist_report_reconciliation.model_dump(mode="json")
            ),
            "initial_attribution": initial_attribution.model_dump(mode="json"),
            "initial_verification": initial_verification.model_dump(mode="json"),
            "evidence_gap": evidence_gap.model_dump(mode="json"),
            "disagreement": disagreement.model_dump(mode="json"),
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
        "publication_eligible": post_draft_execution.publication_eligible,
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
        "artifacts": {
            "directory": normalized_run_id,
            "report": report_path.name,
            "report_sha256": report_sha256,
            "sources": sources_path.name,
            "sources_sha256": rendered_report.sources_sha256,
            "audit": audit_path.name,
            "commit_marker": audit_path.name,
            "bundle_complete": True,
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
        posthoc_retrieval_budget=posthoc_budget_audit,
        run_cost_budget=run_cost_audit,
        stop_diagnostic=stop_diagnostic,
        post_draft_execution=post_draft_execution,
        evidence_tail_reserve=tail_reserve.audit(),
        publication_eligible=post_draft_execution.publication_eligible,
        usage=usage,
    )
