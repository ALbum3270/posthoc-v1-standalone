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
    AttributionStopReason,
    attribute_claims,
)
from open_deep_research.harness.checklist import (
    ChecklistModelClient,
    generate_checklist,
)
from open_deep_research.harness.claims import (
    CitationRequirement,
    ClaimDecompositionResult,
    ClaimDecompositionSettings,
    ClaimModelClient,
    decompose_claims,
)
from open_deep_research.harness.concentration import (
    DomainProxyConcentrationAudit,
    audit_domain_proxy_concentration,
)
from open_deep_research.harness.disagreement import (
    DisagreementBudget,
    DisagreementResult,
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
    VerificationResult,
    VerificationSettings,
    verify_attributions,
)
from open_deep_research.harness.write import (
    ReportDraft,
    WriteModelClient,
    write_report,
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
    rendered_report: RenderedReport
    loop_result: LoopResult
    claim_decomposition: ClaimDecompositionResult
    evaluative_diagnostics: EvaluativeDiagnosticResult
    checklist_report_reconciliation: ChecklistReportReconciliation
    domain_proxy_concentration: DomainProxyConcentrationAudit
    attribution: AttributionResult
    verification: VerificationResult
    evidence_gap: EvidenceGapResult
    disagreement: DisagreementResult
    posthoc_retrieval_budget: PosthocRetrievalBudgetAudit
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
    """Stage all files and publish the audit commit marker last.

    A flat set of files cannot be renamed atomically as one operation. The
    audit file is therefore the commit marker: consumers must treat a run
    without it as incomplete. Best-effort rollback removes already published
    siblings if a later rename fails.
    """

    destination.mkdir(parents=True, exist_ok=True)
    final_paths = (sources_path, report_path, audit_path)
    existing = tuple(path for path in final_paths if path.exists())
    if existing:
        raise FileExistsError(
            "refusing to overwrite an existing artifact bundle: "
            + ", ".join(path.name for path in existing)
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{report_path.stem}-staging-",
            dir=destination,
        )
    )
    staged_sources = staging / sources_path.name
    staged_report = staging / report_path.name
    staged_audit = staging / audit_path.name
    published: list[Path] = []
    try:
        # Required construction and publication order: evidence first, then
        # its digest-bearing report, and the audit commit marker last.
        staged_sources.write_text(sources_markdown, encoding="utf-8")
        staged_report.write_text(report_markdown, encoding="utf-8")
        staged_audit.write_text(audit_json, encoding="utf-8")
        for staged, final in (
            (staged_sources, sources_path),
            (staged_report, report_path),
            (staged_audit, audit_path),
        ):
            os.replace(staged, final)
            published.append(final)
    except BaseException:
        for path in reversed(published):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                # The absent audit commit marker still exposes an incomplete
                # bundle if process-level cleanup cannot finish.
                pass
        raise
    finally:
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
    report_filename = f"{normalized_run_id}.md"
    sources_filename = f"{normalized_run_id}.sources.md"
    audit_filename = f"{normalized_run_id}.json"
    ledger = ResearchLedger(research_id=normalized_run_id, topic=topic.strip())
    active_budget = budget or LoopBudget()
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

    checklist = await generate_checklist(topic, model_client=checklist_model)
    checklist_usage = _usage_from_model(checklist_model)
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
    collection_quote_quality = quote_quality_metrics(ledger.notes)
    settled_without_located_evidence = (
        ledger.settled_without_located_evidence
    )
    settled_without_located_evidence_item_ids = (
        ledger.settled_without_located_evidence_item_ids
    )
    assembled = assemble_notes(loop_result.checklist, ledger.notes)
    report = await write_report(assembled, model_client=write_model)
    claim_decomposition = await decompose_claims(
        report.canonical_draft,
        model_client=claim_model,
        settings=claim_settings,
    )
    evaluative_diagnostics = (
        await diagnose_underspecified_evaluative_claims(
            claim_decomposition.claims,
            model_client=claim_model,
            settings=evaluative_diagnostic_settings,
        )
    )
    checklist_report_reconciliation = await reconcile_checklist_report(
        report.canonical_draft,
        loop_result.checklist,
        blocks=claim_decomposition.blocks,
        claims=claim_decomposition.claims,
        model_client=reconciliation_model,
    )
    if claim_decomposition.claims:
        initial_attribution = await attribute_claims(
            claim_decomposition.claims,
            blocks=claim_decomposition.blocks,
            notes=ledger.notes,
            model_client=attribution_model,
            settings=attribution_settings,
        )
    else:
        initial_attribution = AttributionResult(
            attributions=(),
            stop_reason=AttributionStopReason.COMPLETED,
        )
    corroboration_targets = {
        claim.claim_id: corroboration_target_for_external_claims
        for claim in claim_decomposition.claims
        if claim.citation_requirement == CitationRequirement.EXTERNAL
    }
    initial_verification = await verify_attributions(
        initial_attribution.attributions,
        source_cache=ledger.source_cache,
        model_client=verification_model,
        settings=verification_settings,
        budget=verification_budget,
        corroboration_targets=corroboration_targets,
        estimate_input_tokens=verification_input_token_estimator,
        estimate_cost_usd=verification_cost_estimator,
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
    if effective_evidence_gap_budget is None:
        evidence_gap = EvidenceGapResult(
            stop_reason=EvidenceGapStopReason.DISABLED,
            stop_detail="no independent evidence-gap budget was configured",
            final_attribution=initial_attribution,
            final_verification=initial_verification,
        )
    else:
        evidence_gap = await run_evidence_gap_round(
            canonical_draft=report.canonical_draft,
            checklist=loop_result.checklist,
            blocks=claim_decomposition.blocks,
            ledger=ledger,
            initial_attribution=initial_attribution,
            initial_verification=initial_verification,
            gap_model=decision_model,
            note_model=note_model,
            attribution_model=attribution_model,
            verification_model=verification_model,
            tavily_client=tavily_client,
            budget=effective_evidence_gap_budget,
            attribution_settings=attribution_settings,
            verification_settings=verification_settings,
            corroboration_targets=corroboration_targets,
            estimate_input_tokens=evidence_gap_input_token_estimator,
            estimate_cost_usd=evidence_gap_cost_estimator,
        )
    evidence_gap_attribution = evidence_gap.final_attribution
    evidence_gap_verification = evidence_gap.final_verification
    if disagreement_budget is None:
        disagreement = disabled_disagreement_result(
            evidence_gap_attribution,
            evidence_gap_verification,
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
        disagreement = await run_disagreement_detection(
            canonical_draft=report.canonical_draft,
            checklist=loop_result.checklist,
            blocks=claim_decomposition.blocks,
            ledger=ledger,
            initial_attribution=evidence_gap_attribution,
            initial_verification=evidence_gap_verification,
            selection_model=decision_model,
            note_model=note_model,
            attribution_model=attribution_model,
            verification_model=verification_model,
            tavily_client=tavily_client,
            budget=effective_disagreement_budget,
            attribution_settings=attribution_settings,
            verification_settings=verification_settings,
            corroboration_targets=corroboration_targets,
            estimate_input_tokens=evidence_gap_input_token_estimator,
            estimate_cost_usd=evidence_gap_cost_estimator,
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
    rendered_report = render_verified_report(
        report.canonical_draft,
        verification,
        settled_without_located_evidence=(
            settled_without_located_evidence
        ),
        settled_without_located_evidence_item_ids=(
            settled_without_located_evidence_item_ids
        ),
        registry_coverage=claim_decomposition.registry_coverage,
        checklist_coverage=checklist_report_reconciliation.summary,
        domain_proxy_concentration=domain_proxy_concentration,
        disagreement_attempted_count=len(
            disagreement.disagreement_search_attempted
        ),
        run_id=normalized_run_id,
        report_filename=report_filename,
        sources_filename=sources_filename,
        audit_filename=audit_filename,
    )

    writing_usage = UsageRecord(
        token_count=report.token_count,
        cost_usd=report.cost_usd,
    )
    decomposition_attribution_usage = UsageRecord(
        token_count=(
            claim_decomposition.total_tokens
            + evaluative_diagnostics.total_tokens
            + initial_attribution.total_tokens
        ),
        cost_usd=(
            claim_decomposition.total_cost_usd
            + evaluative_diagnostics.total_cost_usd
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
    report_path = destination / report_filename
    sources_path = destination / sources_filename
    audit_path = destination / audit_filename
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
            "is_success": loop_result.is_success,
        },
        "collection_summary": {
            "settled_without_located_evidence": (
                settled_without_located_evidence
            ),
            "settled_without_located_evidence_item_ids": list(
                settled_without_located_evidence_item_ids
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
            "claim_decomposition": claim_decomposition.model_dump(mode="json"),
            "evaluative_claim_diagnostics": (
                evaluative_diagnostics.model_dump(mode="json")
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
        "models": dict(model_names or {}),
        "canonical_draft": report.canonical_draft,
        "artifacts": {
            "report": report_path.name,
            "report_sha256": report_sha256,
            "sources": sources_path.name,
            "sources_sha256": rendered_report.sources_sha256,
            "audit": audit_path.name,
            "commit_marker": audit_path.name,
            "bundle_complete": True,
            "publication_order": ["sources", "report", "audit"],
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
        usage=usage,
    )
