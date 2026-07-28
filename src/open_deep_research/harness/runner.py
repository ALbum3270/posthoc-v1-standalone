"""End-to-end orchestration and durable harness artifacts."""

from __future__ import annotations

import json
import re
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
from open_deep_research.harness.ledger import ResearchLedger
from open_deep_research.harness.evidence_gap import (
    EvidenceGapBudget,
    EvidenceGapResult,
    EvidenceGapStopReason,
    run_evidence_gap_round,
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
    audit_path: Path
    report: ReportDraft
    rendered_report: RenderedReport
    loop_result: LoopResult
    claim_decomposition: ClaimDecompositionResult
    attribution: AttributionResult
    verification: VerificationResult
    evidence_gap: EvidenceGapResult
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
    verification_usage: UsageRecord,
    evidence_gap_usage: UsageRecord,
) -> tuple[dict[str, UsageRecord], dict[str, Any]]:
    stages = {
        "checklist": checklist_usage,
        "collection": collection_usage,
        "writing": writing_usage,
        "decomposition_attribution": decomposition_attribution_usage,
        "verification": verification_usage,
        "evidence_gap": evidence_gap_usage,
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
    attribution_model: AttributionModelClient,
    verification_model: VerificationModelClient,
    tavily_client: TavilyClient,
    budget: LoopBudget | None = None,
    loop_settings: LoopSettings | None = None,
    claim_settings: ClaimDecompositionSettings | None = None,
    attribution_settings: AttributionSettings | None = None,
    verification_settings: VerificationSettings | None = None,
    verification_budget: VerificationBudget | None = None,
    evidence_gap_budget: EvidenceGapBudget | None = None,
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
    if evidence_gap_budget is None:
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
            budget=evidence_gap_budget,
            attribution_settings=attribution_settings,
            verification_settings=verification_settings,
            corroboration_targets=corroboration_targets,
            estimate_input_tokens=evidence_gap_input_token_estimator,
            estimate_cost_usd=evidence_gap_cost_estimator,
        )
    attribution = evidence_gap.final_attribution
    verification = evidence_gap.final_verification
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
    )

    writing_usage = UsageRecord(
        token_count=report.token_count,
        cost_usd=report.cost_usd,
    )
    decomposition_attribution_usage = UsageRecord(
        token_count=(
            claim_decomposition.total_tokens
            + initial_attribution.total_tokens
        ),
        cost_usd=(
            claim_decomposition.total_cost_usd
            + initial_attribution.total_cost_usd
        ),
    )
    verification_usage = UsageRecord(
        token_count=initial_verification.total_tokens,
        cost_usd=initial_verification.total_cost_usd,
    )
    evidence_gap_usage = UsageRecord(
        token_count=evidence_gap.total_tokens,
        cost_usd=evidence_gap.total_cost_usd,
    )
    usage, usage_audit = _usage_payload(
        checklist_usage=checklist_usage,
        collection_usage=collection_usage,
        writing_usage=writing_usage,
        decomposition_attribution_usage=(
            decomposition_attribution_usage
        ),
        verification_usage=verification_usage,
        evidence_gap_usage=evidence_gap_usage,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / f"{normalized_run_id}.md"
    audit_path = destination / f"{normalized_run_id}.json"
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
            "initial_attribution": initial_attribution.model_dump(mode="json"),
            "initial_verification": initial_verification.model_dump(mode="json"),
            "evidence_gap": evidence_gap.model_dump(mode="json"),
            "attribution": attribution.model_dump(mode="json"),
            "verification": verification.model_dump(mode="json"),
            "rendering": rendered_report.model_dump(
                mode="json",
                exclude={"markdown"},
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
            "audit": audit_path.name,
        },
    }

    report_path.write_text(rendered_report.markdown, encoding="utf-8")
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return HarnessRunResult(
        run_id=normalized_run_id,
        report_path=report_path,
        audit_path=audit_path,
        report=report,
        rendered_report=rendered_report,
        loop_result=loop_result,
        claim_decomposition=claim_decomposition,
        attribution=attribution,
        verification=verification,
        evidence_gap=evidence_gap,
        usage=usage,
    )
