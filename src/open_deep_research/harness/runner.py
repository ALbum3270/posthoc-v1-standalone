"""End-to-end orchestration and durable harness artifacts."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from open_deep_research.harness.assemble import assemble_notes
from open_deep_research.harness.checklist import (
    ChecklistModelClient,
    generate_checklist,
)
from open_deep_research.harness.ledger import ResearchLedger
from open_deep_research.harness.loop import (
    LoopBudget,
    LoopModelClient,
    LoopResult,
    LoopSettings,
    run_research_loop,
)
from open_deep_research.harness.tools import TavilyClient
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
    loop_result: LoopResult
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
) -> tuple[dict[str, UsageRecord], dict[str, Any]]:
    stages = {
        "checklist": checklist_usage,
        "collection": collection_usage,
        "writing": writing_usage,
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
    tavily_client: TavilyClient,
    budget: LoopBudget | None = None,
    loop_settings: LoopSettings | None = None,
    output_dir: str | Path = Path("harness_runs"),
    run_id: str | None = None,
    model_names: Mapping[str, str] | None = None,
) -> HarnessRunResult:
    """Run every pre-verification stage and write report plus audit files."""

    normalized_run_id = _normalize_run_id(run_id)
    ledger = ResearchLedger(research_id=normalized_run_id, topic=topic.strip())

    checklist = await generate_checklist(topic, model_client=checklist_model)
    checklist_usage = _usage_from_model(checklist_model)
    loop_result = await run_research_loop(
        checklist,
        ledger=ledger,
        decision_model=decision_model,
        note_model=note_model,
        tavily_client=tavily_client,
        budget=budget,
        settings=loop_settings,
    )
    assembled = assemble_notes(loop_result.checklist, ledger.notes)
    report = await write_report(assembled, model_client=write_model)

    collection_usage = UsageRecord(
        token_count=ledger.total_tokens,
        cost_usd=ledger.total_cost_usd,
    )
    writing_usage = UsageRecord(
        token_count=report.token_count,
        cost_usd=report.cost_usd,
    )
    usage, usage_audit = _usage_payload(
        checklist_usage=checklist_usage,
        collection_usage=collection_usage,
        writing_usage=writing_usage,
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
        "usage": usage_audit,
        "models": dict(model_names or {}),
        "artifacts": {
            "report": report_path.name,
            "audit": audit_path.name,
        },
    }

    report_path.write_text(report.markdown, encoding="utf-8")
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return HarnessRunResult(
        run_id=normalized_run_id,
        report_path=report_path,
        audit_path=audit_path,
        report=report,
        loop_result=loop_result,
        usage=usage,
    )
