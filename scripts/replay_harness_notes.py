#!/usr/bin/env python
"""Replay cached note-extraction calls with exactly one model variable.

The script reconstructs each successful note-model call from a harness audit:
the cached cleaned source, active checklist item, terminal-state history, note
prompt, parser, cross-item capacity, and source-span resolver all come from the
production harness.  It does not run Tavily, the decision loop, or pending
candidate handling, and it never writes back into ``harness_runs``.

Example:
    python scripts/replay_harness_notes.py \
      harness_runs/519d3215ec474b23b342aa405a1fb6c6.json \
      --model openai/gpt-5.4-mini
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AsyncOpenAI

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE / "src"))

from open_deep_research.harness.checklist import (  # noqa: E402
    ChecklistStatus,
    ResearchChecklist,
)
from open_deep_research.harness.ledger import ResearchLedger  # noqa: E402
from open_deep_research.harness.loop import (  # noqa: E402
    _extract_notes,
    build_note_prompt,
)
from open_deep_research.harness.notes import ResearchNote  # noqa: E402
from open_deep_research.harness.note_span_policy import (  # noqa: E402
    DEFAULT_NOTE_SPAN_MAX_CHARS,
    DEFAULT_NOTE_SPAN_MAX_SEGMENTS,
)

_OPENROUTER_PROXY = "http://127.0.0.1:7890"
_TERMINAL_STATUSES = {
    ChecklistStatus.SETTLED.value,
    ChecklistStatus.EXHAUSTED_NOT_FOUND.value,
}


@dataclass(frozen=True)
class ReplayCase:
    """One historically executed note call reconstructed before replay."""

    round_number: int
    action: str
    active_item_id: str
    url: str
    source_text: str
    checklist: ResearchChecklist
    max_cross_item_seeds: int

    @property
    def prompt(self) -> str:
        return build_note_prompt(
            self.checklist,
            active_item_id=self.active_item_id,
            url=self.url,
            source_text=self.source_text,
            max_cross_item_seeds=self.max_cross_item_seeds,
        )


class ReplayModel(Protocol):
    """The production usage-envelope boundary used by ``_extract_notes``."""

    async def generate(self, prompt: str) -> dict[str, object]:
        """Return content and measured usage."""


class OpenAIReplayModel:
    """Minimal OpenAI-compatible JSON client for an isolated replay."""

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self.client = client
        self.model = model
        self.calls: list[dict[str, Any]] = []

    async def generate(self, prompt: str) -> dict[str, object]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("chat completion returned no text content")
        usage = (
            response.usage.model_dump()
            if response.usage is not None
            else {}
        )
        token_count = int(usage.get("total_tokens", 0) or 0)
        cost_usd = float(
            usage.get("cost_usd", usage.get("cost", 0.0)) or 0.0
        )
        envelope: dict[str, object] = {
            "content": content,
            "token_count": token_count,
            "cost_usd": cost_usd,
        }
        self.calls.append(
            {
                "prompt_sha256": _sha256_text(prompt),
                "raw_content": content,
                "token_count": token_count,
                "cost_usd": cost_usd,
            }
        )
        return envelope


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _round_summary(round_record: dict[str, Any]) -> dict[str, Any]:
    raw = round_record.get("result_summary", "")
    if not isinstance(raw, str):
        raise ValueError("round result_summary must be a JSON string")
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise ValueError("round result_summary must decode to an object")
    return decoded


def _initial_checklist(payload: dict[str, Any]) -> ResearchChecklist:
    checklist = ResearchChecklist.model_validate(payload["checklist"])
    return checklist.model_copy(
        update={
            "items": tuple(
                item.model_copy(
                    update={"status": ChecklistStatus.UNEXPLORED}
                )
                for item in checklist.items
            )
        }
    )


def _apply_recorded_terminal_updates(
    checklist: ResearchChecklist,
    summary: dict[str, Any],
) -> ResearchChecklist:
    statuses = {
        str(update.get("item_id")): str(update.get("status"))
        for update in summary.get("status_updates", ())
        if (
            isinstance(update, dict)
            and update.get("status") in _TERMINAL_STATUSES
        )
    }
    if not statuses:
        return checklist
    return checklist.model_copy(
        update={
            "items": tuple(
                item.model_copy(
                    update={
                        "status": ChecklistStatus(statuses[item.item_id])
                    }
                )
                if item.item_id in statuses
                else item
                for item in checklist.items
            )
        }
    )


def load_replay_cases(
    audit_path: Path,
) -> tuple[dict[str, Any], tuple[ReplayCase, ...]]:
    """Reconstruct the note calls that actually ran in one audit."""

    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("audit JSON must contain one object")
    history = payload.get("ledger", {}).get("checklist_history", ())
    if any(
        isinstance(record, dict) and record.get("event") == "append"
        for record in history
    ):
        raise ValueError(
            "cannot reconstruct appended-item timing from this audit"
        )
    source_cache = payload.get("ledger", {}).get("source_cache")
    rounds = payload.get("ledger", {}).get("rounds")
    if not isinstance(source_cache, dict) or not isinstance(rounds, list):
        raise ValueError("audit is missing ledger source_cache or rounds")

    current = _initial_checklist(payload)
    cases: list[ReplayCase] = []
    for record in sorted(rounds, key=lambda value: value["round_number"]):
        summary = _round_summary(record)
        current = _apply_recorded_terminal_updates(current, summary)
        if (
            record.get("action") not in {"read", "reanalyze"}
            or summary.get("note_model_called") is not True
        ):
            continue
        url = record.get("url")
        active_item_id = record.get("item_id")
        if not isinstance(url, str) or url not in source_cache:
            raise ValueError(
                f"round {record.get('round_number')} note call has no cached source"
            )
        if not isinstance(active_item_id, str):
            raise ValueError(
                f"round {record.get('round_number')} note call has no item_id"
            )
        max_cross = summary.get("cross_item_seed_capacity")
        if not isinstance(max_cross, int) or isinstance(max_cross, bool):
            raise ValueError(
                f"round {record.get('round_number')} lacks cross-item capacity"
            )
        cases.append(
            ReplayCase(
                round_number=int(record["round_number"]),
                action=str(record["action"]),
                active_item_id=active_item_id,
                url=url,
                source_text=str(source_cache[url]),
                checklist=current,
                max_cross_item_seeds=max_cross,
            )
        )
    if not cases:
        raise ValueError("audit contains no successful note-model calls")
    return payload, tuple(cases)


def _length_distribution(lengths: list[int]) -> dict[str, Any]:
    ordered = sorted(lengths)
    if not ordered:
        return {
            "count": 0,
            "min": 0,
            "median": 0,
            "p90": 0,
            "max": 0,
            "mean": 0.0,
            "values": [],
        }

    def nearest_rank(fraction: float) -> int:
        index = max(0, int((len(ordered) * fraction) + 0.999999) - 1)
        return ordered[min(index, len(ordered) - 1)]

    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2
    )
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": median,
        "p90": nearest_rank(0.9),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 3),
        "values": ordered,
    }


def _channel_metrics(notes: list[ResearchNote]) -> dict[str, Any]:
    location_counts = Counter(note.location_status.value for note in notes)
    failures = Counter(
        note.failure_reason.value
        for note in notes
        if note.failure_reason is not None
    )
    return {
        "note_count": len(notes),
        "location_counts": {
            "strict": location_counts["locatable"],
            "repaired": location_counts["repaired_locatable"],
            "unlocatable": location_counts["unlocatable"],
        },
        "noncontiguous_composite_count": failures[
            "noncontiguous_composite"
        ],
        "failure_reason_counts": dict(sorted(failures.items())),
        "quote_length_chars": _length_distribution(
            [len(note.source_quote or note.model_quote or "") for note in notes]
        ),
    }


def _note_payload(note: ResearchNote, *, channel: str) -> dict[str, Any]:
    return {
        "channel": channel,
        **note.model_dump(mode="json"),
        "model_quote_length_chars": len(note.model_quote or ""),
        "source_quote_length_chars": len(note.source_quote or ""),
    }


async def replay_cases(
    cases: tuple[ReplayCase, ...],
    *,
    model_client: ReplayModel,
    model_name: str,
    source_run_id: str,
) -> dict[str, Any]:
    """Execute isolated cases through the production prompt/parser/locator."""

    per_source: list[dict[str, Any]] = []
    all_active: list[ResearchNote] = []
    all_cross: list[ResearchNote] = []
    all_span_rejections: list[dict[str, Any]] = []
    total_tokens = 0
    total_cost = 0.0

    for case in cases:
        ledger = ResearchLedger(topic=case.checklist.topic)
        _, tokens, cost, summary = await _extract_notes(
            case.checklist,
            ledger=ledger,
            note_model=model_client,
            active_item_id=case.active_item_id,
            url=case.url,
            source_text=case.source_text,
            max_cross_item_seeds=case.max_cross_item_seeds,
        )
        active = [
            note for note in ledger.notes
            if note.item_id == case.active_item_id
        ]
        cross = [
            note for note in ledger.notes
            if note.item_id != case.active_item_id
        ]
        all_active.extend(active)
        all_cross.extend(cross)
        total_tokens += tokens
        total_cost += cost
        source_rejections = [
            {
                "round_number": case.round_number,
                "url": case.url,
                **rejection,
            }
            for rejection in summary["note_span_rejections"]
        ]
        all_span_rejections.extend(source_rejections)
        per_source.append(
            {
                "round_number": case.round_number,
                "action": case.action,
                "active_item_id": case.active_item_id,
                "eligible_cross_item_ids": [
                    item.item_id
                    for item in case.checklist.items
                    if (
                        item.item_id != case.active_item_id
                        and not item.is_complete
                    )
                ],
                "url": case.url,
                "source_chars": len(case.source_text),
                "source_sha256": _sha256_text(case.source_text),
                "prompt_sha256": _sha256_text(case.prompt),
                "max_cross_item_seeds": case.max_cross_item_seeds,
                "token_count": tokens,
                "cost_usd": cost,
                "active": _channel_metrics(active),
                "cross": _channel_metrics(cross),
                "notes": [
                    *(_note_payload(note, channel="active") for note in active),
                    *(_note_payload(note, channel="cross") for note in cross),
                ],
                "production_summary": summary,
            }
        )

    combined = [*all_active, *all_cross]
    return {
        "replay_kind": "isolated_note_model",
        "source_run_id": source_run_id,
        "model": model_name,
        "case_count": len(cases),
        "fixed_invariants": {
            "decision_loop_called": False,
            "tavily_called": False,
            "pending_state_machine_called": False,
            "production_note_prompt": True,
            "production_note_schema_parser": True,
            "production_segment_pointer_resolution": True,
            "authoritative_source_slicing": True,
            "legacy_free_text_repair_used": False,
            "oversized_ranges_truncated": False,
            "oversized_ranges_retried": False,
            "cross_item_capacity_from_source_audit": True,
        },
        "usage": {
            "token_count": total_tokens,
            "cost_usd": round(total_cost, 10),
        },
        "active": _channel_metrics(all_active),
        "cross": _channel_metrics(all_cross),
        "combined": _channel_metrics(combined),
        "span_capacity": {
            "limits": {
                "max_segments": DEFAULT_NOTE_SPAN_MAX_SEGMENTS,
                "max_chars": DEFAULT_NOTE_SPAN_MAX_CHARS,
            },
            "rejected_note_count": len(all_span_rejections),
            "failure_reason_counts": dict(
                sorted(
                    Counter(
                        reason
                        for rejection in all_span_rejections
                        for reason in rejection["failure_reasons"]
                    ).items()
                )
            ),
            "rejections": all_span_rejections,
        },
        "per_source": per_source,
    }


def _configure_proxy(base_url: str | None) -> None:
    if (urlparse(base_url or "").hostname or "").casefold() != "openrouter.ai":
        return
    os.environ.setdefault("https_proxy", _OPENROUTER_PROXY)
    os.environ.setdefault("HTTPS_PROXY", os.environ["https_proxy"])


def _default_output_path(run_id: str, model: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "-", model).strip("-")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        _BASE
        / "harness_replays"
        / f"{run_id[:8]}.{safe_model}.{timestamp}.json"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay cached harness note calls while changing only the model."
        )
    )
    parser.add_argument("audit_json", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    return parser


async def _run(args: argparse.Namespace) -> Path:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    _configure_proxy(base_url)

    payload, cases = load_replay_cases(args.audit_json.resolve())
    run_id = str(payload.get("run_id", args.audit_json.stem))
    output_path = (
        args.output.resolve()
        if args.output is not None
        else _default_output_path(run_id, args.model)
    )
    if output_path.resolve() == args.audit_json.resolve():
        raise ValueError("replay output must not overwrite the source audit")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    try:
        model = OpenAIReplayModel(client, args.model)
        result = await replay_cases(
            cases,
            model_client=model,
            model_name=args.model,
            source_run_id=run_id,
        )
        result["provider_calls"] = model.calls
    finally:
        await client.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return output_path


def main() -> int:
    args = build_parser().parse_args()
    output_path = asyncio.run(_run(args))
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
