#!/usr/bin/env python
"""Mechanically inventory source-chain leads in a historical harness audit.

The replay is intentionally narrower than a semantic lead extractor. It reads
only cached source bytes and inventories explicit, reproducible structures:
URLs, DOI-like identifiers, numbered entries, quoted document titles, source
labels, dates, and document locators. It does not call a model, rank sources,
or claim that a lead supports a report assertion.

Example:
    python scripts/replay_harness_source_leads.py \
      /path/to/harness_runs/finance-11/audit.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE / "src"))

from open_deep_research.harness.checklist import ResearchChecklist  # noqa: E402
from open_deep_research.harness.recovery import (  # noqa: E402
    RecoveryTriageAction,
    triage_evidence_recovery,
)
from open_deep_research.harness.source_leads import (  # noqa: E402
    SourceLeadKind,
    inventory_source_lead_candidates,
)
from open_deep_research.harness.verify import VerificationResult  # noqa: E402

_URL = re.compile(r"https?://[^\s<>\]\[()\"']+")
_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_NUMBERED_LINE = re.compile(
    r"(?m)^\s*(?P<number>\d{1,3})\.\s+(?P<text>[^\n]+?)\s*$"
)
_QUOTED_TITLE = re.compile(r"[\"“](?P<title>.+?)[\"”]")
_DATE = re.compile(
    r"\b(?:"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+\d{1,2}(?:,)?\s+\d{4}|"
    r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}"
    r")\b",
    re.IGNORECASE,
)
_LOCATORS = (
    (
        "case_number",
        re.compile(
            r"\bCase\s+(?:No\.?|Number|#)\s*[A-Z0-9][A-Z0-9.\-/]+",
            re.IGNORECASE,
        ),
    ),
    (
        "docket_number",
        re.compile(r"\bDocket\s*#?\s*\d+", re.IGNORECASE),
    ),
    (
        "related_document_numbers",
        re.compile(
            r"\brelated\s+document\(s\)\s*[\d,\s]+",
            re.IGNORECASE,
        ),
    ),
)

_ROUTING_TARGETS = (
    "claim-0035",
    "claim-0036",
    "claim-0040",
    "claim-0047",
    "claim-0056",
)
_DIRECT_ROUTE_TARGETS = {"claim-0047"}
_EDIT_DIRECTLY_TARGETS = {"claim-0053", "claim-0068"}


class _FixedRouteTriageModel:
    """A zero-cost semantic fixture over real finance-11 candidates."""

    def __init__(
        self,
        selected_leads: Mapping[str, str],
        selected_lead_text: Mapping[str, str],
    ) -> None:
        self.selected_leads = dict(selected_leads)
        self.selected_lead_text = dict(selected_lead_text)

    async def generate(self, prompt: str) -> dict[str, Any]:
        claims_marker = "Frozen evidence exceptions:\n"
        leads_marker = "\n\nRegistered source-chain candidates"
        claims_text = prompt.split(claims_marker, 1)[1].split(
            leads_marker, 1
        )[0]
        claims = json.loads(claims_text)
        decisions: list[dict[str, Any]] = []
        for claim in claims:
            claim_id = str(claim["claim_id"])
            research = claim_id in _ROUTING_TARGETS
            action = (
                RecoveryTriageAction.RESEARCH_MORE.value
                if research
                else RecoveryTriageAction.EDIT_DIRECTLY.value
                if claim_id in _EDIT_DIRECTLY_TARGETS
                else RecoveryTriageAction.LEAVE_AS_IS.value
            )
            claim_text = str(claim["claim_text"])
            decisions.append(
                {
                    "claim_id": claim_id,
                    "action": action,
                    "importance": "central" if research else "supporting",
                    "importance_reason": (
                        "fixed offline routing over the historical registry"
                    ),
                    "evidence_need": claim_text if research else None,
                    "preferred_source_role": (
                        "underlying record or independent explanatory reporting"
                        if research
                        else None
                    ),
                    "query": (
                        " ".join(
                            part
                            for part in (
                                self.selected_lead_text.get(claim_id),
                                claim_text,
                                "underlying record independent reporting",
                            )
                            if part
                        )
                        if research
                        else None
                    ),
                    "selected_source_lead_id": (
                        self.selected_leads.get(claim_id) if research else None
                    ),
                }
            )
        return {
            "content": {"decisions": decisions},
            "token_count": 0,
            "cost_usd": 0.0,
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resolve_audit_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "audit.json"
    if not resolved.is_file():
        raise FileNotFoundError(f"audit JSON does not exist: {resolved}")
    return resolved


def _unique_matches(pattern: re.Pattern[str], text: str) -> list[str]:
    return list(dict.fromkeys(match.group(0) for match in pattern.finditer(text)))


def _entry_payload(number: int, text: str) -> dict[str, Any]:
    title_match = _QUOTED_TITLE.search(text)
    title = title_match.group("title").strip() if title_match else None
    source_label = None
    if title_match is not None:
        source_label = text[: title_match.start()].strip().rstrip(".:") or None
    locators = [
        {"kind": kind, "value": match.group(0)}
        for kind, pattern in _LOCATORS
        for match in pattern.finditer(text)
    ]
    return {
        "number": number,
        "verbatim_text": text,
        # Quotation marks alone cannot distinguish a document title from a
        # quotation. Semantic resolution remains a model/human judgement.
        "source_label_candidate": source_label,
        "document_title_candidate": title,
        "dates": _unique_matches(_DATE, text),
        "locators": locators,
        # This is a structural candidate, not a judgement that the entry is a
        # primary source or supports any claim.
        "bibliographic_shape": bool(title or locators),
    }


def inventory_source_text(url: str, text: str) -> dict[str, Any]:
    """Inventory explicit source-chain structures without semantic ranking."""

    numbered_entries = [
        _entry_payload(int(match.group("number")), match.group("text"))
        for match in _NUMBERED_LINE.finditer(text)
    ]
    return {
        "url": url,
        "source_chars": len(text),
        "explicit_urls": _unique_matches(_URL, text),
        "dois": _unique_matches(_DOI, text),
        "numbered_entries": numbered_entries,
        "bibliographic_candidates": [
            entry for entry in numbered_entries if entry["bibliographic_shape"]
        ],
        "locator_entries": [
            entry for entry in numbered_entries if entry["locators"]
        ],
    }


def _select_finance_route_leads(
    source_cache: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Select fixed real candidates for a labelled offline routing replay."""

    candidates = inventory_source_lead_candidates(source_cache)

    def entry(number: int) -> Any:
        matches = tuple(
            candidate
            for candidate in candidates
            if candidate.kind is SourceLeadKind.BIBLIOGRAPHIC_ENTRY
            and candidate.entry_number == number
            and "investopedia.com" in candidate.source_url
        )
        if len(matches) != 1:
            raise ValueError(
                f"expected one cached Investopedia entry {number}, "
                f"found {len(matches)}"
            )
        return matches[0]

    payout_contexts = tuple(
        candidate
        for candidate in candidates
        if candidate.kind is SourceLeadKind.DATED_CONTEXT
        and "cryptoresearch.report" in candidate.source_url
        and "September 30, 2025" in candidate.verbatim_text
        and "FTX Recovery Trust" in candidate.verbatim_text
    )
    if len(payout_contexts) != 1:
        raise ValueError(
            "expected one cached issuer/date payout context, "
            f"found {len(payout_contexts)}"
        )
    selected = {
        "claim-0035": entry(6),
        "claim-0036": entry(7),
        # The disclosure statement is only a lead to inspect. Selecting it
        # does not assert that it supports the repayment/recovery figures.
        "claim-0040": entry(37),
        # This header names the recovery body and distribution context. The
        # model may follow that clue while independent reporting remains in
        # scope for interpretation and cross-checking.
        "claim-0056": payout_contexts[0],
    }
    selected_ids = {
        claim_id: candidate.lead_id
        for claim_id, candidate in selected.items()
    }
    selected_payload = {
        claim_id: candidate.model_dump(mode="json")
        for claim_id, candidate in selected.items()
    }
    return selected_ids, selected_payload


async def _replay_finance_routes(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run production triage routing with a fixed, zero-cost fake model."""

    ledger = payload.get("ledger")
    posthoc = payload.get("posthoc_evidence")
    pre_edit = (
        posthoc.get("pre_edit_evidence")
        if isinstance(posthoc, Mapping)
        else None
    )
    if not isinstance(ledger, Mapping) or not isinstance(pre_edit, Mapping):
        raise ValueError("audit lacks ledger or pre-edit evidence registry")
    source_cache = ledger.get("source_cache")
    if not isinstance(source_cache, Mapping):
        raise ValueError("audit has no ledger.source_cache object")
    selected_ids, selected_payload = _select_finance_route_leads(source_cache)
    triage = await triage_evidence_recovery(
        str(payload["original_canonical_draft"]),
        checklist=ResearchChecklist.model_validate(payload["checklist"]),
        verification=VerificationResult.model_validate(pre_edit["verification"]),
        model_client=_FixedRouteTriageModel(
            selected_ids,
            {
                claim_id: str(candidate["verbatim_text"])
                for claim_id, candidate in selected_payload.items()
            },
        ),
        source_cache={str(url): str(text) for url, text in source_cache.items()},
    )
    by_id = {decision.claim_id: decision for decision in triage.decisions}
    routes = []
    for claim_id in _ROUTING_TARGETS:
        decision = by_id[claim_id]
        routes.append(
            {
                "claim_id": claim_id,
                "query_route": decision.query_route.value,
                "selected_source_lead_id": decision.selected_source_lead_id,
                "source_document_hint": decision.source_document_hint,
                "query": decision.query,
                "route_basis": (
                    "registered_cached_lead"
                    if decision.selected_source_lead_id is not None
                    else "no_registered_upstream_lead"
                ),
            }
        )
    return {
        "mode": "fixed_fake_semantic_selection_over_real_cached_candidates",
        "network_calls": 0,
        "model_cost_usd": 0.0,
        "selected_leads": selected_payload,
        "routes": routes,
        "expected_direct_fallback_claim_ids": sorted(_DIRECT_ROUTE_TARGETS),
        "all_targets_routed": all(
            route["query_route"]
            in {"source_chain", "direct_search_fallback"}
            for route in routes
        ),
    }


def replay_source_leads(audit_path: Path) -> dict[str, Any]:
    """Return a source-lead inventory while proving the audit stayed frozen."""

    audit_path = _resolve_audit_path(audit_path)
    before = audit_path.read_bytes()
    payload = json.loads(before)
    if not isinstance(payload, Mapping):
        raise ValueError("audit JSON must contain one object")
    ledger = payload.get("ledger")
    source_cache = (
        ledger.get("source_cache") if isinstance(ledger, Mapping) else None
    )
    if not isinstance(source_cache, Mapping):
        raise ValueError("audit has no ledger.source_cache object")
    sources = [
        inventory_source_text(str(url), str(text))
        for url, text in source_cache.items()
    ]
    posthoc = payload.get("posthoc_evidence")
    pre_edit = (
        posthoc.get("pre_edit_evidence")
        if isinstance(posthoc, Mapping)
        else None
    )
    verification = (
        pre_edit.get("verification")
        if isinstance(pre_edit, Mapping)
        else None
    )
    observed_claim_ids = {
        str(claim.get("claim", {}).get("claim_id"))
        for claim in verification.get("claims", ())
        if isinstance(claim, Mapping)
    } if isinstance(verification, Mapping) else set()
    routing_replay = (
        asyncio.run(_replay_finance_routes(payload))
        if set(_ROUTING_TARGETS).issubset(observed_claim_ids)
        else {
            "mode": "not_applicable_to_this_historical_registry",
            "required_claim_ids": list(_ROUTING_TARGETS),
            "network_calls": 0,
            "model_cost_usd": 0.0,
        }
    )
    result = {
        "replay_mode": "offline_mechanical_source_lead_inventory",
        "source_audit": str(audit_path),
        "source_run_id": payload.get("run_id"),
        "source_audit_sha256": _sha256_bytes(before),
        "limitations": [
            "no model performed semantic claim-to-lead matching",
            "cleaned cache text may omit hyperlink targets present on the page",
            "quoted text can be a quotation rather than a document title",
            "a bibliographic shape is not proof that a document is primary",
            "a lead is not evidence and does not establish claim support",
        ],
        "summary": {
            "cached_source_count": len(sources),
            "source_with_explicit_url_count": sum(
                bool(source["explicit_urls"]) for source in sources
            ),
            "explicit_url_count": sum(
                len(source["explicit_urls"]) for source in sources
            ),
            "doi_count": sum(len(source["dois"]) for source in sources),
            "numbered_entry_count": sum(
                len(source["numbered_entries"]) for source in sources
            ),
            "bibliographic_candidate_count": sum(
                len(source["bibliographic_candidates"]) for source in sources
            ),
            "locator_entry_count": sum(
                len(source["locator_entries"]) for source in sources
            ),
        },
        "sources": sources,
        "routing_replay": routing_replay,
        "source_audit_unchanged": audit_path.read_bytes() == before,
    }
    return result


def _default_output_path(audit_path: Path, run_id: str | None) -> Path:
    stem = run_id or _resolve_audit_path(audit_path).parent.name
    return _BASE / "harness_replays" / f"{stem}.source-leads-replay.json"


def _write_result(path: Path, result: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory explicit source-chain leads in cached audit text; "
            "uses no model and no network."
        )
    )
    parser.add_argument(
        "audit",
        type=Path,
        help="historical audit.json or its containing run directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output JSON (default: gitignored harness_replays/)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    audit_path = _resolve_audit_path(args.audit)
    result = replay_source_leads(audit_path)
    output = args.output or _default_output_path(
        audit_path,
        str(result.get("source_run_id") or "") or None,
    )
    if output.resolve() == audit_path:
        raise ValueError("replay output cannot overwrite the source audit")
    _write_result(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\nReplay output: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
