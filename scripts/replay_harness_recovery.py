#!/usr/bin/env python
"""Offline replay of evidence recovery against one historical audit.

This diagnostic fixes the historical draft, claim/attribution/verification
registries, checklist, notes, and source cache.  It replaces every model and
network boundary with deterministic fakes, then runs the production recovery
triage, evidence-gap executor, and recovery summarizer.  Nothing is written
back to ``harness_runs``.

The replay deliberately separates two facts:

* observations made from the supplied historical registry; and
* a labelled synthetic contract probe used only when that registry contains
  no non-external recovery anomaly.

Example:
    python scripts/replay_harness_recovery.py \
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
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE / "src"))

from open_deep_research.harness.attribution import (  # noqa: E402
    AttributionResult,
)
from open_deep_research.harness.checklist import (  # noqa: E402
    ResearchChecklist,
)
from open_deep_research.harness.claims import (  # noqa: E402
    CitationRequirement,
    ClaimDecompositionResult,
)
from open_deep_research.harness.evidence_gap import (  # noqa: E402
    EvidenceGapBudget,
    run_evidence_gap_round,
)
from open_deep_research.harness.ledger import ResearchLedger  # noqa: E402
from open_deep_research.harness.recovery import (  # noqa: E402
    RecoveryTriageAction,
    RecoveryTriageResult,
    build_recovery_gap_plan_prompt,
    summarize_evidence_recovery,
    triage_evidence_recovery,
)
from open_deep_research.harness.verify import (  # noqa: E402
    VerificationResult,
    VerificationVerdict,
)


_EXPECTED_RESEARCH = (
    "claim-0035",
    "claim-0036",
    "claim-0040",
    "claim-0047",
    "claim-0056",
)
_EXPECTED_EDIT_DIRECTLY = ("claim-0053", "claim-0068")
_MIXED_VERDICTS = {
    "claim-0035": VerificationVerdict.SUPPORTS,
    "claim-0036": VerificationVerdict.DOES_NOT_SUPPORT,
    "claim-0040": VerificationVerdict.NOT_ENOUGH_INFORMATION,
}
_TRIAGE_CLAIMS_MARKER = "Frozen evidence exceptions:\n"
_TRIAGE_LEADS_MARKER = (
    "\n\nRegistered source-chain candidates "
    "(mechanical text shapes, not evidence):\n"
)
_VERIFICATION_CLAIMS_RE = re.compile(
    r"Claims:\n(?P<claims>\[.*?\])\n\n"
    r"BEGIN COMPLETE CACHED SOURCE WITH ADDRESSABLE SEGMENTS",
    flags=re.DOTALL,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resolve_audit_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "audit.json"
    if not resolved.is_file():
        raise FileNotFoundError(f"audit JSON does not exist: {resolved}")
    return resolved


class _MeasuredFake:
    """Zero-cost deterministic model boundary with admission estimates."""

    def __init__(self) -> None:
        self.calls = 0

    def estimate_tokens(self, prompt: str) -> int:
        del prompt
        return 1

    def estimate_cost_usd(self, prompt: str) -> float:
        del prompt
        return 0.0

    @staticmethod
    def envelope(content: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "content": dict(content),
            "token_count": 1,
            "cost_usd": 0.0,
        }


class _ScriptedTriageModel(_MeasuredFake):
    """Return the expected finance-11 routing for every observed target."""

    @staticmethod
    def _selected_lead_by_claim(
        leads: Sequence[Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        def numbered_entry(number: int) -> Mapping[str, Any]:
            matches = tuple(
                lead
                for lead in leads
                if lead.get("kind") == "bibliographic_entry"
                and lead.get("entry_number") == number
                and "investopedia.com" in str(lead.get("source_url"))
            )
            if len(matches) != 1:
                raise ValueError(
                    f"expected one registered entry {number}, found "
                    f"{len(matches)}"
                )
            return matches[0]

        payout_contexts = tuple(
            lead
            for lead in leads
            if lead.get("kind") == "dated_context"
            and "September 30, 2025" in tuple(lead.get("dates") or ())
            and "FTX Recovery Trust" in str(lead.get("verbatim_text"))
        )
        if len(payout_contexts) != 1:
            raise ValueError(
                "expected one registered issuer/date payout context, found "
                f"{len(payout_contexts)}"
            )
        return {
            "claim-0035": numbered_entry(6),
            "claim-0036": numbered_entry(7),
            "claim-0040": numbered_entry(37),
            "claim-0056": payout_contexts[0],
        }

    async def generate(self, prompt: str) -> dict[str, Any]:
        self.calls += 1
        try:
            claims = json.loads(
                prompt.split(_TRIAGE_CLAIMS_MARKER, 1)[1].split(
                    _TRIAGE_LEADS_MARKER, 1
                )[0]
            )
            leads = json.loads(prompt.split(_TRIAGE_LEADS_MARKER, 1)[1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise ValueError("triage prompt did not expose its target array") from exc
        selected_lead_by_claim = self._selected_lead_by_claim(leads)
        decisions: list[dict[str, Any]] = []
        for claim in claims:
            claim_id = str(claim["claim_id"])
            if claim_id in _EXPECTED_RESEARCH:
                action = RecoveryTriageAction.RESEARCH_MORE
            elif claim_id in _EXPECTED_EDIT_DIRECTLY:
                action = RecoveryTriageAction.EDIT_DIRECTLY
            else:
                action = RecoveryTriageAction.LEAVE_AS_IS
            research = action is RecoveryTriageAction.RESEARCH_MORE
            selected_lead = selected_lead_by_claim.get(claim_id)
            lead_text = (
                str(selected_lead["verbatim_text"])
                if selected_lead is not None
                else None
            )
            decisions.append(
                {
                    "claim_id": claim_id,
                    "action": action.value,
                    "importance": "central" if research else "supporting",
                    "importance_reason": (
                        "fixed offline replay routing for this historical claim"
                    ),
                    "evidence_need": (
                        f"a source that directly addresses {claim_id}"
                        if research
                        else None
                    ),
                    "preferred_source_role": (
                        "underlying record or independent reporting"
                        if research
                        else None
                    ),
                    "query": (
                        " ".join(
                            part
                            for part in (
                                lead_text,
                                f"offline replay query for {claim_id}",
                            )
                            if part
                        )
                        if research
                        else None
                    ),
                    "selected_source_lead_id": (
                        str(selected_lead["lead_id"])
                        if selected_lead is not None
                        else None
                    ),
                }
            )
        return self.envelope({"decisions": decisions})


class _GapPlanModel(_MeasuredFake):
    """Return a fixed offline plan of cached candidates and/or fake queries."""

    def __init__(
        self,
        cached_candidates: Sequence[Mapping[str, Any]],
        queries: Sequence[Mapping[str, Any]],
        deferred_targets: Sequence[Mapping[str, Any]],
    ) -> None:
        super().__init__()
        self.cached_candidates = tuple(dict(item) for item in cached_candidates)
        self.queries = tuple(dict(item) for item in queries)
        self.deferred_targets = tuple(dict(item) for item in deferred_targets)

    async def generate(self, prompt: str) -> dict[str, Any]:
        del prompt
        self.calls += 1
        return self.envelope(
            {
                "cached_candidates": list(self.cached_candidates),
                "queries": list(self.queries),
                "deferred_targets": list(self.deferred_targets),
            }
        )


class _MixedVerifier(_MeasuredFake):
    """Return three different completed semantic outcomes from real claims."""

    async def generate(self, prompt: str) -> dict[str, Any]:
        self.calls += 1
        match = _VERIFICATION_CLAIMS_RE.search(prompt)
        if match is None:
            raise ValueError("verification prompt did not expose its claim array")
        claims = json.loads(match.group("claims"))
        results: list[dict[str, Any]] = []
        for claim in claims:
            claim_id = str(claim["claim_id"])
            verdict = _MIXED_VERDICTS.get(claim_id)
            if verdict is None:
                raise ValueError(f"unexpected mixed-verdict claim: {claim_id}")
            evidentiary = verdict in {
                VerificationVerdict.SUPPORTS,
                VerificationVerdict.CONTRADICTS,
            }
            results.append(
                {
                    "claim_id": claim_id,
                    "verdict": verdict.value,
                    "start_segment_id": "S000001" if evidentiary else None,
                    "end_segment_id": "S000001" if evidentiary else None,
                    "explanation": (
                        "fixed mixed-verdict outcome for offline orchestration"
                    ),
                }
            )
        return self.envelope({"results": results})


class _ForbiddenBoundary(_MeasuredFake):
    """Fail loudly if a cache-only replay crosses an unintended boundary."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    async def generate(self, prompt: str) -> dict[str, Any]:
        del prompt
        self.calls += 1
        raise AssertionError(f"offline replay unexpectedly called {self.name}")


class _NoNetworkTavily:
    """Return a fixed empty search response and forbid source extraction."""

    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        del query, kwargs
        self.calls += 1
        return {"results": []}

    async def extract(self, urls: Any, **kwargs: Any) -> dict[str, Any]:
        del urls, kwargs
        self.calls += 1
        raise AssertionError("offline replay unexpectedly called Tavily.extract")


def _load_registry(
    audit_path: Path,
) -> tuple[
    dict[str, Any],
    str,
    ResearchChecklist,
    ClaimDecompositionResult,
    AttributionResult,
    VerificationResult,
]:
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("audit JSON must contain one object")
    posthoc = payload.get("posthoc_evidence")
    if not isinstance(posthoc, Mapping):
        raise ValueError("audit has no posthoc_evidence object")
    pre_edit = posthoc.get("pre_edit_evidence")
    if not isinstance(pre_edit, Mapping):
        raise ValueError(
            "audit has no pre_edit_evidence registry; recovery precedes editing"
        )
    draft = payload.get("original_canonical_draft")
    if not isinstance(draft, str):
        raise ValueError("audit has no original_canonical_draft")
    return (
        payload,
        draft,
        ResearchChecklist.model_validate(payload["checklist"]),
        ClaimDecompositionResult.model_validate(
            pre_edit["claim_decomposition"]
        ),
        AttributionResult.model_validate(pre_edit["attribution"]),
        VerificationResult.model_validate(pre_edit["verification"]),
    )


def _clone_ledger(payload: Mapping[str, Any]) -> ResearchLedger:
    return ResearchLedger.model_validate(payload["ledger"])


def _decision_map(triage: RecoveryTriageResult) -> dict[str, str]:
    return {
        decision.claim_id: decision.action.value
        for decision in triage.decisions
    }


def _select_cached_hints(
    *,
    ledger: ResearchLedger,
    verification: VerificationResult,
) -> tuple[dict[str, Any], ...]:
    """Select one real, previously unchecked cached relation per fake verdict."""

    result_by_id = {
        result.claim.claim_id: result for result in verification.claims
    }
    notes = tuple(
        note for note in ledger.notes if note.note_id is not None
    )
    selected: list[dict[str, Any]] = []
    used_note_ids: set[str] = set()
    for claim_id in _MIXED_VERDICTS:
        result = result_by_id[claim_id]
        existing_urls = {relation.url for relation in result.relations}
        note = next(
            (
                candidate
                for candidate in notes
                if candidate.url not in existing_urls
                and str(candidate.note_id) not in used_note_ids
                and candidate.url in ledger.source_cache
            ),
            None,
        )
        if note is None:
            raise ValueError(
                f"no unused cached note/source relation for {claim_id}"
            )
        used_note_ids.add(str(note.note_id))
        selected.append(
            {
                "claim_id": claim_id,
                "note_id": str(note.note_id),
                "source_id": note.source_id,
                "independent_from_existing_publishers": True,
                "publisher_identity": note.publisher,
                "independence_rationale": (
                    "fixed cache-only candidate for offline replay"
                ),
            }
        )
    return tuple(selected)


async def _internal_contract_probe(
    *,
    draft: str,
    checklist: ResearchChecklist,
    verification: VerificationResult,
) -> dict[str, Any]:
    """Exercise triage's non-external boundary without claiming observation."""

    source = next(
        claim
        for claim in verification.claims
        if claim.claim.claim_id == "claim-0033"
    )
    internal_claim = source.claim.model_copy(
        update={"citation_requirement": CitationRequirement.INTERNAL}
    )
    internal_result = source.model_copy(update={"claim": internal_claim})
    probe = VerificationResult(
        claims=(internal_result,),
        independence=verification.independence,
    )
    model = _ForbiddenBoundary("triage model for inapplicable-only scope")
    triage = await triage_evidence_recovery(
        draft,
        checklist=checklist,
        verification=probe,
        model_client=model,
    )
    return {
        "kind": "synthetic_contract_probe",
        "derived_from_claim_id": source.claim.claim_id,
        "changed_field_only": "citation_requirement: external -> internal",
        "triage_status": triage.status.value,
        "triage_target_claim_ids": list(triage.target_claim_ids),
        "triage_model_calls": model.calls,
        "inapplicable_claims": [
            claim.model_dump(mode="json")
            for claim in triage.inapplicable_claims
        ],
        "passed": any(
            claim.claim_id == source.claim.claim_id
            for claim in triage.inapplicable_claims
        )
        and not triage.target_claim_ids
        and model.calls == 0,
    }


async def _run_gap_scenario(
    *,
    payload: Mapping[str, Any],
    draft: str,
    checklist: ResearchChecklist,
    decomposition: ClaimDecompositionResult,
    attribution: AttributionResult,
    verification: VerificationResult,
    triage: RecoveryTriageResult,
    cached_candidates: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    mixed_verifier: bool,
) -> tuple[Any, dict[str, int]]:
    ledger = _clone_ledger(payload)
    routed_claim_ids = {
        str(candidate["claim_id"]) for candidate in cached_candidates
    } | {
        str(claim_id)
        for query in queries
        for claim_id in query["claim_ids"]
    }
    deferred_targets = tuple(
        {
            "claim_id": claim_id,
            "reason": "query_capacity_not_allocated",
            "priority_rationale": (
                "offline mixed-verdict probe deliberately has zero web "
                "query capacity"
            ),
        }
        for claim_id in triage.research_target_claim_ids
        if claim_id not in routed_claim_ids
    )
    gap_model = _GapPlanModel(
        cached_candidates,
        queries,
        deferred_targets,
    )
    note_model = _ForbiddenBoundary("note model")
    attribution_model = _ForbiddenBoundary("attribution model")
    verifier: _MeasuredFake = (
        _MixedVerifier()
        if mixed_verifier
        else _ForbiddenBoundary("verification model")
    )
    tavily = _NoNetworkTavily()

    def plan_prompt_builder(**kwargs: Any) -> str:
        return build_recovery_gap_plan_prompt(triage=triage, **kwargs)

    pass_result = await run_evidence_gap_round(
        canonical_draft=draft,
        checklist=checklist,
        blocks=decomposition.blocks,
        ledger=ledger,
        initial_attribution=attribution,
        initial_verification=verification,
        gap_model=gap_model,
        note_model=note_model,
        attribution_model=attribution_model,
        verification_model=verifier,
        tavily_client=tavily,
        budget=EvidenceGapBudget(
            max_tokens=10_000,
            max_cost_usd=1.0,
            max_search_queries=3 if queries else 0,
            max_reads=3,
        ),
        explicit_target_claim_ids=triage.research_target_claim_ids,
        plan_prompt_builder=plan_prompt_builder,
        ledger_event_prefix="recovery_replay",
    )
    recovery = summarize_evidence_recovery(
        triage=triage,
        pass_result=pass_result,
        initial_verification=verification,
        cached_source_urls=tuple(payload["ledger"]["source_cache"]),
    )
    calls = {
        "gap_model": gap_model.calls,
        "note_model": note_model.calls,
        "attribution_model": attribution_model.calls,
        "verification_model": verifier.calls,
        "tavily": tavily.calls,
    }
    return recovery, calls


def _scenario_payload(recovery: Any, calls: Mapping[str, int]) -> dict[str, Any]:
    pass_result = recovery.pass_result
    return {
        "stop_reason": recovery.stop_reason.value,
        "stop_detail": recovery.stop_detail,
        "frozen_target_claim_ids": list(recovery.frozen_target_claim_ids),
        "gap_target_claim_ids": list(pass_result.target_claim_ids),
        "frozen_equals_triage_research_more": (
            recovery.frozen_target_claim_ids
            == recovery.triage.research_target_claim_ids
        ),
        "gap_targets_equal_frozen": (
            pass_result.target_claim_ids
            == recovery.frozen_target_claim_ids
        ),
        "attempted_claim_ids": list(recovery.attempted_claim_ids),
        "unattempted_claim_ids": list(recovery.unattempted_claim_ids),
        "added_source_urls": list(pass_result.added_source_urls),
        "new_completed_relation_count": (
            pass_result.information_yield.new_completed_relation_count
        ),
        "new_completed_verdict_counts": dict(
            pass_result.information_yield.new_completed_verdict_counts
        ),
        "attempts": [
            attempt.model_dump(mode="json") for attempt in recovery.attempts
        ],
        "boundary_call_counts": dict(calls),
    }


async def replay_finance_recovery(audit_path: Path) -> dict[str, Any]:
    """Run both offline scenarios and return one machine-readable report."""

    audit_path = _resolve_audit_path(audit_path)
    audit_before = audit_path.read_bytes()
    source_sha256 = _sha256_bytes(audit_before)
    (
        payload,
        draft,
        checklist,
        decomposition,
        attribution,
        verification,
    ) = _load_registry(audit_path)

    triage_model = _ScriptedTriageModel()
    triage = await triage_evidence_recovery(
        draft,
        checklist=checklist,
        verification=verification,
        model_client=triage_model,
        source_cache=payload["ledger"]["source_cache"],
    )
    decisions = _decision_map(triage)
    actual_research = tuple(
        claim_id
        for claim_id in triage.target_claim_ids
        if decisions.get(claim_id) == RecoveryTriageAction.RESEARCH_MORE.value
    )
    actual_edit = tuple(
        claim_id
        for claim_id in triage.target_claim_ids
        if decisions.get(claim_id) == RecoveryTriageAction.EDIT_DIRECTLY.value
    )

    empty_recovery, empty_calls = await _run_gap_scenario(
        payload=payload,
        draft=draft,
        checklist=checklist,
        decomposition=decomposition,
        attribution=attribution,
        verification=verification,
        triage=triage,
        cached_candidates=(),
        queries=(
            {
                "claim_ids": list(triage.research_target_claim_ids),
                "item_id": checklist.items[0].item_id,
                "query": "offline empty-result retrieval probe",
            },
        ),
        mixed_verifier=False,
    )
    hints = _select_cached_hints(
        ledger=_clone_ledger(payload),
        verification=verification,
    )
    mixed_recovery, mixed_calls = await _run_gap_scenario(
        payload=payload,
        draft=draft,
        checklist=checklist,
        decomposition=decomposition,
        attribution=attribution,
        verification=verification,
        triage=triage,
        cached_candidates=hints,
        queries=(),
        mixed_verifier=True,
    )

    citation_counts = Counter(
        claim.claim.citation_requirement.value
        for claim in verification.claims
    )
    result = {
        "replay_mode": "offline_fixed_models_and_retrieval",
        "source_audit": str(audit_path),
        "source_run_id": payload.get("run_id"),
        "source_audit_sha256": source_sha256,
        "source_registry": "posthoc_evidence.pre_edit_evidence",
        "source_registry_profile": {
            "claim_count": len(verification.claims),
            "citation_requirement_counts": dict(sorted(citation_counts.items())),
            "source_cache_count": len(payload["ledger"]["source_cache"]),
            "note_count": len(payload["ledger"]["notes"]),
            "observed_internal_recovery_claim_ids": [
                claim.claim_id for claim in triage.inapplicable_claims
            ],
        },
        "triage": {
            "status": triage.status.value,
            "target_claim_ids": list(triage.target_claim_ids),
            "decisions": decisions,
            "research_more_claim_ids": list(actual_research),
            "edit_directly_claim_ids": list(actual_edit),
            "inapplicable_claims": [
                claim.model_dump(mode="json")
                for claim in triage.inapplicable_claims
            ],
            "expected_research_more_matches": (
                actual_research == _EXPECTED_RESEARCH
            ),
            "expected_edit_directly_present": all(
                decisions.get(claim_id)
                == RecoveryTriageAction.EDIT_DIRECTLY.value
                for claim_id in _EXPECTED_EDIT_DIRECTLY
            ),
        },
        "internal_boundary": (
            {
                "observed_in_source_registry": True,
                "result": "validated_from_historical_registry",
            }
            if triage.inapplicable_claims
            else {
                "observed_in_source_registry": False,
                "result": "not_observable_in_finance_11",
                "probe": await _internal_contract_probe(
                    draft=draft,
                    checklist=checklist,
                    verification=verification,
                ),
            }
        ),
        "empty_retrieval": _scenario_payload(empty_recovery, empty_calls),
        "mixed_verdict_retrieval": {
            "planned_cached_candidates": list(hints),
            **_scenario_payload(mixed_recovery, mixed_calls),
        },
        "source_audit_unchanged": (
            audit_path.read_bytes() == audit_before
        ),
    }
    return result


def _default_output_path(audit_path: Path, run_id: str | None) -> Path:
    stem = run_id or _resolve_audit_path(audit_path).parent.name
    return _BASE / "harness_replays" / f"{stem}.recovery-replay.json"


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
            "Replay recovery triage and orchestration offline with fixed fake "
            "models and no network calls."
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
    result = asyncio.run(replay_finance_recovery(audit_path))
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
