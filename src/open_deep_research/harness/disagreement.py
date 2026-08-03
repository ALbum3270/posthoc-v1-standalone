"""One bounded, non-goal-seeking pass for alternative-source checks."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from open_deep_research.harness.attribution import (
    AttributionModelClient,
    AttributionResult,
    AttributionSettings,
)
from open_deep_research.harness.checklist import ResearchChecklist
from open_deep_research.harness.claims import CitationRequirement, MarkdownBlock
from open_deep_research.harness.evidence_gap import (
    EvidenceGapBudget,
    EvidenceGapResult,
    EvidenceGapStopReason,
    run_evidence_gap_round,
)
from open_deep_research.harness.jsonio import loads_lenient
from open_deep_research.harness.ledger import ResearchLedger
from open_deep_research.harness.tools import TavilyClient
from open_deep_research.harness.verify import (
    ClaimVerification,
    VerificationModelClient,
    VerificationRecordStatus,
    VerificationResult,
    VerificationSettings,
    VerificationVerdict,
)


class DisagreementStopReason(str, Enum):
    """Why the single disagreement-detection pass ended."""

    DISABLED = "disabled"
    NO_ELIGIBLE_CLAIMS = "no_eligible_claims"
    NO_SELECTION = "no_selection"
    COMPLETED = "completed"
    SINGLE_PASS_ENDED_WITH_UNATTEMPTED_SELECTIONS = (
        "single_pass_ended_with_unattempted_selections"
    )
    BUDGET_EXHAUSTED = "budget_exhausted"
    MODEL_ERROR = "model_error"


class DisagreementBudget(BaseModel):
    """Independent pass cap inside the shared post-hoc envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # The former 30k cap was below the 35.9k/44.2k completion estimates
    # observed in finance-15/16.  This is a capacity cap, not a spend target.
    max_tokens: int = Field(default=50_000, ge=0)
    max_cost_usd: float = Field(default=0.06, ge=0.0)
    max_selected_claims: int = Field(default=6, ge=0, le=20)
    max_search_queries: int = Field(default=3, ge=0, le=20)
    max_reads: int = Field(default=3, ge=0, le=20)
    max_results_per_search: int = Field(default=5, ge=1, le=20)
    provider_timeout_seconds: float = Field(default=60.0, ge=1.0, le=60.0)


class PosthocRetrievalBudget(BaseModel):
    """Shared outer cap across evidence-gap and disagreement passes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # The default is the sum of the two independent defaults: evidence gap
    # 60k/$0.10 plus disagreement 50k/$0.06.  The old 60k/$0.10 value merely
    # preserved the pre-disagreement gap cap and had no empirical basis; it
    # made the two quality passes compete by default.
    max_tokens: int = Field(default=110_000, ge=0)
    max_cost_usd: float = Field(default=0.16, ge=0.0)


class PosthocRetrievalAllocation(BaseModel):
    """Code-owned reservation before either post-hoc pass executes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_gap_budget: EvidenceGapBudget | None
    disagreement_reserved_tokens: int = Field(ge=0)
    disagreement_reserved_cost_usd: float = Field(ge=0.0)


def allocate_posthoc_retrieval_budget(
    *,
    shared_budget: PosthocRetrievalBudget | None,
    evidence_gap_budget: EvidenceGapBudget | None,
    disagreement_budget: DisagreementBudget | None,
) -> PosthocRetrievalAllocation:
    """Reserve disagreement capacity before admitting evidence-gap work."""

    if shared_budget is None:
        return PosthocRetrievalAllocation(
            evidence_gap_budget=evidence_gap_budget,
            disagreement_reserved_tokens=0,
            disagreement_reserved_cost_usd=0.0,
        )
    reserved_tokens = (
        min(shared_budget.max_tokens, disagreement_budget.max_tokens)
        if disagreement_budget is not None
        else 0
    )
    reserved_cost = (
        min(shared_budget.max_cost_usd, disagreement_budget.max_cost_usd)
        if disagreement_budget is not None
        else 0.0
    )
    if evidence_gap_budget is None:
        admitted_gap_budget = None
    else:
        admitted_gap_budget = evidence_gap_budget.model_copy(
            update={
                "max_tokens": min(
                    evidence_gap_budget.max_tokens,
                    max(0, shared_budget.max_tokens - reserved_tokens),
                ),
                "max_cost_usd": min(
                    evidence_gap_budget.max_cost_usd,
                    max(0.0, shared_budget.max_cost_usd - reserved_cost),
                ),
            }
        )
    return PosthocRetrievalAllocation(
        evidence_gap_budget=admitted_gap_budget,
        disagreement_reserved_tokens=reserved_tokens,
        disagreement_reserved_cost_usd=reserved_cost,
    )


class DisagreementCallUsage(BaseModel):
    """One model call charged to disagreement detection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str
    prompt_chars: int = Field(ge=0)
    estimated_input_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0.0)
    token_count: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)


class DisagreementSelection(BaseModel):
    """A model-selected claim worth checking against an alternative source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    reason: str


class DisagreementSearchAttempt(BaseModel):
    """Auditable proof that one frozen claim was actually challenged."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    selection_reason: str
    methods: tuple[str, ...]
    search_queries: tuple[str, ...] = ()
    cached_source_ids: tuple[str, ...] = ()
    search_errors: tuple[str, ...] = ()
    new_completed_relation_count: int = Field(default=0, ge=0)
    completed_verdict_counts: dict[str, int] = Field(default_factory=dict)


class PosthocRetrievalBudgetAudit(BaseModel):
    """Mechanical accounting for the shared post-hoc retrieval cap."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    configured: bool
    max_tokens: int = Field(ge=0)
    max_cost_usd: float = Field(ge=0.0)
    evidence_gap_tokens: int = Field(ge=0)
    evidence_gap_cost_usd: float = Field(ge=0.0)
    disagreement_tokens: int = Field(ge=0)
    disagreement_cost_usd: float = Field(ge=0.0)
    disagreement_reserved_tokens: int = Field(default=0, ge=0)
    disagreement_reserved_cost_usd: float = Field(default=0.0, ge=0.0)
    evidence_gap_admission_max_tokens: int = Field(default=0, ge=0)
    evidence_gap_admission_max_cost_usd: float = Field(default=0.0, ge=0.0)
    remaining_tokens: int = Field(ge=0)
    remaining_cost_usd: float = Field(ge=0.0)
    within_shared_budget: bool
    allocation_method: str = (
        "reserve_disagreement_then_admit_evidence_gap"
    )
    enforcement_limitations: tuple[str, ...] = (
        "calls are admitted using calibrated estimates before provider usage "
        "is known",
        "measured usage can exceed an estimate and is reported rather than "
        "silently discarded",
    )

    @model_validator(mode="after")
    def _reservation_fits_shared_cap(self) -> PosthocRetrievalBudgetAudit:
        if not self.configured:
            return self
        if self.disagreement_reserved_tokens > self.max_tokens:
            raise ValueError("disagreement token reserve exceeds shared cap")
        if self.disagreement_reserved_cost_usd > self.max_cost_usd:
            raise ValueError("disagreement cost reserve exceeds shared cap")
        if (
            self.evidence_gap_admission_max_tokens
            + self.disagreement_reserved_tokens
            > self.max_tokens
        ):
            raise ValueError("post-hoc token allocations exceed shared cap")
        if (
            self.evidence_gap_admission_max_cost_usd
            + self.disagreement_reserved_cost_usd
            > self.max_cost_usd + 1e-12
        ):
            raise ValueError("post-hoc cost allocations exceed shared cap")
        return self


class DisagreementResult(BaseModel):
    """Audit of one bounded pass; conflict count is never its success unit."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    selected_claims: tuple[DisagreementSelection, ...] = ()
    rejected_selections: tuple[dict[str, Any], ...] = ()
    disagreement_search_attempted: tuple[
        DisagreementSearchAttempt, ...
    ] = ()
    completed_verdict_counts: dict[str, int] = Field(default_factory=dict)
    new_completed_relation_count: int = Field(default=0, ge=0)
    conflict_count_is_success_metric: bool = False
    success_unit: str = "attempted_claim_with_audited_search_or_cache_review"
    zero_conflicts_interpretation: str = (
        "no conflict found in attempted checks; this does not establish "
        "absence of disagreement"
    )
    usage: tuple[DisagreementCallUsage, ...] = ()
    stop_reason: DisagreementStopReason
    stop_detail: str
    claim_registry_unchanged: bool = True
    canonical_draft_unchanged: bool = True
    acquisition: EvidenceGapResult | None = None
    final_attribution: AttributionResult = Field(exclude=True)
    final_verification: VerificationResult = Field(exclude=True)

    @property
    def total_tokens(self) -> int:
        return sum(call.token_count for call in self.usage)

    @property
    def total_cost_usd(self) -> float:
        return sum(call.cost_usd for call in self.usage)


_SELECTION_PROMPT = """\
Run one bounded disagreement-detection selection pass. Return json only.
The report and every claim are frozen.

Choose at most {max_claims} external claims for which checking an alternative
source, measurement convention, causal account, attribution, or scope would
be informative. Select a claim because an alternative check is useful, never
because you predict that it will produce a contradiction. Do not optimize the
number of conflicts. Selecting fewer claims or none is legal.

The success unit is that a selected claim is actually checked and the result
is recorded. supports, contradicts, does_not_support, and
not_enough_information are equally valid information outcomes. Zero conflicts
is a normal result and never establishes that no disagreement exists.

Return:
{{"claims":[{{"claim_id":"claim-0001",\
"reason":"why an alternative-source check is informative"}}]}}

External claim registry:
{claims}
"""


_PLAN_PROMPT = """\
Run exactly one bounded alternative-source planning pass. Return json only.
A candidate is only a source worth checking later, never a support verdict.
Claims and report wording are frozen.

For each selected claim, first consider unused cached notes, then web search.
Seek an alternative publishing source, measurement convention, causal
account, attribution, or scope. Do not seek a desired verdict and do not
optimize for contradictions. supports, contradicts, does_not_support, and
not_enough_information are equally valid outcomes.

You have at most {max_queries} web queries. This is an upper bound, not a
target. One focused query may serve several claims when their alternative
check genuinely overlaps. Decide whether a cached candidate or query is useful
for each selected claim; the code does not require every query slot to be used
or decide what source would be informative. A selected claim with no accepted
route remains explicitly unattempted, which is not a conclusion about whether
disagreement exists.

Return only accepted cached candidates and proposed queries. Do not return
deferred_targets. Code records every unrouted selected claim as
query_capacity_not_allocated after parsing the plan. This is a mechanical
capacity fact, not a semantic conclusion about the claim.

Return:
{{"cached_candidates":[{{"claim_id":"claim-0001",\
"note_id":"note-000001","source_id":"source-id",\
"independent_from_existing_publishers":true,\
"publisher_identity":"publishing organization",\
"independence_rationale":"brief reason"}}],\
"queries":[{{"claim_ids":["claim-0001"],\
"item_id":"existing-checklist-item","query":"one focused query"}}]}}

Allowed checklist item IDs:
{item_ids}

Selected claims:
{targets}

Compact cached-note registry:
{notes}
"""


def build_disagreement_selection_prompt(
    claims: Sequence[ClaimVerification],
    *,
    max_claims: int,
) -> str:
    """Ask the model which claims merit a neutral alternative-source check."""

    payload = [
        {
            "claim_id": entry.claim.claim_id,
            "claim_text": entry.claim.claim_text,
            "evidence_state": entry.state.value,
            "publisher_domain_proxies": list(
                entry.publisher_domain_proxies
            ),
            "completed_verdicts": [
                relation.semantic_verdict.value
                for relation in entry.relations
                if (
                    relation.status is VerificationRecordStatus.COMPLETED
                    and relation.semantic_verdict is not None
                )
            ],
        }
        for entry in claims
        if entry.claim.citation_requirement is CitationRequirement.EXTERNAL
    ]
    return _SELECTION_PROMPT.format(
        max_claims=max_claims,
        claims=json.dumps(payload, ensure_ascii=False, sort_keys=True),
    )


def build_disagreement_plan_prompt(
    *,
    targets: Sequence[ClaimVerification],
    notes: Sequence[Any],
    checklist: ResearchChecklist,
    max_queries: int,
) -> str:
    """Plan cache-first checks without preferring any semantic verdict."""

    note_registry = [
        {
            "note_id": note.note_id,
            "source_id": note.source_id,
            "item_id": note.item_id,
            "finding": note.finding,
            "publisher": note.publisher,
            "location_status": note.location_status.value,
        }
        for note in notes
    ]
    target_payload = [
        {
            "claim_id": target.claim.claim_id,
            "claim_text": target.claim.claim_text,
            "existing_publisher_domain_proxies": list(
                target.publisher_domain_proxies
            ),
        }
        for target in targets
    ]
    return _PLAN_PROMPT.format(
        max_queries=max_queries,
        item_ids=json.dumps(
            [item.item_id for item in checklist.items],
            ensure_ascii=False,
        ),
        targets=json.dumps(
            target_payload, ensure_ascii=False, sort_keys=True
        ),
        notes=json.dumps(note_registry, ensure_ascii=False, sort_keys=True),
    )


def _estimate(
    client: Any,
    prompt: str,
    estimator: Callable[[Any, str], int | float] | None,
    method_name: str,
) -> int | float:
    if estimator is not None:
        return estimator(client, prompt)
    method = getattr(client, method_name, None)
    if not callable(method):
        raise RuntimeError(
            f"disagreement {method_name} admission estimator is unavailable"
        )
    return method(prompt)


def _response_envelope(response: Any) -> tuple[Any, int, float]:
    if not isinstance(response, Mapping):
        raise ValueError("model response must be a usage envelope")
    return (
        response.get("content"),
        max(0, int(response.get("token_count", 0))),
        max(0.0, float(response.get("cost_usd", 0.0))),
    )


def _decode_content(content: Any) -> Any:
    return loads_lenient(content) if isinstance(content, str) else content


def _decode_diagnostic(
    content: Any,
    error: Exception,
    *,
    attempt: int,
) -> dict[str, Any]:
    """Retain a mechanical failure record without inventing missing JSON."""

    text = content if isinstance(content, str) else ""
    stripped = text.rstrip()
    return {
        "stage": "disagreement_selection_decode",
        "attempt": attempt,
        "error": f"{type(error).__name__}: {error}",
        "response_chars": len(text),
        "ended_with_json_closer": stripped.endswith(("}", "]")),
    }


def _selection_retry_prompt(
    base_prompt: str,
    diagnostic: Mapping[str, Any],
) -> str:
    """Request one complete replacement after mechanical JSON failure."""

    return (
        base_prompt
        + "\n\nMECHANICAL JSON DECODING FAILED. Return one complete replacement "
        "JSON object, not a patch or continuation. The prior response was "
        "charged but could not be decoded. Diagnostic:\n"
        + json.dumps(dict(diagnostic), ensure_ascii=False, sort_keys=True)
    )


def _parse_selection(
    content: Any,
    *,
    eligible_claim_ids: set[str],
    max_claims: int,
) -> tuple[
    tuple[DisagreementSelection, ...],
    tuple[dict[str, Any], ...],
]:
    raw_entries = content.get("claims") if isinstance(content, Mapping) else None
    if not isinstance(raw_entries, (list, tuple)):
        return (), (
            {
                "stage": "disagreement_selection",
                "error": "claims must be an array",
            },
        )
    accepted: list[DisagreementSelection] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_entries):
        try:
            if not isinstance(raw, Mapping):
                raise ValueError("selection entry must be an object")
            claim_id = str(raw["claim_id"]).strip()
            reason = str(raw["reason"]).strip()
            if not reason:
                raise ValueError("reason must not be blank")
            if claim_id not in eligible_claim_ids:
                raise ValueError("claim_id is not an external claim")
            if claim_id in seen:
                raise ValueError("duplicate claim_id")
            if len(accepted) >= max_claims:
                raise ValueError("selection cap reached")
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append(
                {
                    "stage": "disagreement_selection",
                    "index": index,
                    "error": str(exc),
                    "raw": raw,
                }
            )
            continue
        seen.add(claim_id)
        accepted.append(
            DisagreementSelection(claim_id=claim_id, reason=reason)
        )
    return tuple(accepted), tuple(rejected)


def _completed_relations(
    verification: VerificationResult,
) -> dict[tuple[str, str, str], Any]:
    return {
        (relation.claim_id, relation.source_id, relation.url): relation
        for entry in verification.claims
        for relation in entry.relations
        if relation.status is VerificationRecordStatus.COMPLETED
    }


def _attempts(
    selections: Sequence[DisagreementSelection],
    acquisition: EvidenceGapResult,
    initial_verification: VerificationResult,
) -> tuple[DisagreementSearchAttempt, ...]:
    before = _completed_relations(initial_verification)
    after = _completed_relations(acquisition.final_verification)
    newly_completed = {
        identity: relation
        for identity, relation in after.items()
        if identity not in before
    }
    by_id = {selection.claim_id: selection for selection in selections}
    attempts: list[DisagreementSearchAttempt] = []
    for claim_id, selection in by_id.items():
        hints = tuple(
            hint
            for hint in acquisition.cached_candidate_hints
            if hint.claim_id == claim_id
        )
        searches = tuple(
            search
            for search in acquisition.searches
            if claim_id in search.query.claim_ids
        )
        if not hints and not searches:
            continue
        relations = tuple(
            relation
            for identity, relation in newly_completed.items()
            if identity[0] == claim_id
        )
        counts = {
            verdict.value: sum(
                relation.semantic_verdict is verdict
                for relation in relations
            )
            for verdict in VerificationVerdict
        }
        methods = []
        if hints:
            methods.append("cached_source_review")
        if any(search.error is None for search in searches):
            methods.append("web_search")
        attempts.append(
            DisagreementSearchAttempt(
                claim_id=claim_id,
                selection_reason=selection.reason,
                methods=tuple(methods),
                search_queries=tuple(
                    search.query.query for search in searches
                ),
                cached_source_ids=tuple(
                    dict.fromkeys(hint.source_id for hint in hints)
                ),
                search_errors=tuple(
                    search.error
                    for search in searches
                    if search.error is not None
                ),
                new_completed_relation_count=len(relations),
                completed_verdict_counts=counts,
            )
        )
    return tuple(attempts)


def disabled_disagreement_result(
    attribution: AttributionResult,
    verification: VerificationResult,
    *,
    detail: str = "no disagreement-detection budget was configured",
) -> DisagreementResult:
    """Construct an explicit disabled result without a model or network call."""

    return DisagreementResult(
        stop_reason=DisagreementStopReason.DISABLED,
        stop_detail=detail,
        final_attribution=attribution,
        final_verification=verification,
    )


async def run_disagreement_detection(
    *,
    canonical_draft: str,
    checklist: ResearchChecklist,
    blocks: Sequence[MarkdownBlock],
    ledger: ResearchLedger,
    initial_attribution: AttributionResult,
    initial_verification: VerificationResult,
    selection_model: Any,
    note_model: Any,
    attribution_model: AttributionModelClient,
    verification_model: VerificationModelClient,
    tavily_client: TavilyClient,
    budget: DisagreementBudget,
    attribution_settings: AttributionSettings | None = None,
    verification_settings: VerificationSettings | None = None,
    corroboration_targets: Mapping[str, int] | None = None,
    estimate_input_tokens: Callable[[Any, str], int] | None = None,
    estimate_cost_usd: Callable[[Any, str], float] | None = None,
) -> DisagreementResult:
    """Select and attempt neutral alternative-source checks exactly once."""

    eligible = tuple(
        entry
        for entry in initial_verification.claims
        if entry.claim.citation_requirement is CitationRequirement.EXTERNAL
    )
    if not eligible:
        return DisagreementResult(
            stop_reason=DisagreementStopReason.NO_ELIGIBLE_CLAIMS,
            stop_detail="no external claims were eligible for selection",
            final_attribution=initial_attribution,
            final_verification=initial_verification,
        )
    frozen_claims = {
        entry.claim.claim_id: entry.claim.model_dump(mode="json")
        for entry in initial_attribution.attributions
    }
    base_prompt = build_disagreement_selection_prompt(
        eligible,
        max_claims=budget.max_selected_claims,
    )
    selection_usage: list[DisagreementCallUsage] = []
    rejected: list[dict[str, Any]] = []
    selections: tuple[DisagreementSelection, ...] = ()
    selection_prompt = base_prompt
    for attempt in range(1, 3):
        try:
            estimated_tokens = max(
                0,
                int(
                    _estimate(
                        selection_model,
                        selection_prompt,
                        estimate_input_tokens,
                        "estimate_tokens",
                    )
                ),
            )
            estimated_cost = max(
                0.0,
                float(
                    _estimate(
                        selection_model,
                        selection_prompt,
                        estimate_cost_usd,
                        "estimate_cost_usd",
                    )
                ),
            )
        except Exception as exc:
            return DisagreementResult(
                rejected_selections=tuple(rejected),
                usage=tuple(selection_usage),
                stop_reason=DisagreementStopReason.MODEL_ERROR,
                stop_detail=f"{type(exc).__name__}: {exc}",
                final_attribution=initial_attribution,
                final_verification=initial_verification,
            )
        if (
            sum(call.token_count for call in selection_usage)
            + estimated_tokens
            > budget.max_tokens
            or sum(call.cost_usd for call in selection_usage)
            + estimated_cost
            > budget.max_cost_usd
        ):
            return DisagreementResult(
                rejected_selections=tuple(rejected),
                usage=tuple(selection_usage),
                stop_reason=DisagreementStopReason.BUDGET_EXHAUSTED,
                stop_detail=(
                    "selection admission exceeds remaining disagreement "
                    "pass budget"
                ),
                final_attribution=initial_attribution,
                final_verification=initial_verification,
            )
        try:
            response = selection_model.generate(selection_prompt)
            if inspect.isawaitable(response):
                response = await response
            raw_content, tokens, cost = _response_envelope(response)
        except Exception as exc:
            return DisagreementResult(
                rejected_selections=tuple(rejected),
                usage=tuple(selection_usage),
                stop_reason=DisagreementStopReason.MODEL_ERROR,
                stop_detail=f"{type(exc).__name__}: {exc}",
                final_attribution=initial_attribution,
                final_verification=initial_verification,
            )
        call_usage = DisagreementCallUsage(
            stage=(
                "disagreement_selection"
                if attempt == 1
                else "disagreement_selection_retry"
            ),
            prompt_chars=len(selection_prompt),
            estimated_input_tokens=estimated_tokens,
            estimated_cost_usd=estimated_cost,
            token_count=tokens,
            cost_usd=cost,
        )
        selection_usage.append(call_usage)
        try:
            content = _decode_content(raw_content)
        except Exception as exc:
            diagnostic = _decode_diagnostic(
                raw_content,
                exc,
                attempt=attempt,
            )
            rejected.append(diagnostic)
            if attempt == 1:
                selection_prompt = _selection_retry_prompt(
                    base_prompt,
                    diagnostic,
                )
                continue
            return DisagreementResult(
                rejected_selections=tuple(rejected),
                usage=tuple(selection_usage),
                stop_reason=DisagreementStopReason.MODEL_ERROR,
                stop_detail=(
                    "disagreement selection JSON remained undecodable after "
                    "one bounded retry"
                ),
                final_attribution=initial_attribution,
                final_verification=initial_verification,
            )
        selections, parsed_rejected = _parse_selection(
            content,
            eligible_claim_ids={
                entry.claim.claim_id for entry in eligible
            },
            max_claims=budget.max_selected_claims,
        )
        rejected.extend(parsed_rejected)
        break
    if not selections:
        return DisagreementResult(
            rejected_selections=tuple(rejected),
            usage=tuple(selection_usage),
            stop_reason=DisagreementStopReason.NO_SELECTION,
            stop_detail=(
                "bounded selection completed with no claims selected; "
                "this does not establish absence of disagreement"
            ),
            final_attribution=initial_attribution,
            final_verification=initial_verification,
        )

    remaining_tokens = max(
        0,
        budget.max_tokens
        - sum(call.token_count for call in selection_usage),
    )
    remaining_cost = max(
        0.0,
        budget.max_cost_usd
        - sum(call.cost_usd for call in selection_usage),
    )
    acquisition = await run_evidence_gap_round(
        canonical_draft=canonical_draft,
        checklist=checklist,
        blocks=blocks,
        ledger=ledger,
        initial_attribution=initial_attribution,
        initial_verification=initial_verification,
        gap_model=selection_model,
        note_model=note_model,
        attribution_model=attribution_model,
        verification_model=verification_model,
        tavily_client=tavily_client,
        budget=EvidenceGapBudget(
            max_tokens=remaining_tokens,
            max_cost_usd=remaining_cost,
            max_search_queries=budget.max_search_queries,
            max_reads=budget.max_reads,
            max_results_per_search=budget.max_results_per_search,
            provider_timeout_seconds=budget.provider_timeout_seconds,
        ),
        attribution_settings=attribution_settings,
        verification_settings=verification_settings,
        corroboration_targets=corroboration_targets,
        estimate_input_tokens=estimate_input_tokens,
        estimate_cost_usd=estimate_cost_usd,
        explicit_target_claim_ids=[
            selection.claim_id for selection in selections
        ],
        plan_prompt_builder=build_disagreement_plan_prompt,
        ledger_event_prefix="disagreement",
    )
    usage = (
        *selection_usage,
        *(
            DisagreementCallUsage(
                stage=call.stage,
                prompt_chars=call.prompt_chars,
                estimated_input_tokens=call.estimated_input_tokens,
                estimated_cost_usd=call.estimated_cost_usd,
                token_count=call.token_count,
                cost_usd=call.cost_usd,
            )
            for call in acquisition.usage
        ),
    )
    attempts = _attempts(
        selections,
        acquisition,
        initial_verification,
    )
    counts = {
        verdict.value: sum(
            attempt.completed_verdict_counts.get(verdict.value, 0)
            for attempt in attempts
        )
        for verdict in VerificationVerdict
    }
    mapped_stop = {
        EvidenceGapStopReason.COMPLETED: DisagreementStopReason.COMPLETED,
        EvidenceGapStopReason.BUDGET_EXHAUSTED: (
            DisagreementStopReason.BUDGET_EXHAUSTED
        ),
        EvidenceGapStopReason.MODEL_ERROR: DisagreementStopReason.MODEL_ERROR,
        EvidenceGapStopReason.NO_TARGETS: (
            DisagreementStopReason.NO_SELECTION
        ),
        EvidenceGapStopReason.DISABLED: DisagreementStopReason.DISABLED,
    }[acquisition.stop_reason]
    attempted_ids = {
        attempt.claim_id for attempt in attempts if attempt.methods
    }
    unattempted_ids = tuple(
        selection.claim_id
        for selection in selections
        if selection.claim_id not in attempted_ids
    )
    if (
        mapped_stop is DisagreementStopReason.COMPLETED
        and unattempted_ids
    ):
        mapped_stop = (
            DisagreementStopReason.SINGLE_PASS_ENDED_WITH_UNATTEMPTED_SELECTIONS
        )
        stop_detail = (
            "the only bounded disagreement pass ended with selected claims "
            "that received neither an accepted cached candidate nor an issued "
            "search route: "
            + ", ".join(unattempted_ids)
        )
    elif mapped_stop is DisagreementStopReason.COMPLETED:
        stop_detail = (
            f"single bounded pass attempted {len(attempted_ids)} claim(s); "
            f"new completed relations={sum(counts.values())}; "
            f"conflicts found={counts[VerificationVerdict.CONTRADICTS.value]}. "
            "Zero conflicts is normal and does not establish absence of "
            "disagreement."
        )
    else:
        stop_detail = acquisition.stop_detail
    current_claims = {
        entry.claim.claim_id: entry.claim.model_dump(mode="json")
        for entry in acquisition.final_attribution.attributions
    }
    unchanged = current_claims == frozen_claims
    if not unchanged:
        raise AssertionError(
            "disagreement detection mutated the frozen claim registry"
        )
    return DisagreementResult(
        selected_claims=selections,
        rejected_selections=tuple(rejected),
        disagreement_search_attempted=attempts,
        completed_verdict_counts=counts,
        new_completed_relation_count=sum(counts.values()),
        usage=usage,
        stop_reason=mapped_stop,
        stop_detail=stop_detail,
        claim_registry_unchanged=unchanged,
        canonical_draft_unchanged=True,
        acquisition=acquisition,
        final_attribution=acquisition.final_attribution,
        final_verification=acquisition.final_verification,
    )


def shared_posthoc_budget_audit(
    *,
    budget: PosthocRetrievalBudget | None,
    evidence_gap_tokens: int,
    evidence_gap_cost_usd: float,
    disagreement_tokens: int,
    disagreement_cost_usd: float,
    disagreement_reserved_tokens: int = 0,
    disagreement_reserved_cost_usd: float = 0.0,
    evidence_gap_admission_max_tokens: int = 0,
    evidence_gap_admission_max_cost_usd: float = 0.0,
) -> PosthocRetrievalBudgetAudit:
    """Prove two individually bounded passes also respect their shared cap."""

    total_tokens = evidence_gap_tokens + disagreement_tokens
    total_cost = evidence_gap_cost_usd + disagreement_cost_usd
    max_tokens = budget.max_tokens if budget is not None else total_tokens
    max_cost = budget.max_cost_usd if budget is not None else total_cost
    return PosthocRetrievalBudgetAudit(
        configured=budget is not None,
        max_tokens=max_tokens,
        max_cost_usd=max_cost,
        evidence_gap_tokens=evidence_gap_tokens,
        evidence_gap_cost_usd=evidence_gap_cost_usd,
        disagreement_tokens=disagreement_tokens,
        disagreement_cost_usd=disagreement_cost_usd,
        disagreement_reserved_tokens=disagreement_reserved_tokens,
        disagreement_reserved_cost_usd=disagreement_reserved_cost_usd,
        evidence_gap_admission_max_tokens=evidence_gap_admission_max_tokens,
        evidence_gap_admission_max_cost_usd=(
            evidence_gap_admission_max_cost_usd
        ),
        remaining_tokens=max(0, max_tokens - total_tokens),
        remaining_cost_usd=max(0.0, max_cost - total_cost),
        within_shared_budget=(
            total_tokens <= max_tokens and total_cost <= max_cost
        ),
    )
