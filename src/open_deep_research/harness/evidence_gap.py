"""One bounded post-verification pass for unresolved evidence gaps."""

from __future__ import annotations

import inspect
import json
import math
import unicodedata
from collections.abc import Awaitable, Callable, Mapping, Sequence
from enum import Enum
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from open_deep_research.harness.attribution import (
    AttributionError,
    AttributionModelClient,
    AttributionResult,
    AttributionSettings,
    AttributionStatus,
    AttributionStopReason,
    CandidateSource,
    ClaimAttribution,
    attribute_claims,
    build_attribution_prompt,
)
from open_deep_research.harness.budget import RunCostCapReached
from open_deep_research.harness.checklist import ResearchChecklist
from open_deep_research.harness.claims import (
    AtomicClaim,
    CitationRequirement,
    MarkdownBlock,
    SourceResolution,
)
from open_deep_research.harness.jsonio import loads_lenient
from open_deep_research.harness.ledger import ResearchLedger
from open_deep_research.harness.notes import (
    ResearchNote,
    create_note,
    source_id_for_url,
)
from open_deep_research.harness.source_provenance import SourceLineageStatus
from open_deep_research.harness.tools import (
    SearchResult,
    TavilyClient,
    read_with_links,
    search,
)
from open_deep_research.harness.truth_conditions import (
    TruthConditionRegistry,
    aggregate_truth_condition_claim,
    select_truth_condition_registry,
    truth_condition_registry_sha256,
)
from open_deep_research.harness.verify import (
    ClaimEvidenceState,
    ClaimVerification,
    VerificationBudget,
    VerificationModelClient,
    VerificationRecordStatus,
    VerificationResult,
    VerificationSettings,
    VerificationVerdict,
    PublisherIndependenceAudit,
    VerifiedSourceRelation,
    build_claim_verification,
    build_verification_prompt,
    verify_attributions,
)

_TARGET_STATES = {
    ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    ClaimEvidenceState.SUPPORTED_SINGLE_DOMAIN_PROXY,
    ClaimEvidenceState.SUPPORTED_MULTIPLE_DOMAIN_PROXIES,
    ClaimEvidenceState.SUPPORTED_DISTRIBUTED_ELEMENT_EVIDENCE,
    ClaimEvidenceState.CONFLICTING_EVIDENCE,
}

_TERMINAL_RELATION_STATUSES = {
    VerificationRecordStatus.COMPLETED,
    VerificationRecordStatus.QUOTE_UNLOCATABLE,
}


def select_evidence_gap_targets(
    initial_verification: VerificationResult,
) -> tuple[ClaimVerification, ...]:
    """Return the default frozen denominator for one evidence-gap pass."""

    return tuple(
        result
        for result in initial_verification.claims
        if (
            result.claim.citation_requirement is CitationRequirement.EXTERNAL
            and result.state in _TARGET_STATES
            and (
                result.state
                not in {
                    ClaimEvidenceState.SUPPORTED_SINGLE_DOMAIN_PROXY,
                    ClaimEvidenceState.SUPPORTED_MULTIPLE_DOMAIN_PROXIES,
                    ClaimEvidenceState.SUPPORTED_DISTRIBUTED_ELEMENT_EVIDENCE,
                }
                or result.publisher_domain_proxy_count
                < result.corroboration_target
            )
        )
    )


class EvidenceGapStopReason(str, Enum):
    """Why the single evidence-gap pass ended."""

    DISABLED = "disabled"
    NO_TARGETS = "no_targets"
    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MODEL_ERROR = "model_error"


class EvidenceGapBudget(BaseModel):
    """Independent hard envelope and network caps for one gap pass."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_tokens: int = Field(default=60_000, ge=0)
    max_cost_usd: float = Field(default=0.10, ge=0.0)
    max_search_queries: int = Field(default=6, ge=0, le=20)
    max_reads: int = Field(default=3, ge=0, le=20)
    max_results_per_search: int = Field(default=5, ge=1, le=20)
    provider_timeout_seconds: float = Field(default=60.0, ge=1.0, le=60.0)
    max_planning_input_tokens: int = Field(default=36_000, ge=0)
    max_planning_prompt_chars: int = Field(default=120_000, ge=1)
    planning_output_headroom_ratio: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
    )
    downstream_action_source_chars: int = Field(
        default=4_096,
        ge=1,
        le=100_000,
    )


class EvidenceGapCallUsage(BaseModel):
    """One admitted model call inside the gap envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_number: int = Field(ge=1)
    stage: str
    prompt_chars: int = Field(ge=0)
    estimated_input_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0.0)
    token_count: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)


class EvidenceGapPlanningCapacityAudit(BaseModel):
    """Capacity proof made before the planning model may spend the pass.

    Search results and page lengths do not exist yet at this boundary.  The
    reserve therefore protects one mechanically executable candidate route:
    exact cached-source verification, a query-only investigation when reads
    are disabled, or bounded web read selection followed by note extraction,
    incremental attribution, and verification. This proves capacity, not
    semantic relevance. Exact downstream prompts and source text replace the
    probe as soon as concrete routes exist.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_format_version: str = "compact-gap-plan-v2"
    advertised_max_search_queries: int = Field(default=0, ge=0)
    target_count: int = Field(ge=0)
    cached_note_count: int = Field(ge=0)
    prompt_chars: int = Field(ge=0)
    estimated_planning_input_tokens: int = Field(ge=0)
    estimated_planning_cost_usd: float = Field(ge=0.0)
    max_planning_input_tokens: int = Field(ge=0)
    max_planning_prompt_chars: int = Field(ge=1)
    planning_output_headroom_tokens: int = Field(ge=0)
    planning_output_headroom_cost_usd: float = Field(ge=0.0)
    downstream_action_source_chars: int = Field(ge=1)
    read_selection_estimated_tokens: int = Field(ge=0)
    read_selection_estimated_cost_usd: float = Field(ge=0.0)
    reattribution_estimated_tokens: int = Field(ge=0)
    reattribution_estimated_cost_usd: float = Field(ge=0.0)
    note_and_verification_estimated_tokens: int = Field(ge=0)
    note_and_verification_estimated_cost_usd: float = Field(ge=0.0)
    downstream_action_estimated_tokens: int = Field(ge=0)
    downstream_action_estimated_cost_usd: float = Field(ge=0.0)
    web_downstream_action_estimated_tokens: int = Field(default=0, ge=0)
    web_downstream_action_estimated_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
    )
    reserved_tokens: int = Field(ge=0)
    reserved_cost_usd: float = Field(ge=0.0)
    reserve_fully_funded: bool
    limitations: tuple[str, ...] = (
        "the selected page length is unknown until the free network read",
        "the minimum action reserve covers one source window, not every read",
        "a model-selected cached source can exceed the bounded action window",
        "provider output/reasoning tokens can exceed admission estimates",
    )


class CachedCandidateHint(BaseModel):
    """A model-selected cached candidate, not a support verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    note_id: str
    source_id: str
    publisher_identity: str
    independence_rationale: str


class GapSearchQuery(BaseModel):
    """One ordered web query that may route evidence for several claims."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_ids: tuple[str, ...] = Field(min_length=1)
    target_element_ids: tuple[str, ...] = ()
    item_id: str
    query: str

    @field_validator("target_element_ids")
    @classmethod
    def _element_ids_are_unique(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        if len(set(value)) != len(value) or any(not item.strip() for item in value):
            raise ValueError("target_element_ids must be unique and non-blank")
        return value

    @model_validator(mode="after")
    def _element_ids_belong_to_claims(self) -> GapSearchQuery:
        if any(
            element_id.split("::tc-", 1)[0] not in set(self.claim_ids)
            for element_id in self.target_element_ids
        ):
            raise ValueError("target elements must belong to routed claims")
        return self


class DeferredGapTarget(BaseModel):
    """A target left without a successfully completed acquisition route.

    This is a capacity record, not a semantic judgement that the claim needs
    no evidence or cannot be researched.  Those conclusions require actual
    evidence work and are deliberately unavailable to the planner.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    reason: Literal[
        "query_capacity_not_allocated",
        "search_route_failed",
    ]
    priority_rationale: str
    allocation_source: Literal["code_derived"] = "code_derived"

    @field_validator("priority_rationale")
    @classmethod
    def _rationale_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("priority_rationale must not be blank")
        return normalized


class GapSearchRecord(BaseModel):
    """Provider results retained before the model chooses reads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: GapSearchQuery
    results: tuple[SearchResult, ...] = ()
    error: str | None = None


class GapReadSelection(BaseModel):
    """A model judgement that one result merits a full read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str
    item_id: str
    claim_ids: tuple[str, ...]
    target_element_ids: tuple[str, ...] = ()
    publisher_identity: str
    independence_rationale: str

    @model_validator(mode="after")
    def _element_ids_belong_to_claims(self) -> GapReadSelection:
        if len(set(self.target_element_ids)) != len(self.target_element_ids):
            raise ValueError("target_element_ids must be unique")
        if any(
            element_id.split("::tc-", 1)[0] not in set(self.claim_ids)
            for element_id in self.target_element_ids
        ):
            raise ValueError("target elements must belong to selected claims")
        return self


class GapSourceAcquisition(BaseModel):
    """One selected URL and its durable source/note outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str
    claim_ids: tuple[str, ...]
    target_element_ids: tuple[str, ...] = ()
    publisher_identity: str
    cache_hit: bool
    source_chars: int = Field(default=0, ge=0)
    note_ids: tuple[str, ...] = ()
    outcome: str
    error: str | None = None

    @model_validator(mode="after")
    def _element_ids_belong_to_claims(self) -> GapSourceAcquisition:
        if len(set(self.target_element_ids)) != len(self.target_element_ids):
            raise ValueError("target_element_ids must be unique")
        if any(
            element_id.split("::tc-", 1)[0] not in set(self.claim_ids)
            for element_id in self.target_element_ids
        ):
            raise ValueError("target elements must belong to acquired claims")
        return self


class ProtectedCompletedRelation(BaseModel):
    """A rejected attempt to replace an already completed source verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    source_id: str
    url: str
    attempted_status: VerificationRecordStatus
    reason: str


class VerificationMergeAudit(BaseModel):
    """Mechanical proof that an evidence-gap merge retained prior work."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    initial_relation_count: int = Field(ge=0)
    initial_completed_relation_count: int = Field(ge=0)
    incremental_relation_count: int = Field(ge=0)
    incremental_completed_relation_count: int = Field(ge=0)
    final_relation_count: int = Field(ge=0)
    final_completed_relation_count: int = Field(ge=0)
    preserved_initial_completed_relation_count: int = Field(ge=0)
    completed_relation_count_non_decreasing: bool
    protected_completed_relations: tuple[ProtectedCompletedRelation, ...] = ()


class VerificationReserveAudit(BaseModel):
    """Admission reserve for downstream verification of known source groups.

    A web result has no reliable source length until it has been read.  This
    audit consequently records only cached candidates and sources whose exact
    text was already read in the current pass; it never extrapolates from the
    largest unrelated cache entry.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str = "actual_cached_and_read_source_groups_after_prerequisite_allowance"
    reference_source_url: str | None = None
    reference_source_chars: int = Field(default=0, ge=0)
    cached_hint_batch_count: int = Field(default=0, ge=0)
    admitted_read_source_batch_count: int = Field(default=0, ge=0)
    admitted_read_source_urls: tuple[str, ...] = ()
    web_read_slots: int = Field(default=0, ge=0)
    planned_query_count: int = Field(default=0, ge=0)
    planned_query_claim_count: int = Field(default=0, ge=0)
    estimated_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    prerequisite_stage: str | None = None
    prerequisite_estimated_tokens: int = Field(default=0, ge=0)
    prerequisite_estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    minimum_action_estimated_tokens: int = Field(default=0, ge=0)
    minimum_action_estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    incremental_reattribution_estimated_tokens: int = Field(default=0, ge=0)
    incremental_reattribution_estimated_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
    )
    required_downstream_tokens: int = Field(default=0, ge=0)
    required_downstream_cost_usd: float = Field(default=0.0, ge=0.0)
    reserved_tokens: int = Field(default=0, ge=0)
    reserved_cost_usd: float = Field(default=0.0, ge=0.0)
    reserve_fully_funded: bool = False
    limitations: tuple[str, ...] = (
        "future source length is unknown until its free network read completes",
        "a source rejected before note extraction creates no verification reserve",
        "later reattribution can create relations outside planned query groups",
        "incremental reattribution protects one estimated turn plus headroom; "
        "later inspect or retry turns require exact admission",
        "verification admission estimates do not predict model output tokens exactly",
    )


class EvidenceGapInformationAudit(BaseModel):
    """Factual yield of one bounded pass, independent of corroboration targets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pass_completed_within_budget: bool = False
    new_completed_relation_count: int = Field(default=0, ge=0)
    new_completed_verdict_counts: dict[str, int] = Field(default_factory=dict)
    new_publisher_domain_proxies: tuple[str, ...] = ()
    new_claim_publisher_relation_count: int = Field(default=0, ge=0)
    claims_newly_corroborated: int = Field(default=0, ge=0)
    claims_newly_supported_by_multiple_domain_proxies: int = Field(
        default=0,
        ge=0,
    )
    claims_newly_conflicting: int = Field(default=0, ge=0)


class EvidenceGapResult(BaseModel):
    """Audit of one non-iterative gap pass plus code-owned final registries."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    target_claim_ids: tuple[str, ...] = ()
    routed_target_claim_ids: tuple[str, ...] = ()
    unrouted_target_claim_ids: tuple[str, ...] = ()
    initial_states: dict[str, ClaimEvidenceState] = Field(default_factory=dict)
    cached_candidate_hints: tuple[CachedCandidateHint, ...] = ()
    deferred_targets: tuple[DeferredGapTarget, ...] = ()
    rejected_entries: tuple[dict[str, Any], ...] = ()
    searches: tuple[GapSearchRecord, ...] = ()
    read_selections: tuple[GapReadSelection, ...] = ()
    acquisitions: tuple[GapSourceAcquisition, ...] = ()
    added_source_urls: tuple[str, ...] = ()
    added_note_ids: tuple[str, ...] = ()
    verification_reserve: VerificationReserveAudit | None = None
    verification_reserve_history: tuple[VerificationReserveAudit, ...] = ()
    planning_capacity: EvidenceGapPlanningCapacityAudit | None = None
    planning_attempt_count: int = Field(default=0, ge=0)
    selected_planning_attempt: int | None = Field(default=None, ge=1)
    unused_query_slots: int = Field(default=0, ge=0)
    information_yield: EvidenceGapInformationAudit = Field(
        default_factory=EvidenceGapInformationAudit
    )
    usage: tuple[EvidenceGapCallUsage, ...] = ()
    stop_reason: EvidenceGapStopReason
    stop_detail: str
    claim_registry_unchanged: bool = True
    canonical_draft_unchanged: bool = True
    independence_method: str = (
        "retrieval_model_screen_then_publisher_domain_proxy_deduplication"
    )
    independence_is_strict: bool = False
    independence_limitations: tuple[str, ...] = (
        "model_screening_can_misidentify_common_ownership",
        "syndicated_or_republished_content_can_be_missed",
        "retrieval screening does not establish final source independence",
        "final corroboration requires separately confirmed source lineage",
    )
    query_planning_method: str = (
        "model_routes_with_code_derived_unrouted_audit"
    )
    query_merge_precision_status: str = "pending_observation"
    query_merge_precision_limitations: tuple[str, ...] = (
        "no in-repository A/B currently measures merged-query precision",
        "a broader merged query may trade per-claim precision for coverage",
    )
    verification_merge: VerificationMergeAudit | None = None
    final_attribution: AttributionResult = Field(exclude=True)
    final_verification: VerificationResult = Field(exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _derive_target_routes(cls, value: Any) -> Any:
        """Derive routing coverage from accepted, actually retained routes.

        A target is routed only when it has an accepted cached-note candidate
        or a search record without a provider error.  Merely belonging to the requested target
        set is not work done.  Keeping this derivation inside the result model
        also makes hand-built offline/recovery results obey the same audit
        contract as the live executor.
        """

        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        target_ids = tuple(data.get("target_claim_ids") or ())
        routed: set[str] = set()
        for hint in data.get("cached_candidate_hints") or ():
            claim_id = (
                hint.get("claim_id")
                if isinstance(hint, Mapping)
                else getattr(hint, "claim_id", None)
            )
            if claim_id is not None:
                routed.add(str(claim_id))
        for record in data.get("searches") or ():
            error = (
                record.get("error")
                if isinstance(record, Mapping)
                else getattr(record, "error", None)
            )
            if error is not None:
                continue
            query = (
                record.get("query")
                if isinstance(record, Mapping)
                else getattr(record, "query", None)
            )
            claim_ids = (
                query.get("claim_ids")
                if isinstance(query, Mapping)
                else getattr(query, "claim_ids", ())
            )
            routed.update(str(claim_id) for claim_id in claim_ids or ())
        data["routed_target_claim_ids"] = tuple(
            claim_id for claim_id in target_ids if claim_id in routed
        )
        data["unrouted_target_claim_ids"] = tuple(
            claim_id for claim_id in target_ids if claim_id not in routed
        )
        if data.get("selected_planning_attempt") is not None:
            deferred = list(data.get("deferred_targets") or ())
            deferred_ids = {
                str(
                    item.get("claim_id")
                    if isinstance(item, Mapping)
                    else getattr(item, "claim_id", "")
                )
                for item in deferred
            }
            queried_ids = {
                str(claim_id)
                for record in data.get("searches") or ()
                for claim_id in (
                    (
                        record.get("query", {}).get("claim_ids", ())
                        if isinstance(record, Mapping)
                        else getattr(record.query, "claim_ids", ())
                    )
                    or ()
                )
            }
            for claim_id in data["unrouted_target_claim_ids"]:
                if claim_id in deferred_ids:
                    continue
                failed_route = claim_id in queried_ids
                deferred.append(
                    {
                        "claim_id": claim_id,
                        "reason": (
                            "search_route_failed"
                            if failed_route
                            else "query_capacity_not_allocated"
                        ),
                        "priority_rationale": (
                            "code derived: every issued search route for this "
                            "target ended with a provider error"
                            if failed_route
                            else "code derived: target received no accepted "
                            "cached candidate or issued query route"
                        ),
                        "allocation_source": "code_derived",
                    }
                )
            data["deferred_targets"] = tuple(deferred)
        return data

    @model_validator(mode="after")
    def _target_routes_partition_requested_scope(self) -> EvidenceGapResult:
        target_ids = self.target_claim_ids
        routed = self.routed_target_claim_ids
        unrouted = self.unrouted_target_claim_ids
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("target_claim_ids must be unique")
        if len(set(routed)) != len(routed):
            raise ValueError("routed_target_claim_ids must be unique")
        if len(set(unrouted)) != len(unrouted):
            raise ValueError("unrouted_target_claim_ids must be unique")
        if set(routed) & set(unrouted):
            raise ValueError("routed and unrouted target claims must be disjoint")
        if set(routed) | set(unrouted) != set(target_ids):
            raise ValueError(
                "routed and unrouted target claims must partition targets"
            )
        deferred_ids = tuple(target.claim_id for target in self.deferred_targets)
        if len(set(deferred_ids)) != len(deferred_ids):
            raise ValueError("deferred target claim_ids must be unique")
        if not set(deferred_ids).issubset(set(unrouted)):
            raise ValueError("deferred targets must be unrouted target claims")
        if self.planning_attempt_count:
            if self.selected_planning_attempt is None:
                if self.stop_reason not in {
                    EvidenceGapStopReason.BUDGET_EXHAUSTED,
                    EvidenceGapStopReason.MODEL_ERROR,
                }:
                    raise ValueError(
                        "a completed planning call must select one attempt"
                    )
            elif self.selected_planning_attempt > self.planning_attempt_count:
                raise ValueError(
                    "selected planning attempt exceeds attempted plans"
                )
        if self.selected_planning_attempt is not None:
            if set(deferred_ids) != set(unrouted):
                raise ValueError(
                    "code-derived unrouted records must equal unrouted targets"
                )
        if self.verification_reserve_history:
            if self.verification_reserve != self.verification_reserve_history[-1]:
                raise ValueError(
                    "current verification reserve must equal the final "
                    "recorded reserve snapshot"
                )
        return self

    @property
    def total_tokens(self) -> int:
        return sum(call.token_count for call in self.usage)

    @property
    def total_cost_usd(self) -> float:
        return sum(call.cost_usd for call in self.usage)


class EvidenceGapModelClient(Protocol):
    """Injected cheap model used for the one-shot gap plan and read choice."""

    def generate(self, prompt: str) -> Any | Awaitable[Any]:
        """Return JSON in the measured usage envelope."""


class _Envelope(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    content: Any
    token_count: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)


class _RawCachedHint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    note_id: str
    source_id: str
    independent_from_existing_publishers: bool
    publisher_identity: str
    independence_rationale: str


class _RawSearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_ids: tuple[str, ...] = Field(min_length=1)
    item_id: str
    query: str

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


class _RawRead(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str
    item_id: str
    claim_ids: tuple[str, ...]
    independent_from_existing_publishers: bool
    publisher_identity: str
    independence_rationale: str


class _GapNoteDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str
    finding: str
    quote: str

    @field_validator("item_id", "finding", "quote")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("note fields must not be blank")
        return normalized


_PLAN_PROMPT = """\
Run exactly one evidence-gap planning pass after an initial verification.
Return json only. A candidate is only a source worth checking later, never a
support verdict. Claims and report wording are frozen.

For every target claim, review the complete compact registry of cached notes
before proposing web search. Reusing a cached note is allowed only when its
source is relevant and, for a claim that already has supporting publishers,
it comes from a genuinely different publishing organization. Different
domains owned by the same organization, aggregators, and republished or
syndicated copies are not independent publishers. Do not use domain spelling
alone to manufacture a second publisher.

Each cached note records only the sparse checked_for_target_claim_ids list.
For every other target claim the note is unused; this is the exact compact
inverse of repeating every unused claim ID on every note.

You have a hard budget of at most {max_queries} web search queries. Select and
order the highest-value focused queries, putting the highest priority first.
One query may name several claim_ids only when the same focused search can
plausibly find pages relevant to every named claim. Do not make a query vague
merely to cover more claims.

Treat the current notes and checked sources as leads, not as a verdict that an
answer is complete. For each proposed route, decide whether following an
underlying record, participant, or independently reported account would better
answer the claim in context. Missing material in this pass is never evidence
that the reported fact is absent. You alone judge relevance, source role,
independence, timeliness, and whether another focused route is worth its cost;
code does not rank source types or set a quality threshold.

Return only semantic routes: accepted cached candidates and proposed queries.
Do not return deferred_targets. Code records every unrouted target as
query_capacity_not_allocated; that is a mechanical record of the resulting
route coverage, not a model judgement that evidence is unnecessary or
unavailable. The query limit is an upper resource bound, not a coverage target.

Return:
{{"cached_candidates":[{{"claim_id":"claim-0001",\
"note_id":"note-000001","source_id":"source-id",\
"independent_from_existing_publishers":true,\
"publisher_identity":"publishing organization",\
"independence_rationale":"brief reason"}}],\
"queries":[{{"claim_ids":["claim-0001","claim-0002"],\
"item_id":"existing-checklist-item","query":"one focused query"}}]}}

Search may seek support, contradiction, absence of support, or insufficient
information; do not optimize only for agreement.

Checklist item IDs:
{item_ids}

Checklist corroboration targets for gap-planning priority only:
{corroboration_targets}

Target claims and initial source-level outcomes:
{targets}

Compact cached note registry:
{notes}
"""


_READ_PROMPT = """\
Choose full pages to read for one bounded evidence-gap pass. Return json only.
The search snippets are unverified routing material, never evidence. Claims
and report wording are frozen.

Return:
{{"reads":[{{"url":"an exact candidate URL","item_id":"an allowed item",\
"claim_ids":["claim-0001"],\
"independent_from_existing_publishers":true,\
"publisher_identity":"publishing organization",\
"independence_rationale":"brief reason"}}]}}

Choose only URLs shown below and no more than {max_reads}. A URL must be
plausibly relevant to every listed claim. For claims that already have a
supporting publisher, reject another domain of the same organization and
reject aggregators or republished copies. Set
independent_from_existing_publishers=false when independence is doubtful;
code will retain the rejection in the audit. Zero reads is legal.

Targets:
{targets}

Search candidates:
{candidates}
"""


_NOTE_PROMPT = """\
Extract zero, one, or two research notes from this one complete source. Return json
only:
{{"notes":[{{"item_id":"existing-checklist-item",\
"finding":"what the source says",\
"quote":"one exact continuous source passage"}}]}}

Prioritize the frozen target claims, but retain useful findings for other
listed checklist items too. One quote is one continuous verbatim passage.
Copy it exactly: do not paraphrase, join separated passages, use ellipses,
reorder words, or change punctuation. If two separate passages are needed,
return two notes. Never return more than two notes. Returning zero notes is
legal and is not an error.

Allowed checklist item IDs:
{item_ids}

Target claims:
{claims}

Source URL:
{url}

BEGIN COMPLETE SOURCE
{source_text}
END COMPLETE SOURCE
"""


class _GapBudgetExhausted(RuntimeError):
    pass


class _BudgetTracker:
    def __init__(
        self,
        budget: EvidenceGapBudget,
        *,
        estimate_input_tokens: Callable[[Any, str], int] | None,
        estimate_cost_usd: Callable[[Any, str], float] | None,
    ) -> None:
        self.budget = budget
        self.estimate_input_tokens = estimate_input_tokens
        self.estimate_cost_usd = estimate_cost_usd
        self.usage: list[EvidenceGapCallUsage] = []
        self.verification_reserved_tokens = 0
        self.verification_reserved_cost_usd = 0.0

    @property
    def tokens_used(self) -> int:
        return sum(call.token_count for call in self.usage)

    @property
    def cost_used(self) -> float:
        return sum(call.cost_usd for call in self.usage)

    def _estimate(self, client: Any, prompt: str) -> tuple[int, float]:
        token_estimator = self.estimate_input_tokens
        cost_estimator = self.estimate_cost_usd
        try:
            if token_estimator is not None:
                tokens = int(token_estimator(client, prompt))
            else:
                method = getattr(client, "estimate_tokens", None)
                if not callable(method):
                    raise RuntimeError("estimator method is absent")
                tokens = int(method(prompt))
        except (RuntimeError, TypeError, ValueError) as exc:
            raise _GapBudgetExhausted(
                "gap token admission estimator is unavailable: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if tokens < 0:
            raise _GapBudgetExhausted(
                "gap token admission estimator returned a negative value"
            )
        try:
            if cost_estimator is not None:
                cost = float(cost_estimator(client, prompt))
            else:
                method = getattr(client, "estimate_cost_usd", None)
                if not callable(method):
                    raise RuntimeError("estimator method is absent")
                cost = float(method(prompt))
        except (RuntimeError, TypeError, ValueError) as exc:
            raise _GapBudgetExhausted(
                "gap cost admission estimator is unavailable: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not math.isfinite(cost) or cost < 0.0:
            raise _GapBudgetExhausted(
                "gap cost admission estimator returned an invalid value"
            )
        return tokens, cost

    def reserve_verification(
        self,
        *,
        tokens: int,
        cost_usd: float,
        prerequisite_tokens: int = 0,
        prerequisite_cost_usd: float = 0.0,
    ) -> tuple[int, float]:
        """Protect verification without starving its required precursor."""

        remaining_tokens = max(0, self.budget.max_tokens - self.tokens_used)
        remaining_cost = max(0.0, self.budget.max_cost_usd - self.cost_used)
        verification_token_capacity = max(
            0,
            remaining_tokens - max(0, int(prerequisite_tokens)),
        )
        verification_cost_capacity = max(
            0.0,
            remaining_cost - max(0.0, float(prerequisite_cost_usd)),
        )
        self.verification_reserved_tokens = min(
            max(0, int(tokens)),
            verification_token_capacity,
        )
        self.verification_reserved_cost_usd = min(
            max(0.0, float(cost_usd)),
            verification_cost_capacity,
        )
        return (
            self.verification_reserved_tokens,
            self.verification_reserved_cost_usd,
        )

    async def call(
        self,
        client: Any,
        prompt: str,
        *,
        stage: str,
        allow_verification_reserve: bool = False,
    ) -> Any:
        estimated_tokens, estimated_cost = self._estimate(client, prompt)
        token_limit = self.budget.max_tokens
        cost_limit = self.budget.max_cost_usd
        if not allow_verification_reserve:
            token_limit -= self.verification_reserved_tokens
            cost_limit -= self.verification_reserved_cost_usd
        if self.tokens_used + estimated_tokens > token_limit:
            detail = (
                " while preserving the verification reserve"
                if self.verification_reserved_tokens
                and not allow_verification_reserve
                else ""
            )
            raise _GapBudgetExhausted(
                "estimated input exceeds remaining gap token budget" + detail
            )
        if self.cost_used + estimated_cost > cost_limit:
            detail = (
                " while preserving the verification reserve"
                if self.verification_reserved_cost_usd
                and not allow_verification_reserve
                else ""
            )
            raise _GapBudgetExhausted(
                "estimated call exceeds remaining gap cost budget" + detail
            )
        response = client.generate(prompt)
        if inspect.isawaitable(response):
            response = await response
        tokens, cost = _best_effort_usage(response)
        self.usage.append(
            EvidenceGapCallUsage(
                call_number=len(self.usage) + 1,
                stage=stage,
                prompt_chars=len(prompt),
                estimated_input_tokens=estimated_tokens,
                estimated_cost_usd=estimated_cost,
                token_count=tokens,
                cost_usd=cost,
            )
        )
        return response


class _TrackedClient:
    def __init__(
        self,
        client: Any,
        tracker: _BudgetTracker,
        stage: str,
        *,
        allow_verification_reserve: bool = False,
    ) -> None:
        self.client = client
        self.tracker = tracker
        self.stage = stage
        self.allow_verification_reserve = allow_verification_reserve

    async def generate(self, prompt: str) -> Any:
        return await self.tracker.call(
            self.client,
            prompt,
            stage=self.stage,
            allow_verification_reserve=self.allow_verification_reserve,
        )


def _best_effort_usage(response: Any) -> tuple[int, float]:
    if not isinstance(response, Mapping):
        return 0, 0.0
    try:
        return (
            max(0, int(response.get("token_count", 0))),
            max(0.0, float(response.get("cost_usd", 0.0))),
        )
    except (TypeError, ValueError):
        return 0, 0.0


def _decode_response(response: Any) -> Any:
    envelope = _Envelope.model_validate(response)
    content = envelope.content
    return loads_lenient(content) if isinstance(content, str) else content


def _publisher_proxy(url: str, fallback: str = "") -> str:
    host = (urlparse(url).hostname or "").strip(".").casefold()
    if host.startswith("www."):
        host = host[4:]
    return host or fallback.strip().casefold()


def _supporting_publisher_proxy_sets(
    target: ClaimVerification,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return whole-claim, element-level, and union publisher proxies."""

    whole_claim = tuple(sorted(set(target.publisher_domain_proxies)))
    element_level = tuple(
        sorted(set(target.element_supporting_domain_proxies))
    )
    used = tuple(sorted(set(whole_claim) | set(element_level)))
    return whole_claim, element_level, used


def _source_was_already_checked(
    target: ClaimVerification,
    *,
    source_id: str,
    url: str,
) -> bool:
    """Return whether a route would repeat an existing claim/source check."""

    return any(
        relation.status in _TERMINAL_RELATION_STATUSES
        and (relation.source_id == source_id or relation.url == url)
        for relation in target.relations
    )


def _cached_route_mechanical_error(
    target: ClaimVerification,
    note: ResearchNote,
) -> str | None:
    """Reject only code-provable cached no-ops before semantic planning."""

    if note.note_id is None:
        return "cached note has no stable note_id"
    if _source_was_already_checked(
        target,
        source_id=note.source_id,
        url=note.url,
    ):
        return "source was already checked for this claim"
    used_publishers = set(_supporting_publisher_proxy_sets(target)[2])
    if _publisher_proxy(note.url, note.publisher) in used_publishers:
        return "publisher domain proxy already supports this claim"
    return None


def _has_mechanically_executable_cached_route(
    *,
    targets: Sequence[ClaimVerification],
    notes: Sequence[ResearchNote],
    source_cache: Mapping[str, str],
) -> bool:
    """Return whether at least one target/note pair survives code gates."""

    return any(
        note.url in source_cache
        and _cached_route_mechanical_error(target, note) is None
        for target in targets
        for note in notes
    )


def _identity_key(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", value).casefold()
        if character.isalnum()
    )


def _identity_matches_proxy(identity: str, proxy: str) -> bool:
    """Catch explicit same-brand labels without claiming ownership resolution."""

    key = _identity_key(identity)
    return bool(key) and key in {
        _identity_key(label) for label in proxy.split(".") if label
    }


def _target_payload(
    targets: Sequence[ClaimVerification],
    *,
    truth_condition_registry: TruthConditionRegistry | None = None,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for target in targets:
        whole_claim, element_level, used = _supporting_publisher_proxy_sets(
            target
        )
        aggregate = target.truth_condition_aggregate
        registry_entry = (
            truth_condition_registry.entry_for(target.claim.claim_id)
            if truth_condition_registry is not None
            else None
        )
        element_text_by_id = {
            element.element_id: element.text
            for element in (
                registry_entry.elements if registry_entry is not None else ()
            )
        }
        for relation in target.relations:
            for element_relation in relation.element_relations:
                element_text_by_id.setdefault(
                    element_relation.element_id,
                    element_relation.element_text,
                )
        payload.append(
            {
                "claim_id": target.claim.claim_id,
                "claim_text": target.claim.claim_text,
                "state": target.state.value,
                # Preserve the model's semantic research intent at O(elements)
                # size. Element IDs are code-owned and opaque, so a state-only
                # ID list is not enough for a planner to focus a follow-up
                # query. Per-source element cross-products remain omitted;
                # ``checked_sources`` below retains the compact source view.
                "truth_condition_summary": (
                    {
                        "elementization_execution_status": (
                            aggregate.elementization_execution_status.value
                        ),
                        "elementization_semantic_status": (
                            aggregate.elementization_semantic_status.value
                            if aggregate.elementization_semantic_status is not None
                            else None
                        ),
                        "coverage_state": aggregate.coverage_state.value,
                        "execution_completeness": (
                            aggregate.execution_completeness.value
                        ),
                        "elements": [
                            {
                                "element_id": element.element_id,
                                "truth_condition": element_text_by_id.get(
                                    element.element_id
                                ),
                                "semantic_state": element.semantic_state.value,
                                "execution_completeness": (
                                    element.execution_completeness.value
                                ),
                            }
                            for element in aggregate.elements
                        ],
                    }
                    if aggregate is not None
                    else None
                ),
                "corroboration_target": target.corroboration_target,
                "whole_claim_supporting_publisher_domain_proxies": list(
                    whole_claim
                ),
                "element_supporting_publisher_domain_proxies": list(
                    element_level
                ),
                "used_supporting_publisher_domain_proxies": list(used),
                "checked_sources": [
                    {
                        "url": relation.url,
                        "publisher_domain_proxy": (
                            relation.publisher_domain_proxy
                        ),
                        "verdict": (
                            relation.semantic_verdict.value
                            if relation.semantic_verdict is not None
                            else None
                        ),
                    }
                    for relation in target.relations
                ],
            }
        )
    return payload


def build_evidence_gap_plan_prompt(
    *,
    targets: Sequence[ClaimVerification],
    notes: Sequence[ResearchNote],
    checklist: ResearchChecklist,
    max_queries: int,
    truth_condition_registry: TruthConditionRegistry | None = None,
) -> str:
    """Expose cache and a hard query budget before allowing web search."""

    used_pairs = {
        (target.claim.claim_id, note_id)
        for target in targets
        for relation in target.relations
        for note_id in relation.candidate_note_ids
    }
    note_registry = [
        {
            "note_id": note.note_id,
            "source_id": note.source_id,
            "item_id": note.item_id,
            "finding": note.finding,
            "publisher": note.publisher,
            "location_status": note.location_status.value,
            # Sparse checked IDs are the exact inverse of the old repeated
            # ``unused_by`` arrays. Absence still means unused for a target,
            # while growth becomes O(notes + actual relations), not
            # O(notes * targets).
            "checked_for_target_claim_ids": [
                target.claim.claim_id
                for target in targets
                if (target.claim.claim_id, note.note_id) in used_pairs
            ],
        }
        for note in notes
    ]
    compact_json = {
        "ensure_ascii": False,
        "sort_keys": True,
        "separators": (",", ":"),
    }
    return _PLAN_PROMPT.format(
        max_queries=max_queries,
        item_ids=json.dumps(
            list(checklist.in_scope_item_ids),
            **compact_json,
        ),
        corroboration_targets=json.dumps(
            [
                {
                    "item_id": item.item_id,
                    "corroboration_target": item.corroboration_target,
                }
                for item in checklist.in_scope_items
            ],
            **compact_json,
        ),
        targets=json.dumps(
            _target_payload(
                targets,
                truth_condition_registry=truth_condition_registry,
            ),
            **compact_json,
        ),
        notes=json.dumps(note_registry, **compact_json),
    )


def build_evidence_gap_read_prompt(
    *,
    targets: Sequence[ClaimVerification],
    searches: Sequence[GapSearchRecord],
    max_reads: int,
    truth_condition_registry: TruthConditionRegistry | None = None,
) -> str:
    """Ask the model to choose URLs after seeing bounded search results."""

    candidates = [
        {
            "query_claim_ids": list(record.query.claim_ids),
            "target_element_ids": list(record.query.target_element_ids),
            "allowed_item_id": record.query.item_id,
            "title": result.title,
            "url": result.url,
            "snippet": result.snippet,
        }
        for record in searches
        for result in record.results
    ]
    return _READ_PROMPT.format(
        max_reads=max_reads,
        targets=json.dumps(
            _target_payload(
                targets,
                truth_condition_registry=truth_condition_registry,
            ),
            ensure_ascii=False,
            sort_keys=True,
        ),
        candidates=json.dumps(candidates, ensure_ascii=False, sort_keys=True),
    )


def build_evidence_gap_note_prompt(
    *,
    url: str,
    source_text: str,
    claims: Sequence[AtomicClaim],
    checklist: ResearchChecklist,
    target_element_ids: Sequence[str] = (),
    truth_condition_registry: TruthConditionRegistry | None = None,
) -> str:
    """Build one full-source note pass without changing checklist state."""

    selected_element_ids = set(target_element_ids)
    elements_by_claim = {
        entry.claim_id: [
            {
                "element_id": element.element_id,
                "text": element.text,
            }
            for element in entry.elements
            if element.element_id in selected_element_ids
        ]
        for entry in (
            truth_condition_registry.entries
            if truth_condition_registry is not None
            else ()
        )
    }
    return _NOTE_PROMPT.format(
        item_ids=json.dumps(
            list(checklist.in_scope_item_ids),
            ensure_ascii=False,
        ),
        claims=json.dumps(
            [
                {
                    "claim_id": claim.claim_id,
                    "claim_text": claim.claim_text,
                    "target_truth_conditions": elements_by_claim.get(
                        claim.claim_id, []
                    ),
                }
                for claim in claims
            ],
            ensure_ascii=False,
            sort_keys=True,
        ),
        url=url,
        source_text=source_text,
    )


def _capacity_probe_source(char_count: int) -> str:
    """Return deterministic, token-like source text of the requested size."""

    prefix = "capacity-probe-unique-passage\n"
    seed = "capacity evidence passage "
    if char_count <= len(prefix):
        return prefix[:char_count]
    repeats = math.ceil((char_count - len(prefix)) / len(seed))
    return (prefix + seed * repeats)[:char_count]


def _bounded_web_action_allowances(
    *,
    targets: Sequence[ClaimVerification],
    routed_claim_ids: Sequence[str] | None,
    checklist: ResearchChecklist,
    tracker: _BudgetTracker,
    budget: EvidenceGapBudget,
    note_model: EvidenceGapModelClient,
    attribution_model: AttributionModelClient,
    verification_model: VerificationModelClient,
    attribution_settings: AttributionSettings,
    truth_condition_registry: TruthConditionRegistry | None,
    target_element_ids_by_claim: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[int, float, int, float, int, float]]:
    """Estimate one bounded two-note web tail for each routed claim."""

    active_items = checklist.in_scope_items
    if not active_items:
        return {}
    selected_ids = (
        set(routed_claim_ids) if routed_claim_ids is not None else None
    )
    probe_url = "https://capacity.invalid/evidence"
    probe_source = _capacity_probe_source(
        budget.downstream_action_source_chars
    )
    allowances: dict[str, tuple[int, float, int, float, int, float]] = {}
    for target in targets:
        claim = target.claim
        if selected_ids is not None and claim.claim_id not in selected_ids:
            continue
        probe_registry = (
            select_truth_condition_registry(
                truth_condition_registry,
                (claim.claim_id,),
            )
            if truth_condition_registry is not None
            else None
        )
        verification_prompt = build_verification_prompt(
            url=probe_url,
            source_text=probe_source,
            claims=(claim,),
            registry=probe_registry,
        )
        verification_tokens, verification_cost = tracker._estimate(
            verification_model,
            verification_prompt,
        )
        note_prompt = build_evidence_gap_note_prompt(
            url=probe_url,
            source_text=probe_source,
            claims=(claim,),
            checklist=checklist,
            target_element_ids=target_element_ids_by_claim.get(
                claim.claim_id,
                (),
            ),
            truth_condition_registry=truth_condition_registry,
        )
        note_tokens, note_cost = tracker._estimate(note_model, note_prompt)
        note_tokens += math.ceil(
            note_tokens * budget.planning_output_headroom_ratio
        )
        note_cost *= 1.0 + budget.planning_output_headroom_ratio
        probe_notes = tuple(
            create_note(
                item_id=active_items[0].item_id,
                finding=f"Capacity evidence passage {ordinal}.",
                quote=probe_source.splitlines()[0],
                url=probe_url,
                source_text=probe_source,
            ).model_copy(
                update={"note_id": f"note-capacity-probe-{ordinal}"}
            )
            for ordinal in (1, 2)
        )
        reattribution_prompt = build_attribution_prompt(
            (claim,),
            probe_notes,
            page_size=attribution_settings.note_page_size,
        )
        reattribution_tokens, reattribution_cost = tracker._estimate(
            attribution_model,
            reattribution_prompt,
        )
        reattribution_tokens += math.ceil(
            reattribution_tokens * budget.planning_output_headroom_ratio
        )
        reattribution_cost *= 1.0 + budget.planning_output_headroom_ratio
        allowances[claim.claim_id] = (
            note_tokens + verification_tokens,
            note_cost + verification_cost,
            reattribution_tokens,
            reattribution_cost,
            note_tokens + reattribution_tokens + verification_tokens,
            note_cost + reattribution_cost + verification_cost,
        )
    return allowances


def _planning_capacity_audit(
    *,
    plan_prompt: str,
    tracker: _BudgetTracker,
    budget: EvidenceGapBudget,
    advertised_max_search_queries: int,
    targets: Sequence[ClaimVerification],
    notes: Sequence[ResearchNote],
    source_cache: Mapping[str, str],
    checklist: ResearchChecklist,
    gap_model: EvidenceGapModelClient,
    note_model: EvidenceGapModelClient,
    attribution_model: AttributionModelClient,
    verification_model: VerificationModelClient,
    attribution_settings: AttributionSettings,
    truth_condition_registry: TruthConditionRegistry | None,
    target_element_ids_by_claim: Mapping[str, tuple[str, ...]],
) -> EvidenceGapPlanningCapacityAudit:
    """Protect one executable candidate route before admitting planning.

    Cached verification and a new web acquisition are alternative actions.
    This check removes only mechanically provable no-ops; the model still owns
    relevance. If web does not fit, the caller may rebuild a cache-only prompt
    before paying for planning. Every selected route is then preflighted
    against exact downstream prompts.
    """

    planning_tokens, planning_cost = tracker._estimate(gap_model, plan_prompt)
    headroom_tokens = math.ceil(
        planning_tokens * budget.planning_output_headroom_ratio
    )
    headroom_cost = planning_cost * budget.planning_output_headroom_ratio
    available_after_plan_tokens = max(0, budget.max_tokens - planning_tokens)
    available_after_plan_cost = max(0.0, budget.max_cost_usd - planning_cost)

    # kind, read tokens/cost, note+verification tokens/cost,
    # reattribution tokens/cost, complete downstream tokens/cost
    route_estimates: list[
        tuple[str, int, float, int, float, int, float, int, float]
    ] = []
    active_items = checklist.in_scope_items
    can_probe_search_route = bool(
        targets
        and active_items
        and advertised_max_search_queries
    )
    can_probe_web_route = bool(can_probe_search_route and budget.max_reads)
    probe_url = "https://capacity.invalid/evidence"

    # A cache route uses exact text already owned by the ledger. Enumerate only
    # target/note pairs that survive code-provable no-op gates. Sorting later
    # selects the least resource pressure; this is capacity accounting, not a
    # relevance judgement. A model-selected route is checked exactly again.
    cached_routes = tuple(
        (
            target,
            note,
            source_cache[note.url],
        )
        for target in targets
        for note in notes
        if note.url in source_cache
        and _cached_route_mechanical_error(target, note) is None
    )
    for target, note, cached_text in sorted(
        cached_routes,
        key=lambda route: (
            len(route[2]),
            route[1].url,
            route[0].claim.claim_id,
        ),
    ):
        claim = target.claim
        probe_registry = (
            select_truth_condition_registry(
                truth_condition_registry,
                (claim.claim_id,),
            )
            if truth_condition_registry is not None
            else None
        )
        verification_prompt = build_verification_prompt(
            url=note.url,
            source_text=cached_text,
            claims=(claim,),
            registry=probe_registry,
        )
        verification_tokens, verification_cost = tracker._estimate(
            verification_model,
            verification_prompt,
        )
        route_estimates.append(
            (
                "cache",
                0,
                0.0,
                verification_tokens,
                verification_cost,
                0,
                0.0,
                verification_tokens,
                verification_cost,
            )
        )

    if can_probe_web_route:
        probe_query = GapSearchQuery(
            claim_ids=(targets[0].claim.claim_id,),
            item_id=active_items[0].item_id,
            query="capacity probe",
        )
        read_prompt = build_evidence_gap_read_prompt(
            targets=targets,
            searches=(
                GapSearchRecord(
                    query=probe_query,
                    results=(
                        SearchResult(
                            title="Candidate evidence route",
                            url=probe_url,
                            snippet="Candidate evidence passage.",
                        ),
                    ),
                ),
            ),
            max_reads=1,
            truth_condition_registry=truth_condition_registry,
        )
        read_tokens, read_cost = tracker._estimate(gap_model, read_prompt)
        read_tokens += math.ceil(
            read_tokens * budget.planning_output_headroom_ratio
        )
        read_cost *= 1.0 + budget.planning_output_headroom_ratio
        web_allowances = _bounded_web_action_allowances(
            targets=targets,
            routed_claim_ids=None,
            checklist=checklist,
            tracker=tracker,
            budget=budget,
            note_model=note_model,
            attribution_model=attribution_model,
            verification_model=verification_model,
            attribution_settings=attribution_settings,
            truth_condition_registry=truth_condition_registry,
            target_element_ids_by_claim=target_element_ids_by_claim,
        )
        for (
            note_and_verification_tokens,
            note_and_verification_cost,
            route_reattribution_tokens,
            route_reattribution_cost,
            downstream_tokens,
            downstream_cost,
        ) in web_allowances.values():
            route_estimates.append(
                (
                    "web",
                    read_tokens,
                    read_cost,
                    note_and_verification_tokens,
                    note_and_verification_cost,
                    route_reattribution_tokens,
                    route_reattribution_cost,
                    downstream_tokens,
                    downstream_cost,
                )
            )
    elif can_probe_search_route:
        # A query-only disagreement pass cannot create formal evidence, but it
        # remains an explicit investigation attempt used by the bounded
        # disagreement audit. Search is provider-bounded and model-free after
        # planning, so output headroom is the only downstream capacity needed.
        route_estimates.append(
            ("search", 0, 0.0, 0, 0.0, 0, 0.0, 0, 0.0)
        )

    def route_fits(
        route: tuple[str, int, float, int, float, int, float, int, float],
    ) -> bool:
        return (
            headroom_tokens + route[1] + route[7]
            <= available_after_plan_tokens
            and headroom_cost + route[2] + route[8]
            <= available_after_plan_cost + 1e-12
        )

    feasible_cache = [
        route
        for route in route_estimates
        if route[0] == "cache" and route_fits(route)
    ]
    feasible_other = [
        route
        for route in route_estimates
        if route[0] != "cache" and route_fits(route)
    ]
    web_routes = [route for route in route_estimates if route[0] == "web"]
    search_routes = [route for route in route_estimates if route[0] == "search"]
    cache_routes = [route for route in route_estimates if route[0] == "cache"]
    # If the prompt advertises web queries, its capacity proof must be a full
    # web route. Cache can be used as the cheaper alternative only by a
    # cache-only prompt (the caller may rebuild one before spending planning).
    if can_probe_web_route:
        # Web is an advertised semantic option, so a cache route cannot serve
        # as its resource proof. Preserve one complete web route before the
        # planner may choose between web and cache.
        route_pool = feasible_other or web_routes
    elif can_probe_search_route:
        # Disagreement can deliberately run as a query-only investigation
        # when reads are disabled. Preserve that existing bounded behavior.
        route_pool = [
            route for route in feasible_other if route[0] == "search"
        ] or search_routes
    else:
        route_pool = feasible_cache or cache_routes

    def route_pressure(
        route: tuple[str, int, float, int, float, int, float, int, float],
    ) -> tuple[float, float, int, float]:
        needed_tokens = headroom_tokens + route[1] + route[7]
        needed_cost = headroom_cost + route[2] + route[8]
        token_ratio = (
            needed_tokens / available_after_plan_tokens
            if available_after_plan_tokens
            else (0.0 if needed_tokens == 0 else math.inf)
        )
        cost_ratio = (
            needed_cost / available_after_plan_cost
            if available_after_plan_cost
            else (0.0 if needed_cost <= 1e-12 else math.inf)
        )
        return (
            max(token_ratio, cost_ratio),
            token_ratio + cost_ratio,
            needed_tokens,
            needed_cost,
        )

    chosen = min(route_pool, key=route_pressure) if route_pool else None
    # Once the planner actually selects web search, cache is no longer a
    # substitute for that route's note/attribution/verification tail. Keep a
    # separate conservative one-claim web allowance for the post-plan seam.
    minimum_web_route = (
        min(web_routes, key=route_pressure) if web_routes else None
    )
    web_downstream_action_tokens = (
        minimum_web_route[7] if minimum_web_route is not None else 0
    )
    web_downstream_action_cost = (
        minimum_web_route[8] if minimum_web_route is not None else 0.0
    )
    read_tokens = chosen[1] if chosen is not None else 0
    read_cost = chosen[2] if chosen is not None else 0.0
    action_tokens = chosen[3] if chosen is not None else 0
    action_cost = chosen[4] if chosen is not None else 0.0
    reattribution_tokens = chosen[5] if chosen is not None else 0
    reattribution_cost = chosen[6] if chosen is not None else 0.0
    downstream_action_tokens = chosen[7] if chosen is not None else 0
    downstream_action_cost = chosen[8] if chosen is not None else 0.0
    desired_tokens = headroom_tokens + read_tokens + downstream_action_tokens
    desired_cost = headroom_cost + read_cost + downstream_action_cost
    requested_tokens = min(desired_tokens, available_after_plan_tokens)
    requested_cost = min(desired_cost, available_after_plan_cost)
    reserved_tokens, reserved_cost = tracker.reserve_verification(
        tokens=requested_tokens,
        cost_usd=requested_cost,
    )
    return EvidenceGapPlanningCapacityAudit(
        advertised_max_search_queries=advertised_max_search_queries,
        target_count=len(targets),
        cached_note_count=len(notes),
        prompt_chars=len(plan_prompt),
        estimated_planning_input_tokens=planning_tokens,
        estimated_planning_cost_usd=planning_cost,
        max_planning_input_tokens=min(
            budget.max_planning_input_tokens,
            budget.max_tokens,
        ),
        max_planning_prompt_chars=budget.max_planning_prompt_chars,
        planning_output_headroom_tokens=headroom_tokens,
        planning_output_headroom_cost_usd=headroom_cost,
        downstream_action_source_chars=budget.downstream_action_source_chars,
        read_selection_estimated_tokens=read_tokens,
        read_selection_estimated_cost_usd=read_cost,
        reattribution_estimated_tokens=reattribution_tokens,
        reattribution_estimated_cost_usd=reattribution_cost,
        note_and_verification_estimated_tokens=action_tokens,
        note_and_verification_estimated_cost_usd=action_cost,
        downstream_action_estimated_tokens=downstream_action_tokens,
        downstream_action_estimated_cost_usd=downstream_action_cost,
        web_downstream_action_estimated_tokens=(
            web_downstream_action_tokens
        ),
        web_downstream_action_estimated_cost_usd=(
            web_downstream_action_cost
        ),
        reserved_tokens=reserved_tokens,
        reserved_cost_usd=reserved_cost,
        reserve_fully_funded=(
            chosen is not None
            and reserved_tokens >= desired_tokens
            and reserved_cost + 1e-12 >= desired_cost
        ),
    )


def _estimate_incremental_reattribution_allowance(
    *,
    claims: Sequence[AtomicClaim],
    notes: Sequence[ResearchNote],
    tracker: _BudgetTracker,
    attribution_model: AttributionModelClient,
    attribution_settings: AttributionSettings,
    output_headroom_ratio: float,
) -> tuple[int, float]:
    """Estimate one incremental attribution turn plus provider headroom."""

    if not claims or not notes:
        return 0, 0.0
    prompt = build_attribution_prompt(
        claims,
        notes,
        page_size=attribution_settings.note_page_size,
    )
    estimated_tokens, estimated_cost = tracker._estimate(
        attribution_model,
        prompt,
    )
    return (
        estimated_tokens
        + math.ceil(estimated_tokens * output_headroom_ratio),
        estimated_cost * (1.0 + output_headroom_ratio),
    )


def _prospective_gap_note(
    *,
    item_id: str,
    url: str,
    ordinal: int,
) -> ResearchNote:
    """Create a bounded placeholder for pre-note attribution capacity only."""

    source_text = _capacity_probe_source(256)
    note = create_note(
        item_id=item_id,
        finding="Potential evidence passage selected for this routed claim.",
        quote=source_text.splitlines()[0],
        url=url,
        source_text=source_text,
    )
    return note.model_copy(update={"note_id": f"note-gap-probe-{ordinal:06d}"})


def _estimate_verification_group(
    *,
    claims: Sequence[AtomicClaim],
    url: str,
    source_text: str,
    batch_size: int,
    tracker: _BudgetTracker,
    verification_model: VerificationModelClient,
    truth_condition_registry: TruthConditionRegistry | None = None,
) -> tuple[int, int, float]:
    batch_count = 0
    tokens = 0
    cost = 0.0
    for start in range(0, len(claims), batch_size):
        prompt = build_verification_prompt(
            url=url,
            source_text=source_text,
            claims=claims[start : start + batch_size],
            registry=truth_condition_registry,
        )
        estimated_tokens, estimated_cost = tracker._estimate(
            verification_model,
            prompt,
        )
        batch_count += 1
        tokens += estimated_tokens
        cost += estimated_cost
    return batch_count, tokens, cost


def _reserve_verification_budget(
    *,
    tracker: _BudgetTracker,
    queries: Sequence[GapSearchQuery],
    hints: Sequence[CachedCandidateHint],
    targets: Sequence[ClaimVerification],
    notes: Sequence[ResearchNote],
    source_cache: Mapping[str, str],
    max_reads: int,
    verification_model: VerificationModelClient,
    verification_settings: VerificationSettings,
    truth_condition_registry: TruthConditionRegistry | None = None,
    prerequisite_stage: str | None = None,
    prerequisite_estimated_tokens: int = 0,
    prerequisite_estimated_cost_usd: float = 0.0,
    minimum_action_estimated_tokens: int = 0,
    minimum_action_estimated_cost_usd: float = 0.0,
    incremental_reattribution_estimated_tokens: int = 0,
    incremental_reattribution_estimated_cost_usd: float = 0.0,
    admitted_read_source_claims: (
        Mapping[str, Sequence[AtomicClaim]] | None
    ) = None,
) -> VerificationReserveAudit:
    """Reserve only verification work whose exact source text is known.

    Cache hints are already grounded in a concrete source. A newly read URL
    joins the exact reserve immediately before note extraction and stays only
    after it produces a note. Before read selection, the caller may also hold
    the explicitly audited bounded minimum-action probe; this is not inferred
    from an unrelated cache entry and is replaced as soon as actual page text
    exists.
    """

    claim_by_id = {target.claim.claim_id: target.claim for target in targets}
    note_by_id = {str(note.note_id): note for note in notes}
    hint_claims_by_url: dict[str, dict[str, AtomicClaim]] = {}
    for hint in hints:
        note = note_by_id[hint.note_id]
        hint_claims_by_url.setdefault(note.url, {})[hint.claim_id] = (
            claim_by_id[hint.claim_id]
        )

    admitted_claims_by_url: dict[str, dict[str, AtomicClaim]] = {}
    for url, claims in (admitted_read_source_claims or {}).items():
        for claim in claims:
            admitted_claims_by_url.setdefault(url, {})[claim.claim_id] = claim

    claim_groups_by_url: dict[str, dict[str, AtomicClaim]] = {
        url: dict(claims_by_id)
        for url, claims_by_id in hint_claims_by_url.items()
    }
    for url, claims_by_id in admitted_claims_by_url.items():
        claim_groups_by_url.setdefault(url, {}).update(claims_by_id)

    cached_batch_count = 0
    estimated_tokens = 0
    estimated_cost = 0.0
    admitted_read_source_batch_count = 0
    for url in sorted(claim_groups_by_url):
        source_text = source_cache.get(url)
        if source_text is None:
            continue
        claims = tuple(claim_groups_by_url[url].values())
        batches, tokens, cost = _estimate_verification_group(
            claims=claims,
            url=url,
            source_text=source_text,
            batch_size=verification_settings.batch_size,
            tracker=tracker,
            verification_model=verification_model,
            truth_condition_registry=truth_condition_registry,
        )
        if url in hint_claims_by_url:
            cached_batch_count += batches
        if url in admitted_claims_by_url:
            admitted_read_source_batch_count += batches
        estimated_tokens += tokens
        estimated_cost += cost

    # Reattribution is a required predecessor of verification for newly
    # extracted notes.  It is additive: using max() here allowed note
    # extraction to consume the attribution capacity while an unrelated
    # verification reserve still appeared healthy.
    required_downstream_tokens = (
        estimated_tokens
        + minimum_action_estimated_tokens
        + incremental_reattribution_estimated_tokens
    )
    required_downstream_cost = (
        estimated_cost
        + minimum_action_estimated_cost_usd
        + incremental_reattribution_estimated_cost_usd
    )
    reserved_tokens, reserved_cost = tracker.reserve_verification(
        tokens=required_downstream_tokens,
        cost_usd=required_downstream_cost,
        prerequisite_tokens=prerequisite_estimated_tokens,
        prerequisite_cost_usd=prerequisite_estimated_cost_usd,
    )
    return VerificationReserveAudit(
        method=(
            "actual_source_groups_plus_bounded_minimum_action"
            if minimum_action_estimated_tokens
            or minimum_action_estimated_cost_usd
            else "actual_source_groups_plus_incremental_reattribution"
            if incremental_reattribution_estimated_tokens
            or incremental_reattribution_estimated_cost_usd
            else "actual_cached_and_read_source_groups_after_prerequisite_allowance"
        ),
        reference_source_url=None,
        reference_source_chars=0,
        cached_hint_batch_count=cached_batch_count,
        admitted_read_source_batch_count=admitted_read_source_batch_count,
        admitted_read_source_urls=tuple(sorted(admitted_claims_by_url)),
        web_read_slots=0,
        planned_query_count=len(queries),
        planned_query_claim_count=len(
            {
                claim_id
                for query in queries
                for claim_id in query.claim_ids
            }
        ),
        estimated_tokens=estimated_tokens,
        estimated_cost_usd=estimated_cost,
        prerequisite_stage=prerequisite_stage,
        prerequisite_estimated_tokens=prerequisite_estimated_tokens,
        prerequisite_estimated_cost_usd=prerequisite_estimated_cost_usd,
        minimum_action_estimated_tokens=minimum_action_estimated_tokens,
        minimum_action_estimated_cost_usd=minimum_action_estimated_cost_usd,
        incremental_reattribution_estimated_tokens=(
            incremental_reattribution_estimated_tokens
        ),
        incremental_reattribution_estimated_cost_usd=(
            incremental_reattribution_estimated_cost_usd
        ),
        required_downstream_tokens=required_downstream_tokens,
        required_downstream_cost_usd=required_downstream_cost,
        reserved_tokens=reserved_tokens,
        reserved_cost_usd=reserved_cost,
        reserve_fully_funded=(
            reserved_tokens >= required_downstream_tokens
            and reserved_cost + 1e-12 >= required_downstream_cost
        ),
    )


def _parse_plan(
    content: Any,
    *,
    targets: Sequence[ClaimVerification],
    notes: Sequence[ResearchNote],
    checklist: ResearchChecklist,
    max_queries: int,
    target_element_ids_by_claim: Mapping[str, tuple[str, ...]] | None = None,
) -> tuple[
    tuple[CachedCandidateHint, ...],
    tuple[GapSearchQuery, ...],
    tuple[DeferredGapTarget, ...],
    tuple[dict[str, Any], ...],
    bool,
]:
    target_by_id = {target.claim.claim_id: target for target in targets}
    note_by_id = {str(note.note_id): note for note in notes}
    item_ids = set(checklist.in_scope_item_ids)
    rejected: list[dict[str, Any]] = []
    accepted_hints: list[CachedCandidateHint] = []
    accepted_queries: list[GapSearchQuery] = []
    if not isinstance(content, Mapping):
        return (
            (),
            (),
            (),
            (
                {
                    "stage": "plan",
                    "error": "plan must be a JSON object",
                    "raw": content,
                },
            ),
            False,
        )
    raw_hints = content.get("cached_candidates")
    raw_queries = content.get("queries")
    raw_deferred = content.get("deferred_targets", ())
    if not isinstance(raw_hints, (list, tuple)):
        rejected.append(
            {
                "stage": "plan",
                "error": "cached_candidates must be an array",
                "raw": raw_hints,
            }
        )
        raw_hints = ()
    if not isinstance(raw_queries, (list, tuple)):
        rejected.append(
            {
                "stage": "plan",
                "error": "queries must be an array",
                "raw": raw_queries,
            }
        )
        raw_queries = ()
    if not isinstance(raw_deferred, (list, tuple)):
        rejected.append(
            {
                "stage": "plan",
                "error": "deferred_targets must be an array",
                "raw": raw_deferred,
            }
        )
        raw_deferred = ()
    elif raw_deferred:
        rejected.append(
            {
                "stage": "deferred_target",
                "error": (
                    "planner_supplied_deferred_target; capacity deferrals "
                    "are derived by code"
                ),
                "raw": raw_deferred,
            }
        )
    identities_by_claim: dict[str, set[str]] = {
        claim_id: set() for claim_id in target_by_id
    }
    seen_hint_relations: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(raw_hints):
        try:
            hint = _RawCachedHint.model_validate(raw)
        except (TypeError, ValidationError, ValueError) as exc:
            rejected.append(
                {
                    "stage": "cached_candidate",
                    "index": index,
                    "error": str(exc),
                    "raw": raw,
                }
            )
            continue
        if hint.claim_id not in target_by_id:
            rejected.append(
                {
                    "stage": "cached_candidate",
                    "index": index,
                    "claim_id": hint.claim_id,
                    "error": "unknown target claim_id",
                    "raw": raw,
                }
            )
            continue
        relation_identity = (hint.claim_id, hint.note_id, hint.source_id)
        if relation_identity in seen_hint_relations:
            rejected.append(
                {
                    "stage": "cached_candidate",
                    "index": index,
                    "claim_id": hint.claim_id,
                    "error": "duplicate cached candidate relation",
                    "raw": raw,
                }
            )
            continue
        seen_hint_relations.add(relation_identity)
        target = target_by_id[hint.claim_id]
        _, _, used_publishers = _supporting_publisher_proxy_sets(target)
        existing_publishers = set(used_publishers)
        note = note_by_id.get(hint.note_id)
        identity = hint.publisher_identity.strip().casefold()
        error: str | None = None
        if note is None:
            error = "unknown note_id"
        elif note.source_id != hint.source_id:
            error = "note_id/source_id mismatch"
        elif not hint.independent_from_existing_publishers:
            error = "model did not judge publisher independent"
        elif (
            mechanical_error := _cached_route_mechanical_error(target, note)
        ) is not None:
            error = mechanical_error
        elif any(
            _identity_matches_proxy(hint.publisher_identity, proxy)
            for proxy in existing_publishers
        ):
            error = "publisher identity matches an existing domain label"
        elif (
            not identity
            or identity in identities_by_claim[hint.claim_id]
        ):
            error = "publisher identity is blank or repeated for this claim"
        if error is not None:
            rejected.append(
                {
                    "stage": "cached_candidate",
                    "claim_id": hint.claim_id,
                    "error": error,
                    "raw": hint.model_dump(mode="json"),
                }
            )
            continue
        identities_by_claim[hint.claim_id].add(identity)
        accepted_hints.append(
            CachedCandidateHint(
                claim_id=hint.claim_id,
                note_id=hint.note_id,
                source_id=hint.source_id,
                publisher_identity=hint.publisher_identity.strip(),
                independence_rationale=hint.independence_rationale.strip(),
            )
        )

    for index, raw in enumerate(raw_queries):
        try:
            query = _RawSearchQuery.model_validate(raw)
        except (TypeError, ValidationError, ValueError) as exc:
            rejected.append(
                {
                    "stage": "search_query",
                    "index": index,
                    "error": str(exc),
                    "raw": raw,
                }
            )
            continue
        claim_ids = tuple(dict.fromkeys(query.claim_ids))
        if any(claim_id not in target_by_id for claim_id in claim_ids):
            rejected.append(
                {
                    "stage": "search_query",
                    "index": index,
                    "error": "query must name known target claim_ids",
                    "raw": query.model_dump(mode="json"),
                }
            )
            continue
        if len(accepted_queries) >= max_queries:
            rejected.append(
                {
                    "stage": "search_query",
                    "index": index,
                    "claim_ids": list(claim_ids),
                    "error": "gap search query cap reached",
                    "raw": query.model_dump(mode="json"),
                }
            )
            continue
        if query.item_id not in item_ids:
            rejected.append(
                {
                    "stage": "search_query",
                    "index": index,
                    "claim_ids": list(claim_ids),
                    "error": "unknown checklist item_id",
                    "raw": query.model_dump(mode="json"),
                }
            )
            continue
        accepted_queries.append(
            GapSearchQuery(
                claim_ids=claim_ids,
                target_element_ids=tuple(
                    element_id
                    for claim_id in claim_ids
                    for element_id in (
                        (target_element_ids_by_claim or {}).get(claim_id, ())
                    )
                ),
                item_id=query.item_id,
                query=query.query,
            )
        )

    return (
        tuple(accepted_hints),
        tuple(accepted_queries),
        (),
        tuple(rejected),
        True,
    )


def _plan_route_ids(
    hints: Sequence[CachedCandidateHint],
    queries: Sequence[GapSearchQuery],
) -> set[str]:
    """Return target IDs with one accepted, executable planning route."""

    return {hint.claim_id for hint in hints} | {
        claim_id for query in queries for claim_id in query.claim_ids
    }


def _derive_capacity_deferrals(
    *,
    target_claim_ids: Sequence[str],
    routed_claim_ids: set[str],
    accepted_query_count: int,
    query_cap: int,
) -> tuple[DeferredGapTarget, ...]:
    """Record unrouted scope without asking the model to judge deferral.

    This is a code-owned route-coverage fact. It deliberately says nothing
    about whether evidence exists, whether every available query would help,
    or whether another pass would find it.
    """

    rationale = (
        "code derived: target received no accepted cached candidate or "
        f"issued query route in this bounded plan; issued queries="
        f"{accepted_query_count}/{query_cap}"
    )
    return tuple(
        DeferredGapTarget(
            claim_id=claim_id,
            reason="query_capacity_not_allocated",
            priority_rationale=rationale,
        )
        for claim_id in target_claim_ids
        if claim_id not in routed_claim_ids
    )


def _parse_reads(
    content: Any,
    *,
    targets: Sequence[ClaimVerification],
    searches: Sequence[GapSearchRecord],
    cached_hints: Sequence[CachedCandidateHint],
    checklist: ResearchChecklist,
    max_reads: int,
) -> tuple[
    tuple[GapReadSelection, ...],
    tuple[dict[str, Any], ...],
]:
    target_by_id = {target.claim.claim_id: target for target in targets}
    item_ids = set(checklist.in_scope_item_ids)
    allowed: dict[str, set[tuple[str, str]]] = {}
    routed_elements: dict[tuple[str, str, str], list[str]] = {}
    for record in searches:
        for result in record.results:
            allowed.setdefault(result.url, set()).update(
                (claim_id, record.query.item_id)
                for claim_id in record.query.claim_ids
            )
            for claim_id in record.query.claim_ids:
                key = (result.url, record.query.item_id, claim_id)
                values = routed_elements.setdefault(key, [])
                for element_id in record.query.target_element_ids:
                    if (
                        element_id.split("::tc-", 1)[0] == claim_id
                        and element_id not in values
                    ):
                        values.append(element_id)
    accepted: list[GapReadSelection] = []
    rejected: list[dict[str, Any]] = []
    identities_by_claim: dict[str, set[str]] = {
        claim_id: set() for claim_id in target_by_id
    }
    for hint in cached_hints:
        identities_by_claim[hint.claim_id].add(
            hint.publisher_identity.strip().casefold()
        )
    raw_reads = content.get("reads") if isinstance(content, Mapping) else None
    if not isinstance(raw_reads, (list, tuple)):
        return (), (
            {"stage": "read_selection", "error": "reads must be an array"},
        )

    seen_urls: set[str] = set()
    for index, raw in enumerate(raw_reads):
        try:
            proposal = _RawRead.model_validate(raw)
        except (TypeError, ValidationError, ValueError) as exc:
            rejected.append(
                {
                    "stage": "read_selection",
                    "index": index,
                    "error": str(exc),
                    "raw": raw,
                }
            )
            continue
        error: str | None = None
        claim_ids = tuple(dict.fromkeys(proposal.claim_ids))
        identity = proposal.publisher_identity.strip().casefold()
        if len(accepted) >= max_reads:
            error = "gap read cap reached"
        elif proposal.url in seen_urls:
            error = "duplicate read URL"
        elif proposal.url not in allowed:
            error = "URL was not returned by the bounded searches"
        elif proposal.item_id not in item_ids:
            error = "unknown checklist item_id"
        elif not claim_ids:
            error = "read must name at least one target claim_id"
        elif proposal.item_id not in {
            item_id for _, item_id in allowed[proposal.url]
        }:
            error = "URL/item relation was not present in search routing"
        elif not proposal.independent_from_existing_publishers:
            error = "model did not judge publisher independent"
        elif not identity:
            error = "publisher identity must not be blank"
        if error is not None:
            rejected.append(
                {
                    "stage": "read_selection",
                    "index": index,
                    "error": error,
                    "raw": proposal.model_dump(mode="json"),
                }
            )
            continue

        # Independence is a claim-source property.  A grouped proposal may be
        # redundant for one claim and new for another; rejecting the whole URL
        # lets the redundant sibling erase a valid route.  Validate and audit
        # each claim independently, then read the URL once for the survivors.
        allowed_claim_ids = {
            claim_id
            for claim_id, item_id in allowed[proposal.url]
            if item_id == proposal.item_id
        }
        accepted_claim_ids: list[str] = []
        publisher_proxy = _publisher_proxy(proposal.url)
        for claim_id in claim_ids:
            claim_error: str | None = None
            if claim_id not in target_by_id:
                claim_error = "unknown target claim_id"
            elif claim_id not in allowed_claim_ids:
                claim_error = "claim was not routed to this URL by search"
            elif _source_was_already_checked(
                target_by_id[claim_id],
                source_id=source_id_for_url(proposal.url),
                url=proposal.url,
            ):
                claim_error = "source was already checked for this claim"
            elif publisher_proxy in set(
                _supporting_publisher_proxy_sets(
                    target_by_id[claim_id]
                )[2]
            ):
                claim_error = "publisher domain proxy already supports this claim"
            elif any(
                _identity_matches_proxy(proposal.publisher_identity, proxy)
                for proxy in _supporting_publisher_proxy_sets(
                    target_by_id[claim_id]
                )[2]
            ):
                claim_error = "publisher identity matches an existing domain label"
            elif identity in identities_by_claim[claim_id]:
                claim_error = "publisher identity repeated for this claim"
            if claim_error is not None:
                rejected.append(
                    {
                        "stage": "read_selection_claim",
                        "index": index,
                        "claim_id": claim_id,
                        "error": claim_error,
                        "raw": proposal.model_dump(mode="json"),
                    }
                )
                continue
            accepted_claim_ids.append(claim_id)
        if not accepted_claim_ids:
            continue
        seen_urls.add(proposal.url)
        for claim_id in accepted_claim_ids:
            identities_by_claim[claim_id].add(identity)
        accepted.append(
            GapReadSelection(
                url=proposal.url.strip(),
                item_id=proposal.item_id,
                claim_ids=tuple(accepted_claim_ids),
                target_element_ids=tuple(
                    element_id
                    for claim_id in accepted_claim_ids
                    for element_id in routed_elements.get(
                        (proposal.url, proposal.item_id, claim_id),
                        (),
                    )
                ),
                publisher_identity=proposal.publisher_identity.strip(),
                independence_rationale=(
                    proposal.independence_rationale.strip()
                ),
            )
        )
    return tuple(accepted), tuple(rejected)


def _hint_candidates(
    hints: Sequence[CachedCandidateHint],
    *,
    notes: Sequence[ResearchNote],
) -> dict[str, tuple[CandidateSource, ...]]:
    note_by_id = {str(note.note_id): note for note in notes}
    by_claim: dict[str, list[CandidateSource]] = {}
    for hint in hints:
        note = note_by_id[hint.note_id]
        by_claim.setdefault(hint.claim_id, []).append(
            CandidateSource(
                note_id=hint.note_id,
                source_id=hint.source_id,
                item_id=note.item_id,
                publisher=note.publisher,
                url=note.url,
                location_status=note.location_status,
                resolution=SourceResolution.DIRECT,
            )
        )
    return {
        claim_id: tuple(candidates)
        for claim_id, candidates in by_claim.items()
    }


def _merge_attributions(
    base: AttributionResult,
    refreshed: AttributionResult,
    *,
    hint_candidates: Mapping[str, Sequence[CandidateSource]],
) -> AttributionResult:
    refreshed_by_id = {
        entry.claim.claim_id: entry for entry in refreshed.attributions
    }
    merged: list[ClaimAttribution] = []
    for original in base.attributions:
        claim_id = original.claim.claim_id
        addition = refreshed_by_id.get(claim_id)
        candidates: list[CandidateSource] = list(original.candidates)
        errors: list[AttributionError] = list(original.errors)
        if addition is not None:
            candidates.extend(addition.candidates)
            errors.extend(addition.errors)
        candidates.extend(hint_candidates.get(claim_id, ()))
        unique_candidates: list[CandidateSource] = []
        seen_candidates: set[tuple[str, str, str]] = set()
        for candidate in candidates:
            key = (candidate.note_id, candidate.source_id, candidate.url)
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            unique_candidates.append(candidate)
        unique_errors: list[AttributionError] = []
        seen_errors: set[tuple[str, str, str]] = set()
        for error in errors:
            key = (error.claim_id, error.code, error.detail)
            if key in seen_errors:
                continue
            seen_errors.add(key)
            unique_errors.append(error)
        if unique_candidates and unique_errors:
            status = AttributionStatus.CANDIDATE_SOURCES_WITH_ERRORS
        elif unique_candidates:
            status = AttributionStatus.CANDIDATE_SOURCES
        elif unique_errors:
            status = AttributionStatus.ATTRIBUTION_ERROR
        else:
            status = AttributionStatus.NO_CANDIDATE_SOURCE
        merged.append(
            ClaimAttribution(
                # The claim object is deliberately the initial frozen object.
                claim=original.claim,
                status=status,
                candidates=tuple(unique_candidates),
                errors=tuple(unique_errors),
            )
        )
    return AttributionResult(
        attributions=tuple(merged),
        inspected_pages=base.inspected_pages + refreshed.inspected_pages,
        usage=base.usage + refreshed.usage,
        diagnostics=base.diagnostics + refreshed.diagnostics,
        stop_reason=refreshed.stop_reason,
    )


def _relation_identity(
    relation: VerifiedSourceRelation,
) -> tuple[str, str, str]:
    return (relation.claim_id, relation.source_id, relation.url)


def _candidate_identity(
    claim_id: str,
    candidate: CandidateSource,
) -> tuple[str, str, str]:
    return (claim_id, candidate.source_id, candidate.url)


def _completed_relation_identities(
    verification: VerificationResult,
) -> set[tuple[str, str, str]]:
    return {
        _relation_identity(relation)
        for claim in verification.claims
        for relation in claim.relations
        if relation.status is VerificationRecordStatus.COMPLETED
    }


def _relation_count(verification: VerificationResult) -> int:
    return sum(len(claim.relations) for claim in verification.claims)


def _unchanged_merge_audit(
    verification: VerificationResult,
) -> VerificationMergeAudit:
    completed = len(_completed_relation_identities(verification))
    relations = _relation_count(verification)
    return VerificationMergeAudit(
        initial_relation_count=relations,
        initial_completed_relation_count=completed,
        incremental_relation_count=0,
        incremental_completed_relation_count=0,
        final_relation_count=relations,
        final_completed_relation_count=completed,
        preserved_initial_completed_relation_count=completed,
        completed_relation_count_non_decreasing=True,
    )


def _information_yield(
    initial: VerificationResult,
    final: VerificationResult,
    *,
    stop_reason: EvidenceGapStopReason,
) -> EvidenceGapInformationAudit:
    """Measure newly completed checks without treating support as success."""

    initial_completed = _completed_relation_identities(initial)
    final_relations = {
        _relation_identity(relation): relation
        for claim in final.claims
        for relation in claim.relations
        if relation.status is VerificationRecordStatus.COMPLETED
    }
    newly_completed = tuple(
        relation
        for identity, relation in sorted(final_relations.items())
        if identity not in initial_completed
    )
    verdict_counts = {
        verdict.value: sum(
            relation.semantic_verdict is verdict
            for relation in newly_completed
        )
        for verdict in VerificationVerdict
    }
    initial_publishers = {
        relation.publisher_domain_proxy
        for claim in initial.claims
        for relation in claim.relations
        if relation.status is VerificationRecordStatus.COMPLETED
    }
    new_publishers = {
        relation.publisher_domain_proxy for relation in newly_completed
    }
    initial_claim_publishers = {
        (relation.claim_id, relation.publisher_domain_proxy)
        for claim in initial.claims
        for relation in claim.relations
        if relation.status is VerificationRecordStatus.COMPLETED
    }
    new_claim_publishers = {
        (relation.claim_id, relation.publisher_domain_proxy)
        for relation in newly_completed
    }
    initial_states = {
        claim.claim.claim_id: claim.state for claim in initial.claims
    }
    return EvidenceGapInformationAudit(
        pass_completed_within_budget=(
            stop_reason is EvidenceGapStopReason.COMPLETED
        ),
        new_completed_relation_count=len(newly_completed),
        new_completed_verdict_counts=verdict_counts,
        new_publisher_domain_proxies=tuple(
            sorted(new_publishers - initial_publishers)
        ),
        new_claim_publisher_relation_count=len(
            new_claim_publishers - initial_claim_publishers
        ),
        claims_newly_corroborated=sum(
            claim.state is ClaimEvidenceState.CORROBORATED
            and initial_states.get(claim.claim.claim_id)
            is not ClaimEvidenceState.CORROBORATED
            for claim in final.claims
        ),
        claims_newly_supported_by_multiple_domain_proxies=sum(
            claim.state
            is ClaimEvidenceState.SUPPORTED_MULTIPLE_DOMAIN_PROXIES
            and initial_states.get(claim.claim.claim_id)
            is not ClaimEvidenceState.SUPPORTED_MULTIPLE_DOMAIN_PROXIES
            for claim in final.claims
        ),
        claims_newly_conflicting=sum(
            claim.state is ClaimEvidenceState.CONFLICTING_EVIDENCE
            and initial_states.get(claim.claim.claim_id)
            is not ClaimEvidenceState.CONFLICTING_EVIDENCE
            for claim in final.claims
        ),
    )


def _incremental_attributions(
    attributions: Sequence[ClaimAttribution],
    *,
    initial_verification: VerificationResult,
) -> tuple[ClaimAttribution, ...]:
    """Return only claim/source identities absent from initial verification.

    Claims and cached source text are frozen during the gap pass. Additional
    note IDs for an already checked claim/source identity therefore do not
    change the verifier input and must not trigger a duplicate model call.
    """

    existing = {
        _relation_identity(relation)
        for claim in initial_verification.claims
        for relation in claim.relations
        if relation.status in _TERMINAL_RELATION_STATUSES
    }
    incremental: list[ClaimAttribution] = []
    for attribution in attributions:
        candidates = tuple(
            candidate
            for candidate in attribution.candidates
            if _candidate_identity(attribution.claim.claim_id, candidate)
            not in existing
        )
        if not candidates:
            continue
        incremental.append(
            ClaimAttribution(
                claim=attribution.claim,
                status=(
                    AttributionStatus.CANDIDATE_SOURCES_WITH_ERRORS
                    if attribution.errors
                    else AttributionStatus.CANDIDATE_SOURCES
                ),
                candidates=candidates,
                errors=attribution.errors,
            )
        )
    return tuple(incremental)


def _merge_verifications(
    initial: VerificationResult,
    refreshed_targets: VerificationResult,
    *,
    merged_attribution: AttributionResult,
    truth_condition_registry: TruthConditionRegistry | None = None,
) -> tuple[VerificationResult, VerificationMergeAudit]:
    """Add source relations without allowing failed refreshes to erase work."""

    if truth_condition_registry is not None:
        registry_hash = truth_condition_registry_sha256(truth_condition_registry)
        refreshed_registry_hash = truth_condition_registry_sha256(
            select_truth_condition_registry(
                truth_condition_registry,
                tuple(
                    claim.claim.claim_id for claim in refreshed_targets.claims
                ),
            )
        )
        for label, result_hash, expected_hash in (
            ("initial", initial.truth_condition_registry_sha256, registry_hash),
            (
                "refreshed",
                refreshed_targets.truth_condition_registry_sha256,
                refreshed_registry_hash,
            ),
        ):
            if result_hash != expected_hash:
                raise ValueError(
                    f"{label} verification uses a different truth-condition registry"
                )

    refreshed_by_id = {
        result.claim.claim_id: result
        for result in refreshed_targets.claims
    }
    attribution_by_id = {
        result.claim.claim_id: result
        for result in merged_attribution.attributions
    }
    protected: list[ProtectedCompletedRelation] = []
    claims: list[ClaimVerification] = []
    for original in initial.claims:
        claim_id = original.claim.claim_id
        relation_by_identity = {
            _relation_identity(relation): relation
            for relation in original.relations
        }
        refreshed = refreshed_by_id.get(claim_id)
        if refreshed is not None:
            for relation in refreshed.relations:
                identity = _relation_identity(relation)
                existing = relation_by_identity.get(identity)
                if existing is None:
                    relation_by_identity[identity] = relation
                    continue
                if existing.status is VerificationRecordStatus.COMPLETED:
                    protected.append(
                        ProtectedCompletedRelation(
                            claim_id=claim_id,
                            source_id=existing.source_id,
                            url=existing.url,
                            attempted_status=relation.status,
                            reason=(
                                "a gap refresh cannot replace an already "
                                "completed source verdict"
                            ),
                        )
                    )
                    # Completed source verdicts are immutable in a gap pass.
                    continue
                if relation.status is VerificationRecordStatus.COMPLETED:
                    relation_by_identity[identity] = relation

        attribution = attribution_by_id[claim_id]
        merged_relations = tuple(relation_by_identity.values())
        truth_condition_aggregate = None
        if truth_condition_registry is not None:
            registry_entry = truth_condition_registry.entry_for(claim_id)
            if registry_entry is None:
                raise ValueError(
                    "truth-condition registry omitted evidence-gap claim "
                    f"{claim_id}"
                )
            truth_condition_aggregate = aggregate_truth_condition_claim(
                registry_entry,
                tuple(
                    element.as_assessment()
                    for relation in merged_relations
                    for element in relation.element_relations
                ),
                expected_source_ids=tuple(
                    dict.fromkeys(
                        candidate.source_id
                        for candidate in attribution.candidates
                    )
                ),
            )
        claims.append(
            build_claim_verification(
                original.claim,
                merged_relations,
                required_sources=original.corroboration_target,
                attribution_status=attribution.status,
                truth_condition_aggregate=truth_condition_aggregate,
            )
        )

    unique_relations = {
        (relation.source_id, relation.url): relation
        for claim in claims
        for relation in claim.relations
    }
    confirmed_lineages = sum(
        relation.source_lineage is not None
        and relation.source_lineage.status is SourceLineageStatus.CONFIRMED
        for relation in unique_relations.values()
    )
    proposed_lineages = sum(
        relation.source_lineage is not None
        and relation.source_lineage.status is SourceLineageStatus.PROPOSED
        for relation in unique_relations.values()
    )
    merged = VerificationResult(
        claims=tuple(claims),
        usage=initial.usage + refreshed_targets.usage,
        diagnostics=initial.diagnostics + refreshed_targets.diagnostics,
        independence=PublisherIndependenceAudit(
            confirmed_assessment_count=confirmed_lineages,
            proposed_assessment_count=proposed_lineages,
            unresolved_relation_count=(
                len(unique_relations) - confirmed_lineages - proposed_lineages
            ),
        ),
        truth_condition_registry_sha256=(
            truth_condition_registry_sha256(truth_condition_registry)
            if truth_condition_registry is not None
            else initial.truth_condition_registry_sha256
        ),
    )
    initial_completed = _completed_relation_identities(initial)
    final_completed = _completed_relation_identities(merged)
    missing_completed = initial_completed - final_completed
    if missing_completed:
        raise AssertionError(
            "evidence-gap merge removed completed verification relations: "
            f"{sorted(missing_completed)}"
        )
    audit = VerificationMergeAudit(
        initial_relation_count=_relation_count(initial),
        initial_completed_relation_count=len(initial_completed),
        incremental_relation_count=_relation_count(refreshed_targets),
        incremental_completed_relation_count=len(
            _completed_relation_identities(refreshed_targets)
        ),
        final_relation_count=_relation_count(merged),
        final_completed_relation_count=len(final_completed),
        preserved_initial_completed_relation_count=len(
            initial_completed & final_completed
        ),
        completed_relation_count_non_decreasing=(
            len(final_completed) >= len(initial_completed)
        ),
        protected_completed_relations=tuple(protected),
    )
    if not audit.completed_relation_count_non_decreasing:
        raise AssertionError(
            "evidence-gap completed relation count decreased without an "
            "audited relation change"
        )
    return merged, audit


async def run_evidence_gap_round(
    *,
    canonical_draft: str,
    checklist: ResearchChecklist,
    blocks: Sequence[MarkdownBlock],
    ledger: ResearchLedger,
    initial_attribution: AttributionResult,
    initial_verification: VerificationResult,
    gap_model: EvidenceGapModelClient,
    note_model: EvidenceGapModelClient,
    attribution_model: AttributionModelClient,
    verification_model: VerificationModelClient,
    tavily_client: TavilyClient,
    budget: EvidenceGapBudget,
    attribution_settings: AttributionSettings | None = None,
    verification_settings: VerificationSettings | None = None,
    truth_condition_registry: TruthConditionRegistry | None = None,
    corroboration_targets: Mapping[str, int] | None = None,
    required_independent_sources: Mapping[str, int] | None = None,
    estimate_input_tokens: Callable[[Any, str], int] | None = None,
    estimate_cost_usd: Callable[[Any, str], float] | None = None,
    explicit_target_claim_ids: Sequence[str] | None = None,
    explicit_target_element_ids: (
        Mapping[str, Sequence[str]] | None
    ) = None,
    plan_prompt_builder: Callable[..., str] | None = None,
    ledger_event_prefix: str = "gap",
) -> EvidenceGapResult:
    """Run one cache-first gap pass, then reattribute and reverify once."""

    if (
        corroboration_targets is not None
        and required_independent_sources is not None
    ):
        raise ValueError(
            "use corroboration_targets or legacy "
            "required_independent_sources, not both"
        )
    if not ledger_event_prefix.strip():
        raise ValueError("ledger_event_prefix must not be blank")

    def ledger_event(name: str) -> str:
        if ledger_event_prefix == "gap":
            return "gap_stop" if name == "stop" else name
        return f"{ledger_event_prefix}_{name}"

    if explicit_target_element_ids is not None and explicit_target_claim_ids is None:
        raise ValueError(
            "explicit target elements require explicit target claim IDs"
        )
    if explicit_target_claim_ids is None:
        targets = select_evidence_gap_targets(initial_verification)
    else:
        requested = tuple(dict.fromkeys(explicit_target_claim_ids))
        available = {
            result.claim.claim_id: result
            for result in initial_verification.claims
            if result.claim.citation_requirement
            == CitationRequirement.EXTERNAL
        }
        unknown = tuple(
            claim_id for claim_id in requested if claim_id not in available
        )
        if unknown:
            raise ValueError(
                "explicit evidence-gap targets must be external claims: "
                + ", ".join(unknown)
            )
        targets = tuple(available[claim_id] for claim_id in requested)
    target_element_ids_by_claim: dict[str, tuple[str, ...]] = {}
    if explicit_target_element_ids is not None:
        if truth_condition_registry is None:
            raise ValueError(
                "explicit target elements require a truth-condition registry"
            )
        supplied_claim_ids = set(explicit_target_element_ids)
        requested_claim_ids = {target.claim.claim_id for target in targets}
        if supplied_claim_ids != requested_claim_ids:
            raise ValueError(
                "explicit target-element mapping must cover exactly the "
                "explicit target claims"
            )
        for claim_id in tuple(target.claim.claim_id for target in targets):
            entry = truth_condition_registry.entry_for(claim_id)
            if entry is None:
                raise ValueError(
                    f"truth-condition registry omitted target {claim_id}"
                )
            selected = tuple(
                dict.fromkeys(explicit_target_element_ids[claim_id])
            )
            registered = {element.element_id for element in entry.elements}
            if not selected or not set(selected).issubset(registered):
                raise ValueError(
                    "explicit target elements must be a non-empty registered "
                    f"subset for {claim_id}"
                )
            target_element_ids_by_claim[claim_id] = selected
    initial_states = {
        result.claim.claim_id: result.state
        for result in targets
    }
    if not targets:
        return EvidenceGapResult(
            stop_reason=EvidenceGapStopReason.NO_TARGETS,
            stop_detail="initial verification exposed no eligible evidence gaps",
            verification_merge=_unchanged_merge_audit(initial_verification),
            final_attribution=initial_attribution,
            final_verification=initial_verification,
        )

    frozen_claims = {
        entry.claim.claim_id: entry.claim.model_dump(mode="json")
        for entry in initial_attribution.attributions
    }
    frozen_draft = canonical_draft
    tracker = _BudgetTracker(
        budget,
        estimate_input_tokens=estimate_input_tokens,
        estimate_cost_usd=estimate_cost_usd,
    )
    rejected: list[dict[str, Any]] = []
    searches: list[GapSearchRecord] = []
    selections: tuple[GapReadSelection, ...] = ()
    acquisitions: list[GapSourceAcquisition] = []
    added_source_urls: list[str] = []
    added_note_ids: list[str] = []
    incremental_routed_claim_ids: list[str] = []
    hints: tuple[CachedCandidateHint, ...] = ()
    deferred_targets: tuple[DeferredGapTarget, ...] = ()
    verification_reserve: VerificationReserveAudit | None = None
    verification_reserve_history: list[VerificationReserveAudit] = []
    planning_capacity: EvidenceGapPlanningCapacityAudit | None = None
    plan_attempt_count = 0
    selected_planning_attempt: int | None = None
    unused_query_slots = 0
    stop_reason = EvidenceGapStopReason.COMPLETED
    stop_detail = "single evidence-gap pass completed"
    verification_merge = _unchanged_merge_audit(initial_verification)
    active_verification_settings = (
        verification_settings or VerificationSettings()
    )
    active_attribution_settings = attribution_settings or AttributionSettings()

    try:
        active_plan_builder = (
            plan_prompt_builder or build_evidence_gap_plan_prompt
        )
        supports_registry = plan_prompt_builder is None
        if plan_prompt_builder is not None:
            try:
                plan_signature = inspect.signature(plan_prompt_builder)
            except (TypeError, ValueError):
                plan_signature = None
            supports_registry = bool(
                plan_signature is not None
                and (
                    "truth_condition_registry" in plan_signature.parameters
                    or any(
                        parameter.kind
                        is inspect.Parameter.VAR_KEYWORD
                        for parameter in plan_signature.parameters.values()
                    )
                )
            )
        def build_plan_prompt(max_queries: int) -> str:
            plan_kwargs: dict[str, Any] = {
                "targets": targets,
                "notes": ledger.notes,
                "checklist": checklist,
                "max_queries": max_queries,
            }
            if supports_registry:
                plan_kwargs["truth_condition_registry"] = (
                    truth_condition_registry
                )
            return active_plan_builder(**plan_kwargs)

        effective_max_queries = budget.max_search_queries
        plan_prompt = build_plan_prompt(effective_max_queries)
        queries: tuple[GapSearchQuery, ...] = ()
        plan_attempt_count = 1
        planning_capacity = _planning_capacity_audit(
            plan_prompt=plan_prompt,
            tracker=tracker,
            budget=budget,
            advertised_max_search_queries=effective_max_queries,
            targets=targets,
            notes=ledger.notes,
            source_cache=ledger.source_cache,
            checklist=checklist,
            gap_model=gap_model,
            note_model=note_model,
            attribution_model=attribution_model,
            verification_model=verification_model,
            attribution_settings=active_attribution_settings,
            truth_condition_registry=truth_condition_registry,
            target_element_ids_by_claim=target_element_ids_by_claim,
        )
        if (
            not planning_capacity.reserve_fully_funded
            and effective_max_queries > 0
            and _has_mechanically_executable_cached_route(
                targets=targets,
                notes=ledger.notes,
                source_cache=ledger.source_cache,
            )
        ):
            # A cache route is a real alternative only after web search is
            # removed from the model's action space. Rebuild the unpaid prompt
            # with an explicit zero-query budget, then prove that route. This
            # preserves cache-first work without pretending it funds a web
            # route the planner could otherwise select.
            cache_only_prompt = build_plan_prompt(0)
            cache_only_capacity = _planning_capacity_audit(
                plan_prompt=cache_only_prompt,
                tracker=tracker,
                budget=budget,
                advertised_max_search_queries=0,
                targets=targets,
                notes=ledger.notes,
                source_cache=ledger.source_cache,
                checklist=checklist,
                gap_model=gap_model,
                note_model=note_model,
                attribution_model=attribution_model,
                verification_model=verification_model,
                attribution_settings=active_attribution_settings,
                truth_condition_registry=truth_condition_registry,
                target_element_ids_by_claim=target_element_ids_by_claim,
            )
            if cache_only_capacity.reserve_fully_funded:
                effective_max_queries = 0
                plan_prompt = cache_only_prompt
                planning_capacity = cache_only_capacity
                rejected.append(
                    {
                        "stage": "planning_capacity",
                        "outcome": "downgraded_to_cache_only",
                        "reason": (
                            "a web route did not fit, so the unpaid plan was "
                            "rebuilt with max_queries=0 before model execution"
                        ),
                    }
                )
        if len(plan_prompt) > budget.max_planning_prompt_chars:
            raise _GapBudgetExhausted(
                "gap budget: compacted planning prompt exceeds its "
                "character bound; "
                f"prompt_chars={len(plan_prompt)}; "
                f"max={budget.max_planning_prompt_chars}"
            )
        if (
            planning_capacity.estimated_planning_input_tokens
            > min(budget.max_planning_input_tokens, budget.max_tokens)
        ):
            raise _GapBudgetExhausted(
                "gap budget: compacted planning prompt exceeds its "
                "input-token bound; "
                "estimated_input_tokens="
                f"{planning_capacity.estimated_planning_input_tokens}; "
                "max="
                f"{min(budget.max_planning_input_tokens, budget.max_tokens)}"
            )
        if not planning_capacity.reserve_fully_funded:
            deferred_targets = _derive_capacity_deferrals(
                target_claim_ids=tuple(
                    target.claim.claim_id for target in targets
                ),
                routed_claim_ids=set(),
                accepted_query_count=0,
                query_cap=effective_max_queries,
            )
            unused_query_slots = effective_max_queries
            raise _GapBudgetExhausted(
                "planning cannot preserve output headroom plus one bounded "
                "cached verification or web note-attribution-verification "
                "route"
            )
        plan_response = await tracker.call(
            gap_model,
            plan_prompt,
            stage="cache_review_and_search_plan",
        )
        try:
            plan_content = _decode_response(plan_response)
            (
                hints,
                queries,
                _attempt_deferred,
                plan_rejected,
                _plan_shape_valid,
            ) = _parse_plan(
                plan_content,
                targets=targets,
                notes=ledger.notes,
                checklist=checklist,
                max_queries=effective_max_queries,
                target_element_ids_by_claim=target_element_ids_by_claim,
            )
        except (TypeError, ValidationError, ValueError) as exc:
            hints = ()
            queries = ()
            plan_rejected = (
                {
                    "stage": "plan",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        rejected.extend(
            {**entry, "attempt": plan_attempt_count}
            for entry in plan_rejected
        )
        selected_planning_attempt = plan_attempt_count
        routed_ids = _plan_route_ids(hints, queries)
        deferred_targets = _derive_capacity_deferrals(
            target_claim_ids=tuple(
                target.claim.claim_id for target in targets
            ),
            routed_claim_ids=routed_ids,
            accepted_query_count=len(queries),
            query_cap=effective_max_queries,
        )
        unused_query_slots = max(0, effective_max_queries - len(queries))
        for query in queries:
            try:
                results = tuple(
                    await search(
                        query.query,
                        tavily_client=tavily_client,
                        max_results=budget.max_results_per_search,
                        timeout_seconds=budget.provider_timeout_seconds,
                    )
                )
                searches.append(GapSearchRecord(query=query, results=results))
            except Exception as exc:
                searches.append(
                    GapSearchRecord(
                        query=query,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        successful_route_ids = {hint.claim_id for hint in hints} | {
            claim_id
            for search_record in searches
            if search_record.error is None
            for claim_id in search_record.query.claim_ids
        }
        queried_ids = {
            claim_id
            for search_record in searches
            for claim_id in search_record.query.claim_ids
        }
        deferral_by_id = {
            target.claim_id: target for target in deferred_targets
        }
        for target in targets:
            claim_id = target.claim.claim_id
            if claim_id in successful_route_ids or claim_id in deferral_by_id:
                continue
            deferral_by_id[claim_id] = DeferredGapTarget(
                claim_id=claim_id,
                reason=(
                    "search_route_failed"
                    if claim_id in queried_ids
                    else "query_capacity_not_allocated"
                ),
                priority_rationale=(
                    "code derived: every issued search route for this target "
                    "ended with a provider error"
                    if claim_id in queried_ids
                    else "code derived: target received no accepted cached "
                    "candidate or issued query route"
                ),
            )
        deferred_targets = tuple(
            deferral_by_id[target.claim.claim_id]
            for target in targets
            if target.claim.claim_id in deferral_by_id
            and target.claim.claim_id not in successful_route_ids
        )

        read_prompt: str | None = None
        prerequisite_tokens = 0
        prerequisite_cost = 0.0
        if any(record.results for record in searches) and budget.max_reads:
            read_prompt = build_evidence_gap_read_prompt(
                targets=targets,
                searches=searches,
                max_reads=budget.max_reads,
                truth_condition_registry=truth_condition_registry,
            )
            prerequisite_tokens, prerequisite_cost = tracker._estimate(
                gap_model,
                read_prompt,
            )
            prerequisite_tokens += math.ceil(
                prerequisite_tokens
                * budget.planning_output_headroom_ratio
            )
            prerequisite_cost *= 1.0 + budget.planning_output_headroom_ratio

        selected_web_allowances: tuple[tuple[int, float, int, float, int, float], ...] = ()
        if read_prompt is not None:
            routed_web_claim_ids = tuple(
                dict.fromkeys(
                    claim_id
                    for query in queries
                    for claim_id in query.claim_ids
                )
            )
            selected_web_allowances = tuple(
                _bounded_web_action_allowances(
                    targets=targets,
                    routed_claim_ids=routed_web_claim_ids,
                    checklist=checklist,
                    tracker=tracker,
                    budget=budget,
                    note_model=note_model,
                    attribution_model=attribution_model,
                    verification_model=verification_model,
                    attribution_settings=active_attribution_settings,
                    truth_condition_registry=truth_condition_registry,
                    target_element_ids_by_claim=target_element_ids_by_claim,
                ).values()
            )

        def selected_web_pressure(
            allowance: tuple[int, float, int, float, int, float],
        ) -> tuple[float, float, int, float]:
            remaining_tokens = max(
                0,
                budget.max_tokens
                - tracker.tokens_used
                - prerequisite_tokens,
            )
            remaining_cost = max(
                0.0,
                budget.max_cost_usd
                - tracker.cost_used
                - prerequisite_cost,
            )
            token_ratio = (
                allowance[4] / remaining_tokens
                if remaining_tokens
                else (0.0 if allowance[4] == 0 else math.inf)
            )
            cost_ratio = (
                allowance[5] / remaining_cost
                if remaining_cost
                else (0.0 if allowance[5] <= 1e-12 else math.inf)
            )
            return (
                max(token_ratio, cost_ratio),
                token_ratio + cost_ratio,
                allowance[4],
                allowance[5],
            )

        selected_web_allowance = (
            min(selected_web_allowances, key=selected_web_pressure)
            if selected_web_allowances
            else None
        )

        # Verification is downstream of read selection. Reserving every
        # remaining token for verification before admitting that prerequisite
        # made a real finance-14 pass deadlock despite five search results.
        # The allowance is estimated from the exact post-search prompt; it is
        # capacity accounting, not a judgement about which source to read.
        verification_reserve = _reserve_verification_budget(
            tracker=tracker,
            queries=queries,
            hints=hints,
            targets=targets,
            notes=ledger.notes,
            source_cache=ledger.source_cache,
            max_reads=budget.max_reads,
            verification_model=verification_model,
            verification_settings=active_verification_settings,
            truth_condition_registry=truth_condition_registry,
            prerequisite_stage=(
                "read_selection" if read_prompt is not None else None
            ),
            prerequisite_estimated_tokens=prerequisite_tokens,
            prerequisite_estimated_cost_usd=prerequisite_cost,
            minimum_action_estimated_tokens=(
                selected_web_allowance[4]
                if selected_web_allowance is not None
                else 0
            ),
            minimum_action_estimated_cost_usd=(
                selected_web_allowance[5]
                if selected_web_allowance is not None
                else 0.0
            ),
        )
        verification_reserve_history.append(verification_reserve)
        ledger.record_evidence_gap(
            event=ledger_event("cache_review"),
            result_summary=json.dumps(
                {
                    "target_claim_ids": [target.claim.claim_id for target in targets],
                    "accepted_cached_candidates": len(hints),
                    "search_queries": len(queries),
                    "deferred_target_claim_ids": [
                        target.claim_id for target in deferred_targets
                    ],
                    "plan_attempt_count": plan_attempt_count,
                    "selected_planning_attempt": selected_planning_attempt,
                    "unused_query_slots": unused_query_slots,
                    "rejected_entries": len(rejected),
                    "verification_reserved_tokens": (
                        verification_reserve.reserved_tokens
                    ),
                    "verification_reserved_cost_usd": (
                        verification_reserve.reserved_cost_usd
                    ),
                    "prerequisite_stage": (
                        verification_reserve.prerequisite_stage
                    ),
                    "prerequisite_estimated_tokens": (
                        verification_reserve.prerequisite_estimated_tokens
                    ),
                    "prerequisite_estimated_cost_usd": (
                        verification_reserve.prerequisite_estimated_cost_usd
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )

        if (
            read_prompt is not None
            and not verification_reserve.reserve_fully_funded
        ):
            stop_reason = EvidenceGapStopReason.BUDGET_EXHAUSTED
            stop_detail = (
                "read selection not admitted because the actual planning "
                "usage and search-result prompt cannot preserve accepted "
                "cached checks plus one bounded note-attribution-verification "
                "action"
            )
            rejected.append(
                {
                    "stage": "read_selection",
                    "error": stop_detail,
                }
            )
            read_prompt = None

        if read_prompt is not None:
            try:
                read_response = await tracker.call(
                    gap_model,
                    read_prompt,
                    stage="read_selection",
                )
            except _GapBudgetExhausted as exc:
                stop_reason = EvidenceGapStopReason.BUDGET_EXHAUSTED
                stop_detail = str(exc)
                rejected.append(
                    {
                        "stage": "read_selection",
                        "error": str(exc),
                    }
                )
            else:
                selections, read_rejected = _parse_reads(
                    _decode_response(read_response),
                    targets=targets,
                    searches=searches,
                    cached_hints=hints,
                    checklist=checklist,
                    max_reads=budget.max_reads,
                )
                rejected.extend(read_rejected)

        target_claim_by_id = {
            target.claim.claim_id: target.claim for target in targets
        }
        # Only sources that actually produced notes remain in this mapping.
        # Before each note call, the current newly read source is added
        # provisionally so its own downstream verification can be protected.
        # A zero-note, model-error, or budget-rejected source is omitted on
        # the next iteration and therefore cannot keep starving smaller URLs.
        admitted_read_source_claims: dict[str, tuple[AtomicClaim, ...]] = {}
        for selection in selections:
            existing = ledger.get_source(selection.url)
            cache_hit = existing is not None
            try:
                source_read = (
                    None
                    if existing is not None
                    else await read_with_links(
                        selection.url,
                        tavily_client=tavily_client,
                        timeout_seconds=budget.provider_timeout_seconds,
                    )
                )
                source_text = (
                    existing
                    if existing is not None
                    else source_read.cleaned_text
                )
                if not cache_hit:
                    ledger.cache_source(
                        selection.url,
                        source_text,
                        source_links=source_read.source_links,
                        link_capture=source_read.link_capture,
                    )
            except Exception as exc:  # noqa: BLE001 - one source must not lose the pass
                error = f"{type(exc).__name__}: {exc}"
                acquisitions.append(
                    GapSourceAcquisition(
                        url=selection.url,
                        claim_ids=selection.claim_ids,
                        target_element_ids=selection.target_element_ids,
                        publisher_identity=selection.publisher_identity,
                        cache_hit=cache_hit,
                        outcome="read_error",
                        error=error,
                    )
                )
                ledger.record_evidence_gap(
                    event=ledger_event("source_read_error"),
                    url=selection.url,
                    result_summary=error,
                )
                continue
            if not cache_hit:
                added_source_urls.append(selection.url)

            selection_claims = tuple(
                target_claim_by_id[claim_id]
                for claim_id in selection.claim_ids
            )
            prospective_sources = dict(admitted_read_source_claims)
            existing_group = {
                claim.claim_id: claim
                for claim in prospective_sources.get(selection.url, ())
            }
            existing_group.update(
                {claim.claim_id: claim for claim in selection_claims}
            )
            prospective_sources[selection.url] = tuple(
                existing_group.values()
            )
            existing_added_note_ids = set(added_note_ids)
            prospective_incremental_notes = tuple(
                note
                for note in ledger.notes
                if note.note_id in existing_added_note_ids
            ) + tuple(
                _prospective_gap_note(
                    item_id=selection.item_id,
                    url=selection.url,
                    ordinal=len(added_note_ids) + offset,
                )
                for offset in (1, 2)
            )
            prospective_routed_ids = tuple(
                dict.fromkeys(
                    (*incremental_routed_claim_ids, *selection.claim_ids)
                )
            )
            prospective_routed_claims = tuple(
                target_claim_by_id[claim_id]
                for claim_id in prospective_routed_ids
            )
            (
                prospective_reattribution_tokens,
                prospective_reattribution_cost,
            ) = _estimate_incremental_reattribution_allowance(
                claims=prospective_routed_claims,
                notes=prospective_incremental_notes,
                tracker=tracker,
                attribution_model=attribution_model,
                attribution_settings=active_attribution_settings,
                output_headroom_ratio=budget.planning_output_headroom_ratio,
            )
            note_prompt = build_evidence_gap_note_prompt(
                url=selection.url,
                source_text=source_text,
                claims=selection_claims,
                checklist=checklist,
                target_element_ids=selection.target_element_ids,
                truth_condition_registry=truth_condition_registry,
            )
            note_prerequisite_tokens, note_prerequisite_cost = tracker._estimate(
                note_model,
                note_prompt,
            )
            note_prerequisite_tokens += math.ceil(
                note_prerequisite_tokens
                * budget.planning_output_headroom_ratio
            )
            note_prerequisite_cost *= (
                1.0 + budget.planning_output_headroom_ratio
            )
            verification_reserve = _reserve_verification_budget(
                tracker=tracker,
                queries=queries,
                hints=hints,
                targets=targets,
                notes=ledger.notes,
                source_cache=ledger.source_cache,
                max_reads=budget.max_reads,
                verification_model=verification_model,
                verification_settings=active_verification_settings,
                truth_condition_registry=truth_condition_registry,
                admitted_read_source_claims=prospective_sources,
                prerequisite_stage="note_extraction",
                prerequisite_estimated_tokens=note_prerequisite_tokens,
                prerequisite_estimated_cost_usd=note_prerequisite_cost,
                incremental_reattribution_estimated_tokens=(
                    prospective_reattribution_tokens
                ),
                incremental_reattribution_estimated_cost_usd=(
                    prospective_reattribution_cost
                ),
            )
            verification_reserve_history.append(verification_reserve)

            if not verification_reserve.reserve_fully_funded:
                stop_reason = EvidenceGapStopReason.BUDGET_EXHAUSTED
                stop_detail = (
                    "actual source not admitted because note extraction "
                    "cannot preserve incremental reattribution and "
                    "verification capacity"
                )
                acquisitions.append(
                    GapSourceAcquisition(
                        url=selection.url,
                        claim_ids=selection.claim_ids,
                        target_element_ids=selection.target_element_ids,
                        publisher_identity=selection.publisher_identity,
                        cache_hit=cache_hit,
                        source_chars=len(source_text),
                        outcome=(
                            "note_extraction_not_run_budget_after_actual_read"
                        ),
                        error=stop_detail,
                    )
                )
                rejected.append(
                    {
                        "stage": "note_extraction",
                        "url": selection.url,
                        "source_chars": len(source_text),
                        "error": stop_detail,
                    }
                )
                continue

            try:
                note_response = await tracker.call(
                    note_model,
                    note_prompt,
                    stage="note_extraction",
                )
                note_content = _decode_response(note_response)
            except _GapBudgetExhausted as exc:
                stop_reason = EvidenceGapStopReason.BUDGET_EXHAUSTED
                stop_detail = str(exc)
                actual_read_error = (
                    "actual source could not be admitted for note extraction; "
                    f"source_chars={len(source_text)}; {exc}"
                )
                acquisitions.append(
                    GapSourceAcquisition(
                        url=selection.url,
                        claim_ids=selection.claim_ids,
                        target_element_ids=selection.target_element_ids,
                        publisher_identity=selection.publisher_identity,
                        cache_hit=cache_hit,
                        source_chars=len(source_text),
                        outcome=(
                            "note_extraction_not_run_budget_after_actual_read"
                        ),
                        error=actual_read_error,
                    )
                )
                rejected.append(
                    {
                        "stage": "note_extraction",
                        "url": selection.url,
                        "source_chars": len(source_text),
                        "error": actual_read_error,
                    }
                )
                # This source has no admitted note work, so its provisional
                # verification reserve is intentionally released before the
                # next selected URL.  Do not let one large page turn a
                # bounded pass into a silent all-or-nothing stop.
                continue
            except RunCostCapReached:
                # The shared run controller, rather than this local source
                # loop, owns absolute-cost termination and partial publishing.
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                acquisitions.append(
                    GapSourceAcquisition(
                        url=selection.url,
                        claim_ids=selection.claim_ids,
                        target_element_ids=selection.target_element_ids,
                        publisher_identity=selection.publisher_identity,
                        cache_hit=cache_hit,
                        source_chars=len(source_text),
                        outcome="note_model_error",
                        error=error,
                    )
                )
                ledger.record_evidence_gap(
                    event=ledger_event("note_model_error"),
                    url=selection.url,
                    result_summary=error,
                )
                continue
            raw_notes = (
                note_content.get("notes")
                if isinstance(note_content, Mapping)
                else None
            )
            if not isinstance(raw_notes, (list, tuple)):
                raw_notes = ()
                rejected.append(
                    {
                        "stage": "note_extraction",
                        "url": selection.url,
                        "error": "notes must be an array",
                    }
                )
            elif len(raw_notes) > 2:
                rejected.append(
                    {
                        "stage": "note_extraction",
                        "url": selection.url,
                        "error": (
                            "note protocol permits at most two notes; "
                            f"rejected {len(raw_notes) - 2} overflow entries"
                        ),
                    }
                )
                raw_notes = raw_notes[:2]
            created_ids: list[str] = []
            allowed_items = set(checklist.in_scope_item_ids)
            for index, raw_note in enumerate(raw_notes):
                try:
                    draft = _GapNoteDraft.model_validate(raw_note)
                    if draft.item_id not in allowed_items:
                        raise ValueError("unknown checklist item_id")
                    note = ledger.add_note(
                        create_note(
                            item_id=draft.item_id,
                            finding=draft.finding,
                            quote=draft.quote,
                            url=selection.url,
                            source_text=source_text,
                        )
                    )
                except (TypeError, ValidationError, ValueError) as exc:
                    rejected.append(
                        {
                            "stage": "note_extraction",
                            "url": selection.url,
                            "index": index,
                            "error": str(exc),
                            "raw": raw_note,
                        }
                    )
                    continue
                if note.note_id is None:
                    raise AssertionError("ledger must assign a note_id")
                created_ids.append(note.note_id)
                added_note_ids.append(note.note_id)
            acquisitions.append(
                GapSourceAcquisition(
                    url=selection.url,
                    claim_ids=selection.claim_ids,
                    target_element_ids=selection.target_element_ids,
                    publisher_identity=selection.publisher_identity,
                    cache_hit=cache_hit,
                    source_chars=len(source_text),
                    note_ids=tuple(created_ids),
                    outcome="notes_created" if created_ids else "zero_notes",
                )
            )
            if created_ids:
                admitted_read_source_claims[selection.url] = tuple(
                    prospective_sources[selection.url]
                )
                for claim_id in selection.claim_ids:
                    if claim_id not in incremental_routed_claim_ids:
                        incremental_routed_claim_ids.append(claim_id)
            ledger.record_evidence_gap(
                event=ledger_event("source_acquired"),
                url=selection.url,
                note_ids=tuple(created_ids),
                result_summary=json.dumps(
                    {
                        "cache_hit": cache_hit,
                        "source_chars": len(source_text),
                        "outcome": (
                            "notes_created" if created_ids else "zero_notes"
                        ),
                    },
                    sort_keys=True,
                ),
            )

        # Release any reserve held for a final failed/zero-note candidate and
        # retain only sources that actually created note input for the tail.
        verification_reserve = _reserve_verification_budget(
            tracker=tracker,
            queries=queries,
            hints=hints,
            targets=targets,
            notes=ledger.notes,
            source_cache=ledger.source_cache,
            max_reads=budget.max_reads,
            verification_model=verification_model,
            verification_settings=active_verification_settings,
            truth_condition_registry=truth_condition_registry,
            admitted_read_source_claims=admitted_read_source_claims,
        )
        verification_reserve_history.append(verification_reserve)

        target_claims = [target.claim for target in targets]
        if added_note_ids:
            added_note_id_set = set(added_note_ids)
            incremental_notes = tuple(
                note
                for note in ledger.notes
                if note.note_id in added_note_id_set
            )
            target_claim_by_id = {
                claim.claim_id: claim for claim in target_claims
            }
            incremental_claims = tuple(
                target_claim_by_id[claim_id]
                for claim_id in incremental_routed_claim_ids
            )
            (
                exact_reattribution_tokens,
                exact_reattribution_cost,
            ) = _estimate_incremental_reattribution_allowance(
                claims=incremental_claims,
                notes=incremental_notes,
                tracker=tracker,
                attribution_model=attribution_model,
                attribution_settings=active_attribution_settings,
                output_headroom_ratio=budget.planning_output_headroom_ratio,
            )
            combined_tail_reserve = _reserve_verification_budget(
                tracker=tracker,
                queries=queries,
                hints=hints,
                targets=targets,
                notes=ledger.notes,
                source_cache=ledger.source_cache,
                max_reads=budget.max_reads,
                verification_model=verification_model,
                verification_settings=active_verification_settings,
                truth_condition_registry=truth_condition_registry,
                admitted_read_source_claims=admitted_read_source_claims,
                incremental_reattribution_estimated_tokens=(
                    exact_reattribution_tokens
                ),
                incremental_reattribution_estimated_cost_usd=(
                    exact_reattribution_cost
                ),
            )
            verification_reserve_history.append(combined_tail_reserve)
            if not combined_tail_reserve.reserve_fully_funded:
                stop_reason = EvidenceGapStopReason.BUDGET_EXHAUSTED
                stop_detail = (
                    "new notes preserved, but exact incremental "
                    "reattribution plus verification exceeds the remaining "
                    "gap budget"
                )
                rejected.append(
                    {
                        "stage": "reattribution",
                        "error": stop_detail,
                    }
                )
                refreshed = AttributionResult(
                    attributions=(),
                    stop_reason=AttributionStopReason.COMPLETED,
                )
            else:
                # The combined audit proves both tail actions fit. Release
                # only the attribution allowance so the tracked attribution
                # call can consume it while exact verification stays held.
                verification_reserve = _reserve_verification_budget(
                    tracker=tracker,
                    queries=queries,
                    hints=hints,
                    targets=targets,
                    notes=ledger.notes,
                    source_cache=ledger.source_cache,
                    max_reads=budget.max_reads,
                    verification_model=verification_model,
                    verification_settings=active_verification_settings,
                    truth_condition_registry=truth_condition_registry,
                    admitted_read_source_claims=admitted_read_source_claims,
                )
                verification_reserve_history.append(verification_reserve)
                try:
                    refreshed = await attribute_claims(
                        incremental_claims,
                        blocks=blocks,
                        # Initial attribution is merged below and must not be
                        # paid for a second time. This incremental pass routes
                        # only notes created by this evidence-gap pass.
                        notes=incremental_notes,
                        model_client=_TrackedClient(
                            attribution_model,
                            tracker,
                            "reattribution",
                        ),
                        settings=active_attribution_settings,
                    )
                except _GapBudgetExhausted as exc:
                    stop_reason = EvidenceGapStopReason.BUDGET_EXHAUSTED
                    stop_detail = str(exc)
                    rejected.append(
                        {
                            "stage": "reattribution",
                            "error": str(exc),
                        }
                    )
                    refreshed = AttributionResult(
                        attributions=(),
                        stop_reason=AttributionStopReason.COMPLETED,
                    )
        else:
            # The plan's mechanically validated hints already represent every
            # cache-only addition. Re-running attribution without new notes
            # would spend the verification reserve without adding new input.
            refreshed = AttributionResult(
                attributions=(),
                stop_reason=AttributionStopReason.COMPLETED,
            )
        merged_attribution = _merge_attributions(
            initial_attribution,
            refreshed,
            hint_candidates=_hint_candidates(hints, notes=ledger.notes),
        )
        merged_by_id = {
            entry.claim.claim_id: entry
            for entry in merged_attribution.attributions
        }
        target_attributions = [
            merged_by_id[target.claim.claim_id]
            for target in targets
        ]
        incremental_attributions = _incremental_attributions(
            target_attributions,
            initial_verification=initial_verification,
        )
        remaining_token_budget = max(
            0, budget.max_tokens - tracker.tokens_used
        )
        remaining_cost_budget = max(
            0.0, budget.max_cost_usd - tracker.cost_used
        )

        def verification_token_estimator(prompt: str) -> int:
            return tracker._estimate(verification_model, prompt)[0]

        def verification_cost_estimator(prompt: str) -> float:
            return tracker._estimate(verification_model, prompt)[1]

        if incremental_attributions:
            # The verifier binds the truth-condition registry to the exact
            # attribution denominator *and order*.  Preserve that sequence:
            # a set can reorder an explicit bounded subset and make a valid
            # incremental pass fail closed at the verifier boundary.
            incremental_claim_ids = tuple(
                attribution.claim.claim_id
                for attribution in incremental_attributions
            )
            refreshed_verification = await verify_attributions(
                incremental_attributions,
                source_cache=ledger.source_cache,
                model_client=_TrackedClient(
                    verification_model,
                    tracker,
                    "reverification",
                    allow_verification_reserve=True,
                ),
                settings=active_verification_settings,
                budget=VerificationBudget(
                    max_tokens=remaining_token_budget,
                    max_cost_usd=remaining_cost_budget,
                ),
                corroboration_targets={
                    claim_id: count
                    for claim_id, count in (
                        corroboration_targets
                        if corroboration_targets is not None
                        else (required_independent_sources or {})
                    ).items()
                    if claim_id in incremental_claim_ids
                },
                estimate_input_tokens=verification_token_estimator,
                estimate_cost_usd=verification_cost_estimator,
                registry=(
                    select_truth_condition_registry(
                        truth_condition_registry,
                        incremental_claim_ids,
                    )
                    if truth_condition_registry is not None
                    else None
                ),
            )
        else:
            refreshed_verification = VerificationResult(
                claims=(),
                truth_condition_registry_sha256=(
                    truth_condition_registry_sha256(
                        select_truth_condition_registry(
                            truth_condition_registry,
                            (),
                        )
                    )
                    if truth_condition_registry is not None
                    else None
                ),
            )
        if any(
            relation.status
            is VerificationRecordStatus.VERIFICATION_NOT_RUN_BUDGET
            for claim_result in refreshed_verification.claims
            for relation in claim_result.relations
        ):
            stop_reason = EvidenceGapStopReason.BUDGET_EXHAUSTED
            stop_detail = (
                "verification admission exceeded the remaining reserved "
                "gap budget"
            )
        final_verification, verification_merge = _merge_verifications(
            initial_verification,
            refreshed_verification,
            merged_attribution=merged_attribution,
            truth_condition_registry=truth_condition_registry,
        )
        if (
            tracker.tokens_used >= budget.max_tokens
            or tracker.cost_used >= budget.max_cost_usd
        ):
            stop_reason = EvidenceGapStopReason.BUDGET_EXHAUSTED
            stop_detail = (
                "gap budget reached after the final admitted model call"
            )
    except RunCostCapReached:
        # The absolute run-level controller owns this boundary.  Converting it
        # into a local model error hides the binding cap from the runner and
        # defeats its partial-artifact path.
        raise
    except _GapBudgetExhausted as exc:
        stop_reason = EvidenceGapStopReason.BUDGET_EXHAUSTED
        stop_detail = str(exc)
        merged_attribution = initial_attribution
        final_verification = initial_verification
    except Exception as exc:
        stop_reason = EvidenceGapStopReason.MODEL_ERROR
        stop_detail = f"{type(exc).__name__}: {exc}"
        merged_attribution = initial_attribution
        final_verification = initial_verification

    current_claims = {
        entry.claim.claim_id: entry.claim.model_dump(mode="json")
        for entry in merged_attribution.attributions
    }
    claims_unchanged = current_claims == frozen_claims
    if not claims_unchanged:
        raise AssertionError("evidence-gap round mutated the frozen claim registry")
    if canonical_draft != frozen_draft:
        raise AssertionError("evidence-gap round mutated the canonical draft")

    information_yield = _information_yield(
        initial_verification,
        final_verification,
        stop_reason=stop_reason,
    )
    if stop_reason is EvidenceGapStopReason.COMPLETED:
        routed_claim_ids = {hint.claim_id for hint in hints} | {
            claim_id
            for search_record in searches
            if search_record.error is None
            for claim_id in search_record.query.claim_ids
        }
        routed_count = sum(
            target.claim.claim_id in routed_claim_ids for target in targets
        )
        stop_detail = (
            "single bounded evidence-gap pass ended; "
            f"routed target claims={routed_count}/{len(targets)}; "
            f"unrouted target claims={len(targets) - routed_count}; "
            f"code-derived unrouted records={len(deferred_targets)}; "
            f"issued query slots={len(queries)}/{effective_max_queries}; "
            "new completed claim-source relations="
            f"{information_yield.new_completed_relation_count}"
        )
    ledger.record_evidence_gap(
        event=ledger_event("stop"),
        result_summary=json.dumps(
            {
                "reason": stop_reason.value,
                "detail": stop_detail,
                "added_sources": len(added_source_urls),
                "added_notes": len(added_note_ids),
                "initial_completed_relations": (
                    verification_merge.initial_completed_relation_count
                ),
                "final_completed_relations": (
                    verification_merge.final_completed_relation_count
                ),
                "new_completed_relations": (
                    information_yield.new_completed_relation_count
                ),
                "new_completed_verdict_counts": (
                    information_yield.new_completed_verdict_counts
                ),
                "new_publisher_domain_proxies": list(
                    information_yield.new_publisher_domain_proxies
                ),
                "new_claim_publisher_relations": (
                    information_yield.new_claim_publisher_relation_count
                ),
                "claims_newly_corroborated": (
                    information_yield.claims_newly_corroborated
                ),
                "claims_newly_conflicting": (
                    information_yield.claims_newly_conflicting
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    return EvidenceGapResult(
        target_claim_ids=tuple(
            target.claim.claim_id for target in targets
        ),
        initial_states=initial_states,
        cached_candidate_hints=hints,
        deferred_targets=deferred_targets,
        planning_attempt_count=plan_attempt_count,
        selected_planning_attempt=selected_planning_attempt,
        unused_query_slots=unused_query_slots,
        rejected_entries=tuple(rejected),
        searches=tuple(searches),
        read_selections=selections,
        acquisitions=tuple(acquisitions),
        added_source_urls=tuple(added_source_urls),
        added_note_ids=tuple(added_note_ids),
        verification_reserve=verification_reserve,
        verification_reserve_history=tuple(verification_reserve_history),
        planning_capacity=planning_capacity,
        information_yield=information_yield,
        usage=tuple(tracker.usage),
        stop_reason=stop_reason,
        stop_detail=stop_detail,
        claim_registry_unchanged=claims_unchanged,
        canonical_draft_unchanged=(canonical_draft == frozen_draft),
        verification_merge=verification_merge,
        final_attribution=merged_attribution,
        final_verification=final_verification,
    )
