"""One bounded post-verification pass for unresolved evidence gaps."""

from __future__ import annotations

import inspect
import json
import unicodedata
from collections.abc import Awaitable, Callable, Mapping, Sequence
from enum import Enum
from typing import Any, Protocol
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
)
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
)
from open_deep_research.harness.tools import (
    SearchResult,
    SourceReadError,
    TavilyClient,
    read_with_links,
    search,
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
    VerifiedSourceRelation,
    build_claim_verification,
    build_verification_prompt,
    verify_attributions,
)

_TARGET_STATES = {
    ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER,
    ClaimEvidenceState.CONFLICTING_EVIDENCE,
}


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
    max_search_queries: int = Field(default=3, ge=0, le=20)
    max_reads: int = Field(default=3, ge=0, le=20)
    max_results_per_search: int = Field(default=5, ge=1, le=20)


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
    item_id: str
    query: str


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
    publisher_identity: str
    independence_rationale: str


class GapSourceAcquisition(BaseModel):
    """One selected URL and its durable source/note outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str
    claim_ids: tuple[str, ...]
    publisher_identity: str
    cache_hit: bool
    source_chars: int = Field(default=0, ge=0)
    note_ids: tuple[str, ...] = ()
    outcome: str
    error: str | None = None


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
    """Pre-search admission reserve for later full-source verification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str = "planned_claim_groups_against_largest_cached_source"
    reference_source_url: str | None = None
    reference_source_chars: int = Field(default=0, ge=0)
    cached_hint_batch_count: int = Field(default=0, ge=0)
    web_read_slots: int = Field(default=0, ge=0)
    planned_query_count: int = Field(default=0, ge=0)
    planned_query_claim_count: int = Field(default=0, ge=0)
    estimated_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    reserved_tokens: int = Field(default=0, ge=0)
    reserved_cost_usd: float = Field(default=0.0, ge=0.0)
    limitations: tuple[str, ...] = (
        "future source length is unknown before network reads",
        "web reserve uses the largest source already present in cache",
        "later reattribution can create relations outside planned query groups",
        "admission estimates do not predict model output tokens exactly",
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
    rejected_entries: tuple[dict[str, Any], ...] = ()
    searches: tuple[GapSearchRecord, ...] = ()
    read_selections: tuple[GapReadSelection, ...] = ()
    acquisitions: tuple[GapSourceAcquisition, ...] = ()
    added_source_urls: tuple[str, ...] = ()
    added_note_ids: tuple[str, ...] = ()
    verification_reserve: VerificationReserveAudit | None = None
    information_yield: EvidenceGapInformationAudit = Field(
        default_factory=EvidenceGapInformationAudit
    )
    usage: tuple[EvidenceGapCallUsage, ...] = ()
    stop_reason: EvidenceGapStopReason
    stop_detail: str
    claim_registry_unchanged: bool = True
    canonical_draft_unchanged: bool = True
    independence_method: str = "model_screen_then_publisher_domain_proxy"
    independence_is_strict: bool = False
    independence_limitations: tuple[str, ...] = (
        "model_screening_can_misidentify_common_ownership",
        "syndicated_or_republished_content_can_be_missed",
        "final_counts_still_use_publisher_domain_proxy",
    )
    query_planning_method: str = "model_selected_ordered_merged_queries"
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
        or an issued search record.  Merely belonging to the requested target
        set is not work done.  Keeping this derivation inside the result model
        also makes hand-built offline/recovery results obey the same audit
        contract as the live executor.
        """

        if not isinstance(value, Mapping):
            return value
        data = dict(value)
        if (
            "routed_target_claim_ids" in data
            or "unrouted_target_claim_ids" in data
        ):
            return data
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

You have a hard budget of at most {max_queries} web search queries. This is an
upper bound, not a target: returning fewer or zero queries is legal. Select
and order only the highest-value queries. Put the highest-priority query
first. One query may name several claim_ids when the same focused search can
plausibly find pages relevant to every named claim. Reusing one query for
several claims is encouraged when their information need genuinely overlaps,
but do not make a query vague merely to cover more claims. A target claim
does not need its own output entry.

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
Extract zero or more research notes from this one complete source. Return json
only:
{{"notes":[{{"item_id":"existing-checklist-item",\
"finding":"what the source says",\
"quote":"one exact continuous source passage"}}]}}

Prioritize the frozen target claims, but retain useful findings for other
listed checklist items too. One quote is one continuous verbatim passage.
Copy it exactly: do not paraphrase, join separated passages, use ellipses,
reorder words, or change punctuation. If two separate passages are needed,
return two notes. Returning zero notes is legal and is not an error.

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
        if token_estimator is not None:
            tokens = max(0, int(token_estimator(client, prompt)))
        else:
            method = getattr(client, "estimate_tokens", None)
            if not callable(method):
                raise _GapBudgetExhausted(
                    "gap token admission estimator is unavailable"
                )
            tokens = max(0, int(method(prompt)))
        if cost_estimator is not None:
            cost = max(0.0, float(cost_estimator(client, prompt)))
        else:
            method = getattr(client, "estimate_cost_usd", None)
            if not callable(method):
                raise _GapBudgetExhausted(
                    "gap cost admission estimator is unavailable"
                )
            cost = max(0.0, float(method(prompt)))
        return tokens, cost

    def reserve_verification(
        self,
        *,
        tokens: int,
        cost_usd: float,
    ) -> tuple[int, float]:
        """Protect an estimated verification envelope from earlier stages."""

        remaining_tokens = max(0, self.budget.max_tokens - self.tokens_used)
        remaining_cost = max(0.0, self.budget.max_cost_usd - self.cost_used)
        self.verification_reserved_tokens = min(
            max(0, int(tokens)),
            remaining_tokens,
        )
        self.verification_reserved_cost_usd = min(
            max(0.0, float(cost_usd)),
            remaining_cost,
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
) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": target.claim.claim_id,
            "claim_text": target.claim.claim_text,
            "state": target.state.value,
            "corroboration_target": target.corroboration_target,
            "formal_publisher_domain_proxies": list(
                target.publisher_domain_proxies
            ),
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
        for target in targets
    ]


def build_evidence_gap_plan_prompt(
    *,
    targets: Sequence[ClaimVerification],
    notes: Sequence[ResearchNote],
    checklist: ResearchChecklist,
    max_queries: int,
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
            "unused_by_target_claim_ids": [
                target.claim.claim_id
                for target in targets
                if (target.claim.claim_id, note.note_id) not in used_pairs
            ],
        }
        for note in notes
    ]
    return _PLAN_PROMPT.format(
        max_queries=max_queries,
        item_ids=json.dumps(
            [item.item_id for item in checklist.items],
            ensure_ascii=False,
        ),
        corroboration_targets=json.dumps(
            [
                {
                    "item_id": item.item_id,
                    "corroboration_target": item.corroboration_target,
                }
                for item in checklist.items
            ],
            ensure_ascii=False,
            sort_keys=True,
        ),
        targets=json.dumps(
            _target_payload(targets),
            ensure_ascii=False,
            sort_keys=True,
        ),
        notes=json.dumps(note_registry, ensure_ascii=False, sort_keys=True),
    )


def build_evidence_gap_read_prompt(
    *,
    targets: Sequence[ClaimVerification],
    searches: Sequence[GapSearchRecord],
    max_reads: int,
) -> str:
    """Ask the model to choose URLs after seeing bounded search results."""

    candidates = [
        {
            "query_claim_ids": list(record.query.claim_ids),
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
            _target_payload(targets),
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
) -> str:
    """Build one full-source note pass without changing checklist state."""

    return _NOTE_PROMPT.format(
        item_ids=json.dumps(
            [item.item_id for item in checklist.items],
            ensure_ascii=False,
        ),
        claims=json.dumps(
            [
                {"claim_id": claim.claim_id, "claim_text": claim.claim_text}
                for claim in claims
            ],
            ensure_ascii=False,
            sort_keys=True,
        ),
        url=url,
        source_text=source_text,
    )


def _estimate_verification_group(
    *,
    claims: Sequence[AtomicClaim],
    url: str,
    source_text: str,
    batch_size: int,
    tracker: _BudgetTracker,
    verification_model: VerificationModelClient,
) -> tuple[int, int, float]:
    batch_count = 0
    tokens = 0
    cost = 0.0
    for start in range(0, len(claims), batch_size):
        prompt = build_verification_prompt(
            url=url,
            source_text=source_text,
            claims=claims[start : start + batch_size],
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
) -> VerificationReserveAudit:
    """Reserve a pre-search estimate without pretending future sizes are known."""

    claim_by_id = {target.claim.claim_id: target.claim for target in targets}
    note_by_id = {str(note.note_id): note for note in notes}
    hint_claims_by_url: dict[str, dict[str, AtomicClaim]] = {}
    for hint in hints:
        note = note_by_id[hint.note_id]
        hint_claims_by_url.setdefault(note.url, {})[hint.claim_id] = (
            claim_by_id[hint.claim_id]
        )

    cached_batch_count = 0
    estimated_tokens = 0
    estimated_cost = 0.0
    for url in sorted(hint_claims_by_url):
        source_text = source_cache.get(url)
        if source_text is None:
            continue
        claims = tuple(hint_claims_by_url[url].values())
        batches, tokens, cost = _estimate_verification_group(
            claims=claims,
            url=url,
            source_text=source_text,
            batch_size=verification_settings.batch_size,
            tracker=tracker,
            verification_model=verification_model,
        )
        cached_batch_count += batches
        estimated_tokens += tokens
        estimated_cost += cost

    reference_url: str | None = None
    reference_text = ""
    if source_cache:
        reference_url, reference_text = max(
            sorted(source_cache.items()),
            key=lambda item: len(item[1]),
        )

    web_read_slots = max_reads if queries else 0
    if reference_url is not None and web_read_slots:
        per_query_estimates: list[tuple[int, float]] = []
        for query in queries:
            claims = tuple(claim_by_id[claim_id] for claim_id in query.claim_ids)
            _, tokens, cost = _estimate_verification_group(
                claims=claims,
                url=reference_url,
                source_text=reference_text,
                batch_size=verification_settings.batch_size,
                tracker=tracker,
                verification_model=verification_model,
            )
            per_query_estimates.append((tokens, cost))
        if per_query_estimates:
            largest_tokens = max(tokens for tokens, _ in per_query_estimates)
            largest_cost = max(cost for _, cost in per_query_estimates)
            estimated_tokens += largest_tokens * web_read_slots
            estimated_cost += largest_cost * web_read_slots

    reserved_tokens, reserved_cost = tracker.reserve_verification(
        tokens=estimated_tokens,
        cost_usd=estimated_cost,
    )
    return VerificationReserveAudit(
        reference_source_url=reference_url,
        reference_source_chars=len(reference_text),
        cached_hint_batch_count=cached_batch_count,
        web_read_slots=web_read_slots,
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
        reserved_tokens=reserved_tokens,
        reserved_cost_usd=reserved_cost,
    )


def _parse_plan(
    content: Any,
    *,
    targets: Sequence[ClaimVerification],
    notes: Sequence[ResearchNote],
    checklist: ResearchChecklist,
    max_queries: int,
) -> tuple[
    tuple[CachedCandidateHint, ...],
    tuple[GapSearchQuery, ...],
    tuple[dict[str, Any], ...],
]:
    target_by_id = {target.claim.claim_id: target for target in targets}
    note_by_id = {str(note.note_id): note for note in notes}
    item_ids = {item.item_id for item in checklist.items}
    rejected: list[dict[str, Any]] = []
    accepted_hints: list[CachedCandidateHint] = []
    accepted_queries: list[GapSearchQuery] = []
    if not isinstance(content, Mapping):
        return (), (), (
            {
                "stage": "plan",
                "error": "plan must be a JSON object",
                "raw": content,
            },
        )
    raw_hints = content.get("cached_candidates")
    raw_queries = content.get("queries")
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
        existing_publishers = set(target.publisher_domain_proxies)
        note = note_by_id.get(hint.note_id)
        identity = hint.publisher_identity.strip().casefold()
        error: str | None = None
        if note is None:
            error = "unknown note_id"
        elif note.source_id != hint.source_id:
            error = "note_id/source_id mismatch"
        elif not hint.independent_from_existing_publishers:
            error = "model did not judge publisher independent"
        elif _publisher_proxy(note.url, note.publisher) in existing_publishers:
            error = "publisher domain proxy already supports this claim"
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
                item_id=query.item_id,
                query=query.query,
            )
        )
    return (
        tuple(accepted_hints),
        tuple(accepted_queries),
        tuple(rejected),
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
    item_ids = {item.item_id for item in checklist.items}
    allowed: dict[str, set[tuple[str, str]]] = {}
    for record in searches:
        for result in record.results:
            allowed.setdefault(result.url, set()).update(
                (claim_id, record.query.item_id)
                for claim_id in record.query.claim_ids
            )
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
            elif publisher_proxy in set(
                target_by_id[claim_id].publisher_domain_proxies
            ):
                claim_error = "publisher domain proxy already supports this claim"
            elif any(
                _identity_matches_proxy(proposal.publisher_identity, proxy)
                for proxy in target_by_id[claim_id].publisher_domain_proxies
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
) -> tuple[VerificationResult, VerificationMergeAudit]:
    """Add source relations without allowing failed refreshes to erase work."""

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
        claims.append(
            build_claim_verification(
                original.claim,
                tuple(relation_by_identity.values()),
                required_sources=original.corroboration_target,
                attribution_status=attribution.status,
            )
        )

    merged = VerificationResult(
        claims=tuple(claims),
        usage=initial.usage + refreshed_targets.usage,
        diagnostics=initial.diagnostics + refreshed_targets.diagnostics,
        independence=initial.independence,
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
    corroboration_targets: Mapping[str, int] | None = None,
    required_independent_sources: Mapping[str, int] | None = None,
    estimate_input_tokens: Callable[[Any, str], int] | None = None,
    estimate_cost_usd: Callable[[Any, str], float] | None = None,
    explicit_target_claim_ids: Sequence[str] | None = None,
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

    if explicit_target_claim_ids is None:
        targets = tuple(
            result
            for result in initial_verification.claims
            if (
                result.claim.citation_requirement
                == CitationRequirement.EXTERNAL
                and result.state in _TARGET_STATES
                and (
                    result.state
                    is not ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER
                    or result.publisher_domain_proxy_count
                    < result.corroboration_target
                )
            )
        )
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
    hints: tuple[CachedCandidateHint, ...] = ()
    verification_reserve: VerificationReserveAudit | None = None
    stop_reason = EvidenceGapStopReason.COMPLETED
    stop_detail = "single evidence-gap pass completed"
    verification_merge = _unchanged_merge_audit(initial_verification)
    active_verification_settings = (
        verification_settings or VerificationSettings()
    )

    try:
        active_plan_builder = (
            plan_prompt_builder or build_evidence_gap_plan_prompt
        )
        plan_response = await tracker.call(
            gap_model,
            active_plan_builder(
                targets=targets,
                notes=ledger.notes,
                checklist=checklist,
                max_queries=budget.max_search_queries,
            ),
            stage="cache_review_and_search_plan",
        )
        plan_content = _decode_response(plan_response)
        hints, queries, plan_rejected = _parse_plan(
            plan_content,
            targets=targets,
            notes=ledger.notes,
            checklist=checklist,
            max_queries=budget.max_search_queries,
        )
        rejected.extend(plan_rejected)
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
        )
        ledger.record_evidence_gap(
            event=ledger_event("cache_review"),
            result_summary=json.dumps(
                {
                    "target_claim_ids": [target.claim.claim_id for target in targets],
                    "accepted_cached_candidates": len(hints),
                    "search_queries": len(queries),
                    "rejected_entries": len(plan_rejected),
                    "verification_reserved_tokens": (
                        verification_reserve.reserved_tokens
                    ),
                    "verification_reserved_cost_usd": (
                        verification_reserve.reserved_cost_usd
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )

        for query in queries:
            try:
                results = tuple(
                    await search(
                        query.query,
                        tavily_client=tavily_client,
                        max_results=budget.max_results_per_search,
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

        if any(record.results for record in searches) and budget.max_reads:
            try:
                read_response = await tracker.call(
                    gap_model,
                    build_evidence_gap_read_prompt(
                        targets=targets,
                        searches=searches,
                        max_reads=budget.max_reads,
                    ),
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
                    )
                )
                source_text = (
                    existing
                    if existing is not None
                    else source_read.cleaned_text
                )
            except SourceReadError as exc:
                acquisitions.append(
                    GapSourceAcquisition(
                        url=selection.url,
                        claim_ids=selection.claim_ids,
                        publisher_identity=selection.publisher_identity,
                        cache_hit=False,
                        outcome="read_error",
                        error=str(exc),
                    )
                )
                ledger.record_evidence_gap(
                    event=ledger_event("source_read_error"),
                    url=selection.url,
                    result_summary=str(exc),
                )
                continue
            if not cache_hit:
                ledger.cache_source(
                    selection.url,
                    source_text,
                    source_links=source_read.source_links,
                    link_capture=source_read.link_capture,
                )
                added_source_urls.append(selection.url)

            if cache_hit:
                existing_note_ids = tuple(
                    note_id
                    for group in ledger.note_ids_for_url(selection.url).values()
                    for note_id in group
                )
                acquisitions.append(
                    GapSourceAcquisition(
                        url=selection.url,
                        claim_ids=selection.claim_ids,
                        publisher_identity=selection.publisher_identity,
                        cache_hit=True,
                        source_chars=len(source_text),
                        note_ids=existing_note_ids,
                        outcome="cache_hit_no_reanalysis",
                    )
                )
                ledger.record_evidence_gap(
                    event=ledger_event("source_cache_hit"),
                    url=selection.url,
                    note_ids=existing_note_ids,
                    result_summary="cached source reused without note reanalysis",
                )
                continue

            try:
                note_response = await tracker.call(
                    note_model,
                    build_evidence_gap_note_prompt(
                        url=selection.url,
                        source_text=source_text,
                        claims=[
                            target_claim_by_id[claim_id]
                            for claim_id in selection.claim_ids
                        ],
                        checklist=checklist,
                    ),
                    stage="note_extraction",
                )
                note_content = _decode_response(note_response)
            except _GapBudgetExhausted as exc:
                stop_reason = EvidenceGapStopReason.BUDGET_EXHAUSTED
                stop_detail = str(exc)
                acquisitions.append(
                    GapSourceAcquisition(
                        url=selection.url,
                        claim_ids=selection.claim_ids,
                        publisher_identity=selection.publisher_identity,
                        cache_hit=cache_hit,
                        source_chars=len(source_text),
                        outcome="note_extraction_not_run_budget",
                        error=str(exc),
                    )
                )
                rejected.append(
                    {
                        "stage": "note_extraction",
                        "url": selection.url,
                        "error": str(exc),
                    }
                )
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                acquisitions.append(
                    GapSourceAcquisition(
                        url=selection.url,
                        claim_ids=selection.claim_ids,
                        publisher_identity=selection.publisher_identity,
                        cache_hit=False,
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
            created_ids: list[str] = []
            allowed_items = {item.item_id for item in checklist.items}
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
                    publisher_identity=selection.publisher_identity,
                    cache_hit=cache_hit,
                    source_chars=len(source_text),
                    note_ids=tuple(created_ids),
                    outcome="notes_created" if created_ids else "zero_notes",
                )
            )
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

        target_claims = [target.claim for target in targets]
        if added_note_ids:
            try:
                refreshed = await attribute_claims(
                    target_claims,
                    blocks=blocks,
                    notes=ledger.notes,
                    model_client=_TrackedClient(
                        attribution_model,
                        tracker,
                        "reattribution",
                    ),
                    settings=attribution_settings,
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
            incremental_claim_ids = {
                attribution.claim.claim_id
                for attribution in incremental_attributions
            }
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
            )
        else:
            refreshed_verification = VerificationResult(claims=())
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
        )
        if (
            tracker.tokens_used >= budget.max_tokens
            or tracker.cost_used >= budget.max_cost_usd
        ):
            stop_reason = EvidenceGapStopReason.BUDGET_EXHAUSTED
            stop_detail = (
                "gap budget reached after the final admitted model call"
            )
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
        routed_claim_ids = {
            hint.claim_id for hint in hints
        } | {
            claim_id
            for search_record in searches
            for claim_id in search_record.query.claim_ids
        }
        routed_count = sum(
            target.claim.claim_id in routed_claim_ids for target in targets
        )
        stop_detail = (
            "single bounded evidence-gap pass ended; "
            f"routed target claims={routed_count}/{len(targets)}; "
            f"unrouted target claims={len(targets) - routed_count}; "
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
        rejected_entries=tuple(rejected),
        searches=tuple(searches),
        read_selections=selections,
        acquisitions=tuple(acquisitions),
        added_source_urls=tuple(added_source_urls),
        added_note_ids=tuple(added_note_ids),
        verification_reserve=verification_reserve,
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
