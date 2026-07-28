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
)

from open_deep_research.harness.attribution import (
    AttributionError,
    AttributionModelClient,
    AttributionResult,
    AttributionSettings,
    AttributionStatus,
    CandidateSource,
    ClaimAttribution,
    attribute_claims,
)
from open_deep_research.harness.checklist import ResearchChecklist
from open_deep_research.harness.claims import (
    AtomicClaim,
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
    read,
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
    VerifiedSourceRelation,
    build_claim_verification,
    verify_attributions,
)

_TARGET_STATES = {
    ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    ClaimEvidenceState.SUPPORTED_BELOW_REQUIREMENT,
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
    """One target-bound web query admitted by the single plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
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


class EvidenceGapResult(BaseModel):
    """Audit of one non-iterative gap pass plus code-owned final registries."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    target_claim_ids: tuple[str, ...] = ()
    initial_states: dict[str, ClaimEvidenceState] = Field(default_factory=dict)
    cached_candidate_hints: tuple[CachedCandidateHint, ...] = ()
    rejected_entries: tuple[dict[str, Any], ...] = ()
    searches: tuple[GapSearchRecord, ...] = ()
    read_selections: tuple[GapReadSelection, ...] = ()
    acquisitions: tuple[GapSourceAcquisition, ...] = ()
    added_source_urls: tuple[str, ...] = ()
    added_note_ids: tuple[str, ...] = ()
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
    verification_merge: VerificationMergeAudit | None = None
    final_attribution: AttributionResult = Field(exclude=True)
    final_verification: VerificationResult = Field(exclude=True)

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

    note_id: str
    source_id: str
    independent_from_existing_publishers: bool
    publisher_identity: str
    independence_rationale: str


class _RawSearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str
    query: str


class _RawClaimPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    cached_candidates: tuple[_RawCachedHint, ...] = ()
    needs_web_search: bool
    queries: tuple[_RawSearchQuery, ...] = ()


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

Return:
{{"claims":[{{"claim_id":"claim-0001","cached_candidates":[{{\
"note_id":"note-000001","source_id":"source-id",\
"independent_from_existing_publishers":true,\
"publisher_identity":"publishing organization",\
"independence_rationale":"brief reason"}}],\
"needs_web_search":true,\
"queries":[{{"item_id":"existing-checklist-item","query":"targeted query"}}]}}]}}

Every target claim_id must appear exactly once. Zero cached candidates and
zero queries are legal. Search may seek support, contradiction, absence of
support, or insufficient information; do not optimize only for agreement.

Checklist item IDs:
{item_ids}

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

    async def call(self, client: Any, prompt: str, *, stage: str) -> Any:
        estimated_tokens, estimated_cost = self._estimate(client, prompt)
        if self.tokens_used + estimated_tokens > self.budget.max_tokens:
            raise _GapBudgetExhausted(
                "estimated input exceeds remaining gap token budget"
            )
        if self.cost_used + estimated_cost > self.budget.max_cost_usd:
            raise _GapBudgetExhausted(
                "estimated call exceeds remaining gap cost budget"
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
    def __init__(self, client: Any, tracker: _BudgetTracker, stage: str) -> None:
        self.client = client
        self.tracker = tracker
        self.stage = stage

    async def generate(self, prompt: str) -> Any:
        return await self.tracker.call(self.client, prompt, stage=self.stage)


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
            "required_independent_sources": (
                target.required_independent_sources
            ),
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
) -> str:
    """Expose all cached note summaries before allowing new web search."""

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
        item_ids=json.dumps(
            [item.item_id for item in checklist.items],
            ensure_ascii=False,
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
            "query_claim_id": record.query.claim_id,
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
    raw_claims = content.get("claims") if isinstance(content, Mapping) else None
    if not isinstance(raw_claims, (list, tuple)):
        return (), (), (
            {"stage": "plan", "error": "claims must be an array", "raw": content},
        )

    seen_claims: set[str] = set()
    identities_by_claim: dict[str, set[str]] = {
        claim_id: set() for claim_id in target_by_id
    }
    for index, raw in enumerate(raw_claims):
        try:
            plan = _RawClaimPlan.model_validate(raw)
        except (TypeError, ValidationError, ValueError) as exc:
            rejected.append(
                {"stage": "plan", "index": index, "error": str(exc), "raw": raw}
            )
            continue
        if plan.claim_id not in target_by_id or plan.claim_id in seen_claims:
            rejected.append(
                {
                    "stage": "plan",
                    "index": index,
                    "error": "unknown or duplicate target claim_id",
                    "raw": raw,
                }
            )
            continue
        seen_claims.add(plan.claim_id)
        target = target_by_id[plan.claim_id]
        existing_publishers = set(target.publisher_domain_proxies)
        for hint in plan.cached_candidates:
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
            elif not identity or identity in identities_by_claim[plan.claim_id]:
                error = "publisher identity is blank or repeated for this claim"
            if error is not None:
                rejected.append(
                    {
                        "stage": "cached_candidate",
                        "claim_id": plan.claim_id,
                        "error": error,
                        "raw": hint.model_dump(mode="json"),
                    }
                )
                continue
            identities_by_claim[plan.claim_id].add(identity)
            accepted_hints.append(
                CachedCandidateHint(
                    claim_id=plan.claim_id,
                    note_id=hint.note_id,
                    source_id=hint.source_id,
                    publisher_identity=hint.publisher_identity.strip(),
                    independence_rationale=(
                        hint.independence_rationale.strip()
                    ),
                )
            )
        if not plan.needs_web_search and plan.queries:
            rejected.append(
                {
                    "stage": "plan",
                    "claim_id": plan.claim_id,
                    "error": "queries supplied while needs_web_search was false",
                }
            )
            continue
        if not plan.needs_web_search:
            continue
        for query in plan.queries:
            if len(accepted_queries) >= max_queries:
                rejected.append(
                    {
                        "stage": "search_query",
                        "claim_id": plan.claim_id,
                        "error": "gap search query cap reached",
                        "raw": query.model_dump(mode="json"),
                    }
                )
                continue
            if query.item_id not in item_ids:
                rejected.append(
                    {
                        "stage": "search_query",
                        "claim_id": plan.claim_id,
                        "error": "unknown checklist item_id",
                        "raw": query.model_dump(mode="json"),
                    }
                )
                continue
            accepted_queries.append(
                GapSearchQuery(
                    claim_id=plan.claim_id,
                    item_id=query.item_id,
                    query=query.query.strip(),
                )
            )

    for missing in sorted(set(target_by_id) - seen_claims):
        rejected.append(
            {
                "stage": "plan",
                "claim_id": missing,
                "error": "target claim omitted from cache review",
            }
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
            allowed.setdefault(result.url, set()).add(
                (record.query.claim_id, record.query.item_id)
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
        elif not claim_ids or any(claim_id not in target_by_id for claim_id in claim_ids):
            error = "read must name known target claim_ids"
        elif proposal.item_id not in {
            item_id for _, item_id in allowed[proposal.url]
        }:
            error = "URL/item relation was not present in search routing"
        elif not proposal.independent_from_existing_publishers:
            error = "model did not judge publisher independent"
        elif not identity:
            error = "publisher identity must not be blank"
        elif any(
            _publisher_proxy(proposal.url)
            in set(target_by_id[claim_id].publisher_domain_proxies)
            for claim_id in claim_ids
        ):
            error = "publisher domain proxy already supports a target claim"
        elif any(
            _identity_matches_proxy(
                proposal.publisher_identity,
                proxy,
            )
            for claim_id in claim_ids
            for proxy in target_by_id[claim_id].publisher_domain_proxies
        ):
            error = "publisher identity matches an existing domain label"
        elif any(
            identity in identities_by_claim[claim_id]
            for claim_id in claim_ids
        ):
            error = "publisher identity repeated for a target claim"
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
        seen_urls.add(proposal.url)
        for claim_id in claim_ids:
            identities_by_claim[claim_id].add(identity)
        accepted.append(
            GapReadSelection(
                url=proposal.url.strip(),
                item_id=proposal.item_id,
                claim_ids=claim_ids,
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
                required_sources=original.required_independent_sources,
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
    required_independent_sources: Mapping[str, int] | None = None,
    estimate_input_tokens: Callable[[Any, str], int] | None = None,
    estimate_cost_usd: Callable[[Any, str], float] | None = None,
) -> EvidenceGapResult:
    """Run one cache-first gap pass, then reattribute and reverify once."""

    targets = tuple(
        result
        for result in initial_verification.claims
        if result.state in _TARGET_STATES
    )
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
    stop_reason = EvidenceGapStopReason.COMPLETED
    stop_detail = "single evidence-gap pass completed"
    verification_merge = _unchanged_merge_audit(initial_verification)

    try:
        plan_response = await tracker.call(
            gap_model,
            build_evidence_gap_plan_prompt(
                targets=targets,
                notes=ledger.notes,
                checklist=checklist,
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
        ledger.record_evidence_gap(
            event="cache_review",
            result_summary=json.dumps(
                {
                    "target_claim_ids": [target.claim.claim_id for target in targets],
                    "accepted_cached_candidates": len(hints),
                    "search_queries": len(queries),
                    "rejected_entries": len(plan_rejected),
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
            read_response = await tracker.call(
                gap_model,
                build_evidence_gap_read_prompt(
                    targets=targets,
                    searches=searches,
                    max_reads=budget.max_reads,
                ),
                stage="read_selection",
            )
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
                source_text = existing or await read(
                    selection.url,
                    tavily_client=tavily_client,
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
                    event="source_read_error",
                    url=selection.url,
                    result_summary=str(exc),
                )
                continue
            if not cache_hit:
                ledger.cache_source(selection.url, source_text)
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
                    event="source_cache_hit",
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
            except _GapBudgetExhausted:
                raise
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
                    event="note_model_error",
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
                event="source_acquired",
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
                ),
                settings=verification_settings,
                budget=VerificationBudget(
                    max_tokens=remaining_token_budget,
                    max_cost_usd=remaining_cost_budget,
                ),
                required_independent_sources={
                    claim_id: count
                    for claim_id, count in (
                        required_independent_sources or {}
                    ).items()
                    if claim_id in incremental_claim_ids
                },
                estimate_input_tokens=verification_token_estimator,
                estimate_cost_usd=verification_cost_estimator,
            )
        else:
            refreshed_verification = VerificationResult(claims=())
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

    ledger.record_evidence_gap(
        event="gap_stop",
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
        usage=tuple(tracker.usage),
        stop_reason=stop_reason,
        stop_detail=stop_detail,
        claim_registry_unchanged=claims_unchanged,
        canonical_draft_unchanged=(canonical_draft == frozen_draft),
        verification_merge=verification_merge,
        final_attribution=merged_attribution,
        final_verification=final_verification,
    )
