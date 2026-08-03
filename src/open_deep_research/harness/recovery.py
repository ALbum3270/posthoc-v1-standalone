"""Non-mutating triage and one bounded evidence-recovery pass.

The triage decides which *already written* evidence exceptions are important
enough to research again.  It never edits report bytes or evidence records.
The recovery executor reuses the bounded evidence-gap machinery, but freezes
the triage-selected claim IDs before any search and audits attempts separately
from support outcomes.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Awaitable, Mapping, Sequence
from enum import Enum
from typing import Any, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from open_deep_research.harness.budget import RunCostCapReached
from open_deep_research.harness.checklist import ResearchChecklist
from open_deep_research.harness.claims import CitationRequirement
from open_deep_research.harness.edit import EDITORIAL_TARGET_STATES
from open_deep_research.harness.evidence_gap import (
    EvidenceGapResult,
    EvidenceGapStopReason,
    build_evidence_gap_plan_prompt,
)
from open_deep_research.harness.jsonio import loads_lenient
from open_deep_research.harness.ledger import SourceLinkRecord
from open_deep_research.harness.notes import ResearchNote
from open_deep_research.harness.source_leads import (
    SOURCE_LEAD_INVENTORY_LIMITATIONS,
    SourceLeadCandidate,
    inventory_source_lead_candidates,
)
from open_deep_research.harness.verify import (
    ClaimEvidenceState,
    ClaimVerification,
    VerificationRecordStatus,
    VerificationResult,
)


class RecoveryTriageAction(str, Enum):
    """A content-routing judgement, never an evidence verdict."""

    RESEARCH_MORE = "research_more"
    EDIT_DIRECTLY = "edit_directly"
    LEAVE_AS_IS = "leave_as_is"


class RecoveryImportance(str, Enum):
    """How directly the claim contributes to answering the user."""

    CENTRAL = "central"
    SUPPORTING = "supporting"
    INCIDENTAL = "incidental"


class RecoveryTriageStatus(str, Enum):
    """Whether the semantic triage assessed its frozen target set."""

    NO_TARGETS = "no_targets"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class RecoveryQueryRoute(str, Enum):
    """Code-owned route derived from a registered lead selection."""

    SOURCE_CHAIN = "source_chain"
    DIRECT_SEARCH_FALLBACK = "direct_search_fallback"


class RecoverySourceChainAccess(str, Enum):
    """What happened after a registered source-chain lead was selected."""

    NOT_APPLICABLE_DIRECT_SEARCH = "not_applicable_direct_search"
    LEAD_SEARCH_NOT_EXECUTED = "lead_search_not_executed"
    LEAD_SEARCH_NO_RESULT = "lead_search_no_result"
    LEAD_FOUND_NOT_READ = "lead_found_not_read"
    LEAD_FOUND_AND_READABLE = "lead_found_and_readable"
    LEAD_FOUND_BUT_UNREADABLE = "lead_found_but_unreadable"


class RecoveryInapplicabilityReason(str, Enum):
    """Mechanical reasons an evidence anomaly is outside web recovery."""

    NON_EXTERNAL_CITATION_REQUIREMENT = (
        "non_external_citation_requirement"
    )


class RecoveryInapplicableClaim(BaseModel):
    """An anomalous claim retained in audit but excluded from web retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    citation_requirement: CitationRequirement
    reason: RecoveryInapplicabilityReason = (
        RecoveryInapplicabilityReason.NON_EXTERNAL_CITATION_REQUIREMENT
    )
    explanation: str = (
        "evidence recovery performs external source retrieval; this claim's "
        "citation requirement is not external"
    )

    @model_validator(mode="after")
    def _external_claims_are_applicable(self) -> RecoveryInapplicableClaim:
        if self.citation_requirement is CitationRequirement.EXTERNAL:
            raise ValueError(
                "an external claim cannot be recorded as recovery-inapplicable"
            )
        return self


class RecoveryTriageSettings(BaseModel):
    """Mechanical batching only; it contains no content-quality threshold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(default=12, ge=1, le=30)
    source_lead_prompt_char_limit: int = Field(
        default=80_000,
        ge=2,
        description=(
            "Maximum serialized characters for the mechanically exposed "
            "source-lead array in one triage prompt."
        ),
    )


class RecoveryTriageDecision(BaseModel):
    """One model judgement plus a query intent for research targets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    action: RecoveryTriageAction
    importance: RecoveryImportance
    importance_reason: str = Field(min_length=1)
    evidence_need: str | None = None
    preferred_source_role: str | None = None
    query: str | None = None
    selected_source_lead_id: str | None = None
    source_document_hint: str | None = None
    query_route: RecoveryQueryRoute | None = None
    rejected_source_lead_id: str | None = None

    @field_validator(
        "importance_reason",
        "evidence_need",
        "preferred_source_role",
        "query",
        "selected_source_lead_id",
        "source_document_hint",
        "rejected_source_lead_id",
    )
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def _query_intent_belongs_only_to_research(self) -> RecoveryTriageDecision:
        intent = (
            self.evidence_need,
            self.preferred_source_role,
            self.query,
        )
        if self.action is RecoveryTriageAction.RESEARCH_MORE:
            if any(value is None for value in intent):
                raise ValueError(
                    "research_more requires evidence_need, "
                    "preferred_source_role, and query"
                )
            if self.query_route is RecoveryQueryRoute.SOURCE_CHAIN:
                if (
                    self.selected_source_lead_id is None
                    or self.source_document_hint is None
                ):
                    raise ValueError(
                        "source_chain requires a registered lead and "
                        "code-resolved document hint"
                    )
            elif self.query_route is RecoveryQueryRoute.DIRECT_SEARCH_FALLBACK:
                if (
                    self.selected_source_lead_id is not None
                    or self.source_document_hint is not None
                ):
                    raise ValueError(
                        "direct_search_fallback cannot claim a source lead"
                    )
            else:
                raise ValueError("research_more requires a code-owned route")
        elif any(
            value is not None
            for value in (
                *intent,
                self.selected_source_lead_id,
                self.source_document_hint,
                self.query_route,
                self.rejected_source_lead_id,
            )
        ):
            raise ValueError(
                "only research_more may carry a retrieval intent"
            )
        return self


class RecoveryTriageCallUsage(BaseModel):
    """Measured output of one triage batch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_number: int = Field(ge=1)
    claim_ids: tuple[str, ...]
    outcome: str = Field(min_length=1)
    prompt_chars: int = Field(default=0, ge=0)
    token_count: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)


class RecoveryTriageResult(BaseModel):
    """Complete non-mutating audit of the triage decision boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: RecoveryTriageStatus
    target_claim_ids: tuple[str, ...] = ()
    decisions: tuple[RecoveryTriageDecision, ...] = ()
    failed_claim_ids: tuple[str, ...] = ()
    inapplicable_claims: tuple[RecoveryInapplicableClaim, ...] = ()
    diagnostics: tuple[str, ...] = ()
    usage: tuple[RecoveryTriageCallUsage, ...] = ()
    source_leads: tuple[SourceLeadCandidate, ...] = ()
    source_lead_prompt_inventory_count: int | None = Field(
        default=None,
        ge=0,
    )
    source_lead_prompt_shown_count: int | None = Field(
        default=None,
        ge=0,
    )
    source_lead_prompt_omitted_count: int | None = Field(
        default=None,
        ge=0,
    )
    source_lead_prompt_char_limit: int | None = Field(
        default=None,
        ge=0,
    )
    source_lead_prompt_serialized_chars: int | None = Field(
        default=None,
        ge=0,
    )
    source_lead_prompt_truncated: bool | None = None
    source_lead_inventory_limitations: tuple[str, ...] = (
        SOURCE_LEAD_INVENTORY_LIMITATIONS
    )
    canonical_draft_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_draft_unchanged: bool = True
    claim_registry_unchanged: bool = True
    triage_is_non_mutating: bool = True

    @model_validator(mode="after")
    def _decisions_partition_the_scope(self) -> RecoveryTriageResult:
        target_ids = tuple(self.target_claim_ids)
        decision_ids = tuple(decision.claim_id for decision in self.decisions)
        failed_ids = tuple(self.failed_claim_ids)
        inapplicable_ids = tuple(
            claim.claim_id for claim in self.inapplicable_claims
        )
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("recovery triage target IDs must be unique")
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("recovery triage decisions must be unique")
        if len(set(failed_ids)) != len(failed_ids):
            raise ValueError("recovery triage failures must be unique")
        if len(set(inapplicable_ids)) != len(inapplicable_ids):
            raise ValueError("inapplicable recovery claim IDs must be unique")
        if set(inapplicable_ids) & set(target_ids):
            raise ValueError(
                "an inapplicable recovery claim cannot enter triage scope"
            )
        if set(decision_ids) | set(failed_ids) != set(target_ids):
            raise ValueError(
                "triage decisions and failures must partition targets"
            )
        if set(decision_ids) & set(failed_ids):
            raise ValueError("a triage target cannot succeed and fail")
        expected_status = (
            RecoveryTriageStatus.NO_TARGETS
            if not target_ids
            else RecoveryTriageStatus.COMPLETE
            if not failed_ids
            else RecoveryTriageStatus.PARTIAL
            if decision_ids
            else RecoveryTriageStatus.FAILED
        )
        if self.status is not expected_status:
            raise ValueError("triage status must reflect substantive decisions")
        prompt_counts = (
            self.source_lead_prompt_inventory_count,
            self.source_lead_prompt_shown_count,
            self.source_lead_prompt_omitted_count,
            self.source_lead_prompt_char_limit,
            self.source_lead_prompt_serialized_chars,
        )
        if any(value is not None for value in prompt_counts):
            if any(value is None for value in prompt_counts):
                raise ValueError(
                    "source-lead prompt capacity fields must be recorded together"
                )
            assert self.source_lead_prompt_inventory_count is not None
            assert self.source_lead_prompt_shown_count is not None
            assert self.source_lead_prompt_omitted_count is not None
            assert self.source_lead_prompt_char_limit is not None
            assert self.source_lead_prompt_serialized_chars is not None
            if self.source_lead_prompt_inventory_count != len(self.source_leads):
                raise ValueError(
                    "source-lead prompt inventory count must equal full audit inventory"
                )
            if (
                self.source_lead_prompt_shown_count
                + self.source_lead_prompt_omitted_count
                != self.source_lead_prompt_inventory_count
            ):
                raise ValueError(
                    "shown and omitted source-lead counts must partition inventory"
                )
            if (
                self.source_lead_prompt_serialized_chars
                > self.source_lead_prompt_char_limit
            ):
                raise ValueError(
                    "serialized source-lead prompt payload exceeds its capacity"
                )
            expected_truncated = self.source_lead_prompt_omitted_count > 0
            if self.source_lead_prompt_truncated is not expected_truncated:
                raise ValueError(
                    "source-lead prompt truncation must reflect omitted candidates"
                )
        elif self.source_lead_prompt_truncated is not None:
            raise ValueError(
                "source-lead prompt truncation requires capacity fields"
            )
        if not self.canonical_draft_unchanged:
            raise ValueError("recovery triage cannot mutate report bytes")
        if not self.claim_registry_unchanged:
            raise ValueError("recovery triage cannot mutate claim records")
        return self

    @property
    def research_target_claim_ids(self) -> tuple[str, ...]:
        return tuple(
            decision.claim_id
            for decision in self.decisions
            if decision.action is RecoveryTriageAction.RESEARCH_MORE
        )

    @property
    def total_tokens(self) -> int:
        return sum(record.token_count for record in self.usage)

    @property
    def total_cost_usd(self) -> float:
        return sum(record.cost_usd for record in self.usage)


class RecoveryClaimAttempt(BaseModel):
    """Mechanical record of what was actually attempted for one target."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    query_route: RecoveryQueryRoute
    selected_source_lead_id: str | None = None
    source_document_hint: str | None = None
    source_chain_access: RecoverySourceChainAccess
    planned_query_count: int = Field(ge=0)
    executed_search_count: int = Field(ge=0)
    search_error_count: int = Field(ge=0)
    search_result_count: int = Field(ge=0)
    cached_candidate_count: int = Field(ge=0)
    selected_read_urls: tuple[str, ...] = ()
    source_read_outcomes: dict[str, str] = Field(default_factory=dict)
    found_but_unreadable_urls: tuple[str, ...] = ()
    unread_candidate_urls: tuple[str, ...] = ()
    new_completed_relation_count: int = Field(ge=0)
    new_completed_verdict_counts: dict[str, int] = Field(default_factory=dict)
    attempted: bool


class EvidenceRecoveryStopReason(str, Enum):
    """Why the single recovery pass stopped."""

    NO_RESEARCH_TARGETS = "no_research_targets"
    TARGETS_ATTEMPTED = "targets_attempted"
    NO_INFORMATION_YIELD = "no_information_yield"
    SINGLE_PASS_ENDED_WITH_UNATTEMPTED_TARGETS = (
        "single_pass_ended_with_unattempted_targets"
    )
    BUDGET_EXHAUSTED = "budget_exhausted"
    MODEL_ERROR = "model_error"


class EvidenceRecoveryResult(BaseModel):
    """Triage plus one non-iterative, frozen-target recovery execution."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    triage: RecoveryTriageResult
    frozen_target_claim_ids: tuple[str, ...] = ()
    attempts: tuple[RecoveryClaimAttempt, ...] = ()
    attempted_claim_ids: tuple[str, ...] = ()
    unattempted_claim_ids: tuple[str, ...] = ()
    unread_candidate_urls: tuple[str, ...] = ()
    stop_reason: EvidenceRecoveryStopReason
    stop_detail: str
    pass_result: EvidenceGapResult
    canonical_draft_unchanged: bool = True
    claim_registry_unchanged: bool = True
    target_set_frozen_before_search: bool = True
    automatic_second_pass_allowed: bool = False

    @model_validator(mode="after")
    def _attempts_partition_frozen_targets(self) -> EvidenceRecoveryResult:
        targets = tuple(self.frozen_target_claim_ids)
        if targets != self.triage.research_target_claim_ids:
            raise ValueError(
                "frozen recovery targets must equal triage research targets"
            )
        if tuple(self.pass_result.target_claim_ids) != targets:
            raise ValueError(
                "gap executor targets must equal frozen recovery targets"
            )
        attempts = tuple(attempt.claim_id for attempt in self.attempts)
        if attempts != targets:
            raise ValueError("one recovery attempt record is required per target")
        if set(self.attempted_claim_ids) | set(
            self.unattempted_claim_ids
        ) != set(targets):
            raise ValueError("attempted and unattempted IDs must partition targets")
        if set(self.attempted_claim_ids) & set(self.unattempted_claim_ids):
            raise ValueError("a recovery target cannot be attempted and unattempted")
        if not self.canonical_draft_unchanged:
            raise ValueError("evidence recovery cannot mutate report bytes")
        if not self.claim_registry_unchanged:
            raise ValueError("evidence recovery cannot mutate claim records")
        return self

    @property
    def total_tokens(self) -> int:
        return self.triage.total_tokens + self.pass_result.total_tokens

    @property
    def total_cost_usd(self) -> float:
        return self.triage.total_cost_usd + self.pass_result.total_cost_usd


class RecoveryTriageModelClient(Protocol):
    """Injected model boundary for non-mutating content triage."""

    def generate(self, prompt: str) -> Any | Awaitable[Any]:
        """Return one measured JSON envelope."""


class _Envelope(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    content: Any
    token_count: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)


class _RawDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    action: RecoveryTriageAction
    importance: RecoveryImportance
    importance_reason: str
    evidence_need: str | None = None
    preferred_source_role: str | None = None
    query: str | None = None
    selected_source_lead_id: str | None = None


_TRIAGE_PROMPT = """\
Perform a non-mutating recovery triage over evidence exceptions in an already
written report. Return json only. Do not edit, delete, qualify, or rewrite any
report text or claim. Evidence states are completed audit outcomes, not claims
about everything that exists in the world.

For every claim_id choose exactly one action:
- research_more: a concrete factual point materially answers the user's
  question and should receive one bounded attempt to find a better source;
- edit_directly: further research is not the right response, for example the
  wording is dispensable, over-strong, causal, intentional, or already
  contradicted by completed evidence;
- leave_as_is: retaining the visibly labelled uncertainty is more useful than
  either retrieval or immediate editing.

Never choose research_more for a refuted claim. Research is not a search for
agreement: a later source may support, contradict, fail to support, or provide
insufficient information, and all four are useful outcomes.

For research_more, also provide:
- evidence_need: the exact date, amount, legal status, event, attribution, or
  other proposition the source must address;
- preferred_source_role: the useful role of a source, such as an original
  record or independent explanatory reporting. This is a semantic preference,
  not a domain allowlist and not a claim that a host is authoritative;
- query: one focused initial query;
- selected_source_lead_id: an ID from the registered source-lead array when a
  concrete title, date, identifier, issuing body, or original URL is useful;
  otherwise null. Never invent an ID.

Code, not the model, derives the route: selecting a registered ID means a
source-chain search; null means the supplied query is a direct-search
fallback. A registered lead is only a clue, not evidence and not proof that a
source is original. Original records are useful for their own contents, while
independent reporting can still be needed for explanation and cross-checking;
do not let an official source monopolize interpretive claims. For other
actions, all four retrieval-intent fields and selected_source_lead_id must be
null.

The source-lead object records whether the full inventory was mechanically
truncated to fit a serialized-character capacity. This is not a relevance or
authority ranking. Select only an ID shown in its `leads` array. If
`truncated` is true, omitted candidates may still exist; do not infer their
absence, and use the direct-search fallback when no shown clue is useful.

Return exactly one entry per claim_id:
{{"decisions":[{{"claim_id":"claim-0001",\
"action":"research_more|edit_directly|leave_as_is",\
"importance":"central|supporting|incidental",\
"importance_reason":"how it contributes to the user's question",\
"evidence_need":"specific proposition or null",\
"preferred_source_role":"source role or null",\
"query":"focused query or null",\
"selected_source_lead_id":"lead-... or null"}}]}}

Research topic and checklist:
{checklist}

Frozen evidence exceptions:
{claims}

Registered source-chain candidates (mechanical text shapes, not evidence):
{source_leads}
"""


class _SourceLeadPromptProjection(BaseModel):
    """Capacity-bounded prompt view while retaining the complete audit view."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    inventory_count: int = Field(ge=0)
    shown_leads: tuple[SourceLeadCandidate, ...] = ()
    omitted_count: int = Field(ge=0)
    char_limit: int = Field(ge=2)
    serialized_chars: int = Field(ge=0)

    @model_validator(mode="after")
    def _counts_and_capacity_are_consistent(
        self,
    ) -> _SourceLeadPromptProjection:
        if len(self.shown_leads) + self.omitted_count != self.inventory_count:
            raise ValueError("source-lead prompt projection must partition inventory")
        if self.serialized_chars > self.char_limit:
            raise ValueError("source-lead prompt projection exceeds capacity")
        return self

    @property
    def truncated(self) -> bool:
        return self.omitted_count > 0

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "inventory_count": self.inventory_count,
            "shown_count": len(self.shown_leads),
            "omitted_count": self.omitted_count,
            "serialized_char_limit": self.char_limit,
            "serialized_chars": self.serialized_chars,
            "truncated": self.truncated,
            "selection_scope": (
                "shown leads are the deterministic inventory-order prefix; "
                "this is not a relevance or authority ranking"
            ),
            "leads": [
                lead.model_dump(mode="json") for lead in self.shown_leads
            ],
        }


def _serialized_source_leads_chars(
    leads: Sequence[SourceLeadCandidate],
) -> int:
    """Measure the exact JSON array emitted into the triage prompt."""

    return len(
        json.dumps(
            [lead.model_dump(mode="json") for lead in leads],
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _project_source_leads_for_prompt(
    source_leads: Sequence[SourceLeadCandidate],
    *,
    char_limit: int,
) -> _SourceLeadPromptProjection:
    """Expose a whole-candidate prefix up to a mechanical prompt capacity.

    Candidate text is never cut mid-record: when the next complete candidate
    would exceed capacity, the remaining inventory is omitted from the model
    prompt but remains in the result's complete audit inventory. The prompt
    explicitly offers direct search as the degradation path.
    """

    shown: list[SourceLeadCandidate] = []
    for lead in source_leads:
        candidate = (*shown, lead)
        if _serialized_source_leads_chars(candidate) > char_limit:
            break
        shown.append(lead)
    serialized_chars = _serialized_source_leads_chars(shown)
    return _SourceLeadPromptProjection(
        inventory_count=len(source_leads),
        shown_leads=tuple(shown),
        omitted_count=len(source_leads) - len(shown),
        char_limit=char_limit,
        serialized_chars=serialized_chars,
    )


def _claim_registry_sha256(verification: VerificationResult) -> str:
    payload = json.dumps(
        [claim.model_dump(mode="json") for claim in verification.claims],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def recovery_triage_targets(
    verification: VerificationResult,
) -> tuple[ClaimVerification, ...]:
    """Select external content anomalies accepted by the gap executor."""

    return tuple(
        claim
        for claim in verification.claims
        if claim.state in EDITORIAL_TARGET_STATES
        and claim.claim.citation_requirement is CitationRequirement.EXTERNAL
    )


def recovery_inapplicable_claims(
    verification: VerificationResult,
) -> tuple[RecoveryInapplicableClaim, ...]:
    """Retain non-external anomalies without sending them to web retrieval."""

    return tuple(
        RecoveryInapplicableClaim(
            claim_id=claim.claim.claim_id,
            citation_requirement=claim.claim.citation_requirement,
        )
        for claim in verification.claims
        if claim.state in EDITORIAL_TARGET_STATES
        and claim.claim.citation_requirement is not CitationRequirement.EXTERNAL
    )


def build_recovery_triage_prompt(
    targets: Sequence[ClaimVerification],
    *,
    checklist: ResearchChecklist,
    source_leads: Sequence[SourceLeadCandidate] = (),
    source_lead_prompt_char_limit: int = 80_000,
) -> str:
    """Build the semantic triage prompt without any source allowlist."""

    projection = _project_source_leads_for_prompt(
        source_leads,
        char_limit=source_lead_prompt_char_limit,
    )
    return _build_recovery_triage_prompt(
        targets,
        checklist=checklist,
        source_lead_projection=projection,
    )


def _build_recovery_triage_prompt(
    targets: Sequence[ClaimVerification],
    *,
    checklist: ResearchChecklist,
    source_lead_projection: _SourceLeadPromptProjection,
) -> str:
    """Build one prompt from a previously measured source-lead projection."""

    claims = [
        {
            "claim_id": target.claim.claim_id,
            "claim_text": target.claim.claim_text,
            "anchor_text": target.claim.anchor_text,
            "evidence_state": target.state.value,
            "relations": [
                {
                    "url": relation.url,
                    "status": relation.status.value,
                    "semantic_verdict": (
                        relation.semantic_verdict.value
                        if relation.semantic_verdict is not None
                        else None
                    ),
                    "explanation": relation.explanation,
                    "source_quote": relation.source_quote,
                }
                for relation in target.relations
            ],
        }
        for target in targets
    ]
    checklist_payload = {
        "topic": checklist.topic,
        "items": [
            {"item_id": item.item_id, "question": item.question}
            for item in checklist.items
        ],
    }
    return _TRIAGE_PROMPT.format(
        checklist=json.dumps(
            checklist_payload, ensure_ascii=False, sort_keys=True
        ),
        claims=json.dumps(claims, ensure_ascii=False, sort_keys=True),
        source_leads=json.dumps(
            source_lead_projection.prompt_payload(),
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


async def _call_triage_model(
    model_client: RecoveryTriageModelClient,
    prompt: str,
) -> tuple[Any, int, float]:
    response = model_client.generate(prompt)
    if inspect.isawaitable(response):
        response = await response
    envelope = _Envelope.model_validate(response)
    content = envelope.content
    if isinstance(content, str):
        content = loads_lenient(content)
    return content, envelope.token_count, envelope.cost_usd


def _parse_triage_batch(
    content: Any,
    targets: Sequence[ClaimVerification],
    *,
    source_leads: Sequence[SourceLeadCandidate] = (),
) -> tuple[
    tuple[RecoveryTriageDecision, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    expected = tuple(target.claim.claim_id for target in targets)
    expected_set = set(expected)
    state_by_id = {
        target.claim.claim_id: target.state for target in targets
    }
    source_lead_by_id = {lead.lead_id: lead for lead in source_leads}
    diagnostics: list[str] = []
    failed: set[str] = set()
    by_id: dict[str, RecoveryTriageDecision] = {}
    raw_decisions = (
        content.get("decisions") if isinstance(content, Mapping) else None
    )
    if not isinstance(raw_decisions, (list, tuple)):
        return (), expected, ("recovery_triage_payload_invalid",)
    duplicates: set[str] = set()
    for index, raw in enumerate(raw_decisions):
        raw_id = raw.get("claim_id") if isinstance(raw, Mapping) else None
        try:
            proposal = _RawDecision.model_validate(raw)
            proposal_payload = proposal.model_dump(mode="python")
            selected_lead_id = proposal.selected_source_lead_id
            selected_lead = (
                source_lead_by_id.get(selected_lead_id)
                if selected_lead_id is not None
                else None
            )
            rejected_lead_id: str | None = None
            if selected_lead_id is not None and selected_lead is None:
                rejected_lead_id = selected_lead_id
                diagnostics.append(
                    "recovery_triage_unknown_source_lead_fell_back_direct: "
                    f"{proposal.claim_id} -> {selected_lead_id}"
                )
            proposal_payload.update(
                {
                    "selected_source_lead_id": (
                        selected_lead.lead_id
                        if selected_lead is not None
                        else None
                    ),
                    "source_document_hint": (
                        selected_lead.verbatim_text
                        if selected_lead is not None
                        else None
                    ),
                    "query_route": (
                        RecoveryQueryRoute.SOURCE_CHAIN
                        if selected_lead is not None
                        else RecoveryQueryRoute.DIRECT_SEARCH_FALLBACK
                    )
                    if proposal.action is RecoveryTriageAction.RESEARCH_MORE
                    else None,
                    "rejected_source_lead_id": rejected_lead_id,
                }
            )
            decision = RecoveryTriageDecision.model_validate(
                proposal_payload
            )
        except (TypeError, ValidationError, ValueError) as exc:
            diagnostics.append(f"recovery_triage_entry_invalid[{index}]: {exc}")
            if isinstance(raw_id, str) and raw_id in expected_set:
                failed.add(raw_id)
            continue
        if decision.claim_id not in expected_set:
            diagnostics.append(
                f"recovery_triage_unknown_claim: {decision.claim_id}"
            )
            continue
        if (
            state_by_id[decision.claim_id] is ClaimEvidenceState.REFUTED
            and decision.action is RecoveryTriageAction.RESEARCH_MORE
        ):
            diagnostics.append(
                "recovery_triage_refuted_research_rejected: "
                f"{decision.claim_id}"
            )
            failed.add(decision.claim_id)
            continue
        if decision.claim_id in by_id:
            by_id.pop(decision.claim_id, None)
            duplicates.add(decision.claim_id)
            failed.add(decision.claim_id)
            diagnostics.append(
                f"recovery_triage_duplicate_claim: {decision.claim_id}"
            )
            continue
        if decision.claim_id in duplicates:
            continue
        by_id[decision.claim_id] = decision
    ordered: list[RecoveryTriageDecision] = []
    for claim_id in expected:
        if claim_id in by_id and claim_id not in failed:
            ordered.append(by_id[claim_id])
        else:
            failed.add(claim_id)
            diagnostics.append(f"recovery_triage_missing_or_invalid: {claim_id}")
    return tuple(ordered), tuple(
        claim_id for claim_id in expected if claim_id in failed
    ), tuple(diagnostics)


async def triage_evidence_recovery(
    canonical_draft: str,
    *,
    checklist: ResearchChecklist,
    verification: VerificationResult,
    model_client: RecoveryTriageModelClient,
    settings: RecoveryTriageSettings | None = None,
    source_cache: Mapping[str, str] | None = None,
    source_links: Mapping[str, Sequence[SourceLinkRecord]] | None = None,
) -> RecoveryTriageResult:
    """Assess all completed evidence anomalies without changing any bytes."""

    frozen_draft = canonical_draft
    frozen_registry_hash = _claim_registry_sha256(verification)
    targets = recovery_triage_targets(verification)
    source_leads = inventory_source_lead_candidates(
        source_cache or {},
        source_links=source_links,
    )
    inapplicable_claims = recovery_inapplicable_claims(verification)
    target_ids = tuple(target.claim.claim_id for target in targets)
    draft_hash = hashlib.sha256(canonical_draft.encode("utf-8")).hexdigest()
    active_settings = settings or RecoveryTriageSettings()
    source_lead_projection = _project_source_leads_for_prompt(
        source_leads,
        char_limit=active_settings.source_lead_prompt_char_limit,
    )
    source_lead_audit = {
        "source_lead_prompt_inventory_count": (
            source_lead_projection.inventory_count
        ),
        "source_lead_prompt_shown_count": len(
            source_lead_projection.shown_leads
        ),
        "source_lead_prompt_omitted_count": source_lead_projection.omitted_count,
        "source_lead_prompt_char_limit": source_lead_projection.char_limit,
        "source_lead_prompt_serialized_chars": (
            source_lead_projection.serialized_chars
        ),
        "source_lead_prompt_truncated": source_lead_projection.truncated,
    }
    if not targets:
        return RecoveryTriageResult(
            status=RecoveryTriageStatus.NO_TARGETS,
            inapplicable_claims=inapplicable_claims,
            canonical_draft_sha256=draft_hash,
            claim_registry_sha256=frozen_registry_hash,
            source_leads=source_leads,
            **source_lead_audit,
        )

    decisions: list[RecoveryTriageDecision] = []
    failed_ids: list[str] = []
    diagnostics: list[str] = []
    if source_lead_projection.truncated:
        diagnostics.append(
            "recovery_triage_source_leads_mechanically_truncated: "
            f"shown={len(source_lead_projection.shown_leads)}/"
            f"{source_lead_projection.inventory_count}; "
            f"omitted={source_lead_projection.omitted_count}; "
            f"serialized_chars={source_lead_projection.serialized_chars}; "
            f"char_limit={source_lead_projection.char_limit}; "
            "direct_search_fallback_remains_available"
        )
    usage: list[RecoveryTriageCallUsage] = []
    for start in range(0, len(targets), active_settings.batch_size):
        batch = targets[start : start + active_settings.batch_size]
        batch_number = len(usage) + 1
        batch_ids = tuple(target.claim.claim_id for target in batch)
        prompt_chars = 0
        phase = "prompt_build"
        try:
            prompt = _build_recovery_triage_prompt(
                batch,
                checklist=checklist,
                source_lead_projection=source_lead_projection,
            )
            prompt_chars = len(prompt)
            phase = "model_call_or_response_processing"
            content, tokens, cost = await _call_triage_model(
                model_client,
                prompt,
            )
            parsed, failed, batch_diagnostics = _parse_triage_batch(
                content,
                batch,
                source_leads=source_lead_projection.shown_leads,
            )
        except RunCostCapReached:
            raise
        except Exception as exc:
            tokens = 0
            cost = 0.0
            parsed = ()
            failed = batch_ids
            batch_diagnostics = (
                f"recovery_triage_batch_error[{batch_number}]: "
                f"phase={phase}; claim_ids={list(batch_ids)}; "
                f"prompt_chars={prompt_chars}; source_leads_shown="
                f"{len(source_lead_projection.shown_leads)}/"
                f"{source_lead_projection.inventory_count}; "
                f"source_leads_omitted={source_lead_projection.omitted_count}; "
                f"source_lead_serialized_chars="
                f"{source_lead_projection.serialized_chars}; "
                f"source_lead_char_limit={source_lead_projection.char_limit}; "
                f"{type(exc).__name__}: {exc}",
            )
        decisions.extend(parsed)
        failed_ids.extend(failed)
        diagnostics.extend(batch_diagnostics)
        usage.append(
            RecoveryTriageCallUsage(
                batch_number=batch_number,
                claim_ids=batch_ids,
                outcome=(
                    "completed"
                    if not failed
                    else "failed"
                    if len(failed) == len(batch_ids)
                    else "partial"
                ),
                prompt_chars=prompt_chars,
                token_count=tokens,
                cost_usd=cost,
            )
        )

    if canonical_draft != frozen_draft:
        raise AssertionError("recovery triage mutated the canonical draft")
    if _claim_registry_sha256(verification) != frozen_registry_hash:
        raise AssertionError("recovery triage mutated the claim registry")
    ordered_decisions = tuple(
        decision
        for claim_id in target_ids
        for decision in decisions
        if decision.claim_id == claim_id
    )
    ordered_failed = tuple(
        claim_id for claim_id in target_ids if claim_id in set(failed_ids)
    )
    status = (
        RecoveryTriageStatus.COMPLETE
        if not ordered_failed
        else RecoveryTriageStatus.PARTIAL
        if ordered_decisions
        else RecoveryTriageStatus.FAILED
    )
    return RecoveryTriageResult(
        status=status,
        target_claim_ids=target_ids,
        decisions=ordered_decisions,
        failed_claim_ids=ordered_failed,
        inapplicable_claims=inapplicable_claims,
        diagnostics=tuple(diagnostics),
        usage=tuple(usage),
        source_leads=source_leads,
        **source_lead_audit,
        canonical_draft_sha256=draft_hash,
        claim_registry_sha256=frozen_registry_hash,
    )


def build_recovery_gap_plan_prompt(
    *,
    targets: Sequence[ClaimVerification],
    notes: Sequence[ResearchNote],
    checklist: ResearchChecklist,
    max_queries: int,
    triage: RecoveryTriageResult,
) -> str:
    """Bind ordinary bounded planning to the frozen triage query intents."""

    target_ids = {target.claim.claim_id for target in targets}
    intents = [
        decision.model_dump(mode="json")
        for decision in triage.decisions
        if decision.claim_id in target_ids
        and decision.action is RecoveryTriageAction.RESEARCH_MORE
    ]
    base = build_evidence_gap_plan_prompt(
        targets=targets,
        notes=notes,
        checklist=checklist,
        max_queries=max_queries,
    )
    return (
        "This is the only bounded evidence-recovery pass. The target IDs and "
        "report wording are frozen. First review cached notes. Then use the "
        "recorded evidence_need, source role, query_route, query, and "
        "code-resolved document hint to select or merge at most the stated "
        "number of focused searches. For source_chain, follow the registered "
        "title, date, identifier, issuing body, or URL. For "
        "direct_search_fallback, there was no registered upstream clue: use "
        "the recorded claim/evidence/source-role query rather than silently "
        "omitting the target. Do not invent a document or treat a preferred "
        "role as a host allowlist. Original records and independent reporting "
        "can coexist. Any later verdict is useful; do not search only for "
        "agreement.\n\n"
        "Frozen recovery intents:\n"
        + json.dumps(intents, ensure_ascii=False, sort_keys=True)
        + "\n\n"
        + base
    )


def _completed_relations_by_claim(
    verification: VerificationResult,
) -> dict[str, dict[tuple[str, str], Any]]:
    result: dict[str, dict[tuple[str, str], Any]] = {}
    for claim in verification.claims:
        result[claim.claim.claim_id] = {
            (relation.source_id, relation.url): relation
            for relation in claim.relations
            if relation.status is VerificationRecordStatus.COMPLETED
        }
    return result


def summarize_evidence_recovery(
    *,
    triage: RecoveryTriageResult,
    pass_result: EvidenceGapResult,
    initial_verification: VerificationResult,
    cached_source_urls: Sequence[str],
) -> EvidenceRecoveryResult:
    """Derive claim-level attempts and a non-optimistic single-pass stop."""

    target_ids = triage.research_target_claim_ids
    initial_relations = _completed_relations_by_claim(initial_verification)
    final_relations = _completed_relations_by_claim(
        pass_result.final_verification
    )
    cached_urls = set(cached_source_urls)
    decision_by_id = {
        decision.claim_id: decision for decision in triage.decisions
    }
    attempts: list[RecoveryClaimAttempt] = []
    all_unread: set[str] = set()
    for claim_id in target_ids:
        decision = decision_by_id[claim_id]
        if decision.query_route is None:
            raise AssertionError(
                f"research target has no code-owned query route: {claim_id}"
            )
        searches = tuple(
            search
            for search in pass_result.searches
            if claim_id in search.query.claim_ids
        )
        hints = tuple(
            hint
            for hint in pass_result.cached_candidate_hints
            if hint.claim_id == claim_id
        )
        selections = tuple(
            selection
            for selection in pass_result.read_selections
            if claim_id in selection.claim_ids
        )
        acquisitions = tuple(
            acquisition
            for acquisition in pass_result.acquisitions
            if claim_id in acquisition.claim_ids
        )
        selected_urls = {selection.url for selection in selections}
        result_urls = {
            result.url for search in searches for result in search.results
        }
        unread_urls = tuple(
            sorted(result_urls - selected_urls - cached_urls)
        )
        all_unread.update(unread_urls)
        unreadable = tuple(
            acquisition.url
            for acquisition in acquisitions
            if acquisition.outcome == "read_error"
        )
        previous = initial_relations.get(claim_id, {})
        current = final_relations.get(claim_id, {})
        new_relations = tuple(
            relation
            for identity, relation in current.items()
            if identity not in previous
        )
        verdict_counts: dict[str, int] = {}
        for relation in new_relations:
            verdict = (
                relation.semantic_verdict.value
                if relation.semantic_verdict is not None
                else "none"
            )
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        attempted = bool(searches or hints)
        if decision.query_route is RecoveryQueryRoute.DIRECT_SEARCH_FALLBACK:
            source_chain_access = (
                RecoverySourceChainAccess.NOT_APPLICABLE_DIRECT_SEARCH
            )
        elif any(
            acquisition.outcome != "read_error"
            for acquisition in acquisitions
        ):
            source_chain_access = (
                RecoverySourceChainAccess.LEAD_FOUND_AND_READABLE
            )
        elif acquisitions:
            source_chain_access = (
                RecoverySourceChainAccess.LEAD_FOUND_BUT_UNREADABLE
            )
        elif result_urls:
            source_chain_access = RecoverySourceChainAccess.LEAD_FOUND_NOT_READ
        elif searches:
            source_chain_access = RecoverySourceChainAccess.LEAD_SEARCH_NO_RESULT
        else:
            source_chain_access = (
                RecoverySourceChainAccess.LEAD_SEARCH_NOT_EXECUTED
            )
        attempts.append(
            RecoveryClaimAttempt(
                claim_id=claim_id,
                query_route=decision.query_route,
                selected_source_lead_id=decision.selected_source_lead_id,
                source_document_hint=decision.source_document_hint,
                source_chain_access=source_chain_access,
                planned_query_count=len(searches),
                executed_search_count=len(searches),
                search_error_count=sum(
                    search.error is not None for search in searches
                ),
                search_result_count=sum(
                    len(search.results) for search in searches
                ),
                cached_candidate_count=len(hints),
                selected_read_urls=tuple(
                    selection.url for selection in selections
                ),
                source_read_outcomes={
                    acquisition.url: acquisition.outcome
                    for acquisition in acquisitions
                },
                found_but_unreadable_urls=unreadable,
                unread_candidate_urls=unread_urls,
                new_completed_relation_count=len(new_relations),
                new_completed_verdict_counts=verdict_counts,
                attempted=attempted,
            )
        )

    attempted_ids = tuple(
        attempt.claim_id for attempt in attempts if attempt.attempted
    )
    unattempted_ids = tuple(
        attempt.claim_id for attempt in attempts if not attempt.attempted
    )
    no_yield = (
        not pass_result.added_source_urls
        and pass_result.information_yield.new_completed_relation_count == 0
        and not all_unread
    )
    if not target_ids:
        stop_reason = EvidenceRecoveryStopReason.NO_RESEARCH_TARGETS
        stop_detail = "triage selected no claims for evidence recovery"
    elif pass_result.stop_reason is EvidenceGapStopReason.BUDGET_EXHAUSTED:
        stop_reason = EvidenceRecoveryStopReason.BUDGET_EXHAUSTED
        stop_detail = pass_result.stop_detail
    elif pass_result.stop_reason is EvidenceGapStopReason.MODEL_ERROR:
        stop_reason = EvidenceRecoveryStopReason.MODEL_ERROR
        stop_detail = pass_result.stop_detail
    elif no_yield:
        stop_reason = EvidenceRecoveryStopReason.NO_INFORMATION_YIELD
        stop_detail = (
            "single bounded recovery pass produced no new source, no new "
            "completed claim-source relation, and no unread candidate; "
            f"attempted={len(attempted_ids)}/{len(target_ids)}"
        )
    elif not unattempted_ids:
        stop_reason = EvidenceRecoveryStopReason.TARGETS_ATTEMPTED
        stop_detail = (
            "every frozen recovery target received an explicit cached-source "
            "or search attempt"
        )
    else:
        stop_reason = (
            EvidenceRecoveryStopReason.SINGLE_PASS_ENDED_WITH_UNATTEMPTED_TARGETS
        )
        stop_detail = (
            "the only allowed recovery pass ended with unattempted frozen "
            f"targets: {', '.join(unattempted_ids)}"
        )
    return EvidenceRecoveryResult(
        triage=triage,
        frozen_target_claim_ids=target_ids,
        attempts=tuple(attempts),
        attempted_claim_ids=attempted_ids,
        unattempted_claim_ids=unattempted_ids,
        unread_candidate_urls=tuple(sorted(all_unread)),
        stop_reason=stop_reason,
        stop_detail=stop_detail,
        pass_result=pass_result,
        canonical_draft_unchanged=pass_result.canonical_draft_unchanged,
        claim_registry_unchanged=pass_result.claim_registry_unchanged,
    )


__all__ = [
    "EvidenceRecoveryResult",
    "EvidenceRecoveryStopReason",
    "RecoveryClaimAttempt",
    "RecoveryInapplicabilityReason",
    "RecoveryInapplicableClaim",
    "RecoveryImportance",
    "RecoveryQueryRoute",
    "RecoverySourceChainAccess",
    "RecoveryTriageAction",
    "RecoveryTriageDecision",
    "RecoveryTriageModelClient",
    "RecoveryTriageResult",
    "RecoveryTriageSettings",
    "RecoveryTriageStatus",
    "build_recovery_gap_plan_prompt",
    "build_recovery_triage_prompt",
    "recovery_triage_targets",
    "recovery_inapplicable_claims",
    "summarize_evidence_recovery",
    "triage_evidence_recovery",
]
