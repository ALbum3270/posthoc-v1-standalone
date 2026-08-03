"""Post-hoc candidate attribution without making support judgements."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Mapping, Sequence
from enum import Enum
from hashlib import sha256
from typing import Any, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from open_deep_research.harness.claims import (
    AtomicClaim,
    CitationRequirement,
    MarkdownBlock,
    SourceResolution,
    source_inheritance_allowed,
)
from open_deep_research.harness.jsonio import loads_lenient
from open_deep_research.harness.notes import NoteLocationStatus, ResearchNote


class AttributionStatus(str, Enum):
    """Auditable outcome of candidate attribution for one claim."""

    CANDIDATE_SOURCES = "candidate_sources"
    CANDIDATE_SOURCES_WITH_ERRORS = "candidate_sources_with_errors"
    NO_CANDIDATE_SOURCE = "no_candidate_source"
    ATTRIBUTION_ERROR = "attribution_error"


class AttributionStopReason(str, Enum):
    """Reason the bounded attribution session ended."""

    COMPLETED = "completed"
    MALFORMED_RESPONSE = "malformed_response"
    MODEL_ROUND_LIMIT = "model_round_limit"
    CONTENT_RETRY_LIMIT = "content_retry_limit"


class AttributionSettings(BaseModel):
    """Paging and call-count limits for one attribution session."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    note_page_size: int = Field(default=8, ge=1, le=100)
    max_model_rounds: int = Field(default=20, ge=1, le=100)


class CandidateSource(BaseModel):
    """A source worth checking later, never a support verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    note_id: str
    source_id: str
    item_id: str
    publisher: str
    url: str
    location_status: NoteLocationStatus
    resolution: SourceResolution
    inherited_from_claim_id: str | None = None

    @model_validator(mode="after")
    def _resolution_matches_origin(self) -> CandidateSource:
        if self.resolution == SourceResolution.DIRECT:
            if self.inherited_from_claim_id is not None:
                raise ValueError("direct candidates cannot have an origin claim")
        elif (
            self.resolution != SourceResolution.UNRESOLVED
            and self.inherited_from_claim_id is None
        ):
            raise ValueError("inherited candidates require an origin claim")
        return self


class AttributionError(BaseModel):
    """One rejected or missing model attribution, retained for audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    code: str
    detail: str
    raw: Any = None


class ClaimAttribution(BaseModel):
    """Candidate relations and errors for one retained atomic claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: AtomicClaim
    status: AttributionStatus
    candidates: tuple[CandidateSource, ...] = ()
    errors: tuple[AttributionError, ...] = ()

    @model_validator(mode="after")
    def _status_matches_payload(self) -> ClaimAttribution:
        if self.status == AttributionStatus.NO_CANDIDATE_SOURCE:
            if self.candidates or self.errors:
                raise ValueError("no_candidate_source must be clean and empty")
        elif self.status == AttributionStatus.ATTRIBUTION_ERROR:
            if self.candidates or not self.errors:
                raise ValueError("attribution_error requires errors only")
        elif self.status == AttributionStatus.CANDIDATE_SOURCES:
            if not self.candidates or self.errors:
                raise ValueError("candidate_sources requires clean candidates")
        elif not self.candidates or not self.errors:
            raise ValueError(
                "candidate_sources_with_errors requires candidates and errors"
            )
        return self


class InspectedNotePage(BaseModel):
    """One model-requested full-note page and whether it was already present."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cursor: int = Field(ge=0)
    next_cursor: int | None = Field(default=None, ge=0)
    note_ids: tuple[str, ...]
    cache_hit: bool = False


class AttributionCallUsage(BaseModel):
    """Measured usage and action for one attribution-model call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    round_number: int = Field(ge=1)
    action: str
    token_count: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)


class AttributionResult(BaseModel):
    """Complete candidate-attribution output and paging audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attributions: tuple[ClaimAttribution, ...]
    inspected_pages: tuple[InspectedNotePage, ...] = ()
    usage: tuple[AttributionCallUsage, ...] = ()
    diagnostics: tuple[str, ...] = ()
    stop_reason: AttributionStopReason

    @property
    def total_tokens(self) -> int:
        """Return all measured attribution tokens."""

        return sum(record.token_count for record in self.usage)

    @property
    def total_cost_usd(self) -> float:
        """Return all measured attribution cost."""

        return sum(record.cost_usd for record in self.usage)


class AttributionModelClient(Protocol):
    """Injected model boundary used only for candidate attribution."""

    def generate(self, prompt: str) -> Any | Awaitable[Any]:
        """Return an action in a measured usage envelope."""


class _ModelEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    content: Any
    token_count: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)


class _InspectNotesAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["inspect_notes"]
    cursor: int = Field(ge=0)


class _CandidateProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    note_ref: str = Field(min_length=1)
    inherited_from_claim_id: str | None = None

    @field_validator("note_ref")
    @classmethod
    def _identifier_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("identifier must not be blank")
        return normalized

_ATTRIBUTION_PROMPT = """\
Perform post-hoc candidate attribution for a canonical report draft.

You can see every claim_text and a compact registry of every available note.
A candidate means only that the note's cached source is worth checking for the
claim in the later verification stage. It is not a finding that the source
supports, contradicts, proves, verifies, or grounds the claim. Do not return a
support verdict.

Publisher and URL are provenance context, not proof and not a mechanical
ranking. Use your semantic judgement about each source's relationship to the
claim; do not choose a note merely because its model-written finding repeats
the claim. A compound claim may need multiple candidate notes when distinct
parts require distinct sources. Request a full-note page when the compact
metadata is insufficient to make that choice.

Return json only. Every claim must appear exactly once in the final attribute
action. An empty candidates array is legal and means no candidate source was
identified.
Candidates may cross checklist item boundaries. Notes marked unlocatable are
still legal candidates: their source can be inspected later, but the note is
not evidence by itself.

The compact note registry is complete. Candidate identity is the opaque
note_ref shown in that registry. Persistent note_id and source_id values are
not model-owned fields and are derived by code from note_ref.
When compact metadata is insufficient, request one deterministic full-note
page using:
{{"action":"inspect_notes","cursor":0}}
Only the requested pages shown below contain full notes; no semantic top-N has
been selected for you.

When ready, return:
{{"action":"attribute","claims":[{{"claim_id":"claim-0001",\
"candidates":[{{"note_ref":"nref-example",\
"inherited_from_claim_id":null}}]}}]}}

When you match the current claim to a candidate itself, set
inherited_from_claim_id to null. The same note_ref may be listed this way for
any number of claims: repetition is direct matching, not inheritance. Do not
use inheritance to avoid repeating a note_ref.

Set inherited_from_claim_id only when your semantic judgement is that the
current claim borrows that exact note_ref through a local narrative
continuation from an earlier claim. Code, not you, derives the structural
source_resolution from the Markdown units. Invalid, backward, or nonlocal
lineage remains an unresolved candidate with an audit error; it is never
silently rewritten as direct.

All claims:
{claims}

Compact note registry:
{notes}

Paging:
{paging}

Full note pages requested so far:
{pages}

Protocol feedback from earlier calls:
{feedback}
"""


def _claim_registry(claims: Sequence[AtomicClaim]) -> list[dict[str, Any]]:
    return [
        {
            "claim_id": claim.claim_id,
            "block_id": claim.block_id,
            "claim_text": claim.claim_text,
            "citation_requirement": claim.citation_requirement.value,
            "normalization_status": claim.normalization_status.value,
        }
        for claim in claims
    ]


def _ordered_notes(notes: Sequence[ResearchNote]) -> tuple[ResearchNote, ...]:
    note_ids = [note.note_id for note in notes]
    if any(note_id is None for note_id in note_ids):
        raise ValueError("candidate attribution requires ledger-assigned note_id")
    if len(note_ids) != len(set(note_ids)):
        raise ValueError("candidate attribution requires unique note_id values")
    return tuple(sorted(notes, key=lambda note: str(note.note_id)))


def _note_reference(note: ResearchNote) -> str:
    """Return a stable prompt-local handle with no ordinal relation to claims."""

    if note.note_id is None:
        raise ValueError("candidate attribution requires ledger-assigned note_id")
    identity = f"{note.note_id}\0{note.source_id}".encode("utf-8")
    return f"nref-{sha256(identity).hexdigest()[:16]}"


def _note_reference_map(
    notes: Sequence[ResearchNote],
) -> dict[str, ResearchNote]:
    """Build and mechanically verify the model-visible note handle registry."""

    references: dict[str, ResearchNote] = {}
    for note in notes:
        note_ref = _note_reference(note)
        if note_ref in references:
            raise ValueError("candidate attribution note_ref collision")
        references[note_ref] = note
    return references


def _compact_note_registry(
    notes: Sequence[ResearchNote],
    *,
    page_size: int,
) -> list[dict[str, Any]]:
    return [
        {
            "note_ref": _note_reference(note),
            "item_id": note.item_id,
            "finding": note.finding,
            "publisher": note.publisher,
            "url": note.url,
            "location_status": note.location_status.value,
            "page_cursor": (index // page_size) * page_size,
        }
        for index, note in enumerate(notes)
    ]


def _full_note(note: ResearchNote) -> dict[str, Any]:
    payload = note.model_dump(mode="json")
    payload.pop("note_id", None)
    payload.pop("source_id", None)
    payload["note_ref"] = _note_reference(note)
    return payload


def build_attribution_prompt(
    claims: Sequence[AtomicClaim],
    notes: Sequence[ResearchNote],
    *,
    page_size: int,
    inspected_page_payloads: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
    feedback: Sequence[str] = (),
) -> str:
    """Build one attribution turn with all claims and compact note metadata."""

    ordered_notes = _ordered_notes(notes)
    pages = {
        str(cursor): list(payload)
        for cursor, payload in sorted((inspected_page_payloads or {}).items())
    }
    valid_cursors = list(range(0, len(ordered_notes), page_size))
    paging = {
        "note_count": len(ordered_notes),
        "page_size": page_size,
        "valid_cursors": valid_cursors,
    }
    return _ATTRIBUTION_PROMPT.format(
        claims=json.dumps(
            _claim_registry(claims), ensure_ascii=False, sort_keys=True
        ),
        notes=json.dumps(
            _compact_note_registry(ordered_notes, page_size=page_size),
            ensure_ascii=False,
            sort_keys=True,
        ),
        paging=json.dumps(paging, ensure_ascii=False, sort_keys=True),
        pages=json.dumps(pages, ensure_ascii=False, sort_keys=True),
        feedback=json.dumps(list(feedback), ensure_ascii=False),
    )


async def _generate(
    client: AttributionModelClient,
    prompt: str,
) -> tuple[Any, int, float]:
    response = client.generate(prompt)
    if inspect.isawaitable(response):
        response = await response
    try:
        envelope = _ModelEnvelope.model_validate(response)
    except ValidationError as exc:
        raise ValueError(
            "attribution model returned an invalid usage envelope"
        ) from exc
    content = envelope.content
    if isinstance(content, str):
        try:
            content = loads_lenient(content)
        except json.JSONDecodeError:
            # Preserve the paid malformed output for the caller's audit path.
            pass
    return content, envelope.token_count, envelope.cost_usd


def _error(
    claim_id: str,
    code: str,
    detail: str,
    raw: Any = None,
) -> AttributionError:
    return AttributionError(
        claim_id=claim_id,
        code=code,
        detail=detail,
        raw=raw,
    )


def _resolution_for(candidates: Sequence[CandidateSource]) -> SourceResolution:
    resolutions = {candidate.resolution for candidate in candidates}
    for resolution in (
        SourceResolution.DIRECT,
        SourceResolution.INHERITED_SAME_UNIT,
        SourceResolution.INHERITED_PREVIOUS_UNIT,
    ):
        if resolution in resolutions:
            return resolution
    return SourceResolution.UNRESOLVED


def _failed_attributions(
    claims: Sequence[AtomicClaim],
    *,
    code: str,
    detail: str,
    raw: Any = None,
) -> tuple[ClaimAttribution, ...]:
    return tuple(
        ClaimAttribution(
            claim=claim.model_copy(
                update={"source_resolution": SourceResolution.UNRESOLVED}
            ),
            status=AttributionStatus.ATTRIBUTION_ERROR,
            errors=(_error(claim.claim_id, code, detail, raw),),
        )
        for claim in claims
    )


def _non_external_attribution(claim: AtomicClaim) -> ClaimAttribution:
    """Return the mechanically empty attribution for a non-external claim."""

    return ClaimAttribution(
        claim=claim.model_copy(
            update={"source_resolution": SourceResolution.UNRESOLVED}
        ),
        status=AttributionStatus.NO_CANDIDATE_SOURCE,
    )


def _merge_attribution_scope(
    claims: Sequence[AtomicClaim],
    external_attributions: Sequence[ClaimAttribution],
) -> tuple[ClaimAttribution, ...]:
    """Restore original claim order after external-only model attribution."""

    by_claim_id = {
        attribution.claim.claim_id: attribution
        for attribution in external_attributions
    }
    return tuple(
        (
            by_claim_id[claim.claim_id]
            if claim.citation_requirement == CitationRequirement.EXTERNAL
            else _non_external_attribution(claim)
        )
        for claim in claims
    )


def _parse_final_attributions(
    content: Mapping[str, Any],
    *,
    claims: Sequence[AtomicClaim],
    blocks: Sequence[MarkdownBlock],
    notes: Sequence[ResearchNote],
    claim_context: Sequence[AtomicClaim] | None = None,
    existing_attributions: Sequence[ClaimAttribution] = (),
) -> tuple[tuple[ClaimAttribution, ...], tuple[str, ...]]:
    scope_claim_by_id = {claim.claim_id: claim for claim in claims}
    contextual_claims = tuple(claim_context or claims)
    claim_by_id = {claim.claim_id: claim for claim in contextual_claims}
    claim_order = {
        claim.claim_id: index for index, claim in enumerate(contextual_claims)
    }
    note_by_ref = _note_reference_map(notes)
    diagnostics: list[str] = []

    raw_claims = content.get("claims")
    if not isinstance(raw_claims, (list, tuple)):
        return _failed_attributions(
            claims,
            code="malformed_claim_attributions",
            detail="attribute action claims must be an array",
            raw=raw_claims,
        ), ("attribute action claims was not an array",)

    entries: dict[str, Mapping[str, Any]] = {}
    duplicate_claim_ids: set[str] = set()
    entry_errors: dict[str, list[AttributionError]] = {
        claim.claim_id: [] for claim in claims
    }
    for index, raw_claim in enumerate(raw_claims):
        if not isinstance(raw_claim, Mapping):
            diagnostics.append(f"attribution_entry_invalid[{index}]")
            continue
        claim_id = raw_claim.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id.strip():
            diagnostics.append(f"attribution_entry_missing_claim_id[{index}]")
            continue
        claim_id = claim_id.strip()
        if claim_id not in scope_claim_by_id:
            diagnostics.append(f"attribution_out_of_scope_claim: {claim_id}")
            continue
        if claim_id in entries:
            duplicate_claim_ids.add(claim_id)
            entries.pop(claim_id, None)
            continue
        if claim_id in duplicate_claim_ids:
            continue
        entries[claim_id] = raw_claim

    proposals: dict[str, list[_CandidateProposal]] = {
        claim.claim_id: [] for claim in claims
    }
    explicit_empty: set[str] = set()
    for claim in claims:
        claim_id = claim.claim_id
        if claim_id in duplicate_claim_ids:
            entry_errors[claim_id].append(
                _error(
                    claim_id,
                    "duplicate_claim_attribution",
                    "claim appeared more than once in the attribute action",
                )
            )
            continue
        entry = entries.get(claim_id)
        if entry is None:
            entry_errors[claim_id].append(
                _error(
                    claim_id,
                    "missing_claim_attribution",
                    "claim was omitted from the attribute action",
                )
            )
            continue
        raw_candidates = entry.get("candidates")
        if not isinstance(raw_candidates, (list, tuple)):
            entry_errors[claim_id].append(
                _error(
                    claim_id,
                    "malformed_candidates",
                    "candidates must be an array",
                    raw_candidates,
                )
            )
            continue
        if not raw_candidates:
            explicit_empty.add(claim_id)
            continue
        for index, raw_candidate in enumerate(raw_candidates):
            if (
                isinstance(raw_candidate, Mapping)
                and "note_ref" not in raw_candidate
                and "note_id" in raw_candidate
            ):
                entry_errors[claim_id].append(
                    _error(
                        claim_id,
                        "persistent_note_id_not_accepted",
                        (
                            "candidate identity must use a note_ref from the "
                            "model-visible registry; persistent note_id is "
                            "code-owned"
                        ),
                        raw_candidate,
                    )
                )
                continue
            try:
                proposal = _CandidateProposal.model_validate(raw_candidate)
            except (TypeError, ValidationError, ValueError) as exc:
                entry_errors[claim_id].append(
                    _error(
                        claim_id,
                        "malformed_candidate",
                        f"candidate {index} could not be parsed: {exc}",
                        raw_candidate,
                    )
                )
                continue
            proposals[claim_id].append(proposal)

    valid_identity: dict[str, list[tuple[_CandidateProposal, ResearchNote]]] = {
        claim.claim_id: [] for claim in claims
    }
    for claim in claims:
        claim_id = claim.claim_id
        seen_relations: set[tuple[str, str, str | None]] = set()
        for proposal in proposals[claim_id]:
            note = note_by_ref.get(proposal.note_ref)
            if note is None:
                entry_errors[claim_id].append(
                    _error(
                        claim_id,
                        "unknown_note_ref",
                        f"note_ref does not exist: {proposal.note_ref}",
                        proposal.model_dump(mode="json"),
                    )
                )
                continue
            note_id = str(note.note_id)
            relation_key = (
                note_id,
                note.source_id,
                proposal.inherited_from_claim_id,
            )
            if relation_key in seen_relations:
                entry_errors[claim_id].append(
                    _error(
                        claim_id,
                        "duplicate_candidate_relation",
                        "candidate relation was repeated",
                        proposal.model_dump(mode="json"),
                    )
                )
                continue
            seen_relations.add(relation_key)
            valid_identity[claim_id].append((proposal, note))

    accepted: dict[str, list[CandidateSource]] = {
        claim.claim_id: [] for claim in claims
    }
    direct_pairs: dict[str, set[tuple[str, str]]] = {
        claim.claim_id: set() for claim in contextual_claims
    }
    for attribution in existing_attributions:
        direct_pairs.setdefault(attribution.claim.claim_id, set()).update(
            (candidate.note_id, candidate.source_id)
            for candidate in attribution.candidates
            if candidate.inherited_from_claim_id is None
        )
    for claim in claims:
        claim_id = claim.claim_id
        for proposal, note in valid_identity[claim_id]:
            if proposal.inherited_from_claim_id is not None:
                continue
            note_id = str(note.note_id)
            direct_pairs[claim_id].add((note_id, note.source_id))
            accepted[claim_id].append(
                CandidateSource(
                    note_id=note_id,
                    source_id=note.source_id,
                    item_id=note.item_id,
                    publisher=note.publisher,
                    url=note.url,
                    location_status=note.location_status,
                    resolution=SourceResolution.DIRECT,
                )
            )

    for claim in claims:
        claim_id = claim.claim_id
        for proposal, note in valid_identity[claim_id]:
            if proposal.inherited_from_claim_id is None:
                continue
            origin_id = proposal.inherited_from_claim_id
            resolution = SourceResolution.UNRESOLVED
            if origin_id is None or origin_id not in claim_by_id:
                entry_errors[claim_id].append(
                    _error(
                        claim_id,
                        "invalid_inheritance_origin",
                        "inherited candidate requires an existing origin claim",
                        proposal.model_dump(mode="json"),
                    )
                )
            elif claim_order[origin_id] >= claim_order[claim_id]:
                entry_errors[claim_id].append(
                    _error(
                        claim_id,
                        "backward_inheritance",
                        "inheritance origin must precede the target claim",
                        proposal.model_dump(mode="json"),
                    )
                )
            elif (
                str(note.note_id),
                note.source_id,
            ) not in direct_pairs[origin_id]:
                entry_errors[claim_id].append(
                    _error(
                        claim_id,
                        "inheritance_origin_not_direct",
                        "origin must own the exact pair as a direct candidate",
                        proposal.model_dump(mode="json"),
                    )
                )
            else:
                origin = claim_by_id[origin_id]
                if source_inheritance_allowed(
                    blocks,
                    source_block_id=origin.block_id,
                    target_block_id=claim.block_id,
                    resolution=SourceResolution.INHERITED_SAME_UNIT,
                ):
                    resolution = SourceResolution.INHERITED_SAME_UNIT
                elif source_inheritance_allowed(
                    blocks,
                    source_block_id=origin.block_id,
                    target_block_id=claim.block_id,
                    resolution=SourceResolution.INHERITED_PREVIOUS_UNIT,
                ):
                    resolution = SourceResolution.INHERITED_PREVIOUS_UNIT
                else:
                    entry_errors[claim_id].append(
                        _error(
                            claim_id,
                            "lineage_outside_markdown_boundary",
                            (
                                "candidate retained unresolved because the "
                                "claimed lineage crosses the mechanical "
                                "Markdown boundary"
                            ),
                            proposal.model_dump(mode="json"),
                        )
                    )
            accepted[claim_id].append(
                CandidateSource(
                    note_id=str(note.note_id),
                    source_id=note.source_id,
                    item_id=note.item_id,
                    publisher=note.publisher,
                    url=note.url,
                    location_status=note.location_status,
                    resolution=resolution,
                    inherited_from_claim_id=origin_id,
                )
            )

    result: list[ClaimAttribution] = []
    for claim in claims:
        claim_id = claim.claim_id
        candidates = tuple(accepted[claim_id])
        errors = tuple(entry_errors[claim_id])
        updated_claim = claim.model_copy(
            update={"source_resolution": _resolution_for(candidates)}
        )
        if candidates and errors:
            status = AttributionStatus.CANDIDATE_SOURCES_WITH_ERRORS
        elif candidates:
            status = AttributionStatus.CANDIDATE_SOURCES
        elif errors:
            status = AttributionStatus.ATTRIBUTION_ERROR
        else:
            if claim_id not in explicit_empty:
                errors = (
                    _error(
                        claim_id,
                        "no_candidate_result",
                        "no valid candidate result was returned",
                    ),
                )
                status = AttributionStatus.ATTRIBUTION_ERROR
            else:
                status = AttributionStatus.NO_CANDIDATE_SOURCE
        result.append(
            ClaimAttribution(
                claim=updated_claim,
                status=status,
                candidates=candidates,
                errors=errors,
            )
        )
    return tuple(result), tuple(diagnostics)


def _ordered_attributions(
    claims: Sequence[AtomicClaim],
    by_claim_id: Mapping[str, ClaimAttribution],
) -> tuple[ClaimAttribution, ...]:
    """Restore the complete external-claim order after scoped retries."""

    missing = [
        claim.claim_id
        for claim in claims
        if claim.claim_id not in by_claim_id
    ]
    if missing:
        raise AssertionError(
            f"attribution result missing claims after retry: {missing}"
        )
    return tuple(by_claim_id[claim.claim_id] for claim in claims)


def _content_retry_feedback(
    rejected: Sequence[ClaimAttribution],
    *,
    valid_note_refs: Sequence[str],
) -> str:
    """Describe a mechanical rejection without re-running accepted claims."""

    payload = {
        "retry_claim_ids": [entry.claim.claim_id for entry in rejected],
        "rejected_candidates": [
            {
                "claim_id": entry.claim.claim_id,
                "errors": [
                    {
                        "code": error.code,
                        "detail": error.detail,
                        "raw": error.raw,
                    }
                    for error in entry.errors
                ],
            }
            for entry in rejected
        ],
        "valid_note_refs": list(valid_note_refs),
    }
    return (
        "The previous attribute action contained mechanically rejected "
        "content. Retry exactly retry_claim_ids; accepted claims are frozen "
        "and must not be returned again. Candidate identity must use one "
        "note_ref from valid_note_refs. "
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


async def attribute_claims(
    claims: Sequence[AtomicClaim],
    *,
    blocks: Sequence[MarkdownBlock],
    notes: Sequence[ResearchNote],
    model_client: AttributionModelClient,
    settings: AttributionSettings | None = None,
) -> AttributionResult:
    """Let a model page notes and propose mechanically validated candidates."""

    active_settings = settings or AttributionSettings()
    all_claims = tuple(claims)
    claim_ids = [claim.claim_id for claim in all_claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("candidate attribution requires unique claim_id values")
    external_claims = tuple(
        claim
        for claim in all_claims
        if claim.citation_requirement == CitationRequirement.EXTERNAL
    )
    if not external_claims:
        return AttributionResult(
            attributions=_merge_attribution_scope(all_claims, ()),
            stop_reason=AttributionStopReason.COMPLETED,
        )
    ordered_notes = _ordered_notes(notes)
    inspected_payloads: dict[int, list[dict[str, Any]]] = {}
    inspected_pages: list[InspectedNotePage] = []
    usage: list[AttributionCallUsage] = []
    feedback: list[str] = []
    diagnostics: list[str] = []
    completed: dict[str, ClaimAttribution] = {}
    pending_claims = external_claims
    external_by_id = {
        claim.claim_id: claim for claim in external_claims
    }
    valid_note_refs = tuple(_note_reference_map(ordered_notes))

    for round_number in range(1, active_settings.max_model_rounds + 1):
        content, tokens, cost = await _generate(
            model_client,
            build_attribution_prompt(
                pending_claims,
                ordered_notes,
                page_size=active_settings.note_page_size,
                inspected_page_payloads=inspected_payloads,
                feedback=feedback,
            ),
        )
        action_name = (
            str(content.get("action"))
            if isinstance(content, Mapping)
            else "malformed"
        )
        usage.append(
            AttributionCallUsage(
                round_number=round_number,
                action=action_name,
                token_count=tokens,
                cost_usd=cost,
            )
        )
        if not isinstance(content, Mapping):
            failed = _failed_attributions(
                pending_claims,
                code="malformed_attribution_response",
                detail="model response was not a JSON object",
                raw=content,
            )
            combined = dict(completed)
            combined.update(
                (entry.claim.claim_id, entry) for entry in failed
            )
            return AttributionResult(
                attributions=_merge_attribution_scope(
                    all_claims,
                    _ordered_attributions(external_claims, combined),
                ),
                inspected_pages=tuple(inspected_pages),
                usage=tuple(usage),
                diagnostics=tuple(
                    [*diagnostics, "model response was not a JSON object"]
                ),
                stop_reason=AttributionStopReason.MALFORMED_RESPONSE,
            )

        if content.get("action") == "inspect_notes":
            try:
                action = _InspectNotesAction.model_validate(content)
            except ValidationError as exc:
                message = f"invalid inspect_notes action: {exc}"
                feedback.append(message)
                diagnostics.append(message)
                continue
            cursor = action.cursor
            valid_cursor = (
                cursor < len(ordered_notes)
                and cursor % active_settings.note_page_size == 0
            )
            if not valid_cursor:
                message = f"invalid note page cursor: {cursor}"
                feedback.append(message)
                diagnostics.append(message)
                continue
            page = ordered_notes[
                cursor : cursor + active_settings.note_page_size
            ]
            cache_hit = cursor in inspected_payloads
            if not cache_hit:
                inspected_payloads[cursor] = [_full_note(note) for note in page]
            next_cursor = cursor + active_settings.note_page_size
            if next_cursor >= len(ordered_notes):
                next_cursor = None
            inspected_pages.append(
                InspectedNotePage(
                    cursor=cursor,
                    next_cursor=next_cursor,
                    note_ids=tuple(str(note.note_id) for note in page),
                    cache_hit=cache_hit,
                )
            )
            continue

        if content.get("action") == "attribute":
            attributions, parse_diagnostics = _parse_final_attributions(
                content,
                claims=pending_claims,
                blocks=blocks,
                notes=ordered_notes,
                claim_context=external_claims,
                existing_attributions=tuple(completed.values()),
            )
            diagnostics.extend(parse_diagnostics)
            rejected = tuple(
                entry
                for entry in attributions
                if entry.status == AttributionStatus.ATTRIBUTION_ERROR
            )
            for entry in attributions:
                if entry.status != AttributionStatus.ATTRIBUTION_ERROR:
                    completed[entry.claim.claim_id] = entry
            if not rejected:
                return AttributionResult(
                    attributions=_merge_attribution_scope(
                        all_claims,
                        _ordered_attributions(external_claims, completed),
                    ),
                    inspected_pages=tuple(inspected_pages),
                    usage=tuple(usage),
                    diagnostics=tuple(diagnostics),
                    stop_reason=AttributionStopReason.COMPLETED,
                )

            retry_feedback = _content_retry_feedback(
                rejected,
                valid_note_refs=valid_note_refs,
            )
            diagnostics.append(retry_feedback)
            if round_number == active_settings.max_model_rounds:
                combined = dict(completed)
                combined.update(
                    (entry.claim.claim_id, entry) for entry in rejected
                )
                return AttributionResult(
                    attributions=_merge_attribution_scope(
                        all_claims,
                        _ordered_attributions(external_claims, combined),
                    ),
                    inspected_pages=tuple(inspected_pages),
                    usage=tuple(usage),
                    diagnostics=tuple(diagnostics),
                    stop_reason=AttributionStopReason.CONTENT_RETRY_LIMIT,
                )
            pending_claims = tuple(
                external_by_id[entry.claim.claim_id]
                for entry in rejected
            )
            feedback = [retry_feedback]
            continue

        message = "action must be inspect_notes or attribute"
        feedback.append(message)
        diagnostics.append(message)

    failed = _failed_attributions(
        pending_claims,
        code="attribution_round_limit",
        detail="model did not return a final attribute action",
    )
    combined = dict(completed)
    combined.update((entry.claim.claim_id, entry) for entry in failed)
    return AttributionResult(
        attributions=_merge_attribution_scope(
            all_claims,
            _ordered_attributions(external_claims, combined),
        ),
        inspected_pages=tuple(inspected_pages),
        usage=tuple(usage),
        diagnostics=tuple(diagnostics),
        stop_reason=AttributionStopReason.MODEL_ROUND_LIMIT,
    )
