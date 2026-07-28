"""Post-hoc candidate attribution without making support judgements."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Mapping, Sequence
from enum import Enum
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

    note_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    inherited_from_claim_id: str | None = None

    @field_validator("note_id", "source_id")
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

Return json only. Every claim must appear exactly once in the final attribute
action. An empty candidates array is legal and means no candidate source was
identified.
Candidates may cross checklist item boundaries. Notes marked unlocatable are
still legal candidates: their source can be inspected later, but the note is
not evidence by itself.

The compact note registry is complete. Do not invent note_id or source_id.
When compact metadata is insufficient, request one deterministic full-note
page using:
{{"action":"inspect_notes","cursor":0}}
Only the requested pages shown below contain full notes; no semantic top-N has
been selected for you.

When ready, return:
{{"action":"attribute","claims":[{{"claim_id":"claim-0001",\
"candidates":[{{"note_id":"note-000001","source_id":"source-id",\
"inherited_from_claim_id":null}}]}}]}}

When you match the current claim to a candidate itself, set
inherited_from_claim_id to null. The same note_id/source_id pair may be listed
this way for any number of claims: repetition is direct matching, not
inheritance. Do not use inheritance to avoid repeating a pair.

Set inherited_from_claim_id only when your semantic judgement is that the
current claim borrows that exact candidate pair through a local narrative
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


def _compact_note_registry(
    notes: Sequence[ResearchNote],
    *,
    page_size: int,
) -> list[dict[str, Any]]:
    return [
        {
            "note_id": note.note_id,
            "source_id": note.source_id,
            "item_id": note.item_id,
            "finding": note.finding,
            "publisher": note.publisher,
            "location_status": note.location_status.value,
            "page_cursor": (index // page_size) * page_size,
        }
        for index, note in enumerate(notes)
    ]


def _full_note(note: ResearchNote) -> dict[str, Any]:
    return note.model_dump(mode="json")


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


def _parse_final_attributions(
    content: Mapping[str, Any],
    *,
    claims: Sequence[AtomicClaim],
    blocks: Sequence[MarkdownBlock],
    notes: Sequence[ResearchNote],
) -> tuple[tuple[ClaimAttribution, ...], tuple[str, ...]]:
    claim_by_id = {claim.claim_id: claim for claim in claims}
    claim_order = {claim.claim_id: index for index, claim in enumerate(claims)}
    note_by_id = {str(note.note_id): note for note in notes}
    known_source_ids = {note.source_id for note in notes}
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
        if claim_id not in claim_by_id:
            diagnostics.append(f"attribution_unknown_claim: {claim_id}")
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
            note = note_by_id.get(proposal.note_id)
            identity_valid = True
            if note is None:
                entry_errors[claim_id].append(
                    _error(
                        claim_id,
                        "unknown_note_id",
                        f"note_id does not exist: {proposal.note_id}",
                        proposal.model_dump(mode="json"),
                    )
                )
                identity_valid = False
            if proposal.source_id not in known_source_ids:
                entry_errors[claim_id].append(
                    _error(
                        claim_id,
                        "unknown_source_id",
                        f"source_id does not exist: {proposal.source_id}",
                        proposal.model_dump(mode="json"),
                    )
                )
                identity_valid = False
            if not identity_valid:
                continue
            if note.source_id != proposal.source_id:
                entry_errors[claim_id].append(
                    _error(
                        claim_id,
                        "note_source_mismatch",
                        "note_id is not owned by the proposed source_id",
                        proposal.model_dump(mode="json"),
                    )
                )
                continue
            relation_key = (
                proposal.note_id,
                proposal.source_id,
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
        claim.claim_id: set() for claim in claims
    }
    for claim in claims:
        claim_id = claim.claim_id
        for proposal, note in valid_identity[claim_id]:
            if proposal.inherited_from_claim_id is not None:
                continue
            direct_pairs[claim_id].add((proposal.note_id, proposal.source_id))
            accepted[claim_id].append(
                CandidateSource(
                    note_id=proposal.note_id,
                    source_id=proposal.source_id,
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
                proposal.note_id,
                proposal.source_id,
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
                    note_id=proposal.note_id,
                    source_id=proposal.source_id,
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
    claim_ids = [claim.claim_id for claim in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("candidate attribution requires unique claim_id values")
    ordered_notes = _ordered_notes(notes)
    inspected_payloads: dict[int, list[dict[str, Any]]] = {}
    inspected_pages: list[InspectedNotePage] = []
    usage: list[AttributionCallUsage] = []
    feedback: list[str] = []

    for round_number in range(1, active_settings.max_model_rounds + 1):
        content, tokens, cost = await _generate(
            model_client,
            build_attribution_prompt(
                claims,
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
            return AttributionResult(
                attributions=_failed_attributions(
                    claims,
                    code="malformed_attribution_response",
                    detail="model response was not a JSON object",
                    raw=content,
                ),
                inspected_pages=tuple(inspected_pages),
                usage=tuple(usage),
                diagnostics=("model response was not a JSON object",),
                stop_reason=AttributionStopReason.MALFORMED_RESPONSE,
            )

        if content.get("action") == "inspect_notes":
            try:
                action = _InspectNotesAction.model_validate(content)
            except ValidationError as exc:
                feedback.append(f"invalid inspect_notes action: {exc}")
                continue
            cursor = action.cursor
            valid_cursor = (
                cursor < len(ordered_notes)
                and cursor % active_settings.note_page_size == 0
            )
            if not valid_cursor:
                feedback.append(f"invalid note page cursor: {cursor}")
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
            attributions, diagnostics = _parse_final_attributions(
                content,
                claims=claims,
                blocks=blocks,
                notes=ordered_notes,
            )
            return AttributionResult(
                attributions=attributions,
                inspected_pages=tuple(inspected_pages),
                usage=tuple(usage),
                diagnostics=diagnostics,
                stop_reason=AttributionStopReason.COMPLETED,
            )

        feedback.append(
            "action must be inspect_notes or attribute"
        )

    return AttributionResult(
        attributions=_failed_attributions(
            claims,
            code="attribution_round_limit",
            detail="model did not return a final attribute action",
        ),
        inspected_pages=tuple(inspected_pages),
        usage=tuple(usage),
        diagnostics=tuple(feedback),
        stop_reason=AttributionStopReason.MODEL_ROUND_LIMIT,
    )
