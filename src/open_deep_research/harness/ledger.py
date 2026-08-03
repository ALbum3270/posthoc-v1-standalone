"""Serializable audit ledger and immutable-by-URL source cache."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from open_deep_research.harness.notes import ResearchNote


class SourceLinkCaptureStatus(str, Enum):
    """Mechanical outcome of the optional Markdown link sidecar request."""

    CAPTURED = "captured"
    NO_LINKS_CAPTURED = "no_links_captured"
    NO_MARKDOWN_CONTENT = "no_markdown_content"
    PROVIDER_ERROR = "provider_error"


class SourceLinkRecord(BaseModel):
    """One HTTP(S) link mechanically parsed from provider Markdown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_url: str = Field(min_length=1)
    label: str = ""

    @model_validator(mode="after")
    def _target_is_http_url(self) -> SourceLinkRecord:
        parsed = urlsplit(self.target_url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError("source link target must be an absolute HTTP(S) URL")
        return self


class SourceLinkCaptureAudit(BaseModel):
    """Audit the best-effort sidecar without claiming link completeness."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: SourceLinkCaptureStatus
    requested_format: str = "markdown"
    captured_link_count: int = Field(default=0, ge=0)
    completeness_guaranteed: bool = False
    ordering_basis: Literal[
        "provider_markdown_document_order_first_occurrence",
        "not_applicable_no_captured_links",
        "legacy_unspecified",
    ] = "legacy_unspecified"
    error: str | None = None

    @model_validator(mode="after")
    def _capture_count_matches_status(self) -> SourceLinkCaptureAudit:
        if (
            self.status is SourceLinkCaptureStatus.CAPTURED
            and self.captured_link_count == 0
        ):
            raise ValueError("captured status requires at least one link")
        if (
            self.status is not SourceLinkCaptureStatus.CAPTURED
            and self.captured_link_count != 0
        ):
            raise ValueError("non-captured status requires zero links")
        if self.completeness_guaranteed:
            raise ValueError("provider Markdown link completeness is not guaranteed")
        if (
            self.status is SourceLinkCaptureStatus.CAPTURED
            and self.ordering_basis == "not_applicable_no_captured_links"
        ):
            raise ValueError("captured links require an applicable ordering basis")
        if (
            self.status is not SourceLinkCaptureStatus.CAPTURED
            and self.ordering_basis
            == "provider_markdown_document_order_first_occurrence"
        ):
            raise ValueError("empty link captures cannot claim document ordering")
        if (
            self.status is SourceLinkCaptureStatus.PROVIDER_ERROR
            and not self.error
        ):
            raise ValueError("provider_error status requires an error message")
        if (
            self.status is not SourceLinkCaptureStatus.PROVIDER_ERROR
            and self.error is not None
        ):
            raise ValueError("only provider_error status may carry an error")
        return self


class SettlementEvidence(BaseModel):
    """Evidence visible when the model settled one checklist item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strict_locatable_notes: int = Field(default=0, ge=0)
    repaired_locatable_notes: int = Field(default=0, ge=0)
    located_notes: int = Field(default=0, ge=0)
    publisher_count: int = Field(default=0, ge=0)
    publishers: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _derived_counts_are_consistent(self) -> SettlementEvidence:
        if self.located_notes != (
            self.strict_locatable_notes + self.repaired_locatable_notes
        ):
            raise ValueError(
                "located_notes must equal strict plus repaired note counts"
            )
        if self.publisher_count != len(self.publishers):
            raise ValueError("publisher_count must equal the publisher list length")
        return self


class DismissedCandidateSnapshot(BaseModel):
    """One candidate the model explicitly declined before exhaustion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class UnreadableCandidateSnapshot(BaseModel):
    """One candidate whose attempted read ended in a mechanical tool error."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str = Field(min_length=1)
    error: str = Field(min_length=1)


class ExhaustionAttemptSnapshot(BaseModel):
    """Item-attributed collection history frozen at an exhaustion judgement.

    This is procedural evidence that collection was actually attempted.  It
    does not judge whether the attempts were sufficient or the sources useful.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    search_attempts: int = Field(default=0, ge=0)
    search_successes: int = Field(default=0, ge=0)
    search_errors: int = Field(default=0, ge=0)
    read_attempts: int = Field(default=0, ge=0)
    read_successes: int = Field(default=0, ge=0)
    read_errors: int = Field(default=0, ge=0)
    reanalyze_attempts: int = Field(default=0, ge=0)
    reanalyze_successes: int = Field(default=0, ge=0)
    reanalyze_errors: int = Field(default=0, ge=0)
    surfaced_candidate_urls: tuple[str, ...] = ()
    read_urls: tuple[str, ...] = ()
    dismissed_candidates: tuple[DismissedCandidateSnapshot, ...] = ()
    unreadable_candidates: tuple[UnreadableCandidateSnapshot, ...] = ()
    pending_unread_urls: tuple[str, ...] = ()
    note_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _attempt_outcomes_are_consistent(self) -> ExhaustionAttemptSnapshot:
        for action in ("search", "read", "reanalyze"):
            attempts = getattr(self, f"{action}_attempts")
            successes = getattr(self, f"{action}_successes")
            errors = getattr(self, f"{action}_errors")
            if attempts != successes + errors:
                raise ValueError(
                    f"{action}_attempts must equal successes plus errors"
                )
        return self

    @property
    def qualifying_attempts(self) -> int:
        """Return real acquisition attempts, independent of their outcome."""

        return (
            self.search_attempts
            + self.read_attempts
            + self.reanalyze_attempts
        )

    @property
    def has_qualifying_attempt(self) -> bool:
        """Return whether the mechanical exhaustion precondition is met."""

        return self.qualifying_attempts > 0


class RoundRecord(BaseModel):
    """One research-loop action and its measured usage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    round_number: int = Field(ge=1)
    action: str = Field(min_length=1)
    item_id: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    query: str | None = None
    url: str | None = None
    result_summary: str = ""
    token_count: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)


class ChecklistChangeRecord(BaseModel):
    """One accepted or rejected checklist mutation request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    accepted: bool
    reason: str = ""
    from_status: str | None = None
    to_status: str | None = None
    settlement_evidence: SettlementEvidence | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    exhaustion_attempts: ExhaustionAttemptSnapshot | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class EvidenceGapLedgerRecord(BaseModel):
    """One post-draft ledger mutation, separate from initial collection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event: str = Field(min_length=1)
    result_summary: str = ""
    url: str | None = None
    note_ids: tuple[str, ...] = ()


class ResearchLedger(BaseModel):
    """All durable evidence and audit events for one research run."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    research_id: str = ""
    topic: str = ""
    rounds: list[RoundRecord] = Field(default_factory=list)
    source_cache: dict[str, str] = Field(default_factory=dict)
    source_links: dict[str, tuple[SourceLinkRecord, ...]] = Field(
        default_factory=dict
    )
    source_link_capture: dict[str, SourceLinkCaptureAudit] = Field(
        default_factory=dict
    )
    notes: list[ResearchNote] = Field(default_factory=list)
    checklist_history: list[ChecklistChangeRecord] = Field(default_factory=list)
    evidence_gap_history: list[EvidenceGapLedgerRecord] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def _assign_missing_note_ids(self) -> ResearchLedger:
        """Give legacy serialized notes stable ledger-local identifiers."""

        explicit_ids = [
            note.note_id for note in self.notes if note.note_id is not None
        ]
        if len(explicit_ids) != len(set(explicit_ids)):
            raise ValueError("ledger note_id values must be unique")

        used_ids = set(explicit_ids)
        normalized_notes: list[ResearchNote] = []
        next_number = 1
        for note in self.notes:
            if note.note_id is None:
                note_id = f"note-{next_number:06d}"
                while note_id in used_ids:
                    next_number += 1
                    note_id = f"note-{next_number:06d}"
                note = note.model_copy(update={"note_id": note_id})
                used_ids.add(note_id)
            normalized_notes.append(note)
            next_number += 1

        self.notes[:] = normalized_notes
        return self

    @model_validator(mode="after")
    def _source_link_sidecars_match_cache(self) -> ResearchLedger:
        """Reject orphaned or internally inconsistent sidecar metadata."""

        sidecar_urls = set(self.source_links) | set(self.source_link_capture)
        orphaned = sidecar_urls - set(self.source_cache)
        if orphaned:
            raise ValueError(
                "source link sidecars require cached source text: "
                + ", ".join(sorted(orphaned))
            )
        if set(self.source_links) != set(self.source_link_capture):
            raise ValueError(
                "source links and capture audits must have identical URL keys"
            )
        for url, capture in self.source_link_capture.items():
            if capture.captured_link_count != len(self.source_links[url]):
                raise ValueError(
                    f"source link capture count mismatch for URL: {url}"
                )
        return self

    def record_round(
        self,
        *,
        round_number: int,
        action: str,
        item_id: str | None = None,
        query: str | None = None,
        url: str | None = None,
        result_summary: str = "",
        token_count: int = 0,
        cost_usd: float = 0.0,
    ) -> RoundRecord:
        """Append and return one round record."""

        record = RoundRecord(
            round_number=round_number,
            action=action,
            item_id=item_id,
            query=query,
            url=url,
            result_summary=result_summary,
            token_count=token_count,
            cost_usd=cost_usd,
        )
        self.rounds.append(record)
        return record

    def cache_source(
        self,
        url: str,
        cleaned_text: str,
        *,
        source_links: tuple[SourceLinkRecord, ...] | None = None,
        link_capture: SourceLinkCaptureAudit | None = None,
    ) -> bool:
        """Cache canonical text and optional link sidecar without drift.

        The canonical text has exactly the historical ``clean_text`` contract.
        Link records are separate provider metadata and never rewrite those
        bytes.  Missing sidecar fields remain valid for historical audits.
        """

        normalized_url = url.strip()
        if not normalized_url:
            raise ValueError("source URL must not be blank")
        existing = self.source_cache.get(normalized_url)
        if existing is not None:
            if existing != cleaned_text:
                raise ValueError(f"cached source changed for URL: {normalized_url}")
        normalized_links = self._validate_source_link_sidecar(
            normalized_url,
            source_links=source_links,
            link_capture=link_capture,
        )
        # All input and consistency checks precede mutation.  A rejected
        # sidecar must not leave canonical text or only one sidecar map behind.
        if existing is None:
            self.source_cache[normalized_url] = cleaned_text
        if normalized_links is not None:
            self.source_links[normalized_url] = normalized_links
            self.source_link_capture[normalized_url] = link_capture
        return existing is None

    def _validate_source_link_sidecar(
        self,
        url: str,
        *,
        source_links: tuple[SourceLinkRecord, ...] | None,
        link_capture: SourceLinkCaptureAudit | None,
    ) -> tuple[SourceLinkRecord, ...] | None:
        if (source_links is None) is not (link_capture is None):
            raise ValueError(
                "source links and link capture audit must be cached together"
            )
        if source_links is None:
            return None
        normalized_links = tuple(source_links)
        if link_capture.captured_link_count != len(normalized_links):
            raise ValueError(
                "link capture count must equal cached source link count"
            )
        existing_links = self.source_links.get(url)
        if existing_links is not None and existing_links != normalized_links:
            raise ValueError(f"cached source links changed for URL: {url}")
        existing_capture = self.source_link_capture.get(url)
        if existing_capture is not None and existing_capture != link_capture:
            raise ValueError(f"source link capture changed for URL: {url}")
        return normalized_links

    def get_source(self, url: str) -> str | None:
        """Return cached source text without refetching it."""

        return self.source_cache.get(url.strip())

    def add_note(self, note: ResearchNote) -> ResearchNote:
        """Retain a note regardless of quote-location status."""

        if note.note_id is None:
            used_ids = {
                candidate.note_id
                for candidate in self.notes
                if candidate.note_id is not None
            }
            number = len(self.notes) + 1
            note_id = f"note-{number:06d}"
            while note_id in used_ids:
                number += 1
                note_id = f"note-{number:06d}"
            note = note.model_copy(update={"note_id": note_id})
        elif any(candidate.note_id == note.note_id for candidate in self.notes):
            raise ValueError(f"duplicate note_id: {note.note_id}")
        self.notes.append(note)
        return note

    def record_checklist_change(
        self,
        *,
        event: str,
        item_id: str,
        accepted: bool,
        reason: str,
        from_status: str | None = None,
        to_status: str | None = None,
        settlement_evidence: SettlementEvidence | dict[str, Any] | None = None,
        exhaustion_attempts: (
            ExhaustionAttemptSnapshot | dict[str, Any] | None
        ) = None,
    ) -> ChecklistChangeRecord:
        """Append and return one checklist audit record."""

        record = ChecklistChangeRecord(
            event=event,
            item_id=item_id,
            accepted=accepted,
            reason=reason,
            from_status=from_status,
            to_status=to_status,
            settlement_evidence=settlement_evidence,
            exhaustion_attempts=exhaustion_attempts,
        )
        self.checklist_history.append(record)
        return record

    def record_evidence_gap(
        self,
        *,
        event: str,
        result_summary: str = "",
        url: str | None = None,
        note_ids: tuple[str, ...] = (),
    ) -> EvidenceGapLedgerRecord:
        """Append a post-draft event without changing collection rounds."""

        record = EvidenceGapLedgerRecord(
            event=event,
            result_summary=result_summary,
            url=url,
            note_ids=note_ids,
        )
        self.evidence_gap_history.append(record)
        return record

    def note_ids_for_url(self, url: str) -> dict[str, tuple[str, ...]]:
        """Return deterministic ledger-local note IDs grouped by checklist item.

        Stable note/source identifiers become first-class data in the later
        paging step. Until then, these IDs let a cache hit refer to existing
        notes without changing the persisted note schema prematurely.
        """

        grouped: dict[str, list[str]] = {}
        normalized_url = url.strip()
        for index, note in enumerate(self.notes, start=1):
            if note.url != normalized_url:
                continue
            note_id = note.note_id or f"note-{index:06d}"
            grouped.setdefault(note.item_id, []).append(note_id)
        return {item_id: tuple(ids) for item_id, ids in grouped.items()}

    @property
    def settled_without_located_evidence_item_ids(self) -> tuple[str, ...]:
        """Return uniquely settled items whose settle-time evidence count was zero."""

        item_ids: list[str] = []
        seen: set[str] = set()
        for record in self.checklist_history:
            evidence = record.settlement_evidence
            if (
                not record.accepted
                or record.to_status != "settled"
                or evidence is None
                or evidence.located_notes != 0
                or record.item_id in seen
            ):
                continue
            seen.add(record.item_id)
            item_ids.append(record.item_id)
        return tuple(item_ids)

    @property
    def settled_without_located_evidence(self) -> int:
        """Count items settled without strict or repaired located evidence."""

        return len(self.settled_without_located_evidence_item_ids)

    @property
    def rejected_exhausted_without_collection_attempt_item_ids(
        self,
    ) -> tuple[str, ...]:
        """Return zero-attempt exhaustion judgements rejected by the loop."""

        return self._exhaustion_item_ids(accepted=False, has_attempt=False)

    @property
    def rejected_exhausted_without_collection_attempt(self) -> int:
        """Count zero-attempt exhaustion judgements rejected by the loop."""

        return len(
            self.rejected_exhausted_without_collection_attempt_item_ids
        )

    @property
    def accepted_exhausted_without_collection_attempt_item_ids(
        self,
    ) -> tuple[str, ...]:
        """Return accepted exhausted records with a recorded zero snapshot."""

        return self._exhaustion_item_ids(accepted=True, has_attempt=False)

    @property
    def accepted_exhausted_without_collection_attempt(self) -> int:
        """Count accepted exhausted records whose snapshot proves no attempt."""

        return len(
            self.accepted_exhausted_without_collection_attempt_item_ids
        )

    @property
    def accepted_exhausted_attempt_unknown_legacy_item_ids(
        self,
    ) -> tuple[str, ...]:
        """Return legacy accepted exhaustions lacking a frozen attempt snapshot."""

        item_ids: list[str] = []
        seen: set[str] = set()
        for record in self.checklist_history:
            if (
                not record.accepted
                or record.to_status != "exhausted_not_found"
                or record.exhaustion_attempts is not None
                or record.item_id in seen
            ):
                continue
            seen.add(record.item_id)
            item_ids.append(record.item_id)
        return tuple(item_ids)

    @property
    def accepted_exhausted_attempt_unknown_legacy(self) -> int:
        """Count legacy accepted exhaustions whose attempt history is unknown."""

        return len(self.accepted_exhausted_attempt_unknown_legacy_item_ids)

    @property
    def exhausted_with_unread_candidates_item_ids(self) -> tuple[str, ...]:
        """Return accepted exhaustions whose frozen snapshot retained unread URLs."""

        item_ids: list[str] = []
        seen: set[str] = set()
        for record in self.checklist_history:
            snapshot = record.exhaustion_attempts
            if (
                not record.accepted
                or record.to_status != "exhausted_not_found"
                or snapshot is None
                or not snapshot.pending_unread_urls
                or record.item_id in seen
            ):
                continue
            seen.add(record.item_id)
            item_ids.append(record.item_id)
        return tuple(item_ids)

    @property
    def exhausted_with_unread_candidates(self) -> int:
        """Count accepted exhaustion judgements made with unread candidates."""

        return len(self.exhausted_with_unread_candidates_item_ids)

    def _exhaustion_item_ids(
        self,
        *,
        accepted: bool,
        has_attempt: bool,
    ) -> tuple[str, ...]:
        item_ids: list[str] = []
        seen: set[str] = set()
        for record in self.checklist_history:
            snapshot = record.exhaustion_attempts
            if (
                record.accepted is not accepted
                or record.to_status != "exhausted_not_found"
                or snapshot is None
                or snapshot.has_qualifying_attempt is not has_attempt
                or record.item_id in seen
            ):
                continue
            seen.add(record.item_id)
            item_ids.append(record.item_id)
        return tuple(item_ids)

    @property
    def total_tokens(self) -> int:
        """Return total recorded model tokens."""

        return sum(record.token_count for record in self.rounds)

    @property
    def total_cost_usd(self) -> float:
        """Return total recorded provider cost."""

        return sum(record.cost_usd for record in self.rounds)

    def to_audit_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible audit payload."""

        return self.model_dump(mode="json")

    def to_audit_json(self, *, indent: int | None = 2) -> str:
        """Serialize the complete audit payload deterministically."""

        return json.dumps(
            self.to_audit_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )
