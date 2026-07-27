"""Serializable audit ledger and immutable-by-URL source cache."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from open_deep_research.harness.notes import ResearchNote


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


class RoundRecord(BaseModel):
    """One research-loop action and its measured usage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    round_number: int = Field(ge=1)
    action: str = Field(min_length=1)
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


class ResearchLedger(BaseModel):
    """All durable evidence and audit events for one research run."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    research_id: str = ""
    topic: str = ""
    rounds: list[RoundRecord] = Field(default_factory=list)
    source_cache: dict[str, str] = Field(default_factory=dict)
    notes: list[ResearchNote] = Field(default_factory=list)
    checklist_history: list[ChecklistChangeRecord] = Field(default_factory=list)

    def record_round(
        self,
        *,
        round_number: int,
        action: str,
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
            query=query,
            url=url,
            result_summary=result_summary,
            token_count=token_count,
            cost_usd=cost_usd,
        )
        self.rounds.append(record)
        return record

    def cache_source(self, url: str, cleaned_text: str) -> bool:
        """Cache full cleaned text once, rejecting later content drift."""

        normalized_url = url.strip()
        if not normalized_url:
            raise ValueError("source URL must not be blank")
        existing = self.source_cache.get(normalized_url)
        if existing is not None:
            if existing != cleaned_text:
                raise ValueError(f"cached source changed for URL: {normalized_url}")
            return False
        self.source_cache[normalized_url] = cleaned_text
        return True

    def get_source(self, url: str) -> str | None:
        """Return cached source text without refetching it."""

        return self.source_cache.get(url.strip())

    def add_note(self, note: ResearchNote) -> ResearchNote:
        """Retain a note regardless of quote-location status."""

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
        )
        self.checklist_history.append(record)
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
            grouped.setdefault(note.item_id, []).append(f"note-{index:06d}")
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
