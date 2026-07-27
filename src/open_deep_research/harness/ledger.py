"""Serializable audit ledger and immutable-by-URL source cache."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from open_deep_research.harness.notes import ResearchNote


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
    ) -> ChecklistChangeRecord:
        """Append and return one checklist audit record."""

        record = ChecklistChangeRecord(
            event=event,
            item_id=item_id,
            accepted=accepted,
            reason=reason,
            from_status=from_status,
            to_status=to_status,
        )
        self.checklist_history.append(record)
        return record

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
