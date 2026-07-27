"""Auditable research notes grounded in cached source text."""

from __future__ import annotations

from enum import Enum
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from open_deep_research.graphrag.validation.grounding import locate_source_quote


class NoteLocationStatus(str, Enum):
    """Whether a note's requested quote was found in its source."""

    LOCATABLE = "locatable"
    UNLOCATABLE = "unlocatable"


class QuoteSpan(BaseModel):
    """Exact character offsets for a quote in cached source text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)

    @model_validator(mode="after")
    def _end_is_not_before_start(self) -> QuoteSpan:
        if self.end_char < self.start_char:
            raise ValueError("quote span end must not precede its start")
        return self


class ResearchNote(BaseModel):
    """One finding and its exact or explicitly unlocatable source quote."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1)
    finding: str = Field(min_length=1)
    quote: str
    url: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    span: QuoteSpan | None = None
    location_status: NoteLocationStatus

    @model_validator(mode="after")
    def _location_status_matches_span(self) -> ResearchNote:
        if self.location_status is NoteLocationStatus.LOCATABLE and self.span is None:
            raise ValueError("locatable notes require a quote span")
        if (
            self.location_status is NoteLocationStatus.UNLOCATABLE
            and self.span is not None
        ):
            raise ValueError("unlocatable notes cannot have a quote span")
        return self

    @property
    def is_locatable(self) -> bool:
        """Return whether the quote has a verified source span."""

        return self.location_status is NoteLocationStatus.LOCATABLE


def _publisher_domain(url: str) -> str:
    """Return the normalized host used as the note's publisher."""

    host = (urlparse(url).hostname or "").strip(".").casefold()
    if host.startswith("www."):
        host = host[4:]
    return host


def create_note(
    *,
    item_id: str,
    finding: str,
    quote: str,
    url: str,
    source_text: str,
) -> ResearchNote:
    """Create a note without discarding or blocking on an unlocated quote."""

    normalized_url = url.strip()
    located = locate_source_quote(source_text, quote)
    if located is None:
        stored_quote = quote
        span = None
        location_status = NoteLocationStatus.UNLOCATABLE
    else:
        # Store the authoritative slice from source_text. This preserves exact
        # whitespace even when locate_source_quote used whitespace normalization.
        stored_quote = located.quote or ""
        span = QuoteSpan(
            start_char=located.start_char,
            end_char=located.end_char,
        )
        location_status = NoteLocationStatus.LOCATABLE

    return ResearchNote(
        item_id=item_id.strip(),
        finding=finding.strip(),
        quote=stored_quote,
        url=normalized_url,
        publisher=_publisher_domain(normalized_url),
        span=span,
        location_status=location_status,
    )
