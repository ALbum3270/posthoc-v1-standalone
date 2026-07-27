"""Auditable research notes grounded in cached source text."""

from __future__ import annotations

import re
import unicodedata
from enum import Enum
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from open_deep_research.graphrag.validation.grounding import locate_source_quote

_ELLIPSIS = re.compile(r"\s*(?:\.{3,}|…)\s*")
_DIGIT_RUN = re.compile(r"\d+")
_MAX_REPAIR_SPAN_EXPANSION_CHARS = 64


class NoteLocationStatus(str, Enum):
    """Whether and how a note acquired an authoritative source span."""

    LOCATABLE = "locatable"
    REPAIRED_LOCATABLE = "repaired_locatable"
    UNLOCATABLE = "unlocatable"


class QuoteFailureReason(str, Enum):
    """Mechanical reason strict location and conservative repair did not pass."""

    NONCONTIGUOUS_COMPOSITE = "noncontiguous_composite"
    AMBIGUOUS_FORMAT_MATCH = "ambiguous_format_match"
    NUMBER_SEQUENCE_MISMATCH = "number_sequence_mismatch"
    REPAIR_SPAN_TOO_LARGE = "repair_span_too_large"
    QUOTE_NOT_FOUND = "quote_not_found"


class QuoteRepairMethod(str, Enum):
    """The only conservative repair method implemented by the harness."""

    NFKC_CASEFOLD_ALNUM_UNIQUE_CONTIGUOUS = (
        "nfkc_casefold_alnum_unique_contiguous"
    )


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


class SourceEvidence(BaseModel):
    """Authoritative evidence shared by report writing and later verification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    quote: str
    url: str
    publisher: str
    span: QuoteSpan
    location_status: NoteLocationStatus

    @model_validator(mode="after")
    def _status_has_a_source_span(self) -> SourceEvidence:
        if self.location_status not in {
            NoteLocationStatus.LOCATABLE,
            NoteLocationStatus.REPAIRED_LOCATABLE,
        }:
            raise ValueError("source evidence requires a usable location status")
        return self


class ResearchNote(BaseModel):
    """One finding with separate model-proposed and source-authoritative quotes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1)
    finding: str = Field(min_length=1)
    model_quote: str
    source_quote: str | None = None
    url: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    span: QuoteSpan | None = None
    location_status: NoteLocationStatus
    repair_method: QuoteRepairMethod | None = None
    failure_reason: QuoteFailureReason | None = None
    located_fragment_count: int = Field(default=0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_quote(cls, value: Any) -> Any:
        """Accept frozen pre-dual-field fixtures without changing their bytes."""

        if not isinstance(value, dict) or "quote" not in value:
            return value
        migrated = dict(value)
        legacy_quote = str(migrated.pop("quote"))
        status = migrated.get("location_status")
        migrated.setdefault("model_quote", legacy_quote)
        if status in {
            NoteLocationStatus.LOCATABLE,
            NoteLocationStatus.LOCATABLE.value,
        }:
            migrated.setdefault("source_quote", legacy_quote)
        else:
            migrated.setdefault("source_quote", None)
            migrated.setdefault(
                "failure_reason",
                QuoteFailureReason.QUOTE_NOT_FOUND.value,
            )
        migrated.setdefault("located_fragment_count", 0)
        return migrated

    @model_validator(mode="after")
    def _location_status_matches_evidence(self) -> ResearchNote:
        usable = self.location_status in {
            NoteLocationStatus.LOCATABLE,
            NoteLocationStatus.REPAIRED_LOCATABLE,
        }
        if usable and (self.span is None or self.source_quote is None):
            raise ValueError("located notes require a source quote and span")
        if not usable and (self.span is not None or self.source_quote is not None):
            raise ValueError("unlocatable notes cannot have source evidence")
        if self.location_status is NoteLocationStatus.REPAIRED_LOCATABLE:
            if self.repair_method is None:
                raise ValueError("repaired notes require a repair method")
        elif self.repair_method is not None:
            raise ValueError("only repaired notes may have a repair method")
        if usable and self.failure_reason is not None:
            raise ValueError("located notes cannot have a failure reason")
        if not usable and self.failure_reason is None:
            raise ValueError("unlocatable notes require a failure reason")
        if usable and self.located_fragment_count:
            raise ValueError("located notes cannot retain failed-fragment counts")
        return self

    @property
    def is_locatable(self) -> bool:
        """Return whether the unchanged shared strict locator found the quote."""

        return self.location_status is NoteLocationStatus.LOCATABLE

    @property
    def has_usable_source_span(self) -> bool:
        """Return whether strict location or conservative repair yielded evidence."""

        return self.location_status in {
            NoteLocationStatus.LOCATABLE,
            NoteLocationStatus.REPAIRED_LOCATABLE,
        }


def source_evidence(note: ResearchNote) -> SourceEvidence | None:
    """Export only source-authored text for writing and verification consumers."""

    if (
        note.source_quote is None
        or note.span is None
        or not note.has_usable_source_span
    ):
        return None
    return SourceEvidence(
        quote=note.source_quote,
        url=note.url,
        publisher=note.publisher,
        span=note.span,
        location_status=note.location_status,
    )


def _publisher_domain(url: str) -> str:
    """Return the normalized host used as the note's publisher."""

    host = (urlparse(url).hostname or "").strip(".").casefold()
    if host.startswith("www."):
        host = host[4:]
    return host


def _compact_alphanumeric_with_map(value: str) -> tuple[str, list[int]]:
    """Normalize matching text while retaining source character provenance."""

    characters: list[str] = []
    source_indices: list[int] = []
    for source_index, character in enumerate(value):
        normalized = unicodedata.normalize("NFKC", character).casefold()
        for normalized_character in normalized:
            if normalized_character.isalnum():
                characters.append(normalized_character)
                source_indices.append(source_index)
    return "".join(characters), source_indices


def _all_occurrences(haystack: str, needle: str) -> list[int]:
    starts: list[int] = []
    start = haystack.find(needle)
    while start >= 0:
        starts.append(start)
        start = haystack.find(needle, start + 1)
    return starts


def _number_sequence(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value)
    return tuple(_DIGIT_RUN.findall(normalized))


def _located_fragment_count(source_text: str, model_quote: str) -> int:
    compact_source, _ = _compact_alphanumeric_with_map(source_text)
    fragments = [
        _compact_alphanumeric_with_map(fragment)[0]
        for fragment in _ELLIPSIS.split(model_quote)
    ]
    usable_fragments = [fragment for fragment in fragments if fragment]
    if len(usable_fragments) < 2:
        return 0
    return sum(fragment in compact_source for fragment in usable_fragments)


def _repair_source_span(
    source_text: str,
    model_quote: str,
    *,
    max_span_expansion_chars: int,
) -> tuple[QuoteSpan | None, QuoteFailureReason | None]:
    """Find one unique, complete normalized substring and map it to source."""

    compact_source, source_map = _compact_alphanumeric_with_map(source_text)
    compact_quote, _ = _compact_alphanumeric_with_map(model_quote)
    if not compact_source or not compact_quote:
        return None, QuoteFailureReason.QUOTE_NOT_FOUND

    occurrences = _all_occurrences(compact_source, compact_quote)
    if len(occurrences) > 1:
        return None, QuoteFailureReason.AMBIGUOUS_FORMAT_MATCH
    if not occurrences:
        return None, QuoteFailureReason.QUOTE_NOT_FOUND

    normalized_start = occurrences[0]
    normalized_end = normalized_start + len(compact_quote)
    start_char = source_map[normalized_start]
    end_char = source_map[normalized_end - 1] + 1
    source_quote = source_text[start_char:end_char]

    expansion = len(source_quote) - len(compact_quote)
    if expansion > max_span_expansion_chars:
        return None, QuoteFailureReason.REPAIR_SPAN_TOO_LARGE
    if _number_sequence(source_quote) != _number_sequence(model_quote):
        return None, QuoteFailureReason.NUMBER_SEQUENCE_MISMATCH
    return QuoteSpan(start_char=start_char, end_char=end_char), None


def create_note(
    *,
    item_id: str,
    finding: str,
    quote: str,
    url: str,
    source_text: str,
    max_repair_span_expansion_chars: int = _MAX_REPAIR_SPAN_EXPANSION_CHARS,
) -> ResearchNote:
    """Strictly locate, conservatively repair, or retain an unlocated note."""

    if max_repair_span_expansion_chars < 0:
        raise ValueError("max_repair_span_expansion_chars must be non-negative")
    normalized_url = url.strip()
    located = locate_source_quote(source_text, quote)
    if located is not None:
        source_quote = located.quote or ""
        return ResearchNote(
            item_id=item_id.strip(),
            finding=finding.strip(),
            model_quote=quote,
            source_quote=source_quote,
            url=normalized_url,
            publisher=_publisher_domain(normalized_url),
            span=QuoteSpan(
                start_char=located.start_char,
                end_char=located.end_char,
            ),
            location_status=NoteLocationStatus.LOCATABLE,
        )

    fragment_count = _located_fragment_count(source_text, quote)
    if fragment_count >= 2:
        return ResearchNote(
            item_id=item_id.strip(),
            finding=finding.strip(),
            model_quote=quote,
            url=normalized_url,
            publisher=_publisher_domain(normalized_url),
            location_status=NoteLocationStatus.UNLOCATABLE,
            failure_reason=QuoteFailureReason.NONCONTIGUOUS_COMPOSITE,
            located_fragment_count=fragment_count,
        )

    repaired_span, failure_reason = _repair_source_span(
        source_text,
        quote,
        max_span_expansion_chars=max_repair_span_expansion_chars,
    )
    if repaired_span is not None:
        return ResearchNote(
            item_id=item_id.strip(),
            finding=finding.strip(),
            model_quote=quote,
            source_quote=source_text[
                repaired_span.start_char : repaired_span.end_char
            ],
            url=normalized_url,
            publisher=_publisher_domain(normalized_url),
            span=repaired_span,
            location_status=NoteLocationStatus.REPAIRED_LOCATABLE,
            repair_method=(
                QuoteRepairMethod.NFKC_CASEFOLD_ALNUM_UNIQUE_CONTIGUOUS
            ),
        )

    if failure_reason is None:  # pragma: no cover - paired helper invariant
        raise RuntimeError("failed quote repair did not provide a reason")
    return ResearchNote(
        item_id=item_id.strip(),
        finding=finding.strip(),
        model_quote=quote,
        url=normalized_url,
        publisher=_publisher_domain(normalized_url),
        location_status=NoteLocationStatus.UNLOCATABLE,
        failure_reason=failure_reason,
        located_fragment_count=fragment_count,
    )
