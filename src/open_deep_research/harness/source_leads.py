"""Mechanical source-chain lead inventory for evidence recovery.

This module only exposes addressable text that already exists in the durable
source cache.  It does not decide whether a candidate is an original record,
whether it is relevant to a claim, or whether it supports anything.  Those
remain model judgements; code only makes the candidate identity auditable.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


_URL = re.compile(r"https?://[^\s<>\]\[()\"']+")
_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_NUMBERED_LINE = re.compile(
    r"(?m)^\s*(?P<number>\d{1,3})\.\s+(?P<text>[^\n]+?)\s*$"
)
_QUOTED_TITLE = re.compile(r"[\"“](?P<title>.+?)[\"”]")
_DATE = re.compile(
    r"\b(?:"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+\d{1,2}(?:,)?\s+\d{4}|"
    r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}"
    r")\b",
    re.IGNORECASE,
)
_LOCATORS = (
    (
        "case_number",
        re.compile(
            r"\bCase\s+(?:No\.?|Number|#)\s*[A-Z0-9][A-Z0-9.\-/]+",
            re.IGNORECASE,
        ),
    ),
    (
        "docket_number",
        re.compile(r"\bDocket\s*#?\s*\d+", re.IGNORECASE),
    ),
    (
        "related_document_numbers",
        re.compile(
            r"\brelated\s+document\(s\)\s*[\d,\s]+",
            re.IGNORECASE,
        ),
    ),
)


class SourceLeadKind(str, Enum):
    """Mechanical shape of a cached candidate, not a source-role verdict."""

    EXPLICIT_URL = "explicit_url"
    DOI = "doi"
    BIBLIOGRAPHIC_ENTRY = "bibliographic_entry"
    SOURCE_HEADER = "source_header"
    DATED_CONTEXT = "dated_context"


class SourceLeadLocator(BaseModel):
    """A verbatim case/docket-like locator found in cached text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(min_length=1)
    value: str = Field(min_length=1)


class SourceLeadCandidate(BaseModel):
    """One code-addressable clue that a model may choose to follow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lead_id: str = Field(pattern=r"^lead-[0-9a-f]{16}$")
    kind: SourceLeadKind
    source_url: str = Field(min_length=1)
    verbatim_text: str = Field(min_length=1)
    entry_number: int | None = Field(default=None, ge=0)
    source_label_candidate: str | None = None
    document_title_candidate: str | None = None
    dates: tuple[str, ...] = ()
    locators: tuple[SourceLeadLocator, ...] = ()


def _unique_matches(pattern: re.Pattern[str], text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(match.group(0) for match in pattern.finditer(text))
    )


def _lead_id(
    *,
    source_url: str,
    kind: SourceLeadKind,
    verbatim_text: str,
    entry_number: int | None = None,
) -> str:
    payload = "\x1f".join(
        (source_url, kind.value, str(entry_number), verbatim_text)
    )
    return "lead-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _candidate(
    *,
    source_url: str,
    kind: SourceLeadKind,
    verbatim_text: str,
    entry_number: int | None = None,
    source_label_candidate: str | None = None,
    document_title_candidate: str | None = None,
    dates: Sequence[str] = (),
    locators: Sequence[SourceLeadLocator] = (),
) -> SourceLeadCandidate:
    return SourceLeadCandidate(
        lead_id=_lead_id(
            source_url=source_url,
            kind=kind,
            verbatim_text=verbatim_text,
            entry_number=entry_number,
        ),
        kind=kind,
        source_url=source_url,
        verbatim_text=verbatim_text,
        entry_number=entry_number,
        source_label_candidate=source_label_candidate,
        document_title_candidate=document_title_candidate,
        dates=tuple(dates),
        locators=tuple(locators),
    )


def _line_contexts(text: str) -> tuple[str, ...]:
    """Return bounded full lines containing dates without semantic ranking."""

    contexts: list[str] = []
    for line in text.splitlines():
        normalized = line.strip()
        if not normalized or not _DATE.search(normalized):
            continue
        # A giant provider line is not a useful addressable clue.  Omitting it
        # is recorded by the inventory limitations rather than silently
        # truncating text and inventing a candidate boundary.
        if len(normalized) > 2_000:
            continue
        contexts.append(normalized)
    return tuple(dict.fromkeys(contexts))


def inventory_source_lead_candidates(
    source_cache: Mapping[str, str],
) -> tuple[SourceLeadCandidate, ...]:
    """Inventory generic cached-text structures in deterministic order."""

    candidates: list[SourceLeadCandidate] = []
    seen_ids: set[str] = set()

    def add(candidate: SourceLeadCandidate) -> None:
        if candidate.lead_id in seen_ids:
            return
        seen_ids.add(candidate.lead_id)
        candidates.append(candidate)

    for source_url, text in sorted(source_cache.items()):
        normalized_url = str(source_url).strip()
        source_text = str(text)
        if not normalized_url or not source_text:
            continue

        header_lines = tuple(
            line.strip() for line in source_text.splitlines() if line.strip()
        )[:6]
        if header_lines:
            header = "\n".join(header_lines)
            if len(header) <= 2_000:
                add(
                    _candidate(
                        source_url=normalized_url,
                        kind=SourceLeadKind.SOURCE_HEADER,
                        verbatim_text=header,
                        dates=_unique_matches(_DATE, header),
                    )
                )

        for explicit_url in _unique_matches(_URL, source_text):
            add(
                _candidate(
                    source_url=normalized_url,
                    kind=SourceLeadKind.EXPLICIT_URL,
                    verbatim_text=explicit_url,
                )
            )
        for doi in _unique_matches(_DOI, source_text):
            add(
                _candidate(
                    source_url=normalized_url,
                    kind=SourceLeadKind.DOI,
                    verbatim_text=doi,
                )
            )

        for match in _NUMBERED_LINE.finditer(source_text):
            entry_text = match.group("text").strip()
            title_match = _QUOTED_TITLE.search(entry_text)
            title = (
                title_match.group("title").strip()
                if title_match is not None
                else None
            )
            source_label = None
            if title_match is not None:
                source_label = (
                    entry_text[: title_match.start()].strip().rstrip(".:")
                    or None
                )
            locators = tuple(
                SourceLeadLocator(kind=kind, value=locator.group(0))
                for kind, pattern in _LOCATORS
                for locator in pattern.finditer(entry_text)
            )
            if title is None and not locators:
                continue
            add(
                _candidate(
                    source_url=normalized_url,
                    kind=SourceLeadKind.BIBLIOGRAPHIC_ENTRY,
                    verbatim_text=entry_text,
                    entry_number=int(match.group("number")),
                    source_label_candidate=source_label,
                    document_title_candidate=title,
                    dates=_unique_matches(_DATE, entry_text),
                    locators=locators,
                )
            )

        for context in _line_contexts(source_text):
            add(
                _candidate(
                    source_url=normalized_url,
                    kind=SourceLeadKind.DATED_CONTEXT,
                    verbatim_text=context,
                    dates=_unique_matches(_DATE, context),
                )
            )
    return tuple(candidates)


SOURCE_LEAD_INVENTORY_LIMITATIONS = (
    "candidates are mechanical text shapes, not source-role judgements",
    "quoted text can be a quotation rather than a document title",
    "a selected lead is not evidence and does not establish claim support",
    "the current Tavily text extraction boundary can omit hyperlink targets",
    "date contexts longer than 2000 characters are omitted rather than truncated",
)


__all__ = [
    "SOURCE_LEAD_INVENTORY_LIMITATIONS",
    "SourceLeadCandidate",
    "SourceLeadKind",
    "SourceLeadLocator",
    "inventory_source_lead_candidates",
]
