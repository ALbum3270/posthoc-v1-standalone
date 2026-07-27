"""Canonical narrative drafting plus retained citation-parser diagnostics."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Awaitable, Mapping
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class WriteModelClient(Protocol):
    """Injected report-writing model boundary."""

    def generate(self, prompt: str) -> Any | Awaitable[Any]:
        """Return report markdown in a measured usage envelope."""


class ReportDraft(BaseModel):
    """A canonical citation-free draft and its measured model usage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_draft: str
    token_count: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)


class ParsedCitation(BaseModel):
    """One mechanically resolved claim, verbatim quote and source URL."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: str
    quote: str
    url: str
    reference_id: str


class CitationIssue(BaseModel):
    """A report assertion whose citation cannot be mechanically resolved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: str
    reason: str
    reference_ids: tuple[str, ...] = ()


class CitationParseResult(BaseModel):
    """All resolved triples and all unresolved report assertions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    citations: tuple[ParsedCitation, ...] = ()
    unresolved_claims: tuple[CitationIssue, ...] = ()


class _WriteEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    content: str
    token_count: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)


WRITE_PROMPT = """\
Write the best possible canonical Markdown narrative draft from all of the
assembled research material below. You alone decide the draft's content,
structure, length, and headings.

This is the drafting stage of a post-hoc attribution pipeline. Return narrative
body text only:
- Do not add footnotes, citations, reference markers, source lists, or URLs.
- Do not emit citation or reference numbers in any format.
- Do not reproduce note_id or source_id metadata in the draft.
- Do not present any passage as a verbatim quotation or claim that wording is
  copied exactly from a source.
- Do not add evidence, grounding, confidence, or verification status labels.

Return only the canonical Markdown draft.

Assembled research material:
{assembled_notes}
"""

_REFERENCE_DEFINITION = re.compile(
    r"^\[\^([A-Za-z0-9_-]+)\]:\s*(.*?)\s*$"
)
_REFERENCE_MARKER = re.compile(r"\[\^([A-Za-z0-9_-]+)\]")
_LIST_PREFIX = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")


def build_write_prompt(assembled_notes: str) -> str:
    """Build the one-pass writing prompt without imposing a report template."""

    if not isinstance(assembled_notes, str):
        raise TypeError("assembled_notes must be text")
    return WRITE_PROMPT.format(assembled_notes=assembled_notes)


async def write_report(
    assembled_notes: str,
    *,
    model_client: WriteModelClient,
) -> ReportDraft:
    """Ask one model call to turn the complete assembled material into a report."""

    response = model_client.generate(build_write_prompt(assembled_notes))
    if inspect.isawaitable(response):
        response = await response
    try:
        envelope = _WriteEnvelope.model_validate(response)
    except ValidationError as exc:
        raise ValueError("write model returned an invalid usage envelope") from exc
    return ReportDraft(
        canonical_draft=envelope.content,
        token_count=envelope.token_count,
        cost_usd=envelope.cost_usd,
    )


def _source_definition(value: str) -> tuple[str, str] | None:
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(decoded, Mapping):
        return None
    quote = decoded.get("quote")
    url = decoded.get("url")
    if not isinstance(quote, str) or not isinstance(url, str):
        return None
    if not quote or not url.strip():
        return None
    return quote, url.strip()


def _body_blocks(lines: list[str]) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    in_code_fence = False

    def flush() -> None:
        if current:
            blocks.append("\n".join(current).strip())
            current.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush()
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            continue
        if not stripped:
            flush()
            continue
        if stripped.startswith("#") or stripped in {"---", "***", "___"}:
            flush()
            continue
        current.append(line)
    flush()
    return [block for block in blocks if block]


def _claim_text(block: str) -> str:
    without_markers = _REFERENCE_MARKER.sub("", block)
    lines = [
        _LIST_PREFIX.sub("", line.strip())
        for line in without_markers.splitlines()
        if line.strip()
    ]
    return " ".join(lines).strip()


def parse_report_citations(markdown: str) -> CitationParseResult:
    """Extract valid triples while retaining every unresolvable assertion."""

    if not isinstance(markdown, str):
        raise TypeError("markdown must be text")

    definitions: dict[str, tuple[str, str]] = {}
    invalid_definitions: set[str] = set()
    duplicate_definitions: set[str] = set()
    body_lines: list[str] = []

    for line in markdown.splitlines():
        match = _REFERENCE_DEFINITION.match(line.strip())
        if match is None:
            body_lines.append(line)
            continue
        reference_id, raw_definition = match.groups()
        parsed = _source_definition(raw_definition)
        if reference_id in definitions or reference_id in invalid_definitions:
            duplicate_definitions.add(reference_id)
            definitions.pop(reference_id, None)
            invalid_definitions.add(reference_id)
        elif parsed is None:
            invalid_definitions.add(reference_id)
        else:
            definitions[reference_id] = parsed

    citations: list[ParsedCitation] = []
    unresolved: list[CitationIssue] = []
    for block in _body_blocks(body_lines):
        claim = _claim_text(block)
        if not claim:
            continue
        reference_ids = tuple(_REFERENCE_MARKER.findall(block))
        if not reference_ids:
            unresolved.append(
                CitationIssue(claim=claim, reason="missing_reference")
            )
            continue

        seen: set[str] = set()
        ordered_ids = tuple(
            reference_id
            for reference_id in reference_ids
            if not (reference_id in seen or seen.add(reference_id))
        )
        invalid_ids = tuple(
            reference_id
            for reference_id in ordered_ids
            if reference_id in invalid_definitions
        )
        unknown_ids = tuple(
            reference_id
            for reference_id in ordered_ids
            if reference_id not in definitions
            and reference_id not in invalid_definitions
        )
        if invalid_ids:
            reason = (
                "duplicate_reference_definition"
                if any(value in duplicate_definitions for value in invalid_ids)
                else "malformed_reference_definition"
            )
            unresolved.append(
                CitationIssue(
                    claim=claim,
                    reason=reason,
                    reference_ids=invalid_ids,
                )
            )
        if unknown_ids:
            unresolved.append(
                CitationIssue(
                    claim=claim,
                    reason="unknown_reference",
                    reference_ids=unknown_ids,
                )
            )
        for reference_id in ordered_ids:
            source = definitions.get(reference_id)
            if source is None:
                continue
            quote, url = source
            citations.append(
                ParsedCitation(
                    claim=claim,
                    quote=quote,
                    url=url,
                    reference_id=reference_id,
                )
            )

    return CitationParseResult(
        citations=tuple(citations),
        unresolved_claims=tuple(unresolved),
    )
