"""Deterministically anchor extracted triples to exact source text.

The verified writer guarantees that a fact is not rewritten *after* extraction.
That is necessary but not sufficient: a model can still emit a fluent triple
that the page never stated.  A fact is source-grounded only when the extractor
also returns a verbatim quote that can be located in the selected document.

This module is deliberately mechanical.  It does not judge whether a statement
is true in the world; it establishes the narrower, auditable claim that the
stored report sentence is an exact passage from the cited source.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from open_deep_research.graphrag.ontology import OntologySlot
from open_deep_research.graphrag.schemas import (
    EntityRef,
    ExtractedTriple,
    SourceDocument,
    SourceSpan,
)

_SPACE_RUN = re.compile(r"\s+")
_NUMBER = re.compile(
    r"(?<!\w)[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*%)?"
)


def locate_source_quote(content: str, quote: str) -> SourceSpan | None:
    """Locate a quote in source content, allowing whitespace-only differences.

    The returned quote is sliced from ``content`` itself rather than copied from
    model output.  This makes the span and stored text share one authoritative
    representation.
    """

    source = content or ""
    requested = (quote or "").strip()
    if not source or not requested:
        return None

    start = source.find(requested)
    if start >= 0:
        end = start + len(requested)
        return SourceSpan(start_char=start, end_char=end, quote=source[start:end])

    parts = [part for part in _SPACE_RUN.split(requested) if part]
    if not parts:
        return None
    pattern = _SPACE_RUN.pattern.join(re.escape(part) for part in parts)
    match = re.search(pattern, source)
    if match is None:
        return None
    return SourceSpan(
        start_char=match.start(),
        end_char=match.end(),
        quote=source[match.start() : match.end()],
    )


def normalized_numbers(text: str) -> set[str]:
    """Return normalized numeric tokens for deterministic support checks."""

    values: set[str] = set()
    for match in _NUMBER.finditer(text or ""):
        token = re.sub(r"\s+", "", match.group(0)).replace(",", "")
        values.add(token)
    return values


def row_is_numerically_supported(row: Mapping[str, str], quote: str) -> bool:
    """Require every number in the structured triple to occur in its quote."""

    structured = " ".join(
        str(row.get(key) or "") for key in ("subject", "predicate", "object")
    )
    return normalized_numbers(structured).issubset(normalized_numbers(quote))


def ground_extracted_row(
    *,
    document: SourceDocument,
    slot: OntologySlot,
    row: Mapping[str, str],
    confidence: float = 0.8,
) -> ExtractedTriple | None:
    """Map one model row to a triple only when its quote is in the source.

    Numbers receive an additional subset check.  This catches the most damaging
    class of plausible-looking extraction errors without introducing an LLM
    verifier or a threshold that would need calibration.
    """

    quote = str(row.get("quote") or "").strip()
    span = locate_source_quote(document.content, quote)
    if span is None or not row_is_numerically_supported(row, span.quote or ""):
        return None

    subject = str(row.get("subject") or "").strip()
    predicate = str(row.get("predicate") or "").strip()
    obj = str(row.get("object") or "").strip()
    if not subject or not predicate or not obj:
        return None

    return ExtractedTriple(
        slot_id=slot.slot_id,
        subject=EntityRef(name=subject),
        predicate=predicate,
        object=obj,
        confidence=confidence,
        source_document_id=document.document_id,
        source_span=span,
        rationale="verbatim quote located in source document",
    )
