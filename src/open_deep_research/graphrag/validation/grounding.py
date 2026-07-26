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
import unicodedata
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
_SENTENCE_END = re.compile(r"""[.!?。！？]["'”’)\]]*(?=\s|$)""")
_LEADING_FURNITURE = re.compile(r"""^[\s"'“”‘’([{<\-–—•*]+""")
_DANGLING_CONJUNCTION = re.compile(r"^(?:and|but)\b|^(?:但|但是|并且|而且)", re.I)
_ANAPHORIC_ENGLISH = re.compile(
    r"^(?:they|them|their|theirs|this|these|those|it|its)\b",
    re.I,
)
_ANAPHORIC_CHINESE = re.compile(
    r"^(?:该|其|它们?|他们|她们|此|这些|那些|这(?:个|些)?)"
)
_ANAPHORIC_COMPANY = re.compile(r"^the\s+company(?:'s|’s)\b", re.I)
_LATIN_WORD = re.compile(r"[A-Za-z]+")
_INITIALISM = re.compile(r"^[A-Za-z][A-Za-z0-9]{1,9}$")
_CJK_CHARACTER = re.compile(r"[\u3400-\u9fff]")


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


def _is_abbreviation_period(source: str, punctuation_at: int) -> bool:
    """Avoid treating an initialism's final period as a sentence boundary."""

    prefix = source[: punctuation_at + 1]
    return re.search(r"(?:\b[A-Za-z]\.){2,}$", prefix) is not None


def _trim_source_span(source: str, start: int, end: int) -> SourceSpan:
    while start < end and source[start].isspace():
        start += 1
    while end > start and source[end - 1].isspace():
        end -= 1
    return SourceSpan(start_char=start, end_char=end, quote=source[start:end])


def _sentence_spans(content: str) -> list[SourceSpan]:
    """Split source text into exact, non-overlapping sentence-like spans."""

    source = content or ""
    if not source:
        return []

    sentence_ends = [
        match
        for match in _SENTENCE_END.finditer(source)
        if not (
            source[match.start()] == "."
            and _is_abbreviation_period(source, match.start())
        )
    ]
    hard_breaks = {0, len(source)}
    hard_breaks.update(match.end() for match in sentence_ends)
    for match in re.finditer(r"\n", source):
        hard_breaks.add(match.start())
        hard_breaks.add(match.end())

    boundaries = sorted(hard_breaks)
    spans = [
        _trim_source_span(source, start, end)
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    return [span for span in spans if span.quote]


def _overlapping_sentence_spans(content: str, span: SourceSpan) -> list[SourceSpan]:
    return [
        candidate
        for candidate in _sentence_spans(content)
        if candidate.end_char > span.start_char
        and candidate.start_char < span.end_char
    ]


def expand_span_to_sentence(
    content: str,
    span: SourceSpan,
    *,
    max_chars: int = 200,
) -> SourceSpan:
    """Expand a fragment to one bounded, verbatim source sentence.

    If the enclosing sentence exceeds ``max_chars``, keep the original fragment
    so that a separately self-contained clause can still pass the later gates.
    """

    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    candidates = _overlapping_sentence_spans(content, span)
    endpoint = max(span.start_char, span.end_char - 1)
    candidate = next(
        (
            item
            for item in candidates
            if item.start_char <= endpoint < item.end_char
        ),
        span,
    )
    return candidate if len(candidate.quote or "") <= max_chars else span


def _fold_name_text(text: str) -> str:
    """Normalize Unicode names while preserving token boundaries."""

    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return " ".join(
        "".join(character if character.isalnum() else " " for character in normalized)
        .split()
    )


def _contains_named_form(quote: str, named_form: str) -> bool:
    quote_folded = _fold_name_text(quote)
    name_folded = _fold_name_text(named_form)
    if not quote_folded or not name_folded:
        return False
    if _CJK_CHARACTER.search(name_folded):
        return name_folded.replace(" ", "") in quote_folded.replace(" ", "")
    return f" {name_folded} " in f" {quote_folded} "


def _derived_subject_forms(subject: str) -> list[str]:
    """Derive only structurally justified short forms from a subject name."""

    forms: list[str] = []
    latin_words = _LATIN_WORD.findall(subject)
    capitalized_words = [word for word in latin_words if word[0].isupper()]
    if len(capitalized_words) >= 2:
        initialism = "".join(word[0] for word in capitalized_words)
        if _INITIALISM.fullmatch(initialism):
            forms.append(initialism)

    final_token = subject.strip().split()[-1] if subject.strip() else ""
    hyphen_parts = [part for part in re.split(r"[-‐‑‒–—]", final_token) if part]
    if (
        len(hyphen_parts) >= 2
        and all(part.isalpha() for part in hyphen_parts)
        and sum(len(part) for part in hyphen_parts) >= 5
    ):
        forms.append(final_token)
    return forms


def quote_names_subject(quote: str, subject: str) -> bool:
    """Return whether a quote explicitly names the triple subject.

    Full Unicode names are matched directly. Latin initialisms and compound
    hyphenated surnames are derived from the supplied subject's own structure,
    avoiding event-specific alias dictionaries or arbitrary sliding windows.
    """

    if _contains_named_form(quote, subject):
        return True
    return any(
        _contains_named_form(quote, form)
        for form in _derived_subject_forms(subject)
    )


def quote_is_self_contained(quote: str) -> bool:
    """Reject fragments whose opening depends on missing prior context."""

    candidate = _LEADING_FURNITURE.sub("", quote or "")
    if not candidate:
        return False
    return not any(
        pattern.search(candidate)
        for pattern in (
            _DANGLING_CONJUNCTION,
            _ANAPHORIC_ENGLISH,
            _ANAPHORIC_CHINESE,
            _ANAPHORIC_COMPANY,
        )
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
    max_quote_chars: int = 200,
) -> ExtractedTriple | None:
    """Map one model row to a triple only when its evidence is self-contained.

    A located fragment is expanded to its enclosing source sentence, preserving
    exact offsets and text. Numbers must remain supported, the sentence must not
    begin with an unresolved reference, and it must explicitly name the triple
    subject. These mechanical checks avoid an LLM verifier or a threshold that
    would need calibration.
    """

    subject = str(row.get("subject") or "").strip()
    predicate = str(row.get("predicate") or "").strip()
    obj = str(row.get("object") or "").strip()
    if not subject or not predicate or not obj:
        return None

    quote = str(row.get("quote") or "").strip()
    located_span = locate_source_quote(document.content, quote)
    if located_span is None:
        return None
    if max_quote_chars < 1:
        raise ValueError("max_quote_chars must be positive")

    overlapping_spans = _overlapping_sentence_spans(
        document.content,
        located_span,
    )
    sentence_candidates = [
        candidate
        for candidate in overlapping_spans
        if len(candidate.quote or "") <= max_quote_chars
    ]
    # A quote inside one overlong sentence may already be a concise,
    # self-contained clause. A multi-sentence model quote must instead reduce
    # to one independently valid sentence.
    if len(overlapping_spans) == 1 and not sentence_candidates:
        sentence_candidates.append(located_span)

    valid_candidates = [
        candidate
        for candidate in sentence_candidates
        if len(candidate.quote or "") <= max_quote_chars
        and row_is_numerically_supported(row, candidate.quote or "")
        and quote_is_self_contained(candidate.quote or "")
        and quote_names_subject(candidate.quote or "", subject)
    ]
    if not valid_candidates:
        return None
    span = min(
        valid_candidates,
        key=lambda candidate: (
            not (
                candidate.start_char
                <= max(located_span.start_char, located_span.end_char - 1)
                < candidate.end_char
            ),
            len(candidate.quote or ""),
            candidate.start_char,
        ),
    )

    return ExtractedTriple(
        slot_id=slot.slot_id,
        subject=EntityRef(name=subject),
        predicate=predicate,
        object=obj,
        confidence=confidence,
        source_document_id=document.document_id,
        source_span=span,
        rationale="verbatim self-contained sentence located in source document",
    )
