"""Deterministic, source-bound segment pointers for extractive notes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

SOURCE_SEGMENTATION_VERSION = "markdown-aware-source-segments-v1"

# End a sentence only where terminal punctuation is followed by whitespace or
# the end of its Markdown paragraph. Quotation marks and closing brackets stay
# with the sentence. Exact extraction does not depend on this heuristic being
# linguistically perfect: it only determines the selectable granularity.
_SENTENCE_END = re.compile(r"[.!?。！？]+(?:[\"'”’）)\]}]+)?(?=\s|$)")
_ATX_HEADING = re.compile(r"^[ ]{0,3}#{1,6}[ \t]+.*$")
_SETEXT_HEADING = re.compile(r"^[ ]{0,3}(?:=+|-+)[ \t]*$")
_LIST_ITEM = re.compile(r"^[ \t]{0,3}(?:[-+*]|\d+[.)])[ \t]+")
_FENCE = re.compile(r"^[ ]{0,3}(?P<marks>`{3,}|~{3,})")
_TABLE_DELIMITER_CELL = re.compile(r"^:?-{3,}:?$")


class SourceSegmentKind(str, Enum):
    """Markdown-aware source unit kinds."""

    HEADING = "heading"
    PARAGRAPH_SENTENCE = "paragraph_sentence"
    LIST_ITEM = "list_item"
    TABLE_ROW = "table_row"
    CODE_BLOCK = "code_block"


@dataclass(frozen=True)
class _SourceBlock:
    unit_id: str
    kind: SourceSegmentKind
    start_char: int
    end_char: int


class SourceSegment(BaseModel):
    """One addressable exact slice of a cached source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    segment_id: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    unit_id: str = Field(min_length=1)
    kind: SourceSegmentKind
    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)

    @model_validator(mode="after")
    def _bounds_are_ordered(self) -> SourceSegment:
        if self.end_char <= self.start_char:
            raise ValueError("segment end_char must exceed start_char")
        return self


class SourceSpanRegistry(BaseModel):
    """A deterministic segment registry bound to exact source bytes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    registry_id: str = Field(min_length=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    segmentation_version: str = Field(min_length=1)
    segments: tuple[SourceSegment, ...]

    @model_validator(mode="after")
    def _segments_are_unique_and_ordered(self) -> SourceSpanRegistry:
        ids = [segment.segment_id for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("source segment IDs must be unique")
        for index, segment in enumerate(self.segments):
            if segment.ordinal != index:
                raise ValueError("source segment ordinals must be contiguous")
            if index and self.segments[index - 1].end_char > segment.start_char:
                raise ValueError("source segments must not overlap")
        return self


class ResolvedSourceSpan(BaseModel):
    """A mechanically resolved continuous range in one Markdown unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_segment_id: str
    end_segment_id: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    source_quote: str = Field(min_length=1)


def _trimmed_bounds(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def _paragraph_sentence_bounds(
    source_text: str,
    block: _SourceBlock,
) -> list[tuple[int, int]]:
    """Split a paragraph while retaining absolute, exact source offsets."""

    bounds: list[tuple[int, int]] = []
    relative_start = 0
    block_text = source_text[block.start_char : block.end_char]
    for match in _SENTENCE_END.finditer(block_text):
        relative_end = match.end()
        trimmed = _trimmed_bounds(
            source_text,
            block.start_char + relative_start,
            block.start_char + relative_end,
        )
        if trimmed is not None:
            bounds.append(trimmed)
        relative_start = relative_end
    trimmed = _trimmed_bounds(
        source_text,
        block.start_char + relative_start,
        block.end_char,
    )
    if trimmed is not None:
        bounds.append(trimmed)
    return bounds


def _block_segment_bounds(
    source_text: str,
    block: _SourceBlock,
) -> Sequence[tuple[int, int]]:
    if block.kind is SourceSegmentKind.PARAGRAPH_SENTENCE:
        return _paragraph_sentence_bounds(source_text, block)
    trimmed = _trimmed_bounds(source_text, block.start_char, block.end_char)
    return () if trimmed is None else (trimmed,)


def build_source_span_registry(source_text: str) -> SourceSpanRegistry:
    """Build stable Markdown-aware segments over the complete cached text."""

    if not isinstance(source_text, str):
        raise TypeError("source_text must be text")
    source_hash = sha256(source_text.encode("utf-8")).hexdigest()
    registry_digest = sha256(
        f"{SOURCE_SEGMENTATION_VERSION}\0{source_hash}".encode("utf-8")
    ).hexdigest()[:16]
    segments: list[SourceSegment] = []
    for block in _parse_source_blocks(source_text):
        for start_char, end_char in _block_segment_bounds(source_text, block):
            segment = SourceSegment(
                segment_id=f"S{len(segments) + 1:06d}",
                ordinal=len(segments),
                unit_id=block.unit_id,
                kind=block.kind,
                text=source_text[start_char:end_char],
                start_char=start_char,
                end_char=end_char,
            )
            if segment.text != source_text[start_char:end_char]:
                raise AssertionError("segment must equal its authoritative slice")
            segments.append(segment)
    return SourceSpanRegistry(
        registry_id=f"spanreg-{registry_digest}",
        source_text_sha256=source_hash,
        segmentation_version=SOURCE_SEGMENTATION_VERSION,
        segments=tuple(segments),
    )


def _line_end_without_newline(
    source_text: str,
    start: int,
    raw: str,
) -> int:
    end = start + len(raw)
    while end > start and source_text[end - 1] in "\r\n":
        end -= 1
    return end


def _is_table_delimiter(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(
        _TABLE_DELIMITER_CELL.fullmatch(cell) is not None for cell in cells
    )


def _looks_like_table_row(line: str) -> bool:
    stripped = line.strip()
    return "|" in stripped and not stripped.startswith(("```", "~~~"))


def _parse_source_blocks(source_text: str) -> tuple[_SourceBlock, ...]:
    """Recognize Markdown units while retaining absolute source bounds."""

    if not source_text:
        return ()
    lines = source_text.splitlines(keepends=True)
    starts: list[int] = []
    offset = 0
    for raw in lines:
        starts.append(offset)
        offset += len(raw)

    table_rows: set[int] = set()
    for candidate in range(len(lines) - 1):
        header = lines[candidate].rstrip("\r\n")
        delimiter = lines[candidate + 1].rstrip("\r\n")
        if _looks_like_table_row(header) and _is_table_delimiter(delimiter):
            table_rows.add(candidate)
            cursor = candidate + 2
            while cursor < len(lines):
                row = lines[cursor].rstrip("\r\n")
                if not row.strip() or not _looks_like_table_row(row):
                    break
                table_rows.add(cursor)
                cursor += 1

    blocks: list[_SourceBlock] = []

    def add(kind: SourceSegmentKind, first: int, last: int) -> None:
        start = starts[first]
        end = _line_end_without_newline(
            source_text,
            starts[last],
            lines[last],
        )
        if end > start:
            blocks.append(
                _SourceBlock(
                    unit_id=f"unit-{len(blocks) + 1:06d}",
                    kind=kind,
                    start_char=start,
                    end_char=end,
                )
            )

    index = 0
    while index < len(lines):
        line = lines[index].rstrip("\r\n")
        if not line.strip():
            index += 1
            continue
        if _ATX_HEADING.match(line):
            add(SourceSegmentKind.HEADING, index, index)
            index += 1
            continue
        if (
            index + 1 < len(lines)
            and _SETEXT_HEADING.match(lines[index + 1].rstrip("\r\n"))
        ):
            add(SourceSegmentKind.HEADING, index, index + 1)
            index += 2
            continue
        fence = _FENCE.match(line)
        if fence is not None:
            marker = fence.group("marks")
            last = index
            while last + 1 < len(lines):
                last += 1
                if lines[last].lstrip().startswith(marker[0] * len(marker)):
                    break
            add(SourceSegmentKind.CODE_BLOCK, index, last)
            index = last + 1
            continue
        if _is_table_delimiter(line):
            index += 1
            continue
        if index in table_rows:
            add(SourceSegmentKind.TABLE_ROW, index, index)
            index += 1
            continue
        if _LIST_ITEM.match(line):
            last = index
            while last + 1 < len(lines):
                candidate = lines[last + 1].rstrip("\r\n")
                if (
                    not candidate.strip()
                    or _ATX_HEADING.match(candidate)
                    or _FENCE.match(candidate)
                    or _LIST_ITEM.match(candidate)
                    or last + 1 in table_rows
                ):
                    break
                last += 1
            add(SourceSegmentKind.LIST_ITEM, index, last)
            index = last + 1
            continue

        last = index
        while last + 1 < len(lines):
            candidate = lines[last + 1].rstrip("\r\n")
            if (
                not candidate.strip()
                or _ATX_HEADING.match(candidate)
                or _FENCE.match(candidate)
                or _LIST_ITEM.match(candidate)
                or last + 1 in table_rows
            ):
                break
            if (
                last + 2 < len(lines)
                and _SETEXT_HEADING.match(lines[last + 2].rstrip("\r\n"))
            ):
                break
            last += 1
        add(SourceSegmentKind.PARAGRAPH_SENTENCE, index, last)
        index = last + 1
    return tuple(blocks)


def render_segmented_source(
    source_text: str,
    registry: SourceSpanRegistry,
) -> str:
    """Insert local IDs without deleting or rewriting any source character."""

    _validate_registry_source(source_text, registry)
    pieces: list[str] = []
    cursor = 0
    for segment in registry.segments:
        pieces.append(source_text[cursor : segment.start_char])
        pieces.append(f"<{segment.segment_id}>")
        pieces.append(source_text[segment.start_char : segment.end_char])
        cursor = segment.end_char
    pieces.append(source_text[cursor:])
    return "".join(pieces)


def _validate_registry_source(
    source_text: str,
    registry: SourceSpanRegistry,
) -> None:
    actual_hash = sha256(source_text.encode("utf-8")).hexdigest()
    if actual_hash != registry.source_text_sha256:
        raise ValueError("source text does not match span registry hash")
    if registry.segmentation_version != SOURCE_SEGMENTATION_VERSION:
        raise ValueError("unsupported source segmentation version")
    for segment in registry.segments:
        if source_text[segment.start_char : segment.end_char] != segment.text:
            raise ValueError(
                f"segment {segment.segment_id!r} does not match source text"
            )


def resolve_source_span(
    source_text: str,
    registry: SourceSpanRegistry,
    *,
    start_segment_id: str,
    end_segment_id: str,
) -> ResolvedSourceSpan:
    """Resolve one valid, ordered range without clamping or truncation."""

    _validate_registry_source(source_text, registry)
    by_id = {segment.segment_id: segment for segment in registry.segments}
    start = by_id.get(start_segment_id)
    end = by_id.get(end_segment_id)
    if start is None:
        raise ValueError(f"unknown start_segment_id {start_segment_id!r}")
    if end is None:
        raise ValueError(f"unknown end_segment_id {end_segment_id!r}")
    if start.ordinal > end.ordinal:
        raise ValueError("segment range is reversed")
    if start.unit_id != end.unit_id:
        raise ValueError("segment range crosses a Markdown unit boundary")
    selected = registry.segments[start.ordinal : end.ordinal + 1]
    if any(segment.unit_id != start.unit_id for segment in selected):
        raise ValueError("segment range crosses a Markdown unit boundary")
    source_quote = source_text[start.start_char : end.end_char]
    if not source_quote:
        raise ValueError("segment range resolved to an empty source quote")
    if source_quote != source_text[start.start_char : end.end_char]:
        raise AssertionError("resolved quote must equal its source slice")
    return ResolvedSourceSpan(
        start_segment_id=start.segment_id,
        end_segment_id=end.segment_id,
        start_char=start.start_char,
        end_char=end.end_char,
        source_quote=source_quote,
    )
