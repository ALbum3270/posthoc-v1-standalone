"""Deterministically assemble checklist-ordered notes for report writing."""

from __future__ import annotations

import json
from collections.abc import Iterable

from open_deep_research.harness.checklist import (
    ChecklistDimension,
    ChecklistItem,
    ResearchChecklist,
)
from open_deep_research.harness.notes import ResearchNote

_DIMENSION_ORDER = {
    dimension: index
    for index, dimension in enumerate(
        (
            ChecklistDimension.WHO,
            ChecklistDimension.WHAT,
            ChecklistDimension.WHEN,
            ChecklistDimension.WHERE,
            ChecklistDimension.WHY,
            ChecklistDimension.HOW,
        )
    )
}


def _item_sort_key(item: ChecklistItem) -> tuple[int, int, str]:
    return (_DIMENSION_ORDER[item.dimension], item.priority, item.item_id)


def _note_sort_key(note: ResearchNote) -> tuple[str, str, int, int, str, str]:
    start = note.span.start_char if note.span is not None else -1
    end = note.span.end_char if note.span is not None else -1
    return (note.publisher, note.url, start, end, note.finding, note.quote)


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _render_note(note: ResearchNote, number: int) -> list[str]:
    span = (
        f"{note.span.start_char}:{note.span.end_char}"
        if note.span is not None
        else "unlocatable"
    )
    return [
        f"### Note {number}",
        f"- Finding: {_quoted(note.finding)}",
        f"- Quote: {_quoted(note.quote)}",
        f"- URL: {_quoted(note.url)}",
        f"- Publisher: {_quoted(note.publisher)}",
        f"- Location: {note.location_status.value}",
        f"- Span: {span}",
    ]


def _render_item(item: ChecklistItem, notes: list[ResearchNote]) -> list[str]:
    lines = [
        (
            f"## {item.dimension.value} | priority={item.priority} "
            f"| item_id={item.item_id}"
        ),
        f"- Question: {_quoted(item.question)}",
        f"- Status: {item.status.value}",
        f"- Required source count: {item.required_source_count}",
    ]
    if not notes:
        lines.append("- Notes: none")
        return lines

    for number, note in enumerate(sorted(notes, key=_note_sort_key), start=1):
        lines.extend(("", *_render_note(note, number)))
    return lines


def assemble_notes(
    checklist: ResearchChecklist,
    notes: Iterable[ResearchNote],
) -> str:
    """Return reproducible structured text without discarding unmatched notes."""

    material = list(notes)
    known_item_ids = {item.item_id for item in checklist.items}
    by_item: dict[str, list[ResearchNote]] = {
        item_id: [] for item_id in known_item_ids
    }
    unmatched: list[ResearchNote] = []
    for note in material:
        if note.item_id in by_item:
            by_item[note.item_id].append(note)
        else:
            unmatched.append(note)

    lines = [
        "# Assembled research notes",
        f"Topic: {_quoted(checklist.topic)}",
    ]
    for item in sorted(checklist.items, key=_item_sort_key):
        lines.extend(("", *_render_item(item, by_item[item.item_id])))

    if unmatched:
        lines.extend(("", "## unmatched notes"))
        for number, note in enumerate(
            sorted(unmatched, key=lambda value: (value.item_id, *_note_sort_key(value))),
            start=1,
        ):
            lines.extend(
                (
                    "",
                    f"### Note {number} | item_id={note.item_id}",
                    *_render_note(note, number)[1:],
                )
            )

    return "\n".join(lines) + "\n"
