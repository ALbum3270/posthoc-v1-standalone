"""Deterministically assemble checklist-ordered notes for report writing."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from hashlib import sha256

from open_deep_research.harness.checklist import (
    ChecklistDimension,
    ChecklistItem,
    ResearchChecklist,
)
from open_deep_research.harness.notes import ResearchNote, source_evidence

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
    return (
        note.publisher,
        note.url,
        start,
        end,
        note.finding,
        note.source_quote or "",
    )


def _quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _fallback_note_ids(notes: list[ResearchNote]) -> list[str]:
    """Assign deterministic material-local IDs only when no ledger ID exists."""

    canonical = [
        json.dumps(
            note.model_dump(mode="json", exclude={"note_id"}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for note in notes
    ]
    bases = [
        f"note-material-{sha256(value.encode('utf-8')).hexdigest()[:16]}"
        for value in canonical
    ]
    totals = Counter(bases)
    seen: Counter[str] = Counter()
    result: list[str] = []
    for note, base in zip(notes, bases, strict=True):
        if note.note_id is not None:
            result.append(note.note_id)
            continue
        seen[base] += 1
        suffix = f"-{seen[base]:03d}" if totals[base] > 1 else ""
        result.append(f"{base}{suffix}")
    return result


def _render_note(
    note: ResearchNote,
    number: int,
    note_id: str,
) -> list[str]:
    evidence = source_evidence(note)
    span = (
        f"{evidence.span.start_char}:{evidence.span.end_char}"
        if evidence is not None
        else "unlocatable"
    )
    lines = [
        f"### Note {number} | note_id={note_id}",
        f"- Source ID: {note.source_id}",
        f"- Finding: {_quoted(note.finding)}",
        (
            f"- Source quote: {_quoted(evidence.quote)}"
            if evidence is not None
            else "- Source quote: unavailable"
        ),
        f"- URL: {_quoted(note.url)}",
        f"- Publisher: {_quoted(note.publisher)}",
        f"- Location: {note.location_status.value}",
        f"- Span: {span}",
    ]
    if note.repair_method is not None:
        lines.append(f"- Repair method: {note.repair_method.value}")
    if note.failure_reason is not None:
        lines.append(f"- Failure reason: {note.failure_reason.value}")
        lines.append(
            f"- Located fragment count: {note.located_fragment_count}"
        )
    return lines


def _render_item(
    item: ChecklistItem,
    notes: list[tuple[ResearchNote, str]],
) -> list[str]:
    lines = [
        (
            f"## {item.dimension.value} | priority={item.priority} "
            f"| item_id={item.item_id}"
        ),
        f"- Question: {_quoted(item.question)}",
        f"- Status: {item.status.value}",
    ]
    if not notes:
        lines.append("- Notes: none")
        return lines

    ordered = sorted(notes, key=lambda value: (*_note_sort_key(value[0]), value[1]))
    for number, (note, note_id) in enumerate(ordered, start=1):
        lines.extend(("", *_render_note(note, number, note_id)))
    return lines


def assemble_notes(
    checklist: ResearchChecklist,
    notes: Iterable[ResearchNote],
) -> str:
    """Return reproducible structured text without discarding unmatched notes."""

    material = list(notes)
    note_ids = _fallback_note_ids(material)
    identified = list(zip(material, note_ids, strict=True))
    known_item_ids = {item.item_id for item in checklist.items}
    by_item: dict[str, list[tuple[ResearchNote, str]]] = {
        item_id: [] for item_id in known_item_ids
    }
    unmatched: list[tuple[ResearchNote, str]] = []
    for note, note_id in identified:
        if note.item_id in by_item:
            by_item[note.item_id].append((note, note_id))
        else:
            unmatched.append((note, note_id))

    lines = [
        "# Assembled research notes",
        f"Topic: {_quoted(checklist.topic)}",
    ]
    for item in sorted(checklist.items, key=_item_sort_key):
        lines.extend(("", *_render_item(item, by_item[item.item_id])))

    if unmatched:
        lines.extend(("", "## unmatched notes"))
        for number, (note, note_id) in enumerate(
            sorted(
                unmatched,
                key=lambda value: (
                    value[0].item_id,
                    *_note_sort_key(value[0]),
                    value[1],
                ),
            ),
            start=1,
        ):
            lines.extend(
                (
                    "",
                    (
                        f"### Note {number} | item_id={note.item_id} "
                        f"| note_id={note_id}"
                    ),
                    *_render_note(note, number, note_id)[1:],
                )
            )

    return "\n".join(lines) + "\n"
