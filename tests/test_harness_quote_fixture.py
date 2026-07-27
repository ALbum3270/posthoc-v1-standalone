"""Frozen offline evidence for quote-location and future repair regressions."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from open_deep_research.graphrag.validation.grounding import (
    locate_source_quote,
)
from open_deep_research.harness.loop import quote_quality_metrics
from open_deep_research.harness.notes import (
    NoteLocationStatus,
    QuoteFailureReason,
    QuoteRepairMethod,
    ResearchNote,
    create_note,
)

_FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "harness_quote_grounding_d16bea.json"
)
_ELLIPSIS = re.compile(r"\s*(?:\.{3,}|…)\s*")


def _load_fixture() -> dict:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _compact_alphanumeric(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", value).casefold()
        if character.isalnum()
    )


def _test_only_classification(note: dict, source: str) -> str:
    if locate_source_quote(source, note["quote"]) is not None:
        return "strict_locatable"

    compact_source = _compact_alphanumeric(source)
    segments = [
        _compact_alphanumeric(segment)
        for segment in _ELLIPSIS.split(note["quote"])
        if _compact_alphanumeric(segment)
    ]
    if (
        len(segments) >= 2
        and sum(segment in compact_source for segment in segments) >= 2
    ):
        return "noncontiguous_composite"

    compact_quote = _compact_alphanumeric(note["quote"])
    if compact_quote and compact_quote in compact_source:
        return "format_only"
    return "wording_change"


def test_real_run_fixture_is_self_contained_and_byte_stable() -> None:
    fixture = _load_fixture()
    expectations = fixture["expectations"]
    records = fixture["notes"]
    sources = fixture["source_cache"]

    assert fixture["schema_version"] == 1
    assert fixture["provenance"]["source_run_id"] == (
        "d16bea2078ce42ff80c214130e060842"
    )
    assert len(sources) == expectations["source_count"] == 4
    assert len(records) == expectations["note_count"] == 95
    assert [record["note_index"] for record in records] == list(range(95))
    assert all(record["note"]["url"] in sources for record in records)
    assert all(
        ResearchNote.model_validate(record["note"])
        for record in records
    )

    canonical_notes = json.dumps(
        [record["note"] for record in records],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert hashlib.sha256(canonical_notes).hexdigest() == (
        fixture["provenance"]["notes_sha256"]
    )
    assert {
        url: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for url, text in sources.items()
    } == expectations["source_sha256"]


def test_shared_strict_locator_reproduces_18_of_95_baseline() -> None:
    fixture = _load_fixture()
    records = fixture["notes"]
    sources = fixture["source_cache"]
    expectations = fixture["expectations"]

    strict_indices: list[int] = []
    for record in records:
        note = record["note"]
        located = locate_source_quote(sources[note["url"]], note["quote"])
        if located is None:
            assert note["location_status"] == "unlocatable"
            assert note["span"] is None
            continue

        strict_indices.append(record["note_index"])
        assert note["location_status"] == "locatable"
        assert note["span"] == {
            "start_char": located.start_char,
            "end_char": located.end_char,
        }
        assert sources[note["url"]][located.start_char : located.end_char] == (
            note["quote"]
        )

    assert strict_indices == expectations["strict_locatable_indices"]
    assert len(strict_indices) == expectations["strict_locatable_count"] == 18


def test_frozen_failure_classes_partition_all_77_strict_misses() -> None:
    fixture = _load_fixture()
    sources = fixture["source_cache"]
    records = fixture["notes"]
    expectations = fixture["expectations"]

    actual_by_class: dict[str, list[int]] = {
        "strict_locatable": [],
        "noncontiguous_composite": [],
        "format_only": [],
        "wording_change": [],
    }
    for record in records:
        note = record["note"]
        classification = _test_only_classification(
            note,
            sources[note["url"]],
        )
        assert classification == record["expected_classification"]
        actual_by_class[classification].append(record["note_index"])

    assert actual_by_class["strict_locatable"] == (
        expectations["strict_locatable_indices"]
    )
    assert actual_by_class["noncontiguous_composite"] == (
        expectations["noncontiguous_composite_indices"]
    )
    assert actual_by_class["format_only"] == (
        expectations["format_only_indices"]
    )
    assert actual_by_class["wording_change"] == (
        expectations["wording_change_indices"]
    )
    assert {
        classification: len(indices)
        for classification, indices in actual_by_class.items()
    } == {
        "strict_locatable": 18,
        "noncontiguous_composite": 46,
        "format_only": 26,
        "wording_change": 5,
    }


def test_fixture_marks_only_unique_contiguous_format_repair_candidates() -> None:
    fixture = _load_fixture()
    sources = fixture["source_cache"]
    records = fixture["notes"]
    expectations = fixture["expectations"]

    unique_format_indices: list[int] = []
    ambiguous_format_indices: list[int] = []
    for record in records:
        note = record["note"]
        source = sources[note["url"]]
        occurrence_count = _compact_alphanumeric(source).count(
            _compact_alphanumeric(note["quote"])
        )
        classification = record["expected_classification"]
        if classification == "format_only":
            if occurrence_count == 1:
                unique_format_indices.append(record["note_index"])
                assert record["expected_unique_format_match"] is True
            else:
                ambiguous_format_indices.append(record["note_index"])
                assert record["expected_unique_format_match"] is False
                assert occurrence_count > 1
        elif classification in {
            "noncontiguous_composite",
            "wording_change",
        }:
            assert occurrence_count == 0
            assert record["expected_unique_format_match"] is False

    assert unique_format_indices == expectations["unique_format_match_indices"]
    assert len(unique_format_indices) == (
        expectations["unique_format_match_count"]
    ) == 25
    assert ambiguous_format_indices == (
        expectations["ambiguous_format_match_indices"]
    ) == [68]
    assert expectations["ambiguous_format_match_count"] == 1


def test_fixture_preserves_note_volume_per_source() -> None:
    fixture = _load_fixture()
    counts = Counter(
        record["note"]["url"]
        for record in fixture["notes"]
    )

    assert dict(counts) == fixture["expectations"]["notes_per_url"]


def test_conservative_repair_preserves_all_four_frozen_failure_classes() -> None:
    fixture = _load_fixture()
    records = fixture["notes"]
    sources = fixture["source_cache"]
    expectations = fixture["expectations"]

    notes_by_index = {
        record["note_index"]: create_note(
            item_id=record["note"]["item_id"],
            finding=record["note"]["finding"],
            quote=record["note"]["quote"],
            url=record["note"]["url"],
            source_text=sources[record["note"]["url"]],
        )
        for record in records
    }
    repaired_indices = [
        index
        for index, note in notes_by_index.items()
        if note.location_status is NoteLocationStatus.REPAIRED_LOCATABLE
    ]

    assert repaired_indices == expectations["unique_format_match_indices"]
    assert len(repaired_indices) == 25
    for index in repaired_indices:
        note = notes_by_index[index]
        assert note.repair_method is (
            QuoteRepairMethod.NFKC_CASEFOLD_ALNUM_UNIQUE_CONTIGUOUS
        )
        assert note.span is not None
        source = sources[note.url]
        assert source[note.span.start_char : note.span.end_char] == (
            note.source_quote
        )

    ambiguous = notes_by_index[68]
    assert ambiguous.location_status is NoteLocationStatus.UNLOCATABLE
    assert ambiguous.failure_reason is (
        QuoteFailureReason.AMBIGUOUS_FORMAT_MATCH
    )
    assert ambiguous.source_quote is None

    for index in expectations["noncontiguous_composite_indices"]:
        note = notes_by_index[index]
        assert note.location_status is NoteLocationStatus.UNLOCATABLE
        assert note.failure_reason is (
            QuoteFailureReason.NONCONTIGUOUS_COMPOSITE
        )
        assert note.located_fragment_count >= 2
        assert note.source_quote is None

    for index in expectations["wording_change_indices"]:
        note = notes_by_index[index]
        assert note.location_status is NoteLocationStatus.UNLOCATABLE
        assert note.failure_reason is QuoteFailureReason.QUOTE_NOT_FOUND
        assert note.source_quote is None

    # Fragment diagnosis annotates the original 46 notes; it never materializes
    # fragments as additional notes or usable evidence.
    assert len(notes_by_index) == expectations["note_count"] == 95
    assert sum(
        note.failure_reason
        is QuoteFailureReason.NONCONTIGUOUS_COMPOSITE
        for note in notes_by_index.values()
    ) == 46


def test_fixture_grounding_metrics_keep_strict_repair_and_usable_rates_separate():
    fixture = _load_fixture()
    sources = fixture["source_cache"]
    notes = [
        create_note(
            item_id=record["note"]["item_id"],
            finding=record["note"]["finding"],
            quote=record["note"]["quote"],
            url=record["note"]["url"],
            source_text=sources[record["note"]["url"]],
        )
        for record in fixture["notes"]
    ]

    assert quote_quality_metrics(notes) == {
        "note_count": 95,
        "strict_locatable_count": 18,
        "strict_locatable_rate": 0.189474,
        "repaired_locatable_count": 25,
        "format_repair_rate": 0.263158,
        "usable_source_span_count": 43,
        "usable_source_span_rate": 0.452632,
        "noncontiguous_composite_count": 46,
        "noncontiguous_composite_rate": 0.484211,
    }
