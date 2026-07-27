"""Frozen fifth-run evidence for the post-hoc attribution pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from open_deep_research.graphrag.validation.grounding import (
    locate_source_quote,
)
from open_deep_research.harness.notes import ResearchNote
from open_deep_research.harness.write import parse_report_citations


_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "harness_posthoc_b1407b.json"
)
_CONTRACT_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "open_deep_research"
    / "harness"
    / "POSTHOC_EVIDENCE_CONTRACT.md"
)
_REFERENCE_DEFINITION = re.compile(
    r"^\[\^([^]]+)\]:\s*(.*)$"
)


def _load_fixture() -> dict:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reference_definitions(markdown: str) -> list[dict]:
    definitions: list[dict] = []
    for line_number, line in enumerate(markdown.splitlines(), 1):
        match = _REFERENCE_DEFINITION.match(line)
        if match is None:
            continue
        decoded = json.loads(match.group(2))
        definitions.append(
            {
                "line_number": line_number,
                "quote": decoded["quote"],
                "reference_id": match.group(1),
                "url": decoded["url"],
            }
        )
    return definitions


def test_fifth_run_fixture_is_self_contained_and_byte_stable() -> None:
    fixture = _load_fixture()
    expectations = fixture["expectations"]
    notes = fixture["notes"]
    sources = fixture["source_cache"]
    report = fixture["report_markdown"]

    assert fixture["schema_version"] == 1
    assert fixture["provenance"]["source_run_id"] == (
        "b1407b310c804b46aa9c5e5e6a163678"
    )
    assert fixture["contract_versions"] == {
        "artifact_rendering": "single-report-v1",
        "claim_granularity": "atomic-v1",
        "posthoc_evidence": "1",
    }
    assert len(notes) == expectations["note_count"] == 32
    assert len(sources) == expectations["source_count"] == 4
    assert all(ResearchNote.model_validate(note) for note in notes)
    assert all(note["url"] in sources for note in notes)

    assert _canonical_sha(notes) == expectations["notes_sha256"]
    assert hashlib.sha256(report.encode("utf-8")).hexdigest() == (
        fixture["provenance"]["source_report_sha256"]
    )
    assert {
        url: hashlib.sha256(text.encode("utf-8")).hexdigest()
        for url, text in sources.items()
    } == expectations["source_sha256"]


def test_fifth_run_reproduces_collection_and_terminal_state() -> None:
    fixture = _load_fixture()
    snapshot = fixture["run_snapshot"]
    expectations = fixture["expectations"]
    items = snapshot["checklist"]["items"]

    assert snapshot["round_count"] == expectations["round_count"] == 16
    assert snapshot["action_counts"] == expectations["action_counts"] == {
        "read": 4,
        "search": 11,
        "stop": 1,
    }
    assert len(items) == expectations["checklist_item_count"] == 7
    assert Counter(item["status"] for item in items) == (
        expectations["terminal_status_counts"]
    )
    assert snapshot["stop"]["reason"] == "all_items_terminal"
    assert snapshot["stop"]["is_success"] is True
    assert snapshot["stop"]["open_item_ids"] == []
    assert {
        item["item_id"]
        for item in items
        if item["status"] == "exhausted_not_found"
    } == {"how-01", "how-02"}
    assert (
        snapshot["collection_summary"][
            "settled_without_located_evidence_item_ids"
        ]
        == expectations["settled_without_located_evidence_item_ids"]
        == ["where-01"]
    )


def test_fifth_run_reproduces_26_strict_and_6_unlocatable_notes() -> None:
    fixture = _load_fixture()
    sources = fixture["source_cache"]
    notes = fixture["notes"]
    expectations = fixture["expectations"]

    actual_statuses: Counter[str] = Counter()
    for note in notes:
        located = locate_source_quote(
            sources[note["url"]],
            note["model_quote"],
        )
        actual_statuses[note["location_status"]] += 1
        if note["location_status"] == "locatable":
            assert located is not None
            assert note["span"] == {
                "start_char": located.start_char,
                "end_char": located.end_char,
            }
            assert note["source_quote"] == (
                sources[note["url"]][
                    located.start_char : located.end_char
                ]
            )
        else:
            assert note["location_status"] == "unlocatable"
            assert located is None
            assert note["source_quote"] is None
            assert note["span"] is None

    assert actual_statuses == expectations["location_status_counts"]
    assert actual_statuses["locatable"] == (
        expectations["strict_locatable_count"]
    ) == 26
    assert actual_statuses["unlocatable"] == (
        expectations["unlocatable_count"]
    ) == 6
    assert Counter(note["url"] for note in notes) == (
        expectations["notes_per_url"]
    )
    assert Counter(note["item_id"] for note in notes) == (
        expectations["notes_per_item"]
    )
    assert Counter(note["publisher"] for note in notes) == (
        expectations["notes_per_publisher"]
    )


def test_fixture_freezes_the_legacy_duplicate_footnote_failure() -> None:
    fixture = _load_fixture()
    report = fixture["report_markdown"]
    sources = fixture["source_cache"]
    expectations = fixture["expectations"]
    definitions = _reference_definitions(report)

    assert definitions == expectations["reference_definitions"]
    assert len(definitions) == (
        expectations["report_reference_definition_count"]
    ) == 33
    assert len({item["reference_id"] for item in definitions}) == (
        expectations["report_unique_reference_id_count"]
    ) == 25
    assert Counter(item["reference_id"] for item in definitions) == {
        **{
            str(index): 1
            for index in range(1, 26)
            if str(index)
            not in expectations["duplicated_reference_id_counts"]
        },
        **expectations["duplicated_reference_id_counts"],
    }
    assert expectations["duplicated_reference_id_counts"] == {
        "11": 2,
        "13": 3,
        "14": 3,
        "15": 3,
        "16": 2,
    }
    assert expectations[
        "duplicated_reference_distinct_definition_counts"
    ] == {
        "11": 2,
        "13": 2,
        "14": 2,
        "15": 2,
        "16": 2,
    }

    strict_count = sum(
        locate_source_quote(
            sources[definition["url"]],
            definition["quote"],
        )
        is not None
        for definition in definitions
    )
    assert strict_count == (
        expectations["report_strict_locatable_definition_count"]
    ) == 26
    assert len(definitions) - strict_count == (
        expectations["report_unlocatable_definition_count"]
    ) == 7
    assert len(
        {
            (definition["quote"], definition["url"])
            for definition in definitions
        }
    ) == expectations["report_unique_quote_url_pair_count"] == 30

    parsed = parse_report_citations(report)
    assert len(parsed.citations) == (
        expectations["legacy_parser_resolved_citation_count"]
    ) == 26
    assert Counter(issue.reason for issue in parsed.unresolved_claims) == (
        expectations["legacy_parser_issue_counts"]
    )


def test_atomic_v1_examples_are_unique_verbatim_report_anchors() -> None:
    fixture = _load_fixture()
    contract = fixture["granularity_contract"]
    report = fixture["report_markdown"]
    cases = contract["cases"]

    assert contract["version"] == "atomic-v1"
    assert [case["case_id"] for case in cases] == [
        "selection-narrative-01",
        "atomic-compound-01",
        "decontextualization-pronoun-01",
        "atomic-timeline-01",
    ]
    assert [len(case["claim_texts"]) for case in cases] == [0, 4, 1, 6]
    assert all(report.count(case["anchor_text"]) == 1 for case in cases)
    assert (
        cases[2]["claim_texts"][0] != cases[2]["anchor_text"]
    )
    assert "Sam Bankman-Fried" in cases[2]["claim_texts"][0]
    assert "His parents" in cases[2]["anchor_text"]


def test_contract_has_one_annotated_report_and_no_evaluation_variant() -> None:
    contract = _CONTRACT_PATH.read_text(encoding="utf-8")

    assert "one reader-facing report, `<run_id>.md`" in contract
    assert "plus its audit JSON" in contract
    assert "There is no second report variant" in contract
    assert "report.evaluation.md" not in contract
    assert "may not be removed or weakened" in contract
    assert "canonical narrative draft without citations" in contract
    assert "note/source handles" in contract
    assert "may emit stable note handles" not in contract
    assert "no more than 20 claims" in contract
    assert "full cached source text" in contract
