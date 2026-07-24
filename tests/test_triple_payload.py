"""Tests for liberal triple-payload parsing.

Every shape marked "observed" was produced by openai/gpt-4.1-mini on 2026-07-24
during the first live run, where the bare-object case silently zeroed out five
rounds of extraction.
"""

from __future__ import annotations

import pytest

from open_deep_research.graphrag.extraction.triple_payload import parse_triple_payload

SBF = {"subject": "Sam Bankman-Fried", "predicate": "founded", "object": "FTX"}


def test_bare_single_triple_object() -> None:
    """Observed, and the one that broke the live run.

    Read as "nothing found", it is indistinguishable from a passage that had
    nothing -- so the failure looked like clean, honest misses.
    """

    raw = '{"subject":"Sam Bankman-Fried","predicate":"founded","object":"FTX"}'
    assert parse_triple_payload(raw) == [SBF]


def test_list_under_a_triples_key() -> None:
    """Observed with a shorter system prompt."""

    raw = '{"triples":[{"subject":"Sam Bankman-Fried","predicate":"founded","object":"FTX"}]}'
    assert parse_triple_payload(raw) == [SBF]


def test_fenced_bare_array() -> None:
    """Observed without response_format."""

    raw = '```json\n[{"subject":"Sam Bankman-Fried","predicate":"founded","object":"FTX"}]\n```'
    assert parse_triple_payload(raw) == [SBF]


@pytest.mark.parametrize("key", ["facts", "results", "items", "data", "output", "extractions"])
def test_other_plausible_wrapper_keys(key: str) -> None:
    raw = '{"%s":[{"subject":"Sam Bankman-Fried","predicate":"founded","object":"FTX"}]}' % key
    assert parse_triple_payload(raw) == [SBF]


def test_unknown_key_with_a_single_list_is_still_accepted() -> None:
    """Losing an extraction to a naming choice would be a silly way to fail."""

    raw = '{"whatever_the_model_called_it":[{"subject":"A","predicate":"b","object":"C"}]}'
    assert parse_triple_payload(raw) == [{"subject": "A", "predicate": "b", "object": "C"}]


def test_ambiguous_multiple_lists_are_refused() -> None:
    raw = '{"a":[{"subject":"A","predicate":"b","object":"C"}],"b":[{"subject":"D","predicate":"e","object":"F"}]}'
    assert parse_triple_payload(raw) == []


def test_empty_array_means_no_facts() -> None:
    assert parse_triple_payload("[]") == []
    assert parse_triple_payload('{"triples":[]}') == []


def test_incomplete_triples_are_dropped() -> None:
    """A half-formed fact is not worth guessing at on the way into the graph."""

    raw = """[
        {"subject":"A","predicate":"b","object":"C"},
        {"subject":"D","predicate":"e"},
        {"subject":"","predicate":"f","object":"G"},
        {"subject":"H","predicate":"i","object":"   "}
    ]"""
    assert parse_triple_payload(raw) == [{"subject": "A", "predicate": "b", "object": "C"}]


def test_values_are_stripped() -> None:
    raw = '{"subject":"  FTX  ","predicate":" filed for ","object":" Chapter 11 "}'
    assert parse_triple_payload(raw) == [
        {"subject": "FTX", "predicate": "filed for", "object": "Chapter 11"}
    ]


@pytest.mark.parametrize("raw", [None, "", "   ", "not json", "{broken", "42", '"a string"'])
def test_unparseable_input_yields_nothing(raw) -> None:
    assert parse_triple_payload(raw) == []


def test_non_dict_entries_are_ignored() -> None:
    raw = '["a string", 42, null, {"subject":"A","predicate":"b","object":"C"}]'
    assert parse_triple_payload(raw) == [{"subject": "A", "predicate": "b", "object": "C"}]
