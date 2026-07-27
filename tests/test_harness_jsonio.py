"""Model output arrives wrapped; decoding tolerates packaging, not invention."""

import json

import pytest

from open_deep_research.harness.jsonio import loads_lenient


def test_bare_json_object_is_decoded():
    assert loads_lenient('{"action":"stop"}') == {"action": "stop"}


def test_fenced_json_is_decoded():
    text = '```json\n{"action":"stop"}\n```'
    assert loads_lenient(text) == {"action": "stop"}


def test_unlabelled_fence_is_decoded():
    assert loads_lenient('```\n{"notes":[]}\n```') == {"notes": []}


def test_json_surrounded_by_prose_is_decoded():
    text = 'Here is the action you asked for:\n{"action":"stop"}\nLet me know.'
    assert loads_lenient(text) == {"action": "stop"}


def test_top_level_array_is_decoded():
    assert loads_lenient('```json\n[{"item_id":"a"}]\n```') == [{"item_id": "a"}]


def test_nested_braces_survive_span_extraction():
    text = 'prose {"notes":[{"item_id":"a","finding":"b"}]} trailing'
    assert loads_lenient(text) == {"notes": [{"item_id": "a", "finding": "b"}]}


def test_undecodable_text_still_raises():
    # Tolerance must not become invention: unusable output stays an error.
    with pytest.raises(json.JSONDecodeError):
        loads_lenient("no json here at all")


def test_empty_text_raises():
    with pytest.raises(json.JSONDecodeError):
        loads_lenient("   ")
