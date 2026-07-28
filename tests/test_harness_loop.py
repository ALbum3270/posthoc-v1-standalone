import asyncio
import json

import pytest

from open_deep_research.harness.checklist import (
    ChecklistDimension,
    ChecklistItem,
    ChecklistStatus,
    ResearchChecklist,
)
from open_deep_research.harness.ledger import ResearchLedger
from open_deep_research.harness.loop import (
    LoopBudget,
    LoopSettings,
    StopReason,
    build_note_prompt,
    run_research_loop,
)
from open_deep_research.harness.notes import NoteLocationStatus, create_note


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("model script exhausted")
        return self.responses.pop(0)


class FakeTavily:
    def __init__(self, *, raw_text="A useful exact sentence."):
        self.raw_text = raw_text
        self.search_calls = []
        self.extract_calls = []

    async def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return {
            "results": [
                {
                    "title": "Candidate",
                    "url": "https://example.com/source",
                    "content": "Candidate snippet",
                    "score": 0.8,
                }
            ]
        }

    async def extract(self, urls, **kwargs):
        self.extract_calls.append((urls, kwargs))
        raw_text = (
            self.raw_text[urls[0]]
            if isinstance(self.raw_text, dict)
            else self.raw_text
        )
        return {
            "results": [
                {
                    "url": urls[0],
                    "raw_content": raw_text,
                }
            ]
        }


class FailingExtractTavily(FakeTavily):
    async def extract(self, urls, **kwargs):
        self.extract_calls.append((urls, kwargs))
        raise ValueError("the candidate URL is not fetchable")


def envelope(content, *, tokens=1, cost=0.01):
    return {
        "content": content,
        "token_count": tokens,
        "cost_usd": cost,
    }


def decision_state(prompt):
    return json.loads(prompt.split("Current collection state:\n", 1)[1])


def checklist(*, second=True):
    items = [
        ChecklistItem(
            item_id="what-1",
            dimension=ChecklistDimension.WHAT,
            question="What happened?",
            priority=1,
            required_source_count=1,
        )
    ]
    if second:
        items.append(
            ChecklistItem(
                item_id="how-1",
                dimension=ChecklistDimension.HOW,
                question="How did it happen?",
                priority=1,
                required_source_count=1,
            )
        )
    return ResearchChecklist(topic="A neutral topic", items=tuple(items))


def checklist_with_item_ids(*item_ids):
    dimensions = tuple(ChecklistDimension)
    return ResearchChecklist(
        topic="A neutral topic",
        items=tuple(
            ChecklistItem(
                item_id=item_id,
                dimension=dimensions[index % len(dimensions)],
                question=f"Question {index}?",
                priority=index + 1,
                required_source_count=1,
            )
            for index, item_id in enumerate(item_ids)
        ),
    )


def run_loop(
    decisions,
    *,
    notes=(),
    active_checklist=None,
    active_ledger=None,
    tavily=None,
    budget=None,
    settings=None,
):
    decision_model = ScriptedModel(decisions)
    note_model = ScriptedModel(notes)
    ledger = active_ledger or ResearchLedger(topic="A neutral topic")
    client = tavily or FakeTavily()
    result = asyncio.run(
        run_research_loop(
            active_checklist or checklist(),
            ledger=ledger,
            decision_model=decision_model,
            note_model=note_model,
            tavily_client=client,
            budget=budget,
            settings=settings,
        )
    )
    return result, decision_model, note_model, client


def test_terminal_completion_and_model_stop_are_distinct_outcomes():
    completed, _, _, _ = run_loop(
        [
            envelope(
                {"action": "settle", "item_id": "what-1"}
            ),
            envelope(
                {
                    "action": "search",
                    "item_id": "how-1",
                    "query": "a bounded neutral query",
                }
            ),
            envelope(
                {
                    "action": "mark_exhausted",
                    "item_id": "how-1",
                    "reason": "Reasonable searches found nothing",
                }
            ),
        ]
    )

    assert completed.stop_reason is StopReason.ALL_ITEMS_TERMINAL
    assert completed.is_success is True
    assert completed.checklist.get("what-1").status is ChecklistStatus.SETTLED
    assert (
        completed.checklist.get("how-1").status
        is ChecklistStatus.EXHAUSTED_NOT_FOUND
    )
    assert completed.ledger.rounds[-1].action == "mark_exhausted"
    assert json.loads(completed.ledger.rounds[-1].result_summary)["stop_reason"] == (
        "all_items_terminal"
    )

    stopped, _, _, _ = run_loop([envelope({"action": "stop"})])

    assert stopped.stop_reason is StopReason.MODEL_STOP_WITH_OPEN_ITEMS
    assert stopped.is_success is False
    assert stopped.open_item_ids == ("what-1", "how-1")
    assert stopped.ledger.rounds[-1].action == "stop"
    stop_audit = json.loads(stopped.ledger.rounds[-1].result_summary)
    assert stop_audit["stop_reason"] == "model_stop_with_open_items"
    assert stop_audit["open_item_ids"] == ["what-1", "how-1"]


@pytest.mark.parametrize(
    ("budget", "decision"),
    [
        (
            LoopBudget(max_rounds=1, max_tokens=100, max_cost_usd=10),
            envelope(
                {"action": "search", "item_id": "what-1", "query": "query"}
            ),
        ),
        (
            LoopBudget(max_rounds=10, max_tokens=2, max_cost_usd=10),
            envelope({"action": "stop"}, tokens=2),
        ),
        (
            LoopBudget(max_rounds=10, max_tokens=100, max_cost_usd=0.25),
            envelope({"action": "stop"}, cost=0.25),
        ),
    ],
)
def test_each_hard_budget_stops_as_budget_exhausted(budget, decision):
    result, model, _, _ = run_loop([decision], budget=budget)

    assert result.stop_reason is StopReason.BUDGET_EXHAUSTED
    assert result.is_success is False
    assert len(model.prompts) == 1
    assert (
        json.loads(result.ledger.rounds[-1].result_summary)["stop_reason"]
        == "budget_exhausted"
    )


def test_malformed_actions_are_billed_and_stop_at_the_consecutive_limit():
    result, model, _, _ = run_loop(
        [
            envelope("{", tokens=2, cost=0.1),
            envelope("not json", tokens=3, cost=0.2),
            envelope('{"action":"unknown"}', tokens=4, cost=0.3),
            envelope({"action": "stop"}),
        ]
    )

    assert result.stop_reason is StopReason.MALFORMED_ACTION_LIMIT
    assert result.is_success is False
    assert len(model.prompts) == 3
    assert result.ledger.total_tokens == 9
    assert result.ledger.total_cost_usd == pytest.approx(0.6)
    assert [record.action for record in result.ledger.rounds] == [
        "invalid_action",
        "invalid_action",
        "invalid_action",
    ]
    assert (
        json.loads(result.ledger.rounds[-1].result_summary)["stop_reason"]
        == "malformed_action_limit"
    )


def test_valid_action_resets_the_malformed_streak():
    result, model, _, _ = run_loop(
        [
            envelope("{"),
            envelope({"action": "settle", "item_id": "what-1"}),
        ],
        active_checklist=checklist(second=False),
    )

    assert len(model.prompts) == 2
    assert result.stop_reason is StopReason.ALL_ITEMS_TERMINAL


def test_cached_read_returns_note_ids_without_rerunning_note_model():
    source = (
        "A useful exact sentence.\n\n"
        "A second exact sentence.\n\n"
        "A private-source-marker that no note quotes."
    )
    tavily = FakeTavily(raw_text=source)
    decisions = [
        envelope(
            {
                "action": "read",
                "item_id": "what-1",
                "url": "https://example.com/source",
            },
            tokens=2,
            cost=0.1,
        ),
        envelope(
            {
                "action": "read",
                "item_id": "what-1",
                "url": "https://example.com/source",
            },
            tokens=2,
            cost=0.1,
        ),
        envelope({"action": "stop"}),
    ]
    note_outputs = [
        envelope(
            {
                "active_notes": [],
                "cross_item_seeds": [
                    {
                        "item_id": "how-1",
                        "finding": "The source also answers another item.",
                        "start_segment_id": "S000001",
                        "end_segment_id": "S000001",
                    }
                ]
            },
            tokens=3,
            cost=0.2,
        ),
    ]

    result, decision_model, note_model, client = run_loop(
        decisions,
        notes=note_outputs,
        tavily=tavily,
    )

    assert len(client.extract_calls) == 1
    assert len(note_model.prompts) == 1
    assert all(
        all(
            passage in prompt
            for passage in (
                "A useful exact sentence.",
                "A second exact sentence.",
                "A private-source-marker that no note quotes.",
            )
        )
        for prompt in note_model.prompts
    )
    assert (
        'Active item:\n{"item_id": "what-1", "question": "What happened?"}'
        in note_model.prompts[0]
    )
    assert all(
        "private-source-marker" not in prompt
        for prompt in decision_model.prompts
    )
    assert '"consecutive_collection_failures": 1' in decision_model.prompts[1]
    assert [note.item_id for note in result.ledger.notes] == ["how-1"]
    assert result.consecutive_collection_failures["what-1"] == 2
    assert result.consecutive_failures["how-1"] == 0
    assert result.checklist.get("how-1").status is ChecklistStatus.HAS_MATERIAL
    assert result.checklist.get("what-1").status is ChecklistStatus.UNEXPLORED
    assert json.loads(result.ledger.rounds[0].result_summary)["cache_hit"] is False
    cache_audit = json.loads(result.ledger.rounds[1].result_summary)
    assert cache_audit["cache_hit"] is True
    assert cache_audit["note_model_called"] is False
    assert cache_audit["existing_note_ids"] == ["note-000001"]
    assert cache_audit["existing_notes_by_item"] == {
        "how-1": {"count": 1, "note_ids": ["note-000001"]}
    }
    assert result.ledger.rounds[0].token_count == 5
    assert result.ledger.rounds[0].cost_usd == pytest.approx(0.3)
    assert result.ledger.rounds[1].token_count == 2
    assert result.ledger.rounds[1].cost_usd == pytest.approx(0.1)
    # Source bodies stay out of the decision context until the model recalls one.
    assert all(
        json.loads(record.result_summary)["decision_context"]["recalled_urls"] == []
        for record in result.ledger.rounds
    )


def test_model_recalled_sources_are_injected_newest_first_and_audited():
    older_url = "https://example.com/older"
    recent_url = "https://example.com/recent"
    tavily = FakeTavily(
        raw_text={
            older_url: "AAAAAA",
            recent_url: "BBBBBB",
        }
    )
    decisions = [
        envelope({"action": "read", "item_id": "what-1", "url": older_url}),
        envelope({"action": "read", "item_id": "what-1", "url": recent_url}),
        # The model decides summaries are not enough and asks for both bodies.
        envelope({"action": "recall", "item_id": "what-1", "url": older_url}),
        envelope({"action": "recall", "item_id": "what-1", "url": recent_url}),
        envelope({"action": "stop"}),
    ]

    result, decision_model, _, _ = run_loop(
        decisions,
        notes=[
            envelope({"active_notes": [], "cross_item_seeds": []}),
            envelope({"active_notes": [], "cross_item_seeds": []}),
        ],
        tavily=tavily,
        settings=LoopSettings(decision_source_char_limit=8),
    )

    # Nothing is injected before the first recall lands.
    assert "BBBBBB" not in decision_model.prompts[2]

    final_prompt = decision_model.prompts[-1]
    assert '"content": "BBBBBB"' in final_prompt
    assert '"content": "AA"' in final_prompt
    assert "AAAAAA" not in final_prompt
    final_state = decision_state(final_prompt)
    assert [
        source["url"] for source in final_state["recalled_sources"]
    ] == [recent_url, older_url]

    context_audit = json.loads(
        result.ledger.rounds[-1].result_summary
    )["decision_context"]
    assert context_audit == {
        "recalled_urls": [recent_url, older_url],
        "char_limit": 8,
        "injected_chars": 8,
        "injected_sources": [
            {
                "url": recent_url,
                "source_chars": 6,
                "injected_chars": 6,
                "truncated": False,
            },
            {
                "url": older_url,
                "source_chars": 6,
                "injected_chars": 2,
                "truncated": True,
            },
        ],
        "omitted_urls": [],
        "truncated": True,
    }


def test_recalling_an_unread_url_is_recorded_as_a_miss_not_a_crash():
    decisions = [
        envelope(
            {
                "action": "recall",
                "item_id": "what-1",
                "url": "https://example.com/never-read",
            }
        ),
        envelope({"action": "stop"}),
    ]

    result, decision_model, _, _ = run_loop(decisions)

    summary = json.loads(result.ledger.rounds[0].result_summary)
    assert summary["recalled"] is False
    assert summary["detail"] == "url is not in the source cache"
    assert result.consecutive_collection_failures["what-1"] == 0
    # The run continues; a bad recall is not a malformed action.
    assert result.stop_reason is StopReason.MODEL_STOP_WITH_OPEN_ITEMS
    assert "never-read" not in decision_model.prompts[-1].split('"recalled_urls"')[0]


def test_zero_note_batch_is_valid_audited_and_counts_as_active_item_failure():
    decisions = [
        envelope(
            {
                "action": "read",
                "item_id": "what-1",
                "url": "https://example.com/source",
            }
        ),
        envelope({"action": "stop"}),
    ]
    note_outputs = [
        envelope(
            {"active_notes": [], "cross_item_seeds": []},
            tokens=5,
            cost=0.4,
        )
    ]

    result, _, _, _ = run_loop(decisions, notes=note_outputs)

    assert result.stop_reason is StopReason.MODEL_STOP_WITH_OPEN_ITEMS
    assert result.ledger.notes == []
    assert result.consecutive_failures["what-1"] == 1
    audit = json.loads(result.ledger.rounds[0].result_summary)
    assert audit["notes_created"] == 0
    assert audit["active_item_notes"] == 0
    assert audit["note_output_error"] is None
    assert result.ledger.rounds[0].token_count == 6


def test_note_prompt_has_two_bounded_channels_without_changing_active_pass():
    prompt_checklist = checklist_with_item_ids("what-1", "how-1", "why-1")
    prompt_checklist = prompt_checklist.model_copy(
        update={
            "items": tuple(
                item.model_copy(update={"status": ChecklistStatus.SETTLED})
                if item.item_id == "why-1"
                else item
                for item in prompt_checklist.items
            )
        }
    )
    prompt = build_note_prompt(
        prompt_checklist,
        active_item_id="what-1",
        url="https://example.com/source",
        source_text="A useful exact sentence.",
        max_cross_item_seeds=3,
    )

    assert "active_notes" in prompt
    assert "cross_item_seeds" in prompt
    assert "at most\n3 different items" in prompt
    assert "not a quality threshold" in prompt
    assert "Every entry's\nitem_id must equal the active item_id" in prompt
    assert (
        'Active item:\n{"item_id": "what-1", "question": "Question 0?"}'
        in prompt
    )
    eligible = prompt.split("Eligible cross-item targets:\n", 1)[1].split(
        "\n\nSource URL:", 1
    )[0]
    assert eligible == '[{"item_id": "how-1", "question": "Question 1?"}]'
    assert "what-1" not in eligible
    assert "why-1" not in prompt
    assert '"status"' not in prompt
    assert '"priority"' not in prompt
    assert "Do not copy or generate quote text" in prompt
    assert "one specific finding" in prompt
    assert "shortest continuous" in prompt
    assert "at most 12 segments and\n2000 source characters" in prompt
    assert "not evidence-quality targets" in prompt
    assert "do not\nexpand one range to cover a section" in prompt
    assert "source_text_sha256" in prompt
    assert "segmentation_version" in prompt
    assert "<S000001>A useful exact sentence." in prompt
    assert all(
        term not in prompt.casefold()
        for term in ("receivership", "regulators", "asset_flow")
    )


def test_note_channels_parse_entries_independently_and_cross_does_not_reset_failure():
    source = "Cross evidence is exact."
    decisions = [
        envelope(
            {
                "action": "read",
                "item_id": "what-1",
                "url": "https://example.com/source",
            }
        ),
        envelope({"action": "stop"}),
    ]
    note_outputs = [
        envelope(
            {
                "active_notes": [
                    {
                        "item_id": "what-1",
                        "finding": "Missing pointer makes only this entry bad.",
                    }
                ],
                "cross_item_seeds": [
                    {
                        "item_id": "how-1",
                        "finding": "A valid cross-item seed.",
                        "start_segment_id": "S000001",
                        "end_segment_id": "S000001",
                    }
                ],
            }
        )
    ]

    result, _, _, _ = run_loop(
        decisions,
        notes=note_outputs,
        tavily=FakeTavily(raw_text=source),
    )

    assert [(note.item_id, note.finding) for note in result.ledger.notes] == [
        ("how-1", "A valid cross-item seed.")
    ]
    assert result.consecutive_collection_failures["what-1"] == 1
    assert result.checklist.get("how-1").status is ChecklistStatus.HAS_MATERIAL
    audit = json.loads(result.ledger.rounds[0].result_summary)
    assert audit["active_notes_created"] == 0
    assert audit["cross_item_seeds_created"] == 1
    assert audit["cross_item_seed_item_ids"] == ["how-1"]
    assert len(audit["active_note_errors"]) == 1
    assert audit["cross_item_seed_errors"] == []


def test_note_audit_splits_pointer_location_counts_by_output_channel():
    source = "Exact active sentence.\n\nＡlpha—Beta 2024"
    decisions = [
        envelope(
            {
                "action": "read",
                "item_id": "what-1",
                "url": "https://example.com/source",
            }
        ),
        envelope({"action": "stop"}),
    ]
    note_outputs = [
        envelope(
            {
                "active_notes": [
                    {
                        "item_id": "what-1",
                        "finding": "A strict active note.",
                        "start_segment_id": "S000001",
                        "end_segment_id": "S000001",
                    },
                    {
                        "item_id": "what-1",
                        "finding": "An invalid active pointer.",
                        "start_segment_id": "S999999",
                        "end_segment_id": "S999999",
                    },
                ],
                "cross_item_seeds": [
                    {
                        "item_id": "how-1",
                        "finding": "A strict cross-item seed.",
                        "start_segment_id": "S000002",
                        "end_segment_id": "S000002",
                    }
                ],
            }
        )
    ]

    result, _, _, _ = run_loop(
        decisions,
        notes=note_outputs,
        tavily=FakeTavily(raw_text=source),
    )

    audit = json.loads(result.ledger.rounds[0].result_summary)
    assert audit["note_extraction_mode"] == "segment_pointer"
    assert audit["source_span_registry"]["source_text_sha256"]
    assert audit["source_span_registry"]["segmentation_version"]
    assert audit["active_note_location_counts"] == {
        "strict_locatable": 1,
        "repaired_locatable": 0,
        "unlocatable": 0,
    }
    assert audit["cross_item_seed_location_counts"] == {
        "strict_locatable": 1,
        "repaired_locatable": 0,
        "unlocatable": 0,
    }
    assert any(
        "unknown start_segment_id 'S999999'" in error
        for error in audit["active_note_errors"]
    )


def test_oversized_note_ranges_are_rejected_individually_with_full_audit():
    thirteen_sentences = " ".join(
        f"Compact sentence {index}." for index in range(13)
    )
    long_list_item = "- " + ("x" * 2_001)
    source = (
        thirteen_sentences
        + "\n\n"
        + long_list_item
        + "\n\nA valid compact sentence."
    )
    decisions = [
        envelope(
            {
                "action": "read",
                "item_id": "what-1",
                "url": "https://example.com/source",
            }
        ),
        envelope({"action": "stop"}),
    ]
    note_outputs = [
        envelope(
            {
                "active_notes": [
                    {
                        "item_id": "what-1",
                        "finding": "A section-scale multi-sentence proposal.",
                        "start_segment_id": "S000001",
                        "end_segment_id": "S000013",
                    },
                    {
                        "item_id": "what-1",
                        "finding": "A giant single-segment proposal.",
                        "start_segment_id": "S000014",
                        "end_segment_id": "S000014",
                    },
                    {
                        "item_id": "what-1",
                        "finding": "A compact proposal remains valid.",
                        "start_segment_id": "S000015",
                        "end_segment_id": "S000015",
                    },
                ],
                "cross_item_seeds": [],
            }
        )
    ]

    result, _, _, _ = run_loop(
        decisions,
        notes=note_outputs,
        tavily=FakeTavily(raw_text=source),
    )

    assert [note.finding for note in result.ledger.notes] == [
        "A compact proposal remains valid."
    ]
    audit = json.loads(result.ledger.rounds[0].result_summary)
    assert audit["note_span_capacity"] == {
        "max_segments": 12,
        "max_chars": 2000,
        "provisional_protocol_capacity_not_quality_threshold": True,
    }
    assert audit["note_span_rejections"] == [
        {
            "channel": "active_notes",
            "index": 0,
            "item_id": "what-1",
            "start_segment_id": "S000001",
            "end_segment_id": "S000013",
            "segment_count": 13,
            "char_count": len(thirteen_sentences),
            "max_segments": 12,
            "max_chars": 2000,
            "failure_reasons": ["span_too_many_segments"],
        },
        {
            "channel": "active_notes",
            "index": 1,
            "item_id": "what-1",
            "start_segment_id": "S000014",
            "end_segment_id": "S000014",
            "segment_count": 1,
            "char_count": len(long_list_item),
            "max_segments": 12,
            "max_chars": 2000,
            "failure_reasons": ["span_too_many_chars"],
        },
    ]
    assert audit["active_notes_proposed"] == 3
    assert audit["active_notes_created"] == 1


def test_cross_item_seed_capacity_is_per_distinct_open_item_and_audited():
    active = checklist_with_item_ids(
        "what-1", "who-1", "when-1", "where-1", "why-1", "how-1"
    )
    active = active.model_copy(
        update={
            "items": tuple(
                item.model_copy(update={"status": ChecklistStatus.SETTLED})
                if item.item_id == "how-1"
                else item
                for item in active.items
            )
        }
    )
    sentences = [f"Exact sentence {index}." for index in range(6)]
    decisions = [
        envelope(
            {
                "action": "read",
                "item_id": "what-1",
                "url": "https://example.com/source",
            }
        ),
        envelope({"action": "stop"}),
    ]
    note_outputs = [
        envelope(
            {
                "active_notes": [
                    {
                        "item_id": "what-1",
                        "finding": "Active finding.",
                        "start_segment_id": "S000001",
                        "end_segment_id": "S000001",
                    }
                ],
                "cross_item_seeds": [
                    {
                        "item_id": item_id,
                        "finding": f"Seed {index}.",
                        "start_segment_id": f"S{index + 1:06d}",
                        "end_segment_id": f"S{index + 1:06d}",
                    }
                    for index, item_id in enumerate(
                        ("who-1", "when-1", "where-1", "why-1", "how-1"),
                        start=1,
                    )
                ],
            }
        )
    ]

    result, _, _, _ = run_loop(
        decisions,
        notes=note_outputs,
        active_checklist=active,
        tavily=FakeTavily(raw_text=" ".join(sentences)),
        settings=LoopSettings(max_cross_item_seeds=3),
    )

    assert [note.item_id for note in result.ledger.notes] == [
        "what-1",
        "who-1",
        "when-1",
        "where-1",
    ]
    audit = json.loads(result.ledger.rounds[0].result_summary)
    assert audit["active_notes_created"] == 1
    assert audit["cross_item_seeds_proposed"] == 5
    assert audit["cross_item_seeds_created"] == 3
    assert audit["cross_item_seed_capacity"] == 3
    assert any(
        "exceeds cross-item seed capacity 3" in error
        for error in audit["cross_item_seed_errors"]
    )
    assert any(
        "targets terminal item 'how-1'" in error
        for error in audit["cross_item_seed_errors"]
    )


def test_cross_seed_is_exactly_sliced_and_survives_later_exhaustion():
    source = "The source contains exact cross evidence."
    decisions = [
        envelope(
            {
                "action": "read",
                "item_id": "what-1",
                "url": "https://example.com/source",
            }
        ),
        envelope(
            {
                "action": "search",
                "item_id": "how-1",
                "query": "a bounded neutral query",
            }
        ),
        envelope(
            {
                "status_updates": [
                    {
                        "item_id": "how-1",
                        "status": "exhausted_not_found",
                        "reason": "The bounded search is complete.",
                    }
                ],
                "action": {"action": "stop"},
            }
        ),
    ]
    note_outputs = [
        envelope(
            {
                "active_notes": [],
                "cross_item_seeds": [
                    {
                        "item_id": "how-1",
                        "finding": "An exact seed remains historical.",
                        "start_segment_id": "S000001",
                        "end_segment_id": "S000001",
                    }
                ],
            }
        )
    ]

    result, _, _, _ = run_loop(
        decisions,
        notes=note_outputs,
        tavily=FakeTavily(raw_text=source),
    )

    assert len(result.ledger.notes) == 1
    assert result.ledger.notes[0].location_status is NoteLocationStatus.LOCATABLE
    assert result.ledger.notes[0].source_quote == source
    assert result.ledger.notes[0].model_quote is None
    assert result.checklist.get("how-1").status is (
        ChecklistStatus.EXHAUSTED_NOT_FOUND
    )
    assert result.ledger.notes[0].item_id == "how-1"


def test_search_without_an_active_item_note_is_visible_as_a_failure_next_round():
    decisions = [
        envelope(
            {
                "action": "search",
                "item_id": "what-1",
                "query": "a neutral query",
            }
        ),
        envelope({"action": "stop"}),
    ]

    result, decision_model, _, client = run_loop(decisions)

    assert len(client.search_calls) == 1
    assert result.consecutive_collection_failures["what-1"] == 1
    assert '"consecutive_collection_failures": 1' in decision_model.prompts[1]

    # Searching yields candidates, not notes, so the item is still a failure --
    # but the hits must reach the next decision or the model can only search
    # again, which is the zero-entropy loop this project already measured.
    next_prompt = decision_model.prompts[1]
    assert "Candidate snippet" in next_prompt
    assert '"url": "https://example.com/source"' in next_prompt
    assert '"read": false' in next_prompt


def test_candidates_pending_rejects_a_second_search_for_the_same_item():
    decisions = [
        envelope(
            {
                "action": "search",
                "item_id": "what-1",
                "query": "first neutral query",
            }
        ),
        envelope(
            {
                "action": "search",
                "item_id": "what-1",
                "query": "second neutral query",
            }
        ),
        envelope({"action": "stop"}),
    ]

    result, decision_model, _, client = run_loop(decisions)

    assert len(client.search_calls) == 1
    assert "dismiss_candidates" in decision_model.prompts[1]
    rejected = json.loads(result.ledger.rounds[1].result_summary)
    assert rejected["action_rejected"] is True
    assert rejected["rejection_reason"] == "candidates_pending"
    assert rejected["pending_unread_urls"] == [
        "https://example.com/source"
    ]
    assert rejected["candidate_work"]["what-1"] == {
        "read_count": 0,
        "read_urls": [],
        "dismissed_count": 0,
        "dismissed_candidates": [],
        "unreadable_count": 0,
        "unreadable_candidates": [],
        "pending_unread_count": 1,
        "pending_unread_urls": ["https://example.com/source"],
        "candidates_pending": True,
    }


def test_candidates_pending_allows_read_and_records_the_resolved_state():
    url = "https://example.com/source"
    decisions = [
        envelope(
            {
                "action": "search",
                "item_id": "what-1",
                "query": "a neutral query",
            }
        ),
        envelope({"action": "read", "item_id": "what-1", "url": url}),
        envelope({"action": "stop"}),
    ]
    note_outputs = [
        envelope({"active_notes": [], "cross_item_seeds": []})
    ]

    result, _, _, client = run_loop(decisions, notes=note_outputs)

    assert len(client.search_calls) == 1
    assert len(client.extract_calls) == 1
    read_audit = json.loads(result.ledger.rounds[1].result_summary)
    assert read_audit["candidate_work"]["what-1"] == {
        "read_count": 1,
        "read_urls": [url],
        "dismissed_count": 0,
        "dismissed_candidates": [],
        "unreadable_count": 0,
        "unreadable_candidates": [],
        "pending_unread_count": 0,
        "pending_unread_urls": [],
        "candidates_pending": False,
    }


def test_failed_read_consumes_pending_candidate_and_is_not_called_again():
    url = "https://example.com/source"
    decisions = [
        envelope(
            {
                "action": "search",
                "item_id": "what-1",
                "query": "a neutral query",
            }
        ),
        envelope({"action": "read", "item_id": "what-1", "url": url}),
        envelope({"action": "read", "item_id": "what-1", "url": url}),
        envelope({"action": "stop"}),
    ]
    client = FailingExtractTavily()

    result, decision_model, _, _ = run_loop(decisions, tavily=client)

    assert len(client.extract_calls) == 1
    first_failure = json.loads(result.ledger.rounds[1].result_summary)
    assert first_failure["candidate_marked_unreadable"] is True
    assert first_failure["candidate_work"]["what-1"] == {
        "read_count": 0,
        "read_urls": [],
        "dismissed_count": 0,
        "dismissed_candidates": [],
        "unreadable_count": 1,
        "unreadable_candidates": [
            {
                "url": url,
                "error": (
                    "read failed: the candidate URL is not fetchable"
                ),
            }
        ],
        "pending_unread_count": 0,
        "pending_unread_urls": [],
        "candidates_pending": False,
    }
    repeated = json.loads(result.ledger.rounds[2].result_summary)
    assert repeated["action_rejected"] is True
    assert repeated["rejection_reason"] == "candidate_unreadable"
    assert (
        repeated["acquisition_attempts"]["what-1"]["read_attempts"] == 1
    )
    assert "Never retry a URL listed as unreadable" in (
        decision_model.prompts[2]
    )


def test_dismiss_candidates_with_reasons_allows_the_item_to_search_again():
    url = "https://example.com/source"
    reason = "The snippet does not address the active question."
    decisions = [
        envelope(
            {
                "action": "search",
                "item_id": "what-1",
                "query": "first neutral query",
            }
        ),
        envelope(
            {
                "action": "dismiss_candidates",
                "item_id": "what-1",
                "candidates": [{"url": url, "reason": reason}],
            }
        ),
        envelope(
            {
                "action": "search",
                "item_id": "what-1",
                "query": "second neutral query",
            }
        ),
        envelope({"action": "stop"}),
    ]

    result, _, _, client = run_loop(decisions)

    assert len(client.search_calls) == 2
    dismissal = json.loads(result.ledger.rounds[1].result_summary)
    assert dismissal["dismissed_candidates"] == [
        {"url": url, "reason": reason}
    ]
    assert dismissal["remaining_pending_unread_count"] == 0
    assert dismissal["candidate_work"]["what-1"] == {
        "read_count": 0,
        "read_urls": [],
        "dismissed_count": 1,
        "dismissed_candidates": [{"url": url, "reason": reason}],
        "unreadable_count": 0,
        "unreadable_candidates": [],
        "pending_unread_count": 0,
        "pending_unread_urls": [],
        "candidates_pending": False,
    }
    second_search = json.loads(result.ledger.rounds[2].result_summary)
    assert "action_rejected" not in second_search


def test_switching_items_does_not_clear_pending_and_exhaustion_audits_it():
    decisions = [
        envelope(
            {
                "action": "search",
                "item_id": "what-1",
                "query": "query for the first item",
            }
        ),
        envelope(
            {
                "action": "search",
                "item_id": "how-1",
                "query": "query for the second item",
            }
        ),
        envelope(
            {
                "action": "mark_exhausted",
                "item_id": "what-1",
                "reason": "The model chose to stop work on this item.",
            }
        ),
        envelope({"action": "stop"}),
    ]

    result, decision_model, _, client = run_loop(decisions)

    assert len(client.search_calls) == 2
    before_exhaustion = decision_state(decision_model.prompts[2])
    assert before_exhaustion["candidate_work"]["what-1"][
        "pending_unread_count"
    ] == 1
    assert before_exhaustion["candidate_work"]["how-1"][
        "pending_unread_count"
    ] == 1
    exhausted = json.loads(result.ledger.rounds[2].result_summary)
    assert exhausted["candidate_work"]["what-1"] == {
        "read_count": 0,
        "read_urls": [],
        "dismissed_count": 0,
        "dismissed_candidates": [],
        "unreadable_count": 0,
        "unreadable_candidates": [],
        "pending_unread_count": 1,
        "pending_unread_urls": ["https://example.com/source"],
        "candidates_pending": True,
    }


def test_reanalyze_is_the_only_way_to_rerun_notes_for_a_cached_url():
    url = "https://example.com/source"
    decisions = [
        envelope({"action": "read", "item_id": "what-1", "url": url}),
        envelope(
            {
                "action": "reanalyze",
                "item_id": "what-1",
                "url": url,
                "reason": "The first pass only found cross-item evidence",
            }
        ),
        envelope({"action": "stop"}),
    ]
    note_outputs = [
        envelope(
            {
                "active_notes": [],
                "cross_item_seeds": [
                    {
                        "item_id": "how-1",
                        "finding": "Cross-item evidence.",
                        "start_segment_id": "S000001",
                        "end_segment_id": "S000001",
                    }
                ]
            }
        ),
        envelope(
            {
                "active_notes": [
                    {
                        "item_id": "what-1",
                        "finding": "Active-item evidence.",
                        "start_segment_id": "S000001",
                        "end_segment_id": "S000001",
                    }
                ],
                "cross_item_seeds": [],
            }
        ),
    ]

    result, _, note_model, client = run_loop(decisions, notes=note_outputs)

    assert len(client.extract_calls) == 1
    assert len(note_model.prompts) == 2
    assert all(
        'Active item:\n{"item_id": "what-1", "question": "What happened?"}'
        in prompt
        for prompt in note_model.prompts
    )
    assert [note.item_id for note in result.ledger.notes] == ["how-1", "what-1"]
    reanalyze_audit = json.loads(result.ledger.rounds[1].result_summary)
    assert reanalyze_audit["reanalyze_reason"] == (
        "The first pass only found cross-item evidence"
    )
    assert reanalyze_audit["note_model_called"] is True
    assert result.consecutive_collection_failures["what-1"] == 0


def test_budget_headroom_and_writing_reserve_are_visible_and_enforced():
    budget = LoopBudget(
        max_rounds=5,
        max_tokens=100,
        max_cost_usd=1.0,
        writing_token_reserve=20,
        writing_cost_reserve_usd=0.25,
    )

    result, model, _, _ = run_loop(
        [envelope({"action": "stop"})],
        budget=budget,
    )

    prompt = model.prompts[0]
    assert '"remaining_rounds": 5' in prompt
    assert '"remaining_collection_tokens": 80' in prompt
    assert '"remaining_collection_cost_usd": 0.75' in prompt
    assert '"writing_reserve": {"cost_usd": 0.25, "tokens": 20}' in prompt
    assert "required_source_count" not in prompt
    assert "corroboration_target" not in prompt
    assert result.stop_reason is StopReason.MODEL_STOP_WITH_OPEN_ITEMS

    exhausted, _, _, _ = run_loop(
        [envelope({"action": "stop"}, tokens=80)],
        budget=budget,
    )
    assert exhausted.stop_reason is StopReason.BUDGET_EXHAUSTED


def test_batch_status_updates_audit_settle_evidence_and_keep_success_honest():
    active = checklist()
    active = active.model_copy(
        update={
            "items": (
                *active.items,
                ChecklistItem(
                    item_id="where-1",
                    dimension=ChecklistDimension.WHERE,
                    question="Where did it happen?",
                    priority=2,
                    required_source_count=1,
                ),
            )
        }
    )
    ledger = ResearchLedger(topic=active.topic)
    for url in ("https://one.example/a", "https://two.example/b"):
        source = f"Evidence from {url}."
        ledger.cache_source(url, source)
        ledger.add_note(
            create_note(
                item_id="what-1",
                finding="Located evidence.",
                quote=source,
                url=url,
                source_text=source,
            )
        )

    decision = envelope(
        {
            "status_updates": [
                {
                    "item_id": "what-1",
                    "status": "settled",
                    "reason": "Two publishers answer it",
                },
                {
                    "item_id": "how-1",
                    "status": "settled",
                    "reason": "The model judges it complete",
                },
                {
                    "item_id": "where-1",
                    "status": "exhausted_not_found",
                    "reason": "Searches did not identify a location",
                },
            ],
            "action": {
                "action": "search",
                "item_id": "how-1",
                "query": "must be skipped after terminal updates",
            },
        }
    )

    result, _, _, client = run_loop(
        [
            envelope(
                {
                    "action": "search",
                    "item_id": "where-1",
                    "query": "a bounded location query",
                }
            ),
            decision,
        ],
        active_checklist=active,
        active_ledger=ledger,
    )

    assert result.stop_reason is StopReason.ALL_ITEMS_TERMINAL
    assert result.is_success is True
    assert len(client.search_calls) == 1
    assert ledger.settled_without_located_evidence == 1
    assert ledger.settled_without_located_evidence_item_ids == ("how-1",)
    assert "settled_without_located_evidence=1 (how-1)" in result.stop_detail

    settled = {
        record.item_id: record.settlement_evidence
        for record in ledger.checklist_history
        if record.to_status == "settled"
    }
    assert settled["what-1"].strict_locatable_notes == 2
    assert settled["what-1"].repaired_locatable_notes == 0
    assert settled["what-1"].publisher_count == 2
    assert settled["how-1"].located_notes == 0
    round_audit = json.loads(ledger.rounds[1].result_summary)
    assert round_audit["action_skipped"] is True
    assert len(round_audit["status_updates"]) == 3


def test_zero_attempt_exhaustion_is_rejected_per_item_and_stop_stays_honest():
    decision = envelope(
        {
            "status_updates": [
                {
                    "item_id": "what-1",
                    "status": "settled",
                    "reason": "The model judges this item complete.",
                },
                {
                    "item_id": "how-1",
                    "status": "exhausted_not_found",
                    "reason": "No candidates found or read.",
                },
            ],
            "action": {"action": "stop"},
        }
    )

    result, model, _, client = run_loop([decision])

    assert client.search_calls == []
    assert result.stop_reason is StopReason.MODEL_STOP_WITH_OPEN_ITEMS
    assert result.is_success is False
    assert result.checklist.get("what-1").status is ChecklistStatus.SETTLED
    assert result.checklist.get("how-1").status is ChecklistStatus.UNEXPLORED
    assert result.open_item_ids == ("how-1",)
    assert (
        result.ledger.rejected_exhausted_without_collection_attempt_item_ids
        == ("how-1",)
    )
    assert "rejected_exhausted_without_collection_attempt=1 (how-1)" in (
        result.stop_detail
    )

    audit = json.loads(result.ledger.rounds[0].result_summary)
    assert len(audit["status_updates"]) == 1
    rejected = audit["rejected_status_updates"]
    assert len(rejected) == 1
    assert rejected[0]["rejection_reason"] == "no_prior_collection_attempt"
    assert rejected[0]["reason"] == "No candidates found or read."
    assert rejected[0]["exhaustion_attempts"]["search_attempts"] == 0
    assert rejected[0]["exhaustion_attempts"]["note_count"] == 0
    state = decision_state(model.prompts[0])
    assert state["acquisition_attempts"]["how-1"]["attempt_status"] == (
        "not_attempted"
    )
    assert "not_attempted" in model.prompts[0]
    assert "attempted_no_result" in model.prompts[0]


def test_pending_rejected_search_is_not_an_attempt_and_exhaustion_snapshot_keeps_pending():
    decisions = [
        envelope(
            {
                "action": "search",
                "item_id": "what-1",
                "query": "first bounded query",
            }
        ),
        envelope(
            {
                "action": "search",
                "item_id": "what-1",
                "query": "rejected while pending",
            }
        ),
        envelope(
            {
                "status_updates": [
                    {
                        "item_id": "what-1",
                        "status": "exhausted_not_found",
                        "reason": "The model ends the bounded search.",
                    }
                ],
                "action": {"action": "stop"},
            }
        ),
    ]

    result, _, _, client = run_loop(decisions)

    assert len(client.search_calls) == 1
    assert result.checklist.get("what-1").status is (
        ChecklistStatus.EXHAUSTED_NOT_FOUND
    )
    record = next(
        record
        for record in result.ledger.checklist_history
        if record.item_id == "what-1" and record.accepted
    )
    snapshot = record.exhaustion_attempts
    assert snapshot.search_attempts == 1
    assert snapshot.search_successes == 1
    assert snapshot.search_errors == 0
    assert snapshot.pending_unread_urls == ("https://example.com/source",)
    assert result.ledger.exhausted_with_unread_candidates_item_ids == (
        "what-1",
    )
    assert "exhausted_with_unread_candidates=1 (what-1)" in (
        result.stop_detail
    )
    audit = json.loads(result.ledger.rounds[2].result_summary)
    assert audit["status_updates"][0][
        "exhausted_with_unread_candidates"
    ] is True


def test_zero_result_and_tool_error_searches_both_qualify_as_real_attempts():
    class OutcomeTavily(FakeTavily):
        def __init__(self, *, fail):
            super().__init__()
            self.fail = fail

        async def search(self, query, **kwargs):
            self.search_calls.append((query, kwargs))
            if self.fail:
                raise RuntimeError("provider unavailable")
            return {"results": []}

    for fail in (False, True):
        decisions = [
            envelope(
                {
                    "action": "search",
                    "item_id": "what-1",
                    "query": "a bounded query",
                }
            ),
            envelope(
                {
                    "status_updates": [
                        {
                            "item_id": "what-1",
                            "status": "exhausted_not_found",
                            "reason": "The bounded attempt is complete.",
                        }
                    ],
                    "action": {"action": "stop"},
                }
            ),
        ]
        result, _, _, _ = run_loop(
            decisions,
            active_checklist=checklist(second=False),
            tavily=OutcomeTavily(fail=fail),
        )
        assert result.is_success is True
        snapshot = result.ledger.checklist_history[-1].exhaustion_attempts
        assert snapshot.search_attempts == 1
        assert snapshot.search_successes == (0 if fail else 1)
        assert snapshot.search_errors == (1 if fail else 0)


def test_batch_status_updates_and_collection_action_run_in_the_same_round():
    decisions = [
        envelope(
            {
                "status_updates": [
                    {
                        "item_id": "what-1",
                        "status": "settled",
                        "reason": "The model judges this item complete",
                    }
                ],
                "action": {
                    "action": "search",
                    "item_id": "how-1",
                    "query": "continue collecting for the open item",
                },
            }
        ),
        envelope({"action": "stop"}),
    ]

    result, _, _, client = run_loop(decisions)

    assert len(client.search_calls) == 1
    assert result.checklist.get("what-1").status is ChecklistStatus.SETTLED
    assert result.checklist.get("how-1").status is ChecklistStatus.UNEXPLORED
    first_round = json.loads(result.ledger.rounds[0].result_summary)
    assert first_round["status_updates"][0]["item_id"] == "what-1"
    assert first_round["result_count"] == 1


def test_discontinuous_segment_range_is_rejected_without_clamping():
    source = (
        "First continuous passage.\n\n"
        "Intervening text.\n\n"
        "Second continuous passage."
    )
    decisions = [
        envelope(
            {
                "action": "read",
                "item_id": "what-1",
                "url": "https://example.com/composite",
            }
        ),
        envelope({"action": "stop"}),
    ]
    note_outputs = [
        envelope(
            {
                "active_notes": [
                    {
                        "item_id": "what-1",
                        "finding": "Two passages support the finding.",
                        "start_segment_id": "S000001",
                        "end_segment_id": "S000003",
                    }
                ],
                "cross_item_seeds": [],
            }
        )
    ]

    result, decision_model, note_model, _ = run_loop(
        decisions,
        notes=note_outputs,
        tavily=FakeTavily(raw_text=source),
    )

    note_prompt = note_model.prompts[0]
    assert "select exactly one continuous" in note_prompt
    assert "It may not cross a heading, paragraph" in note_prompt
    assert "return two separate\nnotes" in note_prompt
    assert all(
        term not in note_prompt.casefold()
        for term in ("receivership", "regulators", "asset_flow")
    )

    assert len(note_model.prompts) == 1
    assert result.ledger.notes == []
    audit = json.loads(result.ledger.rounds[0].result_summary)
    assert any(
        "crosses a Markdown unit boundary" in error
        for error in audit["active_note_errors"]
    )
    next_prompt = decision_model.prompts[1]
    assert '"note_count": 0' in next_prompt


def test_invalid_status_update_is_rejected_without_losing_valid_parts():
    decisions = [
        envelope(
            {
                "status_updates": [
                    {
                        "item_id": "what-1",
                        "status": "settled",
                        "reason": "The available evidence is sufficient",
                    },
                    {
                        "item_id": "how-1",
                        "status": "has_material",
                        "reason": "The model tried to mirror system state",
                    },
                ],
                "action": {
                    "action": "search",
                    "item_id": "how-1",
                    "query": "continue researching the open item",
                },
            }
        ),
        envelope({"action": "stop"}),
    ]

    result, decision_model, _, client = run_loop(decisions)

    assert result.checklist.get("what-1").status is ChecklistStatus.SETTLED
    assert result.checklist.get("how-1").status is ChecklistStatus.UNEXPLORED
    assert len(client.search_calls) == 1
    assert [record.action for record in result.ledger.rounds] == [
        "search",
        "stop",
    ]
    rejected = [
        record
        for record in result.ledger.checklist_history
        if record.event == "status_update" and not record.accepted
    ]
    assert len(rejected) == 1
    assert rejected[0].item_id == "how-1"
    assert rejected[0].from_status == "unexplored"
    assert rejected[0].to_status == "has_material"
    first_round = json.loads(result.ledger.rounds[0].result_summary)
    assert first_round["status_updates"][0]["item_id"] == "what-1"
    assert first_round["rejected_status_updates"][0]["item_id"] == "how-1"
    assert "Input should be 'settled' or 'exhausted_not_found'" in (
        first_round["rejected_status_updates"][0]["error"]
    )
    prompt = decision_model.prompts[0]
    assert "status_updates accepts terminal judgements only" in prompt
    assert 'Never put "unexplored" or "has_material"' in prompt


def test_partial_success_resets_malformed_streak_but_no_executable_part_does_not():
    partial = {
        "status_updates": [
            {
                "item_id": "how-1",
                "status": "has_material",
                "reason": "Invalid mirror of system state",
            }
        ],
        "action": {
            "action": "search",
            "item_id": "how-1",
            "query": "valid action survives",
        },
    }
    result, model, _, _ = run_loop(
        [
            envelope("{"),
            envelope(partial),
            envelope("{"),
            envelope({"action": "stop"}),
        ],
        budget=LoopBudget(
            max_rounds=10,
            max_tokens=100,
            max_cost_usd=10,
            max_consecutive_malformed_actions=2,
        ),
    )

    assert len(model.prompts) == 4
    assert result.stop_reason is StopReason.MODEL_STOP_WITH_OPEN_ITEMS
    assert [record.action for record in result.ledger.rounds] == [
        "invalid_action",
        "search",
        "invalid_action",
        "stop",
    ]

    rejected_only, _, _, _ = run_loop(
        [
            envelope(
                {
                    "status_updates": [
                        {
                            "item_id": "what-1",
                            "status": "unexplored",
                            "reason": "No executable component",
                        }
                    ],
                    "action": None,
                }
            )
        ],
        budget=LoopBudget(
            max_rounds=10,
            max_tokens=100,
            max_cost_usd=10,
            max_consecutive_malformed_actions=1,
        ),
    )
    assert rejected_only.stop_reason is StopReason.MALFORMED_ACTION_LIMIT
    assert rejected_only.ledger.rounds[0].action == "invalid_action"


def test_note_index_is_compact_and_model_pages_then_recalls_selected_notes():
    active = checklist(second=False)
    ledger = ResearchLedger(topic=active.topic)
    source = "Exact quote one. Exact quote two. Exact quote three. Exact quote four."
    url = "https://example.com/page"
    ledger.cache_source(url, source)
    for number in range(1, 5):
        quote = f"Exact quote {('one', 'two', 'three', 'four')[number - 1]}."
        ledger.add_note(
            create_note(
                item_id="what-1",
                finding=f"Finding marker {number}",
                quote=quote,
                url=url,
                source_text=source,
            )
        )

    decisions = [
        envelope(
            {
                "action": "inspect_notes",
                "item_id": "what-1",
                "cursor": None,
            }
        ),
        envelope(
            {
                "action": "inspect_notes",
                "item_id": "what-1",
                "cursor": "2",
            }
        ),
        envelope(
            {
                "action": "recall_notes",
                "item_id": "what-1",
                "note_ids": ["note-000002", "note-000004"],
            }
        ),
        envelope({"action": "stop"}),
    ]

    result, decision_model, _, _ = run_loop(
        decisions,
        active_checklist=active,
        active_ledger=ledger,
        settings=LoopSettings(note_page_size=2, max_recalled_notes=2),
    )

    initial_prompt = decision_model.prompts[0]
    assert '"note_count": 4' in initial_prompt
    assert '"can_inspect": true' in initial_prompt
    assert "Finding marker 1" not in initial_prompt
    assert "Exact quote one." not in initial_prompt

    first_page_prompt = decision_model.prompts[1]
    assert "Finding marker 1" in first_page_prompt
    assert "Finding marker 2" in first_page_prompt
    assert "Finding marker 3" not in first_page_prompt
    assert "Exact quote one." not in first_page_prompt

    second_page_prompt = decision_model.prompts[2]
    assert "Finding marker 1" not in second_page_prompt
    assert "Finding marker 3" in second_page_prompt
    assert "Finding marker 4" in second_page_prompt

    recalled_prompt = decision_model.prompts[3]
    assert '"note_id": "note-000002"' in recalled_prompt
    assert '"note_id": "note-000004"' in recalled_prompt
    assert "Exact quote two." in recalled_prompt
    assert "Exact quote four." in recalled_prompt
    assert "Exact quote one." not in recalled_prompt

    assert [note.note_id for note in ledger.notes] == [
        "note-000001",
        "note-000002",
        "note-000003",
        "note-000004",
    ]
    assert len({note.source_id for note in ledger.notes}) == 1
    assert result.consecutive_collection_failures["what-1"] == 0
    first_audit = json.loads(ledger.rounds[0].result_summary)
    second_audit = json.loads(ledger.rounds[1].result_summary)
    recall_audit = json.loads(ledger.rounds[2].result_summary)
    assert first_audit["next_cursor"] == "2"
    assert second_audit["next_cursor"] is None
    assert recall_audit["recalled_note_ids"] == [
        "note-000002",
        "note-000004",
    ]
