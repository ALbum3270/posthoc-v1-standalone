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
    run_research_loop,
)


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


def envelope(content, *, tokens=1, cost=0.01):
    return {
        "content": content,
        "token_count": tokens,
        "cost_usd": cost,
    }


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


def test_read_fetches_once_extracts_cross_item_notes_and_hides_source_from_decider():
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
                "notes": [
                    {
                        "item_id": "how-1",
                        "finding": "The source also answers another item.",
                        "quote": "A useful exact sentence.",
                    }
                ]
            },
            tokens=3,
            cost=0.2,
        ),
        envelope(
            {
                "notes": [
                    {
                        "item_id": "what-1",
                        "finding": "The active item now has material.",
                        "quote": "A second exact sentence.",
                    }
                ]
            },
            tokens=4,
            cost=0.3,
        ),
    ]

    result, decision_model, note_model, client = run_loop(
        decisions,
        notes=note_outputs,
        tavily=tavily,
    )

    assert len(client.extract_calls) == 1
    assert len(note_model.prompts) == 2
    assert all(source in prompt for prompt in note_model.prompts)
    assert all("private-source-marker" not in prompt for prompt in decision_model.prompts)
    assert '"consecutive_failures": 1' in decision_model.prompts[1]
    assert [note.item_id for note in result.ledger.notes] == ["how-1", "what-1"]
    assert result.consecutive_failures["what-1"] == 0
    assert result.consecutive_failures["how-1"] == 0
    assert result.checklist.get("how-1").status is ChecklistStatus.HAS_MATERIAL
    assert result.checklist.get("what-1").status is ChecklistStatus.HAS_MATERIAL
    assert json.loads(result.ledger.rounds[0].result_summary)["cache_hit"] is False
    assert json.loads(result.ledger.rounds[1].result_summary)["cache_hit"] is True
    assert result.ledger.rounds[0].token_count == 5
    assert result.ledger.rounds[0].cost_usd == pytest.approx(0.3)
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
        notes=[envelope({"notes": []}), envelope({"notes": []})],
        tavily=tavily,
        settings=LoopSettings(decision_source_char_limit=8),
    )

    # Nothing is injected before the first recall lands.
    assert "BBBBBB" not in decision_model.prompts[2]

    final_prompt = decision_model.prompts[-1]
    assert '"content": "BBBBBB"' in final_prompt
    assert '"content": "AA"' in final_prompt
    assert "AAAAAA" not in final_prompt
    assert final_prompt.index(recent_url) < final_prompt.index(older_url)

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
    assert result.consecutive_failures["what-1"] == 1
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
    note_outputs = [envelope({"notes": []}, tokens=5, cost=0.4)]

    result, _, _, _ = run_loop(decisions, notes=note_outputs)

    assert result.stop_reason is StopReason.MODEL_STOP_WITH_OPEN_ITEMS
    assert result.ledger.notes == []
    assert result.consecutive_failures["what-1"] == 1
    audit = json.loads(result.ledger.rounds[0].result_summary)
    assert audit["notes_created"] == 0
    assert audit["active_item_notes"] == 0
    assert audit["note_output_error"] is None
    assert result.ledger.rounds[0].token_count == 6


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
    assert result.consecutive_failures["what-1"] == 1
    assert '"consecutive_failures": 1' in decision_model.prompts[1]

    # Searching yields candidates, not notes, so the item is still a failure --
    # but the hits must reach the next decision or the model can only search
    # again, which is the zero-entropy loop this project already measured.
    next_prompt = decision_model.prompts[1]
    assert "Candidate snippet" in next_prompt
    assert '"url": "https://example.com/source"' in next_prompt
    assert '"read": false' in next_prompt
