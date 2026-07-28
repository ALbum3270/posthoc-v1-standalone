import asyncio
import json

from scripts.replay_harness_notes import load_replay_cases, replay_cases


class ScriptedReplayModel:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return self.response


def test_replay_reconstructs_historical_targets_and_uses_production_grounding(
    tmp_path,
):
    source = "An exact active sentence. A separate exact seed sentence."
    url = "https://example.com/source"
    audit = {
        "run_id": "run-123",
        "checklist": {
            "topic": "A neutral topic",
            "items": [
                {
                    "item_id": "active",
                    "dimension": "what",
                    "question": "What happened?",
                    "priority": 1,
                    "corroboration_target": 1,
                    "status": "settled",
                },
                {
                    "item_id": "eligible",
                    "dimension": "how",
                    "question": "How did it happen?",
                    "priority": 2,
                    "corroboration_target": 1,
                    "status": "unexplored",
                },
                {
                    "item_id": "terminal",
                    "dimension": "when",
                    "question": "When did it happen?",
                    "priority": 3,
                    "corroboration_target": 1,
                    "status": "settled",
                },
            ],
        },
        "ledger": {
            "checklist_history": [],
            "source_cache": {url: source},
            "rounds": [
                {
                    "round_number": 1,
                    "action": "read",
                    "item_id": "active",
                    "url": url,
                    "result_summary": json.dumps(
                        {
                            "note_model_called": True,
                            "cross_item_seed_capacity": 3,
                            "status_updates": [
                                {
                                    "item_id": "terminal",
                                    "status": "settled",
                                    "reason": "done",
                                }
                            ],
                        }
                    ),
                    "token_count": 1,
                    "cost_usd": 0.01,
                }
            ],
        },
    }
    audit_path = tmp_path / "run.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    payload, cases = load_replay_cases(audit_path)

    assert payload["run_id"] == "run-123"
    assert len(cases) == 1
    assert [
        item.item_id
        for item in cases[0].checklist.items
        if not item.is_complete and item.item_id != "active"
    ] == ["eligible"]
    model = ScriptedReplayModel(
        {
            "content": {
                "active_notes": [
                    {
                        "item_id": "active",
                        "finding": "Active finding.",
                        "start_segment_id": "S000001",
                        "end_segment_id": "S000001",
                    }
                ],
                "cross_item_seeds": [
                    {
                        "item_id": "eligible",
                        "finding": "Cross finding.",
                        "start_segment_id": "S000002",
                        "end_segment_id": "S000002",
                    }
                ],
            },
            "token_count": 42,
            "cost_usd": 0.0123,
        }
    )

    result = asyncio.run(
        replay_cases(
            cases,
            model_client=model,
            model_name="test-model",
            source_run_id="run-123",
        )
    )

    assert result["fixed_invariants"]["tavily_called"] is False
    assert result["usage"] == {"token_count": 42, "cost_usd": 0.0123}
    assert result["active"]["location_counts"] == {
        "strict": 1,
        "repaired": 0,
        "unlocatable": 0,
    }
    assert result["cross"]["location_counts"] == {
        "strict": 1,
        "repaired": 0,
        "unlocatable": 0,
    }
    assert result["cross"]["noncontiguous_composite_count"] == 0
    assert result["combined"]["quote_length_chars"]["count"] == 2
    assert result["span_capacity"] == {
        "limits": {"max_segments": 12, "max_chars": 2000},
        "rejected_note_count": 0,
        "failure_reason_counts": {},
        "rejections": [],
    }
    assert result["fixed_invariants"]["oversized_ranges_truncated"] is False
    assert result["fixed_invariants"]["oversized_ranges_retried"] is False
    assert result["per_source"][0]["eligible_cross_item_ids"] == [
        "eligible"
    ]
    assert "terminal" not in model.prompts[0].split(
        "Eligible cross-item targets:\n", 1
    )[1].split("\n\nSource URL:", 1)[0]
