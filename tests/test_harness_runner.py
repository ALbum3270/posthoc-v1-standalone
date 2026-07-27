import asyncio
import json
import os

import pytest

import run_harness as harness_cli
from open_deep_research.harness.loop import LoopBudget, StopReason
from open_deep_research.harness.runner import run_harness


class ChecklistModel:
    last_usage = {"token_count": 3, "cost_usd": 0.03}

    def __init__(self, events):
        self.events = events

    async def generate(self, prompt):
        self.events.append("checklist")
        return {
            "items": [
                {
                    "item_id": "what-1",
                    "dimension": "what",
                    "question": "What happened?",
                    "priority": 1,
                    "required_source_count": 1,
                }
            ]
        }


class DecisionModel:
    def __init__(self, events):
        self.events = events

    async def generate(self, prompt):
        self.events.append("decision")
        return {
            "content": {"action": "settle", "item_id": "what-1"},
            "token_count": 2,
            "cost_usd": 0.02,
        }


class UnusedNoteModel:
    async def generate(self, prompt):
        raise AssertionError("note model should not be called")


class WriteModel:
    def __init__(self, events):
        self.events = events
        self.prompts = []

    async def generate(self, prompt):
        self.events.append("write")
        self.prompts.append(prompt)
        return {
            "content": "# Report\n\nThe model wrote this report.",
            "token_count": 5,
            "cost_usd": 0.05,
        }


class UnusedTavily:
    async def search(self, query, **kwargs):
        raise AssertionError("search should not be called")

    async def extract(self, urls, **kwargs):
        raise AssertionError("read should not be called")


def test_runner_executes_pipeline_and_writes_report_and_complete_audit(tmp_path):
    events = []
    writer = WriteModel(events)

    result = asyncio.run(
        run_harness(
            "A topic",
            checklist_model=ChecklistModel(events),
            decision_model=DecisionModel(events),
            note_model=UnusedNoteModel(),
            write_model=writer,
            tavily_client=UnusedTavily(),
            budget=LoopBudget(max_rounds=2, max_tokens=100, max_cost_usd=1),
            output_dir=tmp_path,
            run_id="fixed-run",
            model_names={
                "decision": "cheap-decision",
                "note": "cheap-note",
                "verification": "strong-verifier",
            },
        )
    )

    assert events == ["checklist", "decision", "write"]
    assert result.loop_result.stop_reason is StopReason.ALL_ITEMS_TERMINAL
    assert result.report_path == tmp_path / "fixed-run.md"
    assert result.audit_path == tmp_path / "fixed-run.json"
    assert result.report_path.read_text(encoding="utf-8") == (
        "# Report\n\nThe model wrote this report."
    )
    assert "- Status: settled" in writer.prompts[0]

    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["ledger"]["research_id"] == "fixed-run"
    assert audit["ledger"]["rounds"][0]["action"] == "settle"
    assert audit["checklist"]["items"][0]["status"] == "settled"
    assert audit["stop"] == {
        "detail": "all checklist items reached a terminal state",
        "is_success": True,
        "open_item_ids": [],
        "reason": "all_items_terminal",
    }
    assert audit["usage"] == {
        "checklist": {"cost_usd": 0.03, "token_count": 3},
        "collection": {"cost_usd": 0.02, "token_count": 2},
        "total": {"cost_usd": 0.1, "token_count": 10},
        "writing": {"cost_usd": 0.05, "token_count": 5},
    }
    assert audit["models"]["verification"] == "strong-verifier"
    assert audit["artifacts"] == {
        "audit": "fixed-run.json",
        "report": "fixed-run.md",
    }


def test_runner_rejects_run_id_that_could_escape_output_directory(tmp_path):
    with pytest.raises(ValueError, match="run_id"):
        asyncio.run(
            run_harness(
                "A topic",
                checklist_model=ChecklistModel([]),
                decision_model=DecisionModel([]),
                note_model=UnusedNoteModel(),
                write_model=WriteModel([]),
                tavily_client=UnusedTavily(),
                output_dir=tmp_path,
                run_id="../outside",
            )
        )


def test_cli_configures_openrouter_proxy_without_touching_no_proxy(monkeypatch):
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.setenv("no_proxy", "leave-this-alone")

    harness_cli.configure_openrouter_proxy("https://openrouter.ai/api/v1")

    assert os.environ["https_proxy"] == "http://127.0.0.1:7890"
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7890"
    assert os.environ["no_proxy"] == "leave-this-alone"
