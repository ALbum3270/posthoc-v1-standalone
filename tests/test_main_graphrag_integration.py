import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import open_deep_research.deep_researcher as main_graph
from open_deep_research.configuration import Configuration
from open_deep_research.graphrag.runtime import (
    GraphResearchResult,
    GraphResearchUsage,
)
from open_deep_research.graphrag.schemas import EvidencePack


def fake_result() -> GraphResearchResult:
    pack = EvidencePack(topic="FTX", coverage_ratio=1.0)
    return GraphResearchResult(
        topic="FTX",
        research_id="run-main",
        stop_reason="coverage_reached",
        stop_detail="done",
        coverage_ratio=1.0,
        evidence_pack=pack,
        report="# verified report",
        fact_count=4,
        source_count=2,
        usage=GraphResearchUsage(total_tokens=123, elapsed_seconds=4.5),
    )


def test_legacy_remains_the_compatibility_default() -> None:
    assert Configuration().research_mode == "legacy"
    assert main_graph.research_entrypoint(Configuration()) == "research_supervisor"


def test_graphrag_mode_routes_to_the_verified_engine() -> None:
    config = Configuration(research_mode="graphrag")

    assert main_graph.research_entrypoint(config) == "graph_research"
    assert "graph_research" in main_graph.deep_researcher_builder.nodes


def test_graph_runtime_settings_map_application_controls(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    config = Configuration(
        targeted_extraction_model="openai:gpt-4.1-mini",
        coverage_target=0.75,
        graphrag_max_rounds=12,
        graphrag_max_attempts_per_slot=2,
        graphrag_search_results=4,
        graphrag_max_documents_per_round=2,
        graphrag_min_sources_per_claim=2,
        graphrag_enable_relevance_gate=True,
        graphrag_relevance_reject_threshold=0.85,
        graphrag_enable_slot_applicability=True,
        graphrag_applicability_threshold=0.9,
    )

    settings = main_graph.graph_runtime_settings(config)

    assert settings.model == "openai/gpt-4.1-mini"
    assert settings.coverage_target == 0.75
    assert settings.max_rounds == 12
    assert settings.max_attempts_per_slot == 2
    assert settings.search_results == 4
    assert settings.max_documents_per_round == 2
    assert settings.min_sources_per_claim == 2
    assert settings.enable_relevance_gate is True
    assert settings.relevance_reject_threshold == 0.85
    assert settings.enable_slot_applicability is True
    assert settings.applicability_not_applicable_threshold == 0.9


def test_graph_node_returns_the_evidence_only_report(monkeypatch) -> None:
    captured = {}

    async def fake_run(topic, **kwargs):
        captured["topic"] = topic
        captured.update(kwargs)
        return fake_result()

    monkeypatch.setattr(main_graph, "run_live_graph_research", fake_run)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily")
    monkeypatch.setenv("NEO4J_URI", "bolt://test")
    monkeypatch.setenv("NEO4J_PASSWORD", "test-password")

    command = asyncio.run(
        main_graph.graph_research(
            {
                "research_brief": "FTX collapse",
                "messages": [HumanMessage(content="research FTX")],
            },
            {"configurable": {"research_mode": "graphrag"}},
        )
    )

    assert command.goto == "__end__"
    assert captured["topic"] == "FTX collapse"
    assert command.update["final_report"] == "# verified report"
    assert command.update["evidence_pack"].topic == "FTX"
    assert command.update["research_id"] == "run-main"
    assert command.update["research_metrics"]["total_tokens"] == 123
    assert command.update["messages"][0].content == "# verified report"


def test_non_token_supervisor_errors_are_not_silently_swallowed(monkeypatch) -> None:
    async def fail(*args, **kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(main_graph.researcher_subgraph, "ainvoke", fail)
    monkeypatch.setattr(main_graph, "is_token_limit_exceeded", lambda *args: False)
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ConductResearch",
                "args": {"research_topic": "FTX"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )

    with pytest.raises(RuntimeError, match="database unavailable"):
        asyncio.run(
            main_graph.supervisor_tools(
                {
                    "supervisor_messages": [message],
                    "research_iterations": 1,
                    "research_brief": "FTX",
                },
                {},
            )
        )
