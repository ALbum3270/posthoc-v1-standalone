"""Tests for one research round, and for the state defaults it relies on."""

from __future__ import annotations

import asyncio

import pytest
from graphiti_core.driver.driver import GraphProvider

from open_deep_research.graphrag.control.researcher import run_research_round
from open_deep_research.graphrag.ontology import OntologySlot
from open_deep_research.graphrag.schemas import (
    EntityRef,
    ExtractedTriple,
    SourceDocument,
)
from open_deep_research.state import (
    GRAPH_STATE_DEFAULTS,
    AgentState,
    graph_state_value,
)

SLOT = OntologySlot(
    slot_id="what.core_event",
    dimension="WHAT",
    label="Core Event",
    question="What happened?",
    priority=100,
)


class FakeDriver:
    provider = GraphProvider.NEO4J
    graph_operations_interface = None

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute_query(self, query, **kwargs):
        self.calls.append({"query": str(query), **kwargs})
        return [], None, None


class FakeGraphiti:
    def __init__(self) -> None:
        self.driver = FakeDriver()
        self.embedder = self

    async def create_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts]


def document(url: str, content: str = "body text") -> SourceDocument:
    return SourceDocument(
        document_id=url, title=f"title for {url}", url=url, content=content
    )


def triple(subject: str = "FTX", predicate: str = "filed for", obj: str = "Chapter 11"):
    return ExtractedTriple(
        slot_id=SLOT.slot_id,
        subject=EntityRef(name=subject),
        predicate=predicate,
        object=obj,
        confidence=0.8,
        source_document_id="doc",
    )


def run_round(*, documents, extract_map, **kwargs):
    async def search(*, query, exclude_urls):
        return [d for d in documents if d.url not in set(exclude_urls)]

    async def extract(*, document, slot):
        return extract_map.get(document.url, [])

    return asyncio.run(
        run_research_round(
            FakeGraphiti(),
            topic="FTX",
            research_id="r-1",
            slot=SLOT,
            query="ftx bankruptcy",
            search=search,
            extract=extract,
            **kwargs,
        )
    )


# --------------------------------------------------------------------------


def test_facts_reaching_the_graph_count_as_success() -> None:
    result = run_round(
        documents=[document("https://a.example")],
        extract_map={"https://a.example": [triple()]},
    )

    assert result.succeeded is True
    assert result.facts_written == 1
    assert result.triples_extracted == 1
    assert result.episode_uuids


def test_extraction_yielding_nothing_is_a_failure_not_a_fill() -> None:
    """V1 called this a fill because an episode existed. It is not one."""

    result = run_round(documents=[document("https://a.example")], extract_map={})

    assert result.succeeded is False
    assert result.facts_written == 0
    assert result.documents_seen == ["https://a.example"]
    assert "none reached the graph" in result.note


def test_excluded_urls_are_not_revisited() -> None:
    result = run_round(
        documents=[document("https://seen.example"), document("https://fresh.example")],
        extract_map={"https://fresh.example": [triple()]},
        exclude_urls=["https://seen.example"],
    )

    assert result.documents_seen == ["https://fresh.example"]
    assert result.succeeded is True


def test_search_returning_nothing_is_reported() -> None:
    result = run_round(documents=[], extract_map={})

    assert result.succeeded is False
    assert result.note == "search returned no usable documents"


def test_empty_documents_are_skipped() -> None:
    result = run_round(
        documents=[document("https://empty.example", content="   ")], extract_map={}
    )

    assert result.note == "search returned no usable documents"


def test_stops_at_the_first_document_that_answers_the_slot() -> None:
    """Once the slot is answered, further pages only re-answer it."""

    docs = [document("https://a.example"), document("https://b.example")]
    result = run_round(
        documents=docs,
        extract_map={"https://a.example": [triple()], "https://b.example": [triple()]},
    )

    assert result.documents_seen == ["https://a.example"]
    assert result.facts_written == 1


def test_barren_document_is_still_recorded_before_moving_on() -> None:
    """Its URL has to come back, or the next round searches it again."""

    docs = [document("https://barren.example"), document("https://good.example")]
    result = run_round(
        documents=docs, extract_map={"https://good.example": [triple()]}
    )

    assert result.documents_seen == ["https://barren.example", "https://good.example"]
    assert result.succeeded is True


def test_max_documents_is_respected() -> None:
    docs = [document(f"https://{i}.example") for i in range(5)]
    result = run_round(documents=docs, extract_map={}, max_documents=2)

    assert len(result.documents_seen) == 2


# --------------------------------------------------------------------------
# state defaults (§4: the TypedDict pseudo-default that would KeyError)
# --------------------------------------------------------------------------


def test_agent_state_is_a_plain_dict_at_runtime() -> None:
    """Why `field: float = 0.0` in the class body cannot work as a default."""

    assert issubclass(AgentState, dict)


@pytest.mark.parametrize("key,expected", sorted(GRAPH_STATE_DEFAULTS.items()))
def test_missing_scalar_state_reads_its_default(key: str, expected) -> None:
    """First round: nothing has written these yet, and subscripting raises."""

    empty: dict = {}
    with pytest.raises(KeyError):
        empty[key]
    assert graph_state_value(empty, key) == expected


def test_written_state_values_win_over_defaults() -> None:
    assert graph_state_value({"coverage_ratio": 0.42}, "coverage_ratio") == 0.42


def test_explicit_none_falls_back_to_the_default() -> None:
    assert graph_state_value({"coverage_ratio": None}, "coverage_ratio") == 0.0


def test_undeclared_key_is_a_programming_error() -> None:
    with pytest.raises(KeyError, match="no declared default"):
        graph_state_value({}, "not_a_state_field")
