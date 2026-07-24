"""Tests for one research round, and for the state defaults it relies on."""

from __future__ import annotations

import asyncio

import pytest
from graphiti_core.driver.driver import GraphProvider

from open_deep_research.graphrag.control.researcher import (
    run_research_round,
    run_support_round,
)
from open_deep_research.graphrag.graph.verified_episode import (
    add_verified_episode,
    normalize_entity_name,
)
from open_deep_research.graphrag.ontology import OntologySlot
from open_deep_research.graphrag.schemas import (
    EntityRef,
    ExtractedTriple,
    SourceDocument,
    VerifiedEpisodeInput,
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
        self.existing: dict[str, str] = {}
        self.claim_edges: dict[tuple[str, ...], dict] = {}

    async def execute_query(self, query, **kwargs):
        self.calls.append({"query": str(query), **kwargs})
        if "normalized" in kwargs:
            uuid = self.existing.get(kwargs["normalized"])
            return ([{"uuid": uuid}] if uuid else []), None, None
        if "entity_data" in kwargs:
            data = kwargs["entity_data"]
            self.existing[normalize_entity_name(data["name"])] = data["uuid"]
        if "edge_data" in kwargs:
            data = kwargs["edge_data"]
            if data.get("research_id"):
                key = (
                    data["research_id"],
                    data["slot_id"],
                    data["group_id"],
                    data["source_uuid"],
                    data["target_uuid"],
                    data["name"],
                )
                self.claim_edges[key] = dict(data)
        if "relation_name" in kwargs:
            key = (
                kwargs["research_id"],
                kwargs["slot_id"],
                kwargs["group_id"],
                kwargs["source_uuid"],
                kwargs["target_uuid"],
                kwargs["relation_name"],
            )
            found = self.claim_edges.get(key)
            return ([dict(found)] if found else []), None, None
        if "edge_uuid" in kwargs and "supporting_source_urls" in kwargs:
            for edge in self.claim_edges.values():
                if edge["uuid"] == kwargs["edge_uuid"]:
                    edge.update(
                        {
                            "episodes": kwargs["episodes"],
                            "supporting_source_urls": kwargs[
                                "supporting_source_urls"
                            ],
                            "supporting_source_titles": kwargs[
                                "supporting_source_titles"
                            ],
                            "supporting_source_identities": kwargs[
                                "supporting_source_identities"
                            ],
                            "supporting_quotes": kwargs["supporting_quotes"],
                        }
                    )
                    return [{"uuid": edge["uuid"]}], None, None
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


def run_round(*, documents, extract_map, support_map=None, **kwargs):
    async def search(*, query, exclude_urls):
        return [d for d in documents if d.url not in set(exclude_urls)]

    async def extract(*, document, slot):
        return extract_map.get(document.url, [])

    async def extract_support(*, document, slot, targets):
        active = support_map if support_map is not None else extract_map
        return active.get(document.url, [])

    return asyncio.run(
        run_research_round(
            FakeGraphiti(),
            topic="FTX",
            research_id="r-1",
            slot=SLOT,
            query="ftx bankruptcy",
            search=search,
            extract=extract,
            extract_support=extract_support,
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


def test_multiple_urls_from_one_publisher_do_not_fake_independence() -> None:
    docs = [
        document("https://www.example.com/first"),
        document("https://news.example.com/second"),
        document("https://independent.example/report"),
    ]
    result = run_round(
        documents=docs,
        extract_map={
            "https://www.example.com/first": [triple()],
            "https://news.example.com/second": [triple()],
            "https://independent.example/report": [triple()],
        },
        min_sources=2,
    )

    assert "https://news.example.com/second" not in result.documents_seen
    assert result.contributing_source_identities == [
        "example.com",
        "independent.example",
    ]


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


# --------------------------------------------------------------------------
# cross-corroboration (M4 regression measured 0% on all three topics)
# --------------------------------------------------------------------------


def test_default_stops_at_one_source() -> None:
    """Cheap default: the slot is answered, further pages only re-answer it."""

    docs = [document("https://a.example"), document("https://b.example")]
    result = run_round(
        documents=docs,
        extract_map={"https://a.example": [triple()], "https://b.example": [triple()]},
    )

    assert result.contributing_sources == ["https://a.example"]
    assert result.is_corroborated is False


def test_min_sources_two_collects_a_second_source() -> None:
    docs = [document("https://a.example"), document("https://b.example")]
    result = run_round(
        documents=docs,
        extract_map={"https://a.example": [triple()], "https://b.example": [triple()]},
        min_sources=2,
    )

    assert result.contributing_sources == ["https://a.example", "https://b.example"]
    assert result.is_corroborated is True
    assert result.facts_written == 1
    assert result.supports_added == 1


def test_barren_documents_do_not_count_as_sources() -> None:
    """Only pages that actually put facts in the graph corroborate anything."""

    docs = [
        document("https://a.example"),
        document("https://barren.example"),
        document("https://c.example"),
    ]
    result = run_round(
        documents=docs,
        extract_map={"https://a.example": [triple()], "https://c.example": [triple()]},
        min_sources=2,
    )

    assert result.contributing_sources == ["https://a.example", "https://c.example"]
    assert "https://barren.example" in result.documents_seen


def test_unmet_corroboration_is_reported_but_still_a_success() -> None:
    """One source is better than none; the shortfall belongs in the note."""

    result = run_round(
        documents=[document("https://a.example")],
        extract_map={"https://a.example": [triple()]},
        min_sources=2,
    )

    assert result.succeeded is True
    assert result.is_corroborated is False
    assert "support count of 2" in result.note


def test_an_unrelated_second_source_is_not_corroboration() -> None:
    docs = [document("https://a.example"), document("https://b.example")]
    result = run_round(
        documents=docs,
        extract_map={"https://a.example": [triple()]},
        support_map={
            "https://b.example": [
                triple(subject="Binance", predicate="considered", obj="FTX")
            ]
        },
        min_sources=2,
    )

    assert result.is_corroborated is False
    assert result.contributing_sources == ["https://a.example"]
    assert result.supports_added == 0


def test_round_keeps_looking_until_each_primary_claim_is_supported() -> None:
    first_claim = triple()
    second_claim = triple(
        subject="FTX",
        predicate="announced",
        obj="its collapse",
    )
    docs = [
        document("https://a.example"),
        document("https://b.example"),
        document("https://c.example"),
    ]
    result = run_round(
        documents=docs,
        extract_map={"https://a.example": [first_claim, second_claim]},
        support_map={
            "https://b.example": [first_claim],
            "https://c.example": [second_claim],
        },
        min_sources=2,
    )

    assert result.is_corroborated is True
    assert len(result.target_edge_uuids) == 2
    assert set(result.corroborated_edge_uuids) == set(result.target_edge_uuids)
    assert result.documents_seen == [
        "https://a.example",
        "https://b.example",
        "https://c.example",
    ]


def test_targeted_support_round_can_only_update_the_existing_claim() -> None:
    async def scenario():
        graphiti = FakeGraphiti()
        target = triple(
            subject="Sam Bankman-Fried",
            predicate="controlled",
            obj="$8 billion",
        )
        first = document("https://first.example")
        initial = await add_verified_episode(
            graphiti,
            VerifiedEpisodeInput(
                research_id="r-1",
                slot_id=SLOT.slot_id,
                source=first,
                triples=[target],
            ),
        )

        async def search(*, query, exclude_urls):
            assert first.url in exclude_urls
            return [document("https://independent.example")]

        async def extract_support(*, document, slot, targets):
            assert targets == [target]
            return [
                target.model_copy(
                    update={"source_document_id": document.document_id}
                )
            ]

        result = await run_support_round(
            graphiti,
            research_id="r-1",
            slot=SLOT,
            target=target,
            target_edge_uuid=initial.created_edge_uuids[0],
            query='"Sam Bankman-Fried" "$8 billion"',
            search=search,
            extract_support=extract_support,
            exclude_urls=[first.url],
            exclude_source_identities=["first.example"],
        )
        return graphiti, initial, result

    graphiti, initial, result = asyncio.run(scenario())

    assert result.facts_written == 0
    assert result.supports_added == 1
    assert result.corroborated_edge_uuids == initial.created_edge_uuids
    lookup_names = {
        call["exact_name"]
        for call in graphiti.driver.calls
        if "exact_name" in call
    }
    assert {"sam bankman-fried", "$8 billion"} <= lookup_names
