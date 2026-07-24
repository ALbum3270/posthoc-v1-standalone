from types import SimpleNamespace

import pytest

from open_deep_research.graphrag.graph.client import GraphitiClient, GraphitiClientConfig
from open_deep_research.graphrag.schemas import GraphEpisodePayload


class FakeGraphiti:
    def __init__(self):
        self.calls = []

    async def add_episode(self, **kwargs):
        self.calls.append(("add_episode", kwargs))
        return SimpleNamespace(
            episode=SimpleNamespace(uuid="episode-1"),
            nodes=[SimpleNamespace(uuid="node-1"), SimpleNamespace(uuid="node-2")],
            edges=[SimpleNamespace(uuid="edge-1")],
        )

    async def add_episode_bulk(self, **kwargs):
        self.calls.append(("add_episode_bulk", kwargs))
        return SimpleNamespace(
            episodes=[SimpleNamespace(uuid="episode-a"), SimpleNamespace(uuid="episode-b")],
            # EntityNode has no `episodes` field -- model_fields is
            # ['attributes','created_at','group_id','labels','name',
            #  'name_embedding','summary','uuid']. The previous version of this
            # fake gave nodes one anyway, which made a wrapper bug that always
            # returned empty node_uuids look like a passing test.
            nodes=[
                SimpleNamespace(uuid="node-a"),
                SimpleNamespace(uuid="node-b"),
            ],
            # Episode -> entity attribution really lives here.
            episodic_edges=[
                SimpleNamespace(source_node_uuid="episode-a", target_node_uuid="node-a"),
                SimpleNamespace(source_node_uuid="episode-b", target_node_uuid="node-b"),
            ],
            edges=[
                SimpleNamespace(uuid="edge-a", episodes=["episode-a"]),
                SimpleNamespace(uuid="edge-b", episodes=["episode-b"]),
            ],
        )

    async def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        return ["edge-result"]

    async def search_(self, **kwargs):
        self.calls.append(("search_", kwargs))
        return SimpleNamespace(edges=["e"], nodes=["n"], episodes=["ep"])

    async def get_nodes_and_edges_by_episode(self, episode_uuids):
        self.calls.append(("get_nodes_and_edges_by_episode", episode_uuids))
        return {"episodes": episode_uuids}


class FakeEpisodeType:
    text = "text"
    json = "json"
    message = "message"


class FakeRawEpisode:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_add_episode_maps_payload_and_preserves_metadata():
    client = GraphitiClient(
        GraphitiClientConfig(group_id="tenant-a"),
        graphiti=FakeGraphiti(),
    )
    client._imports = SimpleNamespace(
        episode_type=FakeEpisodeType,
        raw_episode_cls=FakeRawEpisode,
        search_config="fake-config",
    )

    payload = GraphEpisodePayload(
        name="episode-name",
        episode_body="raw finding",
        source_description="tavily result",
        metadata={"source_type": "official", "conflicts_with": ["edge-x"]},
    )

    result = __import__("asyncio").run(client.add_episode(payload))

    assert result.episode_uuid == "episode-1"
    assert result.node_uuids == ["node-1", "node-2"]
    assert result.conflict_ids == ["edge-x"]
    call_name, kwargs = client._graphiti.calls[0]
    assert call_name == "add_episode"
    assert kwargs["group_id"] == "tenant-a"
    assert "Structured metadata:" in kwargs["episode_body"]
    assert "metadata=" in kwargs["source_description"]


def test_add_episode_bulk_maps_results_back_to_payloads():
    client = GraphitiClient(
        GraphitiClientConfig(group_id="tenant-b"),
        graphiti=FakeGraphiti(),
    )
    client._imports = SimpleNamespace(
        episode_type=FakeEpisodeType,
        raw_episode_cls=FakeRawEpisode,
        search_config="fake-config",
    )

    payloads = [
        GraphEpisodePayload(
            name="ep-a",
            episode_body="a",
            source_description="src-a",
            metadata={"conflicts_with": ["c1"]},
        ),
        GraphEpisodePayload(
            name="ep-b",
            episode_body="b",
            source_description="src-b",
        ),
    ]

    results = __import__("asyncio").run(client.add_episode_bulk(payloads))

    assert [result.episode_uuid for result in results] == ["episode-a", "episode-b"]
    assert results[0].node_uuids == ["node-a"]
    assert results[1].edge_uuids == ["edge-b"]
    assert results[0].conflict_ids == ["c1"]


def test_search_methods_use_wrapper_defaults():
    client = GraphitiClient(
        GraphitiClientConfig(group_id="tenant-c", default_search_results=7),
        graphiti=FakeGraphiti(),
    )
    client._imports = SimpleNamespace(
        episode_type=FakeEpisodeType,
        raw_episode_cls=FakeRawEpisode,
        search_config="fake-config",
    )

    basic_results = __import__("asyncio").run(client.search("query"))
    subgraph_results = __import__("asyncio").run(client.search_subgraph("query"))

    assert basic_results == ["edge-result"]
    assert subgraph_results.nodes == ["n"]
    _, kwargs = client._graphiti.calls[0]
    assert kwargs["group_ids"] == ["tenant-c"]
    assert kwargs["num_results"] == 7
    _, kwargs = client._graphiti.calls[1]
    assert kwargs["config"] == "fake-config"


def test_structured_claims_are_refused_by_the_re_extraction_path():
    """Pre-extracted claims must not be flattened back to prose for an LLM.

    That round trip is what rewrote facts and invented dates (§3.12); the
    verified path exists for these payloads, so this one refuses them loudly
    instead of silently degrading the data.
    """
    from open_deep_research.graphrag.graph.client import StructuredClaimsNotSupportedError
    from open_deep_research.graphrag.schemas import ExtractedClaim

    client = GraphitiClient(GraphitiClientConfig(), graphiti=FakeGraphiti())
    payload = GraphEpisodePayload(
        name="episode-name",
        episode_body="raw finding",
        source_description="tavily result",
        claims=[
            ExtractedClaim(
                claim_id="c-1",
                slot_id="what.core_event",
                text="FTX announced bankruptcy in mid-November",
                source_document_id="doc-1",
            )
        ],
    )

    with pytest.raises(StructuredClaimsNotSupportedError, match="add_verified_episode"):
        __import__("asyncio").run(client.add_episode(payload))

    assert client._graphiti.calls == [], "nothing may reach Graphiti"


def test_missing_reference_time_warns_instead_of_defaulting_silently(caplog):
    """now() for an undated document is the original bug; make it visible."""
    client = GraphitiClient(GraphitiClientConfig(), graphiti=FakeGraphiti())
    client._imports = SimpleNamespace(
        episode_type=FakeEpisodeType, raw_episode_cls=FakeRawEpisode, search_config="x"
    )
    payload = GraphEpisodePayload(
        name="undated", episode_body="body", source_description="src"
    )

    with caplog.at_level("WARNING"):
        __import__("asyncio").run(client.add_episode(payload))

    assert any("reference_time" in r.message for r in caplog.records)
