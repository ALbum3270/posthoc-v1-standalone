from types import SimpleNamespace

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
            nodes=[
                SimpleNamespace(uuid="node-a", episodes=["episode-a"]),
                SimpleNamespace(uuid="node-b", episodes=["episode-b"]),
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
