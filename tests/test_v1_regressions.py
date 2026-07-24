import asyncio
from types import SimpleNamespace

from v1.graph_writer import write_slot_episode
from v1.reporter import EvidenceRef, build_report
from v1.searcher import search, select_relevant_text
from v1.supervisor import make_query_novel, select_next_slot


def test_relevant_text_selection_skips_page_chrome_and_keeps_answer():
    raw_text = (
        "Jump to content\nNavigation menu\n"
        + "Wikipedia icon.svg Search Main Page\n" * 200
        + "\n\nFTX disclosed financial liabilities of $8 billion "
        "and more than one million creditors."
    )

    selected = select_relevant_text(
        raw_text,
        query="FTX collapse scale losses",
        focus="What is the financial amount, quantity, or material impact?",
        max_chars=500,
    )

    assert "$8 billion" in selected
    assert len(selected) <= 500
    assert not selected.startswith("Jump to content")


def test_relevant_text_selection_keeps_tavily_focused_snippet_first():
    selected = select_relevant_text(
        "Navigation\n" * 500,
        snippet="The regulator filed its complaint on March 1, 2026.",
        query="complaint filing date",
        focus="When did official intervention begin?",
        max_chars=180,
    )

    assert selected.startswith("The regulator filed its complaint")
    assert len(selected) <= 180


def test_search_filters_seen_urls_and_requests_plain_text(monkeypatch):
    calls = []

    class FakeTavilyClient:
        def __init__(self, api_key):
            assert api_key == "test-key"

        async def search(self, query, **kwargs):
            calls.append((query, kwargs))
            return {
                "results": [
                    {
                        "url": "https://example.com/seen/",
                        "title": "Seen",
                        "content": "Old result",
                        "raw_content": "Old result",
                    },
                    {
                        "url": "https://example.com/new#section",
                        "title": "New",
                        "content": "FTX owed billions of dollars.",
                        "raw_content": "Navigation\n\nFTX liabilities were $8 billion.",
                    },
                ]
            }

    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr("tavily.AsyncTavilyClient", FakeTavilyClient)

    results = asyncio.run(
        search(
            "FTX liabilities",
            focus="What was the financial amount?",
            exclude_urls={"https://example.com/seen#top"},
        )
    )

    assert [result.title for result in results] == ["New"]
    assert "$8 billion" in results[0].raw_text
    assert calls[0][1]["include_raw_content"] == "text"


def test_slot_selection_backs_off_after_failure():
    first = select_next_slot(set(), {})
    after_failure = select_next_slot(set(), {first["slot_id"]: 1})

    assert first["slot_id"] == "who.primary_actor"
    assert after_failure["slot_id"] == "what.core_event"


def test_repeated_query_gets_deterministic_alternative():
    slot = {
        "slot_id": "what.scale",
        "dimension": "WHAT",
        "label": "规模",
        "question": "What was the financial impact?",
    }

    query = make_query_novel(
        "FTX losses amount",
        topic="FTX collapse",
        slot=slot,
        previous_queries=["FTX losses amount"],
    )

    assert query != "FTX losses amount"
    assert query.startswith("FTX losses amount")


class FakeGraphitiWriter:
    def __init__(self):
        self.kwargs = None

    async def add_episode(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(episode=SimpleNamespace(uuid="episode-123"))


def test_graph_writer_persists_research_and_slot_identity():
    graphiti = FakeGraphitiWriter()

    episode_uuid = asyncio.run(
        write_slot_episode(
            graphiti,
            group_id="neo4j",
            research_id="research-abc",
            slot_id="what.scale",
            slot_label="规模",
            triples=[
                {
                    "subject": "FTX",
                    "predicate": "owed",
                    "object": "$8 billion",
                }
            ],
            source_url="https://example.com/source",
            source_title="Source",
            previous_episode_uuids=["episode-earlier"],
        )
    )

    assert episode_uuid == "episode-123"
    assert graphiti.kwargs["name"].startswith("research-abc::what.scale::")
    assert "research_id=research-abc" in graphiti.kwargs["source_description"]
    assert "slot_id=what.scale" in graphiti.kwargs["source_description"]
    assert graphiti.kwargs["previous_episode_uuids"] == ["episode-earlier"]


class FakeGraphitiReporter:
    def __init__(self):
        self.calls = []
        self.facts_by_episode = {
            "episode-who": ["Sam Bankman-Fried was the primary actor."],
            "episode-what": ["FTX filed for bankruptcy."],
        }

    async def get_nodes_and_edges_by_episode(self, episode_uuids):
        self.calls.append(episode_uuids)
        edges = [
            SimpleNamespace(fact=fact)
            for episode_uuid in episode_uuids
            for fact in self.facts_by_episode[episode_uuid]
        ]
        return SimpleNamespace(edges=edges, nodes=[])


def test_report_uses_exact_episode_ids_without_cross_slot_search():
    graphiti = FakeGraphitiReporter()
    report = asyncio.run(
        build_report(
            graphiti,
            topic="FTX collapse",
            evidence_by_slot={
                "who.primary_actor": [
                    EvidenceRef(
                        episode_uuid="episode-who",
                        title="Actor source",
                        url="https://example.com/who",
                    )
                ],
                "what.core_event": [
                    EvidenceRef(
                        episode_uuid="episode-what",
                        title="Event source",
                        url="https://example.com/what",
                    )
                ],
            },
        )
    )

    who_section = report.split("## WHO", 1)[1].split("## WHAT", 1)[0]
    what_section = report.split("## WHAT", 1)[1].split("## WHEN", 1)[0]

    assert "Sam Bankman-Fried was the primary actor." in who_section
    assert "FTX filed for bankruptcy." not in who_section
    assert "FTX filed for bankruptcy." in what_section
    assert "Sam Bankman-Fried was the primary actor." not in what_section
    assert graphiti.calls == [["episode-who"], ["episode-what"]]
    assert "2/16" in report
