"""Tests for the verified (no re-extraction) Graphiti write path.

These run against real ``graphiti_core`` node/edge objects and a recording fake
driver, so they exercise the actual save payloads without a database or any API
spend. What they pin down is the set of properties the default
``add_episode`` route violated in the 2026-07-24 baseline run.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from graphiti_core.driver.driver import GraphProvider

from open_deep_research.graphrag.graph.verified_episode import (
    RESERVED_EDGE_ATTRIBUTES,
    ReservedAttributeError,
    add_verified_episode,
    assert_no_reserved_attributes,
    build_edge_attributes,
    normalize_entity_name,
    render_fact,
    resolve_reference_time,
)
from open_deep_research.graphrag.schemas import (
    EntityRef,
    ExtractedTriple,
    SourceDocument,
    SourceSpan,
    VerifiedEpisodeInput,
)

EMBED_DIM = 8


class FakeDriver:
    """Records every save so tests can assert on payloads and ordering."""

    provider = GraphProvider.NEO4J
    graph_operations_interface = None

    def __init__(self, existing_entities: dict[str, str] | None = None) -> None:
        self.calls: list[dict] = []
        self.existing = existing_entities or {}

    async def execute_query(self, query, **kwargs):
        self.calls.append({"query": str(query), **kwargs})
        if "normalized" in kwargs:
            uuid = self.existing.get(kwargs["normalized"])
            return ([{"uuid": uuid}] if uuid else []), None, None
        return [], None, None

    # -- helpers ---------------------------------------------------------
    # On Neo4j, EpisodicNode.save spreads its args as kwargs while EntityNode and
    # EntityEdge nest theirs under entity_data / edge_data (nodes.py:564,
    # edges.py:359). Each save type is identified by which of those it uses.
    @property
    def episode_saves(self) -> list[dict]:
        return [c for c in self.calls if "entity_edges" in c]

    @property
    def entity_saves(self) -> list[dict]:
        return [c["entity_data"] for c in self.calls if "entity_data" in c]

    @property
    def edge_saves(self) -> list[dict]:
        return [c["edge_data"] for c in self.calls if "edge_data" in c]


class FakeEmbedder:
    async def create_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(i)] * EMBED_DIM for i, _ in enumerate(texts)]


class FakeGraphiti:
    def __init__(self, driver: FakeDriver) -> None:
        self.driver = driver
        self.embedder = FakeEmbedder()


def make_triple(
    subject: str,
    predicate: str,
    obj: str,
    *,
    slot_id: str = "what.core_event",
    quote: str | None = None,
    confidence: float = 0.8,
) -> ExtractedTriple:
    return ExtractedTriple(
        slot_id=slot_id,
        subject=EntityRef(name=subject),
        predicate=predicate,
        object=obj,
        confidence=confidence,
        source_document_id="doc-1",
        source_span=(
            SourceSpan(start_char=0, end_char=len(quote), quote=quote) if quote else None
        ),
    )


def make_payload(
    triples: list[ExtractedTriple],
    *,
    published_at: datetime | None = None,
) -> VerifiedEpisodeInput:
    return VerifiedEpisodeInput(
        research_id="r-123",
        slot_id="what.core_event",
        source=SourceDocument(
            document_id="doc-1",
            title="FTX collapse explained",
            url="https://example.com/ftx",
            content="...",
            published_at=published_at,
        ),
        triples=triples,
    )


# --------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------


def test_normalize_entity_name_is_conservative() -> None:
    assert normalize_entity_name("  FTX  ") == "ftx"
    assert normalize_entity_name("FTX, Inc.") == "ftx inc"
    # Must NOT merge an alias with its full form -- that is a semantic call.
    assert normalize_entity_name("SBF") != normalize_entity_name("Sam Bankman-Fried")


def test_render_fact_prefers_the_verbatim_quote() -> None:
    triple = make_triple("FTX", "filed for", "Chapter 11", quote="FTX filed for Chapter 11.")
    assert render_fact(triple) == "FTX filed for Chapter 11."


def test_render_fact_falls_back_to_the_triple() -> None:
    assert render_fact(make_triple("FTX", "filed for", "Chapter 11")) == "FTX filed for Chapter 11"


def test_reference_time_prefers_publication_date() -> None:
    """§3.12 fix 1: now() for a 2022 article is what produced the 2026 dates."""

    published = datetime(2022, 11, 20, tzinfo=timezone.utc)
    assert resolve_reference_time(make_payload([], published_at=published)) == published


def test_edge_attributes_carry_the_queryable_contract() -> None:
    triple = make_triple("FTX", "filed for", "Chapter 11")
    attributes = build_edge_attributes(
        make_payload([]), triple, "FTX filed for Chapter 11 on 11 November 2022"
    )

    assert attributes["research_id"] == "r-123"
    assert attributes["slot_id"] == "what.core_event"
    assert attributes["source_url"] == "https://example.com/ftx"
    assert attributes["stated_years"] == [2022]
    assert not RESERVED_EDGE_ATTRIBUTES.intersection(attributes)


@pytest.mark.parametrize(
    "reserved", ["fact", "valid_at", "group_id", "episodes", "uuid", "source_uuid", "target_uuid"]
)
def test_reserved_attributes_are_refused(reserved: str) -> None:
    """`edges.py:359` merges attributes into the core field namespace (§2.5).

    A collision overwrites graph state silently rather than erroring, so the
    denylist has to be enforced on the way in.
    """

    with pytest.raises(ReservedAttributeError, match=reserved):
        assert_no_reserved_attributes({"slot_id": "ok", reserved: "clobbered"})


def test_denylist_matches_the_persisted_edge_keys() -> None:
    """Pins the denylist to the names `EntityEdge.save` actually writes.

    These differ from the Python attribute names -- the endpoints serialize as
    source_uuid / target_uuid -- so denylisting the model's field names would
    leave the real collision open.
    """

    persisted = {
        "source_uuid",
        "target_uuid",
        "uuid",
        "name",
        "group_id",
        "fact",
        "fact_embedding",
        "episodes",
        "created_at",
        "expired_at",
        "valid_at",
        "invalid_at",
    }
    assert persisted <= RESERVED_EDGE_ATTRIBUTES


def test_clean_attributes_pass_through_unchanged() -> None:
    attributes = {"slot_id": "who.actors", "extraction_confidence": 0.9}
    assert assert_no_reserved_attributes(attributes) is attributes


# --------------------------------------------------------------------------
# the write path
# --------------------------------------------------------------------------


def test_fact_text_is_written_verbatim() -> None:
    """M1's core acceptance: claim text is byte-identical before and after."""

    text = "FTX filed for Chapter 11 bankruptcy on 11 November 2022."
    driver = FakeDriver()
    payload = make_payload([make_triple("FTX", "filed for", "Chapter 11", quote=text)])

    result = asyncio.run(add_verified_episode(FakeGraphiti(driver), payload))

    assert len(driver.edge_saves) == 1
    assert driver.edge_saves[0]["fact"] == text
    assert result.edge_uuids == [driver.edge_saves[0]["uuid"]]


def test_underspecified_date_produces_no_valid_at() -> None:
    """The exact 2026-07-24 failure: no year in the source, so no date on the edge."""

    driver = FakeDriver()
    payload = make_payload(
        [make_triple("FTX", "announced bankruptcy", "in mid-November")],
        published_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
    )

    asyncio.run(add_verified_episode(FakeGraphiti(driver), payload))

    assert driver.edge_saves[0]["valid_at"] is None


def test_explicit_date_is_preserved() -> None:
    driver = FakeDriver()
    payload = make_payload(
        [make_triple("FTX", "filed for bankruptcy on", "11 November 2022")]
    )

    asyncio.run(add_verified_episode(FakeGraphiti(driver), payload))

    assert driver.edge_saves[0]["valid_at"] == datetime(2022, 11, 11, tzinfo=timezone.utc)


def test_written_years_are_a_subset_of_stated_years() -> None:
    """The §3.12 acceptance assertion, enforced across a mixed batch."""

    driver = FakeDriver()
    payload = make_payload(
        [
            make_triple("FTX", "announced bankruptcy", "in mid-November"),
            make_triple("FTX", "filed for bankruptcy on", "11 November 2022"),
            make_triple("SBF", "was convicted in", "November 2023"),
        ],
        published_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
    )

    asyncio.run(add_verified_episode(FakeGraphiti(driver), payload))

    for saved in driver.edge_saves:
        if saved["valid_at"] is None:
            continue
        assert str(saved["valid_at"].year) in saved["fact"]
    assert 2026 not in {s["valid_at"].year for s in driver.edge_saves if s["valid_at"]}


def test_slot_and_research_id_land_on_edge_attributes() -> None:
    """M0 has to aggregate on these; in an episode-name string they are unusable."""

    driver = FakeDriver()
    payload = make_payload([make_triple("FTX", "filed for", "Chapter 11")])

    asyncio.run(add_verified_episode(FakeGraphiti(driver), payload))

    saved = driver.edge_saves[0]
    assert saved["research_id"] == "r-123"
    assert saved["slot_id"] == "what.core_event"
    assert saved["source_url"] == "https://example.com/ftx"


def test_entity_edges_backlink_is_written_after_the_edges() -> None:
    """§3.2: the trap that makes every provenance lookup silently return nothing."""

    driver = FakeDriver()
    payload = make_payload([make_triple("FTX", "filed for", "Chapter 11")])

    result = asyncio.run(add_verified_episode(FakeGraphiti(driver), payload))

    episode_saves = driver.episode_saves
    assert len(episode_saves) == 2, "episode must be saved again once edge uuids exist"
    assert episode_saves[0]["entity_edges"] == []
    assert episode_saves[1]["entity_edges"] == result.edge_uuids

    first_episode_idx = driver.calls.index(episode_saves[0])
    last_episode_idx = driver.calls.index(episode_saves[1])
    edge_idx = next(i for i, c in enumerate(driver.calls) if "edge_data" in c)
    assert first_episode_idx < edge_idx < last_episode_idx


def test_existing_entities_are_reused_not_duplicated() -> None:
    driver = FakeDriver(existing_entities={"ftx": "existing-ftx-uuid"})
    payload = make_payload([make_triple("FTX", "filed for", "Chapter 11")])

    result = asyncio.run(add_verified_episode(FakeGraphiti(driver), payload))

    assert "existing-ftx-uuid" in result.node_uuids
    assert [c["name"] for c in driver.entity_saves] == ["Chapter 11"]
    assert driver.edge_saves[0]["source_uuid"] == "existing-ftx-uuid"


def test_no_llm_client_is_touched() -> None:
    """The whole point: nothing on this path may call a model.

    FakeGraphiti exposes no llm_client at all, so any attempt to resolve edges or
    entities through an LLM would raise AttributeError.
    """

    driver = FakeDriver()
    payload = make_payload([make_triple("FTX", "filed for", "Chapter 11")])

    result = asyncio.run(add_verified_episode(FakeGraphiti(driver), payload))

    assert result.edge_uuids
    assert not hasattr(FakeGraphiti(driver), "llm_client")


def test_empty_payload_still_anchors_an_episode() -> None:
    driver = FakeDriver()
    result = asyncio.run(add_verified_episode(FakeGraphiti(driver), make_payload([])))

    assert result.episode_uuid
    assert result.edge_uuids == []
    assert len(driver.episode_saves) == 1
