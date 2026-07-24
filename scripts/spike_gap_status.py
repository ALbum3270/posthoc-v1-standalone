"""Spike: does graph-derived coverage actually differ from V1's filled_ids?

M0's whole justification is that reading coverage out of the graph answers a
different question than the in-memory set it replaces (SESSION_HANDOFF §5). If
both always agree, M0 is bookkeeping and the core assumption remains untested.

The discriminating case is set up directly: two slots get an episode written for
them, but only one produces any fact edges.

    V1's filled_ids:  both slots filled  (an episode was written for each)
    graph coverage:   one slot filled    (only one has facts)

Also exercises the Cypher against the real backend -- aggregation and list
functions that pass a fake driver can still fail on Neo4j.

Costs a handful of embedding calls; no chat model. Cleans up after itself.

    python scripts/spike_gap_status.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE / "src"))
load_dotenv(_BASE / ".env")

from open_deep_research.graphrag.graph.queries import (  # noqa: E402
    coverage_ratio_from_gaps,
    get_gap_status,
    get_open_slots_from_graph,
)
from open_deep_research.graphrag.graph.verified_episode import (  # noqa: E402
    add_verified_episode,
)
from open_deep_research.graphrag.ontology import OntologySlot  # noqa: E402
from open_deep_research.graphrag.schemas import (  # noqa: E402
    EntityRef,
    ExtractedTriple,
    SourceDocument,
    VerifiedEpisodeInput,
)

SCHEMA: dict[str, tuple[OntologySlot, ...]] = {
    "WHO": (
        OntologySlot(
            slot_id="who.primary_actor",
            dimension="WHO",
            label="Primary Actor",
            question="Who is the main actor?",
            priority=100,
        ),
        OntologySlot(
            slot_id="who.affected_parties",
            dimension="WHO",
            label="Affected Parties",
            question="Who is affected?",
            priority=90,
        ),
    ),
    "WHY": (
        OntologySlot(
            slot_id="why.motive",
            dimension="WHY",
            label="Motive",
            question="Why did it happen?",
            priority=80,
        ),
    ),
}


def source(doc_id: str) -> SourceDocument:
    return SourceDocument(
        document_id=doc_id,
        title="FTX coverage spike",
        url=f"https://example.com/{doc_id}",
        content="(not re-extracted)",
        published_at=datetime(2022, 11, 20, tzinfo=timezone.utc),
    )


def payload(research_id: str, slot_id: str, triples: list[ExtractedTriple]):
    return VerifiedEpisodeInput(
        research_id=research_id,
        slot_id=slot_id,
        source=source(f"doc-{slot_id}"),
        triples=triples,
    )


def triple(subject: str, predicate: str, obj: str, slot_id: str) -> ExtractedTriple:
    return ExtractedTriple(
        slot_id=slot_id,
        subject=EntityRef(name=subject),
        predicate=predicate,
        object=obj,
        confidence=0.85,
        source_document_id=f"doc-{slot_id}",
    )


async def main() -> int:
    from graphiti_core import Graphiti

    research_id = f"spike-{uuid4().hex[:8]}"
    graphiti = Graphiti(
        uri=os.environ["NEO4J_URI"],
        user=os.environ.get("NEO4J_USER", "neo4j"),
        password=os.environ["NEO4J_PASSWORD"],
    )
    driver = graphiti.driver

    # Slot A: an episode that yields real facts.
    filled = await add_verified_episode(
        graphiti,
        payload(
            research_id,
            "who.primary_actor",
            [
                triple("Sam Bankman-Fried", "founded", "FTX", "who.primary_actor"),
                triple("Sam Bankman-Fried", "resigned as CEO of", "FTX", "who.primary_actor"),
            ],
        ),
    )
    # Slot B: an episode is written, but extraction produced nothing usable.
    # V1 would have marked this slot filled regardless.
    barren = await add_verified_episode(
        graphiti, payload(research_id, "who.affected_parties", [])
    )

    print(f"slot A episode={filled.episode_uuid[:8]} edges={len(filled.edge_uuids)}")
    print(f"slot B episode={barren.episode_uuid[:8]} edges={len(barren.edge_uuids)}  <- barren\n")

    statuses = await get_gap_status(graphiti, research_id=research_id, schema=SCHEMA)
    ratio = coverage_ratio_from_gaps(statuses)
    open_slots = await get_open_slots_from_graph(
        graphiti, research_id=research_id, schema=SCHEMA
    )

    for status in statuses:
        mark = "FILLED" if status.filled else "open  "
        print(f"  {mark}  {status.slot_id:24} conf={status.confidence:.2f}  {status.notes}")
    print(f"\ncoverage = {ratio:.0%}   open slots (priority order) = "
          f"{[s.slot_id for s in open_slots]}\n")

    by_id = {s.slot_id: s for s in statuses}
    v1_filled_ids = {"who.primary_actor", "who.affected_parties"}  # what V1 would say

    checks = [
        ("slot with facts is filled", by_id["who.primary_actor"].filled),
        ("BARREN slot stays open (V1 would call it filled)",
         not by_id["who.affected_parties"].filled),
        ("untouched slot stays open", not by_id["why.motive"].filled),
        ("coverage is 1/3, not V1's 2/3", abs(ratio - 1 / 3) < 1e-9),
        ("graph disagrees with filled_ids", {s.slot_id for s in statuses if s.filled}
         != v1_filled_ids),
        ("confidence came from edge attributes",
         abs(by_id["who.primary_actor"].confidence - 0.85) < 1e-6),
        ("provenance episode is attributed",
         filled.episode_uuid in by_id["who.primary_actor"].supporting_episode_ids),
        ("open slots ordered by priority",
         [s.slot_id for s in open_slots] == ["who.affected_parties", "why.motive"]),
    ]

    print("--- assertions ---")
    ok = True
    for label, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
        ok = ok and passed

    # ---- cleanup ---------------------------------------------------------
    await driver.execute_query(
        "MATCH ()-[e:RELATES_TO {research_id: $rid}]->() DELETE e", rid=research_id
    )
    for episode_uuid in (filled.episode_uuid, barren.episode_uuid):
        await driver.execute_query(
            "MATCH (n:Episodic {uuid: $u}) DETACH DELETE n", u=episode_uuid
        )
    await driver.execute_query(
        "MATCH (n:Entity) WHERE n.uuid IN $uuids AND NOT (n)--() DELETE n",
        uuids=filled.node_uuids + barren.node_uuids,
    )
    print("\ncleaned up")

    await graphiti.close()
    print("SPIKE PASSED" if ok else "SPIKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
