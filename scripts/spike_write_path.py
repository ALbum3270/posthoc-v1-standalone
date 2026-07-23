"""Spike: does a hand-rolled structured write produce a correct provenance chain?

This is the `add_verified_episode` question, isolated from every LLM dependency.
We construct the graph objects ourselves -- no extract_nodes, no extract_edges --
supply placeholder embeddings, save them, then read back and assert the chain:

    Episode --MENTIONS--> Entity        (EpisodicEdge)
    Entity  --RELATES--> Entity         (EntityEdge, with .episodes -> Episode)

If this holds, `add_verified_episode` is implementable in the application layer
using only public graphiti APIs, and forking graphiti-main is unnecessary.

No API key required.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.embedder.client import EMBEDDING_DIM
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode

GROUP = "spike"
NOW = datetime.now(timezone.utc)


def fake_vec(seed: float) -> list[float]:
    """Placeholder embedding. Values are irrelevant; only the dimension matters."""
    return [seed] * EMBEDDING_DIM


async def main() -> int:
    driver = KuzuDriver(db=":memory:")
    print(f"kuzu driver up, provider={driver.provider}, EMBEDDING_DIM={EMBEDDING_DIM}\n")

    # ---- 1. The episode: provenance anchor -------------------------------
    episode = EpisodicNode(
        name="verified::what.core_event::doc-1",
        group_id=GROUP,
        labels=[],
        source=EpisodeType.text,
        content="FTX filed for Chapter 11 bankruptcy on 2022-11-11.",
        source_description="https://example.com/ftx | verified",
        created_at=NOW,
        valid_at=NOW,
    )
    await episode.save(driver)
    print(f"[1] episode saved      uuid={episode.uuid}")

    # ---- 2. The entities -------------------------------------------------
    subject = EntityNode(
        name="FTX", group_id=GROUP, labels=["Entity"], name_embedding=fake_vec(0.11)
    )
    obj = EntityNode(
        name="Chapter 11 bankruptcy",
        group_id=GROUP,
        labels=["Entity"],
        name_embedding=fake_vec(0.22),
    )
    await subject.save(driver)
    await obj.save(driver)
    print(f"[2] entities saved     {subject.name}={subject.uuid[:8]}  {obj.name}={obj.uuid[:8]}")

    # ---- 3. The fact edge, carrying the verification contract in attributes
    attributes = {
        "claim_id": f"claim-{uuid4().hex[:8]}",
        "slot_id": "what.core_event",
        "verification_status": "passed",
        "extraction_confidence": 0.91,
        "truth_score": 0.86,
        "source_document_id": "doc-1",
        "source_url": "https://example.com/ftx",
        "source_span_start": 120,
        "source_span_end": 184,
        "conflicts_with": [],
    }
    edge = EntityEdge(
        source_node_uuid=subject.uuid,
        target_node_uuid=obj.uuid,
        name="FILED_FOR",
        fact="FTX filed for Chapter 11 bankruptcy on 2022-11-11.",
        group_id=GROUP,
        created_at=NOW,
        valid_at=NOW,
        episodes=[episode.uuid],          # <-- provenance link
        fact_embedding=fake_vec(0.33),
        attributes=attributes,
    )
    await edge.save(driver)
    print(f"[3] fact edge saved    uuid={edge.uuid[:8]}  episodes={len(edge.episodes)}")

    # ---- 4. Episode -> Entity MENTIONS edges -----------------------------
    for node in (subject, obj):
        await EpisodicEdge(
            source_node_uuid=episode.uuid,
            target_node_uuid=node.uuid,
            group_id=GROUP,
            created_at=NOW,
        ).save(driver)
    print("[4] 2 MENTIONS edges saved")

    # ---- 4b. ORDERING REQUIREMENT ---------------------------------------
    # `get_nodes_and_edges_by_episode()` resolves edges by reading
    # `episode.entity_edges` (graphiti.py:1439-1441). Saving the episode before
    # the edges exist leaves that list empty, and every provenance lookup then
    # silently returns nothing. The episode must be re-saved once edge uuids are
    # known. graphiti's own add_episode does this; a hand-rolled writer must too.
    episode.entity_edges = [edge.uuid]
    await episode.save(driver)
    print("[4b] episode re-saved with entity_edges backlink")

    # ---- 5. Read everything back ----------------------------------------
    print("\n--- read-back assertions ---")
    ok = True

    got_edge = await EntityEdge.get_by_uuid(driver, edge.uuid)
    checks = [
        ("edge.fact round-trips", got_edge.fact == edge.fact),
        ("edge.episodes -> episode uuid", got_edge.episodes == [episode.uuid]),
        ("edge endpoints preserved", got_edge.source_node_uuid == subject.uuid
                                      and got_edge.target_node_uuid == obj.uuid),
        ("edge.valid_at preserved", got_edge.valid_at is not None),
        ("edge.expired_at is None (== active)", got_edge.expired_at is None),
    ]

    got_ep = await EpisodicNode.get_by_uuid(driver, episode.uuid)
    checks.append(("episode.content round-trips", got_ep.content == episode.content))
    checks.append(
        ("episode.entity_edges links the fact edge", edge.uuid in (got_ep.entity_edges or []))
    )

    got_node = await EntityNode.get_by_uuid(driver, subject.uuid)
    checks.append(("entity.name round-trips", got_node.name == "FTX"))

    for label, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
        ok = ok and passed

    # attributes: on Kuzu these are json.dumps'd (edges.py:353), so this is the
    # place where the flat/queryable contract does NOT hold. Report, don't assert.
    print(f"\n  attributes read back as {type(got_edge.attributes).__name__}: "
          f"{str(got_edge.attributes)[:120]}")

    await driver.close()
    print(f"\n{'SPIKE PASSED' if ok else 'SPIKE FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
