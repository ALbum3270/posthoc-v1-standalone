"""Spike: does add_verified_episode actually stop the date contamination?

Replays the exact triple that broke the 2026-07-24 baseline run --
"FTX | announced bankruptcy | in mid-November", no year -- through the verified
write path, against the real local Neo4j.

Through Graphiti's default route that triple came back out as two contradictory
dated edges, 2023 and 2026 (SESSION_HANDOFF §3.12). The assertion here is that
it now lands with no date at all, alongside a sibling fact that *does* state a
date and must keep it.

Costs a handful of embedding calls and nothing else: no chat model is invoked on
this path. Cleans up everything it writes.

    python scripts/spike_verified_write.py
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

from open_deep_research.graphrag.graph.verified_episode import (  # noqa: E402
    add_verified_episode,
)
from open_deep_research.graphrag.schemas import (  # noqa: E402
    EntityRef,
    ExtractedTriple,
    SourceDocument,
    VerifiedEpisodeInput,
)

GROUP_ID = "neo4j"  # must equal driver._database, see §3.4
SLOT = "what.core_event"
ARTICLE_PUBLISHED = datetime(2022, 11, 20, tzinfo=timezone.utc)


def triple(subject: str, predicate: str, obj: str) -> ExtractedTriple:
    return ExtractedTriple(
        slot_id=SLOT,
        subject=EntityRef(name=subject),
        predicate=predicate,
        object=obj,
        confidence=0.8,
        source_document_id="spike-doc",
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

    payload = VerifiedEpisodeInput(
        research_id=research_id,
        slot_id=SLOT,
        source=SourceDocument(
            document_id="spike-doc",
            title="FTX black swan",
            url="https://example.com/ftx-spike",
            content="(not re-extracted)",
            published_at=ARTICLE_PUBLISHED,
        ),
        triples=[
            # The one that broke. No year anywhere in it.
            triple("FTX", "announced bankruptcy", "in mid-November"),
            # The control: states a date, must keep it.
            triple("FTX", "filed for bankruptcy protection on", "11 November 2022"),
        ],
    )

    result = await add_verified_episode(graphiti, payload)
    print(f"wrote episode={result.episode_uuid[:8]} "
          f"nodes={len(result.node_uuids)} edges={len(result.edge_uuids)}\n")

    records, _, _ = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO {research_id: $rid}]->()
        RETURN e.fact AS fact, e.valid_at AS valid_at,
               e.slot_id AS slot_id, e.research_id AS research_id
        ORDER BY e.fact
        """,
        rid=research_id,
    )

    checks: list[tuple[str, bool]] = []
    by_fact = {r["fact"]: r for r in records}

    undated = "FTX announced bankruptcy in mid-November"
    dated = "FTX filed for bankruptcy protection on 11 November 2022"

    checks.append(("both facts written verbatim", set(by_fact) == {undated, dated}))
    if undated in by_fact:
        row = by_fact[undated]
        checks.append(("undated fact carries NO valid_at", row["valid_at"] is None))
        checks.append(("slot_id queryable on edge", row["slot_id"] == SLOT))
        checks.append(("research_id queryable on edge", row["research_id"] == research_id))
    if dated in by_fact:
        got = by_fact[dated]["valid_at"]
        checks.append(("dated fact keeps 2022-11-11", got is not None and got.year == 2022
                       and got.month == 11 and got.day == 11))

    # No year may appear on an edge that the source never stated (§3.12).
    contaminated = [
        r["fact"] for r in records
        if r["valid_at"] is not None and str(r["valid_at"].year) not in r["fact"]
    ]
    checks.append(("no year invented on any edge", not contaminated))

    # Provenance chain: this resolves edges by reading episode.entity_edges
    # (graphiti.py:1437-1441), so it fails iff the §3.2 backlink was not written.
    found = await graphiti.get_nodes_and_edges_by_episode([result.episode_uuid])
    checks.append(
        ("provenance resolves via episode.entity_edges",
         len(found.edges) == len(result.edge_uuids))
    )

    print("--- assertions ---")
    ok = True
    for label, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")
        ok = ok and passed
    if contaminated:
        print(f"  contaminated facts: {contaminated}")

    # ---- cleanup ---------------------------------------------------------
    await driver.execute_query(
        "MATCH ()-[e:RELATES_TO {research_id: $rid}]->() DELETE e", rid=research_id
    )
    await driver.execute_query(
        "MATCH (n:Episodic {uuid: $u}) DETACH DELETE n", u=result.episode_uuid
    )
    await driver.execute_query(
        """
        MATCH (n:Entity) WHERE n.uuid IN $uuids AND NOT (n)--()
        DELETE n
        """,
        uuids=result.node_uuids,
    )
    print("\ncleaned up")

    await graphiti.close()
    print("SPIKE PASSED" if ok else "SPIKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
