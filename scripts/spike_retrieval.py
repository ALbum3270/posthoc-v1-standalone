"""The last unverified link: does a real write actually come back out of search?

Full stack, no fakes: Kuzu storage + OpenRouter LLM extraction + OpenRouter
embeddings. Writes one episode through graphiti's own add_episode, then queries.

An empty result here is THE silent failure mode we care about -- writes succeed,
Cypher can see the rows, but hybrid search returns nothing, and the symptom
downstream looks like "the stopping logic is broken".
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(r"c:\Users\Lenovo\Desktop\Langgraph\open_deep_research-main\.env"))

from graphiti_core import Graphiti  # noqa: E402

from graphiti_core.embedder.client import EMBEDDING_DIM  # noqa: E402
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig  # noqa: E402
from graphiti_core.llm_client.config import LLMConfig  # noqa: E402
from graphiti_core.llm_client.openai_client import OpenAIClient  # noqa: E402
from graphiti_core.nodes import EpisodeType  # noqa: E402
from graphiti_core.search.search_config import (  # noqa: E402
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    SearchConfig,
)

# graphiti's own recipes all pair bm25 with cosine_similarity. This one isolates
# the vector path, so a fulltext-index problem cannot mask an embedding problem.
VECTOR_ONLY = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.cosine_similarity],
        reranker=EdgeReranker.rrf,
    ),
    limit=3,
)

CHAT_MODEL = "openai/gpt-4.1-mini"
EMBED_MODEL = "openai/text-embedding-3-small"

EPISODE_TEXT = (
    "On 11 November 2022, the cryptocurrency exchange FTX filed for Chapter 11 "
    "bankruptcy protection in Delaware. Founder Sam Bankman-Fried resigned as CEO "
    "the same day and was replaced by John J. Ray III. The collapse followed a "
    "liquidity crisis triggered by reporting on the balance sheet of Alameda "
    "Research, FTX's affiliated trading firm."
)

QUERIES = [
    "Who founded FTX?",
    "When did FTX file for bankruptcy?",
    "What triggered the FTX collapse?",
]


async def main() -> int:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    print(f"endpoint: {base_url}")
    print(f"llm: {CHAT_MODEL}   embed: {EMBED_MODEL}   EMBEDDING_DIM={EMBEDDING_DIM}\n")

    # Neo4j Aura. Kuzu was tried first and abandoned: graphiti 0.28.2's Kuzu
    # backend never materializes the FTS index, and add_episode's entity dedup
    # depends on fulltext search, so ingestion cannot complete there at all.
    graphiti = Graphiti(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"],
        llm_client=OpenAIClient(
            config=LLMConfig(api_key=api_key, base_url=base_url, model=CHAT_MODEL)
        ),
        embedder=OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=api_key, base_url=base_url, embedding_model=EMBED_MODEL
            )
        ),
    )

    await graphiti.build_indices_and_constraints()
    print("[1] indices built")

    # group_id left as None on purpose: passing one makes graphiti reinterpret it
    # as a database name (graphiti.py:887-890), which is not what we want here.
    result = await graphiti.add_episode(
        name="ftx-collapse-doc-1",
        episode_body=EPISODE_TEXT,
        source_description="https://example.com/ftx",
        reference_time=datetime.now(timezone.utc),
        source=EpisodeType.text,
    )
    print(f"[2] episode written  uuid={result.episode.uuid[:8]}")
    print(f"    entities extracted ({len(result.nodes)}): "
          f"{[n.name for n in result.nodes]}")
    print(f"    facts extracted   ({len(result.edges)}):")
    for e in result.edges:
        print(f"      - {e.name}: {e.fact[:78]}")

    # Did the embedder actually run? An unembedded graph is silently unsearchable.
    embedded_nodes = sum(1 for n in result.nodes if n.name_embedding)
    embedded_edges = sum(1 for e in result.edges if e.fact_embedding)
    print(f"\n[3] embeddings present: {embedded_nodes}/{len(result.nodes)} nodes, "
          f"{embedded_edges}/{len(result.edges)} edges")

    print("\n[4a] vector-only recall (isolates the embedding path)")
    recalled = 0
    for q in QUERIES:
        try:
            res = await graphiti.search_(query=q, config=VECTOR_ONLY)
            hits = res.edges
        except Exception as exc:
            print(f"  ERR  {q} -> {type(exc).__name__}: {str(exc)[:90]}")
            continue
        if hits:
            recalled += 1
        print(f"  {'HIT ' if hits else 'MISS'} {q}")
        for h in hits[:2]:
            print(f"         -> {h.fact[:74]}")

    print("\n[4b] full hybrid recall (bm25 + vector; needs the FTS index)")
    hybrid_ok = 0
    for q in QUERIES:
        try:
            hits = await graphiti.search(query=q, num_results=3)
        except Exception as exc:
            print(f"  ERR  {q} -> {type(exc).__name__}: {str(exc)[:90]}")
            continue
        if hits:
            hybrid_ok += 1
        print(f"  {'HIT ' if hits else 'MISS'} {q}")

    await graphiti.close()

    ok = (
        len(result.nodes) > 0
        and len(result.edges) > 0
        and embedded_nodes == len(result.nodes)
        and embedded_edges == len(result.edges)
        and recalled == len(QUERIES)
    )
    print(f"\nextraction={len(result.nodes)}n/{len(result.edges)}e  "
          f"vector_recall={recalled}/{len(QUERIES)}  "
          f"hybrid_recall={hybrid_ok}/{len(QUERIES)}")
    print("SPIKE PASSED" if ok else "SPIKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
