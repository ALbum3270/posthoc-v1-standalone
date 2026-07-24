"""One research round: search, extract, write, report what happened.

Composes the M1 pieces rather than reimplementing them -- the adapter fills
``published_at``, the verified writer keeps facts verbatim and gates dates.

Two rules this node exists to enforce, both learned from the V1 baseline:

* **A slot counts as filled only when facts actually reach the graph.** V1 marked
  it filled as soon as an episode was written, so barren rounds looked like
  progress and coverage overstated what was known (§3.11, §3.15).
* **Every attempt is reported, successful or not.** The supervisor's failure
  memory is what breaks the livelock, and it can only work if failures come back.

Search and extraction are injected. Both need the network or a model, and keeping
them at the edge leaves the round logic testable without either.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field

from open_deep_research.graphrag.graph.verified_episode import add_verified_episode
from open_deep_research.graphrag.ontology import OntologySlot
from open_deep_research.graphrag.schemas import (
    ExtractedTriple,
    GraphWriteResult,
    SourceDocument,
    VerifiedEpisodeInput,
)

SearchFn = Callable[..., Awaitable[list[SourceDocument]]]
ExtractFn = Callable[..., Awaitable[list[ExtractedTriple]]]


class RoundResult(BaseModel):
    """What one round did, in the terms the supervisor needs back."""

    model_config = ConfigDict(extra="forbid")

    slot_id: str
    query: str
    documents_seen: list[str] = Field(default_factory=list)
    triples_extracted: int = 0
    facts_written: int = 0
    episode_uuids: list[str] = Field(default_factory=list)
    contributing_sources: list[str] = Field(default_factory=list)
    note: str = ""

    @property
    def is_corroborated(self) -> bool:
        """Whether more than one source contributed facts to this slot."""

        return len(self.contributing_sources) > 1

    @property
    def succeeded(self) -> bool:
        """True only when facts reached the graph.

        Not "an episode was written" -- that is the V1 definition, and it is what
        made barren rounds look like progress.
        """

        return self.facts_written > 0


async def run_research_round(
    graphiti: Any,
    *,
    topic: str,
    research_id: str,
    slot: OntologySlot,
    query: str,
    search: SearchFn,
    extract: ExtractFn,
    exclude_urls: list[str] | None = None,
    max_documents: int = 3,
    min_sources: int = 1,
    group_id: str = "neo4j",
) -> RoundResult:
    """Search for one slot, extract, and write whatever survives.

    ``min_sources`` is how many *distinct sources* must contribute facts before
    the round stops early. At the default of 1 the round ends as soon as the slot
    is answered, which is cheap but leaves every conclusion resting on a single
    page -- the M4 regression measured 0% cross-corroboration across all three
    topics for exactly this reason. Raising it to 2 buys a second, independent
    source per slot at the cost of one more extraction call.

    Documents that yield nothing are still reported, so their URLs are excluded
    next time round.
    """

    result = RoundResult(slot_id=slot.slot_id, query=query)
    excluded = set(exclude_urls or [])

    documents = await search(query=query, exclude_urls=sorted(excluded))
    candidates = [
        document
        for document in documents
        if document.url not in excluded and document.content.strip()
    ][:max_documents]

    if not candidates:
        result.note = "search returned no usable documents"
        return result

    for document in candidates:
        if document.url:
            result.documents_seen.append(document.url)

        triples = await extract(document=document, slot=slot)
        if not triples:
            continue
        result.triples_extracted += len(triples)

        write: GraphWriteResult = await add_verified_episode(
            graphiti,
            VerifiedEpisodeInput(
                research_id=research_id,
                slot_id=slot.slot_id,
                source=document,
                triples=triples,
                group_id=group_id,
            ),
            default_group_id=group_id,
        )
        result.episode_uuids.append(write.episode_uuid)
        result.facts_written += len(write.edge_uuids)

        if write.edge_uuids:
            source_key = document.url or document.document_id
            if source_key and source_key not in result.contributing_sources:
                result.contributing_sources.append(source_key)
            if len(result.contributing_sources) >= max(min_sources, 1):
                break

    if result.succeeded and not result.is_corroborated and min_sources > 1:
        result.note = (
            f"only {len(result.contributing_sources)} source contributed; "
            f"wanted {min_sources}"
        )
    elif not result.succeeded and not result.note:
        result.note = (
            f"{len(result.documents_seen)} document(s) searched, "
            f"{result.triples_extracted} triple(s) extracted, none reached the graph"
        )
    return result
