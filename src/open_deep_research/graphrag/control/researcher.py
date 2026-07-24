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
from open_deep_research.graphrag.validation.sources import publisher_identity

SearchFn = Callable[..., Awaitable[list[SourceDocument]]]
ExtractFn = Callable[..., Awaitable[list[ExtractedTriple]]]
SupportExtractFn = Callable[..., Awaitable[list[ExtractedTriple]]]


class RoundResult(BaseModel):
    """What one round did, in the terms the supervisor needs back."""

    model_config = ConfigDict(extra="forbid")

    slot_id: str
    query: str
    documents_seen: list[str] = Field(default_factory=list)
    triples_extracted: int = 0
    facts_written: int = 0
    supports_added: int = 0
    episode_uuids: list[str] = Field(default_factory=list)
    contributing_sources: list[str] = Field(default_factory=list)
    contributing_source_identities: list[str] = Field(default_factory=list)
    target_edge_uuids: list[str] = Field(default_factory=list)
    corroborated_edge_uuids: list[str] = Field(default_factory=list)
    note: str = ""

    @property
    def is_corroborated(self) -> bool:
        """Whether every target claim received an independent supporting source."""

        targets = set(self.target_edge_uuids)
        return bool(targets) and targets.issubset(self.corroborated_edge_uuids)

    @property
    def succeeded(self) -> bool:
        """True only when facts reached the graph.

        Not "an episode was written" -- that is the V1 definition, and it is what
        made barren rounds look like progress.
        """

        return self.facts_written > 0 or self.supports_added > 0


async def run_research_round(
    graphiti: Any,
    *,
    topic: str,
    research_id: str,
    slot: OntologySlot,
    query: str,
    search: SearchFn,
    extract: ExtractFn,
    extract_support: SupportExtractFn | None = None,
    exclude_urls: list[str] | None = None,
    max_documents: int = 3,
    min_sources: int = 1,
    group_id: str = "neo4j",
) -> RoundResult:
    """Search for one slot, extract, and write whatever survives.

    ``min_sources`` is how many independent publisher identities must support
    the *same structured claim* before the round stops early.  The first source
    establishes target triples.  Further documents are passed to
    ``extract_support`` and written with ``support_only=True``: an unrelated fact
    in the same broad slot cannot masquerade as corroboration.

    Documents that yield nothing are still reported, so their URLs are excluded
    next time round.
    """

    result = RoundResult(slot_id=slot.slot_id, query=query)
    excluded = set(exclude_urls or [])

    documents = await search(query=query, exclude_urls=sorted(excluded))
    candidates: list[SourceDocument] = []
    candidate_publishers: set[str] = set()
    for document in documents:
        if document.url in excluded or not document.content.strip():
            continue
        identity = publisher_identity(
            document.url,
            fallback=document.document_id,
        )
        if identity and identity in candidate_publishers:
            continue
        candidates.append(document)
        if identity:
            candidate_publishers.add(identity)
        if len(candidates) >= max_documents:
            break

    if not candidates:
        result.note = "search returned no usable documents"
        return result

    primary_targets: list[ExtractedTriple] = []
    for document in candidates:
        if document.url:
            result.documents_seen.append(document.url)

        identity = publisher_identity(
            document.url,
            fallback=document.document_id,
        )
        if primary_targets:
            if identity in result.contributing_source_identities:
                continue
            if extract_support is None:
                continue
            triples = await extract_support(
                document=document,
                slot=slot,
                targets=primary_targets,
            )
        else:
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
                support_only=bool(primary_targets),
            ),
            default_group_id=group_id,
        )
        result.episode_uuids.append(write.episode_uuid)
        result.facts_written += len(write.created_edge_uuids)
        result.supports_added += len(write.supported_edge_uuids)
        result.corroborated_edge_uuids.extend(write.supported_edge_uuids)

        if write.edge_uuids:
            source_key = document.url or document.document_id
            if source_key and source_key not in result.contributing_sources:
                result.contributing_sources.append(source_key)
            if identity and identity not in result.contributing_source_identities:
                result.contributing_source_identities.append(identity)

        if not primary_targets and write.created_edge_uuids:
            primary_targets = triples
            result.target_edge_uuids = list(write.created_edge_uuids)

        if write.supported_edge_uuids:
            result.corroborated_edge_uuids = list(
                dict.fromkeys(result.corroborated_edge_uuids)
            )
            if result.is_corroborated and (
                len(result.contributing_source_identities) >= max(min_sources, 1)
            ):
                break
        elif write.created_edge_uuids and min_sources <= 1:
            break

    if result.succeeded and not result.is_corroborated and min_sources > 1:
        missing = len(
            set(result.target_edge_uuids) - set(result.corroborated_edge_uuids)
        )
        result.note = (
            f"{missing} structured claim(s) did not reach the requested "
            f"independent support count of {min_sources}"
        )
    elif not result.succeeded and not result.note:
        result.note = (
            f"{len(result.documents_seen)} document(s) searched, "
            f"{result.triples_extracted} triple(s) extracted, none reached the graph"
        )
    return result


async def run_support_round(
    graphiti: Any,
    *,
    research_id: str,
    slot: OntologySlot,
    target: ExtractedTriple,
    target_edge_uuid: str,
    query: str,
    search: SearchFn,
    extract_support: SupportExtractFn,
    exclude_urls: list[str] | None = None,
    exclude_source_identities: list[str] | None = None,
    max_documents: int = 3,
    group_id: str = "neo4j",
) -> RoundResult:
    """Run a targeted follow-up whose only allowed outcome is claim support.

    Coverage rounds search for an answer to a slot. This follow-up instead
    searches for one already-persisted structured claim and uses
    ``support_only=True``. It can add provenance to that exact edge, but it
    cannot create a convenient new fact merely to make a support metric rise.
    """

    result = RoundResult(
        slot_id=slot.slot_id,
        query=query,
        target_edge_uuids=[target_edge_uuid],
    )
    excluded_urls = set(exclude_urls or [])
    excluded_identities = set(exclude_source_identities or [])

    documents = await search(query=query, exclude_urls=sorted(excluded_urls))
    candidates: list[SourceDocument] = []
    candidate_publishers: set[str] = set()
    for document in documents:
        if document.url in excluded_urls or not document.content.strip():
            continue
        identity = publisher_identity(
            document.url,
            fallback=document.document_id,
        )
        if identity and (
            identity in excluded_identities or identity in candidate_publishers
        ):
            continue
        candidates.append(document)
        if identity:
            candidate_publishers.add(identity)
        if len(candidates) >= max_documents:
            break

    if not candidates:
        result.note = "support search returned no independent usable documents"
        return result

    for document in candidates:
        if document.url:
            result.documents_seen.append(document.url)
        identity = publisher_identity(
            document.url,
            fallback=document.document_id,
        )
        triples = await extract_support(
            document=document,
            slot=slot,
            targets=[target],
        )
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
                support_only=True,
            ),
            default_group_id=group_id,
        )
        result.episode_uuids.append(write.episode_uuid)
        result.supports_added += len(write.supported_edge_uuids)
        result.corroborated_edge_uuids.extend(write.supported_edge_uuids)

        if target_edge_uuid in write.supported_edge_uuids:
            source_key = document.url or document.document_id
            if source_key:
                result.contributing_sources.append(source_key)
            if identity:
                result.contributing_source_identities.append(identity)
            result.corroborated_edge_uuids = list(
                dict.fromkeys(result.corroborated_edge_uuids)
            )
            break

    if not result.succeeded:
        result.note = (
            f"{len(result.documents_seen)} independent document(s) searched; "
            "none supported the exact structured claim"
        )
    return result
