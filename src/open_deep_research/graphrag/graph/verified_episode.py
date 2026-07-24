"""Structured, verbatim write path into Graphiti.

``add_verified_episode`` exists because the default route -- flatten triples into
prose, hand them to ``Graphiti.add_episode``, let its LLM extract them again --
demonstrably corrupts facts. In the 2026-07-24 baseline run a single source
triple with no year ("FTX announced bankruptcy in mid-November") came back out of
the graph as two contradictory dated edges, 2023 and 2026 (SESSION_HANDOFF
§3.12). Nothing downstream can repair that; coverage, stopping and the final
report all inherit it.

This module writes what extraction produced and nothing else. Design decisions:

* **No LLM anywhere on this path.** Graphiti's ``resolve_extracted_edges`` is an
  LLM edge-resolution step that may rephrase ``fact``; that would defeat the
  point. Entity de-duplication here is deterministic (normalized name within the
  research session), which also avoids the per-triple "2 hybrid searches + 1 LLM
  call" cost noted in §3.1.
* **Dates are gated, not inferred** -- see ``validation.dates``.
* **Provenance is a hard requirement.** The ``episode.entity_edges`` backlink is
  written *after* the edges exist, because ``get_nodes_and_edges_by_episode``
  resolves edges through that field and silently returns nothing when it is empty
  (§3.2). This is the trap the original spike hit first.

Built on the four public APIs proven sufficient by ``scripts/spike_write_path.py``
(8/8, no API cost), so graphiti-core stays unforked and pinned at 0.28.2 (§2.2).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from open_deep_research.graphrag.schemas import (
    ExtractedTriple,
    GraphWriteResult,
    VerifiedEpisodeInput,
)
from open_deep_research.graphrag.validation.dates import resolve_valid_at, stated_years

# Writing any of these through `attributes` would collide with a core EntityEdge
# field. `edges.py:359` does `edge_data.update(self.attributes or {})`, putting
# attributes in the same namespace as the real columns, so a collision silently
# overwrites graph state rather than erroring (§2.5).
#
# These are the persisted key names from `edges.py:337-350`, which are NOT all
# the same as the Python attribute names: the endpoints serialize as
# `source_uuid` / `target_uuid`, not `source_node_uuid` / `target_node_uuid`.
# Denylisting the Python names would have let the real collision through.
# `attributes` is included because the Kuzu branch (`edges.py:353`) uses it as a
# column of its own.
RESERVED_EDGE_ATTRIBUTES = frozenset(
    {
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
        "attributes",
    }
)

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w一-鿿]+", re.UNICODE)


class ReservedAttributeError(ValueError):
    """Raised when a caller tries to write an attribute that would clobber a core field."""


def assert_no_reserved_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Reject attribute names that would silently overwrite a core edge field.

    Returns the attributes unchanged so this can wrap a construction expression.
    """

    collisions = RESERVED_EDGE_ATTRIBUTES.intersection(attributes)
    if collisions:
        raise ReservedAttributeError(
            f"attributes would overwrite core EntityEdge fields: {sorted(collisions)}"
        )
    return attributes


def normalize_entity_name(name: str) -> str:
    """Fold a name to its de-duplication key.

    Deliberately conservative: case, surrounding punctuation and whitespace only.
    It will not merge "SBF" with "Sam Bankman-Fried" -- that is a semantic
    judgement, and making it here would mean guessing on the write path.
    """

    folded = _NON_WORD.sub(" ", name or "").strip().lower()
    return _WS.sub(" ", folded)


def render_fact(triple: ExtractedTriple) -> str:
    """Render a triple to the exact text stored on the edge.

    Prefers the verbatim source quote when extraction captured one, so the graph
    holds the source's own words rather than a reconstruction.
    """

    if triple.source_span is not None and triple.source_span.quote:
        return triple.source_span.quote.strip()

    obj = triple.object if isinstance(triple.object, str) else triple.object.name
    return _WS.sub(" ", f"{triple.subject.name} {triple.predicate} {obj}").strip()


def edge_relation_name(predicate: str) -> str:
    """Derive a stable relation label from a predicate."""

    token = _NON_WORD.sub("_", (predicate or "related_to").strip()).strip("_").upper()
    return token or "RELATED_TO"


def build_edge_attributes(
    payload: VerifiedEpisodeInput,
    triple: ExtractedTriple,
    fact_text: str,
) -> dict[str, Any]:
    """Assemble the queryable contract carried on each fact edge.

    ``slot_id`` and ``research_id`` land here rather than in a formatted string
    because M0 has to aggregate on them: with slot attribution living only in the
    episode name, coverage can only count "did we write an episode for this slot",
    which is what the in-memory ``filled_ids`` already said (§5).
    """

    attributes: dict[str, Any] = {
        "research_id": payload.research_id,
        "slot_id": payload.slot_id,
        "source_document_id": triple.source_document_id or payload.source.document_id,
        "source_url": payload.source.url or "",
        "source_title": payload.source.title,
        "extraction_confidence": float(triple.confidence),
        "stated_years": sorted(stated_years(fact_text)),
        "conflicts_with": [],
    }

    if triple.source_span is not None:
        attributes["source_span_start"] = triple.source_span.start_char
        attributes["source_span_end"] = triple.source_span.end_char

    if payload.verification is not None:
        attributes["verification_status"] = payload.verification.status.value
        attributes["truth_score"] = payload.verification.truth_score
        attributes["verification_confidence"] = payload.verification.confidence_score

    return assert_no_reserved_attributes(attributes)


def resolve_reference_time(payload: VerifiedEpisodeInput) -> datetime:
    """Pick the episode's reference time.

    Publication date first. Graphiti resolves relative time expressions against
    this value (``prompts/extract_edges.py:78``), so passing ``now()`` for a 2022
    article is what produced the 2026 dates (§3.12 fix 1). Nothing on *this* path
    re-extracts, but the value is stored on the episode and reused by anything
    that reads it later, so it must still be right.
    """

    return (
        payload.source.published_at
        or payload.source.retrieved_at
        or datetime.now(timezone.utc)
    )


async def _lookup_entity_uuid_by_name(
    driver: Any, group_id: str, normalized: str
) -> str | None:
    """Find an existing entity in this group whose normalized name matches."""

    records, _, _ = await driver.execute_query(
        """
        MATCH (n:Entity {group_id: $group_id})
        WHERE toLower(trim(n.name)) = $normalized
        RETURN n.uuid AS uuid
        LIMIT 1
        """,
        group_id=group_id,
        normalized=normalized,
    )
    for record in records:
        return record["uuid"]
    return None


async def add_verified_episode(
    graphiti: Any,
    payload: VerifiedEpisodeInput,
    *,
    default_group_id: str = "neo4j",
) -> GraphWriteResult:
    """Write pre-extracted triples to Graphiti without a second extraction pass.

    Returns the episode, entity and edge uuids that were written, plus the facts
    that were skipped and why.
    """

    from graphiti_core.edges import EntityEdge, EpisodicEdge
    from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode

    driver = graphiti.driver
    embedder = graphiti.embedder
    group_id = payload.group_id or default_group_id
    now = datetime.now(timezone.utc)
    reference_time = resolve_reference_time(payload)

    facts = [(triple, render_fact(triple)) for triple in payload.triples]
    usable = [(triple, text) for triple, text in facts if text]
    skipped = [render_fact(t) or f"<empty triple for {payload.slot_id}>" for t, text in facts if not text]

    episode = EpisodicNode(
        name=payload.episode_name
        or f"{payload.research_id}::{payload.slot_id}::{payload.source.document_id}",
        group_id=group_id,
        labels=[],
        source=EpisodeType.text,
        content="\n".join(text for _, text in usable),
        source_description=(
            f"verified | research_id={payload.research_id}"
            f" | slot_id={payload.slot_id}"
            f" | url={payload.source.url or ''}"
        ),
        created_at=now,
        valid_at=reference_time,
    )
    await episode.save(driver)

    if not usable:
        return GraphWriteResult(episode_uuid=episode.uuid, skipped_triples=skipped)

    # ---- entities: deterministic de-duplication, no LLM ---------------------
    names: list[str] = []
    for triple, _ in usable:
        names.append(triple.subject.name)
        names.append(
            triple.object if isinstance(triple.object, str) else triple.object.name
        )

    uuid_by_key: dict[str, str] = {}
    new_nodes: list[Any] = []
    for name in names:
        key = normalize_entity_name(name)
        if not key or key in uuid_by_key:
            continue
        existing = await _lookup_entity_uuid_by_name(driver, group_id, key)
        if existing:
            uuid_by_key[key] = existing
            continue
        node = EntityNode(name=name.strip(), group_id=group_id, labels=["Entity"])
        new_nodes.append(node)
        uuid_by_key[key] = node.uuid

    if new_nodes:
        vectors = await embedder.create_batch([node.name for node in new_nodes])
        for node, vector in zip(new_nodes, vectors, strict=True):
            node.name_embedding = vector
            await node.save(driver)

    # ---- fact edges: verbatim text, gated dates -----------------------------
    fact_vectors = await embedder.create_batch([text for _, text in usable])
    edges: list[Any] = []
    for (triple, text), vector in zip(usable, fact_vectors, strict=True):
        obj_name = (
            triple.object if isinstance(triple.object, str) else triple.object.name
        )
        source_uuid = uuid_by_key.get(normalize_entity_name(triple.subject.name))
        target_uuid = uuid_by_key.get(normalize_entity_name(obj_name))
        if not source_uuid or not target_uuid:
            skipped.append(text)
            continue

        edge = EntityEdge(
            source_node_uuid=source_uuid,
            target_node_uuid=target_uuid,
            name=edge_relation_name(triple.predicate),
            fact=text,
            group_id=group_id,
            created_at=now,
            valid_at=resolve_valid_at(text, published_at=payload.source.published_at),
            episodes=[episode.uuid],
            fact_embedding=vector,
            attributes=build_edge_attributes(payload, triple, text),
        )
        await edge.save(driver)
        edges.append(edge)

    # ---- provenance: MENTIONS edges, then the backlink ----------------------
    for node_uuid in dict.fromkeys(uuid_by_key.values()):
        await EpisodicEdge(
            source_node_uuid=episode.uuid,
            target_node_uuid=node_uuid,
            group_id=group_id,
            created_at=now,
        ).save(driver)

    # Must come last: get_nodes_and_edges_by_episode reads this field, and an
    # empty list makes every provenance lookup silently return nothing (§3.2).
    episode.entity_edges = [edge.uuid for edge in edges]
    await episode.save(driver)

    return GraphWriteResult(
        episode_uuid=episode.uuid,
        node_uuids=list(dict.fromkeys(uuid_by_key.values())),
        edge_uuids=[edge.uuid for edge in edges],
        skipped_triples=skipped,
    )
