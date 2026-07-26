"""Structured, verbatim write path into Graphiti.

``add_verified_episode`` exists because the default route -- flatten triples into
prose, hand them to ``Graphiti.add_episode``, let its LLM extract them again --
demonstrably corrupts facts. In the 2026-07-24 baseline run a single source
triple with no year ("FTX announced bankruptcy in mid-November") came back out of
the graph as two contradictory dated edges, 2023 and 2026 (SESSION_HANDOFF
§3.12). Nothing downstream can repair that; coverage, stopping and the final
report all inherit it.

This module writes what extraction produced and nothing else. Design decisions:

* **No LLM may rewrite persisted facts.** Graphiti's
  ``resolve_extracted_edges`` is an LLM edge-resolution step that may rephrase
  ``fact``; that would defeat the point. Entity de-duplication here is
  deterministic. The only model call is a structured, read-only equivalence
  judgment after deterministic candidate recall; its output is never stored as
  claim text.
* **Dates are gated, not inferred** -- see ``validation.dates``.
* **Provenance is a hard requirement.** The ``episode.entity_edges`` backlink is
  written *after* the edges exist, because ``get_nodes_and_edges_by_episode``
  resolves edges through that field and silently returns nothing when it is empty
  (§3.2). This is the trap the original spike hit first.

Built on the four public APIs proven sufficient by ``scripts/spike_write_path.py``
(8/8, no API cost), so graphiti-core stays unforked and pinned at 0.28.2 (§2.2).
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict

from graphiti_core.prompts.models import Message
from open_deep_research.graphrag.reporting.evidence_pack import _magnitudes
from open_deep_research.graphrag.schemas import (
    ClaimMatchAuditRecord,
    ExtractedTriple,
    GraphWriteResult,
    VerifiedEpisodeInput,
)
from open_deep_research.graphrag.validation.dates import (
    extract_explicit_dates,
    resolve_valid_at,
    stated_years,
)
from open_deep_research.graphrag.validation.sources import publisher_identity

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

DEFAULT_CLAIM_MATCH_SIMILARITY_THRESHOLD = 0.70
DEFAULT_CLAIM_MATCH_TOP_K = 3

_BIDIRECTIONAL_ENTAILMENT_PROMPT = (
    "Judge whether two natural-language claims express exactly the same factual "
    "content by testing strict entailment in both directions. Use only what each "
    "claim explicitly states. Relatedness is not entailment. A claim does not "
    "entail another if it changes or omits a participant, date, quantity, scope, "
    "polarity, modality, causal relationship, or other material detail. Return "
    "the two directional decisions independently and give a concise reason."
)


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
        "supporting_source_urls": [payload.source.url or ""],
        "supporting_source_titles": [payload.source.title],
        "supporting_source_identities": [
            publisher_identity(
                payload.source.url,
                fallback=payload.source.document_id,
            )
        ],
        "supporting_quotes": [fact_text],
        "extraction_confidence": float(triple.confidence),
        "relevance_status": triple.relevance_status.value,
        "relevance_confidence": triple.relevance_confidence,
        "relevance_reason": triple.relevance_reason or "",
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
    driver: Any, group_id: str, name: str
) -> str | None:
    """Find an existing entity in this group whose normalized name matches."""

    normalized = normalize_entity_name(name)
    records, _, _ = await driver.execute_query(
        """
        MATCH (n:Entity {group_id: $group_id})
        WHERE toLower(trim(n.name)) = $exact_name
        SET n.normalized_name = coalesce(n.normalized_name, $normalized)
        RETURN n.uuid AS uuid
        LIMIT 1
        """,
        group_id=group_id,
        normalized=normalized,
        exact_name=(name or "").strip().casefold(),
    )
    for record in records:
        return record["uuid"]

    # New M5 nodes carry this property. The exact-name query above also
    # backfills it on reused pre-M5 nodes, avoiding a migration and making the
    # normalized fallback progressively complete.
    records, _, _ = await driver.execute_query(
        """
        MATCH (n:Entity {group_id: $group_id, normalized_name: $normalized})
        RETURN n.uuid AS uuid
        LIMIT 1
        """,
        group_id=group_id,
        normalized=normalized,
    )
    for record in records:
        return record["uuid"]
    return None


class _BidirectionalEntailmentDecision(BaseModel):
    """Structured LLM decision; both directions must pass."""

    model_config = ConfigDict(extra="forbid")

    candidate_entails_incoming: bool
    incoming_entails_candidate: bool
    reason: str


async def _lookup_matching_claim_edge(
    driver: Any,
    *,
    research_id: str,
    slot_id: str,
    group_id: str,
    source_uuid: str,
    target_uuid: str,
    relation_name: str,
) -> dict[str, Any] | None:
    """Find the persisted claim represented by the same structured triple."""

    records, _, _ = await driver.execute_query(
        """
        MATCH (s:Entity {uuid: $source_uuid})
              -[e:RELATES_TO]->
              (t:Entity {uuid: $target_uuid})
        WHERE e.research_id = $research_id
          AND e.slot_id = $slot_id
          AND e.group_id = $group_id
          AND e.name = $relation_name
          AND e.expired_at IS NULL
        RETURN e.uuid AS uuid,
               e.episodes AS episodes,
               e.source_url AS source_url,
               e.source_title AS source_title,
               e.fact AS fact,
               e.supporting_source_urls AS supporting_source_urls,
               e.supporting_source_titles AS supporting_source_titles,
               e.supporting_source_identities AS supporting_source_identities,
               e.supporting_quotes AS supporting_quotes
        LIMIT 1
        """,
        research_id=research_id,
        slot_id=slot_id,
        group_id=group_id,
        source_uuid=source_uuid,
        target_uuid=target_uuid,
        relation_name=relation_name,
    )
    for record in records:
        return dict(record)
    return None


def _coerce_embedding(value: Any) -> list[float]:
    """Normalize provider-specific embedding storage to a float list."""

    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)):
        return []
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return []


def _cosine_similarity(left: list[float], right: list[float]) -> float | None:
    """Return cosine similarity, rejecting malformed or zero vectors."""

    if not left or len(left) != len(right):
        return None
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


async def _lookup_semantic_claim_candidates(
    driver: Any,
    *,
    research_id: str,
    slot_id: str,
    group_id: str,
    query_embedding: list[float],
    similarity_threshold: float,
    top_k: int,
) -> list[dict[str, Any]]:
    """Recall the most similar active claims within the same investigation slot."""

    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("claim match similarity threshold must be between 0 and 1")
    if not 1 <= top_k <= 3:
        raise ValueError("claim match top_k must be between 1 and 3")

    records, _, _ = await driver.execute_query(
        """
        /* semantic claim candidates */
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.research_id = $research_id
          AND e.slot_id = $slot_id
          AND e.group_id = $group_id
          AND e.expired_at IS NULL
          AND e.fact_embedding IS NOT NULL
        RETURN e.uuid AS uuid,
               e.episodes AS episodes,
               e.source_url AS source_url,
               e.source_title AS source_title,
               e.fact AS fact,
               e.fact_embedding AS fact_embedding,
               e.supporting_source_urls AS supporting_source_urls,
               e.supporting_source_titles AS supporting_source_titles,
               e.supporting_source_identities AS supporting_source_identities,
               e.supporting_quotes AS supporting_quotes
        """,
        research_id=research_id,
        slot_id=slot_id,
        group_id=group_id,
    )
    candidates: list[dict[str, Any]] = []
    for record in records:
        candidate = dict(record)
        similarity = _cosine_similarity(
            query_embedding,
            _coerce_embedding(candidate.get("fact_embedding")),
        )
        if similarity is None or similarity < similarity_threshold:
            continue
        candidate["similarity"] = similarity
        candidates.append(candidate)
    candidates.sort(
        key=lambda item: (-float(item["similarity"]), str(item.get("uuid") or ""))
    )
    return candidates[:top_k]


def _candidate_source_identities(match: dict[str, Any]) -> list[str]:
    """Publisher identities already supporting a persisted claim."""

    identities = [
        str(item)
        for item in (match.get("supporting_source_identities") or [])
        if str(item).strip()
    ]
    if not identities:
        fallback = publisher_identity(match.get("source_url"))
        if fallback:
            identities.append(fallback)
    return list(dict.fromkeys(identities))


def _deterministic_claim_prefilter(
    candidate_fact: str,
    incoming_fact: str,
) -> tuple[bool, str]:
    """Reject explicit date or magnitude contradictions before any LLM call."""

    candidate_years = stated_years(candidate_fact)
    incoming_years = stated_years(incoming_fact)
    if candidate_years and incoming_years and not candidate_years & incoming_years:
        return (
            False,
            "explicit year conflict: "
            f"candidate={sorted(candidate_years)}, incoming={sorted(incoming_years)}",
        )

    candidate_dates = extract_explicit_dates(candidate_fact)
    incoming_dates = extract_explicit_dates(incoming_fact)
    for precision in ("day", "month"):
        candidate_values = {
            evidence.value
            for evidence in candidate_dates
            if evidence.precision == precision
        }
        incoming_values = {
            evidence.value
            for evidence in incoming_dates
            if evidence.precision == precision
        }
        if candidate_values and incoming_values and not candidate_values & incoming_values:
            return (
                False,
                f"explicit {precision} conflict: "
                f"candidate={sorted(candidate_values)}, "
                f"incoming={sorted(incoming_values)}",
            )

    candidate_magnitudes = _magnitudes(candidate_fact)
    incoming_magnitudes = _magnitudes(incoming_fact)
    for unit in sorted(candidate_magnitudes.keys() & incoming_magnitudes.keys()):
        candidate_values = candidate_magnitudes[unit]
        incoming_values = incoming_magnitudes[unit]
        if candidate_values and incoming_values and not candidate_values & incoming_values:
            return (
                False,
                "explicit magnitude conflict: "
                f"unit={unit}, candidate={sorted(candidate_values)}, "
                f"incoming={sorted(incoming_values)}",
            )

    return True, "no explicit date or magnitude conflict"


async def _judge_bidirectional_entailment(
    llm_client: Any,
    *,
    candidate_fact: str,
    incoming_fact: str,
    group_id: str,
) -> _BidirectionalEntailmentDecision:
    """Require strict entailment in both directions for semantic claim identity."""

    response = await llm_client.generate_response(
        [
            Message(role="system", content=_BIDIRECTIONAL_ENTAILMENT_PROMPT),
            Message(
                role="user",
                content=json.dumps(
                    {
                        "candidate_claim": candidate_fact,
                        "incoming_claim": incoming_fact,
                    },
                    ensure_ascii=False,
                ),
            ),
        ],
        response_model=_BidirectionalEntailmentDecision,
        group_id=group_id,
        prompt_name="claim_match.bidirectional_entailment",
    )
    return _BidirectionalEntailmentDecision.model_validate(response)


async def _append_claim_support(
    driver: Any,
    *,
    match: dict[str, Any],
    episode_uuid: str,
    source_url: str,
    source_title: str,
    source_identity: str,
    quote: str,
) -> str | None:
    """Attach one supporting episode and its source metadata to an existing claim."""

    episodes = [str(item) for item in (match.get("episodes") or []) if item]
    urls = [str(item or "") for item in (match.get("supporting_source_urls") or [])]
    titles = [
        str(item or "") for item in (match.get("supporting_source_titles") or [])
    ]
    identities = [
        str(item or "")
        for item in (match.get("supporting_source_identities") or [])
    ]
    quotes = [str(item or "") for item in (match.get("supporting_quotes") or [])]

    # Seed arrays on edges written before M5.
    if not urls and match.get("source_url") is not None:
        legacy_url = str(match.get("source_url") or "")
        urls.append(legacy_url)
        titles.append(str(match.get("source_title") or ""))
        identities.append(publisher_identity(legacy_url))
        quotes.append(str(match.get("fact") or ""))

    if source_identity and source_identity in identities:
        return None

    if episode_uuid in episodes:
        return None

    episodes.append(episode_uuid)
    urls.append(source_url)
    titles.append(source_title)
    identities.append(source_identity)
    quotes.append(quote)

    await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO {uuid: $edge_uuid}]->()
        SET e.episodes = $episodes,
            e.supporting_source_urls = $supporting_source_urls,
            e.supporting_source_titles = $supporting_source_titles,
            e.supporting_source_identities = $supporting_source_identities,
            e.supporting_quotes = $supporting_quotes
        RETURN e.uuid AS uuid
        """,
        edge_uuid=str(match["uuid"]),
        episodes=episodes,
        supporting_source_urls=urls,
        supporting_source_titles=titles,
        supporting_source_identities=identities,
        supporting_quotes=quotes,
    )
    return str(match["uuid"])


async def add_verified_episode(
    graphiti: Any,
    payload: VerifiedEpisodeInput,
    *,
    default_group_id: str = "neo4j",
    claim_match_similarity_threshold: float = (
        DEFAULT_CLAIM_MATCH_SIMILARITY_THRESHOLD
    ),
    claim_match_top_k: int = DEFAULT_CLAIM_MATCH_TOP_K,
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
    skipped = [
        render_fact(triple) or f"<empty triple for {payload.slot_id}>"
        for triple, text in facts
        if not text
    ]

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
        existing = await _lookup_entity_uuid_by_name(driver, group_id, name)
        if existing:
            uuid_by_key[key] = existing
            continue
        if payload.support_only:
            continue
        node = EntityNode(
            name=name.strip(),
            group_id=group_id,
            labels=["Entity"],
            attributes={"normalized_name": key},
        )
        new_nodes.append(node)
        uuid_by_key[key] = node.uuid

    if new_nodes:
        vectors = await embedder.create_batch([node.name for node in new_nodes])
        for node, vector in zip(new_nodes, vectors, strict=True):
            node.name_embedding = vector
            await node.save(driver)

    # ---- fact edges: merge true claim support, otherwise create -------------
    edges: list[Any] = []
    supported_edge_uuids: list[str] = []
    claim_match_audit: list[ClaimMatchAuditRecord] = []
    new_candidates: list[
        tuple[ExtractedTriple, str, str, str, str, list[float]]
    ] = []
    pending_claim_keys: set[tuple[str, str, str]] = set()
    embedding_by_text: dict[str, list[float]] = {}

    async def embedding_for(text: str) -> list[float]:
        if text not in embedding_by_text:
            vectors = await embedder.create_batch([text])
            embedding_by_text[text] = list(vectors[0])
        return embedding_by_text[text]

    source_identity = publisher_identity(
        payload.source.url,
        fallback=payload.source.document_id,
    )
    for triple, text in usable:
        obj_name = (
            triple.object if isinstance(triple.object, str) else triple.object.name
        )
        source_uuid = uuid_by_key.get(normalize_entity_name(triple.subject.name))
        target_uuid = uuid_by_key.get(normalize_entity_name(obj_name))

        relation_name = edge_relation_name(triple.predicate)
        exact_match = (
            await _lookup_matching_claim_edge(
                driver,
                research_id=payload.research_id,
                slot_id=payload.slot_id,
                group_id=group_id,
                source_uuid=source_uuid,
                target_uuid=target_uuid,
                relation_name=relation_name,
            )
            if source_uuid and target_uuid
            else None
        )
        if exact_match is not None:
            prefilter_passed, prefilter_reason = _deterministic_claim_prefilter(
                str(exact_match.get("fact") or ""),
                text,
            )
            independent_source = (
                source_identity
                not in _candidate_source_identities(exact_match)
            )
            if prefilter_passed and not independent_source:
                prefilter_passed = False
                prefilter_reason = (
                    f"same publisher identity is not independent: {source_identity}"
                )
            supported_uuid = None
            if prefilter_passed:
                supported_uuid = await _append_claim_support(
                    driver,
                    match=exact_match,
                    episode_uuid=episode.uuid,
                    source_url=payload.source.url or "",
                    source_title=payload.source.title,
                    source_identity=source_identity,
                    quote=text,
                )
            claim_match_audit.append(
                ClaimMatchAuditRecord(
                    incoming_fact=text,
                    candidate_edge_uuid=str(exact_match["uuid"]),
                    candidate_fact=str(exact_match.get("fact") or ""),
                    match_method="exact",
                    prefilter_passed=prefilter_passed,
                    prefilter_reason=prefilter_reason,
                    entailment_reason=(
                        "exact structured match; semantic entailment not required"
                        if prefilter_passed
                        else "entailment not called because prefilter rejected"
                    ),
                    accepted=supported_uuid is not None,
                )
            )
            if supported_uuid is not None:
                supported_edge_uuids.append(supported_uuid)
                continue
            if payload.support_only or independent_source is False:
                skipped.append(text)
                continue

        query_embedding = await embedding_for(text)
        semantic_candidates = (
            []
            if exact_match is not None
            else await _lookup_semantic_claim_candidates(
                driver,
                research_id=payload.research_id,
                slot_id=payload.slot_id,
                group_id=group_id,
                query_embedding=query_embedding,
                similarity_threshold=claim_match_similarity_threshold,
                top_k=claim_match_top_k,
            )
        )
        semantic_match_uuid: str | None = None
        for candidate in semantic_candidates:
            candidate_fact = str(candidate.get("fact") or "")
            prefilter_passed, prefilter_reason = _deterministic_claim_prefilter(
                candidate_fact,
                text,
            )
            if (
                prefilter_passed
                and source_identity in _candidate_source_identities(candidate)
            ):
                prefilter_passed = False
                prefilter_reason = (
                    f"same publisher identity is not independent: {source_identity}"
                )

            candidate_entails_incoming: bool | None = None
            incoming_entails_candidate: bool | None = None
            entailment_reason = "entailment not called because prefilter rejected"
            accepted = False
            if prefilter_passed:
                try:
                    decision = await _judge_bidirectional_entailment(
                        graphiti.llm_client,
                        candidate_fact=candidate_fact,
                        incoming_fact=text,
                        group_id=group_id,
                    )
                    candidate_entails_incoming = (
                        decision.candidate_entails_incoming
                    )
                    incoming_entails_candidate = (
                        decision.incoming_entails_candidate
                    )
                    entailment_reason = decision.reason
                    if (
                        candidate_entails_incoming
                        and incoming_entails_candidate
                    ):
                        semantic_match_uuid = await _append_claim_support(
                            driver,
                            match=candidate,
                            episode_uuid=episode.uuid,
                            source_url=payload.source.url or "",
                            source_title=payload.source.title,
                            source_identity=source_identity,
                            quote=text,
                        )
                        accepted = semantic_match_uuid is not None
                except Exception as exc:
                    entailment_reason = (
                        "entailment judge failed; candidate rejected: "
                        f"{type(exc).__name__}: {exc}"
                    )

            claim_match_audit.append(
                ClaimMatchAuditRecord(
                    incoming_fact=text,
                    candidate_edge_uuid=str(candidate["uuid"]),
                    candidate_fact=candidate_fact,
                    match_method="semantic",
                    similarity=float(candidate["similarity"]),
                    prefilter_passed=prefilter_passed,
                    prefilter_reason=prefilter_reason,
                    candidate_entails_incoming=candidate_entails_incoming,
                    incoming_entails_candidate=incoming_entails_candidate,
                    entailment_reason=entailment_reason,
                    accepted=accepted,
                )
            )
            if semantic_match_uuid is not None:
                supported_edge_uuids.append(semantic_match_uuid)
                break

        if semantic_match_uuid is not None:
            continue
        if payload.support_only:
            skipped.append(text)
            continue
        if not source_uuid or not target_uuid:
            skipped.append(text)
            continue
        claim_key = (source_uuid, relation_name, target_uuid)
        if claim_key in pending_claim_keys:
            skipped.append(text)
            continue
        pending_claim_keys.add(claim_key)
        new_candidates.append(
            (
                triple,
                text,
                source_uuid,
                target_uuid,
                relation_name,
                query_embedding,
            )
        )

    for (
        triple,
        text,
        source_uuid,
        target_uuid,
        relation_name,
        vector,
    ) in new_candidates:
        edge = EntityEdge(
            source_node_uuid=source_uuid,
            target_node_uuid=target_uuid,
            name=relation_name,
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
    supported_edge_uuids = list(dict.fromkeys(supported_edge_uuids))
    episode.entity_edges = list(
        dict.fromkeys(
            [
                *[edge.uuid for edge in edges],
                *supported_edge_uuids,
            ]
        )
    )
    await episode.save(driver)

    created_edge_uuids = [edge.uuid for edge in edges]
    return GraphWriteResult(
        episode_uuid=episode.uuid,
        node_uuids=list(dict.fromkeys(uuid_by_key.values())),
        edge_uuids=[*created_edge_uuids, *supported_edge_uuids],
        created_edge_uuids=created_edge_uuids,
        supported_edge_uuids=supported_edge_uuids,
        skipped_triples=skipped,
        claim_match_audit=claim_match_audit,
    )
