"""Thin Graphiti wrapper used by the GraphRAG control plane."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict

from open_deep_research.graphrag.schemas import GraphEpisodePayload, GraphWriteResult


class GraphitiClientConfig(BaseModel):
    """Connection and runtime defaults for Graphiti."""

    model_config = ConfigDict(extra="forbid")

    uri: str | None = None
    user: str | None = None
    password: str | None = None
    group_id: str | None = None
    store_raw_episode_content: bool = True
    update_communities: bool = False
    default_search_results: int = 10


@dataclass
class _GraphitiImports:
    graphiti_cls: Any
    episode_type: Any
    raw_episode_cls: Any
    search_config: Any


def _load_graphiti_imports() -> _GraphitiImports:
    """Import Graphiti lazily so unit tests can run without the dependency."""

    try:
        from graphiti_core.graphiti import Graphiti
        from graphiti_core.nodes import EpisodeType
        from graphiti_core.search.search_config_recipes import (
            COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
        )
        from graphiti_core.utils.bulk_utils import RawEpisode
    except ImportError as exc:
        raise ImportError(
            "Graphiti is not installed or not importable. Ensure graphiti-main is "
            "available as a dependency before using GraphitiClient."
        ) from exc

    return _GraphitiImports(
        graphiti_cls=Graphiti,
        episode_type=EpisodeType,
        raw_episode_cls=RawEpisode,
        search_config=COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
    )


class GraphitiClient:
    """Application-facing Graphiti wrapper with stable payload contracts."""

    def __init__(
        self,
        config: GraphitiClientConfig,
        graphiti: Any | None = None,
    ) -> None:
        self.config = config
        self._graphiti = graphiti
        self._imports: _GraphitiImports | None = None

    async def connect(self) -> Any:
        """Create or return the underlying Graphiti client."""

        if self._graphiti is not None:
            return self._graphiti

        self._imports = _load_graphiti_imports()
        self._graphiti = self._imports.graphiti_cls(
            uri=self.config.uri,
            user=self.config.user,
            password=self.config.password,
            store_raw_episode_content=self.config.store_raw_episode_content,
        )
        return self._graphiti

    async def close(self) -> None:
        """Close the underlying Graphiti client if it exposes close()."""

        if self._graphiti is None:
            return

        close = getattr(self._graphiti, "close", None)
        if close is not None:
            await close()

    async def build_indices_and_constraints(self, delete_existing: bool = False) -> None:
        """Initialize graph indices and constraints."""

        graphiti = await self.connect()
        await graphiti.build_indices_and_constraints(delete_existing=delete_existing)

    async def add_episode(self, payload: GraphEpisodePayload) -> GraphWriteResult:
        """Write one validated episode to Graphiti."""

        graphiti = await self.connect()
        imports = self._imports or _load_graphiti_imports()
        reference_time = payload.reference_time or datetime.now(timezone.utc)
        episode_source = self._map_episode_type(payload.source, imports.episode_type)
        content = self._render_episode_content(payload)
        source_description = self._render_source_description(payload)
        group_id = payload.group_id or self.config.group_id

        result = await graphiti.add_episode(
            name=payload.name,
            episode_body=content,
            source_description=source_description,
            reference_time=reference_time,
            source=episode_source,
            group_id=group_id,
            update_communities=self.config.update_communities,
        )
        return self._to_graph_write_result(result, payload)

    async def add_episode_bulk(
        self,
        payloads: list[GraphEpisodePayload],
        *,
        group_id: str | None = None,
    ) -> list[GraphWriteResult]:
        """Write multiple episodes in a single Graphiti batch."""

        if not payloads:
            return []

        graphiti = await self.connect()
        imports = self._imports or _load_graphiti_imports()
        effective_group_id = group_id or self.config.group_id or payloads[0].group_id

        bulk_episodes = [
            imports.raw_episode_cls(
                name=payload.name,
                content=self._render_episode_content(payload),
                source_description=self._render_source_description(payload),
                source=self._map_episode_type(payload.source, imports.episode_type),
                reference_time=payload.reference_time or datetime.now(timezone.utc),
                uuid=None,
            )
            for payload in payloads
        ]

        result = await graphiti.add_episode_bulk(
            bulk_episodes=bulk_episodes,
            group_id=effective_group_id,
        )

        episode_ids = [episode.uuid for episode in result.episodes]
        episode_to_nodes: dict[str, list[str]] = {episode_id: [] for episode_id in episode_ids}
        episode_to_edges: dict[str, list[str]] = {episode_id: [] for episode_id in episode_ids}

        # Nodes must be attributed through episodic_edges, not through a node
        # `episodes` field: EntityNode has no such field, so reading one returned
        # an empty list for every node and node_uuids was silently always empty.
        # Each EpisodicEdge points episode (source) -> entity (target).
        for episodic_edge in getattr(result, "episodic_edges", []) or []:
            episode_id = getattr(episodic_edge, "source_node_uuid", None)
            node_uuid = getattr(episodic_edge, "target_node_uuid", None)
            if episode_id in episode_to_nodes and node_uuid:
                episode_to_nodes[episode_id].append(node_uuid)

        # Edges are different: EntityEdge really does carry `episodes`.
        for edge in result.edges:
            episode_ids_for_edge = self._extract_episode_uuids(edge)
            for episode_id in episode_ids_for_edge:
                if episode_id in episode_to_edges:
                    episode_to_edges[episode_id].append(edge.uuid)

        return [
            GraphWriteResult(
                episode_uuid=episode_id,
                node_uuids=episode_to_nodes.get(episode_id, []),
                edge_uuids=episode_to_edges.get(episode_id, []),
                conflict_ids=self._extract_conflict_ids(payloads[index]),
            )
            for index, episode_id in enumerate(episode_ids)
        ]

    async def search(
        self,
        query: str,
        *,
        group_ids: list[str] | None = None,
        num_results: int | None = None,
        center_node_uuid: str | None = None,
    ) -> Any:
        """Run Graphiti's basic edge search."""

        graphiti = await self.connect()
        return await graphiti.search(
            query=query,
            group_ids=group_ids or self._default_group_ids(),
            num_results=num_results or self.config.default_search_results,
            center_node_uuid=center_node_uuid,
        )

    async def search_subgraph(
        self,
        query: str,
        *,
        group_ids: list[str] | None = None,
        center_node_uuid: str | None = None,
        bfs_origin_node_uuids: list[str] | None = None,
    ) -> Any:
        """Run Graphiti's advanced search and return nodes, edges, and episodes."""

        graphiti = await self.connect()
        imports = self._imports or _load_graphiti_imports()
        return await graphiti.search_(
            query=query,
            config=imports.search_config,
            group_ids=group_ids or self._default_group_ids(),
            center_node_uuid=center_node_uuid,
            bfs_origin_node_uuids=bfs_origin_node_uuids,
        )

    async def get_nodes_and_edges_by_episode(self, episode_uuids: list[str]) -> Any:
        """Hydrate the local subgraph around a set of episode UUIDs."""

        graphiti = await self.connect()
        return await graphiti.get_nodes_and_edges_by_episode(episode_uuids)

    def _default_group_ids(self) -> list[str] | None:
        """Return wrapper-level default group ids when configured."""

        if self.config.group_id is None:
            return None
        return [self.config.group_id]

    @staticmethod
    def _map_episode_type(source: str, episode_type_enum: Any) -> Any:
        """Normalize app-level source strings to Graphiti EpisodeType."""

        source_value = (source or "text").lower()
        if source_value == "message":
            return episode_type_enum.message
        if source_value == "json":
            return episode_type_enum.json
        return episode_type_enum.text

    @staticmethod
    def _extract_conflict_ids(payload: GraphEpisodePayload) -> list[str]:
        """Read conflict ids from the payload metadata if present."""

        raw = payload.metadata.get("conflicts_with", [])
        if isinstance(raw, list):
            return [str(item) for item in raw]
        return []

    @staticmethod
    def _extract_episode_uuids(item: Any) -> list[str]:
        """Best-effort extraction of originating episode ids from Graphiti objects."""

        episode_ids = getattr(item, "episodes", None)
        if episode_ids is None:
            return []
        return [str(episode_id) for episode_id in episode_ids]

    def _to_graph_write_result(
        self,
        graphiti_result: Any,
        payload: GraphEpisodePayload,
    ) -> GraphWriteResult:
        """Map Graphiti add_episode output into the app-level write result."""

        node_uuids = [node.uuid for node in getattr(graphiti_result, "nodes", [])]
        edge_uuids = [edge.uuid for edge in getattr(graphiti_result, "edges", [])]
        episode_uuid = getattr(getattr(graphiti_result, "episode", None), "uuid", "")
        return GraphWriteResult(
            episode_uuid=episode_uuid,
            node_uuids=node_uuids,
            edge_uuids=edge_uuids,
            conflict_ids=self._extract_conflict_ids(payload),
        )

    @staticmethod
    def _render_source_description(payload: GraphEpisodePayload) -> str:
        """Preserve payload metadata even if Graphiti does not store metadata separately."""

        metadata_summary = json.dumps(payload.metadata, ensure_ascii=True, sort_keys=True)
        return f"{payload.source_description} | metadata={metadata_summary}"

    @staticmethod
    def _render_episode_content(payload: GraphEpisodePayload) -> str:
        """Embed metadata and claim summaries into Graphiti's episode content."""

        sections = [payload.episode_body.strip()]

        if payload.claims:
            claim_lines = [f"- {claim.claim_id}: {claim.text}" for claim in payload.claims]
            sections.append("Claims:\n" + "\n".join(claim_lines))

        if payload.metadata:
            sections.append(
                "Structured metadata:\n"
                + json.dumps(payload.metadata, ensure_ascii=True, sort_keys=True)
            )

        return "\n\n".join(section for section in sections if section)
