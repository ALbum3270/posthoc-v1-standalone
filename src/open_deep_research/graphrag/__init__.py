"""Foundational GraphRAG types and ontology helpers."""

from open_deep_research.graphrag.ontology import (
    INVESTIGATION_SCHEMA,
    OntologySlot,
    compute_coverage_ratio,
    extend_ontology,
    get_open_slots,
    iter_slots,
)
from open_deep_research.graphrag.graph.client import GraphitiClient, GraphitiClientConfig
from open_deep_research.graphrag.extraction.targeted_extractor import (
    TargetedExtractionConfig,
    TargetedExtractor,
)
from open_deep_research.graphrag.schemas import (
    ClaimVerificationResult,
    EvidencePack,
    ExtractedClaim,
    ExtractedTriple,
    GapStatus,
    GraphEpisodePayload,
    GraphWriteResult,
    SourceDocument,
)

__all__ = [
    "INVESTIGATION_SCHEMA",
    "OntologySlot",
    "compute_coverage_ratio",
    "extend_ontology",
    "get_open_slots",
    "iter_slots",
    "GraphitiClient",
    "GraphitiClientConfig",
    "TargetedExtractionConfig",
    "TargetedExtractor",
    "ClaimVerificationResult",
    "EvidencePack",
    "ExtractedClaim",
    "ExtractedTriple",
    "GapStatus",
    "GraphEpisodePayload",
    "GraphWriteResult",
    "SourceDocument",
]
