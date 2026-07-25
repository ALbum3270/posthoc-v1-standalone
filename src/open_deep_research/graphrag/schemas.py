"""Shared Pydantic schemas for the GraphRAG integration layer."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceType(str, Enum):
    """Normalized source categories used across retrieval and graph writes."""

    WEB = "web"
    NEWS = "news"
    REPORT = "report"
    PDF = "pdf"
    FORUM = "forum"
    OFFICIAL = "official"
    INTERNAL = "internal"
    OTHER = "other"


class EvidenceDirection(str, Enum):
    """How a piece of evidence relates to a claim."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class VerificationStatus(str, Enum):
    """Verification decision for an extracted claim."""

    PASSED = "passed"
    REVIEW = "review"
    REJECTED = "rejected"


class RelevanceStatus(str, Enum):
    """Whether a grounded fact directly answers its assigned ontology slot."""

    UNCHECKED = "unchecked"
    ACCEPTED = "accepted"
    UNCERTAIN = "uncertain"
    REJECTED = "rejected"


class SlotApplicabilityStatus(str, Enum):
    """Per-investigation applicability of one ontology slot."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    NOT_APPLICABLE = "not_applicable"


class SourceSpan(BaseModel):
    """Character offsets mapping extracted facts back to source content."""

    model_config = ConfigDict(extra="forbid")

    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    quote: str | None = None


class SourceDocument(BaseModel):
    """Canonical representation of a retrieved document or chunk."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str
    url: str | None = None
    source_type: SourceType = SourceType.WEB
    content: str
    snippet: str | None = None
    published_at: datetime | None = None
    retrieved_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntityRef(BaseModel):
    """Lightweight entity reference used during extraction before graph writes."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str | None = None
    name: str
    entity_type: str = "Entity"
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class ExtractedTriple(BaseModel):
    """Targeted extraction output aligned to a pending ontology slot."""

    model_config = ConfigDict(extra="forbid")

    slot_id: str
    subject: EntityRef
    predicate: str
    object: EntityRef | str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    source_document_id: str
    source_span: SourceSpan | None = None
    rationale: str | None = None
    relevance_status: RelevanceStatus = RelevanceStatus.UNCHECKED
    relevance_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    relevance_reason: str | None = None


class ExtractedClaim(BaseModel):
    """Atomic claim passed into verification and graph writing."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    slot_id: str
    text: str
    triples: list[ExtractedTriple] = Field(default_factory=list)
    source_document_id: str
    source_span: SourceSpan | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationEvidence(BaseModel):
    """Evidence item emitted by the verification layer."""

    model_config = ConfigDict(extra="forbid")

    title: str
    url: str | None = None
    source_type: SourceType = SourceType.OTHER
    direction: EvidenceDirection
    snippet: str
    credibility_score: float = Field(ge=0.0, le=1.0, default=0.5)
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.5)
    published_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimVerificationResult(BaseModel):
    """Normalized verification verdict used to gate graph writes."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    status: VerificationStatus
    confidence_score: float = Field(ge=0.0, le=1.0)
    truth_score: float = Field(ge=0.0, le=1.0)
    bias_flags: list[str] = Field(default_factory=list)
    evidence: list[VerificationEvidence] = Field(default_factory=list)
    reviewer_notes: str | None = None

    @property
    def allows_graph_write(self) -> bool:
        """Whether the claim is strong enough to be persisted to Graphiti."""

        return self.status in {VerificationStatus.PASSED, VerificationStatus.REVIEW}


class ConflictRecord(BaseModel):
    """Represents conflicting facts retained in the graph."""

    model_config = ConfigDict(extra="forbid")

    conflict_id: str
    slot_id: str
    active_episode_ids: list[str] = Field(default_factory=list)
    summary: str
    leading_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GapStatus(BaseModel):
    """Coverage snapshot for one ontology slot."""

    model_config = ConfigDict(extra="forbid")

    slot_id: str
    dimension: str
    question: str
    filled: bool = False
    applicability: SlotApplicabilityStatus = SlotApplicabilityStatus.REQUIRED
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    supporting_episode_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class SlotApplicability(BaseModel):
    """Auditable per-topic decision about whether a slot should be researched."""

    model_config = ConfigDict(extra="forbid")

    slot_id: str
    status: SlotApplicabilityStatus
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    reason: str = ""


class GraphEpisodePayload(BaseModel):
    """Payload written to Graphiti after extraction and verification."""

    model_config = ConfigDict(extra="forbid")

    name: str
    episode_body: str
    source_description: str
    source: str = "text"
    reference_time: datetime | None = None
    group_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    claims: list[ExtractedClaim] = Field(default_factory=list)


class VerifiedEpisodeInput(BaseModel):
    """Structured payload for the verified write path.

    Distinct from ``GraphEpisodePayload``: that one carries prose for Graphiti to
    extract from, which is the double-extraction route that rewrote facts and
    invented dates (SESSION_HANDOFF §3.12). This one carries already-extracted
    triples and is written verbatim -- no LLM sees it again.
    """

    model_config = ConfigDict(extra="forbid")

    research_id: str
    slot_id: str
    source: SourceDocument
    triples: list[ExtractedTriple] = Field(default_factory=list)
    group_id: str | None = None
    episode_name: str | None = None
    verification: ClaimVerificationResult | None = None
    support_only: bool = False


class ClaimMatchAuditRecord(BaseModel):
    """Auditable decision for one existing claim considered as corroboration."""

    model_config = ConfigDict(extra="forbid")

    incoming_fact: str
    candidate_edge_uuid: str
    candidate_fact: str
    match_method: str
    similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    prefilter_passed: bool
    prefilter_reason: str
    candidate_entails_incoming: bool | None = None
    incoming_entails_candidate: bool | None = None
    entailment_reason: str = ""
    accepted: bool = False


class GraphWriteResult(BaseModel):
    """Minimal bookkeeping returned after a graph write."""

    model_config = ConfigDict(extra="forbid")

    episode_uuid: str
    node_uuids: list[str] = Field(default_factory=list)
    edge_uuids: list[str] = Field(default_factory=list)
    created_edge_uuids: list[str] = Field(default_factory=list)
    supported_edge_uuids: list[str] = Field(default_factory=list)
    conflict_ids: list[str] = Field(default_factory=list)
    skipped_triples: list[str] = Field(default_factory=list)
    claim_match_audit: list[ClaimMatchAuditRecord] = Field(default_factory=list)


class EvidencePackItem(BaseModel):
    """One report-ready conclusion with provenance."""

    model_config = ConfigDict(extra="forbid")

    slot_id: str
    conclusion: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    provenance_episode_ids: list[str] = Field(default_factory=list)
    source_urls: list[str] = Field(default_factory=list)
    source_count: int = Field(default=0, ge=0)
    required_source_count: int = Field(default=1, ge=1)
    claim_corroborated: bool = False
    support_requirement_met: bool = False
    relevance_status: RelevanceStatus = RelevanceStatus.UNCHECKED
    relevance_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    relevance_reason: str | None = None
    caveats: list[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    """Structured report context assembled from Graphiti."""

    model_config = ConfigDict(extra="forbid")

    topic: str
    coverage_ratio: float = Field(ge=0.0, le=1.0, default=0.0)
    items: list[EvidencePackItem] = Field(default_factory=list)
    unresolved_conflicts: list[ConflictRecord] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)
    slot_applicability: list[SlotApplicability] = Field(default_factory=list)
