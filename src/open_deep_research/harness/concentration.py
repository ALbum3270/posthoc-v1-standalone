"""Deterministic audits of report dependence on publisher-domain proxies."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from open_deep_research.harness.claims import MarkdownBlock
from open_deep_research.harness.notes import ResearchNote, source_id_for_url
from open_deep_research.harness.reconcile import (
    ChecklistReportReconciliation,
)
from open_deep_research.harness.verify import (
    VerificationResult,
    VerifiedSourceRelation,
)

_METHOD = "publisher_domain_proxy"
_COUNTING_UNIT = "formal_claim_source_support_relation"
_LIMITATIONS = (
    "host domains are publisher proxies, not organization identities",
    "different domains may reproduce material from one originating organization",
    "one host domain may carry material from multiple originating organizations",
    "publisher-domain distribution does not establish viewpoint diversity",
)


class PublisherDomainProxyCount(BaseModel):
    """One publisher-domain proxy's share of formal support relations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    publisher_domain_proxy: str
    formal_support_relation_count: int = Field(ge=1)


class DomainProxyDistribution(BaseModel):
    """Unnormalized concentration facts for one report unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    formal_support_relation_count: int = Field(ge=0)
    publisher_domain_proxy_count: int = Field(ge=0)
    publisher_domain_proxy_distribution: tuple[
        PublisherDomainProxyCount, ...
    ] = ()
    largest_publisher_domain_proxy: str | None = None
    largest_publisher_domain_proxy_relation_count: int = Field(ge=0)
    largest_publisher_domain_proxy_share: float = Field(
        ge=0.0,
        le=1.0,
    )
    raw_hhi: float = Field(ge=0.0, le=1.0)
    effective_publisher_domain_proxy_count: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _counts_are_consistent(self) -> DomainProxyDistribution:
        rows = self.publisher_domain_proxy_distribution
        if self.publisher_domain_proxy_count != len(rows):
            raise ValueError(
                "publisher proxy count must match the distribution"
            )
        if self.formal_support_relation_count != sum(
            row.formal_support_relation_count for row in rows
        ):
            raise ValueError(
                "formal relation count must match the distribution"
            )
        if not rows:
            if (
                self.largest_publisher_domain_proxy is not None
                or self.largest_publisher_domain_proxy_relation_count
                or self.largest_publisher_domain_proxy_share
                or self.raw_hhi
                or self.effective_publisher_domain_proxy_count
            ):
                raise ValueError(
                    "an empty distribution cannot have concentration values"
                )
            return self
        largest = rows[0]
        if (
            self.largest_publisher_domain_proxy
            != largest.publisher_domain_proxy
            or self.largest_publisher_domain_proxy_relation_count
            != largest.formal_support_relation_count
        ):
            raise ValueError(
                "largest publisher proxy must be the first distribution row"
            )
        return self


class ReadSourceAudit(BaseModel):
    """One cached source that did not formally support a report unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    url: str
    publisher_domain_proxy: str
    total_note_count: int = Field(ge=0)
    notes_for_checklist_item: int | None = Field(default=None, ge=0)


class ElementOnlySupportUsageAudit(BaseModel):
    """Sources used only for formal truth-condition evidence.

    These relations are deliberately excluded from the whole-claim
    concentration denominator.  Recording them separately prevents a source
    that materially supports one registered condition from being mislabeled
    as read-but-unused.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    counting_unit: Literal[
        "formal_element_only_claim_source_relation"
    ] = "formal_element_only_claim_source_relation"
    claim_source_relation_count: int = Field(ge=1)
    source_ids: tuple[str, ...] = Field(min_length=1)
    publisher_domain_proxies: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _sets_are_canonical(self) -> ElementOnlySupportUsageAudit:
        if self.source_ids != tuple(sorted(set(self.source_ids))):
            raise ValueError("element-only source IDs must be sorted and unique")
        if self.publisher_domain_proxies != tuple(
            sorted(set(self.publisher_domain_proxies))
        ):
            raise ValueError(
                "element-only publisher proxies must be sorted and unique"
            )
        return self


class UnitDomainProxyConcentration(BaseModel):
    """Concentration and unused-source facts for one report unit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_type: Literal["section", "checklist_item"]
    unit_id: str
    label: str
    section_path: tuple[str, ...] = ()
    checklist_item_id: str | None = None
    block_ids: tuple[str, ...] = ()
    claim_ids: tuple[str, ...] = ()
    distribution: DomainProxyDistribution
    is_single_publisher_domain_proxy_monopoly: bool
    monopoly_publisher_domain_proxy: str | None = None
    element_only_support: ElementOnlySupportUsageAudit | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    read_but_unused_sources: tuple[ReadSourceAudit, ...] = ()

    @model_validator(mode="after")
    def _monopoly_matches_distribution(
        self,
    ) -> UnitDomainProxyConcentration:
        expected = (
            self.distribution.formal_support_relation_count > 0
            and self.distribution.publisher_domain_proxy_count == 1
        )
        if self.is_single_publisher_domain_proxy_monopoly != expected:
            raise ValueError(
                "single-publisher monopoly must be derived from distribution"
            )
        expected_proxy = (
            self.distribution.largest_publisher_domain_proxy
            if expected
            else None
        )
        if self.monopoly_publisher_domain_proxy != expected_proxy:
            raise ValueError(
                "monopoly publisher proxy must match the distribution"
            )
        return self


class DomainProxyConcentrationAudit(BaseModel):
    """Whole-report and per-unit source-concentration audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["publisher_domain_proxy"] = _METHOD
    counting_unit: Literal[
        "formal_claim_source_support_relation"
    ] = _COUNTING_UNIT
    is_organization_independence_determination: Literal[False] = False
    is_viewpoint_diversity_determination: Literal[False] = False
    limitations: tuple[str, ...] = _LIMITATIONS
    overall: DomainProxyDistribution
    element_only_support: ElementOnlySupportUsageAudit | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    sections: tuple[UnitDomainProxyConcentration, ...] = ()
    checklist_items: tuple[UnitDomainProxyConcentration, ...] = ()
    single_publisher_monopoly_section_ids: tuple[str, ...] = ()
    single_publisher_monopoly_checklist_item_ids: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


def _distribution(
    relations: Sequence[VerifiedSourceRelation],
) -> DomainProxyDistribution:
    counts = Counter(
        relation.publisher_domain_proxy for relation in relations
    )
    rows = tuple(
        PublisherDomainProxyCount(
            publisher_domain_proxy=proxy,
            formal_support_relation_count=count,
        )
        for proxy, count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    total = sum(counts.values())
    if not total:
        return DomainProxyDistribution(
            formal_support_relation_count=0,
            publisher_domain_proxy_count=0,
            largest_publisher_domain_proxy_relation_count=0,
            largest_publisher_domain_proxy_share=0.0,
            raw_hhi=0.0,
            effective_publisher_domain_proxy_count=0.0,
        )
    shares = tuple(count / total for count in counts.values())
    raw_hhi = sum(share * share for share in shares)
    largest = rows[0]
    return DomainProxyDistribution(
        formal_support_relation_count=total,
        publisher_domain_proxy_count=len(rows),
        publisher_domain_proxy_distribution=rows,
        largest_publisher_domain_proxy=largest.publisher_domain_proxy,
        largest_publisher_domain_proxy_relation_count=(
            largest.formal_support_relation_count
        ),
        largest_publisher_domain_proxy_share=(
            largest.formal_support_relation_count / total
        ),
        raw_hhi=raw_hhi,
        effective_publisher_domain_proxy_count=1.0 / raw_hhi,
    )


def _formal_relations(
    verification: VerificationResult,
) -> tuple[
    tuple[VerifiedSourceRelation, ...],
    Mapping[str, tuple[VerifiedSourceRelation, ...]],
    tuple[str, ...],
]:
    unique: dict[tuple[str, str], VerifiedSourceRelation] = {}
    diagnostics: list[str] = []
    for claim in verification.claims:
        claim_id = claim.claim.claim_id
        for relation in claim.relations:
            if not relation.is_formal_supporting_evidence:
                continue
            if relation.claim_id != claim_id:
                diagnostics.append(
                    f"relation_claim_id_mismatch:{relation.claim_id}:"
                    f"parent={claim_id}"
                )
            key = (claim_id, relation.source_id)
            existing = unique.get(key)
            if existing is None:
                unique[key] = relation
                continue
            diagnostics.append(
                f"duplicate_formal_claim_source_relation:{claim_id}:"
                f"{relation.source_id}"
            )
            if (
                existing.publisher_domain_proxy
                != relation.publisher_domain_proxy
                or existing.url != relation.url
            ):
                diagnostics.append(
                    f"conflicting_duplicate_formal_relation:{claim_id}:"
                    f"{relation.source_id}"
                )

    ordered = tuple(
        relation
        for _, relation in sorted(
            unique.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                item[1].publisher_domain_proxy,
            ),
        )
    )
    by_claim: dict[str, list[VerifiedSourceRelation]] = defaultdict(list)
    for (claim_id, _), relation in unique.items():
        by_claim[claim_id].append(relation)
    return (
        ordered,
        {
            claim_id: tuple(
                sorted(
                    relations,
                    key=lambda relation: (
                        relation.publisher_domain_proxy,
                        relation.source_id,
                    ),
                )
            )
            for claim_id, relations in by_claim.items()
        },
        tuple(diagnostics),
    )


def _element_only_relations(
    verification: VerificationResult,
) -> tuple[
    tuple[VerifiedSourceRelation, ...],
    Mapping[str, tuple[VerifiedSourceRelation, ...]],
    tuple[str, ...],
]:
    """Return claim-source relations used only at truth-condition scope."""

    unique: dict[tuple[str, str], VerifiedSourceRelation] = {}
    diagnostics: list[str] = []
    for claim in verification.claims:
        claim_id = claim.claim.claim_id
        for relation in claim.relations:
            if relation.is_formal_supporting_evidence or not any(
                element.is_formal_supporting_evidence
                for element in relation.element_relations
            ):
                continue
            key = (claim_id, relation.source_id)
            if key in unique:
                diagnostics.append(
                    "duplicate_element_only_claim_source_relation:"
                    f"{claim_id}:{relation.source_id}"
                )
                continue
            unique[key] = relation

    ordered = tuple(
        relation
        for _, relation in sorted(
            unique.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                item[1].publisher_domain_proxy,
            ),
        )
    )
    by_claim: dict[str, list[VerifiedSourceRelation]] = defaultdict(list)
    for (claim_id, _), relation in unique.items():
        by_claim[claim_id].append(relation)
    return (
        ordered,
        {
            claim_id: tuple(
                sorted(
                    relations,
                    key=lambda relation: (
                        relation.publisher_domain_proxy,
                        relation.source_id,
                    ),
                )
            )
            for claim_id, relations in by_claim.items()
        },
        tuple(diagnostics),
    )


def _element_only_usage(
    relations: Sequence[VerifiedSourceRelation],
) -> ElementOnlySupportUsageAudit | None:
    if not relations:
        return None
    return ElementOnlySupportUsageAudit(
        claim_source_relation_count=len(relations),
        source_ids=tuple(sorted({relation.source_id for relation in relations})),
        publisher_domain_proxies=tuple(
            sorted(
                {
                    relation.publisher_domain_proxy
                    for relation in relations
                }
            )
        ),
    )


def _publisher_domain(url: str) -> str:
    host = (urlparse(url).hostname or "").strip(".").casefold()
    if host.startswith("www."):
        host = host[4:]
    return host


def _read_sources(
    source_cache: Mapping[str, str],
    notes: Sequence[ResearchNote],
    *,
    used_source_ids: set[str],
    checklist_item_id: str | None,
) -> tuple[ReadSourceAudit, ...]:
    total_note_counts = Counter(note.source_id for note in notes)
    item_note_counts = Counter(
        note.source_id
        for note in notes
        if checklist_item_id is not None
        and note.item_id == checklist_item_id
    )
    unused: list[ReadSourceAudit] = []
    for url in sorted(source_cache):
        source_id = source_id_for_url(url)
        if source_id in used_source_ids:
            continue
        unused.append(
            ReadSourceAudit(
                source_id=source_id,
                url=url,
                publisher_domain_proxy=_publisher_domain(url),
                total_note_count=total_note_counts[source_id],
                notes_for_checklist_item=(
                    item_note_counts[source_id]
                    if checklist_item_id is not None
                    else None
                ),
            )
        )
    return tuple(unused)


def _unit(
    *,
    unit_type: Literal["section", "checklist_item"],
    unit_id: str,
    label: str,
    section_path: tuple[str, ...] = (),
    checklist_item_id: str | None = None,
    block_ids: Sequence[str],
    claim_ids: Sequence[str],
    relations_by_claim: Mapping[
        str,
        tuple[VerifiedSourceRelation, ...],
    ],
    element_only_relations_by_claim: Mapping[
        str,
        tuple[VerifiedSourceRelation, ...],
    ],
    source_cache: Mapping[str, str],
    notes: Sequence[ResearchNote],
) -> UnitDomainProxyConcentration:
    ordered_claim_ids = tuple(dict.fromkeys(claim_ids))
    relation_by_key: dict[tuple[str, str], VerifiedSourceRelation] = {}
    for claim_id in ordered_claim_ids:
        for relation in relations_by_claim.get(claim_id, ()):
            relation_by_key[(claim_id, relation.source_id)] = relation
    relations = tuple(
        relation
        for _, relation in sorted(relation_by_key.items())
    )
    element_relation_by_key: dict[
        tuple[str, str], VerifiedSourceRelation
    ] = {}
    for claim_id in ordered_claim_ids:
        for relation in element_only_relations_by_claim.get(claim_id, ()):
            element_relation_by_key[(claim_id, relation.source_id)] = relation
    element_only_relations = tuple(
        relation
        for _, relation in sorted(element_relation_by_key.items())
    )
    distribution = _distribution(relations)
    # The cache side is keyed by URL and derives its source identity with
    # source_id_for_url().  Relation source_id remains an auditable provider
    # identity, but the schema intentionally permits historical/external IDs
    # that are not URL-derived.  Compare like with like so a formally used
    # source cannot be mislabeled as read-but-unused.
    used_source_ids = {
        source_id_for_url(relation.url)
        for relation in (*relations, *element_only_relations)
    }
    monopoly = (
        distribution.formal_support_relation_count > 0
        and distribution.publisher_domain_proxy_count == 1
    )
    return UnitDomainProxyConcentration(
        unit_type=unit_type,
        unit_id=unit_id,
        label=label,
        section_path=section_path,
        checklist_item_id=checklist_item_id,
        block_ids=tuple(dict.fromkeys(block_ids)),
        claim_ids=ordered_claim_ids,
        distribution=distribution,
        is_single_publisher_domain_proxy_monopoly=monopoly,
        monopoly_publisher_domain_proxy=(
            distribution.largest_publisher_domain_proxy
            if monopoly
            else None
        ),
        element_only_support=_element_only_usage(element_only_relations),
        read_but_unused_sources=_read_sources(
            source_cache,
            notes,
            used_source_ids=used_source_ids,
            checklist_item_id=checklist_item_id,
        ),
    )


def audit_domain_proxy_concentration(
    verification: VerificationResult,
    *,
    blocks: Sequence[MarkdownBlock],
    reconciliation: ChecklistReportReconciliation,
    source_cache: Mapping[str, str],
    notes: Sequence[ResearchNote],
) -> DomainProxyConcentrationAudit:
    """Measure source concentration without assigning a success threshold."""

    formal, relations_by_claim, diagnostics = _formal_relations(verification)
    (
        element_only,
        element_only_relations_by_claim,
        element_diagnostics,
    ) = _element_only_relations(verification)
    diagnostics += element_diagnostics
    claim_by_id = {
        claim.claim.claim_id: claim.claim for claim in verification.claims
    }
    blocks_by_id = {block.block_id: block for block in blocks}

    ordered_blocks = sorted(blocks, key=lambda block: block.ordinal)
    section_runs: list[list[MarkdownBlock]] = []
    for block in ordered_blocks:
        if (
            not section_runs
            or section_runs[-1][-1].section_path != block.section_path
        ):
            section_runs.append([block])
        else:
            section_runs[-1].append(block)

    sections: list[UnitDomainProxyConcentration] = []
    for index, section_blocks in enumerate(section_runs, start=1):
        block_ids = tuple(block.block_id for block in section_blocks)
        block_id_set = set(block_ids)
        claim_ids = tuple(
            claim_id
            for claim_id, claim in sorted(claim_by_id.items())
            if claim.block_id in block_id_set
        )
        section_path = section_blocks[0].section_path
        sections.append(
            _unit(
                unit_type="section",
                unit_id=f"section-{index:04d}",
                label=" / ".join(section_path) if section_path else "(root)",
                section_path=section_path,
                block_ids=block_ids,
                claim_ids=claim_ids,
                relations_by_claim=relations_by_claim,
                element_only_relations_by_claim=(
                    element_only_relations_by_claim
                ),
                source_cache=source_cache,
                notes=notes,
            )
        )

    checklist_items: list[UnitDomainProxyConcentration] = []
    for record in reconciliation.records:
        claim_ids = tuple(
            reference.claim_id for reference in record.references
        )
        block_ids = tuple(
            reference.block_id for reference in record.references
        )
        for block_id in block_ids:
            if block_id not in blocks_by_id:
                diagnostics += (
                    f"reconciliation_block_not_found:{record.item_id}:"
                    f"{block_id}",
                )
        checklist_items.append(
            _unit(
                unit_type="checklist_item",
                unit_id=f"checklist-item:{record.item_id}",
                label=record.question,
                checklist_item_id=record.item_id,
                block_ids=block_ids,
                claim_ids=claim_ids,
                relations_by_claim=relations_by_claim,
                element_only_relations_by_claim=(
                    element_only_relations_by_claim
                ),
                source_cache=source_cache,
                notes=notes,
            )
        )

    return DomainProxyConcentrationAudit(
        overall=_distribution(formal),
        element_only_support=_element_only_usage(element_only),
        sections=tuple(sections),
        checklist_items=tuple(checklist_items),
        single_publisher_monopoly_section_ids=tuple(
            section.unit_id
            for section in sections
            if section.is_single_publisher_domain_proxy_monopoly
        ),
        single_publisher_monopoly_checklist_item_ids=tuple(
            item.checklist_item_id
            for item in checklist_items
            if item.is_single_publisher_domain_proxy_monopoly
            and item.checklist_item_id is not None
        ),
        diagnostics=diagnostics,
    )
