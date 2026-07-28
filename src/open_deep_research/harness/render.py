"""Deterministically render verified evidence into a canonical report draft."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from open_deep_research.harness.claims import (
    CitationRequirement,
    ClaimNormalizationStatus,
    ClaimRegistryCoverage,
)
from open_deep_research.harness.verify import (
    ClaimEvidenceState,
    ClaimVerification,
    VerificationRecordStatus,
    VerificationResult,
    VerificationVerdict,
    VerifiedSourceRelation,
)

_FOOTNOTE_DEFINITION = re.compile(
    r"^[ \t]*\[\^([A-Za-z0-9_-]+)\]:",
)
_FOOTNOTE_MARKER = re.compile(r"\[\^([A-Za-z0-9_-]+)\]")


class EvidenceRegistryKey(BaseModel):
    """The sole identity used for a generated footnote."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)

    @model_validator(mode="after")
    def _span_is_nonempty(self) -> EvidenceRegistryKey:
        if self.end_char <= self.start_char:
            raise ValueError("evidence registry spans must be nonempty")
        return self


class RenderedFootnote(BaseModel):
    """One globally unique code-owned footnote definition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    number: int = Field(ge=1)
    key: EvidenceRegistryKey
    source_quote: str
    url: str


class ClaimRenderAnnotation(BaseModel):
    """One visible claim annotation placed at a mechanically verified anchor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    anchor_end: int = Field(ge=0)
    evidence_state: ClaimEvidenceState
    footnote_numbers: tuple[int, ...] = ()
    rendered_suffix: str


class EvidenceSummary(BaseModel):
    """Compact reader-facing counts rendered above the model-authored body."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    external_claims: int = Field(ge=0)
    corroborated: int = Field(ge=0)
    source_shortfall: int = Field(ge=0)
    conflicting: int = Field(ge=0)
    refuted: int = Field(ge=0)
    inspected_not_supporting: int = Field(ge=0)
    no_candidate: int = Field(ge=0)
    verification_incomplete: int = Field(ge=0)
    verification_not_run: int = Field(ge=0)
    support_quote_unlocatable: int = Field(ge=0)
    claim_normalization_failed: int = Field(ge=0)
    attribution_error: int = Field(ge=0)
    unverified: int = Field(ge=0)
    settled_without_located_evidence: int = Field(ge=0)
    settled_without_located_evidence_item_ids: tuple[str, ...] = ()
    registry_coverage: ClaimRegistryCoverage | None = None

    @model_validator(mode="after")
    def _unverified_total_matches_components(self) -> EvidenceSummary:
        components = (
            self.verification_incomplete
            + self.verification_not_run
            + self.support_quote_unlocatable
            + self.claim_normalization_failed
            + self.attribution_error
        )
        if self.unverified != components:
            raise ValueError(
                "unverified must equal its reader-facing component counts"
            )
        return self


class RenderedReport(BaseModel):
    """Final Markdown plus the deterministic rendering audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    markdown: str
    evidence_summary_line: str
    summary: EvidenceSummary
    footnotes: tuple[RenderedFootnote, ...] = ()
    annotations: tuple[ClaimRenderAnnotation, ...] = ()
    removed_model_footnote_definitions: int = Field(ge=0)
    removed_model_footnote_markers: int = Field(ge=0)
    unanchored_claim_ids: tuple[str, ...] = ()


def _definition_spans(markdown: str) -> list[tuple[int, int]]:
    lines = markdown.splitlines(keepends=True)
    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)

    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        if _FOOTNOTE_DEFINITION.match(lines[index]) is None:
            index += 1
            continue
        start = starts[index]
        end = start + len(lines[index])
        index += 1
        while index < len(lines):
            line = lines[index]
            if line.startswith(("    ", "\t")):
                end = starts[index] + len(line)
                index += 1
                continue
            break
        spans.append((start, end))
    return spans


def _prohibited_footnote_spans(
    markdown: str,
) -> tuple[list[tuple[int, int]], int, int]:
    definitions = _definition_spans(markdown)
    markers: list[tuple[int, int]] = []
    for match in _FOOTNOTE_MARKER.finditer(markdown):
        if any(start <= match.start() < end for start, end in definitions):
            continue
        markers.append((match.start(), match.end()))
    return [*definitions, *markers], len(definitions), len(markers)


def _located_relation(relation: VerifiedSourceRelation) -> bool:
    return (
        relation.source_quote is not None
        and relation.span is not None
        and relation.status == VerificationRecordStatus.COMPLETED
    )


def _renderable_relations(
    verification: ClaimVerification,
) -> tuple[VerifiedSourceRelation, ...]:
    relations = [
        relation
        for relation in verification.relations
        if _located_relation(relation)
        and (
            relation.is_formal_supporting_evidence
            or relation.semantic_verdict == VerificationVerdict.CONTRADICTS
        )
    ]
    return tuple(
        sorted(
            relations,
            key=lambda relation: (
                0
                if relation.semantic_verdict == VerificationVerdict.SUPPORTS
                else 1,
                relation.source_id,
                relation.span.start_char if relation.span else -1,
                relation.span.end_char if relation.span else -1,
            ),
        )
    )


def _unverified_reasons(verification: ClaimVerification) -> tuple[str, ...]:
    status_labels = {
        VerificationRecordStatus.VERIFICATION_NOT_RUN_BUDGET: "预算耗尽",
        VerificationRecordStatus.VERIFICATION_MODEL_ERROR: "模型错误",
        VerificationRecordStatus.SOURCE_TOO_LARGE_FOR_ADMISSION: (
            "原文超出准入上限"
        ),
        VerificationRecordStatus.SOURCE_MISSING_FROM_CACHE: "缓存原文缺失",
    }
    ordered_statuses = (
        VerificationRecordStatus.VERIFICATION_NOT_RUN_BUDGET,
        VerificationRecordStatus.VERIFICATION_MODEL_ERROR,
        VerificationRecordStatus.SOURCE_TOO_LARGE_FOR_ADMISSION,
        VerificationRecordStatus.SOURCE_MISSING_FROM_CACHE,
    )
    present = {relation.status for relation in verification.relations}
    return tuple(
        status_labels[status]
        for status in ordered_statuses
        if status in present
    )


def _warning_label(verification: ClaimVerification) -> str:
    state = verification.state
    if state == ClaimEvidenceState.CORROBORATED:
        return ""
    if state == ClaimEvidenceState.SUPPORTED_BELOW_REQUIREMENT:
        actual = verification.publisher_domain_proxy_count
        required = verification.required_independent_sources
        if actual == 1:
            return f"〔单一来源：{actual}/{required}〕"
        return f"〔来源不足：{actual}/{required}〕"
    if state == ClaimEvidenceState.CONFLICTING_EVIDENCE:
        return "〔来源冲突〕"
    if state == ClaimEvidenceState.REFUTED:
        return "〔所检来源反驳〕"
    if state == ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT:
        return "〔所检来源未支持〕"
    if state == ClaimEvidenceState.NO_CANDIDATE_SOURCE:
        return "〔未找到候选来源〕"
    if state == ClaimEvidenceState.ATTRIBUTION_ERROR:
        return "〔未核验：归因错误〕"
    if state == ClaimEvidenceState.SUPPORT_QUOTE_UNLOCATABLE:
        return "〔未核验：支持性引文无法定位〕"
    if state in {
        ClaimEvidenceState.VERIFICATION_INCOMPLETE,
        ClaimEvidenceState.VERIFICATION_NOT_RUN,
    }:
        reasons = _unverified_reasons(verification)
        detail = "、".join(reasons) if reasons else "核验未完成"
        return f"〔未核验：{detail}〕"
    if state == ClaimEvidenceState.NORMALIZATION_FAILED:
        return ""
    raise ValueError(f"unsupported claim evidence state: {state}")


def _summary(
    verifications: Sequence[ClaimVerification],
    *,
    settled_without_located_evidence: int,
    settled_without_located_evidence_item_ids: Sequence[str],
    registry_coverage: ClaimRegistryCoverage | None,
) -> EvidenceSummary:
    external = [
        verification
        for verification in verifications
        if verification.claim.citation_requirement
        == CitationRequirement.EXTERNAL
    ]

    def count(*states: ClaimEvidenceState) -> int:
        accepted = set(states)
        return sum(
            verification.state in accepted for verification in external
        )

    verification_incomplete = count(
        ClaimEvidenceState.VERIFICATION_INCOMPLETE
    )
    verification_not_run = count(ClaimEvidenceState.VERIFICATION_NOT_RUN)
    support_quote_unlocatable = count(
        ClaimEvidenceState.SUPPORT_QUOTE_UNLOCATABLE
    )
    claim_normalization_failed = count(
        ClaimEvidenceState.NORMALIZATION_FAILED
    )
    attribution_error = count(ClaimEvidenceState.ATTRIBUTION_ERROR)
    return EvidenceSummary(
        external_claims=len(external),
        corroborated=count(ClaimEvidenceState.CORROBORATED),
        source_shortfall=count(
            ClaimEvidenceState.SUPPORTED_BELOW_REQUIREMENT,
        ),
        conflicting=count(ClaimEvidenceState.CONFLICTING_EVIDENCE),
        refuted=count(ClaimEvidenceState.REFUTED),
        inspected_not_supporting=count(
            ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT
        ),
        no_candidate=count(ClaimEvidenceState.NO_CANDIDATE_SOURCE),
        verification_incomplete=verification_incomplete,
        verification_not_run=verification_not_run,
        support_quote_unlocatable=support_quote_unlocatable,
        claim_normalization_failed=claim_normalization_failed,
        attribution_error=attribution_error,
        unverified=(
            verification_incomplete
            + verification_not_run
            + support_quote_unlocatable
            + claim_normalization_failed
            + attribution_error
        ),
        settled_without_located_evidence=(
            settled_without_located_evidence
        ),
        settled_without_located_evidence_item_ids=tuple(
            settled_without_located_evidence_item_ids
        ),
        registry_coverage=registry_coverage,
    )


def _summary_line(summary: EvidenceSummary) -> str:
    item_ids = ", ".join(summary.settled_without_located_evidence_item_ids)
    collection = (
        f"settled_without_located_evidence="
        f"{summary.settled_without_located_evidence}"
    )
    if item_ids:
        collection += f" ({item_ids})"
    coverage = summary.registry_coverage
    if coverage is None:
        coverage_prefix = ""
        claim_scope = ""
        unverified_scope = ""
    elif coverage.is_complete:
        coverage_prefix = (
            f"正文块评估 {coverage.evaluated_blocks}/"
            f"{coverage.total_blocks}；"
        )
        claim_scope = ""
        unverified_scope = ""
    else:
        coverage_prefix = (
            f"正文块评估 {coverage.evaluated_blocks}/"
            f"{coverage.total_blocks}；"
            f"未评估块 {coverage.unassessed_blocks}；"
            "以下断言统计仅覆盖已评估块："
        )
        claim_scope = "已识别"
        unverified_scope = "已识别断言中"
    return (
        "> 证据摘要："
        f"{coverage_prefix}"
        f"{claim_scope}外部可核验断言 {summary.external_claims}；"
        f"充分支持 {summary.corroborated}；"
        f"来源不足 {summary.source_shortfall}；"
        f"来源冲突 {summary.conflicting}；"
        f"所检来源反驳 {summary.refuted}；"
        f"所检来源未支持 {summary.inspected_not_supporting}；"
        f"未找到候选来源 {summary.no_candidate}；"
        f"{unverified_scope}核验不完整 "
        f"{summary.verification_incomplete}；"
        f"完全未核验 {summary.verification_not_run}；"
        f"支持性引文无法定位 {summary.support_quote_unlocatable}；"
        f"claim 定位失败 {summary.claim_normalization_failed}；"
        f"归因错误 {summary.attribution_error}。"
        f"采集信号：{collection}。"
    )


def _apply_edits(
    markdown: str,
    *,
    removals: Sequence[tuple[int, int]],
    insertions: dict[int, Sequence[str]],
) -> str:
    edits: list[tuple[int, int, str]] = [
        (start, end, "") for start, end in removals
    ]
    edits.extend(
        (position, position, "".join(values))
        for position, values in insertions.items()
    )
    rendered = markdown
    for start, end, replacement in sorted(
        edits,
        key=lambda edit: (edit[0], edit[1] - edit[0]),
        reverse=True,
    ):
        rendered = rendered[:start] + replacement + rendered[end:]
    return rendered


def render_verified_report(
    canonical_draft: str,
    verification: VerificationResult,
    *,
    settled_without_located_evidence: int = 0,
    settled_without_located_evidence_item_ids: Sequence[str] = (),
    registry_coverage: ClaimRegistryCoverage | None = None,
) -> RenderedReport:
    """Insert code-owned evidence markers without rewriting narrative text."""

    if not isinstance(canonical_draft, str):
        raise TypeError("canonical_draft must be text")
    if settled_without_located_evidence < 0:
        raise ValueError(
            "settled_without_located_evidence must be non-negative"
        )

    ordered = sorted(
        verification.claims,
        key=lambda entry: (
            entry.claim.start_char
            if entry.claim.start_char is not None
            else len(canonical_draft) + 1,
            entry.claim.end_char
            if entry.claim.end_char is not None
            else len(canonical_draft) + 1,
            entry.claim.claim_id,
        ),
    )
    footnote_by_key: dict[tuple[str, int, int], RenderedFootnote] = {}
    annotations: list[ClaimRenderAnnotation] = []
    insertion_values: dict[int, list[str]] = defaultdict(list)
    unanchored: list[str] = []

    for entry in ordered:
        claim = entry.claim
        if claim.citation_requirement != CitationRequirement.EXTERNAL:
            continue
        if (
            claim.normalization_status
            != ClaimNormalizationStatus.LOCATED
            or claim.start_char is None
            or claim.end_char is None
            or claim.anchor_text is None
            or canonical_draft[claim.start_char : claim.end_char]
            != claim.anchor_text
        ):
            unanchored.append(claim.claim_id)
            continue

        relations = _renderable_relations(entry)
        relation_numbers: list[tuple[VerifiedSourceRelation, int]] = []
        for relation in relations:
            if relation.span is None or relation.source_quote is None:
                continue
            key = (
                relation.source_id,
                relation.span.start_char,
                relation.span.end_char,
            )
            existing = footnote_by_key.get(key)
            if existing is None:
                existing = RenderedFootnote(
                    number=len(footnote_by_key) + 1,
                    key=EvidenceRegistryKey(
                        source_id=relation.source_id,
                        start_char=relation.span.start_char,
                        end_char=relation.span.end_char,
                    ),
                    source_quote=relation.source_quote,
                    url=relation.url,
                )
                footnote_by_key[key] = existing
            elif (
                existing.source_quote != relation.source_quote
                or existing.url != relation.url
            ):
                raise ValueError(
                    "one evidence registry key resolved to conflicting content"
                )
            relation_numbers.append((relation, existing.number))

        warning = _warning_label(entry)
        if entry.state == ClaimEvidenceState.CONFLICTING_EVIDENCE:
            supporting = tuple(
                number
                for relation, number in relation_numbers
                if relation.semantic_verdict == VerificationVerdict.SUPPORTS
            )
            contradicting = tuple(
                number
                for relation, number in relation_numbers
                if relation.semantic_verdict
                == VerificationVerdict.CONTRADICTS
            )
            pieces: list[str] = []
            if supporting:
                pieces.append(
                    "支持" + "".join(f"[^{number}]" for number in supporting)
                )
            if contradicting:
                pieces.append(
                    "反驳"
                    + "".join(f"[^{number}]" for number in contradicting)
                )
            suffix = (
                "〔来源冲突：" + "；".join(pieces) + "〕"
                if pieces
                else warning
            )
        else:
            numbers = tuple(
                dict.fromkeys(number for _, number in relation_numbers)
            )
            markers = "".join(f"[^{number}]" for number in numbers)
            suffix = markers + warning

        if not suffix:
            continue
        values = insertion_values[claim.end_char]
        if suffix not in values:
            values.append(suffix)
        footnote_numbers = tuple(
            dict.fromkeys(number for _, number in relation_numbers)
        )
        annotations.append(
            ClaimRenderAnnotation(
                claim_id=claim.claim_id,
                anchor_end=claim.end_char,
                evidence_state=entry.state,
                footnote_numbers=footnote_numbers,
                rendered_suffix=suffix,
            )
        )

    removals, definition_count, marker_count = _prohibited_footnote_spans(
        canonical_draft
    )
    body = _apply_edits(
        canonical_draft,
        removals=removals,
        insertions=insertion_values,
    )
    summary = _summary(
        verification.claims,
        settled_without_located_evidence=settled_without_located_evidence,
        settled_without_located_evidence_item_ids=(
            settled_without_located_evidence_item_ids
        ),
        registry_coverage=registry_coverage,
    )
    summary_line = _summary_line(summary)
    footnotes = tuple(
        sorted(footnote_by_key.values(), key=lambda footnote: footnote.number)
    )
    rendered = summary_line + "\n\n" + body
    if footnotes:
        definitions = [
            f"[^{footnote.number}]: "
            + json.dumps(
                {
                    "end_char": footnote.key.end_char,
                    "quote": footnote.source_quote,
                    "source_id": footnote.key.source_id,
                    "start_char": footnote.key.start_char,
                    "url": footnote.url,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for footnote in footnotes
        ]
        separator = "\n" if rendered.endswith("\n") else "\n\n"
        rendered = (
            rendered
            + separator
            + "## 证据来源\n\n"
            + "\n".join(definitions)
            + "\n"
        )
    return RenderedReport(
        markdown=rendered,
        evidence_summary_line=summary_line,
        summary=summary,
        footnotes=footnotes,
        annotations=tuple(annotations),
        removed_model_footnote_definitions=definition_count,
        removed_model_footnote_markers=marker_count,
        unanchored_claim_ids=tuple(unanchored),
    )
