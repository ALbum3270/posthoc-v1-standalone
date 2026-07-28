"""Deterministically render verified evidence into a canonical report draft."""

from __future__ import annotations

import hashlib
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
from open_deep_research.harness.budget_diagnostics import (
    BudgetDecisionSignal,
    RunStopDiagnostic,
)
from open_deep_research.harness.concentration import (
    DomainProxyConcentrationAudit,
)
from open_deep_research.harness.reconcile import ChecklistCoverageSummary
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
_SOURCE_REFERENCE_DEFINITION = re.compile(
    r"^\[(source-\d+)\]: <([^\n]+)>$",
    re.MULTILINE,
)
_EVIDENCE_LEGEND_LINE = (
    "> 图例：带脚注且无额外状态标签 = "
    "单一发布方提供了可定位支持引文"
)
_FOOTNOTE_FORMAT_LINE = (
    "> 脚注格式：`域名代理` · 语义关系 · 逐字证据 · 原文。"
)
_NO_FORMAL_SUPPORT_LINE = (
    "> **证据状态：本报告没有任何可定位的正式支持关系。"
    "下列清单内容覆盖只表示正文讨论了相应调查项，"
    "不表示相关陈述获得来源支持。**"
)


class InitialCollectionSnapshot(BaseModel):
    """Collection-time counts frozen before any post-hoc retrieval can run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cached_source_count: int = Field(ge=0)
    note_count: int = Field(ge=0)
    usable_note_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _usable_notes_cannot_exceed_all_notes(
        self,
    ) -> InitialCollectionSnapshot:
        if self.usable_note_count > self.note_count:
            raise ValueError("usable_note_count cannot exceed note_count")
        return self


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
    publisher_domain_proxy: str
    semantic_verdicts: tuple[VerificationVerdict, ...]
    claim_ids: tuple[str, ...]
    claim_anchors: tuple[str, ...]


class EvidenceBundleValidation(BaseModel):
    """Mechanical proof that report references and source entries are bijective."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    footnote_count: int = Field(ge=0)
    local_definition_count: int = Field(ge=0)
    source_anchor_count: int = Field(ge=0)
    unique_source_url_count: int = Field(ge=0)
    report_url_reference_definition_count: int = Field(ge=0)
    sources_url_reference_definition_count: int = Field(ge=0)
    every_marker_has_local_definition: bool
    every_definition_has_marker: bool
    every_definition_links_to_source_anchor: bool
    every_definition_has_unique_source_anchor: bool
    every_source_anchor_has_definition: bool
    every_source_entry_contains_full_quote: bool
    no_duplicate_definitions: bool
    no_duplicate_source_anchors: bool
    every_footnote_uses_expected_url_reference: bool
    report_url_references_are_unique: bool
    sources_url_references_are_unique: bool
    report_and_sources_url_references_match: bool
    sources_sha256_matches: bool

    @model_validator(mode="after")
    def _all_guarantees_hold(self) -> EvidenceBundleValidation:
        guarantees = (
            self.every_marker_has_local_definition,
            self.every_definition_has_marker,
            self.every_definition_links_to_source_anchor,
            self.every_definition_has_unique_source_anchor,
            self.every_source_anchor_has_definition,
            self.every_source_entry_contains_full_quote,
            self.no_duplicate_definitions,
            self.no_duplicate_source_anchors,
            self.every_footnote_uses_expected_url_reference,
            self.report_url_references_are_unique,
            self.sources_url_references_are_unique,
            self.report_and_sources_url_references_match,
            self.sources_sha256_matches,
        )
        if not all(guarantees):
            raise ValueError("rendered evidence bundle failed validation")
        if not (
            self.footnote_count
            == self.local_definition_count
            == self.source_anchor_count
        ):
            raise ValueError("evidence bundle counts must agree")
        if not (
            self.unique_source_url_count
            == self.report_url_reference_definition_count
            == self.sources_url_reference_definition_count
        ):
            raise ValueError("URL reference definition counts must agree")
        return self


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
    claims_with_located_support: int = Field(ge=0)
    single_publisher_support: int = Field(ge=0)
    multi_publisher_support: int = Field(ge=0)
    zero_publisher_support: int = Field(ge=0)
    corroborated: int = Field(ge=0)
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
    rejected_exhausted_without_collection_attempt: int = Field(ge=0)
    rejected_exhausted_without_collection_attempt_item_ids: tuple[str, ...] = ()
    accepted_exhausted_without_collection_attempt: int = Field(ge=0)
    accepted_exhausted_without_collection_attempt_item_ids: tuple[str, ...] = ()
    accepted_exhausted_attempt_unknown_legacy: int = Field(ge=0)
    accepted_exhausted_attempt_unknown_legacy_item_ids: tuple[str, ...] = ()
    exhausted_with_unread_candidates: int = Field(ge=0)
    exhausted_with_unread_candidates_item_ids: tuple[str, ...] = ()
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
        if (
            self.single_publisher_support
            + self.multi_publisher_support
            + self.zero_publisher_support
            != self.external_claims
        ):
            raise ValueError(
                "publisher-support distribution must cover external claims"
            )
        if self.claims_with_located_support != (
            self.single_publisher_support
            + self.multi_publisher_support
        ):
            raise ValueError(
                "located-support total must match publisher-support counts"
            )
        return self


class RenderedReport(BaseModel):
    """Final Markdown plus the deterministic rendering audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    markdown: str
    sources_markdown: str
    report_filename: str
    sources_filename: str
    audit_filename: str
    sources_sha256: str
    evidence_bundle_line: str
    bundle_validation: EvidenceBundleValidation
    evidence_status_line: str | None = None
    evidence_summary_line: str
    evidence_legend_line: str
    footnote_format_line: str
    checklist_coverage_line: str
    domain_proxy_concentration_line: str | None = None
    summary: EvidenceSummary
    footnotes: tuple[RenderedFootnote, ...] = ()
    annotations: tuple[ClaimRenderAnnotation, ...] = ()
    removed_model_footnote_definitions: int = Field(ge=0)
    removed_model_footnote_markers: int = Field(ge=0)
    unanchored_claim_ids: tuple[str, ...] = ()


_SOURCE_ANCHOR = re.compile(r'^<a id="evidence-(\d+)"></a>$', re.MULTILINE)


def _verdict_label(verdicts: Sequence[VerificationVerdict]) -> str:
    labels = {
        VerificationVerdict.SUPPORTS: "支持",
        VerificationVerdict.CONTRADICTS: "反驳",
    }
    return "、".join(
        labels[verdict]
        for verdict in verdicts
        if verdict in labels
    )


def _quote_fence(text: str) -> str:
    runs = [len(match.group(0)) for match in re.finditer(r"~+", text)]
    return "~" * max(3, (max(runs) + 1) if runs else 3)


def _render_sources_document(
    footnotes: Sequence[RenderedFootnote],
    *,
    run_id: str,
    report_filename: str,
    source_alias_by_url: dict[str, str],
) -> str:
    lines = [
        "# 逐字证据",
        "",
        f"- Run ID：`{run_id}`",
        f"- 对应报告：[{report_filename}]({report_filename})",
        "- 说明：域名仅作发布方代理，不代表机构独立性认定。",
    ]
    for footnote in footnotes:
        lines.extend(
            [
                "",
                f'<a id="evidence-{footnote.number}"></a>',
                f"## 证据 {footnote.number}",
                "",
                f"- 域名代理：`{footnote.publisher_domain_proxy}`",
                f"- 关系：{_verdict_label(footnote.semantic_verdicts)}",
                f"- 原文：[查看原文][{source_alias_by_url[footnote.url]}]",
            ]
        )
        if len(footnote.claim_anchors) > 1:
            lines.extend(["- 用于多个正文锚点："])
            for anchor in footnote.claim_anchors:
                lines.append(
                    "  - "
                    + json.dumps(anchor, ensure_ascii=False)
                )
        fence = _quote_fence(footnote.source_quote)
        lines.extend(
            [
                "",
                "完整逐字引文：",
                "",
                f"{fence}text",
                footnote.source_quote,
                fence,
            ]
        )
    if source_alias_by_url:
        lines.extend(["", "## 原文链接", ""])
        lines.extend(
            f"[{alias}]: <{url}>"
            for url, alias in source_alias_by_url.items()
        )
    return "\n".join(lines) + "\n"


def _validate_evidence_bundle(
    report_markdown: str,
    sources_markdown: str,
    footnotes: Sequence[RenderedFootnote],
    *,
    sources_filename: str,
    sources_sha256: str,
) -> EvidenceBundleValidation:
    definitions = [
        match.group(1)
        for line in report_markdown.splitlines()
        if (match := _FOOTNOTE_DEFINITION.match(line)) is not None
    ]
    markers = _FOOTNOTE_MARKER.findall(
        "".join(
            line
            for line in report_markdown.splitlines(keepends=True)
            if _FOOTNOTE_DEFINITION.match(line) is None
        )
    )
    anchors = _SOURCE_ANCHOR.findall(sources_markdown)
    expected = {str(footnote.number) for footnote in footnotes}
    source_alias_by_url = _source_alias_by_url(footnotes)
    expected_url_references = {
        alias: url for url, alias in source_alias_by_url.items()
    }
    report_url_definitions = _SOURCE_REFERENCE_DEFINITION.findall(
        report_markdown
    )
    sources_url_definitions = _SOURCE_REFERENCE_DEFINITION.findall(
        sources_markdown
    )
    definition_set = set(definitions)
    anchor_set = set(anchors)
    links_complete = True
    url_links_complete = True
    quote_complete = True
    anchor_matches = list(_SOURCE_ANCHOR.finditer(sources_markdown))
    sections: dict[str, str] = {}
    for index, match in enumerate(anchor_matches):
        end = (
            anchor_matches[index + 1].start()
            if index + 1 < len(anchor_matches)
            else len(sources_markdown)
        )
        sections[match.group(1)] = sources_markdown[match.start() : end]
    for footnote in footnotes:
        section = sections.get(str(footnote.number), "")
        if footnote.source_quote not in section:
            quote_complete = False
            break
        link = (
            f"[逐字证据]"
            f"({sources_filename}#evidence-{footnote.number})"
        )
        definition_line = next(
            (
                line
                for line in report_markdown.splitlines()
                if line.startswith(f"[^{footnote.number}]:")
            ),
            "",
        )
        if link not in definition_line:
            links_complete = False
        expected_url_link = (
            f"[原文][{source_alias_by_url[footnote.url]}]"
        )
        if expected_url_link not in definition_line:
            url_links_complete = False
    actual_sha256 = hashlib.sha256(
        sources_markdown.encode("utf-8")
    ).hexdigest()
    return EvidenceBundleValidation(
        footnote_count=len(footnotes),
        local_definition_count=len(definitions),
        source_anchor_count=len(anchors),
        unique_source_url_count=len(source_alias_by_url),
        report_url_reference_definition_count=len(report_url_definitions),
        sources_url_reference_definition_count=len(sources_url_definitions),
        every_marker_has_local_definition=set(markers) <= definition_set,
        every_definition_has_marker=definition_set <= set(markers),
        every_definition_links_to_source_anchor=links_complete,
        every_definition_has_unique_source_anchor=(
            definition_set == anchor_set == expected
        ),
        every_source_anchor_has_definition=anchor_set <= definition_set,
        every_source_entry_contains_full_quote=quote_complete,
        no_duplicate_definitions=len(definitions) == len(definition_set),
        no_duplicate_source_anchors=len(anchors) == len(anchor_set),
        every_footnote_uses_expected_url_reference=url_links_complete,
        report_url_references_are_unique=(
            len(report_url_definitions)
            == len(dict(report_url_definitions))
        ),
        sources_url_references_are_unique=(
            len(sources_url_definitions)
            == len(dict(sources_url_definitions))
        ),
        report_and_sources_url_references_match=(
            dict(report_url_definitions)
            == dict(sources_url_definitions)
            == expected_url_references
        ),
        sources_sha256_matches=actual_sha256 == sources_sha256,
    )


def _source_alias_by_url(
    footnotes: Sequence[RenderedFootnote],
) -> dict[str, str]:
    """Assign stable URL aliases by first footnote occurrence."""

    aliases: dict[str, str] = {}
    for footnote in footnotes:
        if footnote.url not in aliases:
            aliases[footnote.url] = f"source-{len(aliases) + 1}"
    return aliases


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
        return "〔多发布方交叉支持〕"
    if state == ClaimEvidenceState.SUPPORTED_SINGLE_PUBLISHER:
        return ""
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
    rejected_exhausted_without_collection_attempt: int,
    rejected_exhausted_without_collection_attempt_item_ids: Sequence[str],
    accepted_exhausted_without_collection_attempt: int,
    accepted_exhausted_without_collection_attempt_item_ids: Sequence[str],
    accepted_exhausted_attempt_unknown_legacy: int,
    accepted_exhausted_attempt_unknown_legacy_item_ids: Sequence[str],
    exhausted_with_unread_candidates: int,
    exhausted_with_unread_candidates_item_ids: Sequence[str],
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
    single_publisher_support = sum(
        verification.publisher_domain_proxy_count == 1
        for verification in external
    )
    multi_publisher_support = sum(
        verification.publisher_domain_proxy_count >= 2
        for verification in external
    )
    zero_publisher_support = sum(
        verification.publisher_domain_proxy_count == 0
        for verification in external
    )
    return EvidenceSummary(
        external_claims=len(external),
        claims_with_located_support=(
            single_publisher_support + multi_publisher_support
        ),
        single_publisher_support=single_publisher_support,
        multi_publisher_support=multi_publisher_support,
        zero_publisher_support=zero_publisher_support,
        corroborated=count(ClaimEvidenceState.CORROBORATED),
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
        rejected_exhausted_without_collection_attempt=(
            rejected_exhausted_without_collection_attempt
        ),
        rejected_exhausted_without_collection_attempt_item_ids=tuple(
            rejected_exhausted_without_collection_attempt_item_ids
        ),
        accepted_exhausted_without_collection_attempt=(
            accepted_exhausted_without_collection_attempt
        ),
        accepted_exhausted_without_collection_attempt_item_ids=tuple(
            accepted_exhausted_without_collection_attempt_item_ids
        ),
        accepted_exhausted_attempt_unknown_legacy=(
            accepted_exhausted_attempt_unknown_legacy
        ),
        accepted_exhausted_attempt_unknown_legacy_item_ids=tuple(
            accepted_exhausted_attempt_unknown_legacy_item_ids
        ),
        exhausted_with_unread_candidates=exhausted_with_unread_candidates,
        exhausted_with_unread_candidates_item_ids=tuple(
            exhausted_with_unread_candidates_item_ids
        ),
        registry_coverage=registry_coverage,
    )


def _summary_line(
    summary: EvidenceSummary,
    *,
    disagreement_attempted_count: int | None,
) -> str:
    item_ids = ", ".join(summary.settled_without_located_evidence_item_ids)
    collection = (
        f"settled_without_located_evidence="
        f"{summary.settled_without_located_evidence}"
    )
    if item_ids:
        collection += f" ({item_ids})"
    rejected_ids = ", ".join(
        summary.rejected_exhausted_without_collection_attempt_item_ids
    )
    if summary.rejected_exhausted_without_collection_attempt:
        collection += (
            "；拒绝无采集尝试的查遍未找到声明 "
            f"{summary.rejected_exhausted_without_collection_attempt}"
            f" ({rejected_ids})"
        )
    accepted_ids = ", ".join(
        summary.accepted_exhausted_without_collection_attempt_item_ids
    )
    if summary.accepted_exhausted_without_collection_attempt:
        collection += (
            "；采集期接受的零尝试查遍未找到声明 "
            f"{summary.accepted_exhausted_without_collection_attempt}"
            f" ({accepted_ids})"
        )
    unknown_ids = ", ".join(
        summary.accepted_exhausted_attempt_unknown_legacy_item_ids
    )
    if summary.accepted_exhausted_attempt_unknown_legacy:
        collection += (
            "；历史查遍未找到声明缺少尝试快照 "
            f"{summary.accepted_exhausted_attempt_unknown_legacy}"
            f" ({unknown_ids})"
        )
    unread_ids = ", ".join(
        summary.exhausted_with_unread_candidates_item_ids
    )
    if summary.exhausted_with_unread_candidates:
        collection += (
            "；仍有未读候选时判为查遍未找到 "
            f"{summary.exhausted_with_unread_candidates}"
            f" ({unread_ids})"
        )
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
    if summary.conflicting:
        conflict = f"来源冲突 {summary.conflicting}"
        if disagreement_attempted_count is not None:
            conflict += f"（分歧探测已尝试 {disagreement_attempted_count} 条）"
    elif disagreement_attempted_count:
        conflict = (
            "来源冲突 0（分歧探测已尝试 "
            f"{disagreement_attempted_count} 条，未发现冲突；"
            "不表示已确认无争议）"
        )
    else:
        conflict = (
            "来源冲突 0（仅表示现有候选中未发现；未执行分歧探测）"
        )
    return (
        "> 证据摘要："
        f"{coverage_prefix}"
        f"{claim_scope}外部可核验断言 {summary.external_claims}；"
        f"其中 {summary.claims_with_located_support} 条有至少一条"
        "可定位的支持引文；"
        f"单一发布方支持 {summary.single_publisher_support}；"
        f"多发布方交叉支持 {summary.multi_publisher_support}；"
        f"零发布方支持 {summary.zero_publisher_support}；"
        f"{conflict}；"
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


def _checklist_coverage_line(
    summary: ChecklistCoverageSummary | None,
) -> str:
    if summary is None:
        return "> 清单内容覆盖（不表示来源支持）：未执行。"
    uncovered = (
        ", ".join(summary.not_covered_item_ids)
        if summary.not_covered_item_ids
        else "无"
    )
    failed = (
        ", ".join(summary.assessment_failed_item_ids)
        if summary.assessment_failed_item_ids
        else "无"
    )
    return (
        "> 清单内容覆盖（不表示来源支持）："
        f"已评估 {summary.assessed_items}/{summary.total_items}；"
        f"完整覆盖 {summary.covered_items}/{summary.total_items}"
        f"（{summary.covered_rate:.1%}）；"
        f"部分覆盖 {summary.partially_covered_items}；"
        f"未覆盖 {summary.not_covered_items}（{uncovered}）；"
        f"对账失败 {summary.assessment_failed_items}（{failed}）。"
    )


def _domain_proxy_concentration_line(
    audit: DomainProxyConcentrationAudit,
) -> str:
    overall = audit.overall
    if overall.formal_support_relation_count == 0:
        return (
            "> 域名代理集中度：没有正式 claim–source 支持关系；"
            "域名仅作发布方代理。"
        )
    return (
        "> 域名代理集中度：最大域名代理 "
        f"{overall.largest_publisher_domain_proxy} 占正式 claim–source "
        "支持关系的 "
        f"{overall.largest_publisher_domain_proxy_share:.1%}"
        f"（{overall.largest_publisher_domain_proxy_relation_count}/"
        f"{overall.formal_support_relation_count}）；"
        "域名仅作发布方代理。"
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


_BUDGET_SIGNAL_LABEL = {
    BudgetDecisionSignal.MORE_BUDGET_MAY_HELP: (
        "近期取材仍在产出且仍有待办，提高上限可能有用"
    ),
    BudgetDecisionSignal.FIX_MECHANISM_FIRST: (
        "近期取材未产出新材料，先修机制而非加预算"
    ),
    BudgetDecisionSignal.INDETERMINATE: (
        "近期取材有产出也有空转，证据不足以判断加预算是否有用"
    ),
}


def _render_cutoff_line(diagnostic: RunStopDiagnostic | None) -> str | None:
    """Say in the report itself that a ceiling cut the work short.

    A reader deciding whether to pay for more has to see that the run was
    interrupted at all. Leaving this only in the audit file means the decision
    gets made by whoever reads the report, without the one fact that matters.
    """

    if diagnostic is None or not diagnostic.work_was_curtailed:
        return None
    boundary = diagnostic.boundary
    if not diagnostic.cap_was_binding:
        # No call was refused; a pre-check skipped whole stages instead. Say
        # which, because "the cap bound" and "the round never started" leave
        # the reader with different work missing.
        skipped = "、".join(diagnostic.budget_curtailed_stages)
        judgement = _BUDGET_SIGNAL_LABEL.get(
            diagnostic.budget_decision_signal, "无判断"
        )
        return (
            f"> **预算不足以启动以下阶段，它们被跳过：{skipped}。** "
            f"未拒绝任何调用，但计划中的工作没有执行。{judgement}。"
            "详见审计 `stop.diagnostic`。"
        )
    where = (
        f"{boundary.scope} {boundary.resource} "
        f"{boundary.used:g}/{boundary.limit:g}"
        if boundary is not None
        else diagnostic.resource_stop_reason.value
    )
    owed = diagnostic.outstanding
    owed_parts = []
    if owed.open_checklist_items:
        owed_parts.append(f"未结清单项 {owed.open_checklist_items}")
    if owed.unverified_relations:
        owed_parts.append(f"因预算未核验关系 {owed.unverified_relations}")
    if owed.evidence_gap_plan_unexecuted:
        owed_parts.append("证据缺口补采未执行")
    if owed.disagreement_plan_unexecuted:
        owed_parts.append("分歧检测未执行")
    owed_text = "；".join(owed_parts) if owed_parts else "无剩余待办"
    judgement = _BUDGET_SIGNAL_LABEL.get(
        diagnostic.budget_decision_signal,
        "未触及上限，无需判断",
    )
    return (
        f"> **本次运行被成本上限截断（{where}）。** 截止时：{owed_text}。"
        f"{judgement}。详见审计 `stop.diagnostic`。"
    )


def render_verified_report(
    canonical_draft: str,
    verification: VerificationResult,
    *,
    settled_without_located_evidence: int = 0,
    settled_without_located_evidence_item_ids: Sequence[str] = (),
    rejected_exhausted_without_collection_attempt: int = 0,
    rejected_exhausted_without_collection_attempt_item_ids: Sequence[str] = (),
    accepted_exhausted_without_collection_attempt: int = 0,
    accepted_exhausted_without_collection_attempt_item_ids: Sequence[str] = (),
    accepted_exhausted_attempt_unknown_legacy: int = 0,
    accepted_exhausted_attempt_unknown_legacy_item_ids: Sequence[str] = (),
    exhausted_with_unread_candidates: int = 0,
    exhausted_with_unread_candidates_item_ids: Sequence[str] = (),
    registry_coverage: ClaimRegistryCoverage | None = None,
    checklist_coverage: ChecklistCoverageSummary | None = None,
    domain_proxy_concentration: DomainProxyConcentrationAudit | None = None,
    disagreement_attempted_count: int | None = None,
    initial_collection_snapshot: InitialCollectionSnapshot | None = None,
    stop_diagnostic: RunStopDiagnostic | None = None,
    run_id: str = "standalone",
    report_filename: str = "report.md",
    sources_filename: str = "report.sources.md",
    audit_filename: str = "report.json",
) -> RenderedReport:
    """Insert code-owned evidence markers without rewriting narrative text."""

    if not isinstance(canonical_draft, str):
        raise TypeError("canonical_draft must be text")
    if settled_without_located_evidence < 0:
        raise ValueError(
            "settled_without_located_evidence must be non-negative"
        )
    for label, count in (
        (
            "rejected_exhausted_without_collection_attempt",
            rejected_exhausted_without_collection_attempt,
        ),
        (
            "accepted_exhausted_without_collection_attempt",
            accepted_exhausted_without_collection_attempt,
        ),
        (
            "accepted_exhausted_attempt_unknown_legacy",
            accepted_exhausted_attempt_unknown_legacy,
        ),
        (
            "exhausted_with_unread_candidates",
            exhausted_with_unread_candidates,
        ),
    ):
        if count < 0:
            raise ValueError(f"{label} must be non-negative")
    for label, filename in (
        ("report_filename", report_filename),
        ("sources_filename", sources_filename),
        ("audit_filename", audit_filename),
    ):
        if (
            not filename
            or filename != filename.strip()
            or "/" in filename
            or "\\" in filename
            or filename in {".", ".."}
        ):
            raise ValueError(f"{label} must be one relative filename")

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
                if relation.semantic_verdict is None:
                    raise AssertionError(
                        "renderable relation requires a semantic verdict"
                    )
                existing = RenderedFootnote(
                    number=len(footnote_by_key) + 1,
                    key=EvidenceRegistryKey(
                        source_id=relation.source_id,
                        start_char=relation.span.start_char,
                        end_char=relation.span.end_char,
                    ),
                    source_quote=relation.source_quote,
                    url=relation.url,
                    publisher_domain_proxy=(
                        relation.publisher_domain_proxy
                    ),
                    semantic_verdicts=(relation.semantic_verdict,),
                    claim_ids=(claim.claim_id,),
                    claim_anchors=(claim.anchor_text,),
                )
                footnote_by_key[key] = existing
            elif (
                existing.source_quote != relation.source_quote
                or existing.url != relation.url
                or existing.publisher_domain_proxy
                != relation.publisher_domain_proxy
            ):
                raise ValueError(
                    "one evidence registry key resolved to conflicting content"
                )
            else:
                if relation.semantic_verdict is None:
                    raise AssertionError(
                        "renderable relation requires a semantic verdict"
                    )
                existing = existing.model_copy(
                    update={
                        "semantic_verdicts": tuple(
                            dict.fromkeys(
                                (
                                    *existing.semantic_verdicts,
                                    relation.semantic_verdict,
                                )
                            )
                        ),
                        "claim_ids": tuple(
                            dict.fromkeys(
                                (*existing.claim_ids, claim.claim_id)
                            )
                        ),
                        "claim_anchors": tuple(
                            dict.fromkeys(
                                (*existing.claim_anchors, claim.anchor_text)
                            )
                        ),
                    }
                )
                footnote_by_key[key] = existing
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
        rejected_exhausted_without_collection_attempt=(
            rejected_exhausted_without_collection_attempt
        ),
        rejected_exhausted_without_collection_attempt_item_ids=(
            rejected_exhausted_without_collection_attempt_item_ids
        ),
        accepted_exhausted_without_collection_attempt=(
            accepted_exhausted_without_collection_attempt
        ),
        accepted_exhausted_without_collection_attempt_item_ids=(
            accepted_exhausted_without_collection_attempt_item_ids
        ),
        accepted_exhausted_attempt_unknown_legacy=(
            accepted_exhausted_attempt_unknown_legacy
        ),
        accepted_exhausted_attempt_unknown_legacy_item_ids=(
            accepted_exhausted_attempt_unknown_legacy_item_ids
        ),
        exhausted_with_unread_candidates=exhausted_with_unread_candidates,
        exhausted_with_unread_candidates_item_ids=(
            exhausted_with_unread_candidates_item_ids
        ),
        registry_coverage=registry_coverage,
    )
    summary_line = _summary_line(
        summary,
        disagreement_attempted_count=disagreement_attempted_count,
    )
    checklist_line = _checklist_coverage_line(checklist_coverage)
    concentration_line = (
        _domain_proxy_concentration_line(domain_proxy_concentration)
        if domain_proxy_concentration is not None
        else None
    )
    footnotes = tuple(
        sorted(footnote_by_key.values(), key=lambda footnote: footnote.number)
    )
    source_alias_by_url = _source_alias_by_url(footnotes)
    formal_support_relation_count = len(
        {
            (entry.claim.claim_id, relation.source_id)
            for entry in verification.claims
            for relation in entry.relations
            if relation.is_formal_supporting_evidence
        }
    )
    if formal_support_relation_count == 0:
        evidence_status_line = _NO_FORMAL_SUPPORT_LINE
        if (
            initial_collection_snapshot is not None
            and initial_collection_snapshot.cached_source_count == 0
        ):
            evidence_status_line += " **初次采集阶段未取得任何原文。**"
    else:
        evidence_status_line = None
    sources_markdown = _render_sources_document(
        footnotes,
        run_id=run_id,
        report_filename=report_filename,
        source_alias_by_url=source_alias_by_url,
    )
    sources_sha256 = hashlib.sha256(
        sources_markdown.encode("utf-8")
    ).hexdigest()
    bundle_line = (
        "> 证据包：完整逐字证据见 "
        f"[{sources_filename}]({sources_filename})；"
        f"SHA-256 `{sources_sha256}`。审计提交标记见 "
        f"[{audit_filename}]({audit_filename})；"
        "缺失逐字证据、提交标记或摘要不符则证据包不完整。"
    )
    header_lines = [bundle_line]
    cutoff_line = _render_cutoff_line(stop_diagnostic)
    if cutoff_line is not None:
        header_lines.append(cutoff_line)
    if evidence_status_line is not None:
        header_lines.append(evidence_status_line)
    header_lines.append(summary_line)
    if concentration_line is not None:
        header_lines.append(concentration_line)
    header_lines.extend(
        [_FOOTNOTE_FORMAT_LINE, _EVIDENCE_LEGEND_LINE, checklist_line]
    )
    rendered = "\n".join(header_lines) + "\n\n" + body
    if footnotes:
        definitions = [
            (
                f"[^{footnote.number}]: "
                f"`{footnote.publisher_domain_proxy}` · "
                f"{_verdict_label(footnote.semantic_verdicts)} · "
                "[逐字证据]"
                f"({sources_filename}#evidence-{footnote.number}) · "
                f"[原文][{source_alias_by_url[footnote.url]}]"
            )
            for footnote in footnotes
        ]
        url_definitions = [
            f"[{alias}]: <{url}>"
            for url, alias in source_alias_by_url.items()
        ]
        separator = "\n" if rendered.endswith("\n") else "\n\n"
        rendered = (
            rendered
            + separator
            + "## 证据来源\n\n"
            + "\n".join(definitions)
            + "\n\n"
            + "\n".join(url_definitions)
            + "\n"
        )
    bundle_validation = _validate_evidence_bundle(
        rendered,
        sources_markdown,
        footnotes,
        sources_filename=sources_filename,
        sources_sha256=sources_sha256,
    )
    return RenderedReport(
        markdown=rendered,
        sources_markdown=sources_markdown,
        report_filename=report_filename,
        sources_filename=sources_filename,
        audit_filename=audit_filename,
        sources_sha256=sources_sha256,
        evidence_bundle_line=bundle_line,
        bundle_validation=bundle_validation,
        evidence_status_line=evidence_status_line,
        evidence_summary_line=summary_line,
        evidence_legend_line=_EVIDENCE_LEGEND_LINE,
        footnote_format_line=_FOOTNOTE_FORMAT_LINE,
        checklist_coverage_line=checklist_line,
        domain_proxy_concentration_line=concentration_line,
        summary=summary,
        footnotes=footnotes,
        annotations=tuple(annotations),
        removed_model_footnote_definitions=definition_count,
        removed_model_footnote_markers=marker_count,
        unanchored_claim_ids=tuple(unanchored),
    )
