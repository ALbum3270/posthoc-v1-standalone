"""Deterministically render verified evidence into a canonical report draft."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from enum import Enum

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

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
from open_deep_research.harness.notes import QuoteSpan
from open_deep_research.harness.reconcile import ChecklistCoverageSummary
from open_deep_research.harness.truth_conditions import (
    ClaimCoverageState,
    ElementAssessmentExecutionStatus,
    ElementVerificationVerdict,
    ElementizationSemanticStatus,
    ExecutionCompleteness,
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
_SOURCE_REFERENCE_DEFINITION = re.compile(
    r"^\[(source-\d+)\]: <([^\n]+)>$",
    re.MULTILINE,
)
_EVIDENCE_LEGEND_LINE = (
    "> 图例：带脚注且无额外状态标签 = "
    "至少一个来源提供了可定位支持引文；域名代理数量不表示来源独立"
)
_FOOTNOTE_FORMAT_LINE = (
    "> 脚注格式：`域名代理` · 语义关系 · 逐字证据 · 原文。"
)
_NO_FORMAL_SUPPORT_LINE = (
    "> **证据状态：本报告没有任何可定位的正式支持关系。"
    "下列清单内容覆盖只表示正文讨论了相应调查项，"
    "不表示相关陈述获得来源支持。**"
)


class ReaderReportStyle(str, Enum):
    """Versioned separation between reader prose and audit presentation."""

    AUDIT_ANNOTATED = "audit-annotated-v1"
    CLEAN = "clean-reader-v2"


_READER_RENDER_CONTRACT = {
    ReaderReportStyle.AUDIT_ANNOTATED: "posthoc-evidence-v1",
    ReaderReportStyle.CLEAN: "posthoc-evidence-v1.1-clean-reader",
}

_CLEAN_FOOTNOTE_FORMAT_LINE = (
    "脚注格式：发布方链接 · 证据摘录；语义关系与完整逐字证据见伴随文件。"
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
    element_ids: tuple[str, ...] = Field(
        default=(),
        exclude_if=lambda value: not value,
    )


class _RenderableEvidence(BaseModel):
    """One located quote, flattened from either verification protocol.

    Element-v2 keeps truth-condition judgements below the source relation.  The
    renderer flattens only for presentation; the audit retains the nested
    denominator and no child verdict is promoted into a whole-claim verdict.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    url: str
    publisher_domain_proxy: str
    semantic_verdict: VerificationVerdict
    source_quote: str
    span: QuoteSpan
    element_ids: tuple[str, ...] = ()


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
    single_domain_proxy_support: int = Field(
        ge=0,
        validation_alias=AliasChoices(
            "single_domain_proxy_support",
            "single_publisher_support",
        ),
    )
    multiple_domain_proxy_support: int = Field(
        ge=0,
        validation_alias=AliasChoices(
            "multiple_domain_proxy_support",
            "multi_publisher_support",
        ),
    )
    element_level_support: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    distributed_element_support: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    zero_located_support: int = Field(
        ge=0,
        validation_alias=AliasChoices(
            "zero_located_support",
            "zero_publisher_support",
        ),
    )
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
    # These fields are additive and default to zero so historical audit
    # payloads remain readable.  The older located-support counts answer only
    # whether at least one usable quote exists; they must not be read as proof
    # that every material condition of a compound claim was supported.
    truth_condition_claims: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    truth_condition_fully_supported: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    truth_condition_partially_supported: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    truth_condition_mixed: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    truth_condition_not_supported: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    truth_condition_contradicted: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    truth_condition_conflicted: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    truth_condition_unresolved: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    truth_condition_execution_complete: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    truth_condition_execution_partial: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    truth_condition_execution_failed: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    truth_condition_execution_not_run: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    truth_condition_execution_incomplete_overlap: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    truth_condition_elementization_complete: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    truth_condition_elementization_incomplete: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    truth_condition_elementization_uncertain: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
    truth_condition_elementization_unresolved: int = Field(
        default=0,
        ge=0,
        exclude_if=lambda value: value == 0,
    )
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
        legacy_components = (
            self.verification_incomplete
            + self.verification_not_run
            + self.support_quote_unlocatable
            + self.claim_normalization_failed
            + self.attribution_error
        )
        truth_execution_incomplete = (
            self.truth_condition_execution_partial
            + self.truth_condition_execution_failed
            + self.truth_condition_execution_not_run
        )
        overlap = self.truth_condition_execution_incomplete_overlap
        if overlap > min(legacy_components, truth_execution_incomplete):
            raise ValueError(
                "truth-condition incomplete overlap exceeds either scope"
            )
        if self.unverified != (
            legacy_components + truth_execution_incomplete - overlap
        ):
            raise ValueError(
                "unverified must equal the union of legacy and element "
                "execution-incomplete claim scopes"
            )
        if (
            self.single_domain_proxy_support
            + self.multiple_domain_proxy_support
            + self.element_level_support
            + self.zero_located_support
            != self.external_claims
        ):
            raise ValueError(
                "publisher-support distribution must cover external claims"
            )
        if self.claims_with_located_support != (
            self.single_domain_proxy_support
            + self.multiple_domain_proxy_support
            + self.element_level_support
        ):
            raise ValueError(
                "located-support total must match publisher-support counts"
            )
        if self.distributed_element_support > self.element_level_support:
            raise ValueError(
                "distributed element support must be a subset of "
                "element-level support"
            )
        truth_condition_partition = (
            self.truth_condition_fully_supported
            + self.truth_condition_partially_supported
            + self.truth_condition_mixed
            + self.truth_condition_not_supported
            + self.truth_condition_contradicted
            + self.truth_condition_conflicted
            + self.truth_condition_unresolved
        )
        if truth_condition_partition != self.truth_condition_claims:
            raise ValueError(
                "truth-condition coverage counts must partition their claims"
            )
        if self.truth_condition_claims > self.external_claims:
            raise ValueError(
                "truth-condition claims cannot exceed external claim scope"
            )
        truth_condition_execution_partition = (
            self.truth_condition_execution_complete
            + self.truth_condition_execution_partial
            + self.truth_condition_execution_failed
            + self.truth_condition_execution_not_run
        )
        if truth_condition_execution_partition != self.truth_condition_claims:
            raise ValueError(
                "truth-condition execution counts must partition their claims"
            )
        elementization_partition = (
            self.truth_condition_elementization_complete
            + self.truth_condition_elementization_incomplete
            + self.truth_condition_elementization_uncertain
            + self.truth_condition_elementization_unresolved
        )
        # A zero partition is the historical payload shape from before this
        # disclosure existed. New summaries always emit a complete partition.
        if elementization_partition not in {0, self.truth_condition_claims}:
            raise ValueError(
                "truth-condition elementization counts must partition their "
                "claims"
            )
        return self

    @property
    def single_publisher_support(self) -> int:
        """Read historical callers without re-emitting publisher semantics."""

        return self.single_domain_proxy_support

    @property
    def multi_publisher_support(self) -> int:
        """Read historical callers without re-emitting publisher semantics."""

        return self.multiple_domain_proxy_support

    @property
    def zero_publisher_support(self) -> int:
        """Read historical callers without re-emitting publisher semantics."""

        return self.zero_located_support


class RenderedReport(BaseModel):
    """Final Markdown plus the deterministic rendering audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    markdown: str
    sources_markdown: str
    reader_report_style: ReaderReportStyle
    reader_render_contract: str
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
    audit_lines: Sequence[str] = (),
) -> str:
    lines = [
        "# 证据与审计伴随文件" if audit_lines else "# 逐字证据",
        "",
        f"- Run ID：`{run_id}`",
        f"- 对应报告：[{report_filename}]({report_filename})",
        "- 说明：域名仅作发布方代理，不代表机构独立性认定。",
    ]
    if audit_lines:
        lines.extend(["", "## 机械审计摘要", ""])
        lines.extend(line.removeprefix("> ") for line in audit_lines)
        lines.extend(["", "## 逐字证据"])
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
        if footnote.element_ids:
            lines.append(
                "- 真值条件："
                + "、".join(f"`{element_id}`" for element_id in footnote.element_ids)
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
    reader_report_style: ReaderReportStyle,
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
        link_label = (
            "逐字证据"
            if reader_report_style is ReaderReportStyle.AUDIT_ANNOTATED
            else "证据摘录"
        )
        link = (
            f"[{link_label}]"
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
        alias = source_alias_by_url[footnote.url]
        expected_url_link = (
            f"[原文][{alias}]"
            if reader_report_style is ReaderReportStyle.AUDIT_ANNOTATED
            else f"[{footnote.publisher_domain_proxy}][{alias}]"
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
) -> tuple[_RenderableEvidence, ...]:
    relations: list[_RenderableEvidence] = []
    for relation in verification.relations:
        if relation.element_relations:
            for element in relation.element_relations:
                if (
                    element.status
                    is not ElementAssessmentExecutionStatus.COMPLETE
                    or element.source_quote is None
                    or element.span is None
                    or element.semantic_verdict
                    not in {
                        ElementVerificationVerdict.SUPPORTS,
                        ElementVerificationVerdict.CONTRADICTS,
                    }
                    or (
                        element.semantic_verdict
                        is ElementVerificationVerdict.SUPPORTS
                        and not element.is_formal_supporting_evidence
                    )
                ):
                    continue
                relations.append(
                    _RenderableEvidence(
                        source_id=relation.source_id,
                        url=relation.url,
                        publisher_domain_proxy=(
                            relation.publisher_domain_proxy
                        ),
                        semantic_verdict=VerificationVerdict(
                            element.semantic_verdict.value
                        ),
                        source_quote=element.source_quote,
                        span=element.span,
                        element_ids=(element.element_id,),
                    )
                )
            continue
        if (
            _located_relation(relation)
            and relation.semantic_verdict
            in {
                VerificationVerdict.SUPPORTS,
                VerificationVerdict.CONTRADICTS,
            }
            and (
                relation.is_formal_supporting_evidence
                or relation.semantic_verdict is VerificationVerdict.CONTRADICTS
            )
        ):
            assert relation.source_quote is not None
            assert relation.span is not None
            assert relation.semantic_verdict is not None
            relations.append(
                _RenderableEvidence(
                    source_id=relation.source_id,
                    url=relation.url,
                    publisher_domain_proxy=relation.publisher_domain_proxy,
                    semantic_verdict=relation.semantic_verdict,
                    source_quote=relation.source_quote,
                    span=relation.span,
                )
            )
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


def _truth_condition_process_limitations(
    verification: ClaimVerification,
) -> tuple[str, ...]:
    """Return non-verdict limitations that must survive label formatting."""

    aggregate = verification.truth_condition_aggregate
    if aggregate is None:
        return ()
    details: list[str] = []
    semantic_status = aggregate.elementization_semantic_status
    if semantic_status is ElementizationSemanticStatus.INCOMPLETE:
        details.append("真值条件拆分不完整")
    elif semantic_status is ElementizationSemanticStatus.UNCERTAIN:
        details.append("真值条件拆分完整性未决")
    execution_detail = {
        ExecutionCompleteness.COMPLETE: None,
        ExecutionCompleteness.PARTIAL: "真值条件核验执行部分完成",
        ExecutionCompleteness.FAILED: "真值条件核验执行失败",
        ExecutionCompleteness.NOT_RUN: "真值条件核验未运行",
    }[aggregate.execution_completeness]
    if execution_detail is not None:
        details.append(execution_detail)
    return tuple(details)


def _warning_label(verification: ClaimVerification) -> str:
    state = verification.state
    # These states are decided before, or independently from, external source
    # verification.  An empty element-source denominator must not replace the
    # more actionable reason (for example, "no candidate source") with a
    # generic execution warning.
    if state is ClaimEvidenceState.NO_CANDIDATE_SOURCE:
        return "〔未找到候选来源〕"
    if state is ClaimEvidenceState.ATTRIBUTION_ERROR:
        return "〔未核验：归因错误〕"
    if state is ClaimEvidenceState.NORMALIZATION_FAILED:
        return ""
    if state is ClaimEvidenceState.INTERNAL_SUPPORTED:
        return ""
    if state is ClaimEvidenceState.INTERNAL_NOT_SUPPORTED:
        return "〔报告内部依据不支持〕"
    if state is ClaimEvidenceState.EVIDENCE_OBLIGATION_UNRESOLVED:
        return "〔证据义务未决〕"
    aggregate = verification.truth_condition_aggregate
    if aggregate is not None:
        coverage = aggregate.coverage_state
        details: list[str] = []
        semantic_status = aggregate.elementization_semantic_status
        semantic_denominator_unresolved = semantic_status in {
            ElementizationSemanticStatus.INCOMPLETE,
            ElementizationSemanticStatus.UNCERTAIN,
        }
        if (
            coverage is ClaimCoverageState.PARTIALLY_SUPPORTED
            and semantic_denominator_unresolved
        ):
            details.append(
                "真值条件支持结论仅覆盖已登记条件，不能视为完整断言支持"
            )
        elif coverage is ClaimCoverageState.PARTIALLY_SUPPORTED:
            details.append("部分真值条件获得支持，其余未获支持")
        elif coverage is ClaimCoverageState.MIXED:
            details.append("部分真值条件获得支持，另有条件被反驳")
        elif coverage is ClaimCoverageState.NOT_SUPPORTED:
            details.append("真值条件未获支持")
        elif coverage is ClaimCoverageState.CONTRADICTED:
            details.append("所检来源反驳真值条件")
        elif coverage is ClaimCoverageState.CONFLICTED:
            details.append("真值条件来源冲突")
        elif coverage is ClaimCoverageState.UNRESOLVED:
            details.append("真值条件或其核验未决")
        details.extend(_truth_condition_process_limitations(verification))
        if (
            state
            is ClaimEvidenceState.SUPPORTED_DISTRIBUTED_ELEMENT_EVIDENCE
        ):
            details.append(
                "完整真值条件由不同来源分别支持，无单一来源支持整条断言"
            )
        if details:
            return "〔" + "；".join(details) + "〕"
    if state == ClaimEvidenceState.CORROBORATED:
        return "〔经来源谱系评估的交叉支持〕"
    if state in {
        ClaimEvidenceState.SUPPORTED_SINGLE_DOMAIN_PROXY,
        ClaimEvidenceState.SUPPORTED_MULTIPLE_DOMAIN_PROXIES,
    }:
        return ""
    if (
        state
        is ClaimEvidenceState.SUPPORTED_DISTRIBUTED_ELEMENT_EVIDENCE
    ):
        # A truth-condition aggregate should have produced the more detailed
        # branch above. Keep malformed historical callers visible rather than
        # silently presenting distributed evidence as whole-claim support.
        return "〔不同来源分别支持真值条件；无单一来源支持整条断言〕"
    if state == ClaimEvidenceState.CONFLICTING_EVIDENCE:
        return "〔来源冲突〕"
    if state == ClaimEvidenceState.REFUTED:
        return "〔所检来源反驳〕"
    if state == ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT:
        return "〔所检来源未支持〕"
    if state == ClaimEvidenceState.SUPPORT_QUOTE_UNLOCATABLE:
        return "〔未核验：支持性引文无法定位〕"
    if state in {
        ClaimEvidenceState.VERIFICATION_INCOMPLETE,
        ClaimEvidenceState.VERIFICATION_NOT_RUN,
    }:
        reasons = _unverified_reasons(verification)
        detail = "、".join(reasons) if reasons else "核验未完成"
        return f"〔未核验：{detail}〕"
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
    single_domain_proxy_support = sum(
        verification.publisher_domain_proxy_count == 1
        for verification in external
    )
    multiple_domain_proxy_support = sum(
        verification.publisher_domain_proxy_count >= 2
        for verification in external
    )
    element_level_support = sum(
        verification.publisher_domain_proxy_count == 0
        and verification.element_supporting_domain_proxy_count > 0
        for verification in external
    )
    distributed_element_support = count(
        ClaimEvidenceState.SUPPORTED_DISTRIBUTED_ELEMENT_EVIDENCE
    )
    zero_located_support = sum(
        verification.publisher_domain_proxy_count == 0
        and verification.element_supporting_domain_proxy_count == 0
        for verification in external
    )
    truth_condition_coverage = {
        state: sum(
            verification.truth_condition_aggregate is not None
            and verification.truth_condition_aggregate.coverage_state is state
            for verification in external
        )
        for state in ClaimCoverageState
    }
    truth_condition_claims = sum(truth_condition_coverage.values())
    truth_condition_execution = {
        state: sum(
            verification.truth_condition_aggregate is not None
            and verification.truth_condition_aggregate.execution_completeness
            is state
            for verification in external
        )
        for state in ExecutionCompleteness
    }
    truth_condition_elementization = {
        state: sum(
            verification.truth_condition_aggregate is not None
            and verification.truth_condition_aggregate
            .elementization_semantic_status
            is state
            for verification in external
        )
        for state in ElementizationSemanticStatus
    }
    truth_condition_elementization_unresolved = sum(
        verification.truth_condition_aggregate is not None
        and verification.truth_condition_aggregate.elementization_semantic_status
        is None
        for verification in external
    )
    claims_with_incomplete_execution = sum(
        verification.state
        in {
            ClaimEvidenceState.VERIFICATION_INCOMPLETE,
            ClaimEvidenceState.VERIFICATION_NOT_RUN,
            ClaimEvidenceState.SUPPORT_QUOTE_UNLOCATABLE,
            ClaimEvidenceState.NORMALIZATION_FAILED,
            ClaimEvidenceState.ATTRIBUTION_ERROR,
        }
        or (
            verification.truth_condition_aggregate is not None
            and verification.truth_condition_aggregate.execution_completeness
            is not ExecutionCompleteness.COMPLETE
        )
        for verification in external
    )
    truth_condition_execution_incomplete_overlap = sum(
        verification.state
        in {
            ClaimEvidenceState.VERIFICATION_INCOMPLETE,
            ClaimEvidenceState.VERIFICATION_NOT_RUN,
            ClaimEvidenceState.SUPPORT_QUOTE_UNLOCATABLE,
            ClaimEvidenceState.NORMALIZATION_FAILED,
            ClaimEvidenceState.ATTRIBUTION_ERROR,
        }
        and verification.truth_condition_aggregate is not None
        and verification.truth_condition_aggregate.execution_completeness
        is not ExecutionCompleteness.COMPLETE
        for verification in external
    )
    return EvidenceSummary(
        external_claims=len(external),
        claims_with_located_support=(
            single_domain_proxy_support
            + multiple_domain_proxy_support
            + element_level_support
        ),
        single_domain_proxy_support=single_domain_proxy_support,
        multiple_domain_proxy_support=multiple_domain_proxy_support,
        element_level_support=element_level_support,
        distributed_element_support=distributed_element_support,
        zero_located_support=zero_located_support,
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
        unverified=claims_with_incomplete_execution,
        truth_condition_claims=truth_condition_claims,
        truth_condition_fully_supported=truth_condition_coverage[
            ClaimCoverageState.FULLY_SUPPORTED
        ],
        truth_condition_partially_supported=truth_condition_coverage[
            ClaimCoverageState.PARTIALLY_SUPPORTED
        ],
        truth_condition_mixed=truth_condition_coverage[
            ClaimCoverageState.MIXED
        ],
        truth_condition_not_supported=truth_condition_coverage[
            ClaimCoverageState.NOT_SUPPORTED
        ],
        truth_condition_contradicted=truth_condition_coverage[
            ClaimCoverageState.CONTRADICTED
        ],
        truth_condition_conflicted=truth_condition_coverage[
            ClaimCoverageState.CONFLICTED
        ],
        truth_condition_unresolved=truth_condition_coverage[
            ClaimCoverageState.UNRESOLVED
        ],
        truth_condition_execution_complete=truth_condition_execution[
            ExecutionCompleteness.COMPLETE
        ],
        truth_condition_execution_partial=truth_condition_execution[
            ExecutionCompleteness.PARTIAL
        ],
        truth_condition_execution_failed=truth_condition_execution[
            ExecutionCompleteness.FAILED
        ],
        truth_condition_execution_not_run=truth_condition_execution[
            ExecutionCompleteness.NOT_RUN
        ],
        truth_condition_execution_incomplete_overlap=(
            truth_condition_execution_incomplete_overlap
        ),
        truth_condition_elementization_complete=(
            truth_condition_elementization[
                ElementizationSemanticStatus.COMPLETE
            ]
        ),
        truth_condition_elementization_incomplete=(
            truth_condition_elementization[
                ElementizationSemanticStatus.INCOMPLETE
            ]
        ),
        truth_condition_elementization_uncertain=(
            truth_condition_elementization[
                ElementizationSemanticStatus.UNCERTAIN
            ]
        ),
        truth_condition_elementization_unresolved=(
            truth_condition_elementization_unresolved
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
    truth_condition_summary = ""
    support_scope_summary = (
        f"单一域名代理支持 {summary.single_domain_proxy_support}；"
        "多个域名代理支持（不表示来源独立） "
        f"{summary.multiple_domain_proxy_support}；"
    )
    if summary.truth_condition_claims:
        support_scope_summary = (
            "单一域名代理整条断言支持 "
            f"{summary.single_domain_proxy_support}；"
            "多个域名代理各自整条断言支持（不表示来源独立） "
            f"{summary.multiple_domain_proxy_support}；"
            f"仅真值条件级支持 {summary.element_level_support}"
            "（其中跨来源分布式完整覆盖 "
            f"{summary.distributed_element_support}）；"
        )
        truth_condition_summary = (
            "真值条件覆盖："
            f"完整支持 {summary.truth_condition_fully_supported}/"
            f"{summary.truth_condition_claims}；"
            f"部分支持 {summary.truth_condition_partially_supported}；"
            f"支持与反驳并存 {summary.truth_condition_mixed}；"
            f"未支持 {summary.truth_condition_not_supported}；"
            f"整体反驳 {summary.truth_condition_contradicted}；"
            f"条件冲突 {summary.truth_condition_conflicted}；"
            f"未决 {summary.truth_condition_unresolved}；"
            "真值条件执行："
            f"完整 {summary.truth_condition_execution_complete}/"
            f"{summary.truth_condition_claims}；"
            f"部分 {summary.truth_condition_execution_partial}；"
            f"失败 {summary.truth_condition_execution_failed}；"
            f"未运行 {summary.truth_condition_execution_not_run}；"
            "真值条件拆分语义审查："
            f"完整 {summary.truth_condition_elementization_complete}/"
            f"{summary.truth_condition_claims}；"
            f"不完整 {summary.truth_condition_elementization_incomplete}；"
            f"完整性未决 {summary.truth_condition_elementization_uncertain}；"
            f"未取得语义结论 "
            f"{summary.truth_condition_elementization_unresolved}；"
        )
    return (
        "> 证据摘要："
        f"{coverage_prefix}"
        f"{claim_scope}外部可核验断言 {summary.external_claims}；"
        f"其中 {summary.claims_with_located_support} 条有至少一条"
        "可定位的支持引文；"
        f"{support_scope_summary}"
        f"经来源谱系评估的交叉支持 {summary.corroborated}；"
        f"无可定位支持引文 {summary.zero_located_support}；"
        f"{truth_condition_summary}"
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
        # No call was refused; each stage checked its own allowance and cut
        # itself short. It may have spent money and done partial work first,
        # so this must not be phrased as "never started".
        curtailed = "、".join(diagnostic.budget_curtailed_stages)
        judgement = _BUDGET_SIGNAL_LABEL.get(
            diagnostic.budget_decision_signal, "无判断"
        )
        return (
            f"> **以下阶段因预算不足提前停止：{curtailed}。** "
            f"未拒绝任何调用；这些阶段可能已完成部分工作后停下，"
            f"计划中的其余部分没有执行。{judgement}。"
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
    reader_report_style: ReaderReportStyle = (
        ReaderReportStyle.AUDIT_ANNOTATED
    ),
) -> RenderedReport:
    """Insert code-owned evidence markers without rewriting narrative text."""

    if not isinstance(canonical_draft, str):
        raise TypeError("canonical_draft must be text")
    try:
        reader_report_style = ReaderReportStyle(reader_report_style)
    except ValueError as exc:
        raise ValueError("unknown reader report style") from exc
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
        if (
            claim.citation_requirement != CitationRequirement.EXTERNAL
            and entry.state
            not in {
                ClaimEvidenceState.INTERNAL_NOT_SUPPORTED,
                ClaimEvidenceState.EVIDENCE_OBLIGATION_UNRESOLVED,
            }
        ):
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
        relation_numbers: list[tuple[_RenderableEvidence, int]] = []
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
                    element_ids=relation.element_ids,
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
                        "element_ids": tuple(
                            dict.fromkeys(
                                (*existing.element_ids, *relation.element_ids)
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
            process_limitations = _truth_condition_process_limitations(entry)
            if pieces and process_limitations:
                suffix += "〔" + "；".join(process_limitations) + "〕"
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
            or any(
                element.is_formal_supporting_evidence
                for element in relation.element_relations
            )
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
    cutoff_line = _render_cutoff_line(stop_diagnostic)
    footnote_format_line = (
        _FOOTNOTE_FORMAT_LINE
        if reader_report_style is ReaderReportStyle.AUDIT_ANNOTATED
        else _CLEAN_FOOTNOTE_FORMAT_LINE
    )
    companion_audit_lines = [summary_line]
    if cutoff_line is not None:
        companion_audit_lines.insert(0, cutoff_line)
    if evidence_status_line is not None:
        companion_audit_lines.insert(0, evidence_status_line)
    if concentration_line is not None:
        companion_audit_lines.append(concentration_line)
    companion_audit_lines.extend(
        [footnote_format_line, _EVIDENCE_LEGEND_LINE, checklist_line]
    )
    sources_markdown = _render_sources_document(
        footnotes,
        run_id=run_id,
        report_filename=report_filename,
        source_alias_by_url=source_alias_by_url,
        audit_lines=(
            companion_audit_lines
            if reader_report_style is ReaderReportStyle.CLEAN
            else ()
        ),
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
    if reader_report_style is ReaderReportStyle.AUDIT_ANNOTATED:
        header_lines = [bundle_line]
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
    else:
        # Reader prose stays clean. Only material run-level uncertainty remains
        # visible here; exhaustive counts and definitions live in sources/audit.
        header_lines = [
            line
            for line in (cutoff_line, evidence_status_line)
            if line is not None
        ]
    rendered = (
        "\n".join(header_lines) + "\n\n" + body
        if header_lines
        else body
    )
    if footnotes:
        if reader_report_style is ReaderReportStyle.AUDIT_ANNOTATED:
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
        else:
            definitions = [
                (
                    f"[^{footnote.number}]: "
                    f"[{footnote.publisher_domain_proxy}]"
                    f"[{source_alias_by_url[footnote.url]}] · "
                    "[证据摘录]"
                    f"({sources_filename}#evidence-{footnote.number})"
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
        reader_report_style=reader_report_style,
    )
    return RenderedReport(
        markdown=rendered,
        sources_markdown=sources_markdown,
        reader_report_style=reader_report_style,
        reader_render_contract=_READER_RENDER_CONTRACT[reader_report_style],
        report_filename=report_filename,
        sources_filename=sources_filename,
        audit_filename=audit_filename,
        sources_sha256=sources_sha256,
        evidence_bundle_line=bundle_line,
        bundle_validation=bundle_validation,
        evidence_status_line=evidence_status_line,
        evidence_summary_line=summary_line,
        evidence_legend_line=_EVIDENCE_LEGEND_LINE,
        footnote_format_line=footnote_format_line,
        checklist_coverage_line=checklist_line,
        domain_proxy_concentration_line=concentration_line,
        summary=summary,
        footnotes=footnotes,
        annotations=tuple(annotations),
        removed_model_footnote_definitions=definition_count,
        removed_model_footnote_markers=marker_count,
        unanchored_claim_ids=tuple(unanchored),
    )
