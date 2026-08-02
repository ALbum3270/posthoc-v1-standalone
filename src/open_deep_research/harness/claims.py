"""Post-hoc atomic-claim decomposition over a canonical report draft."""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Awaitable, Mapping, Sequence
from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from open_deep_research.harness.jsonio import loads_lenient
from open_deep_research.harness.notes import (
    NoteLocationStatus,
    QuoteFailureReason,
    locate_verification_quote,
)
from open_deep_research.harness.source_spans import (
    SourceSpanRegistry,
    build_source_span_registry,
    render_segmented_source,
    resolve_source_span,
)


class MarkdownBlockKind(str, Enum):
    """Mechanically recognized Markdown content-unit kinds."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE_ROW = "table_row"
    CODE_BLOCK = "code_block"


class CitationRequirement(str, Enum):
    """The kind of evidence a selected assertion requires."""

    EXTERNAL = "external"
    INTERNAL = "internal"
    NONE = "none"


class SourceResolution(str, Enum):
    """How a later attribution stage resolved a claim to source material."""

    DIRECT = "direct"
    INHERITED_SAME_UNIT = "inherited_same_unit"
    INHERITED_PREVIOUS_UNIT = "inherited_previous_unit"
    UNRESOLVED = "unresolved"


class BlockDisposition(str, Enum):
    """Selection outcome retained for every report content block."""

    CLAIMS_SELECTED = "claims_selected"
    NO_VERIFIABLE_CLAIMS = "no_verifiable_claims"
    SELECTION_FAILED = "selection_failed"


class ClaimNormalizationStatus(str, Enum):
    """Whether a selected claim has a mechanically valid report anchor.

    This is deliberately not a faithfulness or entailment verdict.  A located
    semantic claim can still be a bad derivation of its report surface span;
    that separate relation is represented by :class:`ClaimDerivation`.
    """

    LOCATED = "located"
    NORMALIZATION_FAILED = "normalization_failed"


class ClaimRepresentationVersion(str, Enum):
    """Version the report-surface/semantic-claim boundary explicitly."""

    LEGACY_FLAT_V1 = "legacy_flat_v1"
    LAYERED_V2 = "layered_v2"


class ClaimDerivationStatus(str, Enum):
    """Independent assessment of report-surface -> semantic-claim fidelity."""

    NOT_EVALUATED = "not_evaluated"
    ENTAILED = "entailed"
    NOT_ENTAILED = "not_entailed"
    AMBIGUOUS = "ambiguous"
    FAILED = "failed"


class MarkdownBlock(BaseModel):
    """One Markdown structure unit with stable absolute character bounds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    block_id: str
    ordinal: int = Field(ge=0)
    kind: MarkdownBlockKind
    text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    section_path: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _bounds_are_ordered(self) -> MarkdownBlock:
        if self.end_char <= self.start_char:
            raise ValueError("block end_char must be greater than start_char")
        return self


class ContextSpan(BaseModel):
    """A verbatim report span used only to resolve claim context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)

    @model_validator(mode="after")
    def _bounds_are_ordered(self) -> ContextSpan:
        if self.end_char <= self.start_char:
            raise ValueError("context span end_char must exceed start_char")
        return self


class ContextSpanProposal(BaseModel):
    """Model-proposed context text retained separately from code-owned bounds."""

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    text: str = Field(min_length=1)
    proposed_start_char: Any = Field(
        default=None,
        validation_alias="start_char",
    )
    proposed_end_char: Any = Field(
        default=None,
        validation_alias="end_char",
    )

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("context proposal text must not be blank")
        return value


class ReportSurfaceSpan(BaseModel):
    """A code-owned, byte-stable slice of the canonical report.

    Exact matching applies here, at the report-side surface boundary.  It does
    not apply to the semantic ``claim_text`` derived from this span.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    block_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    start_segment_id: str = Field(min_length=1)
    end_segment_id: str = Field(min_length=1)
    span_registry_id: str = Field(min_length=1)
    report_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    segmentation_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _surface_bounds_match_text(self) -> ReportSurfaceSpan:
        if self.end_char <= self.start_char:
            raise ValueError("report surface end_char must exceed start_char")
        if self.end_char - self.start_char != len(self.text):
            raise ValueError("report surface bounds must match text length")
        return self


class ClaimDerivation(BaseModel):
    """Audit the semantic relation without pretending it has been judged.

    Claim generation and claim-quality evaluation are intentionally separate,
    following Claimify's entailment/coverage/decontextualization evaluation
    boundary.  The live extractor therefore records ``not_evaluated``.  A
    frozen evaluation or later reviewer may create a new assessed record; the
    generator never promotes its own output to ``entailed``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ClaimDerivationStatus = ClaimDerivationStatus.NOT_EVALUATED
    evaluator: str | None = None
    rationale: str | None = None

    @model_validator(mode="after")
    def _assessment_metadata_matches_status(self) -> ClaimDerivation:
        if self.status is ClaimDerivationStatus.NOT_EVALUATED:
            if self.evaluator is not None or self.rationale is not None:
                raise ValueError(
                    "not_evaluated derivations cannot carry assessment metadata"
                )
        elif self.evaluator is None:
            raise ValueError("evaluated derivations require an evaluator")
        return self


class SelectedAssertion(BaseModel):
    """A code-resolved, exact assertion selected from one report block.

    ``selected_text`` is always the authoritative report slice retained for
    downstream decomposition.  When selection used the pointer protocol, the
    optional registry binding records the model's addressable container; the
    model never supplied ``selected_text`` itself.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_text: str = Field(min_length=1)
    citation_requirement: CitationRequirement
    selection_start_char: int | None = Field(default=None, ge=0)
    selection_end_char: int | None = Field(default=None, ge=0)
    selection_start_segment_id: str | None = None
    selection_end_segment_id: str | None = None
    selection_span_registry_id: str | None = None
    selection_report_text_sha256: str | None = None
    selection_segmentation_version: str | None = None

    @field_validator("selected_text")
    @classmethod
    def _selected_text_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("selected_text must not be blank")
        return normalized

    @model_validator(mode="after")
    def _selection_pointer_is_completely_bound(self) -> SelectedAssertion:
        pointer = (
            self.selection_start_segment_id,
            self.selection_end_segment_id,
            self.selection_span_registry_id,
            self.selection_report_text_sha256,
            self.selection_segmentation_version,
        )
        if any(value is not None for value in pointer) and not all(
            value is not None for value in pointer
        ):
            raise ValueError(
                "selection segment pointers require complete registry binding"
            )
        bounds = (self.selection_start_char, self.selection_end_char)
        if any(value is not None for value in bounds) and not all(
            value is not None for value in bounds
        ):
            raise ValueError("selection character bounds must be complete")
        if self.selection_start_char is not None:
            if not all(value is not None for value in pointer):
                raise ValueError(
                    "selection character bounds require a bound segment pointer"
                )
            assert self.selection_end_char is not None
            if self.selection_end_char <= self.selection_start_char:
                raise ValueError("selection end_char must exceed start_char")
            if (
                self.selection_end_char - self.selection_start_char
                != len(self.selected_text)
            ):
                raise ValueError(
                    "selection bounds must match selected_text length"
                )
        return self


class BlockSelection(BaseModel):
    """Auditable selection disposition for exactly one Markdown block."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    block_id: str
    disposition: BlockDisposition
    rationale: str = ""
    assertions: tuple[SelectedAssertion, ...] = ()

    @model_validator(mode="after")
    def _disposition_matches_assertions(self) -> BlockSelection:
        if (
            self.disposition == BlockDisposition.CLAIMS_SELECTED
            and not self.assertions
        ):
            raise ValueError("claims_selected requires at least one assertion")
        if (
            self.disposition != BlockDisposition.CLAIMS_SELECTED
            and self.assertions
        ):
            raise ValueError(
                "only claims_selected may contain selected assertions"
            )
        return self


class _SelectedAssertionProposal(BaseModel):
    """Model selection proposal before code owns the exact report text.

    Only the span-pointer form is accepted at runtime. Historical audits hold
    the already code-resolved :class:`SelectedAssertion` records, so replaying
    them does not require keeping a model-controlled textual-selection escape
    hatch in the live protocol.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_requirement: CitationRequirement
    start_segment_id: str = Field(min_length=1)
    end_segment_id: str = Field(min_length=1)

    @field_validator("start_segment_id", "end_segment_id")
    @classmethod
    def _segment_id_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("selection segment ID must not be blank")
        return normalized


class _BlockSelectionProposal(BaseModel):
    """Model-owned assertion proposal; disposition is code-owned.

    ``disposition`` is accepted only as a typed legacy input so historical
    scripted clients and audits remain replayable.  It is never copied into
    :class:`BlockSelection`: assertion presence mechanically derives the only
    structurally consistent disposition.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    block_id: str
    rationale: str = ""
    assertions: tuple[_SelectedAssertionProposal, ...] = ()
    proposed_disposition: BlockDisposition | None = Field(
        default=None,
        validation_alias="disposition",
        exclude=True,
    )


class AtomicClaim(BaseModel):
    """One retained verification unit and its model/report representations.

    The historical class name is retained for audit and API compatibility.  A
    live verification unit is not promised to be logically atomic: splitting
    a sentence can destroy shared scope, attribution, causality, or temporal
    relations.  Those semantic choices belong to the model and later quality
    review, while exact report-surface ownership remains code-enforced.

    The segment IDs identify the model-selected addressable *container*.
    ``anchor_text`` and its character bounds are independently narrowed by
    code to the selected assertion inside that container.  They therefore do
    not promise that the complete segment range itself is the render anchor.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    block_id: str
    representation_version: ClaimRepresentationVersion = (
        ClaimRepresentationVersion.LEGACY_FLAT_V1
    )
    report_surface: ReportSurfaceSpan | None = None
    selected_text: str
    claim_text: str | None
    derivation: ClaimDerivation | None = None
    anchor_text: str | None
    anchor_text_proposal: str | None = None
    anchor_start_segment_id: str | None = None
    anchor_end_segment_id: str | None = None
    anchor_span_registry_id: str | None = None
    anchor_report_text_sha256: str | None = None
    anchor_segmentation_version: str | None = None
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)
    context_spans: tuple[ContextSpan, ...] = ()
    context_span_proposals: tuple[ContextSpanProposal, ...] = ()
    citation_requirement: CitationRequirement
    source_resolution: SourceResolution = SourceResolution.UNRESOLVED
    normalization_status: ClaimNormalizationStatus
    normalization_failure: str | None = None

    @model_validator(mode="after")
    def _location_fields_match_status(self) -> AtomicClaim:
        if self.representation_version is ClaimRepresentationVersion.LAYERED_V2:
            if self.report_surface is None:
                raise ValueError(
                    "layered claims require a code-owned report_surface"
                )
            if self.report_surface.block_id != self.block_id:
                raise ValueError("report_surface must belong to claim block")
            if self.report_surface.text != self.selected_text:
                raise ValueError(
                    "selected_text must mirror the layered report_surface"
                )
            if self.claim_text is None and self.derivation is not None:
                raise ValueError(
                    "claims without semantic text cannot carry a derivation"
                )
            if self.claim_text is not None and self.derivation is None:
                raise ValueError(
                    "layered semantic claims require a derivation record"
                )
        pointer = (
            self.anchor_start_segment_id,
            self.anchor_end_segment_id,
            self.anchor_span_registry_id,
            self.anchor_report_text_sha256,
            self.anchor_segmentation_version,
        )
        if any(value is not None for value in pointer) and not all(
            value is not None for value in pointer
        ):
            raise ValueError(
                "anchor segment pointers require complete registry binding"
            )
        location = (self.anchor_text, self.start_char, self.end_char)
        if self.normalization_status == ClaimNormalizationStatus.LOCATED:
            if any(value is None for value in location):
                raise ValueError("located claims require anchor text and bounds")
            if self.claim_text is None:
                raise ValueError("located claims require claim_text")
            if self.end_char <= self.start_char:
                raise ValueError("claim end_char must exceed start_char")
            if self.normalization_failure is not None:
                raise ValueError("located claims cannot have a failure reason")
        else:
            if self.normalization_failure is None:
                raise ValueError(
                    "normalization_failed requires a failure reason"
                )
            if self.start_char is not None or self.end_char is not None:
                raise ValueError(
                    "normalization_failed cannot retain untrusted bounds"
                )
        return self


class ClaimStageUsage(BaseModel):
    """Measured usage for one claim-processing model stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    token_count: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)


class ClaimDecompositionSettings(BaseModel):
    """Mechanical capacity limits shared by the three claim stages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(default=8, ge=1)


class ClaimBatchRecord(BaseModel):
    """One auditable model batch and the exact inputs it failed to cover."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Literal["selection", "decontextualization", "extraction"]
    batch_number: int = Field(ge=1)
    input_ids: tuple[str, ...]
    output_ids: tuple[str, ...] = ()
    failed_input_ids: tuple[str, ...] = ()
    outcome: Literal["completed", "partial", "failed"]
    error: str | None = None
    usage: ClaimStageUsage = ClaimStageUsage()


class ClaimRegistryCoverage(BaseModel):
    """Mechanical report-block coverage of the claim registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluated_blocks: int = Field(ge=0)
    total_blocks: int = Field(ge=0)
    unassessed_blocks: int = Field(ge=0)
    unassessed_block_ids: tuple[str, ...] = ()
    is_complete: bool

    @model_validator(mode="after")
    def _counts_are_consistent(self) -> ClaimRegistryCoverage:
        if self.evaluated_blocks + self.unassessed_blocks != self.total_blocks:
            raise ValueError("block coverage counts must sum to total_blocks")
        if self.unassessed_blocks != len(self.unassessed_block_ids):
            raise ValueError(
                "unassessed_blocks must match unassessed_block_ids"
            )
        if self.is_complete != (self.unassessed_blocks == 0):
            raise ValueError("is_complete must reflect unassessed_blocks")
        return self


class ClaimDecompositionResult(BaseModel):
    """All blocks, dispositions, claims, diagnostics, and measured usage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    blocks: tuple[MarkdownBlock, ...]
    selections: tuple[BlockSelection, ...]
    claims: tuple[AtomicClaim, ...]
    registry_coverage: ClaimRegistryCoverage
    batches: tuple[ClaimBatchRecord, ...] = ()
    diagnostics: tuple[str, ...] = ()
    selection_usage: ClaimStageUsage
    decontextualization_usage: ClaimStageUsage
    extraction_usage: ClaimStageUsage
    anchor_proposal_count: int = Field(default=0, ge=0)
    anchor_copied_from_selection_count: int = Field(default=0, ge=0)
    anchor_copied_from_selection_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def _anchor_copy_metrics_are_consistent(self) -> ClaimDecompositionResult:
        if self.anchor_copied_from_selection_count > self.anchor_proposal_count:
            raise ValueError("copied anchors cannot exceed anchor proposals")
        expected = (
            self.anchor_copied_from_selection_count
            / self.anchor_proposal_count
            if self.anchor_proposal_count
            else 0.0
        )
        if abs(self.anchor_copied_from_selection_rate - expected) > 1e-12:
            raise ValueError("anchor copy rate must match its audited counts")
        return self

    @property
    def total_tokens(self) -> int:
        """Return usage across all three ordered stages."""

        return sum(
            usage.token_count
            for usage in (
                self.selection_usage,
                self.decontextualization_usage,
                self.extraction_usage,
            )
        )

    @property
    def total_cost_usd(self) -> float:
        """Return cost across all three ordered stages."""

        return sum(
            usage.cost_usd
            for usage in (
                self.selection_usage,
                self.decontextualization_usage,
                self.extraction_usage,
            )
        )


class ClaimModelClient(Protocol):
    """Injected model boundary for post-hoc claim decomposition."""

    def generate(self, prompt: str) -> Any | Awaitable[Any]:
        """Return JSON in a measured usage envelope."""


class _ModelEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    content: Any
    token_count: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)


class _DecontextualizedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    claim_text: str = Field(min_length=1)
    context_spans: tuple[ContextSpanProposal, ...] = ()

    @field_validator("context_spans", mode="before")
    @classmethod
    def _accept_verbatim_text_list(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return [
                {"text": item} if isinstance(item, str) else item
                for item in value
            ]
        return value

    @field_validator("claim_text")
    @classmethod
    def _claim_text_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("claim_text must not be blank")
        return normalized


class _ExtractedAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    start_segment_id: str = Field(min_length=1)
    end_segment_id: str = Field(min_length=1)

    @field_validator("start_segment_id", "end_segment_id")
    @classmethod
    def _segment_id_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("anchor segment IDs must not be blank")
        return normalized


_ATX_HEADING = re.compile(
    r"^[ ]{0,3}(?P<marks>#{1,6})[ \t]+(?P<title>.*?)[ \t]*#*[ \t]*$"
)
_SETEXT_HEADING = re.compile(r"^[ ]{0,3}(?P<marks>=+|-+)[ \t]*$")
_LIST_ITEM = re.compile(r"^(?P<indent>[ \t]{0,3})(?:[-+*]|\d+[.)])[ \t]+")
_FENCE = re.compile(r"^[ ]{0,3}(?P<marks>`{3,}|~{3,})")
_TABLE_DELIMITER_CELL = re.compile(r"^:?-{3,}:?$")


def _line_end_without_newline(report: str, start: int, raw: str) -> int:
    end = start + len(raw)
    while end > start and report[end - 1] in "\r\n":
        end -= 1
    return end


def _is_table_delimiter(line: str) -> bool:
    stripped = line.strip().strip("|")
    cells = [cell.strip() for cell in stripped.split("|")]
    return bool(cells) and all(
        _TABLE_DELIMITER_CELL.fullmatch(cell) is not None for cell in cells
    )


def _looks_like_table_row(line: str) -> bool:
    stripped = line.strip()
    return "|" in stripped and not stripped.startswith(("```", "~~~"))


def parse_markdown_blocks(report: str) -> tuple[MarkdownBlock, ...]:
    """Parse Markdown into bounded content units without sentence numbering."""

    if not isinstance(report, str):
        raise TypeError("report must be text")
    if not report:
        return ()

    lines = report.splitlines(keepends=True)
    starts: list[int] = []
    offset = 0
    for raw in lines:
        starts.append(offset)
        offset += len(raw)

    table_rows: set[int] = set()
    for candidate in range(len(lines) - 1):
        header = lines[candidate].rstrip("\r\n")
        delimiter = lines[candidate + 1].rstrip("\r\n")
        if (
            _looks_like_table_row(header)
            and "|" in delimiter
            and _is_table_delimiter(delimiter)
        ):
            table_rows.add(candidate)
            cursor = candidate + 2
            while cursor < len(lines):
                row = lines[cursor].rstrip("\r\n")
                if not row.strip() or not _looks_like_table_row(row):
                    break
                table_rows.add(cursor)
                cursor += 1

    blocks: list[MarkdownBlock] = []
    section_titles: list[str] = []
    index = 0

    def add_block(
        kind: MarkdownBlockKind,
        first_line: int,
        last_line: int,
        *,
        section_path: Sequence[str] | None = None,
    ) -> None:
        start = starts[first_line]
        end = _line_end_without_newline(
            report,
            starts[last_line],
            lines[last_line],
        )
        if end <= start:
            return
        blocks.append(
            MarkdownBlock(
                block_id=f"block-{len(blocks) + 1:04d}",
                ordinal=len(blocks),
                kind=kind,
                text=report[start:end],
                start_char=start,
                end_char=end,
                section_path=tuple(
                    section_titles if section_path is None else section_path
                ),
            )
        )

    while index < len(lines):
        raw = lines[index]
        line = raw.rstrip("\r\n")
        if not line.strip():
            index += 1
            continue

        atx = _ATX_HEADING.match(line)
        if atx is not None:
            level = len(atx.group("marks"))
            title = atx.group("title").strip()
            section_titles[level - 1 :] = [title]
            add_block(
                MarkdownBlockKind.HEADING,
                index,
                index,
                section_path=section_titles,
            )
            index += 1
            continue

        if index + 1 < len(lines):
            setext = _SETEXT_HEADING.match(lines[index + 1].rstrip("\r\n"))
            if setext is not None and line.strip():
                level = 1 if setext.group("marks").startswith("=") else 2
                title = line.strip()
                section_titles[level - 1 :] = [title]
                add_block(
                    MarkdownBlockKind.HEADING,
                    index,
                    index + 1,
                    section_path=section_titles,
                )
                index += 2
                continue

        fence = _FENCE.match(line)
        if fence is not None:
            marker = fence.group("marks")
            marker_char = marker[0]
            marker_length = len(marker)
            last = index
            while last + 1 < len(lines):
                last += 1
                candidate = lines[last].lstrip()
                if candidate.startswith(marker_char * marker_length):
                    break
            add_block(MarkdownBlockKind.CODE_BLOCK, index, last)
            index = last + 1
            continue

        if _is_table_delimiter(line):
            index += 1
            continue
        if index in table_rows:
            add_block(MarkdownBlockKind.TABLE_ROW, index, index)
            index += 1
            continue

        if _LIST_ITEM.match(line) is not None:
            last = index
            while last + 1 < len(lines):
                candidate = lines[last + 1].rstrip("\r\n")
                if not candidate.strip():
                    break
                if (
                    _ATX_HEADING.match(candidate)
                    or _FENCE.match(candidate)
                    or _LIST_ITEM.match(candidate)
                    or last + 1 in table_rows
                ):
                    break
                last += 1
            add_block(MarkdownBlockKind.LIST_ITEM, index, last)
            index = last + 1
            continue

        last = index
        while last + 1 < len(lines):
            candidate = lines[last + 1].rstrip("\r\n")
            if not candidate.strip():
                break
            if (
                _ATX_HEADING.match(candidate)
                or _FENCE.match(candidate)
                or _LIST_ITEM.match(candidate)
                or last + 1 in table_rows
            ):
                break
            if (
                last + 2 < len(lines)
                and _SETEXT_HEADING.match(
                    lines[last + 2].rstrip("\r\n")
                )
                is not None
            ):
                break
            last += 1
        add_block(MarkdownBlockKind.PARAGRAPH, index, last)
        index = last + 1

    return tuple(blocks)


def source_inheritance_allowed(
    blocks: Sequence[MarkdownBlock],
    *,
    source_block_id: str,
    target_block_id: str,
    resolution: SourceResolution,
) -> bool:
    """Mechanically validate a proposed source-inheritance boundary."""

    by_id = {block.block_id: block for block in blocks}
    source = by_id.get(source_block_id)
    target = by_id.get(target_block_id)
    if source is None or target is None:
        return False
    if resolution == SourceResolution.INHERITED_SAME_UNIT:
        return source.block_id == target.block_id
    if resolution != SourceResolution.INHERITED_PREVIOUS_UNIT:
        return False
    if source.kind != MarkdownBlockKind.PARAGRAPH:
        return False
    if target.kind != MarkdownBlockKind.PARAGRAPH:
        return False
    if source.section_path != target.section_path:
        return False
    return target.ordinal == source.ordinal + 1


_SELECTION_PROMPT = """\
Stage 1 of 3 — selection.

Select the smallest sufficient verification units that preserve the report's
meaning. A unit may contain more than one proposition when their shared
attribution, modality, comparison, cause, time sequence, or other relation is
part of what the report says. Split coordinated material only when each part
can be checked independently without losing or changing those truth
conditions. Preserve every entity, time, place, quantity, negation, modality,
reporting marker, and attribution qualifier that affects truth. The report
text is authoritative: do not paraphrase it, resolve a pronoun, add an omitted
subject, or copy text into the response.

Each block exposes stable addressable source segments as <S000001>. For every
assertion, select one shortest continuous start_segment_id/end_segment_id
range inside that block. Code, not the model, copies the final selected_text
from that range. If isolating a sub-clause would require rewriting, omitting
context, or stitching fragments, select the larger exact segment range rather
than reconstructing a smaller assertion. Do not return selected_text or
character offsets.

For every supplied block_id, return exactly one block entry. Put each chosen
verification unit in its assertions array. Return an empty assertions array
when there is no factual or report-internal assertion to check. Normative
advice, a rhetorical transition, or a statement of writing intent alone is
not an external factual assertion. Do not return a disposition: code mechanically derives claims_selected from a non-empty array and no_verifiable_claims from an empty array.

Classify each selected assertion independently on the orthogonal evidence
dimension:
- external: its truth depends on evidence outside this report;
- internal: it can be checked against the report artifact itself;
- none: it makes no evidence-bearing factual assertion.

Do not omit a block. Selection is independent of whether a source has already
been found. Return JSON only:
{{"blocks":[{{"block_id":"block-0001","rationale":"...",\
"assertions":[{{"start_segment_id":"S000001",\
"end_segment_id":"S000001","citation_requirement":"external|internal|none"}}]}}]}}

Purely structural examples: two unrelated dated events may be two units. By
contrast, "a witness said A caused B" may remain one unit when splitting it
would turn attributed speech into direct assertion or erase the causal link.

Markdown blocks:
{blocks}
"""

_DECONTEXTUALIZATION_PROMPT = """\
Stage 2 of 3 — decontextualization.

Turn each selected assertion into a self-contained claim_text. Resolve only
the references needed to make the assertion independently understandable,
such as a pronoun or an omitted subject. Do not add facts, remove qualifiers,
resolve uncertainty, infer a cause, calculate a new value, or otherwise make
the assertion stronger.

For every piece of surrounding report text used to resolve context, put its
exact verbatim text directly in the context_spans string array. Do not wrap
the strings in objects. Do not calculate or return character offsets; code
will require each proposed text to have one unique occurrence and will assign
its absolute start_char/end_char mechanically. If the selected assertion is
already self-contained, context_spans must be empty.

Return exactly one entry per claim_id as JSON only:
{{"claims":[{{"claim_id":"claim-0001","claim_text":"...",\
"context_spans":["..."]}}]}}

Purely structural example: in "A group opened a facility. It expanded the
facility.", the second assertion may become "The group expanded the
facility."; the first sentence is recorded as context, but no new reason,
date, or degree may be added.

Canonical report:
{report}

Selected assertions:
{claims}
"""

_EXTRACTION_PROMPT = """\
Stage 3 of 3 — extraction.

For each claim_id, identify the shortest continuous segment range in the
canonical report that contains the selected assertion. Return only its
start_segment_id and end_segment_id. The range must stay inside that claim's
supplied block_id. Do not copy anchor text or calculate character offsets. Do
not join noncontiguous fragments.

Code resolves the IDs against a registry bound to the exact report bytes,
then derives the final anchor independently from the selection-stage text
inside that range. It rejects unknown, reversed, cross-unit, wrong-block, or
non-containing ranges instead of guessing.

Return exactly one entry per claim_id as JSON only:
{{"claims":[{{"claim_id":"claim-0001",\
"start_segment_id":"S000001","end_segment_id":"S000001"}}]}}

Canonical report with addressable segments:
{report}

Decontextualized claims:
{claims}
"""


def _render_selection_block(
    report: str,
    *,
    block: MarkdownBlock,
    registry: SourceSpanRegistry,
) -> Mapping[str, Any]:
    """Render one block with only code-owned, in-block segment addresses."""

    contained = tuple(
        segment
        for segment in registry.segments
        if (
            block.start_char <= segment.start_char
            and segment.end_char <= block.end_char
        )
    )
    overlapping = tuple(
        segment
        for segment in registry.segments
        if segment.start_char < block.end_char and block.start_char < segment.end_char
    )
    if not contained or contained != overlapping:
        raise ValueError(
            "selection block must contain complete addressable source segments: "
            f"{block.block_id}"
        )

    pieces: list[str] = []
    cursor = block.start_char
    for segment in contained:
        pieces.append(report[cursor : segment.start_char])
        pieces.append(f"<{segment.segment_id}>")
        pieces.append(report[segment.start_char : segment.end_char])
        cursor = segment.end_char
    pieces.append(report[cursor : block.end_char])
    return {
        "block_id": block.block_id,
        "kind": block.kind.value,
        "section_path": list(block.section_path),
        "addressable_text": "".join(pieces),
    }


def build_selection_prompt(
    report: str,
    blocks: Sequence[MarkdownBlock],
    *,
    span_registry: SourceSpanRegistry | None = None,
) -> str:
    """Build the pointer-only, code-grounded block-selection prompt."""

    registry = span_registry or build_source_span_registry(report)
    payload = [
        _render_selection_block(report, block=block, registry=registry)
        for block in blocks
    ]
    return _SELECTION_PROMPT.format(
        blocks=json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def build_decontextualization_prompt(
    report: str,
    claims: Sequence[Mapping[str, Any]],
) -> str:
    """Build the claim decontextualization prompt."""

    return _DECONTEXTUALIZATION_PROMPT.format(
        report=report,
        claims=json.dumps(list(claims), ensure_ascii=False, sort_keys=True),
    )


def build_extraction_prompt(
    report: str,
    claims: Sequence[Mapping[str, Any]],
    *,
    span_registry: SourceSpanRegistry | None = None,
) -> str:
    """Build the code-resolved anchor-pointer extraction prompt."""

    registry = span_registry or build_source_span_registry(report)
    return _EXTRACTION_PROMPT.format(
        report=render_segmented_source(report, registry),
        claims=json.dumps(list(claims), ensure_ascii=False, sort_keys=True),
    )


async def _call_model(
    client: ClaimModelClient,
    prompt: str,
) -> tuple[Any, ClaimStageUsage]:
    response = client.generate(prompt)
    if inspect.isawaitable(response):
        response = await response
    try:
        envelope = _ModelEnvelope.model_validate(response)
    except ValidationError as exc:
        raise ValueError(
            "claim model returned an invalid usage envelope"
        ) from exc
    content = envelope.content
    if isinstance(content, str):
        content = loads_lenient(content)
    return content, ClaimStageUsage(
        token_count=envelope.token_count,
        cost_usd=envelope.cost_usd,
    )


def _selection_for_every_block(
    blocks: Sequence[MarkdownBlock],
    content: Any,
    *,
    report: str,
    span_registry: SourceSpanRegistry,
) -> tuple[tuple[BlockSelection, ...], tuple[str, ...]]:
    if not isinstance(content, Mapping) or not isinstance(
        content.get("blocks"), (list, tuple)
    ):
        failed = tuple(
            BlockSelection(
                block_id=block.block_id,
                disposition=BlockDisposition.SELECTION_FAILED,
                rationale="selection payload could not be validated",
            )
            for block in blocks
        )
        return failed, ("selection_payload_invalid",)

    block_by_id = {block.block_id: block for block in blocks}
    expected = set(block_by_id)
    grouped: dict[str, list[BlockSelection]] = {}
    diagnostics: list[str] = []
    for index, raw_selection in enumerate(content["blocks"]):
        try:
            proposal = _BlockSelectionProposal.model_validate(raw_selection)
        except (TypeError, ValidationError, ValueError) as exc:
            diagnostics.append(
                f"selection_entry_invalid[{index}]: {exc}"
            )
            continue
        if proposal.block_id not in expected:
            diagnostics.append(
                f"selection_unknown_block: {proposal.block_id}"
            )
            continue
        block = block_by_id[proposal.block_id]
        assertions: list[SelectedAssertion] = []
        try:
            for assertion in proposal.assertions:
                resolved = resolve_source_span(
                    report,
                    span_registry,
                    start_segment_id=assertion.start_segment_id,
                    end_segment_id=assertion.end_segment_id,
                )
                if not (
                    block.start_char <= resolved.start_char
                    and resolved.end_char <= block.end_char
                ):
                    raise ValueError(
                        "selection segment range leaves its declared block"
                    )
                assertions.append(
                    SelectedAssertion(
                        selected_text=resolved.source_quote,
                        citation_requirement=assertion.citation_requirement,
                        selection_start_char=resolved.start_char,
                        selection_end_char=resolved.end_char,
                        selection_start_segment_id=resolved.start_segment_id,
                        selection_end_segment_id=resolved.end_segment_id,
                        selection_span_registry_id=span_registry.registry_id,
                        selection_report_text_sha256=(
                            span_registry.source_text_sha256
                        ),
                        selection_segmentation_version=(
                            span_registry.segmentation_version
                        ),
                    )
                )
        except ValueError as exc:
            diagnostics.append(
                f"selection_entry_invalid[{index}]: {exc}"
            )
            continue

        derived_disposition = (
            BlockDisposition.CLAIMS_SELECTED
            if assertions
            else BlockDisposition.NO_VERIFIABLE_CLAIMS
        )
        selection = BlockSelection(
            block_id=proposal.block_id,
            disposition=derived_disposition,
            rationale=proposal.rationale,
            assertions=tuple(assertions),
        )
        if (
            proposal.proposed_disposition is not None
            and proposal.proposed_disposition is not derived_disposition
        ):
            diagnostics.append(
                "selection_legacy_disposition_ignored"
                f"[{index}]: proposed="
                f"{proposal.proposed_disposition.value}, derived="
                f"{derived_disposition.value}"
            )
        grouped.setdefault(selection.block_id, []).append(selection)

    ordered: list[BlockSelection] = []
    for block in blocks:
        candidates = grouped.get(block.block_id, ())
        if len(candidates) == 1:
            ordered.append(candidates[0])
            continue
        reason = (
            "selection omitted this block"
            if not candidates
            else "selection returned this block more than once"
        )
        diagnostics.append(f"{reason}: {block.block_id}")
        ordered.append(
            BlockSelection(
                block_id=block.block_id,
                disposition=BlockDisposition.SELECTION_FAILED,
                rationale=reason,
            )
        )
    return tuple(ordered), tuple(diagnostics)


def _claim_seed(
    selections: Sequence[BlockSelection],
) -> tuple[dict[str, Any], ...]:
    claims: list[dict[str, Any]] = []
    for selection in selections:
        for assertion in selection.assertions:
            claims.append(
                {
                    "claim_id": f"claim-{len(claims) + 1:04d}",
                    "block_id": selection.block_id,
                    "selected_text": assertion.selected_text,
                    "selection_start_char": assertion.selection_start_char,
                    "selection_end_char": assertion.selection_end_char,
                    "selection_start_segment_id": (
                        assertion.selection_start_segment_id
                    ),
                    "selection_end_segment_id": (
                        assertion.selection_end_segment_id
                    ),
                    "selection_span_registry_id": (
                        assertion.selection_span_registry_id
                    ),
                    "selection_report_text_sha256": (
                        assertion.selection_report_text_sha256
                    ),
                    "selection_segmentation_version": (
                        assertion.selection_segmentation_version
                    ),
                    "citation_requirement": (
                        assertion.citation_requirement.value
                    ),
                }
            )
    return tuple(claims)


def _chunks(
    values: Sequence[Any],
    size: int,
) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        tuple(values[start : start + size])
        for start in range(0, len(values), size)
    )


def _sum_usage(
    usages: Sequence[ClaimStageUsage],
) -> ClaimStageUsage:
    return ClaimStageUsage(
        token_count=sum(usage.token_count for usage in usages),
        cost_usd=sum(usage.cost_usd for usage in usages),
    )


def _batch_outcome(
    input_ids: Sequence[str],
    failed_input_ids: Sequence[str],
) -> Literal["completed", "partial", "failed"]:
    if not failed_input_ids:
        return "completed"
    if len(failed_input_ids) == len(input_ids):
        return "failed"
    return "partial"


def _registry_coverage(
    blocks: Sequence[MarkdownBlock],
    selections: Sequence[BlockSelection],
) -> ClaimRegistryCoverage:
    disposition_by_id = {
        selection.block_id: selection.disposition
        for selection in selections
    }
    unassessed = tuple(
        block.block_id
        for block in blocks
        if disposition_by_id.get(block.block_id)
        in {None, BlockDisposition.SELECTION_FAILED}
    )
    return ClaimRegistryCoverage(
        evaluated_blocks=len(blocks) - len(unassessed),
        total_blocks=len(blocks),
        unassessed_blocks=len(unassessed),
        unassessed_block_ids=unassessed,
        is_complete=not unassessed,
    )


def _locate_context_spans(
    report: str,
    proposals: Sequence[ContextSpanProposal],
    blocks: Sequence[MarkdownBlock],
    *,
    target_block: MarkdownBlock,
) -> tuple[tuple[ContextSpan, ...], str | None]:
    spans: list[ContextSpan] = []
    target_text = report[target_block.start_char : target_block.end_char]
    for proposal in proposals:
        # The selection stage already binds this claim to one Markdown block.
        # Prefer that mechanical boundary before asking whether a short name or
        # date is unique across the whole report.  This prevents a repeated
        # entity in another section from making local pronoun resolution
        # impossible, without guessing when the same text is repeated inside
        # the actual claim block.
        local_occurrences = _unique_occurrences(target_text, proposal.text)
        if len(local_occurrences) > 1:
            return (), "context_span_not_unique"
        if local_occurrences:
            start = target_block.start_char + local_occurrences[0]
            end = start + len(proposal.text)
            source_text = proposal.text
            spans.append(
                ContextSpan(text=source_text, start_char=start, end_char=end)
            )
            continue

        local_repaired = locate_verification_quote(target_text, proposal.text)
        if (
            local_repaired.location_status
            == NoteLocationStatus.REPAIRED_LOCATABLE
            and local_repaired.span is not None
            and local_repaired.source_quote is not None
        ):
            start = target_block.start_char + local_repaired.span.start_char
            end = target_block.start_char + local_repaired.span.end_char
            spans.append(
                ContextSpan(
                    text=local_repaired.source_quote,
                    start_char=start,
                    end_char=end,
                )
            )
            continue
        if (
            local_repaired.failure_reason
            == QuoteFailureReason.AMBIGUOUS_FORMAT_MATCH
        ):
            return (), "context_span_not_unique"

        # A decontextualization may legitimately use the preceding narrative
        # unit. Preserve the old whole-report path only when the target block
        # contains no exact or conservatively repaired match.
        occurrences = _unique_occurrences(report, proposal.text)
        if len(occurrences) > 1:
            return (), "context_span_not_unique"
        if occurrences:
            start = occurrences[0]
            end = start + len(proposal.text)
            source_text = proposal.text
        else:
            repaired = locate_verification_quote(report, proposal.text)
            if (
                repaired.location_status
                != NoteLocationStatus.REPAIRED_LOCATABLE
                or repaired.span is None
                or repaired.source_quote is None
            ):
                reason = (
                    "context_span_not_unique"
                    if repaired.failure_reason
                    == QuoteFailureReason.AMBIGUOUS_FORMAT_MATCH
                    else "context_span_not_verbatim"
                )
                return (), reason
            start = repaired.span.start_char
            end = repaired.span.end_char
            source_text = repaired.source_quote
            if not any(
                block.start_char <= start and end <= block.end_char
                for block in blocks
            ):
                return (), "context_span_repair_crosses_markdown_unit"
        spans.append(
            ContextSpan(
                text=source_text,
                start_char=start,
                end_char=end,
            )
        )
    return tuple(spans), None


def _unique_occurrences(text: str, needle: str) -> tuple[int, ...]:
    starts: list[int] = []
    cursor = 0
    while True:
        position = text.find(needle, cursor)
        if position < 0:
            return tuple(starts)
        starts.append(position)
        cursor = position + 1


def _normalization_failure(
    *,
    claim: Mapping[str, Any],
    claim_text: str | None,
    context_spans: Sequence[ContextSpan] = (),
    context_span_proposals: Sequence[ContextSpanProposal] = (),
    anchor_text_proposal: str | None = None,
    anchor_text: str | None = None,
    anchor_start_segment_id: str | None = None,
    anchor_end_segment_id: str | None = None,
    anchor_span_registry_id: str | None = None,
    anchor_report_text_sha256: str | None = None,
    anchor_segmentation_version: str | None = None,
    reason: str,
) -> AtomicClaim:
    report_surface = _report_surface_from_claim_seed(claim)
    return AtomicClaim(
        claim_id=str(claim["claim_id"]),
        block_id=str(claim["block_id"]),
        representation_version=ClaimRepresentationVersion.LAYERED_V2,
        report_surface=report_surface,
        selected_text=str(claim["selected_text"]),
        claim_text=claim_text,
        derivation=(ClaimDerivation() if claim_text is not None else None),
        anchor_text_proposal=anchor_text_proposal,
        anchor_text=anchor_text,
        anchor_start_segment_id=anchor_start_segment_id,
        anchor_end_segment_id=anchor_end_segment_id,
        anchor_span_registry_id=anchor_span_registry_id,
        anchor_report_text_sha256=anchor_report_text_sha256,
        anchor_segmentation_version=anchor_segmentation_version,
        context_spans=tuple(context_spans),
        context_span_proposals=tuple(context_span_proposals),
        citation_requirement=CitationRequirement(
            claim["citation_requirement"]
        ),
        source_resolution=SourceResolution.UNRESOLVED,
        normalization_status=ClaimNormalizationStatus.NORMALIZATION_FAILED,
        normalization_failure=reason,
    )


def _report_surface_from_claim_seed(
    claim: Mapping[str, Any],
) -> ReportSurfaceSpan:
    """Build the report-side layer from a live pointer-resolved selection."""

    required = {
        "selection_start_char": claim.get("selection_start_char"),
        "selection_end_char": claim.get("selection_end_char"),
        "selection_start_segment_id": claim.get("selection_start_segment_id"),
        "selection_end_segment_id": claim.get("selection_end_segment_id"),
        "selection_span_registry_id": claim.get("selection_span_registry_id"),
        "selection_report_text_sha256": claim.get(
            "selection_report_text_sha256"
        ),
        "selection_segmentation_version": claim.get(
            "selection_segmentation_version"
        ),
    }
    missing = tuple(name for name, value in required.items() if value is None)
    if missing:
        raise AssertionError(
            "live claim seed lacks code-owned report surface fields: "
            + ", ".join(missing)
        )
    return ReportSurfaceSpan(
        block_id=str(claim["block_id"]),
        text=str(claim["selected_text"]),
        start_char=int(required["selection_start_char"]),
        end_char=int(required["selection_end_char"]),
        start_segment_id=str(required["selection_start_segment_id"]),
        end_segment_id=str(required["selection_end_segment_id"]),
        span_registry_id=str(required["selection_span_registry_id"]),
        report_text_sha256=str(required["selection_report_text_sha256"]),
        segmentation_version=str(required["selection_segmentation_version"]),
    )


def _selected_assertion_bounds(
    report: str,
    *,
    block: MarkdownBlock,
    selected_text: str,
    span_registry: SourceSpanRegistry,
    selection_start_segment_id: str | None = None,
    selection_end_segment_id: str | None = None,
    selection_span_registry_id: str | None = None,
    selection_report_text_sha256: str | None = None,
    selection_segmentation_version: str | None = None,
) -> tuple[int | None, int | None, str | None]:
    """Resolve the selection-stage assertion within its one declared block.

    Pointer-based selection already identifies an exact code-owned source
    range, including when the same sentence occurs twice in a block.  Legacy
    historical selections have only text and therefore retain the conservative
    exact-in-block lookup below.  Extraction merely selects a containing
    addressable range; it cannot widen the final render anchor.
    """

    selection_pointer = (
        selection_start_segment_id,
        selection_end_segment_id,
        selection_span_registry_id,
        selection_report_text_sha256,
        selection_segmentation_version,
    )
    if any(value is not None for value in selection_pointer):
        if not all(value is not None for value in selection_pointer):
            raise AssertionError(
                "selection pointer binding must be complete in a claim seed"
            )
        if (
            selection_span_registry_id != span_registry.registry_id
            or selection_report_text_sha256 != span_registry.source_text_sha256
            or selection_segmentation_version
            != span_registry.segmentation_version
        ):
            return None, None, "selection_pointer_registry_mismatch"
        try:
            resolved = resolve_source_span(
                report,
                span_registry,
                start_segment_id=str(selection_start_segment_id),
                end_segment_id=str(selection_end_segment_id),
            )
        except ValueError:
            return None, None, "selection_pointer_invalid"
        if resolved.source_quote != selected_text:
            raise AssertionError(
                "code-resolved selection text must equal its authoritative span"
            )
        if not (
            block.start_char <= resolved.start_char
            and resolved.end_char <= block.end_char
        ):
            return None, None, "selection_pointer_outside_declared_block"
        return resolved.start_char, resolved.end_char, None

    block_text = report[block.start_char : block.end_char]
    occurrences = _unique_occurrences(block_text, selected_text)
    if len(occurrences) == 1:
        start = block.start_char + occurrences[0]
        return start, start + len(selected_text), None
    if len(occurrences) > 1:
        return None, None, "selected_assertion_not_unique_in_block"

    # Selection should be verbatim, but it may omit Markdown-only formatting
    # such as ``**`` around a date.  Reuse the established conservative
    # formatter repair only inside the declared block; it must still resolve
    # to one contiguous authoritative report slice.  A semantic rewrite does
    # not pass this path and remains an explicit normalization failure.
    repaired = locate_verification_quote(block_text, selected_text)
    if (
        repaired.location_status == NoteLocationStatus.REPAIRED_LOCATABLE
        and repaired.span is not None
    ):
        start = block.start_char + repaired.span.start_char
        end = block.start_char + repaired.span.end_char
        return start, end, None
    if repaired.failure_reason == QuoteFailureReason.AMBIGUOUS_FORMAT_MATCH:
        return None, None, "selected_assertion_not_unique_in_block"
    return None, None, "selected_assertion_not_verbatim_in_block"


async def decompose_claims(
    report: str,
    *,
    model_client: ClaimModelClient,
    settings: ClaimDecompositionSettings | None = None,
) -> ClaimDecompositionResult:
    """Run three capacity-bounded stages without dropping an input silently."""

    if not isinstance(report, str):
        raise TypeError("report must be text")
    if not report.strip():
        raise ValueError("report must not be blank")

    active_settings = settings or ClaimDecompositionSettings()
    blocks = parse_markdown_blocks(report)
    block_by_id = {block.block_id: block for block in blocks}
    report_span_registry = build_source_span_registry(report)
    diagnostics: list[str] = []
    batch_records: list[ClaimBatchRecord] = []
    selection_usages: list[ClaimStageUsage] = []
    ordered_selections: list[BlockSelection] = []
    for batch_number, block_batch in enumerate(
        _chunks(blocks, active_settings.batch_size),
        start=1,
    ):
        input_ids = tuple(block.block_id for block in block_batch)
        call_error: str | None = None
        try:
            selection_content, batch_usage = await _call_model(
                model_client,
                build_selection_prompt(
                    report,
                    block_batch,
                    span_registry=report_span_registry,
                ),
            )
        except Exception as exc:
            selection_content = None
            batch_usage = ClaimStageUsage()
            call_error = f"{type(exc).__name__}: {exc}"
            diagnostics.append(
                f"selection_batch_error[{batch_number}] "
                f"block_ids={','.join(input_ids)}: {call_error}"
            )
        batch_selections, batch_diagnostics = _selection_for_every_block(
            block_batch,
            selection_content,
            report=report,
            span_registry=report_span_registry,
        )
        diagnostics.extend(batch_diagnostics)
        failed_ids = tuple(
            selection.block_id
            for selection in batch_selections
            if selection.disposition == BlockDisposition.SELECTION_FAILED
        )
        output_ids = tuple(
            selection.block_id
            for selection in batch_selections
            if selection.disposition != BlockDisposition.SELECTION_FAILED
        )
        ordered_selections.extend(batch_selections)
        selection_usages.append(batch_usage)
        batch_records.append(
            ClaimBatchRecord(
                stage="selection",
                batch_number=batch_number,
                input_ids=input_ids,
                output_ids=output_ids,
                failed_input_ids=failed_ids,
                outcome=_batch_outcome(input_ids, failed_ids),
                error=call_error,
                usage=batch_usage,
            )
        )

    selections = tuple(ordered_selections)
    selection_usage = _sum_usage(selection_usages)
    registry_coverage = _registry_coverage(blocks, selections)
    seed_claims = _claim_seed(selections)
    zero_usage = ClaimStageUsage()
    if not seed_claims:
        return ClaimDecompositionResult(
            blocks=blocks,
            selections=selections,
            claims=(),
            registry_coverage=registry_coverage,
            batches=tuple(batch_records),
            diagnostics=tuple(diagnostics),
            selection_usage=selection_usage,
            decontextualization_usage=zero_usage,
            extraction_usage=zero_usage,
        )

    valid_decontext: dict[
        str,
        tuple[
            str,
            tuple[ContextSpan, ...],
            tuple[ContextSpanProposal, ...],
        ],
    ] = {}
    failures: dict[str, AtomicClaim] = {}
    decontext_usages: list[ClaimStageUsage] = []
    for batch_number, claim_batch in enumerate(
        _chunks(seed_claims, active_settings.batch_size),
        start=1,
    ):
        input_ids = tuple(str(claim["claim_id"]) for claim in claim_batch)
        expected_ids = set(input_ids)
        call_error: str | None = None
        try:
            decontext_content, batch_usage = await _call_model(
                model_client,
                build_decontextualization_prompt(report, claim_batch),
            )
        except Exception as exc:
            decontext_content = None
            batch_usage = ClaimStageUsage()
            call_error = f"{type(exc).__name__}: {exc}"
            diagnostics.append(
                f"decontextualization_batch_error[{batch_number}] "
                f"claim_ids={','.join(input_ids)}: {call_error}"
            )
        decontext_usages.append(batch_usage)
        decontext_by_id: dict[str, _DecontextualizedClaim] = {}
        duplicate_ids: set[str] = set()
        invalid_ids: set[str] = set()
        raw_decontext = (
            decontext_content.get("claims")
            if isinstance(decontext_content, Mapping)
            else None
        )
        if not isinstance(raw_decontext, (list, tuple)):
            diagnostics.append("decontextualization_payload_invalid")
            raw_decontext = ()
        for index, raw_claim in enumerate(raw_decontext):
            raw_claim_id = (
                raw_claim.get("claim_id")
                if isinstance(raw_claim, Mapping)
                else None
            )
            try:
                decontext = _DecontextualizedClaim.model_validate(raw_claim)
            except (TypeError, ValidationError, ValueError) as exc:
                diagnostics.append(
                    f"decontextualization_entry_invalid[{index}]: {exc}"
                )
                if isinstance(raw_claim_id, str):
                    invalid_ids.add(raw_claim_id)
                continue
            if decontext.claim_id not in expected_ids:
                diagnostics.append(
                    f"decontextualization_unknown_claim: "
                    f"{decontext.claim_id}"
                )
                continue
            if decontext.claim_id in decontext_by_id:
                duplicate_ids.add(decontext.claim_id)
                decontext_by_id.pop(decontext.claim_id, None)
                diagnostics.append(
                    f"decontextualization_duplicate_claim: "
                    f"{decontext.claim_id}"
                )
                continue
            if decontext.claim_id in duplicate_ids:
                continue
            decontext_by_id[decontext.claim_id] = decontext

        failed_ids: list[str] = []
        output_ids: list[str] = []
        for claim in claim_batch:
            claim_id = str(claim["claim_id"])
            decontext = decontext_by_id.get(claim_id)
            if decontext is None:
                failure_reason = (
                    "decontextualization_duplicate"
                    if claim_id in duplicate_ids
                    else (
                        "decontextualization_invalid"
                        if claim_id in invalid_ids
                        else "decontextualization_missing"
                    )
                )
                diagnostics.append(f"{failure_reason}: {claim_id}")
                failures[claim_id] = _normalization_failure(
                    claim=claim,
                    claim_text=None,
                    reason=failure_reason,
                )
                failed_ids.append(claim_id)
                continue
            valid_spans, context_error = _locate_context_spans(
                report,
                decontext.context_spans,
                blocks,
                target_block=block_by_id[str(claim["block_id"])],
            )
            if context_error is not None:
                diagnostics.append(f"{context_error}: {claim_id}")
                failures[claim_id] = _normalization_failure(
                    claim=claim,
                    claim_text=decontext.claim_text,
                    context_span_proposals=decontext.context_spans,
                    reason=context_error,
                )
                failed_ids.append(claim_id)
                continue
            valid_decontext[claim_id] = (
                decontext.claim_text,
                valid_spans,
                decontext.context_spans,
            )
            output_ids.append(claim_id)
        batch_records.append(
            ClaimBatchRecord(
                stage="decontextualization",
                batch_number=batch_number,
                input_ids=input_ids,
                output_ids=tuple(output_ids),
                failed_input_ids=tuple(failed_ids),
                outcome=_batch_outcome(input_ids, failed_ids),
                error=call_error,
                usage=batch_usage,
            )
        )

    decontext_usage = _sum_usage(decontext_usages)
    extraction_usages: list[ClaimStageUsage] = []
    extraction_by_id: dict[str, _ExtractedAnchor] = {}
    duplicate_extraction_ids: set[str] = set()
    invalid_extraction_ids: set[str] = set()
    if valid_decontext:
        extraction_input = [
            {
                **claim,
                "claim_text": valid_decontext[str(claim["claim_id"])][0],
            }
            for claim in seed_claims
            if str(claim["claim_id"]) in valid_decontext
        ]
        for batch_number, extraction_batch in enumerate(
            _chunks(extraction_input, active_settings.batch_size),
            start=1,
        ):
            input_ids = tuple(
                str(claim["claim_id"]) for claim in extraction_batch
            )
            try:
                extraction_content, batch_usage = await _call_model(
                    model_client,
                    build_extraction_prompt(
                        report,
                        extraction_batch,
                        span_registry=report_span_registry,
                    ),
                )
                call_error = None
            except Exception as exc:
                extraction_content = None
                batch_usage = ClaimStageUsage()
                call_error = f"{type(exc).__name__}: {exc}"
                diagnostics.append(
                    f"extraction_batch_error[{batch_number}] "
                    f"claim_ids={','.join(input_ids)}: {call_error}"
                )
            extraction_usages.append(batch_usage)
            expected_ids = set(input_ids)
            batch_by_id: dict[str, _ExtractedAnchor] = {}
            batch_duplicate_ids: set[str] = set()
            batch_invalid_ids: set[str] = set()
            raw_extraction = (
                extraction_content.get("claims")
                if isinstance(extraction_content, Mapping)
                else None
            )
            if not isinstance(raw_extraction, (list, tuple)):
                diagnostics.append("extraction_payload_invalid")
                raw_extraction = ()
            for index, raw_claim in enumerate(raw_extraction):
                raw_claim_id = (
                    raw_claim.get("claim_id")
                    if isinstance(raw_claim, Mapping)
                    else None
                )
                try:
                    extraction = _ExtractedAnchor.model_validate(raw_claim)
                except (TypeError, ValidationError, ValueError) as exc:
                    diagnostics.append(
                        f"extraction_entry_invalid[{index}]: {exc}"
                    )
                    if isinstance(raw_claim_id, str):
                        batch_invalid_ids.add(raw_claim_id)
                    continue
                if extraction.claim_id not in expected_ids:
                    diagnostics.append(
                        f"extraction_unknown_claim: {extraction.claim_id}"
                    )
                    continue
                if extraction.claim_id in batch_by_id:
                    batch_duplicate_ids.add(extraction.claim_id)
                    batch_by_id.pop(extraction.claim_id, None)
                    diagnostics.append(
                        f"extraction_duplicate_claim: "
                        f"{extraction.claim_id}"
                    )
                    continue
                if extraction.claim_id in batch_duplicate_ids:
                    continue
                batch_by_id[extraction.claim_id] = extraction

            failed_ids = tuple(
                claim_id
                for claim_id in input_ids
                if claim_id not in batch_by_id
            )
            batch_records.append(
                ClaimBatchRecord(
                    stage="extraction",
                    batch_number=batch_number,
                    input_ids=input_ids,
                    output_ids=tuple(
                        claim_id
                        for claim_id in input_ids
                        if claim_id in batch_by_id
                    ),
                    failed_input_ids=failed_ids,
                    outcome=_batch_outcome(input_ids, failed_ids),
                    error=call_error,
                    usage=batch_usage,
                )
            )
            extraction_by_id.update(batch_by_id)
            duplicate_extraction_ids.update(batch_duplicate_ids)
            invalid_extraction_ids.update(batch_invalid_ids)

    extraction_usage = _sum_usage(extraction_usages)

    claims: list[AtomicClaim] = []
    for seed in seed_claims:
        claim_id = str(seed["claim_id"])
        if claim_id in failures:
            claims.append(failures[claim_id])
            continue
        claim_text, context_spans, context_span_proposals = (
            valid_decontext[claim_id]
        )
        extraction = extraction_by_id.get(claim_id)
        if extraction is None:
            failure_reason = (
                "extraction_duplicate"
                if claim_id in duplicate_extraction_ids
                else (
                    "extraction_invalid"
                    if claim_id in invalid_extraction_ids
                    else "extraction_missing"
                )
            )
            diagnostics.append(f"{failure_reason}: {claim_id}")
            claims.append(
                _normalization_failure(
                    claim=seed,
                    claim_text=claim_text,
                    context_spans=context_spans,
                    context_span_proposals=context_span_proposals,
                    reason=failure_reason,
                )
            )
            continue

        block = block_by_id[str(seed["block_id"])]
        reason: str | None = None
        selected_start, selected_end, selected_error = _selected_assertion_bounds(
            report,
            block=block,
            selected_text=str(seed["selected_text"]),
            span_registry=report_span_registry,
            selection_start_segment_id=seed.get("selection_start_segment_id"),
            selection_end_segment_id=seed.get("selection_end_segment_id"),
            selection_span_registry_id=seed.get("selection_span_registry_id"),
            selection_report_text_sha256=seed.get(
                "selection_report_text_sha256"
            ),
            selection_segmentation_version=seed.get(
                "selection_segmentation_version"
            ),
        )
        if selected_error is not None:
            reason = selected_error
        anchor_start: int | None = None
        anchor_end: int | None = None
        anchor_text: str | None = None
        if reason is None:
            try:
                resolved_anchor = resolve_source_span(
                    report,
                    report_span_registry,
                    start_segment_id=extraction.start_segment_id,
                    end_segment_id=extraction.end_segment_id,
                )
            except ValueError:
                reason = "anchor_pointer_invalid"
            else:
                if (
                    selected_start is None
                    or selected_end is None
                    or selected_start < resolved_anchor.start_char
                    or selected_end > resolved_anchor.end_char
                ):
                    reason = "anchor_does_not_cover_selected_assertion"
                else:
                    # The model points to an addressable container.  Code owns
                    # the final exact bounds and refuses to inherit unrelated
                    # neighbouring assertions from that container.
                    anchor_start = selected_start
                    anchor_end = selected_end
                    anchor_text = report[anchor_start:anchor_end]
        if reason is None and anchor_start is not None and anchor_end is not None:
            if report[anchor_start:anchor_end] != anchor_text:
                raise AssertionError("code-owned anchor bounds must round-trip")
            if not (
                block.start_char <= anchor_start
                and anchor_end <= block.end_char
            ):
                reason = "anchor_outside_selected_block"

        if reason is not None:
            diagnostics.append(f"{reason}: {claim_id}")
            claims.append(
                _normalization_failure(
                    claim=seed,
                    claim_text=claim_text,
                    context_spans=context_spans,
                    context_span_proposals=context_span_proposals,
                    anchor_text=anchor_text,
                    anchor_start_segment_id=extraction.start_segment_id,
                    anchor_end_segment_id=extraction.end_segment_id,
                    anchor_span_registry_id=report_span_registry.registry_id,
                    anchor_report_text_sha256=(
                        report_span_registry.source_text_sha256
                    ),
                    anchor_segmentation_version=(
                        report_span_registry.segmentation_version
                    ),
                    reason=reason,
                )
            )
            continue
        if anchor_start is None or anchor_end is None:
            raise AssertionError("located anchor requires code-owned bounds")
        claims.append(
            AtomicClaim(
                claim_id=claim_id,
                block_id=str(seed["block_id"]),
                representation_version=ClaimRepresentationVersion.LAYERED_V2,
                report_surface=_report_surface_from_claim_seed(seed),
                selected_text=str(seed["selected_text"]),
                claim_text=claim_text,
                derivation=ClaimDerivation(),
                anchor_text_proposal=None,
                anchor_text=anchor_text,
                anchor_start_segment_id=extraction.start_segment_id,
                anchor_end_segment_id=extraction.end_segment_id,
                anchor_span_registry_id=report_span_registry.registry_id,
                anchor_report_text_sha256=(
                    report_span_registry.source_text_sha256
                ),
                anchor_segmentation_version=(
                    report_span_registry.segmentation_version
                ),
                start_char=anchor_start,
                end_char=anchor_end,
                context_spans=context_spans,
                context_span_proposals=context_span_proposals,
                citation_requirement=CitationRequirement(
                    seed["citation_requirement"]
                ),
                source_resolution=SourceResolution.UNRESOLVED,
                normalization_status=ClaimNormalizationStatus.LOCATED,
            )
        )

    anchor_proposal_count = len(extraction_by_id)
    # Pointer extraction never asks the model to copy selected_text. Keep the
    # historical audit metric at its mechanically true value: zero copies.
    anchor_copied_from_selection_count = 0
    anchor_copied_from_selection_rate = 0.0
    return ClaimDecompositionResult(
        blocks=blocks,
        selections=selections,
        claims=tuple(claims),
        registry_coverage=registry_coverage,
        batches=tuple(batch_records),
        diagnostics=tuple(diagnostics),
        selection_usage=selection_usage,
        decontextualization_usage=decontext_usage,
        extraction_usage=extraction_usage,
        anchor_proposal_count=anchor_proposal_count,
        anchor_copied_from_selection_count=(
            anchor_copied_from_selection_count
        ),
        anchor_copied_from_selection_rate=anchor_copied_from_selection_rate,
    )
