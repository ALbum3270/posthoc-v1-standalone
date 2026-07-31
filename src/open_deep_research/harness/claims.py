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
    """Whether a selected claim has a mechanically valid report anchor."""

    LOCATED = "located"
    NORMALIZATION_FAILED = "normalization_failed"


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


class SelectedAssertion(BaseModel):
    """An atomic assertion selected from one report block."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_text: str = Field(min_length=1)
    citation_requirement: CitationRequirement

    @field_validator("selected_text")
    @classmethod
    def _selected_text_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("selected_text must not be blank")
        return normalized


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
    assertions: tuple[SelectedAssertion, ...] = ()
    proposed_disposition: BlockDisposition | None = Field(
        default=None,
        validation_alias="disposition",
        exclude=True,
    )


class AtomicClaim(BaseModel):
    """One retained atomic claim and its distinct model/report representations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    block_id: str
    selected_text: str
    claim_text: str | None
    anchor_text: str | None
    anchor_text_proposal: str | None = None
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
    # Ignore legacy model-supplied offsets during rollout. They are neither
    # trusted nor consulted; code uniquely locates anchor_text below.
    model_config = ConfigDict(extra="ignore", frozen=True)

    claim_id: str
    anchor_text: str = Field(min_length=1)

    @field_validator("anchor_text")
    @classmethod
    def _anchor_not_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("anchor_text must not be empty")
        return value


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

Apply the frozen atomic-v1 rule: one assertion is one independently
truth-valued event or state. If either coordinated clause could be true while
the other is false, select them separately. Preserve every entity, time,
place, quantity, negation, modality, and attribution qualifier that affects
truth.

For every supplied block_id, return exactly one block entry. Put every
independently truth-valued assertion in its assertions array. Return an empty
assertions array when there is no assertion to select. Do not return a
disposition: code mechanically derives claims_selected from a non-empty array
and no_verifiable_claims from an empty array.

Classify each selected assertion independently on the orthogonal evidence
dimension:
- external: its truth depends on evidence outside this report;
- internal: it can be checked against the report artifact itself;
- none: it makes no evidence-bearing factual assertion.

Do not omit a block. Selection is independent of whether a source has already
been found. Return JSON only:
{{"blocks":[{{"block_id":"block-0001","rationale":"...",\
"assertions":[{{"selected_text":"...","citation_requirement":"external|\
internal|none"}}]}}]}}

Purely structural example: a block saying that an object changed twice may
contain two independently truth-valued assertions; return two assertions
instead of joining them.

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

For each claim_id, identify one exact, contiguous anchor_text in the canonical
report that expresses the selected assertion. Copy it verbatim, including its
original punctuation and capitalization. Do not calculate or return character
offsets. Do not use claim_text as anchor_text unless those exact bytes occur
in the report. Do not join noncontiguous fragments.

The code will require exactly one occurrence of anchor_text in the entire
report, calculate start_char/end_char from that occurrence, and mechanically
check report[start_char:end_char] == anchor_text. It will retain a failed
normalization instead of guessing.

Return exactly one entry per claim_id as JSON only:
{{"claims":[{{"claim_id":"claim-0001",\
"anchor_text":"A device stopped."}}]}}

Canonical report:
{report}

Decontextualized claims:
{claims}
"""


def build_selection_prompt(
    blocks: Sequence[MarkdownBlock],
) -> str:
    """Build the complete block-selection prompt."""

    payload = [block.model_dump(mode="json") for block in blocks]
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
) -> str:
    """Build the exact-anchor extraction prompt."""

    return _EXTRACTION_PROMPT.format(
        report=report,
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

    expected = {block.block_id for block in blocks}
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
        derived_disposition = (
            BlockDisposition.CLAIMS_SELECTED
            if proposal.assertions
            else BlockDisposition.NO_VERIFIABLE_CLAIMS
        )
        selection = BlockSelection(
            block_id=proposal.block_id,
            disposition=derived_disposition,
            rationale=proposal.rationale,
            assertions=proposal.assertions,
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
        if selection.block_id not in expected:
            diagnostics.append(
                f"selection_unknown_block: {selection.block_id}"
            )
            continue
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
) -> tuple[tuple[ContextSpan, ...], str | None]:
    spans: list[ContextSpan] = []
    for proposal in proposals:
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
    reason: str,
) -> AtomicClaim:
    return AtomicClaim(
        claim_id=str(claim["claim_id"]),
        block_id=str(claim["block_id"]),
        selected_text=str(claim["selected_text"]),
        claim_text=claim_text,
        anchor_text_proposal=anchor_text_proposal,
        anchor_text=anchor_text,
        context_spans=tuple(context_spans),
        context_span_proposals=tuple(context_span_proposals),
        citation_requirement=CitationRequirement(
            claim["citation_requirement"]
        ),
        source_resolution=SourceResolution.UNRESOLVED,
        normalization_status=ClaimNormalizationStatus.NORMALIZATION_FAILED,
        normalization_failure=reason,
    )


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
                build_selection_prompt(block_batch),
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
                    build_extraction_prompt(report, extraction_batch),
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

    block_by_id = {block.block_id: block for block in blocks}
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

        occurrences = _unique_occurrences(report, extraction.anchor_text)
        block = block_by_id[str(seed["block_id"])]
        reason: str | None = None
        anchor_start: int | None = None
        anchor_end: int | None = None
        anchor_text = extraction.anchor_text
        if len(occurrences) > 1:
            reason = "anchor_not_unique"
        elif occurrences:
            anchor_start = occurrences[0]
            anchor_end = anchor_start + len(extraction.anchor_text)
            if report[anchor_start:anchor_end] != anchor_text:
                raise AssertionError("code-owned anchor bounds must round-trip")
        else:
            repaired = locate_verification_quote(report, extraction.anchor_text)
            if (
                repaired.location_status
                == NoteLocationStatus.REPAIRED_LOCATABLE
                and repaired.span is not None
                and repaired.source_quote is not None
            ):
                anchor_start = repaired.span.start_char
                anchor_end = repaired.span.end_char
                anchor_text = repaired.source_quote
            else:
                reason = (
                    "anchor_not_unique"
                    if repaired.failure_reason
                    == QuoteFailureReason.AMBIGUOUS_FORMAT_MATCH
                    else "anchor_not_found"
                )
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
                    anchor_text_proposal=extraction.anchor_text,
                    anchor_text=anchor_text,
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
                selected_text=str(seed["selected_text"]),
                claim_text=claim_text,
                anchor_text_proposal=extraction.anchor_text,
                anchor_text=anchor_text,
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

    seed_by_id = {
        str(seed["claim_id"]): seed
        for seed in seed_claims
    }
    anchor_proposal_count = len(extraction_by_id)
    anchor_copied_from_selection_count = sum(
        extraction.anchor_text
        == str(seed_by_id[claim_id]["selected_text"])
        for claim_id, extraction in extraction_by_id.items()
    )
    anchor_copied_from_selection_rate = (
        anchor_copied_from_selection_count / anchor_proposal_count
        if anchor_proposal_count
        else 0.0
    )
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
