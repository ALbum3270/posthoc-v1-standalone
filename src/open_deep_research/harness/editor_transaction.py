"""Mechanical lineage and closed review contracts for editorial proposals.

An editorial proposal is a mutation of an already audited draft.  The claim
and Markdown registries are rebuilt after that mutation, so their local IDs
cannot establish identity across the two drafts.  This module instead binds a
proposal to exact character ranges, UTF-8 content hashes, and Markdown
structure.  Semantic acceptance remains a model judgement, but the set of
facts that judgement must cover is mechanically closed.

Nothing here publishes or mutates a report.  A caller may commit only a
``complete`` + ``accept`` :class:`EditorialTransactionResult`; every protocol
failure is represented as an explicit rollback result.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from open_deep_research.harness.claims import (
    AtomicClaim,
    MarkdownBlock,
    parse_markdown_blocks,
)
from open_deep_research.harness.jsonio import loads_lenient


_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_Side = Literal["pre", "post"]


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _overlaps(
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> bool:
    """Return whether two non-empty character intervals intersect."""

    return left_start < right_end and right_start < left_end


class CharacterRange(BaseModel):
    """A half-open Python-string character range.

    This is deliberately named a character range, not a byte range.  The
    surrounding harness slices Python ``str`` values at these offsets; UTF-8
    is used only for the accompanying hashes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)

    @model_validator(mode="after")
    def _range_is_ordered(self) -> CharacterRange:
        if self.end_char < self.start_char:
            raise ValueError("character range end must not precede start")
        return self


class TextSpanSnapshot(BaseModel):
    """An exact, non-empty text slice in one frozen draft."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    text_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _text_matches_bounds_and_hash(self) -> TextSpanSnapshot:
        if self.end_char <= self.start_char:
            raise ValueError("text span end must exceed start")
        if self.end_char - self.start_char != len(self.text):
            raise ValueError("text span bounds must match character length")
        if self.text_sha256 != _text_sha256(self.text):
            raise ValueError("text span hash does not match text")
        return self


def _text_span(text: str, start_char: int, end_char: int) -> TextSpanSnapshot:
    return TextSpanSnapshot(
        text=text,
        start_char=start_char,
        end_char=end_char,
        text_sha256=_text_sha256(text),
    )


class CharacterEdit(BaseModel):
    """One proposed replacement located without a Markdown or claim ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    original_text: str = Field(min_length=1)
    replacement_text: str

    @model_validator(mode="after")
    def _edit_is_a_real_exact_replacement(self) -> CharacterEdit:
        if self.end_char <= self.start_char:
            raise ValueError("editorial replacements require a non-empty input")
        if self.end_char - self.start_char != len(self.original_text):
            raise ValueError("edit bounds must match original text length")
        if self.replacement_text == self.original_text:
            raise ValueError("change manifest cannot contain a no-op edit")
        return self


class EditorialEditSpan(BaseModel):
    """One edit bound to both draft coordinate spaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    edit_ref: str = Field(min_length=1)
    pre_range: CharacterRange
    post_range: CharacterRange
    original_text: str = Field(min_length=1)
    replacement_text: str
    original_text_sha256: str = Field(pattern=_SHA256_PATTERN)
    replacement_text_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _edit_payload_is_hash_bound(self) -> EditorialEditSpan:
        if self.pre_range.end_char - self.pre_range.start_char != len(
            self.original_text
        ):
            raise ValueError("pre-edit range must match original text")
        if self.post_range.end_char - self.post_range.start_char != len(
            self.replacement_text
        ):
            raise ValueError("post-edit range must match replacement text")
        if self.original_text_sha256 != _text_sha256(self.original_text):
            raise ValueError("original edit text hash mismatch")
        if self.replacement_text_sha256 != _text_sha256(self.replacement_text):
            raise ValueError("replacement edit text hash mismatch")
        expected_ref = "edit-" + _canonical_sha256(
            {
                "pre_range": self.pre_range.model_dump(mode="json"),
                "post_range": self.post_range.model_dump(mode="json"),
                "original_text_sha256": self.original_text_sha256,
                "replacement_text_sha256": self.replacement_text_sha256,
            }
        )[:24]
        if self.edit_ref != expected_ref:
            raise ValueError("edit_ref does not match edit ranges and hashes")
        return self


class EditorialBlockSnapshot(BaseModel):
    """A structural block fingerprint local to one draft.

    The reference intentionally excludes the parser's ``block_id`` and
    ordinal.  Both are regenerated after editing and therefore are locators,
    not lineage identities.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    side: _Side
    block_ref: str = Field(min_length=1)
    draft_sha256: str = Field(pattern=_SHA256_PATTERN)
    kind: str = Field(min_length=1)
    section_path: tuple[str, ...] = ()
    span: TextSpanSnapshot

    @model_validator(mode="after")
    def _reference_matches_structure(self) -> EditorialBlockSnapshot:
        expected = f"{self.side}-block-" + _canonical_sha256(
            {
                "draft_sha256": self.draft_sha256,
                "kind": self.kind,
                "section_path": self.section_path,
                "start_char": self.span.start_char,
                "end_char": self.span.end_char,
                "text_sha256": self.span.text_sha256,
            }
        )[:24]
        if self.block_ref != expected:
            raise ValueError("block_ref does not match structural fingerprint")
        return self


class StableBlockPair(BaseModel):
    """One unchanged structural block mapped across draft coordinate spaces."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pre_block_ref: str = Field(min_length=1)
    post_block_ref: str = Field(min_length=1)


class EditorialChangeManifest(BaseModel):
    """Closed mechanical lineage for one complete editor proposal."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["editorial-change-manifest-v1"] = (
        "editorial-change-manifest-v1"
    )
    original_draft_sha256: str = Field(pattern=_SHA256_PATTERN)
    proposed_draft_sha256: str = Field(pattern=_SHA256_PATTERN)
    edits: tuple[EditorialEditSpan, ...]
    pre_blocks: tuple[EditorialBlockSnapshot, ...]
    post_blocks: tuple[EditorialBlockSnapshot, ...]
    stable_block_pairs: tuple[StableBlockPair, ...]
    directly_edited_pre_block_refs: tuple[str, ...]
    affected_pre_block_refs: tuple[str, ...]
    affected_post_block_refs: tuple[str, ...]
    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _manifest_is_closed_and_hash_bound(self) -> EditorialChangeManifest:
        if not self.edits:
            raise ValueError("change manifest requires at least one edit")
        edit_refs = tuple(edit.edit_ref for edit in self.edits)
        if len(edit_refs) != len(set(edit_refs)):
            raise ValueError("edit references must be unique")
        pre_cursor = 0
        post_cursor = 0
        cumulative_delta = 0
        for edit in self.edits:
            if edit.pre_range.start_char < pre_cursor:
                raise ValueError("pre-edit ranges must be ordered and non-overlapping")
            if edit.post_range.start_char < post_cursor:
                raise ValueError("post-edit ranges must be ordered and non-overlapping")
            expected_post_start = edit.pre_range.start_char + cumulative_delta
            if edit.post_range.start_char != expected_post_start:
                raise ValueError("pre/post edit coordinates do not share one delta map")
            pre_cursor = edit.pre_range.end_char
            post_cursor = edit.post_range.end_char
            cumulative_delta += len(edit.replacement_text) - len(edit.original_text)
        pre_refs = tuple(block.block_ref for block in self.pre_blocks)
        post_refs = tuple(block.block_ref for block in self.post_blocks)
        if len(pre_refs) != len(set(pre_refs)):
            raise ValueError("pre block references must be unique")
        if len(post_refs) != len(set(post_refs)):
            raise ValueError("post block references must be unique")
        for refs, label in (
            (self.directly_edited_pre_block_refs, "directly edited block"),
            (self.affected_pre_block_refs, "affected pre block"),
            (self.affected_post_block_refs, "affected post block"),
        ):
            if len(refs) != len(set(refs)):
                raise ValueError(f"{label} references must be unique")
        if any(
            block.side != "pre"
            or block.draft_sha256 != self.original_draft_sha256
            for block in self.pre_blocks
        ):
            raise ValueError("pre blocks must belong to the original draft")
        if any(
            block.side != "post"
            or block.draft_sha256 != self.proposed_draft_sha256
            for block in self.post_blocks
        ):
            raise ValueError("post blocks must belong to the proposed draft")
        stable_pre = tuple(pair.pre_block_ref for pair in self.stable_block_pairs)
        stable_post = tuple(
            pair.post_block_ref for pair in self.stable_block_pairs
        )
        if len(stable_pre) != len(set(stable_pre)) or len(stable_post) != len(
            set(stable_post)
        ):
            raise ValueError("stable block pairing must be one-to-one")
        if set(stable_pre) | set(self.affected_pre_block_refs) != set(pre_refs):
            raise ValueError("stable and affected pre blocks must partition scope")
        if set(stable_pre) & set(self.affected_pre_block_refs):
            raise ValueError("a pre block cannot be stable and affected")
        if set(stable_post) | set(self.affected_post_block_refs) != set(post_refs):
            raise ValueError("stable and affected post blocks must partition scope")
        if set(stable_post) & set(self.affected_post_block_refs):
            raise ValueError("a post block cannot be stable and affected")
        if not set(self.directly_edited_pre_block_refs).issubset(
            self.affected_pre_block_refs
        ):
            raise ValueError("directly edited blocks must be affected")
        expected_hash = _canonical_sha256(
            self.model_dump(mode="json", exclude={"manifest_sha256"})
        )
        if self.manifest_sha256 != expected_hash:
            raise ValueError("manifest_sha256 does not match manifest payload")
        return self


def _snapshot_block(
    block: MarkdownBlock,
    *,
    side: _Side,
    draft: str,
    draft_sha256: str,
) -> EditorialBlockSnapshot:
    text = draft[block.start_char : block.end_char]
    if text != block.text:
        raise ValueError("Markdown block does not round-trip to draft")
    span = _text_span(text, block.start_char, block.end_char)
    block_ref = f"{side}-block-" + _canonical_sha256(
        {
            "draft_sha256": draft_sha256,
            "kind": block.kind.value,
            "section_path": tuple(block.section_path),
            "start_char": span.start_char,
            "end_char": span.end_char,
            "text_sha256": span.text_sha256,
        }
    )[:24]
    return EditorialBlockSnapshot(
        side=side,
        block_ref=block_ref,
        draft_sha256=draft_sha256,
        kind=block.kind.value,
        section_path=tuple(block.section_path),
        span=span,
    )


def _normalized_edits(
    original_draft: str,
    edits: Sequence[CharacterEdit],
) -> tuple[CharacterEdit, ...]:
    ordered = tuple(sorted(edits, key=lambda item: item.start_char))
    if not ordered:
        raise ValueError("at least one character edit is required")
    previous_end = -1
    for edit in ordered:
        if edit.start_char < previous_end:
            raise ValueError("editorial character edits must not overlap")
        if edit.end_char > len(original_draft):
            raise ValueError("editorial character edit exceeds draft bounds")
        if original_draft[edit.start_char : edit.end_char] != edit.original_text:
            raise ValueError("editorial character edit does not match draft")
        previous_end = edit.end_char
    return ordered


def apply_character_edits(
    original_draft: str,
    edits: Sequence[CharacterEdit],
) -> str:
    """Apply exact, non-overlapping replacements to a frozen draft."""

    ordered = _normalized_edits(original_draft, edits)
    revised = original_draft
    for edit in reversed(ordered):
        revised = (
            revised[: edit.start_char]
            + edit.replacement_text
            + revised[edit.end_char :]
        )
    return revised


def _post_position(position: int, edits: Sequence[CharacterEdit]) -> int:
    delta = 0
    for edit in edits:
        if edit.end_char <= position:
            delta += len(edit.replacement_text) - len(edit.original_text)
            continue
        if edit.start_char < position < edit.end_char:
            raise ValueError("cannot map a position inside an edited range")
    return position + delta


def build_editorial_change_manifest(
    original_draft: str,
    proposed_draft: str,
    *,
    edits: Sequence[CharacterEdit],
) -> EditorialChangeManifest:
    """Build ID-independent block lineage for an exact editor proposal."""

    ordered = _normalized_edits(original_draft, edits)
    if apply_character_edits(original_draft, ordered) != proposed_draft:
        raise ValueError("character edits do not reconstruct proposed draft")
    original_hash = _text_sha256(original_draft)
    proposed_hash = _text_sha256(proposed_draft)
    if original_hash == proposed_hash:
        raise ValueError("editorial proposal must change draft bytes")

    edit_spans: list[EditorialEditSpan] = []
    cumulative_delta = 0
    for edit in ordered:
        post_start = edit.start_char + cumulative_delta
        post_end = post_start + len(edit.replacement_text)
        pre_range = CharacterRange(
            start_char=edit.start_char,
            end_char=edit.end_char,
        )
        post_range = CharacterRange(start_char=post_start, end_char=post_end)
        original_text_sha256 = _text_sha256(edit.original_text)
        replacement_text_sha256 = _text_sha256(edit.replacement_text)
        edit_ref = "edit-" + _canonical_sha256(
            {
                "pre_range": pre_range.model_dump(mode="json"),
                "post_range": post_range.model_dump(mode="json"),
                "original_text_sha256": original_text_sha256,
                "replacement_text_sha256": replacement_text_sha256,
            }
        )[:24]
        edit_spans.append(
            EditorialEditSpan(
                edit_ref=edit_ref,
                pre_range=pre_range,
                post_range=post_range,
                original_text=edit.original_text,
                replacement_text=edit.replacement_text,
                original_text_sha256=original_text_sha256,
                replacement_text_sha256=replacement_text_sha256,
            )
        )
        cumulative_delta += len(edit.replacement_text) - len(edit.original_text)

    parsed_pre = parse_markdown_blocks(original_draft)
    parsed_post = parse_markdown_blocks(proposed_draft)
    pre_blocks = tuple(
        _snapshot_block(
            block,
            side="pre",
            draft=original_draft,
            draft_sha256=original_hash,
        )
        for block in parsed_pre
    )
    post_blocks = tuple(
        _snapshot_block(
            block,
            side="post",
            draft=proposed_draft,
            draft_sha256=proposed_hash,
        )
        for block in parsed_post
    )
    parsed_pre_by_ref = {
        snapshot.block_ref: block
        for snapshot, block in zip(pre_blocks, parsed_pre, strict=True)
    }
    post_by_bounds = {
        (block.span.start_char, block.span.end_char): block
        for block in post_blocks
    }
    directly_edited = {
        block.block_ref
        for block in pre_blocks
        if any(
            _overlaps(
                block.span.start_char,
                block.span.end_char,
                edit.start_char,
                edit.end_char,
            )
            for edit in ordered
        )
    }

    stable_pairs: list[StableBlockPair] = []
    used_post_refs: set[str] = set()
    for pre_snapshot in pre_blocks:
        if pre_snapshot.block_ref in directly_edited:
            continue
        parsed_block = parsed_pre_by_ref[pre_snapshot.block_ref]
        try:
            mapped_start = _post_position(parsed_block.start_char, ordered)
            mapped_end = _post_position(parsed_block.end_char, ordered)
        except ValueError:
            continue
        candidate = post_by_bounds.get((mapped_start, mapped_end))
        if candidate is None or candidate.block_ref in used_post_refs:
            continue
        if (
            candidate.span.text_sha256 != pre_snapshot.span.text_sha256
            or candidate.kind != pre_snapshot.kind
            or candidate.section_path != pre_snapshot.section_path
        ):
            continue
        stable_pairs.append(
            StableBlockPair(
                pre_block_ref=pre_snapshot.block_ref,
                post_block_ref=candidate.block_ref,
            )
        )
        used_post_refs.add(candidate.block_ref)

    stable_pre = {pair.pre_block_ref for pair in stable_pairs}
    stable_post = {pair.post_block_ref for pair in stable_pairs}
    affected_pre = tuple(
        block.block_ref for block in pre_blocks if block.block_ref not in stable_pre
    )
    affected_post = tuple(
        block.block_ref
        for block in post_blocks
        if block.block_ref not in stable_post
    )
    manifest_payload = {
        "version": "editorial-change-manifest-v1",
        "original_draft_sha256": original_hash,
        "proposed_draft_sha256": proposed_hash,
        "edits": [edit.model_dump(mode="json") for edit in edit_spans],
        "pre_blocks": [block.model_dump(mode="json") for block in pre_blocks],
        "post_blocks": [block.model_dump(mode="json") for block in post_blocks],
        "stable_block_pairs": [
            pair.model_dump(mode="json") for pair in stable_pairs
        ],
        "directly_edited_pre_block_refs": [
            block.block_ref
            for block in pre_blocks
            if block.block_ref in directly_edited
        ],
        "affected_pre_block_refs": list(affected_pre),
        "affected_post_block_refs": list(affected_post),
    }
    return EditorialChangeManifest(
        **manifest_payload,
        manifest_sha256=_canonical_sha256(manifest_payload),
    )


class AuditUnitInput(BaseModel):
    """One registry-local semantic unit projected without using its local ID."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report_surface: TextSpanSnapshot
    semantic_text: str = Field(min_length=1)
    context_spans: tuple[TextSpanSnapshot, ...] = ()
    audit_payload: dict[str, Any] = Field(default_factory=dict)
    # Kept only to help a caller locate its source record.  It is deliberately
    # excluded from fingerprints, manifests, affected scope, and prompts.
    registry_locator: str | None = Field(default=None, exclude=True)


def audit_unit_input_from_claim(
    claim: AtomicClaim,
    *,
    audit_payload: Mapping[str, Any] | None = None,
) -> AuditUnitInput:
    """Project a live claim into the ID-independent transaction boundary."""

    surface = claim.report_surface
    if surface is None:
        raise ValueError("editor transaction requires a report_surface")
    semantic_text = claim.claim_text or claim.selected_text
    return AuditUnitInput(
        report_surface=_text_span(
            surface.text,
            surface.start_char,
            surface.end_char,
        ),
        semantic_text=semantic_text,
        context_spans=tuple(
            _text_span(span.text, span.start_char, span.end_char)
            for span in claim.context_spans
        ),
        audit_payload=dict(audit_payload or {}),
        registry_locator=claim.claim_id,
    )


class AffectedAuditUnit(BaseModel):
    """A semantic unit whose report surface or context touches the mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    side: _Side
    unit_ref: str = Field(min_length=1)
    block_ref: str = Field(min_length=1)
    draft_sha256: str = Field(pattern=_SHA256_PATTERN)
    report_surface: TextSpanSnapshot
    semantic_text: str = Field(min_length=1)
    semantic_text_sha256: str = Field(pattern=_SHA256_PATTERN)
    context_spans: tuple[TextSpanSnapshot, ...] = ()
    audit_payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unit_reference_is_content_bound(self) -> AffectedAuditUnit:
        if self.semantic_text_sha256 != _text_sha256(self.semantic_text):
            raise ValueError("semantic text hash mismatch")
        expected = f"{self.side}-unit-" + _canonical_sha256(
            {
                "draft_sha256": self.draft_sha256,
                "block_ref": self.block_ref,
                "report_surface": self.report_surface.model_dump(mode="json"),
                "semantic_text_sha256": self.semantic_text_sha256,
                "context_spans": [
                    span.model_dump(mode="json") for span in self.context_spans
                ],
            }
        )[:24]
        if self.unit_ref != expected:
            raise ValueError("unit_ref does not match semantic surface fingerprint")
        return self


class EditorialAffectedScope(BaseModel):
    """Fixed-point structural and context dependency closure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    affected_pre_block_refs: tuple[str, ...]
    affected_post_block_refs: tuple[str, ...]
    affected_pre_units: tuple[AffectedAuditUnit, ...]
    affected_post_units: tuple[AffectedAuditUnit, ...]
    closure_rounds: int = Field(ge=1)

    @model_validator(mode="after")
    def _scope_references_are_closed(self) -> EditorialAffectedScope:
        for refs, label in (
            (self.affected_pre_block_refs, "pre block"),
            (self.affected_post_block_refs, "post block"),
            (
                tuple(unit.unit_ref for unit in self.affected_pre_units),
                "pre unit",
            ),
            (
                tuple(unit.unit_ref for unit in self.affected_post_units),
                "post unit",
            ),
        ):
            if len(refs) != len(set(refs)):
                raise ValueError(f"affected {label} references must be unique")
        if any(
            unit.block_ref not in self.affected_pre_block_refs
            for unit in self.affected_pre_units
        ):
            raise ValueError("affected pre units must belong to affected blocks")
        if any(
            unit.block_ref not in self.affected_post_block_refs
            for unit in self.affected_post_units
        ):
            raise ValueError("affected post units must belong to affected blocks")
        return self


def _validate_unit_inputs(
    draft: str,
    draft_sha256: str,
    blocks: Sequence[EditorialBlockSnapshot],
    units: Sequence[AuditUnitInput],
    *,
    side: _Side,
) -> tuple[AffectedAuditUnit, ...]:
    projected: list[AffectedAuditUnit] = []
    seen: dict[str, AffectedAuditUnit] = {}
    for unit in units:
        surface = unit.report_surface
        if surface.end_char > len(draft):
            raise ValueError(f"{side} report surface exceeds draft")
        if draft[surface.start_char : surface.end_char] != surface.text:
            raise ValueError(f"{side} report surface does not match draft")
        containing = tuple(
            block
            for block in blocks
            if block.span.start_char <= surface.start_char
            and surface.end_char <= block.span.end_char
        )
        if len(containing) != 1:
            raise ValueError(f"{side} report surface must belong to one block")
        for context in unit.context_spans:
            if context.end_char > len(draft):
                raise ValueError(f"{side} context span exceeds draft")
            if draft[context.start_char : context.end_char] != context.text:
                raise ValueError(f"{side} context span does not match draft")
        block_ref = containing[0].block_ref
        semantic_hash = _text_sha256(unit.semantic_text)
        unit_ref = f"{side}-unit-" + _canonical_sha256(
            {
                "draft_sha256": draft_sha256,
                "block_ref": block_ref,
                "report_surface": surface.model_dump(mode="json"),
                "semantic_text_sha256": semantic_hash,
                "context_spans": [
                    span.model_dump(mode="json") for span in unit.context_spans
                ],
            }
        )[:24]
        item = AffectedAuditUnit(
            side=side,
            unit_ref=unit_ref,
            block_ref=block_ref,
            draft_sha256=draft_sha256,
            report_surface=surface,
            semantic_text=unit.semantic_text,
            semantic_text_sha256=semantic_hash,
            context_spans=unit.context_spans,
            audit_payload=dict(unit.audit_payload),
        )
        previous = seen.get(unit_ref)
        if previous is not None and previous != item:
            raise ValueError("duplicate unit fingerprint has conflicting payload")
        if previous is None:
            seen[unit_ref] = item
            projected.append(item)
    return tuple(projected)


def _block_intervals(
    refs: set[str],
    blocks_by_ref: Mapping[str, EditorialBlockSnapshot],
    raw_ranges: Sequence[CharacterRange],
) -> tuple[CharacterRange, ...]:
    return (
        *raw_ranges,
        *(
            CharacterRange(
                start_char=blocks_by_ref[ref].span.start_char,
                end_char=blocks_by_ref[ref].span.end_char,
            )
            for ref in refs
        ),
    )


def _span_hits_ranges(
    span: TextSpanSnapshot,
    ranges: Sequence[CharacterRange],
) -> bool:
    return any(
        _overlaps(
            span.start_char,
            span.end_char,
            candidate.start_char,
            candidate.end_char,
        )
        for candidate in ranges
        if candidate.end_char > candidate.start_char
    )


def build_editorial_affected_scope(
    manifest: EditorialChangeManifest,
    *,
    original_draft: str,
    proposed_draft: str,
    pre_units: Sequence[AuditUnitInput],
    post_units: Sequence[AuditUnitInput],
) -> EditorialAffectedScope:
    """Close changed Markdown structure over report/context dependencies."""

    if _text_sha256(original_draft) != manifest.original_draft_sha256:
        raise ValueError("original draft does not match change manifest")
    if _text_sha256(proposed_draft) != manifest.proposed_draft_sha256:
        raise ValueError("proposed draft does not match change manifest")
    pre_blocks = {block.block_ref: block for block in manifest.pre_blocks}
    post_blocks = {block.block_ref: block for block in manifest.post_blocks}
    projected_pre = _validate_unit_inputs(
        original_draft,
        manifest.original_draft_sha256,
        manifest.pre_blocks,
        pre_units,
        side="pre",
    )
    projected_post = _validate_unit_inputs(
        proposed_draft,
        manifest.proposed_draft_sha256,
        manifest.post_blocks,
        post_units,
        side="post",
    )
    affected_pre = set(manifest.affected_pre_block_refs)
    affected_post = set(manifest.affected_post_block_refs)
    pre_to_post = {
        pair.pre_block_ref: pair.post_block_ref
        for pair in manifest.stable_block_pairs
    }
    post_to_pre = {
        pair.post_block_ref: pair.pre_block_ref
        for pair in manifest.stable_block_pairs
    }
    pre_edit_ranges = tuple(edit.pre_range for edit in manifest.edits)
    post_edit_ranges = tuple(edit.post_range for edit in manifest.edits)

    rounds = 0
    while True:
        rounds += 1
        before = (frozenset(affected_pre), frozenset(affected_post))
        pre_ranges = _block_intervals(
            affected_pre,
            pre_blocks,
            pre_edit_ranges,
        )
        post_ranges = _block_intervals(
            affected_post,
            post_blocks,
            post_edit_ranges,
        )
        for unit in projected_pre:
            if unit.block_ref in affected_pre or any(
                _span_hits_ranges(context, pre_ranges)
                for context in unit.context_spans
            ):
                affected_pre.add(unit.block_ref)
        for unit in projected_post:
            if unit.block_ref in affected_post or any(
                _span_hits_ranges(context, post_ranges)
                for context in unit.context_spans
            ):
                affected_post.add(unit.block_ref)
        for pre_ref in tuple(affected_pre):
            post_ref = pre_to_post.get(pre_ref)
            if post_ref is not None:
                affected_post.add(post_ref)
        for post_ref in tuple(affected_post):
            pre_ref = post_to_pre.get(post_ref)
            if pre_ref is not None:
                affected_pre.add(pre_ref)
        after = (frozenset(affected_pre), frozenset(affected_post))
        if after == before:
            break

    ordered_pre_refs = tuple(
        block.block_ref
        for block in manifest.pre_blocks
        if block.block_ref in affected_pre
    )
    ordered_post_refs = tuple(
        block.block_ref
        for block in manifest.post_blocks
        if block.block_ref in affected_post
    )
    return EditorialAffectedScope(
        manifest_sha256=manifest.manifest_sha256,
        affected_pre_block_refs=ordered_pre_refs,
        affected_post_block_refs=ordered_post_refs,
        affected_pre_units=tuple(
            unit for unit in projected_pre if unit.block_ref in affected_pre
        ),
        affected_post_units=tuple(
            unit for unit in projected_post if unit.block_ref in affected_post
        ),
        closure_rounds=rounds,
    )


class EditorialIntent(str, Enum):
    REMOVE = "remove"
    QUALIFY = "qualify"
    RETAIN_WITH_LABEL = "retain_with_label"


class TransactionTargetRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pre_ref: str = Field(min_length=1)
    intended_action: EditorialIntent


class TransactionReviewRequest(BaseModel):
    """Closed semantic denominators supplied to an independent reviewer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    targets: tuple[TransactionTargetRequirement, ...]
    preserved_pre_refs: tuple[str, ...] = ()
    post_refs: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _review_denominators_are_unique(self) -> TransactionReviewRequest:
        target_refs = tuple(target.pre_ref for target in self.targets)
        if not target_refs:
            raise ValueError("transaction review requires an edited target")
        for refs, label in (
            (target_refs, "target"),
            (self.preserved_pre_refs, "preserved pre"),
            (self.post_refs, "post"),
        ):
            if len(refs) != len(set(refs)):
                raise ValueError(f"{label} references must be unique")
        if set(target_refs) & set(self.preserved_pre_refs):
            raise ValueError("target and preserved pre references must be disjoint")
        return self


def build_transaction_review_request(
    affected_scope: EditorialAffectedScope,
    *,
    target_intents: Mapping[str, EditorialIntent],
) -> TransactionReviewRequest:
    """Close review denominators over every affected semantic unit.

    Callers choose only the semantic intent for actual edit targets.  They do
    not choose which neighbouring facts or post-edit units the reviewer sees:
    every other affected pre unit is mechanically preserved scope, and every
    affected post unit is mechanically included.  This prevents a caller from
    obtaining an apparently complete review by silently omitting a difficult
    unit from the denominator.
    """

    affected_pre_refs = tuple(
        unit.unit_ref for unit in affected_scope.affected_pre_units
    )
    affected_post_refs = tuple(
        unit.unit_ref for unit in affected_scope.affected_post_units
    )
    unknown_targets = set(target_intents) - set(affected_pre_refs)
    if unknown_targets:
        raise ValueError(
            "transaction targets fall outside affected scope: "
            + ", ".join(sorted(unknown_targets))
        )
    if not target_intents:
        raise ValueError("transaction review requires an affected edit target")
    return TransactionReviewRequest(
        manifest_sha256=affected_scope.manifest_sha256,
        targets=tuple(
            TransactionTargetRequirement(
                pre_ref=ref,
                intended_action=target_intents[ref],
            )
            for ref in affected_pre_refs
            if ref in target_intents
        ),
        preserved_pre_refs=tuple(
            ref for ref in affected_pre_refs if ref not in target_intents
        ),
        post_refs=affected_post_refs,
    )


class TargetResolutionOutcome(str, Enum):
    RESOLVED = "resolved"
    RETAINED = "retained"
    NOT_RESOLVED = "not_resolved"
    UNCERTAIN = "uncertain"


class PreservationOutcome(str, Enum):
    PRESERVED = "preserved"
    DEGRADED = "degraded"
    UNCERTAIN = "uncertain"


class PostLineage(str, Enum):
    DERIVED_FROM_PRE = "derived_from_pre"
    NEW_MATERIAL = "new_material"
    UNCERTAIN = "uncertain"


class PostAssessment(str, Enum):
    ACCEPTABLE = "acceptable"
    DEGRADED = "degraded"
    UNCERTAIN = "uncertain"


class AnswerPreservation(str, Enum):
    PRESERVED = "preserved"
    NARROWED_WITH_EVIDENCE = "narrowed_with_evidence"
    DEGRADED = "degraded"
    UNCERTAIN = "uncertain"


class TargetResolutionReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pre_ref: str = Field(min_length=1)
    intended_action: EditorialIntent
    outcome: TargetResolutionOutcome
    post_refs: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)


class PreservedFactReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pre_ref: str = Field(min_length=1)
    outcome: PreservationOutcome
    post_refs: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)


class PostUnitReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    post_ref: str = Field(min_length=1)
    lineage: PostLineage
    pre_refs: tuple[str, ...] = ()
    assessment: PostAssessment
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _lineage_has_matching_predecessors(self) -> PostUnitReview:
        if self.lineage is PostLineage.DERIVED_FROM_PRE and not self.pre_refs:
            raise ValueError("derived post units require predecessor refs")
        if self.lineage is not PostLineage.DERIVED_FROM_PRE and self.pre_refs:
            raise ValueError("new or uncertain post units cannot assert predecessors")
        return self


class TransactionExecutionStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class TransactionDecision(str, Enum):
    ACCEPT = "accept"
    ROLLBACK = "rollback"


class EditorialTransactionResult(BaseModel):
    """A mechanically closed acceptance or an explicit safe rollback."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    execution_status: TransactionExecutionStatus
    decision: TransactionDecision
    target_refs: tuple[str, ...]
    preserved_pre_refs: tuple[str, ...]
    post_refs: tuple[str, ...]
    target_reviews: tuple[TargetResolutionReview, ...] = ()
    preserved_reviews: tuple[PreservedFactReview, ...] = ()
    post_reviews: tuple[PostUnitReview, ...] = ()
    answer_preservation: AnswerPreservation | None = None
    answer_preservation_rationale: str = ""
    unreviewed_refs: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _result_is_closed_and_safe(self) -> EditorialTransactionResult:
        target_refs = set(self.target_refs)
        preserved_refs = set(self.preserved_pre_refs)
        post_refs = set(self.post_refs)
        for refs, label in (
            (self.target_refs, "target"),
            (self.preserved_pre_refs, "preserved"),
            (self.post_refs, "post"),
            (
                tuple(review.pre_ref for review in self.target_reviews),
                "target review",
            ),
            (
                tuple(review.pre_ref for review in self.preserved_reviews),
                "preserved review",
            ),
            (
                tuple(review.post_ref for review in self.post_reviews),
                "post review",
            ),
        ):
            if len(refs) != len(set(refs)):
                raise ValueError(f"transaction {label} references must be unique")
        if target_refs & preserved_refs:
            raise ValueError("target and preserved result scope must be disjoint")
        known_pre = target_refs | preserved_refs
        reviewed_target_refs = {
            review.pre_ref for review in self.target_reviews
        }
        reviewed_preserved_refs = {
            review.pre_ref for review in self.preserved_reviews
        }
        reviewed_post_refs = {review.post_ref for review in self.post_reviews}
        if not reviewed_target_refs.issubset(target_refs):
            raise ValueError("target reviews contain unknown pre references")
        if not reviewed_preserved_refs.issubset(preserved_refs):
            raise ValueError("preserved reviews contain unknown pre references")
        if not reviewed_post_refs.issubset(post_refs):
            raise ValueError("post reviews contain unknown post references")
        if any(
            not set(review.post_refs).issubset(post_refs)
            for review in (*self.target_reviews, *self.preserved_reviews)
        ):
            raise ValueError("pre reviews contain unknown post references")
        if any(
            not set(review.pre_refs).issubset(known_pre)
            for review in self.post_reviews
        ):
            raise ValueError("post reviews contain unknown pre references")
        if len(self.unreviewed_refs) != len(set(self.unreviewed_refs)):
            raise ValueError("unreviewed references must be unique")
        known_refs = known_pre | post_refs
        if not set(self.unreviewed_refs).issubset(known_refs):
            raise ValueError("unreviewed scope contains unknown references")
        mechanically_unreviewed = (
            (target_refs - reviewed_target_refs)
            | (preserved_refs - reviewed_preserved_refs)
            | (post_refs - reviewed_post_refs)
        )
        if set(self.unreviewed_refs) != mechanically_unreviewed:
            raise ValueError("unreviewed refs must equal the uncovered denominator")
        if self.execution_status is TransactionExecutionStatus.COMPLETE:
            if reviewed_target_refs != target_refs:
                raise ValueError("complete review must cover every target")
            if reviewed_preserved_refs != preserved_refs:
                raise ValueError("complete review must cover preserved facts")
            if reviewed_post_refs != post_refs:
                raise ValueError("complete review must cover every post unit")
            if self.unreviewed_refs:
                raise ValueError("complete review cannot retain unreviewed refs")
            if self.answer_preservation is None:
                raise ValueError("complete review requires answer preservation")
            forward = {
                (review.pre_ref, post_ref)
                for review in (*self.target_reviews, *self.preserved_reviews)
                for post_ref in review.post_refs
            }
            reverse = {
                (pre_ref, review.post_ref)
                for review in self.post_reviews
                for pre_ref in review.pre_refs
            }
            if forward != reverse:
                raise ValueError("pre/post semantic lineage must be bidirectional")
        elif self.decision is not TransactionDecision.ROLLBACK:
            raise ValueError("incomplete transaction review must roll back")

        acceptance_ready = (
            self.execution_status is TransactionExecutionStatus.COMPLETE
            and all(
                (
                    review.outcome is TargetResolutionOutcome.RESOLVED
                    and (
                        (
                            review.intended_action is EditorialIntent.REMOVE
                            and not review.post_refs
                        )
                        or (
                            review.intended_action is EditorialIntent.QUALIFY
                            and bool(review.post_refs)
                        )
                    )
                )
                or (
                    review.outcome is TargetResolutionOutcome.RETAINED
                    and review.intended_action
                    is EditorialIntent.RETAIN_WITH_LABEL
                    and bool(review.post_refs)
                )
                for review in self.target_reviews
            )
            and all(
                review.outcome is PreservationOutcome.PRESERVED
                for review in self.preserved_reviews
            )
            and all(
                review.lineage is PostLineage.DERIVED_FROM_PRE
                and review.assessment is PostAssessment.ACCEPTABLE
                for review in self.post_reviews
            )
            and self.answer_preservation
            in {
                AnswerPreservation.PRESERVED,
                AnswerPreservation.NARROWED_WITH_EVIDENCE,
            }
        )
        if (self.decision is TransactionDecision.ACCEPT) != acceptance_ready:
            raise ValueError("transaction decision must be derived from reviews")
        return self

    @property
    def may_commit(self) -> bool:
        return (
            self.execution_status is TransactionExecutionStatus.COMPLETE
            and self.decision is TransactionDecision.ACCEPT
        )


class EditorialTransactionAudit(BaseModel):
    """Durable transaction envelope, including safe pre-review failures.

    A manifest or reviewer protocol can fail before an
    :class:`EditorialTransactionResult` exists. The runner must still retain
    the candidate hashes and exact failure instead of collapsing that state
    into ``editorial_transaction=None``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_draft_sha256: str = Field(pattern=_SHA256_PATTERN)
    proposed_draft_sha256: str = Field(pattern=_SHA256_PATTERN)
    manifest: EditorialChangeManifest | None = None
    affected_scope: EditorialAffectedScope | None = None
    review_request: TransactionReviewRequest | None = None
    result: EditorialTransactionResult | None = None
    token_count: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    diagnostics: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _envelope_is_ordered(self) -> EditorialTransactionAudit:
        if self.manifest is not None:
            if (
                self.manifest.original_draft_sha256
                != self.original_draft_sha256
                or self.manifest.proposed_draft_sha256
                != self.proposed_draft_sha256
            ):
                raise ValueError("transaction manifest hashes do not match audit")
        if self.affected_scope is not None:
            if self.manifest is None:
                raise ValueError("affected scope requires a manifest")
            if self.affected_scope.manifest_sha256 != self.manifest.manifest_sha256:
                raise ValueError("affected scope and manifest do not match")
        if self.review_request is not None:
            if self.affected_scope is None:
                raise ValueError("review request requires affected scope")
            if self.review_request.manifest_sha256 != self.affected_scope.manifest_sha256:
                raise ValueError("review request and affected scope do not match")
        if self.result is not None:
            if self.review_request is None:
                raise ValueError("transaction result requires review request")
            if self.result.manifest_sha256 != self.review_request.manifest_sha256:
                raise ValueError("transaction result and review request do not match")
        if self.result is None and not self.diagnostics:
            raise ValueError("transaction without a result requires a diagnostic")
        return self

    @property
    def may_commit(self) -> bool:
        return self.result is not None and self.result.may_commit


class _RawTargetReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pre_ref: str = Field(min_length=1)
    outcome: TargetResolutionOutcome
    post_refs: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)


class _RawPreservedReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pre_ref: str = Field(min_length=1)
    outcome: PreservationOutcome
    post_refs: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)


class _RawPostReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    post_ref: str = Field(min_length=1)
    lineage: PostLineage
    pre_refs: tuple[str, ...] = ()
    assessment: PostAssessment
    rationale: str = Field(min_length=1)


class _RawTransactionReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    targets: tuple[_RawTargetReview, ...]
    preserved: tuple[_RawPreservedReview, ...]
    post_units: tuple[_RawPostReview, ...]
    answer_preservation: AnswerPreservation
    answer_preservation_rationale: str = Field(min_length=1)


_TRANSACTION_REVIEW_PROMPT = """\
Independently review one proposed edit against its already-audited baseline.
This is a mutation-safety review, not permission to suppress report artifacts.
If the proposal is rejected, the audited baseline will still be published.

References are opaque, hash-bound semantic-surface references. Never infer
identity from a claim number, block number, list order, or matching position.
Judge whether each editor target was resolved as intended, whether every
supported fact in the affected scope was preserved, and whether every post
unit is derived from those pre-edit facts rather than newly invented material.
Use uncertain whenever the supplied audits do not establish the comparison.

Return JSON only:
{{"manifest_sha256":"...","targets":[{{"pre_ref":"...",\
"outcome":"resolved|retained|not_resolved|uncertain",\
"post_refs":["..."],"rationale":"..."}}],"preserved":[{{\
"pre_ref":"...","outcome":"preserved|degraded|uncertain",\
"post_refs":["..."],"rationale":"..."}}],"post_units":[{{\
"post_ref":"...","lineage":"derived_from_pre|new_material|uncertain",\
"pre_refs":["..."],"assessment":"acceptable|degraded|uncertain",\
"rationale":"..."}}],\
"answer_preservation":"preserved|narrowed_with_evidence|degraded|uncertain",\
"answer_preservation_rationale":"..."}}

Every requested pre_ref and post_ref must appear exactly once in its array.
Mappings must be bidirectional: every pre->post reference must also appear in
that post unit's pre_refs, and vice versa. A removed target has no post_refs; a
qualified or retained target must identify its post units. Do not compare raw
counts and do not treat the editor's own rationale as proof.

Closed review request:
{request}

Mechanically affected scope:
{scope}

Frozen pre/post audit payload:
{payload}
"""


def build_transaction_reviewer_prompt(
    request: TransactionReviewRequest,
    *,
    affected_scope: EditorialAffectedScope,
    audit_payload: Mapping[str, Any],
) -> str:
    """Build a setwise reviewer prompt over opaque, closed references."""

    if request.manifest_sha256 != affected_scope.manifest_sha256:
        raise ValueError("review request and affected scope use different manifests")
    requested_pre_refs = {
        *(target.pre_ref for target in request.targets),
        *request.preserved_pre_refs,
    }
    affected_pre_refs = {
        unit.unit_ref for unit in affected_scope.affected_pre_units
    }
    if requested_pre_refs != affected_pre_refs:
        raise ValueError(
            "review request must cover every affected pre-edit unit exactly once"
        )
    if set(request.post_refs) != {
        unit.unit_ref for unit in affected_scope.affected_post_units
    }:
        raise ValueError(
            "review request must cover every affected post-edit unit exactly once"
        )
    return _TRANSACTION_REVIEW_PROMPT.format(
        request=json.dumps(
            request.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        ),
        scope=json.dumps(
            affected_scope.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
        ),
        payload=json.dumps(
            dict(audit_payload),
            ensure_ascii=False,
            sort_keys=True,
        ),
    )


def _rollback_result(
    request: TransactionReviewRequest,
    *,
    status: TransactionExecutionStatus,
    diagnostic: str,
) -> EditorialTransactionResult:
    all_refs = tuple(
        dict.fromkeys(
            (
                *(target.pre_ref for target in request.targets),
                *request.preserved_pre_refs,
                *request.post_refs,
            )
        )
    )
    return EditorialTransactionResult(
        manifest_sha256=request.manifest_sha256,
        execution_status=status,
        decision=TransactionDecision.ROLLBACK,
        target_refs=tuple(target.pre_ref for target in request.targets),
        preserved_pre_refs=request.preserved_pre_refs,
        post_refs=request.post_refs,
        unreviewed_refs=all_refs,
        diagnostics=(diagnostic,),
    )


def parse_transaction_review(
    content: Any,
    *,
    request: TransactionReviewRequest,
) -> EditorialTransactionResult:
    """Parse a reviewer response; every malformed shape safely rolls back."""

    try:
        decoded = loads_lenient(content) if isinstance(content, str) else content
        raw = _RawTransactionReview.model_validate(decoded)
    except (TypeError, ValueError, ValidationError) as exc:
        return _rollback_result(
            request,
            status=TransactionExecutionStatus.FAILED,
            diagnostic=f"transaction_review_invalid: {exc}",
        )
    if raw.manifest_sha256 != request.manifest_sha256:
        return _rollback_result(
            request,
            status=TransactionExecutionStatus.FAILED,
            diagnostic="transaction_review_manifest_mismatch",
        )

    actions = {target.pre_ref: target.intended_action for target in request.targets}
    target_reviews: list[TargetResolutionReview] = []
    try:
        for review in raw.targets:
            action = actions.get(review.pre_ref)
            if action is None:
                raise ValueError(f"unknown transaction target: {review.pre_ref}")
            target_reviews.append(
                TargetResolutionReview(
                    pre_ref=review.pre_ref,
                    intended_action=action,
                    outcome=review.outcome,
                    post_refs=review.post_refs,
                    rationale=review.rationale,
                )
            )
        preserved_reviews = tuple(
            PreservedFactReview.model_validate(review.model_dump(mode="python"))
            for review in raw.preserved
        )
        post_reviews = tuple(
            PostUnitReview.model_validate(review.model_dump(mode="python"))
            for review in raw.post_units
        )
        result_payload = {
            "manifest_sha256": request.manifest_sha256,
            "execution_status": TransactionExecutionStatus.COMPLETE,
            "target_refs": tuple(target.pre_ref for target in request.targets),
            "preserved_pre_refs": request.preserved_pre_refs,
            "post_refs": request.post_refs,
            "target_reviews": tuple(target_reviews),
            "preserved_reviews": preserved_reviews,
            "post_reviews": post_reviews,
            "answer_preservation": raw.answer_preservation,
            "answer_preservation_rationale": (
                raw.answer_preservation_rationale
            ),
        }
    except (TypeError, ValueError, ValidationError) as exc:
        # A semantically negative but closed review is represented below as a
        # complete rollback. Reaching this branch means the protocol itself did
        # not close its denominators or lineage references.
        return _rollback_result(
            request,
            status=TransactionExecutionStatus.FAILED,
            diagnostic=f"transaction_review_scope_error: {exc}",
        )

    # Try the positive mutation decision first. The result model independently
    # derives whether the semantic records permit it. A closed negative review
    # then validates as a completed rollback; malformed lineage validates as
    # neither and falls through to a protocol failure.
    try:
        return EditorialTransactionResult(
            **result_payload,
            decision=TransactionDecision.ACCEPT,
        )
    except ValidationError:
        try:
            return EditorialTransactionResult(
                **result_payload,
                decision=TransactionDecision.ROLLBACK,
            )
        except ValidationError as exc:
            return _rollback_result(
                request,
                status=TransactionExecutionStatus.FAILED,
                diagnostic=f"transaction_review_scope_error: {exc}",
            )


__all__ = [
    "AffectedAuditUnit",
    "AnswerPreservation",
    "AuditUnitInput",
    "CharacterEdit",
    "CharacterRange",
    "EditorialAffectedScope",
    "EditorialBlockSnapshot",
    "EditorialChangeManifest",
    "EditorialEditSpan",
    "EditorialIntent",
    "EditorialTransactionAudit",
    "EditorialTransactionResult",
    "PostAssessment",
    "PostLineage",
    "PostUnitReview",
    "PreservationOutcome",
    "PreservedFactReview",
    "StableBlockPair",
    "TargetResolutionOutcome",
    "TargetResolutionReview",
    "TextSpanSnapshot",
    "TransactionDecision",
    "TransactionExecutionStatus",
    "TransactionReviewRequest",
    "TransactionTargetRequirement",
    "apply_character_edits",
    "audit_unit_input_from_claim",
    "build_editorial_affected_scope",
    "build_editorial_change_manifest",
    "build_transaction_review_request",
    "build_transaction_reviewer_prompt",
    "parse_transaction_review",
]
