"""Read-only semantic reconciliation between a checklist and a report."""

from __future__ import annotations

import inspect
import json
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

from open_deep_research.harness.checklist import ResearchChecklist
from open_deep_research.harness.claims import (
    AtomicClaim,
    ClaimNormalizationStatus,
    MarkdownBlock,
)
from open_deep_research.harness.jsonio import loads_lenient


class ChecklistCoverageDisposition(str, Enum):
    """The model's semantic judgement about one checklist item."""

    COVERED = "covered"
    PARTIALLY_COVERED = "partially_covered"
    NOT_COVERED = "not_covered"


class CoverageAssessmentStatus(str, Enum):
    """Whether a semantic judgement survived mechanical validation."""

    COMPLETED = "completed"
    COMPLETED_WITH_ERRORS = "completed_with_errors"
    ASSESSMENT_FAILED = "assessment_failed"


class ChecklistCoverageReference(BaseModel):
    """A code-owned, mechanically located report reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    block_id: str
    anchor_text: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)


class ChecklistCoverageRecord(BaseModel):
    """One auditable semantic proposal and its mechanically checked result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str
    question: str
    proposed_disposition: ChecklistCoverageDisposition | None = None
    disposition: ChecklistCoverageDisposition | None = None
    rationale: str = ""
    references: tuple[ChecklistCoverageReference, ...] = ()
    proposed_claim_ids: tuple[str, ...] = ()
    invalid_claim_ids: tuple[str, ...] = ()
    assessment_status: CoverageAssessmentStatus
    diagnostics: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _accepted_coverage_has_a_location(self) -> ChecklistCoverageRecord:
        failed = (
            self.assessment_status
            == CoverageAssessmentStatus.ASSESSMENT_FAILED
        )
        if failed != (self.disposition is None):
            raise ValueError(
                "only failed assessments may omit the final disposition"
            )
        if self.disposition in {
            ChecklistCoverageDisposition.COVERED,
            ChecklistCoverageDisposition.PARTIALLY_COVERED,
        } and not self.references:
            raise ValueError(
                "accepted covered dispositions require a report reference"
            )
        return self


class ChecklistCoverageSummary(BaseModel):
    """Reader-facing checklist/report reconciliation counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_items: int = Field(ge=0)
    assessed_items: int = Field(ge=0)
    covered_items: int = Field(ge=0)
    partially_covered_items: int = Field(ge=0)
    not_covered_items: int = Field(ge=0)
    assessment_failed_items: int = Field(ge=0)
    covered_rate: float = Field(ge=0.0, le=1.0)
    partially_covered_item_ids: tuple[str, ...] = ()
    not_covered_item_ids: tuple[str, ...] = ()
    assessment_failed_item_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _counts_are_consistent(self) -> ChecklistCoverageSummary:
        if self.assessed_items + self.assessment_failed_items != self.total_items:
            raise ValueError("assessed and failed counts must cover the checklist")
        if (
            self.covered_items
            + self.partially_covered_items
            + self.not_covered_items
            != self.assessed_items
        ):
            raise ValueError("coverage dispositions must cover assessed items")
        if self.partially_covered_items != len(
            self.partially_covered_item_ids
        ):
            raise ValueError("partial count must match its item IDs")
        if self.not_covered_items != len(self.not_covered_item_ids):
            raise ValueError("not-covered count must match its item IDs")
        if self.assessment_failed_items != len(
            self.assessment_failed_item_ids
        ):
            raise ValueError("failure count must match its item IDs")
        expected_rate = (
            self.covered_items / self.total_items
            if self.total_items
            else 0.0
        )
        if abs(self.covered_rate - expected_rate) > 1e-12:
            raise ValueError("covered_rate must equal covered_items/total_items")
        return self


class ReconciliationCallUsage(BaseModel):
    """Measured usage for the single read-only reconciliation call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    token_count: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)


class ChecklistReportReconciliation(BaseModel):
    """Complete checklist/report coverage registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    records: tuple[ChecklistCoverageRecord, ...]
    summary: ChecklistCoverageSummary
    usage: ReconciliationCallUsage = ReconciliationCallUsage()
    diagnostics: tuple[str, ...] = ()
    affects_report_content: Literal[False] = False
    blocks_artifact_write: Literal[False] = False

    @property
    def total_tokens(self) -> int:
        return self.usage.token_count

    @property
    def total_cost_usd(self) -> float:
        return self.usage.cost_usd


class ReconciliationModelClient(Protocol):
    """Injected semantic-judgement boundary for report reconciliation."""

    def generate(self, prompt: str) -> Any | Awaitable[Any]:
        """Return reconciliation JSON in a measured usage envelope."""


class _ModelEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    content: Any
    token_count: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)


class _CoverageProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1)
    disposition: ChecklistCoverageDisposition
    reason: str = Field(min_length=1)
    claim_ids: tuple[str, ...] = ()

    @field_validator("reason")
    @classmethod
    def _reason_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("reason must not be blank")
        return normalized


_PROMPT = """\
Audit how the report addresses every frozen checklist item. This is a
read-only reconciliation pass: do not rewrite the report or checklist, do not
suggest additions, and do not make evidence-support judgements.

Judge semantic coverage:
- covered: the report directly and substantively answers the checklist item.
- partially_covered: the report addresses a relevant part but does not answer
  the item fully.
- not_covered: the report does not answer the item. Merely adjacent subject
  matter is not coverage.

Read each question for the material relationships it asks for. A causal,
temporal, comparative, actor, scale, or current-status question is only covered
when the report substantively supplies the relationship the question requests;
nearby facts or endpoints alone may be partial coverage. Decide what is
material from the item's own wording and the report. Do not apply a fixed topic
event list or a numeric coverage threshold.

For covered or partially_covered, cite one or more claim_id values from the
registry that actually carry the answer. Reusing a claim_id is allowed.
For not_covered, return an empty claim_ids list and explain what is absent.
Return exactly one entry for every checklist item. Do not invent identifiers.
Return JSON only:
{{"items":[{{"item_id":"...","disposition":"covered|partially_covered|\
not_covered","reason":"...","claim_ids":["claim-..."]}}]}}

Checklist and mechanically indexed report registry:
{payload}
"""


def build_reconciliation_prompt(
    checklist: ResearchChecklist,
    *,
    blocks: Sequence[MarkdownBlock],
    claims: Sequence[AtomicClaim],
) -> str:
    """Build the complete, read-only semantic reconciliation prompt."""

    payload = {
        "checklist": [
            {
                "dimension": item.dimension.value,
                "item_id": item.item_id,
                "question": item.question,
            }
            for item in checklist.items
        ],
        "report_blocks": [
            {
                "block_id": block.block_id,
                "kind": block.kind.value,
                "section_path": list(block.section_path),
                "text": block.text,
            }
            for block in blocks
        ],
        "report_claims": [
            {
                "anchor_text": claim.anchor_text,
                "block_id": claim.block_id,
                "claim_id": claim.claim_id,
                "claim_text": claim.claim_text,
                "normalization_status": claim.normalization_status.value,
            }
            for claim in claims
        ],
    }
    return _PROMPT.format(
        payload=json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _decode_content(content: Any) -> Mapping[str, Any]:
    if isinstance(content, str):
        content = loads_lenient(content)
    if not isinstance(content, Mapping):
        raise ValueError("reconciliation response must be a JSON object")
    return content


async def _call_model(
    model_client: ReconciliationModelClient,
    prompt: str,
) -> tuple[Any, ReconciliationCallUsage]:
    response = model_client.generate(prompt)
    if inspect.isawaitable(response):
        response = await response
    envelope = _ModelEnvelope.model_validate(response)
    return envelope.content, ReconciliationCallUsage(
        token_count=envelope.token_count,
        cost_usd=envelope.cost_usd,
    )


def _validated_reference(
    claim_id: str,
    *,
    canonical_draft: str,
    claims_by_id: Mapping[str, AtomicClaim],
    blocks_by_id: Mapping[str, MarkdownBlock],
) -> tuple[ChecklistCoverageReference | None, str | None]:
    claim = claims_by_id.get(claim_id)
    if claim is None:
        return None, "claim_id_not_found"
    if (
        claim.normalization_status != ClaimNormalizationStatus.LOCATED
        or claim.anchor_text is None
        or claim.start_char is None
        or claim.end_char is None
    ):
        return None, "claim_anchor_not_located"
    block = blocks_by_id.get(claim.block_id)
    if block is None:
        return None, "claim_block_not_found"
    if canonical_draft[block.start_char : block.end_char] != block.text:
        return None, "block_bounds_do_not_match_report"
    if not (
        block.start_char
        <= claim.start_char
        < claim.end_char
        <= block.end_char
    ):
        return None, "claim_anchor_outside_block"
    if canonical_draft[claim.start_char : claim.end_char] != claim.anchor_text:
        return None, "claim_anchor_not_verbatim"
    return (
        ChecklistCoverageReference(
            claim_id=claim.claim_id,
            block_id=claim.block_id,
            anchor_text=claim.anchor_text,
            start_char=claim.start_char,
            end_char=claim.end_char,
        ),
        None,
    )


def _summary(
    records: Sequence[ChecklistCoverageRecord],
) -> ChecklistCoverageSummary:
    completed = [
        record
        for record in records
        if record.assessment_status
        != CoverageAssessmentStatus.ASSESSMENT_FAILED
    ]
    covered = [
        record
        for record in completed
        if record.disposition == ChecklistCoverageDisposition.COVERED
    ]
    partial = [
        record
        for record in completed
        if record.disposition
        == ChecklistCoverageDisposition.PARTIALLY_COVERED
    ]
    uncovered = [
        record
        for record in completed
        if record.disposition == ChecklistCoverageDisposition.NOT_COVERED
    ]
    failed = [
        record
        for record in records
        if record.assessment_status
        == CoverageAssessmentStatus.ASSESSMENT_FAILED
    ]
    total = len(records)
    return ChecklistCoverageSummary(
        total_items=total,
        assessed_items=len(completed),
        covered_items=len(covered),
        partially_covered_items=len(partial),
        not_covered_items=len(uncovered),
        assessment_failed_items=len(failed),
        covered_rate=(len(covered) / total if total else 0.0),
        partially_covered_item_ids=tuple(
            record.item_id for record in partial
        ),
        not_covered_item_ids=tuple(
            record.item_id for record in uncovered
        ),
        assessment_failed_item_ids=tuple(
            record.item_id for record in failed
        ),
    )


def _failed_records(
    checklist: ResearchChecklist,
    diagnostic: str,
) -> tuple[ChecklistCoverageRecord, ...]:
    return tuple(
        ChecklistCoverageRecord(
            item_id=item.item_id,
            question=item.question,
            assessment_status=CoverageAssessmentStatus.ASSESSMENT_FAILED,
            diagnostics=(diagnostic,),
        )
        for item in checklist.items
    )


async def reconcile_checklist_report(
    canonical_draft: str,
    checklist: ResearchChecklist,
    *,
    blocks: Sequence[MarkdownBlock],
    claims: Sequence[AtomicClaim],
    model_client: ReconciliationModelClient,
) -> ChecklistReportReconciliation:
    """Judge semantic coverage, then mechanically validate every citation."""

    prompt = build_reconciliation_prompt(
        checklist,
        blocks=blocks,
        claims=claims,
    )
    try:
        content, usage = await _call_model(model_client, prompt)
    except Exception as exc:
        diagnostic = f"reconciliation_model_error: {exc}"
        records = _failed_records(checklist, diagnostic)
        return ChecklistReportReconciliation(
            records=records,
            summary=_summary(records),
            diagnostics=(diagnostic,),
        )
    try:
        payload = _decode_content(content)
    except (TypeError, ValueError) as exc:
        diagnostic = f"reconciliation_output_invalid: {exc}"
        records = _failed_records(checklist, diagnostic)
        return ChecklistReportReconciliation(
            records=records,
            summary=_summary(records),
            usage=usage,
            diagnostics=(diagnostic,),
        )

    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        diagnostic = "reconciliation_items_missing_or_not_array"
        records = _failed_records(checklist, diagnostic)
        return ChecklistReportReconciliation(
            records=records,
            summary=_summary(records),
            usage=usage,
            diagnostics=(diagnostic,),
        )

    checklist_by_id = {item.item_id: item for item in checklist.items}
    parsed_by_id: dict[str, list[_CoverageProposal]] = {}
    item_errors: dict[str, list[str]] = {}
    global_diagnostics: list[str] = []
    for index, raw in enumerate(raw_items):
        try:
            proposal = _CoverageProposal.model_validate(raw)
        except (TypeError, ValueError, ValidationError) as exc:
            raw_id = raw.get("item_id") if isinstance(raw, Mapping) else None
            diagnostic = f"entry_invalid[{index}]: {exc}"
            if isinstance(raw_id, str) and raw_id in checklist_by_id:
                item_errors.setdefault(raw_id, []).append(diagnostic)
            else:
                global_diagnostics.append(diagnostic)
            continue
        if proposal.item_id not in checklist_by_id:
            global_diagnostics.append(
                f"unknown_item_id[{index}]: {proposal.item_id}"
            )
            continue
        parsed_by_id.setdefault(proposal.item_id, []).append(proposal)

    claims_by_id = {claim.claim_id: claim for claim in claims}
    blocks_by_id = {block.block_id: block for block in blocks}
    records: list[ChecklistCoverageRecord] = []
    for item in checklist.items:
        proposals = parsed_by_id.get(item.item_id, [])
        diagnostics = list(item_errors.get(item.item_id, ()))
        if not proposals:
            diagnostics.append("checklist_item_omitted")
            records.append(
                ChecklistCoverageRecord(
                    item_id=item.item_id,
                    question=item.question,
                    assessment_status=(
                        CoverageAssessmentStatus.ASSESSMENT_FAILED
                    ),
                    diagnostics=tuple(diagnostics),
                )
            )
            continue
        if len(proposals) > 1:
            diagnostics.append("duplicate_checklist_item_entries")
            records.append(
                ChecklistCoverageRecord(
                    item_id=item.item_id,
                    question=item.question,
                    assessment_status=(
                        CoverageAssessmentStatus.ASSESSMENT_FAILED
                    ),
                    diagnostics=tuple(diagnostics),
                )
            )
            continue

        proposal = proposals[0]
        references: list[ChecklistCoverageReference] = []
        invalid_claim_ids: list[str] = []
        for claim_id in dict.fromkeys(proposal.claim_ids):
            reference, error = _validated_reference(
                claim_id,
                canonical_draft=canonical_draft,
                claims_by_id=claims_by_id,
                blocks_by_id=blocks_by_id,
            )
            if reference is None:
                invalid_claim_ids.append(claim_id)
                diagnostics.append(f"{claim_id}: {error}")
            else:
                references.append(reference)

        needs_reference = proposal.disposition in {
            ChecklistCoverageDisposition.COVERED,
            ChecklistCoverageDisposition.PARTIALLY_COVERED,
        }
        if needs_reference and not references:
            diagnostics.append(
                f"{proposal.disposition.value}_without_valid_report_reference"
            )
            records.append(
                ChecklistCoverageRecord(
                    item_id=item.item_id,
                    question=item.question,
                    proposed_disposition=proposal.disposition,
                    rationale=proposal.reason,
                    proposed_claim_ids=proposal.claim_ids,
                    invalid_claim_ids=tuple(invalid_claim_ids),
                    assessment_status=(
                        CoverageAssessmentStatus.ASSESSMENT_FAILED
                    ),
                    diagnostics=tuple(diagnostics),
                )
            )
            continue

        if (
            proposal.disposition
            == ChecklistCoverageDisposition.NOT_COVERED
            and proposal.claim_ids
        ):
            diagnostics.append("not_covered_with_claim_references")

        records.append(
            ChecklistCoverageRecord(
                item_id=item.item_id,
                question=item.question,
                proposed_disposition=proposal.disposition,
                disposition=proposal.disposition,
                rationale=proposal.reason,
                references=tuple(references),
                proposed_claim_ids=proposal.claim_ids,
                invalid_claim_ids=tuple(invalid_claim_ids),
                assessment_status=(
                    CoverageAssessmentStatus.COMPLETED_WITH_ERRORS
                    if diagnostics
                    else CoverageAssessmentStatus.COMPLETED
                ),
                diagnostics=tuple(diagnostics),
            )
        )

    result_records = tuple(records)
    return ChecklistReportReconciliation(
        records=result_records,
        summary=_summary(result_records),
        usage=usage,
        diagnostics=tuple(global_diagnostics),
    )
