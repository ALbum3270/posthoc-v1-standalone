"""Non-gating diagnostics for underspecified evaluative claims."""

from __future__ import annotations

import hashlib
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

from open_deep_research.harness.claims import (
    AtomicClaim,
    CitationRequirement,
    ClaimStageUsage,
)
from open_deep_research.harness.jsonio import loads_lenient


class EvaluativeUnderspecification(str, Enum):
    """Stable, deliberately coarse kinds of missing evaluative boundaries."""

    COMPARISON_SCOPE_UNSPECIFIED = "comparison_scope_unspecified"
    EVALUATION_CRITERION_UNSPECIFIED = "evaluation_criterion_unspecified"
    TEMPORAL_SCOPE_UNSPECIFIED = "temporal_scope_unspecified"


class EvaluativeDiagnosticStatus(str, Enum):
    """Whether an external claim received a usable diagnostic."""

    NOT_UNDERSPECIFIED = "not_underspecified"
    UNDERSPECIFIED = "underspecified_evaluative_claim"
    DIAGNOSTIC_FAILED = "diagnostic_failed"


class EvaluativeClaimAssessment(BaseModel):
    """One advisory diagnosis attached by claim_id, never by mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    status: EvaluativeDiagnosticStatus
    categories: tuple[EvaluativeUnderspecification, ...] = ()
    reason: str

    @field_validator("reason")
    @classmethod
    def _reason_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("diagnostic reason must not be blank")
        return normalized

    @model_validator(mode="after")
    def _categories_match_status(self) -> EvaluativeClaimAssessment:
        if (
            self.status is EvaluativeDiagnosticStatus.UNDERSPECIFIED
            and not self.categories
        ):
            raise ValueError(
                "underspecified_evaluative_claim requires a category"
            )
        if (
            self.status is not EvaluativeDiagnosticStatus.UNDERSPECIFIED
            and self.categories
        ):
            raise ValueError(
                "only underspecified_evaluative_claim may have categories"
            )
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("diagnostic categories must be unique")
        return self


class EvaluativeDiagnosticSettings(BaseModel):
    """Mechanical capacity limit for the independent diagnostic pass."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(default=8, ge=1)


class EvaluativeDiagnosticBatch(BaseModel):
    """Per-batch coverage so omissions never become negative diagnoses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_number: int = Field(ge=1)
    input_claim_ids: tuple[str, ...]
    assessed_claim_ids: tuple[str, ...] = ()
    failed_claim_ids: tuple[str, ...] = ()
    outcome: Literal["completed", "partial", "failed"]
    error: str | None = None
    usage: ClaimStageUsage = ClaimStageUsage()


class EvaluativeDiagnosticResult(BaseModel):
    """Advisory registry whose schema cannot rewrite or filter claims."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assessments: tuple[EvaluativeClaimAssessment, ...] = ()
    batches: tuple[EvaluativeDiagnosticBatch, ...] = ()
    diagnostics: tuple[str, ...] = ()
    usage: ClaimStageUsage = ClaimStageUsage()
    registry_claim_count: int = Field(ge=0)
    external_denominator_before: int = Field(ge=0)
    external_denominator_after: int = Field(ge=0)
    external_claim_ids: tuple[str, ...] = ()
    underspecified_claim_count: int = Field(ge=0)
    diagnostic_failed_count: int = Field(ge=0)
    category_counts: dict[str, int] = Field(default_factory=dict)
    claim_registry_sha256_before: str
    claim_registry_sha256_after: str
    claim_registry_unchanged: bool
    citation_requirements_unchanged: bool
    diagnostic_is_non_gating: bool = True

    @model_validator(mode="after")
    def _prove_non_gating_invariants(self) -> EvaluativeDiagnosticResult:
        if (
            self.external_denominator_before
            != self.external_denominator_after
        ):
            raise ValueError(
                "evaluative diagnostics cannot change the external denominator"
            )
        if self.claim_registry_sha256_before != (
            self.claim_registry_sha256_after
        ):
            raise ValueError(
                "evaluative diagnostics cannot mutate the claim registry"
            )
        if not self.claim_registry_unchanged:
            raise ValueError("claim_registry_unchanged must remain true")
        if not self.citation_requirements_unchanged:
            raise ValueError(
                "citation_requirements_unchanged must remain true"
            )
        assessment_ids = tuple(
            assessment.claim_id for assessment in self.assessments
        )
        if assessment_ids != self.external_claim_ids:
            raise ValueError(
                "every external claim must retain one ordered assessment"
            )
        if self.underspecified_claim_count != sum(
            assessment.status
            is EvaluativeDiagnosticStatus.UNDERSPECIFIED
            for assessment in self.assessments
        ):
            raise ValueError("underspecified count must match assessments")
        if self.diagnostic_failed_count != sum(
            assessment.status
            is EvaluativeDiagnosticStatus.DIAGNOSTIC_FAILED
            for assessment in self.assessments
        ):
            raise ValueError("failed count must match assessments")
        return self

    @property
    def total_tokens(self) -> int:
        return self.usage.token_count

    @property
    def total_cost_usd(self) -> float:
        return self.usage.cost_usd


class EvaluativeDiagnosticModelClient(Protocol):
    """Injected model used only for this advisory semantic judgement."""

    def generate(self, prompt: str) -> Any | Awaitable[Any]:
        """Return JSON in the measured usage envelope."""


class _Envelope(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    content: Any
    token_count: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)


_PROMPT = """\
Run an independent, non-gating audit over retained external claims. Return
json only. This step cannot delete a claim, alter citation_requirement,
change evidence state, or remove anything from a denominator.

For every claim_id, decide whether evaluative or comparative wording lacks an
explicit boundary needed to interpret it. Use zero or more of these stable
categories:
- comparison_scope_unspecified: the comparison set, population, or baseline
  is not defined;
- evaluation_criterion_unspecified: a degree or evaluative term has no stated
  operational criterion;
- temporal_scope_unspecified: a relative historical or recent-time range has
  no explicit interval.

These categories may overlap. Do not diagnose a claim merely because it uses
ordinary uncertainty or approximate wording. A diagnosed claim remains an
external claim requiring the same evidence treatment as before.

Return exactly one entry per claim_id:
{{"claims":[{{"claim_id":"claim-0001",\
"status":"not_underspecified|underspecified_evaluative_claim",\
"categories":["comparison_scope_unspecified"],\
"reason":"brief audit reason"}}]}}

If status is not_underspecified, categories must be empty. Do not return a
diagnostic_failed status; code owns failure records.

Frozen external claims:
{claims}
"""


def build_evaluative_diagnostic_prompt(
    claims: Sequence[AtomicClaim],
) -> str:
    """Build a prompt that cannot redefine claim inclusion or evidence need."""

    payload = [
        {
            "claim_id": claim.claim_id,
            "selected_text": claim.selected_text,
            "claim_text": claim.claim_text,
            "anchor_text": claim.anchor_text,
            "citation_requirement": claim.citation_requirement.value,
        }
        for claim in claims
    ]
    return _PROMPT.format(
        claims=json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def _fingerprint(claims: Sequence[AtomicClaim]) -> str:
    payload = json.dumps(
        [claim.model_dump(mode="json") for claim in claims],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _chunks(
    claims: Sequence[AtomicClaim],
    size: int,
) -> tuple[tuple[AtomicClaim, ...], ...]:
    return tuple(
        tuple(claims[start : start + size])
        for start in range(0, len(claims), size)
    )


async def _call(
    model_client: EvaluativeDiagnosticModelClient,
    prompt: str,
) -> tuple[Any, ClaimStageUsage]:
    response = model_client.generate(prompt)
    if inspect.isawaitable(response):
        response = await response
    envelope = _Envelope.model_validate(response)
    content = envelope.content
    if isinstance(content, str):
        content = loads_lenient(content)
    return content, ClaimStageUsage(
        token_count=envelope.token_count,
        cost_usd=envelope.cost_usd,
    )


def _failure(claim_id: str, reason: str) -> EvaluativeClaimAssessment:
    return EvaluativeClaimAssessment(
        claim_id=claim_id,
        status=EvaluativeDiagnosticStatus.DIAGNOSTIC_FAILED,
        reason=reason,
    )


def _parse_batch(
    content: Any,
    claims: Sequence[AtomicClaim],
) -> tuple[
    tuple[EvaluativeClaimAssessment, ...],
    tuple[str, ...],
]:
    expected = tuple(claim.claim_id for claim in claims)
    expected_set = set(expected)
    raw_entries = (
        content.get("claims") if isinstance(content, Mapping) else None
    )
    diagnostics: list[str] = []
    if not isinstance(raw_entries, (list, tuple)):
        return (
            tuple(
                _failure(claim_id, "diagnostic payload was not an array")
                for claim_id in expected
            ),
            ("evaluative_diagnostic_payload_invalid",),
        )

    by_id: dict[str, EvaluativeClaimAssessment] = {}
    invalid_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for index, raw in enumerate(raw_entries):
        raw_id = raw.get("claim_id") if isinstance(raw, Mapping) else None
        try:
            assessment = EvaluativeClaimAssessment.model_validate(raw)
            if assessment.status is (
                EvaluativeDiagnosticStatus.DIAGNOSTIC_FAILED
            ):
                raise ValueError("model cannot declare diagnostic_failed")
        except (TypeError, ValidationError, ValueError) as exc:
            diagnostics.append(
                f"evaluative_diagnostic_entry_invalid[{index}]: {exc}"
            )
            if isinstance(raw_id, str) and raw_id in expected_set:
                invalid_ids.add(raw_id)
            continue
        if assessment.claim_id not in expected_set:
            diagnostics.append(
                "evaluative_diagnostic_unknown_claim: "
                f"{assessment.claim_id}"
            )
            continue
        if assessment.claim_id in by_id:
            duplicate_ids.add(assessment.claim_id)
            by_id.pop(assessment.claim_id, None)
            diagnostics.append(
                "evaluative_diagnostic_duplicate_claim: "
                f"{assessment.claim_id}"
            )
            continue
        if assessment.claim_id in duplicate_ids:
            continue
        by_id[assessment.claim_id] = assessment

    ordered: list[EvaluativeClaimAssessment] = []
    for claim_id in expected:
        assessment = by_id.get(claim_id)
        if assessment is not None:
            ordered.append(assessment)
            continue
        if claim_id in duplicate_ids:
            reason = "diagnostic returned the claim more than once"
        elif claim_id in invalid_ids:
            reason = "diagnostic entry was invalid"
        else:
            reason = "diagnostic omitted this external claim"
        diagnostics.append(
            f"evaluative_diagnostic_failed: {claim_id}: {reason}"
        )
        ordered.append(_failure(claim_id, reason))
    return tuple(ordered), tuple(diagnostics)


async def diagnose_underspecified_evaluative_claims(
    claims: Sequence[AtomicClaim],
    *,
    model_client: EvaluativeDiagnosticModelClient,
    settings: EvaluativeDiagnosticSettings | None = None,
) -> EvaluativeDiagnosticResult:
    """Annotate every external claim without changing the frozen registry."""

    frozen_claims = tuple(claims)
    fingerprint_before = _fingerprint(frozen_claims)
    citation_requirements_before = tuple(
        (claim.claim_id, claim.citation_requirement)
        for claim in frozen_claims
    )
    external = tuple(
        claim
        for claim in frozen_claims
        if claim.citation_requirement is CitationRequirement.EXTERNAL
    )
    active_settings = settings or EvaluativeDiagnosticSettings()
    assessments: list[EvaluativeClaimAssessment] = []
    batches: list[EvaluativeDiagnosticBatch] = []
    diagnostics: list[str] = []
    usages: list[ClaimStageUsage] = []

    for batch_number, batch in enumerate(
        _chunks(external, active_settings.batch_size),
        start=1,
    ):
        input_ids = tuple(claim.claim_id for claim in batch)
        error: str | None = None
        try:
            content, usage = await _call(
                model_client,
                build_evaluative_diagnostic_prompt(batch),
            )
            batch_assessments, batch_diagnostics = _parse_batch(
                content,
                batch,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            usage = ClaimStageUsage()
            batch_assessments = tuple(
                _failure(claim_id, error) for claim_id in input_ids
            )
            batch_diagnostics = (
                f"evaluative_diagnostic_batch_error[{batch_number}] "
                f"claim_ids={','.join(input_ids)}: {error}",
            )
        assessments.extend(batch_assessments)
        diagnostics.extend(batch_diagnostics)
        usages.append(usage)
        failed_ids = tuple(
            assessment.claim_id
            for assessment in batch_assessments
            if assessment.status
            is EvaluativeDiagnosticStatus.DIAGNOSTIC_FAILED
        )
        assessed_ids = tuple(
            claim_id for claim_id in input_ids if claim_id not in failed_ids
        )
        outcome = (
            "completed"
            if not failed_ids
            else ("failed" if len(failed_ids) == len(input_ids) else "partial")
        )
        batches.append(
            EvaluativeDiagnosticBatch(
                batch_number=batch_number,
                input_claim_ids=input_ids,
                assessed_claim_ids=assessed_ids,
                failed_claim_ids=failed_ids,
                outcome=outcome,
                error=error,
                usage=usage,
            )
        )

    fingerprint_after = _fingerprint(frozen_claims)
    citation_requirements_after = tuple(
        (claim.claim_id, claim.citation_requirement)
        for claim in frozen_claims
    )
    category_counts = {
        category.value: sum(
            category in assessment.categories
            for assessment in assessments
        )
        for category in EvaluativeUnderspecification
    }
    return EvaluativeDiagnosticResult(
        assessments=tuple(assessments),
        batches=tuple(batches),
        diagnostics=tuple(diagnostics),
        usage=ClaimStageUsage(
            token_count=sum(usage.token_count for usage in usages),
            cost_usd=sum(usage.cost_usd for usage in usages),
        ),
        registry_claim_count=len(frozen_claims),
        external_denominator_before=len(external),
        external_denominator_after=len(external),
        external_claim_ids=tuple(claim.claim_id for claim in external),
        underspecified_claim_count=sum(
            assessment.status
            is EvaluativeDiagnosticStatus.UNDERSPECIFIED
            for assessment in assessments
        ),
        diagnostic_failed_count=sum(
            assessment.status
            is EvaluativeDiagnosticStatus.DIAGNOSTIC_FAILED
            for assessment in assessments
        ),
        category_counts=category_counts,
        claim_registry_sha256_before=fingerprint_before,
        claim_registry_sha256_after=fingerprint_after,
        claim_registry_unchanged=fingerprint_before == fingerprint_after,
        citation_requirements_unchanged=(
            citation_requirements_before == citation_requirements_after
        ),
    )
