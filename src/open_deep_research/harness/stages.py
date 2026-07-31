"""Auditable post-draft stage execution and publication eligibility.

The absence of a stage result is not a zero-valued result.  These records keep
``not_run`` and ``partial`` explicit so a cost cutoff cannot silently become
``no_candidate_source``, ``0/0`` coverage, or any other domain conclusion.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StageExecutionStatus(str, Enum):
    """How far one post-draft stage actually ran."""

    NOT_RUN = "not_run"
    PARTIAL = "partial"
    COMPLETE = "complete"
    FAILED = "failed"


class StageScope(BaseModel):
    """A mechanically countable unit of stage work."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit: str = Field(min_length=1)
    count: int = Field(ge=0)


class StageExecutionRecord(BaseModel):
    """One stage result without manufacturing unobserved denominators."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: StageExecutionStatus
    reason: str = Field(min_length=1)
    expected_scope: StageScope | None = None
    evaluated_scope: StageScope | None = None
    unevaluated_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _scope_is_honest(self) -> StageExecutionRecord:
        expected = self.expected_scope
        evaluated = self.evaluated_scope
        if expected is None:
            if evaluated is not None:
                raise ValueError(
                    "evaluated_scope requires a mechanically established "
                    "expected_scope"
                )
            if self.unevaluated_ids:
                raise ValueError(
                    "unevaluated_ids require a mechanically established scope"
                )
            return self
        if evaluated is not None and evaluated.unit != expected.unit:
            raise ValueError("expected and evaluated scope units must match")
        if evaluated is not None and evaluated.count > expected.count:
            raise ValueError("evaluated scope cannot exceed expected scope")
        if len(self.unevaluated_ids) > expected.count:
            raise ValueError("unevaluated IDs cannot exceed expected scope")
        if self.status is StageExecutionStatus.NOT_RUN:
            if evaluated is not None and evaluated.count:
                raise ValueError("not_run cannot claim evaluated work")
        if self.status is StageExecutionStatus.COMPLETE:
            if evaluated is None or evaluated.count != expected.count:
                raise ValueError(
                    "complete requires the full expected scope to be evaluated"
                )
            if self.unevaluated_ids:
                raise ValueError("complete cannot retain unevaluated IDs")
        return self


MANDATORY_PUBLICATION_STAGES = (
    "claim_decomposition",
    "attribution",
    "initial_verification",
    "checklist_reconciliation",
    "deterministic_rendering",
)


class PostDraftExecutionAudit(BaseModel):
    """Stage ledger plus a mechanically validated publication decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stages: dict[str, StageExecutionRecord]
    mandatory_publication_stages: tuple[str, ...] = (
        MANDATORY_PUBLICATION_STAGES
    )
    publication_eligible: bool
    publication_reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _publication_is_mechanically_gated(self) -> PostDraftExecutionAudit:
        missing = [
            name
            for name in self.mandatory_publication_stages
            if name not in self.stages
        ]
        if missing:
            raise ValueError(
                "publication audit is missing mandatory stages: "
                + ", ".join(missing)
            )
        all_complete = all(
            self.stages[name].status is StageExecutionStatus.COMPLETE
            for name in self.mandatory_publication_stages
        )
        if self.publication_eligible != all_complete:
            raise ValueError(
                "publication_eligible must equal mechanical completion of "
                "every mandatory publication stage"
            )
        return self


def publication_audit(
    stages: dict[str, StageExecutionRecord],
) -> PostDraftExecutionAudit:
    """Derive publication eligibility; callers cannot assert it themselves."""

    complete = all(
        stages.get(name) is not None
        and stages[name].status is StageExecutionStatus.COMPLETE
        for name in MANDATORY_PUBLICATION_STAGES
    )
    incomplete = tuple(
        name
        for name in MANDATORY_PUBLICATION_STAGES
        if stages.get(name) is None
        or stages[name].status is not StageExecutionStatus.COMPLETE
    )
    reason = (
        "all mandatory post-draft stages completed"
        if complete
        else "mandatory stages incomplete: " + ", ".join(incomplete)
    )
    return PostDraftExecutionAudit(
        stages=stages,
        publication_eligible=complete,
        publication_reason=reason,
    )


# Where each stage's own output lands in the audit. A stage that claims to have
# completed must have left something behind; the pairing is what makes that
# checkable without reading eight bespoke record sites by hand.
#
# Stages absent from this mapping have no separate payload: deterministic
# rendering *is* the artifact bundle, and the enhancement passes record their
# outcome in their own stop reasons rather than a posthoc_evidence entry.
STAGE_AUDIT_PAYLOAD_KEYS = {
    "claim_decomposition": "claim_decomposition",
    "attribution": "attribution",
    "initial_verification": "verification",
    "checklist_reconciliation": "checklist_report_reconciliation",
    "evaluative_diagnostics": "evaluative_claim_diagnostics",
}


def stages_claiming_completion_without_output(
    audit: dict,
) -> tuple[str, ...]:
    """Name stages recorded as complete whose audit payload is missing.

    A run once reported ``evaluative_diagnostics`` as complete over 87 of 87
    claims while every one of its calls had been refused and its payload was
    null, because the record counted the work requested rather than the work
    done. That shape is invisible to per-stage review and trivially visible
    here, so it is checked mechanically instead.
    """

    posthoc = audit.get("posthoc_evidence") or {}
    stages = (posthoc.get("stage_execution") or {}).get("stages") or {}
    offenders = []
    for stage_name, payload_key in sorted(STAGE_AUDIT_PAYLOAD_KEYS.items()):
        record = stages.get(stage_name)
        if record is None:
            continue
        if record.get("status") != StageExecutionStatus.COMPLETE.value:
            continue
        if posthoc.get(payload_key) is None:
            offenders.append(stage_name)
    return tuple(offenders)
