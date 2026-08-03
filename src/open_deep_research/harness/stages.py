"""Auditable post-draft execution separated from report-quality review.

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


class QualityReviewStatus(str, Enum):
    """Independent report-quality review, not pipeline execution state."""

    NOT_REVIEWED = "not_reviewed"
    PASSED = "passed"
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


MANDATORY_PIPELINE_STAGES = (
    "claim_decomposition",
    "attribution",
    "initial_verification",
    "checklist_reconciliation",
    "deterministic_rendering",
)
# Historical import compatibility.  New audits and code use the narrower name.
MANDATORY_PUBLICATION_STAGES = MANDATORY_PIPELINE_STAGES


class PostDraftExecutionAudit(BaseModel):
    """Stage ledger separated from independent report-quality review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stages: dict[str, StageExecutionRecord]
    mandatory_pipeline_stages: tuple[str, ...] = MANDATORY_PIPELINE_STAGES
    pipeline_complete: bool
    pipeline_completion_reason: str = Field(min_length=1)
    quality_review_status: QualityReviewStatus = QualityReviewStatus.NOT_REVIEWED
    quality_review_passed: bool | None = None
    quality_review_reason: str = (
        "no independent report-quality review has been completed"
    )

    @model_validator(mode="before")
    @classmethod
    def _read_historical_publication_shape(cls, value: object) -> object:
        """Read old audits without re-emitting their over-strong semantics."""

        if not isinstance(value, dict):
            return value
        data = dict(value)
        old = data.pop("mandatory_publication_stages", None)
        if "mandatory_pipeline_stages" not in data and old is not None:
            data["mandatory_pipeline_stages"] = old
        old = data.pop("publication_eligible", None)
        if "pipeline_complete" not in data and old is not None:
            data["pipeline_complete"] = old
        old = data.pop("publication_reason", None)
        if "pipeline_completion_reason" not in data and old is not None:
            data["pipeline_completion_reason"] = old
        return data

    @model_validator(mode="after")
    def _completion_and_quality_are_independent(self) -> PostDraftExecutionAudit:
        missing = [
            name
            for name in self.mandatory_pipeline_stages
            if name not in self.stages
        ]
        if missing:
            raise ValueError(
                "publication audit is missing mandatory stages: "
                + ", ".join(missing)
            )
        all_complete = all(
            self.stages[name].status is StageExecutionStatus.COMPLETE
            for name in self.mandatory_pipeline_stages
        )
        if self.pipeline_complete != all_complete:
            raise ValueError(
                "pipeline_complete must equal mechanical completion of "
                "every mandatory pipeline stage"
            )
        expected_review_value = {
            QualityReviewStatus.NOT_REVIEWED: None,
            QualityReviewStatus.PASSED: True,
            QualityReviewStatus.FAILED: False,
        }[self.quality_review_status]
        if self.quality_review_passed is not expected_review_value:
            raise ValueError(
                "quality_review_passed must match quality_review_status"
            )
        return self

    @property
    def publication_eligible(self) -> bool:
        """Compatibility accessor; new serialized audits do not emit it."""

        return self.pipeline_complete and self.quality_review_passed is True

    @property
    def publication_reason(self) -> str:
        """Compatibility accessor that does not equate completion with quality."""

        if not self.pipeline_complete:
            return self.pipeline_completion_reason
        return self.quality_review_reason


def publication_audit(
    stages: dict[str, StageExecutionRecord],
    *,
    mandatory_stages: tuple[str, ...] | None = None,
) -> PostDraftExecutionAudit:
    """Derive pipeline completion; quality remains independently unreviewed."""

    required = mandatory_stages or MANDATORY_PIPELINE_STAGES
    complete = all(
        stages.get(name) is not None
        and stages[name].status is StageExecutionStatus.COMPLETE
        for name in required
    )
    incomplete = tuple(
        name
        for name in required
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
        mandatory_pipeline_stages=required,
        pipeline_complete=complete,
        pipeline_completion_reason=reason,
        quality_review_status=QualityReviewStatus.NOT_REVIEWED,
        quality_review_passed=None,
        quality_review_reason=(
            "pipeline completion does not establish report quality; "
            "independent verifier and preservation review are pending"
        ),
    )


# Where each stage's own output lands in the audit. A stage that claims to have
# completed must have left something behind; the pairing is what makes that
# checkable without reading eight bespoke record sites by hand.
#
# Stages absent from this mapping have no separate payload: deterministic
# rendering *is* the artifact bundle.  Evidence gap is included because its
# payload contains the accepted target routes needed to distinguish a bounded
# pass ending from every requested target actually being evaluated.
STAGE_AUDIT_PAYLOAD_KEYS = {
    "claim_decomposition": "claim_decomposition",
    "attribution": "attribution",
    "initial_verification": "verification",
    "checklist_reconciliation": "checklist_report_reconciliation",
    "evaluative_diagnostics": "evaluative_claim_diagnostics",
    "evidence_gap": "evidence_gap",
    "recovery_triage": "recovery_triage",
    "evidence_recovery": "evidence_recovery",
    "audit_editing": "editorial_revision",
    "post_edit_evaluative_diagnostics": (
        "post_edit_evaluative_claim_diagnostics"
    ),
    "post_edit_claim_decomposition": "claim_decomposition",
    "post_edit_attribution": "attribution",
    "post_edit_initial_verification": "verification",
    "post_edit_checklist_reconciliation": (
        "checklist_report_reconciliation"
    ),
}


def _evidence_gap_routed_claim_ids(payload: dict) -> tuple[str, ...]:
    """Recover substantive routing from new or historical gap payloads."""

    explicit = payload.get("routed_target_claim_ids")
    if isinstance(explicit, (list, tuple)):
        return tuple(dict.fromkeys(str(claim_id) for claim_id in explicit))
    routed: list[str] = []
    for hint in payload.get("cached_candidate_hints") or ():
        if isinstance(hint, dict) and hint.get("claim_id") is not None:
            routed.append(str(hint["claim_id"]))
    for search in payload.get("searches") or ():
        if not isinstance(search, dict):
            continue
        query = search.get("query") or {}
        if not isinstance(query, dict):
            continue
        routed.extend(str(claim_id) for claim_id in query.get("claim_ids") or ())
    return tuple(dict.fromkeys(routed))


def stages_claiming_completion_without_output(
    audit: dict,
) -> tuple[str, ...]:
    """Name complete stages missing substantive, scope-matching output.

    A run once reported ``evaluative_diagnostics`` as complete over 87 of 87
    claims while every one of its calls had been refused and its payload was
    null, because the record counted the work requested rather than the work
    done. A later gap pass left a non-null but sparse payload and made the same
    mistake by calling two routed claims 58 evaluated claims. Both shapes are
    checked mechanically here.
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
        payload = posthoc.get(payload_key)
        if (
            stage_name == "evaluative_diagnostics"
            and posthoc.get("pre_edit_evidence") is not None
        ):
            payload = posthoc["pre_edit_evidence"].get(
                "evaluative_claim_diagnostics"
            )
        if payload is None:
            offenders.append(stage_name)
            continue
        expected_scope = record.get("expected_scope") or {}
        evaluated_scope = record.get("evaluated_scope") or {}
        expected_count = expected_scope.get("count")
        evaluated_count = evaluated_scope.get("count")
        if (
            isinstance(expected_count, int)
            and expected_count > 0
            and evaluated_count != expected_count
        ):
            offenders.append(stage_name)
            continue
        if stage_name == "evidence_gap" and isinstance(payload, dict):
            routed_count = len(_evidence_gap_routed_claim_ids(payload))
            if (
                evaluated_count != routed_count
                or (
                    isinstance(expected_count, int)
                    and routed_count != expected_count
                )
            ):
                offenders.append(stage_name)
    return tuple(offenders)


# Which stage each stage's scope depends on. Attribution can only know how many
# claims exist if decomposition built the registry; verification can only know
# how many relations exist if attribution proposed them.
STAGE_SCOPE_DEPENDS_ON = {
    "attribution": "claim_decomposition",
    "evaluative_diagnostics": "claim_decomposition",
    "evidence_gap": "initial_verification",
    "recovery_triage": "initial_verification",
    "evidence_recovery": "recovery_triage",
    "initial_verification": "attribution",
    "audit_editing": "initial_verification",
    "post_edit_attribution": "post_edit_claim_decomposition",
    "post_edit_evaluative_diagnostics": "post_edit_claim_decomposition",
    "post_edit_initial_verification": "post_edit_attribution",
    "post_edit_checklist_reconciliation": "post_edit_claim_decomposition",
}


def demote_vacuous_completions(
    stages: dict[str, StageExecutionRecord],
) -> dict[str, StageExecutionRecord]:
    """Refuse ``complete`` earned only by an upstream stage being cut off.

    A run whose budget ran out before decomposition produced no claims, so
    attribution had an empty scope and reported ``complete`` over 0 of 0 --
    the exact zero-denominator this design forbids, arriving through the back
    door. An empty scope is honest only when the stage that establishes it
    actually finished; otherwise the emptiness is an artefact of truncation.

    A genuinely claim-free report still yields ``complete``, because there
    decomposition completed and simply found nothing to attribute.
    """

    adjusted = dict(stages)
    for stage_name, upstream_name in STAGE_SCOPE_DEPENDS_ON.items():
        record = adjusted.get(stage_name)
        upstream = adjusted.get(upstream_name)
        if record is None or upstream is None:
            continue
        if record.status is not StageExecutionStatus.COMPLETE:
            continue
        if upstream.status is StageExecutionStatus.COMPLETE:
            continue
        if record.expected_scope is not None and record.expected_scope.count:
            continue
        adjusted[stage_name] = StageExecutionRecord(
            status=StageExecutionStatus.NOT_RUN,
            reason=(
                f"scope was empty only because {upstream_name} did not "
                f"complete ({upstream.status.value}); nothing was evaluated"
            ),
        )
    return adjusted
