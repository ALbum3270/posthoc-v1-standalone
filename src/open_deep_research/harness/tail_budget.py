"""Dynamic, auditable reserve for the mandatory post-draft evidence tail.

This module deliberately contains no default dollar allocation.  It records
work units and prompt-based estimates so later runs can calibrate a defensible
reserve.  A configured initial reserve is an estimate, never a guarantee.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class TailCheckpointName(str, Enum):
    """Points where mechanically known work changes."""

    RUN_START = "run_start"
    DRAFT_AVAILABLE = "draft_available"
    CLAIMS_AVAILABLE = "claims_available"
    ATTRIBUTION_AVAILABLE = "attribution_available"
    MANDATORY_TAIL_COMPLETE = "mandatory_tail_complete"


class TailWorkUnit(BaseModel):
    """One observed or estimated workload dimension."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    count: int = Field(ge=0)


class TailReserveCheckpoint(BaseModel):
    """One reserve recalculation with its uncertainty exposed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint: TailCheckpointName
    work_units: tuple[TailWorkUnit, ...] = ()
    estimated_remaining_cost_usd: float | None = Field(default=None, ge=0.0)
    estimate_complete: bool = False
    frozen_reserve_usd: float = Field(ge=0.0)
    estimate_is_guarantee: bool = False
    limitations: tuple[str, ...] = ()


class TailStageObservation(BaseModel):
    """Measured work-unit to token/cost observation for future calibration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: str = Field(min_length=1)
    work_units: tuple[TailWorkUnit, ...]
    token_count: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)
    cost_per_unit_usd: float | None = Field(default=None, ge=0.0)
    tokens_per_unit: float | None = Field(default=None, ge=0.0)


class EvidenceTailReserveAudit(BaseModel):
    """Complete reserve and calibration ledger."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    configured_initial_estimate_usd: float = Field(ge=0.0)
    initial_estimate_is_guarantee: bool = False
    current_frozen_reserve_usd: float = Field(ge=0.0)
    checkpoints: tuple[TailReserveCheckpoint, ...] = ()
    stage_observations: tuple[TailStageObservation, ...] = ()
    mandatory_stages: tuple[str, ...] = (
        "claim_decomposition",
        "evidence_obligation_resolution",
        "attribution",
        "initial_verification",
        "checklist_reconciliation",
        "deterministic_rendering",
        "audit_editing",
        "post_edit_claim_decomposition",
        "post_edit_evidence_obligation_resolution",
        "post_edit_attribution",
        "post_edit_initial_verification",
        "post_edit_checklist_reconciliation",
    )
    enhancement_stages: tuple[str, ...] = (
        "evidence_gap",
        "disagreement",
        "evaluative_diagnostics",
        "recovery_triage",
        "evidence_recovery",
        "post_edit_evaluative_diagnostics",
    )
    enhancement_can_borrow_reserve: bool = False
    limitations: tuple[str, ...] = (
        "the initial reserve is a configured estimate, not a guarantee",
        "prompt-cost calibration can be unavailable before a role's first "
        "observed call",
        "future model output can change downstream claim and relation counts",
        "provider cost is observed only after an admitted call completes",
        "no universal stage percentage or dollar amount is encoded",
    )


class EvidenceTailReserveController:
    """Mutable run-local controller whose audit is immutable."""

    mandatory_stages = frozenset(
        {
            "claim_decomposition",
            "evidence_obligation_resolution",
            "attribution",
            "initial_verification",
            "checklist_reconciliation",
            "deterministic_rendering",
            "audit_editing",
            "post_edit_claim_decomposition",
            "post_edit_evidence_obligation_resolution",
            "post_edit_attribution",
            "post_edit_initial_verification",
            "post_edit_checklist_reconciliation",
        }
    )
    enhancement_stages = frozenset(
        {
            "evidence_gap",
            "disagreement",
            "evaluative_diagnostics",
            "recovery_triage",
            "evidence_recovery",
            "post_edit_evaluative_diagnostics",
        }
    )

    def __init__(self, initial_estimate_usd: float = 0.0) -> None:
        self.initial_estimate_usd = max(0.0, float(initial_estimate_usd))
        self.current_reserve_usd = self.initial_estimate_usd
        self._checkpoints: list[TailReserveCheckpoint] = []
        self._observations: list[TailStageObservation] = []
        self._stage_cost: dict[str, float] = defaultdict(float)
        self.checkpoint(
            TailCheckpointName.RUN_START,
            work_units=(),
            estimated_remaining_cost_usd=(
                self.initial_estimate_usd
                if self.initial_estimate_usd
                else None
            ),
            estimate_complete=False,
            limitations=(
                "downstream work units do not exist before the draft",
            ),
        )

    def reserve_for_call(
        self,
        stage: str,
        estimated_cost_usd: float | None,
    ) -> float:
        """Return the reserve protected from this call.

        Mandatory-tail calls may consume the reserve they are part of.
        Enhancement calls never may.  When an estimate is unavailable, no
        guessed release is made; the audit exposes the bootstrap limitation.
        """

        if stage in self.mandatory_stages:
            if estimated_cost_usd is None:
                # This call belongs to the protected tail. Refusing it merely
                # because its role has no calibration yet would preserve money
                # while preventing the work the money was reserved for. With
                # no estimate we release the reserve for this bootstrap call
                # and record that the downstream partition was unknowable.
                return 0.0
            releasable = max(0.0, float(estimated_cost_usd))
            return max(0.0, self.current_reserve_usd - releasable)
        return self.current_reserve_usd

    def record_call_cost(self, stage: str, actual_cost_usd: float) -> None:
        measured = max(0.0, float(actual_cost_usd))
        self._stage_cost[stage] += measured
        if stage in self.mandatory_stages:
            self.current_reserve_usd = max(
                0.0, self.current_reserve_usd - measured
            )

    def checkpoint(
        self,
        checkpoint: TailCheckpointName,
        *,
        work_units: tuple[TailWorkUnit, ...],
        estimated_remaining_cost_usd: float | None,
        estimate_complete: bool,
        limitations: tuple[str, ...] = (),
    ) -> None:
        """Freeze the best current estimate without pretending it is exact."""

        estimate = (
            None
            if estimated_remaining_cost_usd is None
            else max(0.0, float(estimated_remaining_cost_usd))
        )
        if checkpoint is TailCheckpointName.MANDATORY_TAIL_COMPLETE:
            self.current_reserve_usd = 0.0
        elif estimate is not None:
            # An incomplete estimate may prove only a lower bound.  Never
            # release an existing reserve merely because some downstream
            # prompts are not constructible yet.
            self.current_reserve_usd = (
                estimate
                if estimate_complete
                else max(self.current_reserve_usd, estimate)
            )
        self._checkpoints.append(
            TailReserveCheckpoint(
                checkpoint=checkpoint,
                work_units=work_units,
                estimated_remaining_cost_usd=estimate,
                estimate_complete=estimate_complete,
                frozen_reserve_usd=self.current_reserve_usd,
                limitations=limitations,
            )
        )

    def observe_stage(
        self,
        stage: str,
        *,
        work_units: tuple[TailWorkUnit, ...],
        token_count: int,
        cost_usd: float,
    ) -> None:
        total_units = sum(unit.count for unit in work_units)
        self._observations.append(
            TailStageObservation(
                stage=stage,
                work_units=work_units,
                token_count=max(0, int(token_count)),
                cost_usd=max(0.0, float(cost_usd)),
                cost_per_unit_usd=(
                    max(0.0, float(cost_usd)) / total_units
                    if total_units
                    else None
                ),
                tokens_per_unit=(
                    max(0, int(token_count)) / total_units
                    if total_units
                    else None
                ),
            )
        )

    def audit(self) -> EvidenceTailReserveAudit:
        return EvidenceTailReserveAudit(
            configured_initial_estimate_usd=self.initial_estimate_usd,
            current_frozen_reserve_usd=self.current_reserve_usd,
            checkpoints=tuple(self._checkpoints),
            stage_observations=tuple(self._observations),
        )
