"""Auditable source-role and lineage assessments.

Host domains are mechanically reproducible, but they are not organizations,
editorial origins, or independence determinations.  This module keeps those
semantic judgements explicit and optional.  A model may propose an assessment,
but only a separately confirmed assessment is eligible for claim-level
corroboration.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceRole(str, Enum):
    """The evidentiary role played by a source, not a domain allow-list."""

    ORIGINAL_RECORD = "original_record"
    FIRSTHAND_ACCOUNT = "firsthand_account"
    INDEPENDENT_REPORTING = "independent_reporting"
    ANALYSIS = "analysis"
    AGGREGATOR = "aggregator"
    REPUBLISHED = "republished"
    UNKNOWN = "unknown"


class SourceLineageStatus(str, Enum):
    """Whether a semantic lineage judgement exists and who may rely on it."""

    PROPOSED = "proposed"
    CONFIRMED = "confirmed"


class SourceLineageAssessment(BaseModel):
    """One reviewable source-lineage judgement bound to source bytes.

    ``proposed`` records may come from a model and remain visible, but cannot
    establish independence.  ``confirmed`` means a separate evaluator or
    human accepted the judgement; code still does not claim that this makes
    the assessment infallible.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    status: SourceLineageStatus
    source_role: SourceRole
    originating_organization: str = Field(min_length=1)
    lineage_id: str = Field(min_length=1)
    independence_eligible: bool
    evaluator: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    basis_quote: str = Field(min_length=1)
    basis_start_char: int = Field(ge=0)
    basis_end_char: int = Field(ge=0)

    @model_validator(mode="after")
    def _basis_bounds_match_quote(self) -> SourceLineageAssessment:
        if self.basis_end_char <= self.basis_start_char:
            raise ValueError("lineage basis end must exceed start")
        if self.basis_end_char - self.basis_start_char != len(self.basis_quote):
            raise ValueError("lineage basis bounds must match quote length")
        if self.source_role is SourceRole.UNKNOWN:
            raise ValueError("assessed source role cannot remain unknown")
        return self

    @property
    def establishes_independence(self) -> bool:
        """Only separately confirmed, eligible lineage can corroborate."""

        return (
            self.status is SourceLineageStatus.CONFIRMED
            and self.independence_eligible
        )


def assessment_matches_source(
    assessment: SourceLineageAssessment,
    *,
    source_id: str,
    url: str,
    source_text: str,
) -> tuple[bool, str | None]:
    """Mechanically validate identity, hash, bounds, and exact basis bytes."""

    import hashlib

    if assessment.source_id != source_id:
        return False, "lineage source_id does not match verified source"
    if assessment.url != url:
        return False, "lineage URL does not match verified source"
    digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    if assessment.source_text_sha256 != digest:
        return False, "lineage source text hash does not match cache"
    if assessment.basis_end_char > len(source_text):
        return False, "lineage basis lies outside cached source"
    if (
        source_text[assessment.basis_start_char : assessment.basis_end_char]
        != assessment.basis_quote
    ):
        return False, "lineage basis quote does not match cached source"
    return True, None
