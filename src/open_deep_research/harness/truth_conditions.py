"""Truth-condition registries and element-level verification aggregation.

This module deliberately contains no provider calls and no domain heuristics.
Models propose and independently review semantic elements; code owns stable
identifiers, denominator closure, execution accounting, and aggregation.  A
``None`` registry is an explicit legacy mode rather than an empty successful
registry.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


REGISTRY_VERSION = "truth-condition-elements-v1"


_TRUTH_CONDITION_COMPLETENESS_GUIDANCE = (
    "For each claim, explicitly consider every semantic axis in turn: entity "
    "identity and relationship direction; number, unit, and comparison; "
    "attribution; negation and exception; time and order; causality; purpose, "
    "use, and disposition; and modality, uncertainty, and qualification. Only "
    "retain material conditions actually asserted by the authoritative claim "
    "wording. Omit an axis when it is not asserted; do not manufacture an "
    "element merely to cover the checklist. Context may resolve references or "
    "ellipsis but must not add a condition. This is a general semantic reasoning "
    "checklist, not a field-specific vocabulary, an output schema, or a request "
    "for per-axis Boolean flags. Do not add JSON keys or flags for the axes. You, "
    "not code, must decide which asserted conditions are material and whether "
    "the element list is complete. "
)


class TruthConditionProtocol(str, Enum):
    """Verification protocol selected for one run."""

    LEGACY_WHOLE_CLAIM = "legacy_whole_claim_v1"
    ELEMENT_REGISTRY = "truth_condition_elements_v1"


class ElementizationSemanticStatus(str, Enum):
    """Independent review of whether the semantic denominator is adequate."""

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    UNCERTAIN = "uncertain"


class ElementizationExecutionStatus(str, Enum):
    """Mechanical outcome of the proposal-and-review protocol."""

    COMPLETE = "complete"
    INVALID_RESPONSE = "invalid_response"
    MODEL_ERROR = "model_error"
    NOT_RUN = "not_run"


class ElementizationStage(str, Enum):
    """Protocol stage that failed."""

    PROPOSAL = "proposal"
    REVIEW = "review"


class ExecutionCompleteness(str, Enum):
    """Execution coverage, kept separate from semantic correctness."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    NOT_RUN = "not_run"


class ElementVerificationVerdict(str, Enum):
    """Model judgement for one element against one source."""

    SUPPORTS = "supports"
    DOES_NOT_SUPPORT = "does_not_support"
    CONTRADICTS = "contradicts"
    NOT_ENOUGH_INFORMATION = "not_enough_information"


class ElementAssessmentExecutionStatus(str, Enum):
    """Mechanical outcome for one expected element-source assessment."""

    COMPLETE = "complete"
    QUOTE_UNLOCATABLE = "quote_unlocatable"
    MODEL_ERROR = "model_error"
    SOURCE_UNAVAILABLE = "source_unavailable"
    NOT_RUN = "not_run"


class ElementSemanticState(str, Enum):
    """Aggregated semantic state for one registered element."""

    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    CONTRADICTED = "contradicted"
    CONFLICTED = "conflicted"
    UNRESOLVED = "unresolved"


class ClaimCoverageState(str, Enum):
    """Truth-condition coverage for one claim."""

    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    MIXED = "mixed"
    NOT_SUPPORTED = "not_supported"
    CONTRADICTED = "contradicted"
    CONFLICTED = "conflicted"
    UNRESOLVED = "unresolved"


def _clean_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _clean_text_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be an array of strings")
    cleaned = tuple(_clean_text(item, field_name=field_name) for item in value)
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{field_name} cannot contain exact duplicates")
    return cleaned


def _clean_unique_ids(values: Iterable[str], *, field_name: str) -> tuple[str, ...]:
    cleaned = tuple(_clean_text(value, field_name=field_name) for value in values)
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{field_name} must be unique")
    return cleaned


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def make_truth_condition_element_id(claim_id: str, ordinal: int) -> str:
    """Allocate a deterministic element identifier owned by code."""

    clean_claim_id = _clean_text(claim_id, field_name="claim_id")
    if ordinal < 0:
        raise ValueError("ordinal must be non-negative")
    return f"{clean_claim_id}::tc-{ordinal + 1:04d}"


class ElementizationProposal(BaseModel):
    """Untrusted model proposal before independent semantic review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    elements: tuple[str, ...] = ()
    rationale: str | None = None

    @field_validator("claim_id", mode="before")
    @classmethod
    def _claim_id_is_clean(cls, value: object) -> str:
        return _clean_text(value, field_name="claim_id")

    @field_validator("elements", mode="before")
    @classmethod
    def _elements_are_clean(cls, value: object) -> tuple[str, ...]:
        return _clean_text_tuple(value, field_name="elements")

    @field_validator("rationale", mode="before")
    @classmethod
    def _rationale_is_clean(cls, value: object) -> str | None:
        if value is None:
            return None
        return _clean_text(value, field_name="rationale")


class ElementizationReview(BaseModel):
    """Independent semantic review that may replace the proposed elements."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    semantic_status: ElementizationSemanticStatus
    elements: tuple[str, ...] = ()
    missing_conditions: tuple[str, ...] = ()
    rationale: str = Field(min_length=1)

    @field_validator("claim_id", "rationale", mode="before")
    @classmethod
    def _text_is_clean(cls, value: object, info: object) -> str:
        return _clean_text(value, field_name=getattr(info, "field_name", "text"))

    @field_validator("elements", "missing_conditions", mode="before")
    @classmethod
    def _tuples_are_clean(cls, value: object, info: object) -> tuple[str, ...]:
        return _clean_text_tuple(
            value,
            field_name=getattr(info, "field_name", "items"),
        )

    @model_validator(mode="after")
    def _status_matches_payload(self) -> ElementizationReview:
        if self.semantic_status is ElementizationSemanticStatus.COMPLETE:
            if not self.elements:
                raise ValueError("complete review requires at least one element")
            if self.missing_conditions:
                raise ValueError("complete review cannot report missing conditions")
        elif self.semantic_status is ElementizationSemanticStatus.INCOMPLETE:
            if not self.missing_conditions:
                raise ValueError("incomplete review requires missing_conditions")
        return self


class ElementizationFailure(BaseModel):
    """Recoverable protocol failure attributed to one expected claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    stage: ElementizationStage
    execution_status: ElementizationExecutionStatus
    diagnostic: str = Field(min_length=1)

    @field_validator("claim_id", "diagnostic", mode="before")
    @classmethod
    def _text_is_clean(cls, value: object, info: object) -> str:
        return _clean_text(value, field_name=getattr(info, "field_name", "text"))

    @model_validator(mode="after")
    def _failure_is_not_complete(self) -> ElementizationFailure:
        if self.execution_status is ElementizationExecutionStatus.COMPLETE:
            raise ValueError("failure execution_status cannot be complete")
        return self


class ParsedElementizationProposals(BaseModel):
    """Partially recoverable proposal parse with a closed denominator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_claim_ids: tuple[str, ...]
    proposals: tuple[ElementizationProposal, ...]
    failures: tuple[ElementizationFailure, ...]
    diagnostics: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _denominator_is_closed(self) -> ParsedElementizationProposals:
        if len(set(self.expected_claim_ids)) != len(self.expected_claim_ids):
            raise ValueError("expected proposal claim IDs must be unique")
        expected = set(self.expected_claim_ids)
        proposed = {item.claim_id for item in self.proposals}
        failed = {item.claim_id for item in self.failures}
        if len(proposed) != len(self.proposals) or len(failed) != len(self.failures):
            raise ValueError("proposal parse claim IDs must be unique")
        if proposed & failed or proposed | failed != expected:
            raise ValueError("proposal parse must partition every expected claim")
        if any(item.stage is not ElementizationStage.PROPOSAL for item in self.failures):
            raise ValueError("proposal parse failures must identify proposal stage")
        return self


class ParsedElementizationReviews(BaseModel):
    """Partially recoverable review parse with a closed denominator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expected_claim_ids: tuple[str, ...]
    reviews: tuple[ElementizationReview, ...]
    failures: tuple[ElementizationFailure, ...]
    diagnostics: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _denominator_is_closed(self) -> ParsedElementizationReviews:
        if len(set(self.expected_claim_ids)) != len(self.expected_claim_ids):
            raise ValueError("expected review claim IDs must be unique")
        expected = set(self.expected_claim_ids)
        reviewed = {item.claim_id for item in self.reviews}
        failed = {item.claim_id for item in self.failures}
        if len(reviewed) != len(self.reviews) or len(failed) != len(self.failures):
            raise ValueError("review parse claim IDs must be unique")
        if reviewed & failed or reviewed | failed != expected:
            raise ValueError("review parse must partition every expected claim")
        if any(item.stage is not ElementizationStage.REVIEW for item in self.failures):
            raise ValueError("review parse failures must identify review stage")
        return self


def _invalid_failures(
    expected_claim_ids: tuple[str, ...],
    *,
    stage: ElementizationStage,
    diagnostic: str,
) -> tuple[ElementizationFailure, ...]:
    return tuple(
        ElementizationFailure(
            claim_id=claim_id,
            stage=stage,
            execution_status=ElementizationExecutionStatus.INVALID_RESPONSE,
            diagnostic=diagnostic,
        )
        for claim_id in expected_claim_ids
    )


def _parse_json_object(content: str) -> tuple[dict[str, object] | None, str | None]:
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        return None, f"invalid_json: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, "invalid_root: expected a JSON object"
    if set(payload) != {"claims"}:
        return None, "invalid_root: expected exactly the 'claims' field"
    if not isinstance(payload["claims"], list):
        return None, "invalid_root: 'claims' must be an array"
    return payload, None


def _parse_items(
    content: str,
    expected_claim_ids: Sequence[str],
    *,
    stage: ElementizationStage,
    model_type: type[ElementizationProposal] | type[ElementizationReview],
) -> tuple[tuple[object, ...], tuple[ElementizationFailure, ...], tuple[str, ...]]:
    expected = _clean_unique_ids(expected_claim_ids, field_name="expected_claim_ids")
    payload, root_error = _parse_json_object(content)
    if root_error is not None:
        return (), _invalid_failures(expected, stage=stage, diagnostic=root_error), (
            root_error,
        )

    assert payload is not None
    valid_by_id: dict[str, object] = {}
    failure_by_id: dict[str, ElementizationFailure] = {}
    diagnostics: list[str] = []
    expected_set = set(expected)
    for index, raw_item in enumerate(payload["claims"]):  # type: ignore[index]
        raw_claim_id = raw_item.get("claim_id") if isinstance(raw_item, dict) else None
        claim_id = raw_claim_id.strip() if isinstance(raw_claim_id, str) else None
        if claim_id not in expected_set:
            diagnostics.append(f"claims[{index}]: unexpected or missing claim_id")
            continue
        if claim_id in valid_by_id or claim_id in failure_by_id:
            valid_by_id.pop(claim_id, None)
            failure_by_id[claim_id] = ElementizationFailure(
                claim_id=claim_id,
                stage=stage,
                execution_status=ElementizationExecutionStatus.INVALID_RESPONSE,
                diagnostic="duplicate claim_id in model response",
            )
            diagnostics.append(f"{claim_id}: duplicate claim_id")
            continue
        try:
            valid_by_id[claim_id] = model_type.model_validate(raw_item)
        except Exception as exc:
            failure_by_id[claim_id] = ElementizationFailure(
                claim_id=claim_id,
                stage=stage,
                execution_status=ElementizationExecutionStatus.INVALID_RESPONSE,
                diagnostic=f"invalid item: {type(exc).__name__}: {exc}",
            )
            diagnostics.append(f"{claim_id}: invalid item")

    for claim_id in expected:
        if claim_id not in valid_by_id and claim_id not in failure_by_id:
            failure_by_id[claim_id] = ElementizationFailure(
                claim_id=claim_id,
                stage=stage,
                execution_status=ElementizationExecutionStatus.INVALID_RESPONSE,
                diagnostic="claim_id missing from model response",
            )
            diagnostics.append(f"{claim_id}: missing from model response")

    return (
        tuple(valid_by_id[claim_id] for claim_id in expected if claim_id in valid_by_id),
        tuple(failure_by_id[claim_id] for claim_id in expected if claim_id in failure_by_id),
        tuple(diagnostics),
    )


def parse_elementization_proposals(
    content: str,
    expected_claim_ids: Sequence[str],
) -> ParsedElementizationProposals:
    """Parse model proposals without losing valid sibling claims."""

    expected = _clean_unique_ids(expected_claim_ids, field_name="expected_claim_ids")
    parsed, failures, diagnostics = _parse_items(
        content,
        expected,
        stage=ElementizationStage.PROPOSAL,
        model_type=ElementizationProposal,
    )
    return ParsedElementizationProposals(
        expected_claim_ids=expected,
        proposals=tuple(item for item in parsed if isinstance(item, ElementizationProposal)),
        failures=failures,
        diagnostics=diagnostics,
    )


def parse_elementization_reviews(
    content: str,
    expected_claim_ids: Sequence[str],
) -> ParsedElementizationReviews:
    """Parse independent reviews without silently accepting missing claims."""

    expected = _clean_unique_ids(expected_claim_ids, field_name="expected_claim_ids")
    parsed, failures, diagnostics = _parse_items(
        content,
        expected,
        stage=ElementizationStage.REVIEW,
        model_type=ElementizationReview,
    )
    return ParsedElementizationReviews(
        expected_claim_ids=expected,
        reviews=tuple(item for item in parsed if isinstance(item, ElementizationReview)),
        failures=failures,
        diagnostics=diagnostics,
    )


class TruthConditionElement(BaseModel):
    """One code-identified semantic verification denominator item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    element_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    text: str = Field(min_length=1)

    @field_validator("element_id", "claim_id", "text", mode="before")
    @classmethod
    def _text_is_clean(cls, value: object, info: object) -> str:
        return _clean_text(value, field_name=getattr(info, "field_name", "text"))

    @model_validator(mode="after")
    def _identifier_is_code_owned(self) -> TruthConditionElement:
        expected = make_truth_condition_element_id(self.claim_id, self.ordinal)
        if self.element_id != expected:
            raise ValueError(f"element_id must equal code allocation {expected!r}")
        return self


class ClaimTruthConditionRegistryEntry(BaseModel):
    """Auditable proposal, review outcome, and final elements for one claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)
    claim_text_sha256: str = Field(min_length=64, max_length=64)
    proposal_elements: tuple[str, ...] = ()
    elements: tuple[TruthConditionElement, ...] = ()
    execution_status: ElementizationExecutionStatus
    semantic_status: ElementizationSemanticStatus | None = None
    missing_conditions: tuple[str, ...] = ()
    review_rationale: str | None = None
    diagnostic: str | None = None

    @field_validator("claim_id", "claim_text", mode="before")
    @classmethod
    def _text_is_clean(cls, value: object, info: object) -> str:
        return _clean_text(value, field_name=getattr(info, "field_name", "text"))

    @field_validator("proposal_elements", "missing_conditions", mode="before")
    @classmethod
    def _tuples_are_clean(cls, value: object, info: object) -> tuple[str, ...]:
        return _clean_text_tuple(
            value,
            field_name=getattr(info, "field_name", "items"),
        )

    @field_validator("review_rationale", "diagnostic", mode="before")
    @classmethod
    def _optional_text_is_clean(cls, value: object, info: object) -> str | None:
        if value is None:
            return None
        return _clean_text(value, field_name=getattr(info, "field_name", "text"))

    @model_validator(mode="after")
    def _entry_is_coherent(self) -> ClaimTruthConditionRegistryEntry:
        if not _is_sha256(self.claim_text_sha256):
            raise ValueError("claim_text_sha256 must be a lowercase SHA-256 digest")
        if self.claim_text_sha256 != _sha256_text(self.claim_text):
            raise ValueError("claim_text_sha256 must match claim_text")
        for ordinal, element in enumerate(self.elements):
            if element.claim_id != self.claim_id or element.ordinal != ordinal:
                raise ValueError("elements must be contiguous and belong to the entry claim")
        if len({element.element_id for element in self.elements}) != len(self.elements):
            raise ValueError("element IDs must be unique")
        if self.execution_status is ElementizationExecutionStatus.COMPLETE:
            if self.semantic_status is None or not self.review_rationale:
                raise ValueError("completed elementization requires semantic review")
            if self.diagnostic is not None:
                raise ValueError("completed elementization cannot carry a failure diagnostic")
        else:
            if self.semantic_status is not None or self.review_rationale is not None:
                raise ValueError("failed elementization cannot carry a semantic judgement")
            if not self.diagnostic:
                raise ValueError("failed elementization requires a diagnostic")
        if self.semantic_status is ElementizationSemanticStatus.COMPLETE and not self.elements:
            raise ValueError("semantically complete entry requires elements")
        return self


class TruthConditionDenominatorAudit(BaseModel):
    """Closed claim denominator for production elementization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    selected_claim_ids: tuple[str, ...]
    complete_claim_ids: tuple[str, ...]
    incomplete_claim_ids: tuple[str, ...]
    unresolved_claim_ids: tuple[str, ...]
    silent_bypass_claim_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _partitions_selected_claims(self) -> TruthConditionDenominatorAudit:
        if len(set(self.selected_claim_ids)) != len(self.selected_claim_ids):
            raise ValueError("selected claim IDs must be unique")
        selected = set(self.selected_claim_ids)
        groups = (
            set(self.complete_claim_ids),
            set(self.incomplete_claim_ids),
            set(self.unresolved_claim_ids),
            set(self.silent_bypass_claim_ids),
        )
        if any(len(group) != len(values) for group, values in zip(groups, (
            self.complete_claim_ids,
            self.incomplete_claim_ids,
            self.unresolved_claim_ids,
            self.silent_bypass_claim_ids,
        ))):
            raise ValueError("denominator category IDs must be unique")
        for index, group in enumerate(groups):
            if any(group & other for other in groups[index + 1 :]):
                raise ValueError("denominator categories must be disjoint")
        if set().union(*groups) != selected:
            raise ValueError("denominator categories must partition selected claims")
        return self

    @property
    def is_closed(self) -> bool:
        """Return whether no selected claim silently bypassed elementization."""

        return not self.silent_bypass_claim_ids

    @property
    def silent_bypass_count(self) -> int:
        """Return the explicit bypass count."""

        return len(self.silent_bypass_claim_ids)


class TruthConditionRegistry(BaseModel):
    """Independent, versioned truth-condition registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    registry_version: Literal["truth-condition-elements-v1"] = REGISTRY_VERSION
    entries: tuple[ClaimTruthConditionRegistryEntry, ...]
    denominator: TruthConditionDenominatorAudit

    @model_validator(mode="after")
    def _registry_matches_denominator(self) -> TruthConditionRegistry:
        entry_ids = tuple(entry.claim_id for entry in self.entries)
        if len(set(entry_ids)) != len(entry_ids):
            raise ValueError("registry claim IDs must be unique")
        if entry_ids != self.denominator.selected_claim_ids:
            raise ValueError("registry entries must match selected denominator order")
        if not self.denominator.is_closed:
            raise ValueError("production registry cannot contain silent bypasses")
        complete = tuple(
            entry.claim_id
            for entry in self.entries
            if entry.execution_status is ElementizationExecutionStatus.COMPLETE
            and entry.semantic_status is ElementizationSemanticStatus.COMPLETE
        )
        incomplete = tuple(
            entry.claim_id
            for entry in self.entries
            if entry.execution_status is ElementizationExecutionStatus.COMPLETE
            and entry.semantic_status is ElementizationSemanticStatus.INCOMPLETE
        )
        unresolved = tuple(
            entry.claim_id
            for entry in self.entries
            if entry.claim_id not in set(complete) | set(incomplete)
        )
        if (
            complete != self.denominator.complete_claim_ids
            or incomplete != self.denominator.incomplete_claim_ids
            or unresolved != self.denominator.unresolved_claim_ids
        ):
            raise ValueError("registry semantic states must match denominator audit")
        return self

    def entry_for(self, claim_id: str) -> ClaimTruthConditionRegistryEntry | None:
        """Return one entry without treating absence as successful elementization."""

        return next((entry for entry in self.entries if entry.claim_id == claim_id), None)


def select_truth_condition_registry(
    registry: TruthConditionRegistry,
    claim_ids: Sequence[str],
) -> TruthConditionRegistry:
    """Create an exact, ordered registry for a bounded claim subset.

    Incremental evidence passes verify only routed claims.  Passing the full
    run registry would either violate the verifier's exact denominator or tempt
    it to silently skip unrelated entries.  This projection closes a new
    denominator over precisely the caller-selected IDs without changing any
    semantic element.
    """

    selected_ids = _clean_unique_ids(claim_ids, field_name="claim_ids")
    by_id = {entry.claim_id: entry for entry in registry.entries}
    unknown = tuple(claim_id for claim_id in selected_ids if claim_id not in by_id)
    if unknown:
        raise ValueError(
            "truth-condition subset contains unknown claims: "
            + ", ".join(unknown)
        )
    entries = tuple(by_id[claim_id] for claim_id in selected_ids)
    complete = tuple(
        entry.claim_id
        for entry in entries
        if entry.execution_status is ElementizationExecutionStatus.COMPLETE
        and entry.semantic_status is ElementizationSemanticStatus.COMPLETE
    )
    incomplete = tuple(
        entry.claim_id
        for entry in entries
        if entry.execution_status is ElementizationExecutionStatus.COMPLETE
        and entry.semantic_status is ElementizationSemanticStatus.INCOMPLETE
    )
    resolved = set(complete) | set(incomplete)
    unresolved = tuple(
        entry.claim_id for entry in entries if entry.claim_id not in resolved
    )
    return TruthConditionRegistry(
        entries=entries,
        denominator=TruthConditionDenominatorAudit(
            selected_claim_ids=selected_ids,
            complete_claim_ids=complete,
            incomplete_claim_ids=incomplete,
            unresolved_claim_ids=unresolved,
        ),
    )


def _allocate_elements(
    claim_id: str,
    texts: Sequence[str],
) -> tuple[TruthConditionElement, ...]:
    return tuple(
        TruthConditionElement(
            element_id=make_truth_condition_element_id(claim_id, ordinal),
            claim_id=claim_id,
            ordinal=ordinal,
            text=text,
        )
        for ordinal, text in enumerate(texts)
    )


def build_truth_condition_registry(
    claim_surfaces: Mapping[str, str],
    *,
    proposals: Sequence[ElementizationProposal] = (),
    reviews: Sequence[ElementizationReview] = (),
    failures: Sequence[ElementizationFailure] = (),
) -> TruthConditionRegistry:
    """Resolve proposal and independent review outputs into a closed registry.

    Missing outputs are explicit unresolved entries.  A failed review retains
    proposal text for diagnosis but cannot acquire a semantic success status.
    """

    claim_ids = _clean_unique_ids(claim_surfaces, field_name="claim_surfaces keys")
    clean_surfaces = {
        claim_id: _clean_text(claim_surfaces[claim_id], field_name="claim_text")
        for claim_id in claim_ids
    }
    proposal_by_id = {item.claim_id: item for item in proposals}
    review_by_id = {item.claim_id: item for item in reviews}
    if len(proposal_by_id) != len(proposals) or len(review_by_id) != len(reviews):
        raise ValueError("proposal and review claim IDs must be unique")
    failure_by_key = {(item.claim_id, item.stage): item for item in failures}
    if len(failure_by_key) != len(failures):
        raise ValueError("failures must be unique per claim and stage")
    supplied_ids = set(proposal_by_id) | set(review_by_id) | {
        item.claim_id for item in failures
    }
    unknown_ids = supplied_ids - set(claim_ids)
    if unknown_ids:
        raise ValueError(f"outputs contain unknown claim IDs: {sorted(unknown_ids)!r}")

    entries: list[ClaimTruthConditionRegistryEntry] = []
    for claim_id in claim_ids:
        claim_text = clean_surfaces[claim_id]
        proposal = proposal_by_id.get(claim_id)
        review = review_by_id.get(claim_id)
        proposal_failure = failure_by_key.get((claim_id, ElementizationStage.PROPOSAL))
        review_failure = failure_by_key.get((claim_id, ElementizationStage.REVIEW))

        if proposal_failure is not None or proposal is None:
            failure = proposal_failure
            entries.append(
                ClaimTruthConditionRegistryEntry(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    claim_text_sha256=_sha256_text(claim_text),
                    execution_status=(
                        failure.execution_status
                        if failure is not None
                        else ElementizationExecutionStatus.INVALID_RESPONSE
                    ),
                    diagnostic=(
                        failure.diagnostic
                        if failure is not None
                        else "proposal missing from protocol output"
                    ),
                )
            )
            continue

        if review_failure is not None or review is None:
            failure = review_failure
            entries.append(
                ClaimTruthConditionRegistryEntry(
                    claim_id=claim_id,
                    claim_text=claim_text,
                    claim_text_sha256=_sha256_text(claim_text),
                    proposal_elements=proposal.elements,
                    elements=_allocate_elements(claim_id, proposal.elements),
                    execution_status=(
                        failure.execution_status
                        if failure is not None
                        else ElementizationExecutionStatus.NOT_RUN
                    ),
                    diagnostic=(
                        failure.diagnostic
                        if failure is not None
                        else "independent review was not run"
                    ),
                )
            )
            continue

        entries.append(
            ClaimTruthConditionRegistryEntry(
                claim_id=claim_id,
                claim_text=claim_text,
                claim_text_sha256=_sha256_text(claim_text),
                proposal_elements=proposal.elements,
                elements=_allocate_elements(claim_id, review.elements),
                execution_status=ElementizationExecutionStatus.COMPLETE,
                semantic_status=review.semantic_status,
                missing_conditions=review.missing_conditions,
                review_rationale=review.rationale,
            )
        )

    complete_ids = tuple(
        entry.claim_id
        for entry in entries
        if entry.execution_status is ElementizationExecutionStatus.COMPLETE
        and entry.semantic_status is ElementizationSemanticStatus.COMPLETE
    )
    incomplete_ids = tuple(
        entry.claim_id
        for entry in entries
        if entry.execution_status is ElementizationExecutionStatus.COMPLETE
        and entry.semantic_status is ElementizationSemanticStatus.INCOMPLETE
    )
    classified = set(complete_ids) | set(incomplete_ids)
    unresolved_ids = tuple(
        entry.claim_id for entry in entries if entry.claim_id not in classified
    )
    return TruthConditionRegistry(
        entries=tuple(entries),
        denominator=TruthConditionDenominatorAudit(
            selected_claim_ids=claim_ids,
            complete_claim_ids=complete_ids,
            incomplete_claim_ids=incomplete_ids,
            unresolved_claim_ids=unresolved_ids,
        ),
    )


def build_registry_from_parse_results(
    claim_surfaces: Mapping[str, str],
    proposal_result: ParsedElementizationProposals,
    review_result: ParsedElementizationReviews,
) -> TruthConditionRegistry:
    """Build a registry from independently parsed proposal and review calls."""

    claim_ids = tuple(claim_surfaces)
    if proposal_result.expected_claim_ids != claim_ids:
        raise ValueError("proposal denominator does not match claim surfaces")
    if review_result.expected_claim_ids != claim_ids:
        raise ValueError("review denominator does not match claim surfaces")
    return build_truth_condition_registry(
        claim_surfaces,
        proposals=proposal_result.proposals,
        reviews=review_result.reviews,
        failures=proposal_result.failures + review_result.failures,
    )


def truth_condition_protocol(
    registry: TruthConditionRegistry | None,
) -> TruthConditionProtocol:
    """Select legacy behavior only when the caller explicitly passes ``None``."""

    if registry is None:
        return TruthConditionProtocol.LEGACY_WHOLE_CLAIM
    return TruthConditionProtocol.ELEMENT_REGISTRY


def truth_condition_registry_sha256(registry: TruthConditionRegistry) -> str:
    """Hash the canonical registry payload for audit binding."""

    canonical = json.dumps(
        registry.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_elementization_proposal_prompt(
    claim_surfaces: Mapping[str, str],
    *,
    claim_contexts: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Build a semantic proposal prompt; identifiers remain code-owned."""

    claim_ids = _clean_unique_ids(claim_surfaces, field_name="claim_surfaces keys")
    payload = []
    for claim_id in claim_ids:
        context = tuple((claim_contexts or {}).get(claim_id, ()))
        payload.append(
            {
                "claim_id": claim_id,
                "claim_text": _clean_text(
                    claim_surfaces[claim_id], field_name="claim_text"
                ),
                "context": list(context),
            }
        )
    return (
        "Propose the complete truth conditions that must all hold for each claim "
        "to be supported. "
        + _TRUTH_CONDITION_COMPLETENESS_GUIDANCE
        + "Preserve relationships, qualifications, scope, and "
        "attribution; do not turn related facts into substitutes for the claim. "
        "Return semantic element text only. Do not invent element IDs: code assigns "
        "them after independent review. If the claim cannot be safely decomposed, "
        "return an empty elements array and explain why in rationale.\n\n"
        "Return exactly this JSON shape:\n"
        '{"claims":[{"claim_id":"...","elements":["..."],'
        '"rationale":"..."}]}\n\n'
        f"CLAIMS:\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def build_elementization_review_prompt(
    claim_surfaces: Mapping[str, str],
    proposals: Sequence[ElementizationProposal],
    *,
    claim_contexts: Mapping[str, Sequence[str]] | None = None,
    claim_glosses: Mapping[str, str] | None = None,
) -> str:
    """Build an independent-review prompt that may correct the proposal.

    The registry identity is always the verbatim report surface. Necessary
    context and the model-derived decontextualized gloss are advisory semantic
    inputs only; neither may silently replace the statement being reviewed.
    """

    claim_ids = _clean_unique_ids(claim_surfaces, field_name="claim_surfaces keys")
    proposal_by_id = {item.claim_id: item for item in proposals}
    if len(proposal_by_id) != len(proposals) or set(proposal_by_id) != set(claim_ids):
        raise ValueError("review prompt requires exactly one proposal per claim")
    payload = [
        {
            "claim_id": claim_id,
            "report_surface_text": _clean_text(
                claim_surfaces[claim_id], field_name="claim_text"
            ),
            "necessary_context": list((claim_contexts or {}).get(claim_id, ())),
            "retrieval_gloss": (claim_glosses or {}).get(claim_id),
            "proposal": list(proposal_by_id[claim_id].elements),
        }
        for claim_id in claim_ids
    ]
    return (
        "Independently review each proposed truth-condition denominator. The "
        "verbatim report_surface_text, interpreted with necessary_context, is "
        "authoritative. retrieval_gloss is a model-derived aid only and must "
        "not strengthen, weaken, or replace the report wording. Do not "
        "assume the proposal is correct. "
        + _TRUTH_CONDITION_COMPLETENESS_GUIDANCE
        + "Return the corrected full element list. "
        "Use semantic_status=complete only when the list preserves every material "
        "condition of the original claim. Use incomplete when you can identify an "
        "omission but cannot supply a complete reliable list, and uncertain when "
        "you cannot determine completeness. Describe omissions in "
        "missing_conditions. When semantic_status is incomplete, "
        "missing_conditions must contain at least one specific omitted condition; "
        "if no omission can be named, use uncertain instead. This is semantic "
        "review; code will only validate the "
        "response shape and allocate IDs.\n\n"
        "Return exactly this JSON shape:\n"
        '{"claims":[{"claim_id":"...","semantic_status":"complete|incomplete|uncertain",'
        '"elements":["..."],"missing_conditions":["..."],'
        '"rationale":"..."}]}\n\n'
        f"CLAIMS_AND_PROPOSALS:\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def build_elementization_review_recovery_prompt(
    claim_surfaces: Mapping[str, str],
    proposals: Sequence[ElementizationProposal],
    failures: Sequence[ElementizationFailure],
    *,
    claim_contexts: Mapping[str, Sequence[str]] | None = None,
    claim_glosses: Mapping[str, str] | None = None,
) -> str:
    """Build one strict, item-local repair prompt for invalid review output.

    This does not relax :class:`ElementizationReview`.  It narrows the second
    call to the invalid claim IDs and exposes the mechanical validation errors
    so the model can repair protocol shape without reconsidering valid sibling
    judgements.
    """

    claim_ids = _clean_unique_ids(claim_surfaces, field_name="claim_surfaces keys")
    failure_by_id = {item.claim_id: item for item in failures}
    if len(failure_by_id) != len(failures) or set(failure_by_id) != set(claim_ids):
        raise ValueError("recovery prompt requires one failure per claim")
    if any(item.stage is not ElementizationStage.REVIEW for item in failures):
        raise ValueError("recovery prompt accepts review-stage failures only")
    if any(
        item.execution_status is not ElementizationExecutionStatus.INVALID_RESPONSE
        for item in failures
    ):
        raise ValueError("recovery prompt accepts invalid responses only")

    validation_errors = {
        claim_id: failure_by_id[claim_id].diagnostic for claim_id in claim_ids
    }
    return (
        "This is the single bounded protocol-recovery attempt for only the "
        "claims whose previous review items were invalid. Do not revisit or "
        "emit any valid sibling claim. Keep the same semantic task and exact "
        "schema. Re-evaluate inconsistent fields together; do not label an item "
        "complete merely to satisfy validation. Repair each listed item in "
        "light of its validation error.\n"
        f"PRIOR_VALIDATION_ERRORS:\n"
        f"{json.dumps(validation_errors, ensure_ascii=False, sort_keys=True)}\n\n"
        + build_elementization_review_prompt(
            claim_surfaces,
            proposals,
            claim_contexts=claim_contexts,
            claim_glosses=claim_glosses,
        )
    )


class ElementSourceAssessment(BaseModel):
    """One element-source result with semantic and mechanical facts separated."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    element_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    execution_status: ElementAssessmentExecutionStatus
    verdict: ElementVerificationVerdict | None = None
    evidence_located: bool = False
    formal_supporting_evidence: bool = False
    diagnostic: str | None = None

    @field_validator("claim_id", "element_id", "source_id", mode="before")
    @classmethod
    def _text_is_clean(cls, value: object, info: object) -> str:
        return _clean_text(value, field_name=getattr(info, "field_name", "text"))

    @field_validator("diagnostic", mode="before")
    @classmethod
    def _diagnostic_is_clean(cls, value: object) -> str | None:
        if value is None:
            return None
        return _clean_text(value, field_name="diagnostic")

    @model_validator(mode="after")
    def _assessment_is_coherent(self) -> ElementSourceAssessment:
        if self.execution_status is ElementAssessmentExecutionStatus.COMPLETE:
            if self.verdict is None:
                raise ValueError("completed assessment requires a verdict")
        elif self.execution_status is ElementAssessmentExecutionStatus.QUOTE_UNLOCATABLE:
            if self.verdict is None:
                raise ValueError("quote_unlocatable assessment requires a verdict")
            if self.evidence_located or self.formal_supporting_evidence:
                raise ValueError("unlocatable quote cannot be formal evidence")
        else:
            if self.verdict is not None or self.evidence_located:
                raise ValueError("failed assessment cannot carry a semantic verdict")
            if self.formal_supporting_evidence:
                raise ValueError("failed assessment cannot be formal support")
        if self.formal_supporting_evidence and not (
            self.execution_status is ElementAssessmentExecutionStatus.COMPLETE
            and self.verdict is ElementVerificationVerdict.SUPPORTS
            and self.evidence_located
        ):
            raise ValueError("formal support requires a located completed support verdict")
        return self


class ElementTruthConditionAggregate(BaseModel):
    """Aggregate over all expected candidate sources for one element."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    element_id: str
    semantic_state: ElementSemanticState
    execution_completeness: ExecutionCompleteness
    expected_source_ids: tuple[str, ...]
    evaluated_source_ids: tuple[str, ...]
    supporting_source_ids: tuple[str, ...] = ()
    contradicting_source_ids: tuple[str, ...] = ()
    not_supporting_source_ids: tuple[str, ...] = ()
    insufficient_source_ids: tuple[str, ...] = ()
    unresolved_source_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _source_denominator_is_coherent(self) -> ElementTruthConditionAggregate:
        expected = set(self.expected_source_ids)
        evaluated = set(self.evaluated_source_ids)
        if len(expected) != len(self.expected_source_ids):
            raise ValueError("expected source IDs must be unique")
        if len(evaluated) != len(self.evaluated_source_ids):
            raise ValueError("evaluated source IDs must be unique")
        if not evaluated <= expected:
            raise ValueError("evaluated sources must belong to expected denominator")
        raw_groups = (
            self.supporting_source_ids,
            self.contradicting_source_ids,
            self.not_supporting_source_ids,
            self.insufficient_source_ids,
            self.unresolved_source_ids,
        )
        if any(len(values) != len(set(values)) for values in raw_groups):
            raise ValueError("source outcome category IDs must be unique")
        groups = tuple(set(values) for values in raw_groups)
        if any(
            groups[i] & groups[j]
            for i in range(len(groups))
            for j in range(i + 1, len(groups))
        ):
            raise ValueError("source outcome categories must be disjoint")
        classified = set().union(*groups)
        if classified != expected:
            raise ValueError("source outcome categories must cover expected sources")
        classified_completed = set().union(*groups[:-1])
        if not classified_completed <= evaluated:
            raise ValueError(
                "semantic source outcomes require completed evaluations"
            )
        if groups[-1] != expected - classified_completed:
            raise ValueError(
                "unresolved sources must be the unclassified denominator remainder"
            )

        if not expected:
            derived_execution = ExecutionCompleteness.NOT_RUN
            allowed_execution = {derived_execution}
        elif evaluated == expected:
            derived_execution = ExecutionCompleteness.COMPLETE
            allowed_execution = {derived_execution}
        elif evaluated:
            derived_execution = ExecutionCompleteness.PARTIAL
            allowed_execution = {derived_execution}
        else:
            # This aggregate intentionally does not retain whether a zero-result
            # denominator was never attempted or every attempt failed.  The
            # enclosing ClaimVerification can distinguish those cases from its
            # element relations; both are mechanically valid here.
            allowed_execution = {
                ExecutionCompleteness.FAILED,
                ExecutionCompleteness.NOT_RUN,
            }
        if self.execution_completeness not in allowed_execution:
            raise ValueError(
                "execution completeness does not match evaluated source scope"
            )

        supporting, contradicting, not_supporting, _, _ = groups
        if supporting and contradicting:
            derived_semantic = ElementSemanticState.CONFLICTED
        elif contradicting:
            derived_semantic = ElementSemanticState.CONTRADICTED
        elif supporting:
            derived_semantic = ElementSemanticState.SUPPORTED
        elif (
            self.execution_completeness is ExecutionCompleteness.COMPLETE
            and not_supporting == expected
        ):
            derived_semantic = ElementSemanticState.NOT_SUPPORTED
        else:
            derived_semantic = ElementSemanticState.UNRESOLVED
        if self.semantic_state is not derived_semantic:
            raise ValueError(
                "element semantic state does not match source outcome categories"
            )
        return self


class ClaimTruthConditionAggregate(BaseModel):
    """Claim-level coverage over the independently reviewed element registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    elementization_execution_status: ElementizationExecutionStatus
    elementization_semantic_status: ElementizationSemanticStatus | None
    coverage_state: ClaimCoverageState
    execution_completeness: ExecutionCompleteness
    elements: tuple[ElementTruthConditionAggregate, ...]
    supported_element_ids: tuple[str, ...] = ()
    not_supported_element_ids: tuple[str, ...] = ()
    contradicted_element_ids: tuple[str, ...] = ()
    conflicted_element_ids: tuple[str, ...] = ()
    unresolved_element_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _element_denominator_is_coherent(self) -> ClaimTruthConditionAggregate:
        if any(item.claim_id != self.claim_id for item in self.elements):
            raise ValueError("claim aggregate elements must belong to claim_id")
        expected = {item.element_id for item in self.elements}
        if len(expected) != len(self.elements):
            raise ValueError("claim aggregate element IDs must be unique")
        raw_groups = (
            self.supported_element_ids,
            self.not_supported_element_ids,
            self.contradicted_element_ids,
            self.conflicted_element_ids,
            self.unresolved_element_ids,
        )
        if any(len(values) != len(set(values)) for values in raw_groups):
            raise ValueError("element semantic category IDs must be unique")
        groups = tuple(set(values) for values in raw_groups)
        if any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i + 1, len(groups))):
            raise ValueError("element semantic categories must be disjoint")
        if set().union(*groups) != expected:
            raise ValueError("element semantic categories must cover registered elements")

        derived_groups = {
            state: {
                item.element_id
                for item in self.elements
                if item.semantic_state is state
            }
            for state in ElementSemanticState
        }
        declared_groups = {
            ElementSemanticState.SUPPORTED: groups[0],
            ElementSemanticState.NOT_SUPPORTED: groups[1],
            ElementSemanticState.CONTRADICTED: groups[2],
            ElementSemanticState.CONFLICTED: groups[3],
            ElementSemanticState.UNRESOLVED: groups[4],
        }
        if declared_groups != derived_groups:
            raise ValueError(
                "claim element categories must match child semantic states"
            )

        if self.elementization_execution_status is ElementizationExecutionStatus.COMPLETE:
            if self.elementization_semantic_status is None:
                raise ValueError(
                    "completed elementization requires a semantic status"
                )
        elif self.elementization_semantic_status is not None:
            raise ValueError(
                "failed elementization cannot carry a semantic status"
            )

        element_executions = tuple(
            item.execution_completeness for item in self.elements
        )
        if self.elementization_execution_status is not ElementizationExecutionStatus.COMPLETE:
            derived_execution = (
                ExecutionCompleteness.NOT_RUN
                if self.elementization_execution_status
                is ElementizationExecutionStatus.NOT_RUN
                else ExecutionCompleteness.FAILED
            )
        elif not element_executions or all(
            state is ExecutionCompleteness.NOT_RUN
            for state in element_executions
        ):
            derived_execution = ExecutionCompleteness.NOT_RUN
        elif all(
            state is ExecutionCompleteness.COMPLETE
            for state in element_executions
        ):
            derived_execution = ExecutionCompleteness.COMPLETE
        elif all(
            state is ExecutionCompleteness.FAILED
            for state in element_executions
        ):
            derived_execution = ExecutionCompleteness.FAILED
        else:
            derived_execution = ExecutionCompleteness.PARTIAL
        if self.execution_completeness is not derived_execution:
            raise ValueError(
                "claim execution completeness does not match child elements"
            )

        supported = derived_groups[ElementSemanticState.SUPPORTED]
        not_supported = derived_groups[ElementSemanticState.NOT_SUPPORTED]
        contradicted = derived_groups[ElementSemanticState.CONTRADICTED]
        conflicted = derived_groups[ElementSemanticState.CONFLICTED]
        if self.elementization_execution_status is not ElementizationExecutionStatus.COMPLETE:
            derived_coverage = ClaimCoverageState.UNRESOLVED
        elif conflicted:
            derived_coverage = ClaimCoverageState.CONFLICTED
        elif supported and contradicted:
            derived_coverage = ClaimCoverageState.MIXED
        elif contradicted:
            derived_coverage = ClaimCoverageState.CONTRADICTED
        elif (
            self.elementization_semantic_status
            is ElementizationSemanticStatus.COMPLETE
            and self.elements
            and len(supported) == len(self.elements)
        ):
            derived_coverage = ClaimCoverageState.FULLY_SUPPORTED
        elif supported:
            derived_coverage = ClaimCoverageState.PARTIALLY_SUPPORTED
        elif (
            self.elementization_semantic_status
            is ElementizationSemanticStatus.COMPLETE
            and self.elements
            and len(not_supported) == len(self.elements)
        ):
            derived_coverage = ClaimCoverageState.NOT_SUPPORTED
        else:
            derived_coverage = ClaimCoverageState.UNRESOLVED
        if self.coverage_state is not derived_coverage:
            raise ValueError(
                "claim coverage state does not match child semantic states"
            )
        return self


class TruthConditionAggregationResult(BaseModel):
    """Registry-wide aggregation bound to one exact registry payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    registry_sha256: str
    claims: tuple[ClaimTruthConditionAggregate, ...]


def _execution_completeness(
    expected_source_ids: tuple[str, ...],
    assessments: Sequence[ElementSourceAssessment],
) -> ExecutionCompleteness:
    if not expected_source_ids:
        return ExecutionCompleteness.NOT_RUN
    completed = sum(
        item.execution_status is ElementAssessmentExecutionStatus.COMPLETE
        for item in assessments
    )
    if completed == len(expected_source_ids):
        return ExecutionCompleteness.COMPLETE
    if completed:
        return ExecutionCompleteness.PARTIAL
    if assessments:
        return ExecutionCompleteness.FAILED
    return ExecutionCompleteness.NOT_RUN


def aggregate_truth_condition_element(
    element: TruthConditionElement,
    assessments: Sequence[ElementSourceAssessment],
    *,
    expected_source_ids: Sequence[str],
) -> ElementTruthConditionAggregate:
    """Aggregate one closed element-source denominator without semantic heuristics."""

    expected = _clean_unique_ids(expected_source_ids, field_name="expected_source_ids")
    by_source: dict[str, ElementSourceAssessment] = {}
    for assessment in assessments:
        if assessment.claim_id != element.claim_id or assessment.element_id != element.element_id:
            raise ValueError("assessment does not belong to the requested element")
        if assessment.source_id not in set(expected):
            raise ValueError("assessment source is outside the expected denominator")
        if assessment.source_id in by_source:
            raise ValueError("assessment source IDs must be unique per element")
        by_source[assessment.source_id] = assessment

    support_ids = tuple(
        source_id
        for source_id in expected
        if (item := by_source.get(source_id)) is not None
        and item.formal_supporting_evidence
    )
    contradiction_ids = tuple(
        source_id
        for source_id in expected
        if (item := by_source.get(source_id)) is not None
        and item.execution_status is ElementAssessmentExecutionStatus.COMPLETE
        and item.verdict is ElementVerificationVerdict.CONTRADICTS
        and item.evidence_located
    )
    not_supporting_ids = tuple(
        source_id
        for source_id in expected
        if (item := by_source.get(source_id)) is not None
        and item.execution_status is ElementAssessmentExecutionStatus.COMPLETE
        and item.verdict is ElementVerificationVerdict.DOES_NOT_SUPPORT
    )
    insufficient_ids = tuple(
        source_id
        for source_id in expected
        if (item := by_source.get(source_id)) is not None
        and item.execution_status is ElementAssessmentExecutionStatus.COMPLETE
        and item.verdict is ElementVerificationVerdict.NOT_ENOUGH_INFORMATION
    )
    execution = _execution_completeness(expected, tuple(by_source.values()))
    if support_ids and contradiction_ids:
        semantic = ElementSemanticState.CONFLICTED
    elif contradiction_ids:
        semantic = ElementSemanticState.CONTRADICTED
    elif support_ids:
        semantic = ElementSemanticState.SUPPORTED
    elif execution is ExecutionCompleteness.COMPLETE and len(
        not_supporting_ids
    ) == len(expected):
        semantic = ElementSemanticState.NOT_SUPPORTED
    else:
        semantic = ElementSemanticState.UNRESOLVED

    unresolved_ids = tuple(
        source_id
        for source_id in expected
        if source_id
        not in set(support_ids)
        | set(contradiction_ids)
        | set(not_supporting_ids)
        | set(insufficient_ids)
    )
    return ElementTruthConditionAggregate(
        claim_id=element.claim_id,
        element_id=element.element_id,
        semantic_state=semantic,
        execution_completeness=execution,
        expected_source_ids=expected,
        evaluated_source_ids=tuple(
            source_id
            for source_id in expected
            if (item := by_source.get(source_id)) is not None
            and item.execution_status is ElementAssessmentExecutionStatus.COMPLETE
        ),
        supporting_source_ids=support_ids,
        contradicting_source_ids=contradiction_ids,
        not_supporting_source_ids=not_supporting_ids,
        insufficient_source_ids=insufficient_ids,
        unresolved_source_ids=unresolved_ids,
    )


def _claim_execution(
    entry: ClaimTruthConditionRegistryEntry,
    elements: Sequence[ElementTruthConditionAggregate],
) -> ExecutionCompleteness:
    if entry.execution_status is not ElementizationExecutionStatus.COMPLETE:
        if entry.execution_status is ElementizationExecutionStatus.NOT_RUN:
            return ExecutionCompleteness.NOT_RUN
        return ExecutionCompleteness.FAILED
    states = tuple(item.execution_completeness for item in elements)
    if not states or all(state is ExecutionCompleteness.NOT_RUN for state in states):
        return ExecutionCompleteness.NOT_RUN
    if all(state is ExecutionCompleteness.COMPLETE for state in states):
        return ExecutionCompleteness.COMPLETE
    if all(state is ExecutionCompleteness.FAILED for state in states):
        return ExecutionCompleteness.FAILED
    return ExecutionCompleteness.PARTIAL


def aggregate_truth_condition_claim(
    entry: ClaimTruthConditionRegistryEntry,
    assessments: Sequence[ElementSourceAssessment],
    *,
    expected_source_ids: Sequence[str],
) -> ClaimTruthConditionAggregate:
    """Aggregate all registered elements while preserving both status axes."""

    known_element_ids = {element.element_id for element in entry.elements}
    if any(item.element_id not in known_element_ids for item in assessments):
        raise ValueError("assessment references an unregistered element")
    element_aggregates = tuple(
        aggregate_truth_condition_element(
            element,
            tuple(item for item in assessments if item.element_id == element.element_id),
            expected_source_ids=expected_source_ids,
        )
        for element in entry.elements
    )
    by_state = {
        state: tuple(
            item.element_id for item in element_aggregates if item.semantic_state is state
        )
        for state in ElementSemanticState
    }
    supported = by_state[ElementSemanticState.SUPPORTED]
    not_supported = by_state[ElementSemanticState.NOT_SUPPORTED]
    contradicted = by_state[ElementSemanticState.CONTRADICTED]
    conflicted = by_state[ElementSemanticState.CONFLICTED]
    unresolved = by_state[ElementSemanticState.UNRESOLVED]

    if entry.execution_status is not ElementizationExecutionStatus.COMPLETE:
        coverage = ClaimCoverageState.UNRESOLVED
    elif conflicted:
        coverage = ClaimCoverageState.CONFLICTED
    elif supported and contradicted:
        coverage = ClaimCoverageState.MIXED
    elif contradicted:
        coverage = ClaimCoverageState.CONTRADICTED
    elif (
        entry.execution_status is ElementizationExecutionStatus.COMPLETE
        and entry.semantic_status is ElementizationSemanticStatus.COMPLETE
        and element_aggregates
        and len(supported) == len(element_aggregates)
    ):
        coverage = ClaimCoverageState.FULLY_SUPPORTED
    elif supported:
        coverage = ClaimCoverageState.PARTIALLY_SUPPORTED
    elif (
        entry.execution_status is ElementizationExecutionStatus.COMPLETE
        and entry.semantic_status is ElementizationSemanticStatus.COMPLETE
        and element_aggregates
        and len(not_supported) == len(element_aggregates)
    ):
        coverage = ClaimCoverageState.NOT_SUPPORTED
    else:
        coverage = ClaimCoverageState.UNRESOLVED

    return ClaimTruthConditionAggregate(
        claim_id=entry.claim_id,
        elementization_execution_status=entry.execution_status,
        elementization_semantic_status=entry.semantic_status,
        coverage_state=coverage,
        execution_completeness=_claim_execution(entry, element_aggregates),
        elements=element_aggregates,
        supported_element_ids=supported,
        not_supported_element_ids=not_supported,
        contradicted_element_ids=contradicted,
        conflicted_element_ids=conflicted,
        unresolved_element_ids=unresolved,
    )


def aggregate_truth_condition_registry(
    registry: TruthConditionRegistry | None,
    assessments: Sequence[ElementSourceAssessment],
    *,
    expected_source_ids_by_claim: Mapping[str, Sequence[str]],
) -> TruthConditionAggregationResult | None:
    """Aggregate a registry, or return ``None`` for the explicit legacy path."""

    if registry is None:
        return None
    known_claim_ids = {entry.claim_id for entry in registry.entries}
    if set(expected_source_ids_by_claim) - known_claim_ids:
        raise ValueError("expected source mapping contains unknown claims")
    if any(item.claim_id not in known_claim_ids for item in assessments):
        raise ValueError("assessment references an unknown claim")
    claims = tuple(
        aggregate_truth_condition_claim(
            entry,
            tuple(item for item in assessments if item.claim_id == entry.claim_id),
            expected_source_ids=expected_source_ids_by_claim.get(entry.claim_id, ()),
        )
        for entry in registry.entries
    )
    return TruthConditionAggregationResult(
        registry_sha256=truth_condition_registry_sha256(registry),
        claims=claims,
    )
