"""One bounded, post-audit editorial pass over an already written draft.

The editor never decides whether evidence exists.  It receives the immutable
verification registry and may remove dispensable unsupported prose, qualify a
claim, or retain an important unresolved claim for the renderer to label.  A
changed draft is only a proposal until the runner rebuilds the entire claim and
evidence registry against it.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections import defaultdict
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

from open_deep_research.harness.budget import RunCostCapReached
from open_deep_research.harness.claims import (
    CitationRequirement,
    ClaimNormalizationStatus,
    MarkdownBlock,
)
from open_deep_research.harness.jsonio import loads_lenient
from open_deep_research.harness.verify import (
    ClaimEvidenceState,
    ClaimVerification,
    VerificationRecordStatus,
    VerificationResult,
)


class EditorialAction(str, Enum):
    """A semantic editorial decision, never an evidence verdict."""

    REMOVE = "remove"
    QUALIFY = "qualify"
    RETAIN_WITH_LABEL = "retain_with_label"


class EditorialRevisionStatus(str, Enum):
    """Whether every mechanically established edit target was assessed."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class EditorialSettings(BaseModel):
    """Mechanical batching only; no quality threshold lives here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    block_batch_size: int = Field(default=6, ge=1, le=20)


class EditorialDecision(BaseModel):
    """One retained model judgement about an audited claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    action: EditorialAction
    reason: str = Field(min_length=1)


class EditorialBlockEdit(BaseModel):
    """One mechanically bound replacement for a Markdown block."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    block_id: str = Field(min_length=1)
    original_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_text: str
    replacement_text: str
    target_claim_ids: tuple[str, ...]
    decisions: tuple[EditorialDecision, ...]

    @model_validator(mode="after")
    def _decisions_cover_the_block_targets(self) -> EditorialBlockEdit:
        target_ids = tuple(self.target_claim_ids)
        decision_ids = tuple(decision.claim_id for decision in self.decisions)
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("editorial block target claim IDs must be unique")
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("editorial block decisions must be unique")
        if set(target_ids) != set(decision_ids):
            raise ValueError("editorial decisions must cover every block target")
        expected_hash = hashlib.sha256(
            self.original_text.encode("utf-8")
        ).hexdigest()
        if self.original_text_sha256 != expected_hash:
            raise ValueError("editorial block hash does not match original text")
        all_retained = all(
            decision.action is EditorialAction.RETAIN_WITH_LABEL
            for decision in self.decisions
        )
        if all_retained and self.replacement_text != self.original_text:
            raise ValueError("retain-only decisions cannot rewrite the block")
        if not all_retained and self.replacement_text == self.original_text:
            raise ValueError("remove or qualify decisions must change the block")
        return self


class EditorialCallUsage(BaseModel):
    """Measured use of one block batch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_number: int = Field(ge=1)
    block_ids: tuple[str, ...]
    outcome: str = Field(min_length=1)
    token_count: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)


class EditorialBlockDependencyFailure(BaseModel):
    """One incomplete claim that mechanically blocks rewriting its block."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    block_id: str = Field(min_length=1)
    blocked_target_claim_ids: tuple[str, ...]
    dependency_claim_id: str = Field(min_length=1)
    dependency_role: Literal["edit_target", "co_located_non_target"]
    failure_reasons: tuple[str, ...]

    @model_validator(mode="after")
    def _failure_is_specific(self) -> EditorialBlockDependencyFailure:
        if not self.blocked_target_claim_ids:
            raise ValueError("blocked dependency must identify affected targets")
        if not self.failure_reasons:
            raise ValueError("blocked dependency must retain a failure reason")
        dependency_is_target = (
            self.dependency_claim_id in self.blocked_target_claim_ids
        )
        if (
            self.dependency_role == "edit_target"
        ) != dependency_is_target:
            raise ValueError(
                "dependency_role must state whether the failed claim is itself "
                "an edit target"
            )
        return self


class EditorialAdmissionAudit(BaseModel):
    """Mechanical block-closure gate, separate from publication eligibility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gating_unit: Literal["markdown_block"] = "markdown_block"
    target_claim_ids: tuple[str, ...] = ()
    eligible_target_claim_ids: tuple[str, ...] = ()
    blocked_target_claim_ids: tuple[str, ...] = ()
    eligible_block_ids: tuple[str, ...] = ()
    blocked_block_ids: tuple[str, ...] = ()
    blocked_dependencies: tuple[EditorialBlockDependencyFailure, ...] = ()
    unrelated_incomplete_claim_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _scope_is_a_partition(self) -> EditorialAdmissionAudit:
        targets = tuple(self.target_claim_ids)
        eligible = tuple(self.eligible_target_claim_ids)
        blocked = tuple(self.blocked_target_claim_ids)
        if len(set(targets)) != len(targets):
            raise ValueError("editorial admission target IDs must be unique")
        if set(eligible) | set(blocked) != set(targets):
            raise ValueError("eligible and blocked claims must partition targets")
        if set(eligible) & set(blocked):
            raise ValueError("an editorial target cannot be eligible and blocked")
        if set(self.eligible_block_ids) & set(self.blocked_block_ids):
            raise ValueError("an editorial block cannot be eligible and blocked")
        if len(set(self.eligible_block_ids)) != len(self.eligible_block_ids):
            raise ValueError("eligible editorial block IDs must be unique")
        if len(set(self.blocked_block_ids)) != len(self.blocked_block_ids):
            raise ValueError("blocked editorial block IDs must be unique")
        dependencies_by_target = {
            target_id
            for dependency in self.blocked_dependencies
            for target_id in dependency.blocked_target_claim_ids
        }
        if blocked and dependencies_by_target != set(blocked):
            raise ValueError(
                "every blocked target must be explained by a block dependency"
            )
        if any(
            dependency.block_id not in self.blocked_block_ids
            for dependency in self.blocked_dependencies
        ):
            raise ValueError("blocked dependencies must belong to blocked blocks")
        if set(self.unrelated_incomplete_claim_ids) & set(targets):
            raise ValueError("unrelated incomplete claims cannot be edit targets")
        return self


class EditorialRevisionResult(BaseModel):
    """Auditable proposal; incomplete proposals are never partly applied."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: EditorialRevisionStatus
    original_draft: str
    edited_draft: str
    original_draft_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    edited_draft_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_claim_ids: tuple[str, ...] = ()
    gating_unit: Literal["markdown_block"] = "markdown_block"
    eligible_target_claim_ids: tuple[str, ...] = ()
    blocked_target_claim_ids: tuple[str, ...] = ()
    eligible_block_ids: tuple[str, ...] = ()
    blocked_block_ids: tuple[str, ...] = ()
    blocked_dependencies: tuple[EditorialBlockDependencyFailure, ...] = ()
    unrelated_incomplete_claim_ids: tuple[str, ...] = ()
    evaluated_claim_ids: tuple[str, ...] = ()
    unevaluated_claim_ids: tuple[str, ...] = ()
    block_edits: tuple[EditorialBlockEdit, ...] = ()
    diagnostics: tuple[str, ...] = ()
    usage: tuple[EditorialCallUsage, ...] = ()
    changes_applied: bool = False
    requires_reaudit: bool = False
    committed_after_reaudit: bool = False

    @model_validator(mode="after")
    def _result_is_atomic_and_hash_bound(self) -> EditorialRevisionResult:
        original_hash = hashlib.sha256(
            self.original_draft.encode("utf-8")
        ).hexdigest()
        edited_hash = hashlib.sha256(
            self.edited_draft.encode("utf-8")
        ).hexdigest()
        if self.original_draft_sha256 != original_hash:
            raise ValueError("original editorial draft hash mismatch")
        if self.edited_draft_sha256 != edited_hash:
            raise ValueError("edited editorial draft hash mismatch")
        targets = tuple(self.target_claim_ids)
        eligible = tuple(self.eligible_target_claim_ids)
        blocked = tuple(self.blocked_target_claim_ids)
        evaluated = tuple(self.evaluated_claim_ids)
        unevaluated = tuple(self.unevaluated_claim_ids)
        if len(set(targets)) != len(targets):
            raise ValueError("editorial target claim IDs must be unique")
        if set(eligible) | set(blocked) != set(targets):
            raise ValueError("eligible and blocked claims must partition targets")
        if set(eligible) & set(blocked):
            raise ValueError("an editorial target cannot be eligible and blocked")
        if set(self.eligible_block_ids) & set(self.blocked_block_ids):
            raise ValueError("an editorial block cannot be eligible and blocked")
        dependencies_by_target = {
            target_id
            for dependency in self.blocked_dependencies
            for target_id in dependency.blocked_target_claim_ids
        }
        if blocked and dependencies_by_target != set(blocked):
            raise ValueError(
                "every blocked revision target needs a dependency failure"
            )
        if set(self.unrelated_incomplete_claim_ids) & set(targets):
            raise ValueError("unrelated incomplete claims cannot be edit targets")
        if set(evaluated) | set(unevaluated) != set(eligible):
            raise ValueError(
                "evaluated and unevaluated claims must partition eligible targets"
            )
        if set(evaluated) & set(unevaluated):
            raise ValueError(
                "an eligible target cannot be evaluated and unevaluated"
            )
        if self.status is EditorialRevisionStatus.COMPLETE and unevaluated:
            raise ValueError("complete editorial revision cannot omit targets")
        changed = self.edited_draft != self.original_draft
        if self.changes_applied != changed:
            raise ValueError("changes_applied must reflect the actual draft bytes")
        if self.requires_reaudit != changed:
            raise ValueError("every changed draft requires a fresh evidence audit")
        if self.status is not EditorialRevisionStatus.COMPLETE and changed:
            raise ValueError("incomplete editorial output cannot be partly applied")
        if self.committed_after_reaudit and not changed:
            raise ValueError("only a changed draft can be committed after re-audit")
        return self

    @property
    def total_tokens(self) -> int:
        return sum(record.token_count for record in self.usage)

    @property
    def total_cost_usd(self) -> float:
        return sum(record.cost_usd for record in self.usage)


class EditorialModelClient(Protocol):
    """Injected model boundary for the independent editorial judgement."""

    def generate(self, prompt: str) -> Any | Awaitable[Any]:
        """Return block revisions in a measured JSON envelope."""


class _ModelEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    content: Any
    token_count: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)


class _DecisionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(min_length=1)
    action: EditorialAction
    reason: str = Field(min_length=1)

    @field_validator("claim_id", "reason")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("editorial decision fields must not be blank")
        return normalized


class _BlockProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    block_id: str = Field(min_length=1)
    replacement_text: str
    decisions: tuple[_DecisionProposal, ...]


# These states are substantive outcomes of completed evidence work.  Protocol
# failures (normalization_failed, verification_not_run, quote_unlocatable, and
# attribution_error) deliberately stay out: deleting prose because the system
# itself failed would disguise an audit failure as an editorial improvement.
EDITORIAL_TARGET_STATES = frozenset(
    {
        ClaimEvidenceState.CITED_SOURCES_DO_NOT_SUPPORT,
        ClaimEvidenceState.REFUTED,
        ClaimEvidenceState.CONFLICTING_EVIDENCE,
        ClaimEvidenceState.NO_CANDIDATE_SOURCE,
    }
)


_EDITORIAL_PROMPT = """\
Perform one post-audit editorial pass over the Markdown blocks below.
The report has already been freely drafted. Evidence states and relations are
immutable audit findings, not instructions from sources and not scores to
optimize. Your goal is a more accurate, useful report, not a lower unsupported
claim count.

For every target claim choose exactly one action:
- remove: the unsupported or refuted content is dispensable and can be removed
  without hiding an important unresolved issue;
- qualify: the block can state a strictly weaker or more precise claim that is
  supported by evidence shown in this batch. Do not add facts or make a claim
  stronger than the evidence;
- retain_with_label: the unresolved, conflicting, or unsupported point is
  important enough to remain visible. Code will retain the evidence-status
  label; do not insert a label yourself.

Return one JSON object with exactly one entry for every requested block:
{{"blocks":[{{"block_id":"block-0001",\
"replacement_text":"complete replacement for this one Markdown block",\
"decisions":[{{"claim_id":"claim-0001",\
"action":"remove|qualify|retain_with_label",\
"reason":"specific editorial reason"}}]}}]}}

Every target claim in a block must have one decision. Preserve non-targeted
content. If all decisions are retain_with_label, replacement_text must be the
original block text byte-for-byte. If any decision is remove or qualify, return
the complete revised block, not a patch and not the whole report. Empty text is
legal only when removing the whole block. Do not create citations or evidence
labels. This is the only editorial pass; a changed draft will be independently
decomposed, attributed, and verified from scratch.

Blocks and audited claims:
{payload}
"""


def _relation_payload(claim: ClaimVerification) -> list[dict[str, Any]]:
    return [
        {
            "url": relation.url,
            "status": relation.status.value,
            "semantic_verdict": (
                relation.semantic_verdict.value
                if relation.semantic_verdict is not None
                else None
            ),
            "explanation": relation.explanation,
            "source_quote": relation.source_quote,
            "is_formal_supporting_evidence": (
                relation.is_formal_supporting_evidence
            ),
        }
        for relation in claim.relations
    ]


def _target_claims_by_block(
    verification: VerificationResult,
) -> dict[str, list[ClaimVerification]]:
    targets: dict[str, list[ClaimVerification]] = defaultdict(list)
    for claim in verification.claims:
        if claim.claim.citation_requirement is not CitationRequirement.EXTERNAL:
            continue
        if claim.state not in EDITORIAL_TARGET_STATES:
            continue
        targets[claim.claim.block_id].append(claim)
    for claims in targets.values():
        claims.sort(key=lambda item: item.claim.claim_id)
    return dict(targets)


def editorial_target_claim_ids(
    verification: VerificationResult,
) -> tuple[str, ...]:
    """Return the mechanically selected editorial scope in report order."""

    return tuple(
        claim.claim.claim_id
        for claim in verification.claims
        if claim.claim.citation_requirement is CitationRequirement.EXTERNAL
        and claim.state in EDITORIAL_TARGET_STATES
    )


_INCOMPLETE_CLAIM_STATES = frozenset(
    {
        ClaimEvidenceState.NORMALIZATION_FAILED,
        ClaimEvidenceState.ATTRIBUTION_ERROR,
        ClaimEvidenceState.VERIFICATION_INCOMPLETE,
        ClaimEvidenceState.VERIFICATION_NOT_RUN,
        ClaimEvidenceState.SUPPORT_QUOTE_UNLOCATABLE,
    }
)


def _claim_audit_failure_reasons(
    claim: ClaimVerification,
) -> tuple[str, ...]:
    """Return only protocol/audit failures, never negative evidence findings."""

    reasons: list[str] = []
    if (
        claim.claim.normalization_status
        is ClaimNormalizationStatus.NORMALIZATION_FAILED
    ):
        detail = claim.claim.normalization_failure or "unspecified"
        reasons.append(f"normalization_failed:{detail}")
    elif claim.state in _INCOMPLETE_CLAIM_STATES:
        reasons.append(f"claim_state:{claim.state.value}")
    for relation in claim.relations:
        if relation.status is not VerificationRecordStatus.COMPLETED:
            reasons.append(
                "claim_source_relation_incomplete:"
                f"{relation.source_id}:{relation.status.value}"
            )
    return tuple(dict.fromkeys(reasons))


def audit_editorial_admission(
    verification: VerificationResult,
    *,
    blocks: Sequence[MarkdownBlock],
) -> EditorialAdmissionAudit:
    """Admit only target blocks whose complete claim dependency closure is audited.

    The gate is deliberately local to the bytes the editor may replace. A
    failure in another block remains visible as unrelated incomplete work and
    still affects global publication eligibility, but cannot veto a safe block.
    """

    block_by_id = {block.block_id: block for block in blocks}
    if len(block_by_id) != len(blocks):
        raise ValueError("editorial admission requires unique block IDs")
    claim_ids = tuple(
        claim.claim.claim_id for claim in verification.claims
    )
    if len(set(claim_ids)) != len(claim_ids):
        raise ValueError("editorial admission requires unique claim IDs")
    unknown_claim_blocks = tuple(
        sorted(
            {
                claim.claim.block_id
                for claim in verification.claims
                if claim.claim.block_id not in block_by_id
            }
        )
    )
    if unknown_claim_blocks:
        raise ValueError(
            "verification refers to unknown editorial blocks: "
            + ", ".join(unknown_claim_blocks)
        )
    targets = _target_claims_by_block(verification)

    claims_by_block: dict[str, list[ClaimVerification]] = defaultdict(list)
    for claim in verification.claims:
        claims_by_block[claim.claim.block_id].append(claim)
    ordered_target_blocks = sorted(
        targets,
        key=lambda block_id: block_by_id[block_id].ordinal,
    )
    target_ids = tuple(
        claim.claim.claim_id
        for block_id in ordered_target_blocks
        for claim in targets[block_id]
    )
    target_id_set = set(target_ids)
    eligible_target_ids: list[str] = []
    blocked_target_ids: list[str] = []
    eligible_block_ids: list[str] = []
    blocked_block_ids: list[str] = []
    blocked_dependencies: list[EditorialBlockDependencyFailure] = []

    for block_id in ordered_target_blocks:
        block_target_ids = tuple(
            claim.claim.claim_id for claim in targets[block_id]
        )
        incomplete_dependencies = [
            (claim, _claim_audit_failure_reasons(claim))
            for claim in claims_by_block.get(block_id, ())
        ]
        incomplete_dependencies = [
            (claim, reasons)
            for claim, reasons in incomplete_dependencies
            if reasons
        ]
        if not incomplete_dependencies:
            eligible_block_ids.append(block_id)
            eligible_target_ids.extend(block_target_ids)
            continue
        blocked_block_ids.append(block_id)
        blocked_target_ids.extend(block_target_ids)
        for dependency, reasons in incomplete_dependencies:
            dependency_id = dependency.claim.claim_id
            blocked_dependencies.append(
                EditorialBlockDependencyFailure(
                    block_id=block_id,
                    blocked_target_claim_ids=block_target_ids,
                    dependency_claim_id=dependency_id,
                    dependency_role=(
                        "edit_target"
                        if dependency_id in target_id_set
                        else "co_located_non_target"
                    ),
                    failure_reasons=reasons,
                )
            )

    target_block_ids = set(ordered_target_blocks)
    unrelated_incomplete = tuple(
        claim.claim.claim_id
        for claim in verification.claims
        if claim.claim.block_id not in target_block_ids
        and _claim_audit_failure_reasons(claim)
    )
    return EditorialAdmissionAudit(
        target_claim_ids=target_ids,
        eligible_target_claim_ids=tuple(eligible_target_ids),
        blocked_target_claim_ids=tuple(blocked_target_ids),
        eligible_block_ids=tuple(eligible_block_ids),
        blocked_block_ids=tuple(blocked_block_ids),
        blocked_dependencies=tuple(blocked_dependencies),
        unrelated_incomplete_claim_ids=unrelated_incomplete,
    )


def build_editorial_prompt(
    blocks: Sequence[MarkdownBlock],
    *,
    verification: VerificationResult,
) -> str:
    """Build one batch prompt with exact blocks and immutable audit outcomes."""

    target_by_block = _target_claims_by_block(verification)
    all_claims_by_block: dict[str, list[ClaimVerification]] = defaultdict(list)
    for claim in verification.claims:
        all_claims_by_block[claim.claim.block_id].append(claim)
    payload = []
    for block in blocks:
        targets = target_by_block.get(block.block_id, ())
        if not targets:
            continue
        target_ids = {claim.claim.claim_id for claim in targets}
        payload.append(
            {
                "block_id": block.block_id,
                "kind": block.kind.value,
                "original_text": block.text,
                "target_claim_ids": sorted(target_ids),
                "claims": [
                    {
                        "claim_id": claim.claim.claim_id,
                        "claim_text": claim.claim.claim_text,
                        "anchor_text": claim.claim.anchor_text,
                        "evidence_state": claim.state.value,
                        "is_edit_target": claim.claim.claim_id in target_ids,
                        "relations": _relation_payload(claim),
                    }
                    for claim in sorted(
                        all_claims_by_block.get(block.block_id, ()),
                        key=lambda item: item.claim.claim_id,
                    )
                ],
            }
        )
    return _EDITORIAL_PROMPT.format(
        payload=json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


async def _call_model(
    client: EditorialModelClient,
    prompt: str,
) -> tuple[Any, int, float, str | None]:
    try:
        response = client.generate(prompt)
        if inspect.isawaitable(response):
            response = await response
    except RunCostCapReached:
        raise
    except Exception as exc:  # provider failure becomes an auditable result
        return None, 0, 0.0, f"{type(exc).__name__}: {exc}"
    tokens = 0
    cost = 0.0
    if isinstance(response, Mapping):
        try:
            tokens = max(0, int(response.get("token_count", 0)))
            cost = max(0.0, float(response.get("cost_usd", 0.0)))
        except (TypeError, ValueError):
            pass
    try:
        envelope = _ModelEnvelope.model_validate(response)
    except ValidationError as exc:
        return None, tokens, cost, f"invalid usage envelope: {exc}"
    content = envelope.content
    if isinstance(content, str):
        try:
            content = loads_lenient(content)
        except json.JSONDecodeError as exc:
            return None, envelope.token_count, envelope.cost_usd, str(exc)
    return content, envelope.token_count, envelope.cost_usd, None


def _parse_batch(
    content: Any,
    *,
    blocks: Mapping[str, MarkdownBlock],
    targets: Mapping[str, Sequence[ClaimVerification]],
) -> tuple[list[EditorialBlockEdit], list[str]]:
    diagnostics: list[str] = []
    if not isinstance(content, Mapping):
        return [], ["editorial response was not a JSON object"]
    raw_blocks = content.get("blocks")
    if not isinstance(raw_blocks, (list, tuple)):
        return [], ["editorial blocks was not an array"]
    parsed_by_id: dict[str, _BlockProposal] = {}
    duplicates: set[str] = set()
    for index, raw in enumerate(raw_blocks):
        try:
            proposal = _BlockProposal.model_validate(raw)
        except (TypeError, ValidationError, ValueError) as exc:
            diagnostics.append(f"editorial_block_invalid[{index}]: {exc}")
            continue
        if proposal.block_id not in blocks:
            diagnostics.append(f"unknown_editorial_block: {proposal.block_id}")
            continue
        if proposal.block_id in parsed_by_id:
            duplicates.add(proposal.block_id)
            parsed_by_id.pop(proposal.block_id, None)
            diagnostics.append(f"duplicate_editorial_block: {proposal.block_id}")
            continue
        if proposal.block_id not in duplicates:
            parsed_by_id[proposal.block_id] = proposal

    edits: list[EditorialBlockEdit] = []
    for block_id, block in blocks.items():
        proposal = parsed_by_id.get(block_id)
        if proposal is None:
            diagnostics.append(f"editorial_block_missing: {block_id}")
            continue
        target_ids = tuple(
            claim.claim.claim_id for claim in targets[block_id]
        )
        decision_ids = tuple(
            decision.claim_id for decision in proposal.decisions
        )
        if len(set(decision_ids)) != len(decision_ids):
            diagnostics.append(f"duplicate_editorial_claim: {block_id}")
            continue
        if set(decision_ids) != set(target_ids):
            missing = sorted(set(target_ids) - set(decision_ids))
            unknown = sorted(set(decision_ids) - set(target_ids))
            diagnostics.append(
                f"editorial_claim_coverage_error: {block_id}; "
                f"missing={missing}; unknown={unknown}"
            )
            continue
        try:
            edits.append(
                EditorialBlockEdit(
                    block_id=block_id,
                    original_text_sha256=hashlib.sha256(
                        block.text.encode("utf-8")
                    ).hexdigest(),
                    original_text=block.text,
                    replacement_text=proposal.replacement_text,
                    target_claim_ids=target_ids,
                    decisions=tuple(
                        EditorialDecision.model_validate(
                            decision.model_dump(mode="python")
                        )
                        for decision in proposal.decisions
                    ),
                )
            )
        except ValidationError as exc:
            diagnostics.append(f"editorial_block_rejected: {block_id}; {exc}")
    return edits, diagnostics


def _apply_block_edits(
    draft: str,
    blocks: Mapping[str, MarkdownBlock],
    edits: Sequence[EditorialBlockEdit],
) -> str:
    revised = draft
    for edit in sorted(
        edits,
        key=lambda item: blocks[item.block_id].start_char,
        reverse=True,
    ):
        block = blocks[edit.block_id]
        if draft[block.start_char : block.end_char] != edit.original_text:
            raise AssertionError("editorial block no longer matches canonical draft")
        revised = (
            revised[: block.start_char]
            + edit.replacement_text
            + revised[block.end_char :]
        )
    return revised


async def revise_audited_draft(
    canonical_draft: str,
    *,
    blocks: Sequence[MarkdownBlock],
    verification: VerificationResult,
    model_client: EditorialModelClient,
    settings: EditorialSettings | None = None,
) -> EditorialRevisionResult:
    """Run exactly one editorial pass and apply it only when fully covered."""

    active_settings = settings or EditorialSettings()
    block_by_id = {block.block_id: block for block in blocks}
    if len(block_by_id) != len(blocks):
        raise ValueError("editorial revision requires unique block IDs")
    for block in blocks:
        if canonical_draft[block.start_char : block.end_char] != block.text:
            raise ValueError(f"block {block.block_id} is not bound to the draft")
    admission = audit_editorial_admission(verification, blocks=blocks)
    all_targets = _target_claims_by_block(verification)
    targets = {
        block_id: all_targets[block_id]
        for block_id in admission.eligible_block_ids
    }
    target_ids = admission.target_claim_ids
    eligible_target_ids = admission.eligible_target_claim_ids
    draft_hash = hashlib.sha256(canonical_draft.encode("utf-8")).hexdigest()
    if not eligible_target_ids:
        return EditorialRevisionResult(
            status=EditorialRevisionStatus.COMPLETE,
            original_draft=canonical_draft,
            edited_draft=canonical_draft,
            original_draft_sha256=draft_hash,
            edited_draft_sha256=draft_hash,
            target_claim_ids=target_ids,
            eligible_target_claim_ids=eligible_target_ids,
            blocked_target_claim_ids=admission.blocked_target_claim_ids,
            eligible_block_ids=admission.eligible_block_ids,
            blocked_block_ids=admission.blocked_block_ids,
            blocked_dependencies=admission.blocked_dependencies,
            unrelated_incomplete_claim_ids=(
                admission.unrelated_incomplete_claim_ids
            ),
        )

    target_blocks = sorted(
        (block_by_id[block_id] for block_id in targets),
        key=lambda block: block.ordinal,
    )
    edits: list[EditorialBlockEdit] = []
    diagnostics: list[str] = []
    usage: list[EditorialCallUsage] = []
    for start in range(0, len(target_blocks), active_settings.block_batch_size):
        batch = target_blocks[start : start + active_settings.block_batch_size]
        batch_ids = tuple(block.block_id for block in batch)
        prompt = build_editorial_prompt(batch, verification=verification)
        content, tokens, cost, error = await _call_model(model_client, prompt)
        batch_number = len(usage) + 1
        if error is not None:
            diagnostics.append(f"editorial_batch_failed[{batch_number}]: {error}")
            usage.append(
                EditorialCallUsage(
                    batch_number=batch_number,
                    block_ids=batch_ids,
                    outcome="failed",
                    token_count=tokens,
                    cost_usd=cost,
                )
            )
            continue
        batch_blocks = {block.block_id: block for block in batch}
        batch_targets = {
            block_id: targets[block_id] for block_id in batch_ids
        }
        batch_edits, batch_diagnostics = _parse_batch(
            content,
            blocks=batch_blocks,
            targets=batch_targets,
        )
        edits.extend(batch_edits)
        diagnostics.extend(
            f"batch[{batch_number}]: {message}"
            for message in batch_diagnostics
        )
        usage.append(
            EditorialCallUsage(
                batch_number=batch_number,
                block_ids=batch_ids,
                outcome=("completed" if not batch_diagnostics else "partial"),
                token_count=tokens,
                cost_usd=cost,
            )
        )

    evaluated_ids = tuple(
        decision.claim_id
        for edit in sorted(
            edits,
            key=lambda item: block_by_id[item.block_id].ordinal,
        )
        for decision in edit.decisions
    )
    evaluated_set = set(evaluated_ids)
    unevaluated_ids = tuple(
        claim_id
        for claim_id in eligible_target_ids
        if claim_id not in evaluated_set
    )
    complete = (
        not unevaluated_ids
        and len(evaluated_ids) == len(eligible_target_ids)
    )
    if complete:
        edited_draft = _apply_block_edits(canonical_draft, block_by_id, edits)
        status = EditorialRevisionStatus.COMPLETE
    else:
        # Valid block decisions remain in the audit, but a partial rewrite is
        # never applied. This prevents one malformed block from manufacturing a
        # report whose editorial contract was only partly executed.
        edited_draft = canonical_draft
        status = (
            EditorialRevisionStatus.PARTIAL
            if evaluated_ids
            else EditorialRevisionStatus.FAILED
        )
    edited_hash = hashlib.sha256(edited_draft.encode("utf-8")).hexdigest()
    return EditorialRevisionResult(
        status=status,
        original_draft=canonical_draft,
        edited_draft=edited_draft,
        original_draft_sha256=draft_hash,
        edited_draft_sha256=edited_hash,
        target_claim_ids=target_ids,
        eligible_target_claim_ids=eligible_target_ids,
        blocked_target_claim_ids=admission.blocked_target_claim_ids,
        eligible_block_ids=admission.eligible_block_ids,
        blocked_block_ids=admission.blocked_block_ids,
        blocked_dependencies=admission.blocked_dependencies,
        unrelated_incomplete_claim_ids=admission.unrelated_incomplete_claim_ids,
        evaluated_claim_ids=evaluated_ids,
        unevaluated_claim_ids=unevaluated_ids,
        block_edits=tuple(edits),
        diagnostics=tuple(diagnostics),
        usage=tuple(usage),
        changes_applied=(edited_draft != canonical_draft),
        requires_reaudit=(edited_draft != canonical_draft),
    )


__all__ = [
    "EDITORIAL_TARGET_STATES",
    "EditorialAdmissionAudit",
    "EditorialAction",
    "EditorialBlockEdit",
    "EditorialBlockDependencyFailure",
    "EditorialCallUsage",
    "EditorialDecision",
    "EditorialModelClient",
    "EditorialRevisionResult",
    "EditorialRevisionStatus",
    "EditorialSettings",
    "audit_editorial_admission",
    "build_editorial_prompt",
    "editorial_target_claim_ids",
    "revise_audited_draft",
]
