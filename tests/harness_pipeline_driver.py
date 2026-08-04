"""Reusable offline driver for exercising seams across ``run_harness``.

The driver composes the long-lived deterministic fakes in
``test_harness_runner`` and adds only orchestration: all optional post-draft
passes are enabled, and a real ``RunCostController`` can reject the first call
owned by any named stage.  No production interface exists solely for tests.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from open_deep_research.harness.budget import RunCostBudget
from open_deep_research.harness.disagreement import (
    DisagreementBudget,
    PosthocRetrievalBudget,
)
from open_deep_research.harness.evidence_gap import EvidenceGapBudget
from open_deep_research.harness.edit import EditorialSettings
from open_deep_research.harness.loop import LoopBudget
from open_deep_research.harness.recovery import RecoveryTriageSettings
from open_deep_research.harness.runner import HarnessRunResult, run_harness
from open_deep_research.harness.stages import STAGE_AUDIT_PAYLOAD_KEYS

import test_harness_runner as base


# These are the fourteen model-backed stages whose payloads are independently
# named in the durable audit. Disagreement and deterministic rendering remain
# observable in stage_execution but do not have entries in the payload map.
POST_DRAFT_SEAM_STAGES = tuple(STAGE_AUDIT_PAYLOAD_KEYS)


class _EstimatedStageModel:
    """Forward a fake model while making one stage unaffordable."""

    def __init__(
        self,
        delegate: Any,
        classify: Callable[[str, int], str],
        *,
        rejected_stage: str | None,
    ) -> None:
        self.delegate = delegate
        self.classify = classify
        self.rejected_stage = rejected_stage
        self.completed_calls = 0

    def estimate_tokens(self, prompt: str) -> int:
        return 1

    def estimate_cost_usd(self, prompt: str) -> float:
        stage = self.classify(prompt, self.completed_calls)
        return 10.0 if stage == self.rejected_stage else 0.0001

    async def generate(self, prompt: str) -> Any:
        response = self.delegate.generate(prompt)
        if hasattr(response, "__await__"):
            response = await response
        self.completed_calls += 1
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


class _SeamDecisionModel(base.ReadThenSettleDecisionModel):
    """Collection plus the three bounded retrieval planners."""

    def __init__(
        self,
        events: list[str],
        url: str,
        *,
        select_all_disagreements: bool = False,
    ) -> None:
        super().__init__(events, url)
        self.select_all_disagreements = select_all_disagreements

    async def generate(self, prompt: str) -> Any:
        if "Choose full pages to read for one bounded evidence-gap pass" in prompt:
            return _envelope(
                {
                    "reads": [
                        {
                            "url": "https://alternative.example/record",
                            "item_id": "what-1",
                            "claim_ids": ["claim-0002"],
                            "independent_from_existing_publishers": True,
                            "publisher_identity": "Alternative Example",
                            "independence_rationale": "A distinct synthetic source.",
                        }
                    ]
                }
            )
        if "only bounded evidence-recovery pass" in prompt:
            return _envelope(
                {
                    "cached_candidates": [],
                    "queries": [_query("recovery seam query", "claim-0001")],
                }
            )
        if "disagreement-detection selection pass" in prompt:
            claim_ids = (
                ("claim-0001", "claim-0002")
                if self.select_all_disagreements
                else ("claim-0001",)
            )
            return _envelope(
                {
                    "claims": [
                        {
                            "claim_id": claim_id,
                            "reason": "An alternative source check is useful.",
                        }
                        for claim_id in claim_ids
                    ]
                }
            )
        if "alternative-source planning pass" in prompt:
            return _envelope(
                {
                    "cached_candidates": [],
                    "queries": [_query("disagreement seam query")],
                }
            )
        if "evidence-gap planning pass" in prompt:
            return _envelope(
                {
                    "cached_candidates": [],
                    "queries": [_query("evidence gap seam query", "claim-0002")],
                }
            )
        return await super().generate(prompt)


class _SearchEmptyReadAvailable(base.ReadingTavily):
    def __init__(
        self,
        url: str,
        gap_url: str,
        *,
        fail_gap_search: bool = False,
        source_lead_count: int = 0,
    ) -> None:
        super().__init__(url)
        self.gap_url = gap_url
        self.fail_gap_search = fail_gap_search
        self.source_lead_count = source_lead_count

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        if "evidence gap seam" in query:
            if self.fail_gap_search:
                raise RuntimeError("synthetic provider failure")
            return {
                "results": [
                    {
                        "url": self.gap_url,
                        "title": "Alternative record",
                        "content": "ContrarySourceEvidence 2026.",
                    }
                ]
            }
        return {"results": []}

    async def extract(self, urls: list[str], **kwargs: Any) -> dict[str, Any]:
        if urls == [self.gap_url]:
            return {
                "results": [
                    {
                        "url": self.gap_url,
                        "raw_content": "ContrarySourceEvidence 2026.",
                    }
                ]
            }
        if urls == [self.url] and self.source_lead_count:
            leads = "\n".join(
                f"https://records.example/document-{index:04d}.pdf"
                for index in range(self.source_lead_count)
            )
            return {
                "results": [
                    {
                        "url": self.url,
                        "raw_content": f"ExactSourceEvidence 2026.\n{leads}",
                    }
                ]
            }
        return await super().extract(urls, **kwargs)


class _SeamNoteModel(base.OneNoteModel):
    async def generate(self, prompt: str) -> Any:
        if "Extract zero or more research notes" in prompt:
            return _envelope(
                {
                    "notes": [
                        {
                            "item_id": "what-1",
                            "finding": "An alternative source disagrees.",
                            "quote": "ContrarySourceEvidence 2026.",
                        }
                    ]
                }
            )
        return await super().generate(prompt)


class _SeamAttributionModel:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt: str) -> Any:
        self.calls += 1
        refs = re.findall(r"nref-[0-9a-f]{16}", prompt)
        assert refs
        unique_refs = tuple(dict.fromkeys(refs))
        selected = unique_refs[-1] if self.calls == 2 else unique_refs[0]
        claim_ids = tuple(
            dict.fromkeys(re.findall(r'"claim_id": "(claim-[0-9]+)"', prompt))
        )
        return _envelope(
            {
                "action": "attribute",
                "claims": [
                    {
                        "claim_id": claim_id,
                        "candidates": [
                            {
                                "note_ref": selected,
                                "inherited_from_claim_id": None,
                            }
                        ],
                    }
                    for claim_id in claim_ids
                ],
            }
        )


class _SeamVerificationModel:
    def __init__(
        self,
        both_claims_unsupported: bool = False,
        both_claims_supported: bool = False,
    ) -> None:
        self.calls = 0
        self.both_claims_unsupported = both_claims_unsupported
        self.both_claims_supported = both_claims_supported

    async def generate(self, prompt: str) -> Any:
        self.calls += 1
        claim_ids = tuple(
            dict.fromkeys(re.findall(r'"claim_id": "(claim-[0-9]+)"', prompt))
        )
        results = []
        for claim_id in claim_ids:
            if self.calls == 1 and self.both_claims_supported:
                verdict = "supports"
            elif self.calls == 1 and (
                claim_id == "claim-0001" or self.both_claims_unsupported
            ):
                verdict = "does_not_support"
            elif self.calls == 2:
                verdict = "contradicts"
            else:
                verdict = "supports"
            evidentiary = verdict in {"supports", "contradicts"}
            results.append(
                {
                    "claim_id": claim_id,
                    "verdict": verdict,
                    "start_segment_id": "S000001" if evidentiary else None,
                    "end_segment_id": "S000001" if evidentiary else None,
                    "explanation": "Synthetic seam verdict.",
                }
            )
        return _envelope({"results": results})


class _SeamRecoveryModel:
    async def generate(self, prompt: str) -> Any:
        claim_ids = tuple(
            dict.fromkeys(re.findall(r'"claim_id": "(claim-[0-9]+)"', prompt))
        )
        decisions = []
        for claim_id in claim_ids:
            research = claim_id == "claim-0001"
            decisions.append(
                {
                    "claim_id": claim_id,
                    "action": "research_more" if research else "edit_directly",
                    "importance": "central",
                    "importance_reason": "The claim is in the requested answer.",
                    "evidence_need": "A focused record" if research else None,
                    "preferred_source_role": "underlying record" if research else None,
                    "query": "recovery seam query" if research else None,
                    "selected_source_lead_id": None,
                }
            )
        return _envelope({"decisions": decisions})


class _PromptCapacityRecoveryModel(_SeamRecoveryModel):
    """Behave like a provider that returns no choice for an oversized prompt."""

    async def generate(self, prompt: str) -> Any:
        if len(prompt) > 100_000:
            return None
        return await super().generate(prompt)


class _SeamEditorModel:
    async def generate(self, prompt: str) -> Any:
        return _envelope(
            {
                "blocks": [
                    {
                        "block_id": "block-0002",
                        "replacement_text": "A narrower supported fact.",
                        "decisions": [
                            {
                                "claim_id": "claim-0001",
                                "action": "qualify",
                                "reason": "Use the narrower audited wording.",
                            }
                        ],
                    },
                    {
                        "block_id": "block-0003",
                        "replacement_text": "A separate audited assertion.",
                        "decisions": [
                            {
                                "claim_id": "claim-0002",
                                "action": "retain_with_label",
                                "reason": "Keep the unresolved assertion labelled.",
                            }
                        ],
                    },
                ]
            }
        )


def _query(text: str, claim_id: str = "claim-0001") -> dict[str, Any]:
    return {
        "claim_ids": [claim_id],
        "item_id": "what-1",
        "query": text,
    }


def _envelope(content: Any) -> dict[str, Any]:
    return {
        "content": json.dumps(content),
        "token_count": 2,
        "cost_usd": 0.001,
    }


def _decision_stage(prompt: str, call: int) -> str:
    if "only bounded evidence-recovery pass" in prompt:
        return "evidence_recovery"
    if "disagreement-detection selection pass" in prompt or (
        "alternative-source planning pass" in prompt
    ):
        return "disagreement"
    if "evidence-gap planning pass" in prompt:
        return "evidence_gap"
    return "collection"


def _claim_stage(prompt: str, call: int) -> str:
    post_edit = "A narrower supported fact." in prompt
    phase = call % 4
    if post_edit:
        return (
            "post_edit_evaluative_diagnostics"
            if phase == 3
            else "post_edit_claim_decomposition"
        )
    return "evaluative_diagnostics" if phase == 3 else "claim_decomposition"


def _ordinal_stage(first: str, second: str) -> Callable[[str, int], str]:
    return lambda prompt, call: first if call == 0 else second


def _attribution_stage(prompt: str, call: int) -> str:
    if "A narrower supported fact." in prompt:
        return "post_edit_attribution"
    return "attribution" if call == 0 else "evidence_gap"


def _verification_stage(prompt: str, call: int) -> str:
    if "A narrower supported fact." in prompt:
        return "post_edit_initial_verification"
    return "initial_verification" if call == 0 else "evidence_gap"


def _constant_stage(stage: str) -> Callable[[str, int], str]:
    return lambda prompt, call: stage


@dataclass
class FullPipelineSeamDriver:
    """Two-claim run that reaches every optional post-draft subsystem."""

    output_root: Path
    rejected_stage: str | None = None
    select_all_disagreements: bool = False
    fail_gap_search: bool = False
    source_lead_count: int = 0
    prompt_capacity_recovery: bool = False
    both_claims_unsupported: bool = False
    both_claims_supported: bool = False
    recovery_model: Any | None = None
    editor_model: Any | None = None
    recovery_triage_settings: RecoveryTriageSettings | None = None
    editorial_settings: EditorialSettings | None = None
    loop_budget: LoopBudget | None = None
    run_cost_budget: RunCostBudget | None = None

    async def run(self, run_id: str) -> HarnessRunResult:
        events: list[str] = []
        source_url = "https://source.example/full-pipeline"
        gap_url = "https://alternative.example/record"
        decision = _EstimatedStageModel(
            _SeamDecisionModel(
                events,
                source_url,
                select_all_disagreements=self.select_all_disagreements,
            ),
            _decision_stage,
            rejected_stage=self.rejected_stage,
        )
        claim = _EstimatedStageModel(
            base.TwoBlockReauditClaimModel(),
            _claim_stage,
            rejected_stage=self.rejected_stage,
        )
        reconciliation = _EstimatedStageModel(
            base.CoverageModel(events),
            _ordinal_stage(
                "checklist_reconciliation",
                "post_edit_checklist_reconciliation",
            ),
            rejected_stage=self.rejected_stage,
        )
        attribution = _EstimatedStageModel(
            _SeamAttributionModel(),
            _attribution_stage,
            rejected_stage=self.rejected_stage,
        )
        verification = _EstimatedStageModel(
            _SeamVerificationModel(
                self.both_claims_unsupported,
                self.both_claims_supported,
            ),
            _verification_stage,
            rejected_stage=self.rejected_stage,
        )
        recovery = _EstimatedStageModel(
            self.recovery_model
            or (
                _PromptCapacityRecoveryModel()
                if self.prompt_capacity_recovery
                else _SeamRecoveryModel()
            ),
            _constant_stage("recovery_triage"),
            rejected_stage=self.rejected_stage,
        )
        editor = _EstimatedStageModel(
            self.editor_model or _SeamEditorModel(),
            _constant_stage("audit_editing"),
            rejected_stage=self.rejected_stage,
        )

        return await run_harness(
            "A seam-test topic",
            checklist_model=_EstimatedStageModel(
                base.ChecklistModel(events),
                _constant_stage("checklist"),
                rejected_stage=self.rejected_stage,
            ),
            decision_model=decision,
            note_model=_EstimatedStageModel(
                _SeamNoteModel(events),
                _constant_stage("collection"),
                rejected_stage=self.rejected_stage,
            ),
            write_model=_EstimatedStageModel(
                base.TwoBlockAuditWriteModel(),
                _constant_stage("writing"),
                rejected_stage=self.rejected_stage,
            ),
            claim_model=claim,
            reconciliation_model=reconciliation,
            attribution_model=attribution,
            verification_model=verification,
            editor_model=editor,
            recovery_model=recovery,
            tavily_client=_SearchEmptyReadAvailable(
                source_url,
                gap_url,
                fail_gap_search=self.fail_gap_search,
                source_lead_count=self.source_lead_count,
            ),
            budget=self.loop_budget
            or LoopBudget(max_rounds=3, max_tokens=None, max_cost_usd=1),
            run_cost_budget=self.run_cost_budget
            or RunCostBudget(max_cost_usd=2.0),
            evidence_gap_budget=EvidenceGapBudget(
                max_tokens=10_000,
                max_cost_usd=1.0,
                max_search_queries=1,
                max_reads=1,
            ),
            disagreement_budget=DisagreementBudget(
                max_tokens=10_000,
                max_cost_usd=1.0,
                max_selected_claims=(
                    2 if self.select_all_disagreements else 1
                ),
                max_search_queries=1,
                max_reads=1,
            ),
            posthoc_retrieval_budget=PosthocRetrievalBudget(
                max_tokens=20_000,
                max_cost_usd=2.0,
            ),
            evidence_recovery_budget=EvidenceGapBudget(
                max_tokens=10_000,
                max_cost_usd=1.0,
                max_search_queries=1,
                max_reads=1,
            ),
            recovery_triage_settings=self.recovery_triage_settings,
            editorial_settings=self.editorial_settings,
            output_dir=self.output_root,
            run_id=run_id,
        )
