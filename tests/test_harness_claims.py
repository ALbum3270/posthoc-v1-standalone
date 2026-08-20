from __future__ import annotations

import asyncio
import copy
import json
import re

import pytest

from open_deep_research.harness.claims import (
    BlockDisposition,
    BlockSelection,
    CitationRequirement,
    ClaimDerivationStatus,
    ClaimDecompositionResult,
    ClaimDecompositionSettings,
    ClaimNormalizationStatus,
    ClaimRepresentationVersion,
    ClaimStageUsage,
    EvidenceObligationStatus,
    MarkdownBlockKind,
    SelectedAssertion,
    SourceResolution,
    build_decontextualization_prompt,
    build_extraction_prompt,
    build_negative_selection_review_prompt,
    build_selection_prompt,
    decompose_claims,
    parse_markdown_blocks,
    source_inheritance_allowed,
)
from open_deep_research.harness.source_spans import build_source_span_registry
from open_deep_research.harness.truth_conditions import (
    ElementizationExecutionStatus,
    ElementizationSemanticStatus,
)


class ScriptedClaimModel:
    def __init__(
        self,
        *contents: dict[str, object],
        upgrade_selection_text: bool = True,
    ) -> None:
        self.contents = list(contents)
        self.prompts: list[str] = []
        self.upgrade_selection_text = upgrade_selection_text

    async def generate(self, prompt: str) -> dict[str, object]:
        self.prompts.append(prompt)
        content = copy.deepcopy(self.contents.pop(0))
        if self.upgrade_selection_text and prompt.startswith("Stage 1 of 3"):
            _upgrade_legacy_selection_text_to_pointers(content, prompt)
        return {
            "content": json.dumps(content),
            "token_count": 10,
            "cost_usd": 0.01,
        }


class FailingClaimModel:
    async def generate(self, prompt: str) -> dict[str, object]:
        raise RuntimeError("synthetic independent-review failure")


def _upgrade_legacy_selection_text_to_pointers(
    content: dict[str, object],
    prompt: str,
) -> None:
    """Keep old fixture prose readable while exercising the live interface.

    Tests written before selection pointers describe exact assertion text.  The
    production schema must reject that shape, so this *test-client-only*
    adapter turns uniquely identifiable fixture text into the pointer protocol
    emitted by a real selection model.  A rewrite or repeated text is left
    unchanged deliberately; those tests then prove the live schema rejects it
    rather than giving it an unsafe anchor.
    """

    if not isinstance(content.get("blocks"), list):
        return
    payload = json.loads(prompt.split("Markdown blocks:\n", 1)[1])
    addressable_by_block = {
        str(block["block_id"]): str(block["addressable_text"])
        for block in payload
    }
    marker = re.compile(r"<(S\d{6})>")
    for block in content["blocks"]:
        if not isinstance(block, dict):
            continue
        addressable = addressable_by_block.get(str(block.get("block_id")))
        assertions = block.get("assertions")
        if addressable is None or not isinstance(assertions, list):
            continue
        matches = list(marker.finditer(addressable))
        if not matches:
            continue
        plain_parts: list[str] = []
        positions: list[tuple[str, int, int]] = []
        cursor = 0
        for index, match in enumerate(matches):
            plain_parts.append(addressable[cursor : match.start()])
            chunk_end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(addressable)
            )
            start = sum(len(part) for part in plain_parts)
            chunk = addressable[match.end() : chunk_end]
            plain_parts.append(chunk)
            positions.append((match.group(1), start, start + len(chunk)))
            cursor = chunk_end
        plain = "".join(plain_parts)
        for assertion in assertions:
            if not isinstance(assertion, dict):
                continue
            selected_text = assertion.get("selected_text")
            if not isinstance(selected_text, str):
                continue
            if plain.count(selected_text) != 1:
                continue
            start = plain.index(selected_text)
            end = start + len(selected_text)
            selected = [
                segment
                for segment in positions
                if segment[1] < end and start < segment[2]
            ]
            if not selected:
                continue
            assertion.pop("selected_text")
            assertion["start_segment_id"] = selected[0][0]
            assertion["end_segment_id"] = selected[-1][0]


def _span(report: str, text: str) -> dict[str, object]:
    start = report.index(text)
    return {
        "text": text,
        "start_char": start,
        "end_char": start + len(text),
    }


def _pointer(
    report: str,
    text: str,
    *,
    occurrence: int = 0,
) -> dict[str, str]:
    starts: list[int] = []
    cursor = 0
    while True:
        start = report.find(text, cursor)
        if start < 0:
            break
        starts.append(start)
        cursor = start + 1
    start = starts[occurrence]
    end = start + len(text)
    registry = build_source_span_registry(report)
    selected = [
        segment
        for segment in registry.segments
        if segment.start_char < end and start < segment.end_char
    ]
    assert selected
    return {
        "start_segment_id": selected[0].segment_id,
        "end_segment_id": selected[-1].segment_id,
    }


def test_markdown_blocks_have_exact_offsets_and_structural_boundaries() -> None:
    report = """\
# First section
First paragraph line one.
First paragraph line two.

Second paragraph.

- First item
- Second item

| Name | Value |
| --- | --- |
| A | 1 |

## Next section
Last paragraph.
"""

    blocks = parse_markdown_blocks(report)

    assert [block.kind for block in blocks] == [
        MarkdownBlockKind.HEADING,
        MarkdownBlockKind.PARAGRAPH,
        MarkdownBlockKind.PARAGRAPH,
        MarkdownBlockKind.LIST_ITEM,
        MarkdownBlockKind.LIST_ITEM,
        MarkdownBlockKind.TABLE_ROW,
        MarkdownBlockKind.TABLE_ROW,
        MarkdownBlockKind.HEADING,
        MarkdownBlockKind.PARAGRAPH,
    ]
    assert all(
        report[block.start_char : block.end_char] == block.text
        for block in blocks
    )
    assert blocks[1].text == (
        "First paragraph line one.\nFirst paragraph line two."
    )
    assert blocks[1].section_path == ("First section",)
    assert blocks[-1].section_path == ("First section", "Next section")

    assert source_inheritance_allowed(
        blocks,
        source_block_id=blocks[1].block_id,
        target_block_id=blocks[2].block_id,
        resolution=SourceResolution.INHERITED_PREVIOUS_UNIT,
    )
    assert not source_inheritance_allowed(
        blocks,
        source_block_id=blocks[3].block_id,
        target_block_id=blocks[4].block_id,
        resolution=SourceResolution.INHERITED_PREVIOUS_UNIT,
    )
    assert not source_inheritance_allowed(
        blocks,
        source_block_id=blocks[6].block_id,
        target_block_id=blocks[8].block_id,
        resolution=SourceResolution.INHERITED_PREVIOUS_UNIT,
    )
    assert source_inheritance_allowed(
        blocks,
        source_block_id=blocks[4].block_id,
        target_block_id=blocks[4].block_id,
        resolution=SourceResolution.INHERITED_SAME_UNIT,
    )


def test_three_stages_keep_claim_text_and_anchor_text_separate() -> None:
    report = """\
# Overview

A group opened a facility.

It expanded the facility in 2022.
"""
    blocks = parse_markdown_blocks(report)
    context = _span(report, "A group opened a facility.")
    anchor_pointer = _pointer(report, "It expanded the facility in 2022.")
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "disposition": "no_verifiable_claims",
                    "rationale": "heading",
                    "assertions": [],
                },
                {
                    "block_id": blocks[1].block_id,
                    "disposition": "no_verifiable_claims",
                    "rationale": "used only as context",
                    "assertions": [],
                },
                {
                    "block_id": blocks[2].block_id,
                    "disposition": "claims_selected",
                    "rationale": "one assertion",
                    "assertions": [
                        {
                            "selected_text": (
                                "It expanded the facility in 2022."
                            ),
                            "citation_requirement": "external",
                        }
                    ],
                },
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": (
                        "The group expanded the facility in 2022."
                    ),
                    "context_spans": [context["text"]],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    **anchor_pointer,
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    assert len(model.prompts) == 3
    assert model.prompts[0].startswith("Stage 1 of 3")
    assert model.prompts[1].startswith("Stage 2 of 3")
    assert model.prompts[2].startswith("Stage 3 of 3")
    claim = result.claims[0]
    assert claim.claim_text == "The group expanded the facility in 2022."
    assert claim.claim_text not in report
    assert claim.representation_version is ClaimRepresentationVersion.LAYERED_V2
    assert claim.report_surface is not None
    assert claim.report_surface.text == "It expanded the facility in 2022."
    assert report[
        claim.report_surface.start_char : claim.report_surface.end_char
    ] == claim.report_surface.text
    assert claim.derivation is not None
    assert claim.derivation.status is ClaimDerivationStatus.NOT_EVALUATED
    assert claim.anchor_text_proposal is None
    assert claim.anchor_text == "It expanded the facility in 2022."
    assert claim.anchor_start_segment_id == anchor_pointer["start_segment_id"]
    assert claim.anchor_end_segment_id == anchor_pointer["end_segment_id"]
    assert claim.anchor_span_registry_id is not None
    assert claim.anchor_report_text_sha256 is not None
    assert report[claim.start_char : claim.end_char] == claim.anchor_text
    assert claim.context_spans[0].text == "A group opened a facility."
    assert claim.context_spans[0].start_char == report.index(
        "A group opened a facility."
    )
    assert claim.context_span_proposals[0].text == context["text"]
    assert claim.context_span_proposals[0].proposed_start_char is None
    assert claim.context_span_proposals[0].proposed_end_char is None
    assert claim.citation_requirement == CitationRequirement.EXTERNAL
    assert claim.source_resolution == SourceResolution.UNRESOLVED
    assert claim.normalization_status == ClaimNormalizationStatus.LOCATED
    assert result.total_tokens == 30
    assert result.total_cost_usd == 0.03
    assert result.anchor_proposal_count == 1
    assert result.anchor_copied_from_selection_count == 0
    assert result.anchor_copied_from_selection_rate == 0.0
    assert result.truth_condition_registry is not None
    assert result.truth_condition_registry.entries[0].claim_text == (
        "It expanded the facility in 2022."
    )
    assert result.truth_condition_registry.entries[0].claim_text != (
        claim.claim_text
    )
    assert result.truth_condition_registry.denominator.unresolved_claim_ids == (
        "claim-0001",
    )
    assert result.truth_condition_registry.denominator.silent_bypass_count == 0
    assert result.truth_condition_review_usage is not None
    assert result.truth_condition_review_usage.token_count == 0

    assert "Do not add facts" in model.prompts[1]
    assert "otherwise make\nthe assertion stronger" in model.prompts[1]
    assert "Do not copy anchor text" in model.prompts[2]
    assert '"start_char":0' not in model.prompts[2]


def test_truth_conditions_are_independently_reviewed_in_one_batch() -> None:
    report = "Alpha acquired Beta for $2 billion in 2024."
    block = parse_markdown_blocks(report)[0]
    pointer = _pointer(report, report)
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": block.block_id,
                    "assertions": [
                        {
                            **pointer,
                            "citation_requirement": "external",
                        }
                    ],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": report,
                    "context_spans": [],
                    "truth_conditions": [
                        "Alpha acquired Beta.",
                        "The price was $2 billion.",
                    ],
                }
            ]
        },
        {"claims": [{"claim_id": "claim-0001", **pointer}]},
    )
    review_model = ScriptedClaimModel(
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "semantic_status": "complete",
                    "elements": [
                        "Alpha acquired Beta.",
                        "The price was $2 billion.",
                        "The acquisition occurred in 2024.",
                    ],
                    "missing_conditions": [],
                    "rationale": "The proposal omitted the time condition.",
                }
            ]
        }
    )

    result = asyncio.run(
        decompose_claims(
            report,
            model_client=model,
            review_model_client=review_model,
        )
    )

    assert result.claims[0].normalization_status is ClaimNormalizationStatus.LOCATED
    assert result.truth_condition_registry is not None
    entry = result.truth_condition_registry.entries[0]
    assert entry.semantic_status is ElementizationSemanticStatus.COMPLETE
    assert entry.proposal_elements == (
        "Alpha acquired Beta.",
        "The price was $2 billion.",
    )
    assert [element.element_id for element in entry.elements] == [
        "claim-0001::tc-0001",
        "claim-0001::tc-0002",
        "claim-0001::tc-0003",
    ]
    assert result.truth_condition_registry.denominator.complete_claim_ids == (
        "claim-0001",
    )
    assert result.truth_condition_review_usage is not None
    assert result.truth_condition_review_usage.token_count == 10
    assert result.total_tokens == 40
    review_batches = [
        batch
        for batch in result.batches
        if batch.stage == "truth_condition_review"
    ]
    assert len(review_batches) == 1
    assert review_batches[0].input_ids == ("claim-0001",)
    assert review_batches[0].outcome == "completed"
    assert len(review_model.prompts) == 1
    assert "Do not assume the proposal is correct" in review_model.prompts[0]
    assert "element IDs" in model.prompts[1]


def test_truth_condition_review_failure_is_unresolved_without_losing_claim() -> None:
    report = "A material event occurred."
    block = parse_markdown_blocks(report)[0]
    pointer = _pointer(report, report)
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": block.block_id,
                    "assertions": [
                        {**pointer, "citation_requirement": "external"}
                    ],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": report,
                    "context_spans": [],
                    "truth_conditions": ["A material event occurred."],
                }
            ]
        },
        {"claims": [{"claim_id": "claim-0001", **pointer}]},
    )

    result = asyncio.run(
        decompose_claims(
            report,
            model_client=model,
            review_model_client=FailingClaimModel(),
        )
    )

    assert len(result.claims) == 1
    assert result.claims[0].normalization_status is ClaimNormalizationStatus.LOCATED
    assert result.truth_condition_registry is not None
    entry = result.truth_condition_registry.entries[0]
    assert entry.execution_status is ElementizationExecutionStatus.MODEL_ERROR
    assert entry.semantic_status is None
    assert entry.proposal_elements == ("A material event occurred.",)
    assert result.truth_condition_registry.denominator.unresolved_claim_ids == (
        "claim-0001",
    )
    assert result.truth_condition_registry.denominator.silent_bypass_count == 0
    assert result.truth_condition_review_usage is not None
    assert result.truth_condition_review_usage.token_count == 0
    review_batch = next(
        batch
        for batch in result.batches
        if batch.stage == "truth_condition_review"
    )
    assert review_batch.outcome == "failed"
    assert any(
        diagnostic.startswith("truth_condition_review_batch_error[1]")
        for diagnostic in result.diagnostics
    )


def test_malformed_truth_condition_proposal_does_not_fail_decontextualization() -> None:
    report = "A material event occurred."
    block = parse_markdown_blocks(report)[0]
    pointer = _pointer(report, report)
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": block.block_id,
                    "assertions": [
                        {**pointer, "citation_requirement": "external"}
                    ],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": report,
                    "context_spans": [],
                    "truth_conditions": {"not": "an array"},
                }
            ]
        },
        {"claims": [{"claim_id": "claim-0001", **pointer}]},
    )
    unused_review_model = ScriptedClaimModel()

    result = asyncio.run(
        decompose_claims(
            report,
            model_client=model,
            review_model_client=unused_review_model,
        )
    )

    assert result.claims[0].normalization_status is ClaimNormalizationStatus.LOCATED
    assert result.truth_condition_registry is not None
    assert result.truth_condition_registry.entries[0].execution_status is (
        ElementizationExecutionStatus.INVALID_RESPONSE
    )
    assert result.truth_condition_registry.denominator.unresolved_claim_ids == (
        "claim-0001",
    )
    assert unused_review_model.prompts == []
    assert "truth_condition_proposal_invalid: claim-0001" in result.diagnostics


def test_legacy_decomposition_payload_without_truth_fields_still_validates() -> None:
    report = "A material event occurred."
    block = parse_markdown_blocks(report)[0]
    pointer = _pointer(report, report)
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": block.block_id,
                    "assertions": [
                        {**pointer, "citation_requirement": "external"}
                    ],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": report,
                    "context_spans": [],
                }
            ]
        },
        {"claims": [{"claim_id": "claim-0001", **pointer}]},
    )
    result = asyncio.run(decompose_claims(report, model_client=model))
    legacy_payload = result.model_dump(mode="json")
    legacy_payload.pop("truth_condition_registry")
    legacy_payload.pop("truth_condition_review_usage")

    restored = ClaimDecompositionResult.model_validate(legacy_payload)

    assert restored.truth_condition_registry is None
    assert restored.truth_condition_review_usage is None
    assert restored.total_tokens == result.total_tokens == 30
    restored_payload = restored.model_dump(mode="json")
    assert "truth_condition_registry" not in restored_payload
    assert "truth_condition_review_usage" not in restored_payload


def test_each_stage_omission_has_a_claim_specific_diagnostic() -> None:
    report = "First fact. Second fact. Third fact."
    blocks = parse_markdown_blocks(report)
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "disposition": "claims_selected",
                    "rationale": "three independent assertions",
                    "assertions": [
                        {
                            "selected_text": "First fact.",
                            "citation_requirement": "external",
                        },
                        {
                            "selected_text": "Second fact.",
                            "citation_requirement": "external",
                        },
                        {
                            "selected_text": "Third fact.",
                            "citation_requirement": "external",
                        },
                    ],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": "First fact.",
                    "context_spans": [],
                },
                {
                    "claim_id": "claim-0002",
                    "claim_text": "Second fact.",
                    "context_spans": [],
                },
                {
                    "claim_id": "claim-0003",
                    "claim_text": "",
                    "context_spans": [],
                },
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    **_pointer(report, "First fact."),
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    by_id = {claim.claim_id: claim for claim in result.claims}
    assert result.truth_condition_registry is not None
    assert tuple(
        entry.claim_id for entry in result.truth_condition_registry.entries
    ) == tuple(claim.claim_id for claim in result.claims)
    assert by_id["claim-0001"].normalization_status == (
        ClaimNormalizationStatus.LOCATED
    )
    assert by_id["claim-0002"].normalization_failure == "extraction_missing"
    assert by_id["claim-0003"].normalization_failure == (
        "decontextualization_invalid"
    )
    assert "extraction_missing: claim-0002" in result.diagnostics
    assert "decontextualization_invalid: claim-0003" in result.diagnostics
    assert '"claim_id": "claim-0003"' not in model.prompts[2]


def test_selection_pointer_derives_verbatim_finance_19_text_without_rewrite() -> None:
    """A real finance-19 rewrite must become a code-owned exact span.

    The old selection shape handwrote ``2022年12月，SBF随后移送美国。``
    after splitting a single timeline sentence.  It could not be located
    without guessing a new anchor.  The pointer protocol instead retains the
    whole authoritative sentence; pre-pointer selection rejects this payload
    because it has no ``selected_text``, so this is a genuine interface
    red/green regression rather than a synthetic happy path.
    """

    report = (
        "- **2022年12月**：SBF在巴哈马被捕，随后移送美国，"
        "面临多项刑事指控，案件重心转入司法审理阶段。"
    )
    blocks = parse_markdown_blocks(report)
    selection_pointer = _pointer(report, report)
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "rationale": "one exact timeline sentence",
                    "assertions": [
                        {
                            **selection_pointer,
                            "citation_requirement": "external",
                        }
                    ],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": report,
                    "context_spans": [],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    **selection_pointer,
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    claim = result.claims[0]
    assertion = result.selections[0].assertions[0]
    assert claim.normalization_status is ClaimNormalizationStatus.LOCATED
    assert claim.selected_text == report
    assert claim.anchor_text == report
    assert claim.report_surface is not None
    assert claim.report_surface.text == report
    assert claim.report_surface.start_char == 0
    assert claim.report_surface.end_char == len(report)
    assert claim.derivation is not None
    assert claim.derivation.status is ClaimDerivationStatus.NOT_EVALUATED
    assert assertion.selected_text == report
    assert assertion.selection_start_segment_id == (
        selection_pointer["start_segment_id"]
    )
    assert assertion.selection_end_segment_id == selection_pointer[
        "end_segment_id"
    ]
    assert "selected_assertion_not_verbatim_in_block" not in result.diagnostics
    assert '"selected_text"' not in model.prompts[0]
    assert "<S000001>" in model.prompts[0]


def test_selection_pointer_outside_its_block_fails_without_textual_fallback() -> None:
    report = "First fact.\n\nSecond fact."
    blocks = parse_markdown_blocks(report)
    second_pointer = _pointer(report, "Second fact.")
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "rationale": "invalid pointer",
                    "assertions": [
                        {
                            **second_pointer,
                            "citation_requirement": "external",
                        }
                    ],
                },
                {
                    "block_id": blocks[1].block_id,
                    "rationale": "empty",
                    "assertions": [],
                },
            ]
        },
        upgrade_selection_text=False,
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    assert result.selections[0].disposition is BlockDisposition.SELECTION_FAILED
    assert result.selections[1].disposition is (
        BlockDisposition.NO_VERIFIABLE_CLAIMS
    )
    assert result.claims == ()
    assert any(
        diagnostic.startswith("selection_entry_invalid[0]: ")
        and "leaves its declared block" in diagnostic
        for diagnostic in result.diagnostics
    )


def test_textual_selection_is_rejected_even_when_it_happens_to_be_verbatim() -> None:
    """The live protocol cannot fall back to model-copied assertion text."""

    report = "A verbatim fact."
    blocks = parse_markdown_blocks(report)
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "rationale": "old response shape",
                    "assertions": [
                        {
                            "selected_text": report,
                            "citation_requirement": "external",
                        }
                    ],
                }
            ]
        },
        upgrade_selection_text=False,
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    assert result.claims == ()
    assert result.selections[0].disposition is BlockDisposition.SELECTION_FAILED
    assert any(
        diagnostic.startswith("selection_entry_invalid[0]: ")
        and "Extra inputs are not permitted" in diagnostic
        for diagnostic in result.diagnostics
    )


def test_historical_pointer_selection_without_new_offsets_remains_readable() -> None:
    """Layered-v2 adds bounds without invalidating already persisted audits."""

    assertion = SelectedAssertion(
        selected_text="A historical exact span.",
        citation_requirement=CitationRequirement.EXTERNAL,
        selection_start_segment_id="S000001",
        selection_end_segment_id="S000001",
        selection_span_registry_id="registry-old",
        selection_report_text_sha256="0" * 64,
        selection_segmentation_version="markdown-aware-source-segments-v4",
    )

    assert assertion.selection_start_char is None
    assert assertion.selection_end_char is None
    assert assertion.selection_start_segment_id == "S000001"


def test_selection_omission_is_retained_as_failed_disposition() -> None:
    report = "# Heading\n\nA paragraph."
    blocks = parse_markdown_blocks(report)
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "disposition": "no_verifiable_claims",
                    "rationale": "heading",
                    "assertions": [],
                }
            ]
        }
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    assert len(model.prompts) == 1
    assert [selection.disposition for selection in result.selections] == [
        BlockDisposition.NO_VERIFIABLE_CLAIMS,
        BlockDisposition.SELECTION_FAILED,
    ]
    assert result.selections[1].block_id == blocks[1].block_id
    assert result.claims == ()
    assert any("selection omitted this block" in item for item in result.diagnostics)


def test_live_claim_free_run_has_explicit_closed_truth_condition_denominator() -> None:
    """A real empty denominator must not masquerade as a legacy payload."""

    report = "# Overview"
    block = parse_markdown_blocks(report)[0]
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": block.block_id,
                    "disposition": "no_verifiable_claims",
                    "rationale": "heading contains no factual assertion",
                    "assertions": [],
                }
            ]
        }
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    assert result.registry_coverage.is_complete is True
    assert result.claims == ()
    assert result.truth_condition_registry is not None
    assert result.truth_condition_registry.entries == ()
    denominator = result.truth_condition_registry.denominator
    assert denominator.selected_claim_ids == ()
    assert denominator.complete_claim_ids == ()
    assert denominator.unresolved_claim_ids == ()
    assert denominator.silent_bypass_count == 0
    assert result.truth_condition_review_usage == ClaimStageUsage()
    assert len(model.prompts) == 1


def test_live_selection_rejects_historical_none_route() -> None:
    """Historical ``none`` remains readable but cannot enter a live registry.

    The old protocol let the model say both ``no_verifiable_claims`` and
    return assertions classified ``none``. Accepting that entry would let a
    selected assertion exit all evidence denominators, so the live proposal
    rejects it instead of deriving an apparently successful disposition.
    """

    report = "A broad observation appears in the report."
    blocks = parse_markdown_blocks(report)
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "disposition": "no_verifiable_claims",
                    "rationale": "no evidence-bearing factual assertion",
                    "assertions": [
                        {
                            "selected_text": report,
                            "citation_requirement": "none",
                        }
                    ],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": report,
                    "context_spans": [],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    **_pointer(report, report),
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    assert result.registry_coverage.is_complete is False
    assert result.registry_coverage.evaluated_blocks == 0
    assert result.selections[0].disposition is BlockDisposition.SELECTION_FAILED
    assert result.claims == ()
    assert any(
        item.startswith("selection_entry_invalid[0]")
        for item in result.diagnostics
    )
    assert 'no "none"' in model.prompts[0]


def test_strict_block_selection_still_rejects_contradictory_registry_data() -> None:
    with pytest.raises(ValueError, match="only claims_selected"):
        BlockSelection(
            block_id="block-0001",
            disposition=BlockDisposition.NO_VERIFIABLE_CLAIMS,
            assertions=(
                SelectedAssertion(
                    selected_text="A broad observation.",
                    citation_requirement=CitationRequirement.NONE,
                ),
            ),
        )


def test_selection_batches_preserve_order_and_expose_incomplete_coverage() -> None:
    report = "\n\n".join(f"Paragraph {index}." for index in range(1, 11))
    blocks = parse_markdown_blocks(report)

    def dispositions(batch):
        return [
            {
                "block_id": block.block_id,
                "disposition": "no_verifiable_claims",
                "rationale": "evaluated",
                "assertions": [],
            }
            for block in batch
        ]

    model = ScriptedClaimModel(
        {"blocks": dispositions(blocks[:4])},
        {"blocks": dispositions(blocks[4:7])},
        {"blocks": dispositions(blocks[8:])},
    )

    result = asyncio.run(
        decompose_claims(
            report,
            model_client=model,
            settings=ClaimDecompositionSettings(batch_size=4),
        )
    )

    assert [selection.block_id for selection in result.selections] == [
        block.block_id for block in blocks
    ]
    assert result.registry_coverage.model_dump() == {
        "evaluated_blocks": 9,
        "total_blocks": 10,
        "unassessed_blocks": 1,
        "unassessed_block_ids": (blocks[7].block_id,),
        "is_complete": False,
    }
    selection_batches = [
        batch for batch in result.batches if batch.stage == "selection"
    ]
    assert [batch.batch_number for batch in selection_batches] == [1, 2, 3]
    assert selection_batches[1].outcome == "partial"
    assert selection_batches[1].input_ids == tuple(
        block.block_id for block in blocks[4:8]
    )
    assert selection_batches[1].failed_input_ids == (blocks[7].block_id,)
    assert blocks[0].block_id in model.prompts[0]
    assert blocks[4].block_id not in model.prompts[0]
    assert blocks[4].block_id in model.prompts[1]
    assert blocks[8].block_id in model.prompts[2]
    assert result.selection_usage.token_count == 30


def test_selection_batch_call_failure_names_every_affected_block() -> None:
    report = "\n\n".join(f"Paragraph {index}." for index in range(1, 6))
    blocks = parse_markdown_blocks(report)
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": block.block_id,
                    "disposition": "no_verifiable_claims",
                    "rationale": "evaluated",
                    "assertions": [],
                }
                for block in blocks[:4]
            ]
        }
    )

    result = asyncio.run(
        decompose_claims(
            report,
            model_client=model,
            settings=ClaimDecompositionSettings(batch_size=4),
        )
    )

    failed_batch = result.batches[1]
    assert failed_batch.stage == "selection"
    assert failed_batch.batch_number == 2
    assert failed_batch.input_ids == (blocks[4].block_id,)
    assert failed_batch.failed_input_ids == (blocks[4].block_id,)
    assert failed_batch.outcome == "failed"
    assert failed_batch.error is not None
    assert any(
        "selection_batch_error[2]" in diagnostic
        and blocks[4].block_id in diagnostic
        for diagnostic in result.diagnostics
    )
    assert result.registry_coverage.unassessed_block_ids == (
        blocks[4].block_id,
    )


def test_decontextualization_and_extraction_batch_every_selected_claim() -> None:
    anchors = [f"Fact number {index}." for index in range(1, 6)]
    report = " ".join(anchors)
    blocks = parse_markdown_blocks(report)
    assertions = [
        {
            "selected_text": anchor,
            "citation_requirement": "external",
        }
        for anchor in anchors
    ]
    decontext_batches = [
        {
            "claims": [
                {
                    "claim_id": f"claim-{index:04d}",
                    "claim_text": anchors[index - 1],
                    "context_spans": [],
                }
                for index in indexes
            ]
        }
        for indexes in ((1, 2), (3, 4), (5,))
    ]
    extraction_batches = [
        {
            "claims": [
                {
                    "claim_id": f"claim-{index:04d}",
                    **_pointer(report, anchors[index - 1]),
                }
                for index in indexes
            ]
        }
        for indexes in ((1, 2), (3, 4), (5,))
    ]
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "disposition": "claims_selected",
                    "rationale": "five assertions",
                    "assertions": assertions,
                }
            ]
        },
        *decontext_batches,
        *extraction_batches,
    )

    result = asyncio.run(
        decompose_claims(
            report,
            model_client=model,
            settings=ClaimDecompositionSettings(batch_size=2),
        )
    )

    assert len(result.claims) == 5
    assert all(
        claim.normalization_status == ClaimNormalizationStatus.LOCATED
        for claim in result.claims
    )
    assert [batch.input_ids for batch in result.batches] == [
        (blocks[0].block_id,),
        ("claim-0001", "claim-0002"),
        ("claim-0003", "claim-0004"),
        ("claim-0005",),
        ("claim-0001", "claim-0002"),
        ("claim-0003", "claim-0004"),
        ("claim-0005",),
    ]
    assert all(batch.outcome == "completed" for batch in result.batches)
    assert result.registry_coverage.is_complete is True
    assert result.total_tokens == 70


def test_malformed_selection_entry_does_not_discard_valid_sibling() -> None:
    report = "A device stopped.\n\nAnother paragraph."
    blocks = parse_markdown_blocks(report)
    anchor_pointer = _pointer(report, "A device stopped.")
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "disposition": "claims_selected",
                    "rationale": "selected",
                    "assertions": [
                        {
                            "selected_text": "A device stopped.",
                            "citation_requirement": "external",
                        }
                    ],
                },
                {
                    "block_id": blocks[1].block_id,
                    "disposition": "not-a-disposition",
                    "assertions": [],
                },
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": "A device stopped.",
                    "context_spans": [],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    **anchor_pointer,
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    assert len(result.claims) == 1
    assert result.claims[0].normalization_status == (
        ClaimNormalizationStatus.LOCATED
    )
    assert result.selections[0].disposition == (
        BlockDisposition.CLAIMS_SELECTED
    )
    assert result.selections[1].disposition == BlockDisposition.SELECTION_FAILED
    assert any(
        "selection_entry_invalid[1]" in item for item in result.diagnostics
    )


def test_segment_pointer_disambiguates_repeated_report_text() -> None:
    report = "The system changed.\n\nThe system changed."
    blocks = parse_markdown_blocks(report)
    second_start = report.rindex("The system changed.")
    second_pointer = _pointer(
        report,
        "The system changed.",
        occurrence=1,
    )
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "disposition": "no_verifiable_claims",
                    "rationale": "not selected",
                    "assertions": [],
                },
                {
                    "block_id": blocks[1].block_id,
                    "disposition": "claims_selected",
                    "rationale": "selected",
                    "assertions": [
                        {
                            "selected_text": "The system changed.",
                            "citation_requirement": "external",
                        }
                    ],
                },
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": "The system changed.",
                    "context_spans": [],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    **second_pointer,
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    claim = result.claims[0]
    assert claim.normalization_status == ClaimNormalizationStatus.LOCATED
    assert claim.normalization_failure is None
    assert claim.start_char == second_start
    assert claim.end_char == second_start + len("The system changed.")
    assert claim.anchor_text == "The system changed."
    assert claim.anchor_start_segment_id == second_pointer["start_segment_id"]
    assert claim.source_resolution == SourceResolution.UNRESOLVED


def test_invalid_context_span_cannot_become_a_located_claim() -> None:
    report = "A group acted.\n\nIt stopped."
    blocks = parse_markdown_blocks(report)
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "disposition": "no_verifiable_claims",
                    "rationale": "context",
                    "assertions": [],
                },
                {
                    "block_id": blocks[1].block_id,
                    "disposition": "claims_selected",
                    "rationale": "selected",
                    "assertions": [
                        {
                            "selected_text": "It stopped.",
                            "citation_requirement": "external",
                        }
                    ],
                },
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": "The group stopped.",
                    "context_spans": [
                        {
                            "text": "Different text",
                            "start_char": 0,
                            "end_char": len("A group acted."),
                        }
                    ],
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    assert len(model.prompts) == 2
    claim = result.claims[0]
    assert claim.claim_text == "The group stopped."
    assert claim.normalization_status == (
        ClaimNormalizationStatus.NORMALIZATION_FAILED
    )
    assert claim.normalization_failure == "context_span_not_verbatim"
    assert claim.context_spans == ()
    assert len(claim.context_span_proposals) == 1
    assert claim.context_span_proposals[0].text == "Different text"
    assert claim.context_span_proposals[0].proposed_start_char == 0
    assert claim.context_span_proposals[0].proposed_end_char == len(
        "A group acted."
    )


def test_context_span_uses_conservative_repair_within_one_markdown_unit() -> None:
    report = "- **Phase 1:** A group acted. It stopped."
    blocks = parse_markdown_blocks(report)
    proposed_context = "Phase 1: A group acted."
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "disposition": "claims_selected",
                    "rationale": "one assertion",
                    "assertions": [
                        {
                            "selected_text": "It stopped.",
                            "citation_requirement": "external",
                        }
                    ],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": "The group stopped.",
                    "context_spans": [proposed_context],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    **_pointer(report, "It stopped."),
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    claim = result.claims[0]
    assert claim.normalization_status == ClaimNormalizationStatus.LOCATED
    assert claim.context_span_proposals[0].text == proposed_context
    assert claim.context_spans[0].text != proposed_context
    assert report[
        claim.context_spans[0].start_char : claim.context_spans[0].end_char
    ] == claim.context_spans[0].text
    assert "**" in claim.context_spans[0].text


def test_context_span_is_scoped_to_claim_block_before_global_uniqueness() -> None:
    """Regression for finance-17's repeated names in separate report blocks."""

    report = (
        "Sam Bankman-Fried founded the company.\n\n"
        "- **Sam Bankman-Fried**：创始人。他后来辞去CEO职务。"
    )
    blocks = parse_markdown_blocks(report)
    target = blocks[1]
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "rationale": "context only",
                    "assertions": [],
                },
                {
                    "block_id": target.block_id,
                    "rationale": "one contextual assertion",
                    "assertions": [
                        {
                            "selected_text": "他后来辞去CEO职务。",
                            "citation_requirement": "external",
                        }
                    ],
                },
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": "Sam Bankman-Fried后来辞去CEO职务。",
                    "context_spans": ["Sam Bankman-Fried"],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    **_pointer(report, target.text),
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    claim = result.claims[0]
    assert report.count("Sam Bankman-Fried") == 2
    assert target.text.count("Sam Bankman-Fried") == 1
    assert claim.normalization_status is ClaimNormalizationStatus.LOCATED
    assert claim.context_spans[0].start_char == target.start_char + 4
    assert report[
        claim.context_spans[0].start_char : claim.context_spans[0].end_char
    ] == "Sam Bankman-Fried"


def test_context_span_repeated_inside_claim_block_remains_ambiguous() -> None:
    report = "Sam acted. Sam stopped."
    blocks = parse_markdown_blocks(report)
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                        "rationale": "one contextual assertion",
                        "assertions": [
                            {
                                **_pointer(report, "Sam stopped."),
                                "citation_requirement": "external",
                            }
                    ],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": "Sam stopped.",
                    "context_spans": ["Sam"],
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    claim = result.claims[0]
    assert claim.normalization_status is (
        ClaimNormalizationStatus.NORMALIZATION_FAILED
    )
    assert claim.normalization_failure == "context_span_not_unique"
    assert len(model.prompts) == 2


def test_markdown_date_context_repairs_inside_its_own_list_item() -> None:
    """Regression for finance-17's repeated bold timeline dates."""

    report = (
        "- **11月11日**：FTX申请破产保护。\n"
        "- **11月11日**：John J. Ray III接任CEO。"
    )
    blocks = parse_markdown_blocks(report)
    target = blocks[1]
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "rationale": "context only",
                    "assertions": [],
                },
                {
                    "block_id": target.block_id,
                        "rationale": "one timeline assertion",
                        "assertions": [
                            {
                                **_pointer(report, target.text),
                                "citation_requirement": "external",
                            }
                    ],
                },
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": "John J. Ray III于11月11日接任CEO。",
                    "context_spans": ["11月11日："],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    **_pointer(report, target.text),
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    claim = result.claims[0]
    assert claim.normalization_status is ClaimNormalizationStatus.LOCATED
    assert claim.context_span_proposals[0].text == "11月11日："
    assert claim.context_spans[0].text == "11月11日"
    assert target.start_char <= claim.context_spans[0].start_char
    assert claim.context_spans[0].end_char <= target.end_char


def test_context_repair_cannot_cross_markdown_units() -> None:
    report = "# Context\n\nA group acted.\n\nIt stopped."
    blocks = parse_markdown_blocks(report)
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": block.block_id,
                    "disposition": (
                        "claims_selected" if block is blocks[-1]
                        else "no_verifiable_claims"
                    ),
                    "rationale": "evaluated",
                    "assertions": (
                        [
                            {
                                "selected_text": "It stopped.",
                                "citation_requirement": "external",
                            }
                        ]
                        if block is blocks[-1]
                        else []
                    ),
                }
                for block in blocks
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": "The group stopped.",
                    "context_spans": ["Context: A group acted."],
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    assert len(model.prompts) == 2
    claim = result.claims[0]
    assert claim.normalization_status == (
        ClaimNormalizationStatus.NORMALIZATION_FAILED
    )
    assert claim.normalization_failure == (
        "context_span_repair_crosses_markdown_unit"
    )
    assert claim.context_span_proposals[0].text == (
        "Context: A group acted."
    )


def test_anchor_pointer_narrows_to_exact_selected_assertion() -> None:
    report = "The event happened, affecting others."
    blocks = parse_markdown_blocks(report)
    selected_assertion = "The event happened, affecting others."
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "disposition": "claims_selected",
                    "rationale": "one assertion",
                    "assertions": [
                        {
                            "selected_text": selected_assertion,
                            "citation_requirement": "external",
                        }
                    ],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": selected_assertion,
                    "context_spans": [],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    **_pointer(report, report),
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    claim = result.claims[0]
    assert claim.normalization_status == ClaimNormalizationStatus.LOCATED
    assert claim.anchor_text_proposal is None
    assert claim.anchor_text == selected_assertion
    assert report[claim.start_char : claim.end_char] == claim.anchor_text
    assert result.anchor_proposal_count == 1
    assert result.anchor_copied_from_selection_count == 0
    assert result.anchor_copied_from_selection_rate == 0.0


def test_invalid_anchor_pointer_remains_failed_without_clamping() -> None:
    report = "A person, a visible figure, led the group."
    blocks = parse_markdown_blocks(report)
    selected_assertion = report
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "disposition": "claims_selected",
                    "rationale": "one assertion",
                    "assertions": [
                        {
                            "selected_text": selected_assertion,
                            "citation_requirement": "external",
                        }
                    ],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": selected_assertion,
                    "context_spans": [],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "start_segment_id": "S999998",
                    "end_segment_id": "S999999",
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    claim = result.claims[0]
    assert claim.normalization_status == (
        ClaimNormalizationStatus.NORMALIZATION_FAILED
    )
    assert claim.normalization_failure == "anchor_pointer_invalid"
    assert claim.anchor_text_proposal is None
    assert claim.anchor_text is None
    assert claim.anchor_start_segment_id == "S999998"
    assert claim.anchor_end_segment_id == "S999999"
    assert claim.start_char is None
    assert claim.end_char is None


def test_finance07_style_rewritten_selection_is_not_given_a_broader_anchor() -> None:
    """A semantic selection rewrite must not inherit a nearby report span."""

    report = (
        "- **Milestone**  \n"
        "  On day one, an initial plan was announced. "
        "On day two, the plan was reversed after a review."
    )
    blocks = parse_markdown_blocks(report)
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "rationale": "one assertion",
                    "assertions": [
                        {
                            "selected_text": "On day two, the plan changed.",
                            "citation_requirement": "external",
                        }
                    ],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": "The plan changed on day two.",
                    "context_spans": [],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    **_pointer(
                        report,
                        "On day two, the plan was reversed after a review.",
                    ),
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    assert result.claims == ()
    assert result.selections[0].disposition is BlockDisposition.SELECTION_FAILED
    assert any(
        diagnostic.startswith("selection_entry_invalid[0]: ")
        for diagnostic in result.diagnostics
    )


def test_repeated_assertions_are_selected_by_explicit_segment_not_text_search():
    """A pointer, unlike copied text, identifies the intended repetition."""

    selected_assertion = "The transaction failed."
    report = f"{selected_assertion} {selected_assertion}"
    blocks = parse_markdown_blocks(report)
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                        "rationale": "one repeated assertion",
                        "assertions": [
                            {
                                **_pointer(
                                    report,
                                    selected_assertion,
                                    occurrence=1,
                                ),
                                "citation_requirement": "external",
                            }
                    ],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": selected_assertion,
                    "context_spans": [],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    **_pointer(report, selected_assertion, occurrence=1),
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    claim = result.claims[0]
    assert claim.normalization_status is ClaimNormalizationStatus.LOCATED
    assert claim.start_char == report.rindex(selected_assertion)
    assert claim.anchor_text == selected_assertion


def test_anchor_pointer_outside_selected_block_is_still_rejected() -> None:
    report = "First statement.\n\nSecond statement."
    blocks = parse_markdown_blocks(report)
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "rationale": "selected",
                    "assertions": [
                        {
                            "selected_text": "First statement.",
                            "citation_requirement": "external",
                        }
                    ],
                },
                {
                    "block_id": blocks[1].block_id,
                    "rationale": "empty",
                    "assertions": [],
                },
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": "First statement.",
                    "context_spans": [],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    **_pointer(report, "Second statement."),
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    claim = result.claims[0]
    assert claim.normalization_status is (
        ClaimNormalizationStatus.NORMALIZATION_FAILED
    )
    assert claim.normalization_failure == "anchor_does_not_cover_selected_assertion"


def test_finance18_style_broad_pointer_is_narrowed_to_atomic_claim_anchor() -> None:
    """A CJK list-item pointer cannot attach one claim's label to two facts."""

    selected_assertion = "SBF 的管理风格被员工形容为鲁莽且不负责任。"
    report = (
        "- **内部管理失控和风险失察**："
        f"{selected_assertion}"
        "公司缺乏有效的内部控制和风险管理体系。"
    )
    blocks = parse_markdown_blocks(report)
    registry = build_source_span_registry(report)
    assert len(registry.segments) == 2
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "rationale": "one atomic assertion",
                    "assertions": [
                        {
                            "selected_text": selected_assertion,
                            "citation_requirement": "external",
                        }
                    ],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": selected_assertion,
                    "context_spans": [],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    # Simulate finance-18: extraction points at the entire
                    # CJK list item rather than the one selected assertion.
                    "start_segment_id": registry.segments[0].segment_id,
                    "end_segment_id": registry.segments[1].segment_id,
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    claim = result.claims[0]
    assert claim.normalization_status is ClaimNormalizationStatus.LOCATED
    expected_exact_range = report[: report.index("公司缺乏有效")]
    assert claim.anchor_text == expected_exact_range
    assert claim.start_char == 0
    assert claim.end_char == claim.start_char + len(expected_exact_range)
    assert claim.anchor_end_segment_id == registry.segments[1].segment_id
    assert report[claim.start_char : claim.end_char] == expected_exact_range
    assert "公司缺乏有效" not in claim.anchor_text


def test_selection_rewrite_is_visible_failure_not_a_guess_at_offsets() -> None:
    report = "The plan was reversed after a review."
    blocks = parse_markdown_blocks(report)
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "rationale": "one assertion",
                    "assertions": [
                        {
                            "selected_text": "The plan changed.",
                            "citation_requirement": "external",
                        }
                    ],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": "The plan changed.",
                    "context_spans": [],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    **_pointer(report, report),
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    assert result.claims == ()
    assert result.selections[0].disposition is BlockDisposition.SELECTION_FAILED
    assert any(
        diagnostic.startswith("selection_entry_invalid[0]: ")
        for diagnostic in result.diagnostics
    )


def test_prompts_are_topic_neutral_and_expose_only_stage_owned_fields() -> None:
    report = "A device stopped."
    blocks = parse_markdown_blocks(report)

    selection = build_selection_prompt(report, blocks)
    decontext = build_decontextualization_prompt(
        report,
        [
            {
                "claim_id": "claim-0001",
                "selected_text": report,
                "citation_requirement": "external",
            }
        ],
    )
    extraction = build_extraction_prompt(
        report,
        [{"claim_id": "claim-0001", "claim_text": report}],
    )

    assert "source_resolution" not in selection
    assert "source_resolution" not in decontext
    assert "source_resolution" not in extraction
    assert '"start_char":0' not in decontext
    assert '"start_char":0' not in extraction
    assert '"anchor_text"' not in extraction
    assert '"start_segment_id"' in extraction
    assert '"selected_text"' not in selection
    assert '"addressable_text"' in selection
    assert "<S000001>" in selection
    assert "atomic-v1" not in selection
    assert "independently truth-valued" not in selection
    assert "smallest sufficient verification units" in selection
    assert all(
        relation in selection
        for relation in ("attribution", "modality", "cause", "time sequence")
    )
    assert "reporting marker" in selection
    assert all("json" in prompt.lower() for prompt in (
        selection,
        decontext,
        extraction,
    ))


def test_independent_negative_review_can_restore_a_missed_claim() -> None:
    report = "# Report\n\nA material fact was omitted from selection."
    blocks = parse_markdown_blocks(report)
    paragraph = blocks[1]
    claim_model = ScriptedClaimModel(
        {
            "blocks": [
                {"block_id": block.block_id, "assertions": []}
                for block in blocks
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": paragraph.text,
                    "context_spans": [],
                }
            ]
        },
        {
            "claims": [
                {"claim_id": "claim-0001", **_pointer(report, paragraph.text)}
            ]
        },
    )
    review_model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "outcome": "confirmed_empty",
                    "rationale": "structural heading",
                    "assertions": [],
                },
                {
                    "block_id": paragraph.block_id,
                    "outcome": "claims_selected",
                    "rationale": "the first selector missed a factual claim",
                    "assertions": [
                        {
                            **_pointer(report, paragraph.text),
                            "citation_requirement": "external",
                        }
                    ],
                },
            ]
        },
    )

    result = asyncio.run(
        decompose_claims(
            report,
            model_client=claim_model,
            review_model_client=review_model,
            settings=ClaimDecompositionSettings(
                require_independent_evidence_review=True
            ),
        )
    )

    assert result.registry_coverage.is_complete is True
    assert len(result.claims) == 1
    assert result.block_denominator_audit is not None
    assert result.block_denominator_audit.silent_bypass_count == 0
    assert result.block_denominator_audit.selected_block_ids == (
        paragraph.block_id,
    )
    assert result.claim_obligation_audit is not None
    assert result.claim_obligation_audit.externally_routed_claim_ids == (
        "claim-0001",
    )


def test_negative_review_failure_is_explicit_not_confirmed_empty() -> None:
    report = "A block whose empty selection cannot be independently reviewed."
    blocks = parse_markdown_blocks(report)
    claim_model = ScriptedClaimModel(
        {"blocks": [{"block_id": blocks[0].block_id, "assertions": []}]}
    )

    result = asyncio.run(
        decompose_claims(
            report,
            model_client=claim_model,
            review_model_client=FailingClaimModel(),
            settings=ClaimDecompositionSettings(
                require_independent_evidence_review=True
            ),
        )
    )

    assert result.registry_coverage.is_complete is False
    assert result.selections[0].disposition is (
        BlockDisposition.SELECTION_UNRESOLVED
    )
    assert result.block_denominator_audit is not None
    assert result.block_denominator_audit.unresolved_block_ids == (
        blocks[0].block_id,
    )
    assert result.block_denominator_audit.silent_bypass_count == 0


def test_internal_route_is_independently_promoted_to_external() -> None:
    report = "# Report\n\nA company transferred customer funds in 2022."
    blocks = parse_markdown_blocks(report)
    paragraph = blocks[1]
    claim_model = ScriptedClaimModel(
        {
            "blocks": [
                {"block_id": blocks[0].block_id, "assertions": []},
                {
                    "block_id": paragraph.block_id,
                    "assertions": [
                        {
                            **_pointer(report, paragraph.text),
                            "citation_requirement": "internal",
                        }
                    ],
                },
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": paragraph.text,
                    "context_spans": [],
                }
            ]
        },
        {
            "claims": [
                {"claim_id": "claim-0001", **_pointer(report, paragraph.text)}
            ]
        },
    )
    review_model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "outcome": "confirmed_empty",
                    "rationale": "structural heading",
                    "assertions": [],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "outcome": "external_required",
                    "rationale": "truth depends on records outside the report",
                    "evidence_spans": [],
                }
            ]
        },
    )

    result = asyncio.run(
        decompose_claims(
            report,
            model_client=claim_model,
            review_model_client=review_model,
            settings=ClaimDecompositionSettings(
                require_independent_evidence_review=True
            ),
        )
    )

    claim = result.claims[0]
    assert claim.proposed_citation_requirement is CitationRequirement.INTERNAL
    assert claim.citation_requirement is CitationRequirement.EXTERNAL
    assert claim.evidence_obligation is not None
    assert claim.evidence_obligation.status is (
        EvidenceObligationStatus.EXTERNAL_REQUIRED
    )
    assert result.claim_obligation_audit is not None
    assert result.claim_obligation_audit.silent_bypass_count == 0
    assert "claim's own report_surface is not independent" in (
        review_model.prompts[1]
    )


def test_internal_claim_cannot_use_its_own_surface_as_evidence() -> None:
    report = "# Report\n\nThe report contains four findings."
    blocks = parse_markdown_blocks(report)
    paragraph = blocks[1]
    claim_model = ScriptedClaimModel(
        {
            "blocks": [
                {"block_id": blocks[0].block_id, "assertions": []},
                {
                    "block_id": paragraph.block_id,
                    "assertions": [
                        {
                            **_pointer(report, paragraph.text),
                            "citation_requirement": "internal",
                        }
                    ],
                },
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": paragraph.text,
                    "context_spans": [],
                }
            ]
        },
        {
            "claims": [
                {"claim_id": "claim-0001", **_pointer(report, paragraph.text)}
            ]
        },
    )
    review_model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "outcome": "confirmed_empty",
                    "assertions": [],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "outcome": "internal_supported",
                    "rationale": "incorrectly cites itself",
                    "evidence_spans": [_pointer(report, paragraph.text)],
                }
            ]
        },
    )

    result = asyncio.run(
        decompose_claims(
            report,
            model_client=claim_model,
            review_model_client=review_model,
            settings=ClaimDecompositionSettings(
                require_independent_evidence_review=True
            ),
        )
    )

    obligation = result.claims[0].evidence_obligation
    assert obligation is not None
    assert obligation.status is EvidenceObligationStatus.UNRESOLVED
    assert obligation.failure_reason == "internal_evidence_self_reference"
    assert result.claim_obligation_audit is not None
    assert result.claim_obligation_audit.unresolved_claim_ids == (
        "claim-0001",
    )
    assert result.claim_obligation_audit.silent_bypass_count == 0


def test_internal_not_supported_is_a_completed_negative_conclusion() -> None:
    report = "The report artifact does not establish this conclusion."
    block = parse_markdown_blocks(report)[0]
    claim_model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": block.block_id,
                    "assertions": [
                        {
                            **_pointer(report, block.text),
                            "citation_requirement": "internal",
                        }
                    ],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "claim_text": block.text,
                    "context_spans": [],
                }
            ]
        },
        {
            "claims": [
                {"claim_id": "claim-0001", **_pointer(report, block.text)}
            ]
        },
    )
    review_model = ScriptedClaimModel(
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "outcome": "internal_not_supported",
                    "rationale": "No independent report-artifact span exists.",
                    "evidence_spans": [],
                }
            ]
        }
    )

    result = asyncio.run(
        decompose_claims(
            report,
            model_client=claim_model,
            review_model_client=review_model,
            settings=ClaimDecompositionSettings(
                require_independent_evidence_review=True
            ),
        )
    )

    obligation = result.claims[0].evidence_obligation
    assert obligation is not None
    assert obligation.status is EvidenceObligationStatus.INTERNAL_NOT_SUPPORTED
    assert result.claim_obligation_audit is not None
    assert result.claim_obligation_audit.internally_unsupported_claim_ids == (
        "claim-0001",
    )
    assert result.claim_obligation_audit.unresolved_claim_ids == ()
    obligation_batch = next(
        batch for batch in result.batches if batch.stage == "evidence_obligation"
    )
    assert obligation_batch.outcome == "completed"
    assert obligation_batch.output_ids == ("claim-0001",)
    assert obligation_batch.failed_input_ids == ()


def test_review_prompts_keep_empty_and_internal_decisions_explicit() -> None:
    report = "# Report\n\nA claim."
    blocks = parse_markdown_blocks(report)
    negative = build_negative_selection_review_prompt(report, blocks[:1])
    assert "confirmed_empty|claims_selected|uncertain" in negative
    assert 'a "none" route' in negative
