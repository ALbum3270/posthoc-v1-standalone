from __future__ import annotations

import asyncio
import json

from open_deep_research.harness.claims import (
    BlockDisposition,
    CitationRequirement,
    ClaimNormalizationStatus,
    MarkdownBlockKind,
    SourceResolution,
    build_decontextualization_prompt,
    build_extraction_prompt,
    build_selection_prompt,
    decompose_claims,
    parse_markdown_blocks,
    source_inheritance_allowed,
)


class ScriptedClaimModel:
    def __init__(self, *contents: dict[str, object]) -> None:
        self.contents = list(contents)
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> dict[str, object]:
        self.prompts.append(prompt)
        return {
            "content": json.dumps(self.contents.pop(0)),
            "token_count": 10,
            "cost_usd": 0.01,
        }


def _span(report: str, text: str) -> dict[str, object]:
    start = report.index(text)
    return {
        "text": text,
        "start_char": start,
        "end_char": start + len(text),
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
    anchor = _span(report, "It expanded the facility in 2022.")
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
                    "context_spans": [context],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "anchor_text": anchor["text"],
                    "start_char": anchor["start_char"],
                    "end_char": anchor["end_char"],
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
    assert claim.anchor_text == "It expanded the facility in 2022."
    assert report[claim.start_char : claim.end_char] == claim.anchor_text
    assert claim.context_spans[0].text == "A group opened a facility."
    assert claim.citation_requirement == CitationRequirement.EXTERNAL
    assert claim.source_resolution == SourceResolution.UNRESOLVED
    assert claim.normalization_status == ClaimNormalizationStatus.LOCATED
    assert result.total_tokens == 30
    assert result.total_cost_usd == 0.03

    assert "Do not add facts" in model.prompts[1]
    assert "otherwise make\nthe assertion stronger" in model.prompts[1]
    assert "report[start_char:end_char] == anchor_text" in model.prompts[2]


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


def test_malformed_selection_entry_does_not_discard_valid_sibling() -> None:
    report = "A device stopped.\n\nAnother paragraph."
    blocks = parse_markdown_blocks(report)
    anchor = _span(report, "A device stopped.")
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
                    "anchor_text": anchor["text"],
                    "start_char": anchor["start_char"],
                    "end_char": anchor["end_char"],
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


def test_repeated_anchor_is_normalization_failed_without_position_guess() -> None:
    report = "The system changed.\n\nThe system changed."
    blocks = parse_markdown_blocks(report)
    second_start = report.rindex("The system changed.")
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
                    "anchor_text": "The system changed.",
                    "start_char": second_start,
                    "end_char": second_start + len("The system changed."),
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    claim = result.claims[0]
    assert claim.normalization_status == (
        ClaimNormalizationStatus.NORMALIZATION_FAILED
    )
    assert claim.normalization_failure == "anchor_not_unique"
    assert claim.start_char is None
    assert claim.end_char is None
    assert claim.anchor_text == "The system changed."
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


def test_prompts_are_topic_neutral_and_expose_only_stage_owned_fields() -> None:
    report = "A device stopped."
    blocks = parse_markdown_blocks(report)

    selection = build_selection_prompt(blocks)
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
