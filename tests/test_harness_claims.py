from __future__ import annotations

import asyncio
import json

from open_deep_research.harness.claims import (
    BlockDisposition,
    CitationRequirement,
    ClaimDecompositionSettings,
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
                    "context_spans": [context["text"]],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "anchor_text": anchor["text"],
                    "start_char": 0,
                    "end_char": 1,
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
    assert claim.anchor_text_proposal == "It expanded the facility in 2022."
    assert claim.anchor_text == "It expanded the facility in 2022."
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
    assert result.anchor_copied_from_selection_count == 1
    assert result.anchor_copied_from_selection_rate == 1.0

    assert "Do not add facts" in model.prompts[1]
    assert "otherwise make\nthe assertion stronger" in model.prompts[1]
    assert "calculate start_char/end_char" in model.prompts[2]
    assert '"start_char":0' not in model.prompts[2]


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
                    "anchor_text": "First fact.",
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    by_id = {claim.claim_id: claim for claim in result.claims}
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
                    "anchor_text": anchors[index - 1],
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
                    "anchor_text": "It stopped.",
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


def test_anchor_repair_retains_model_proposal_and_uses_report_text() -> None:
    report = "The event happened, affecting others."
    blocks = parse_markdown_blocks(report)
    proposed_anchor = "The event happened."
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "disposition": "claims_selected",
                    "rationale": "one assertion",
                    "assertions": [
                        {
                            "selected_text": proposed_anchor,
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
                    "claim_text": proposed_anchor,
                    "context_spans": [],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "anchor_text": proposed_anchor,
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    claim = result.claims[0]
    assert claim.normalization_status == ClaimNormalizationStatus.LOCATED
    assert claim.anchor_text_proposal == proposed_anchor
    assert claim.anchor_text == "The event happened"
    assert report[claim.start_char : claim.end_char] == claim.anchor_text
    assert result.anchor_proposal_count == 1
    assert result.anchor_copied_from_selection_count == 1
    assert result.anchor_copied_from_selection_rate == 1.0


def test_anchor_rewrite_remains_failed_after_conservative_repair() -> None:
    report = "A person, a visible figure, led the group."
    blocks = parse_markdown_blocks(report)
    rewritten = "The person was a visible figure."
    model = ScriptedClaimModel(
        {
            "blocks": [
                {
                    "block_id": blocks[0].block_id,
                    "disposition": "claims_selected",
                    "rationale": "one assertion",
                    "assertions": [
                        {
                            "selected_text": rewritten,
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
                    "claim_text": rewritten,
                    "context_spans": [],
                }
            ]
        },
        {
            "claims": [
                {
                    "claim_id": "claim-0001",
                    "anchor_text": rewritten,
                }
            ]
        },
    )

    result = asyncio.run(decompose_claims(report, model_client=model))

    claim = result.claims[0]
    assert claim.normalization_status == (
        ClaimNormalizationStatus.NORMALIZATION_FAILED
    )
    assert claim.normalization_failure == "anchor_not_found"
    assert claim.anchor_text_proposal == rewritten
    assert claim.anchor_text == rewritten
    assert claim.start_char is None
    assert claim.end_char is None


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
    assert '"start_char":0' not in decontext
    assert '"start_char":0' not in extraction
    assert all("json" in prompt.lower() for prompt in (
        selection,
        decontext,
        extraction,
    ))
