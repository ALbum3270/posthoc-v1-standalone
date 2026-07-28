from __future__ import annotations

import asyncio
import json

from open_deep_research.harness.attribution import (
    AttributionSettings,
    AttributionStatus,
    AttributionStopReason,
    attribute_claims,
)
from open_deep_research.harness.claims import (
    AtomicClaim,
    CitationRequirement,
    ClaimNormalizationStatus,
    MarkdownBlock,
    MarkdownBlockKind,
    SourceResolution,
)
from open_deep_research.harness.ledger import ResearchLedger
from open_deep_research.harness.notes import create_note


class ScriptedAttributionModel:
    def __init__(self, *contents: dict[str, object]) -> None:
        self.contents = list(contents)
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> dict[str, object]:
        self.prompts.append(prompt)
        return {
            "content": json.dumps(self.contents.pop(0)),
            "token_count": 7,
            "cost_usd": 0.02,
        }


def _block(
    block_id: str,
    ordinal: int,
    *,
    section: tuple[str, ...] = ("Section",),
    kind: MarkdownBlockKind = MarkdownBlockKind.PARAGRAPH,
) -> MarkdownBlock:
    start = ordinal * 100
    return MarkdownBlock(
        block_id=block_id,
        ordinal=ordinal,
        kind=kind,
        text=f"Block {ordinal}",
        start_char=start,
        end_char=start + len(f"Block {ordinal}"),
        section_path=section,
    )


def _claim(
    claim_id: str,
    block_id: str,
    text: str,
    *,
    citation_requirement: CitationRequirement = CitationRequirement.EXTERNAL,
) -> AtomicClaim:
    return AtomicClaim(
        claim_id=claim_id,
        block_id=block_id,
        selected_text=text,
        claim_text=text,
        anchor_text=text,
        start_char=0,
        end_char=len(text),
        citation_requirement=citation_requirement,
        source_resolution=SourceResolution.UNRESOLVED,
        normalization_status=ClaimNormalizationStatus.LOCATED,
    )


def _ledger_note(
    ledger: ResearchLedger,
    *,
    item_id: str,
    finding: str,
    quote: str,
    source_text: str,
    url: str,
):
    return ledger.add_note(
        create_note(
            item_id=item_id,
            finding=finding,
            quote=quote,
            url=url,
            source_text=source_text,
        )
    )


def _by_claim(result):
    return {
        attribution.claim.claim_id: attribution
        for attribution in result.attributions
    }


def test_all_claims_and_compact_registry_are_visible_without_full_notes() -> None:
    blocks = (_block("block-1", 0),)
    claims = (
        _claim("claim-1", "block-1", "The first independent claim."),
        _claim("claim-2", "block-1", "The second independent claim."),
    )
    ledger = ResearchLedger()
    candidate = _ledger_note(
        ledger,
        item_id="different-checklist-item",
        finding="A useful finding from another checklist item.",
        quote="MODEL WORDING NOT PRESENT IN THE SOURCE",
        source_text="The source uses different words.",
        url="https://one.example/page",
    )
    other = _ledger_note(
        ledger,
        item_id="another-item",
        finding="A second registry finding.",
        quote="Exact source phrase.",
        source_text="Exact source phrase.",
        url="https://two.example/page",
    )
    model = ScriptedAttributionModel(
        {
            "action": "attribute",
            "claims": [
                {
                    "claim_id": "claim-1",
                    "candidates": [
                        {
                            "note_id": candidate.note_id,
                            "source_id": candidate.source_id,
                            "inherited_from_claim_id": None,
                        }
                    ],
                },
                {"claim_id": "claim-2", "candidates": []},
            ],
        }
    )

    result = asyncio.run(
        attribute_claims(
            claims,
            blocks=blocks,
            notes=ledger.notes,
            model_client=model,
        )
    )

    prompt = model.prompts[0]
    assert "The first independent claim." in prompt
    assert "The second independent claim." in prompt
    assert candidate.note_id in prompt
    assert candidate.source_id in prompt
    assert candidate.item_id in prompt
    assert candidate.finding in prompt
    assert candidate.publisher in prompt
    assert candidate.location_status.value in prompt
    assert other.note_id in prompt
    assert other.finding in prompt
    assert "MODEL WORDING NOT PRESENT IN THE SOURCE" not in prompt
    assert "Exact source phrase." not in prompt
    assert "https://one.example/page" not in prompt
    assert '"resolution"' not in prompt
    assert "repetition is direct matching, not\ninheritance" in prompt

    by_claim = _by_claim(result)
    first = by_claim["claim-1"]
    assert first.status == AttributionStatus.CANDIDATE_SOURCES
    assert first.claim.source_resolution == SourceResolution.DIRECT
    assert first.candidates[0].item_id == "different-checklist-item"
    assert first.candidates[0].location_status.value == "unlocatable"
    assert not hasattr(first.candidates[0], "verdict")
    assert by_claim["claim-2"].status == (
        AttributionStatus.NO_CANDIDATE_SOURCE
    )
    assert by_claim["claim-2"].errors == ()
    assert by_claim["claim-2"].claim.source_resolution == (
        SourceResolution.UNRESOLVED
    )
    assert result.stop_reason == AttributionStopReason.COMPLETED


def test_model_requests_full_note_page_then_uses_previous_unit_inheritance() -> None:
    blocks = (
        _block("block-1", 0),
        _block("block-2", 1),
    )
    claims = (
        _claim("claim-1", "block-1", "A self-contained first claim."),
        _claim("claim-2", "block-2", "A self-contained second claim."),
    )
    ledger = ResearchLedger()
    note = _ledger_note(
        ledger,
        item_id="item-x",
        finding="The compact finding.",
        quote="Exact full-note quote.",
        source_text="Exact full-note quote.",
        url="https://source.example/page",
    )
    model = ScriptedAttributionModel(
        {"action": "inspect_notes", "cursor": 0},
        {
            "action": "attribute",
            "claims": [
                {
                    "claim_id": "claim-1",
                    "candidates": [
                        {
                            "note_id": note.note_id,
                            "source_id": note.source_id,
                            "inherited_from_claim_id": None,
                        }
                    ],
                },
                {
                    "claim_id": "claim-2",
                    "candidates": [
                        {
                            "note_id": note.note_id,
                            "source_id": note.source_id,
                            "inherited_from_claim_id": "claim-1",
                        }
                    ],
                },
            ],
        },
    )

    result = asyncio.run(
        attribute_claims(
            claims,
            blocks=blocks,
            notes=ledger.notes,
            model_client=model,
            settings=AttributionSettings(note_page_size=1),
        )
    )

    assert len(model.prompts) == 2
    assert "Exact full-note quote." not in model.prompts[0]
    assert "https://source.example/page" not in model.prompts[0]
    assert "Exact full-note quote." in model.prompts[1]
    assert "https://source.example/page" in model.prompts[1]
    assert result.inspected_pages[0].note_ids == (note.note_id,)
    assert result.inspected_pages[0].cache_hit is False
    second = _by_claim(result)["claim-2"]
    assert second.status == AttributionStatus.CANDIDATE_SOURCES
    assert second.claim.source_resolution == (
        SourceResolution.INHERITED_PREVIOUS_UNIT
    )
    assert second.candidates[0].inherited_from_claim_id == "claim-1"
    assert result.total_tokens == 14
    assert result.total_cost_usd == 0.04


def test_invented_identifiers_are_errors_not_no_candidate_source() -> None:
    blocks = (_block("block-1", 0),)
    claims = (_claim("claim-1", "block-1", "One claim."),)
    ledger = ResearchLedger()
    note = _ledger_note(
        ledger,
        item_id="item-1",
        finding="Finding.",
        quote="Exact.",
        source_text="Exact.",
        url="https://source.example/page",
    )
    model = ScriptedAttributionModel(
        {
            "action": "attribute",
            "claims": [
                {
                    "claim_id": "claim-1",
                    "candidates": [
                        {
                            "note_id": "note-does-not-exist",
                            "source_id": note.source_id,
                            "inherited_from_claim_id": None,
                        },
                        {
                            "note_id": note.note_id,
                            "source_id": "source-does-not-exist",
                            "inherited_from_claim_id": None,
                        },
                    ],
                }
            ],
        }
    )

    result = asyncio.run(
        attribute_claims(
            claims,
            blocks=blocks,
            notes=ledger.notes,
            model_client=model,
        )
    )

    attribution = result.attributions[0]
    assert attribution.status == AttributionStatus.ATTRIBUTION_ERROR
    assert attribution.status != AttributionStatus.NO_CANDIDATE_SOURCE
    assert attribution.candidates == ()
    assert {error.code for error in attribution.errors} == {
        "unknown_note_id",
        "unknown_source_id",
    }
    assert all(error.raw is not None for error in attribution.errors)
    assert attribution.claim.source_resolution == SourceResolution.UNRESOLVED


def test_nonlocal_lineage_keeps_candidate_unresolved_with_audit_error() -> None:
    blocks = (
        _block("block-1", 0, section=("First",)),
        _block("block-2", 1, section=("Second",)),
    )
    claims = (
        _claim("claim-1", "block-1", "First claim."),
        _claim("claim-2", "block-2", "Second claim."),
    )
    ledger = ResearchLedger()
    note = _ledger_note(
        ledger,
        item_id="item-1",
        finding="Finding.",
        quote="Exact.",
        source_text="Exact.",
        url="https://source.example/page",
    )
    model = ScriptedAttributionModel(
        {
            "action": "attribute",
            "claims": [
                {
                    "claim_id": "claim-1",
                    "candidates": [
                        {
                            "note_id": note.note_id,
                            "source_id": note.source_id,
                            "inherited_from_claim_id": None,
                        }
                    ],
                },
                {
                    "claim_id": "claim-2",
                    "candidates": [
                        {
                            "note_id": note.note_id,
                            "source_id": note.source_id,
                            "inherited_from_claim_id": "claim-1",
                        }
                    ],
                },
            ],
        }
    )

    result = asyncio.run(
        attribute_claims(
            claims,
            blocks=blocks,
            notes=ledger.notes,
            model_client=model,
        )
    )

    second = _by_claim(result)["claim-2"]
    assert second.status == AttributionStatus.CANDIDATE_SOURCES_WITH_ERRORS
    assert second.errors[0].code == "lineage_outside_markdown_boundary"
    assert len(second.candidates) == 1
    assert second.candidates[0].resolution == SourceResolution.UNRESOLVED
    assert second.candidates[0].inherited_from_claim_id == "claim-1"
    assert second.claim.source_resolution == SourceResolution.UNRESOLVED


def test_same_unit_inheritance_uses_an_earlier_direct_candidate() -> None:
    blocks = (_block("block-1", 0),)
    claims = (
        _claim("claim-1", "block-1", "First atomic claim."),
        _claim("claim-2", "block-1", "Second atomic claim."),
    )
    ledger = ResearchLedger()
    note = _ledger_note(
        ledger,
        item_id="item-1",
        finding="Finding.",
        quote="Exact.",
        source_text="Exact.",
        url="https://source.example/page",
    )
    model = ScriptedAttributionModel(
        {
            "action": "attribute",
            "claims": [
                {
                    "claim_id": "claim-1",
                    "candidates": [
                        {
                            "note_id": note.note_id,
                            "source_id": note.source_id,
                            "inherited_from_claim_id": None,
                        }
                    ],
                },
                {
                    "claim_id": "claim-2",
                    "candidates": [
                        {
                            "note_id": note.note_id,
                            "source_id": note.source_id,
                            "inherited_from_claim_id": "claim-1",
                        }
                    ],
                },
            ],
        }
    )

    result = asyncio.run(
        attribute_claims(
            claims,
            blocks=blocks,
            notes=ledger.notes,
            model_client=model,
        )
    )

    second = _by_claim(result)["claim-2"]
    assert second.status == AttributionStatus.CANDIDATE_SOURCES
    assert second.claim.source_resolution == (
        SourceResolution.INHERITED_SAME_UNIT
    )
    assert second.candidates[0].inherited_from_claim_id == "claim-1"


def test_inheritance_cannot_chain_beyond_a_direct_origin() -> None:
    blocks = (
        _block("block-1", 0),
        _block("block-2", 1),
        _block("block-3", 2),
    )
    claims = (
        _claim("claim-1", "block-1", "First claim."),
        _claim("claim-2", "block-2", "Second claim."),
        _claim("claim-3", "block-3", "Third claim."),
    )
    ledger = ResearchLedger()
    note = _ledger_note(
        ledger,
        item_id="item-1",
        finding="Finding.",
        quote="Exact.",
        source_text="Exact.",
        url="https://source.example/page",
    )
    relation = {
        "note_id": note.note_id,
        "source_id": note.source_id,
    }
    model = ScriptedAttributionModel(
        {
            "action": "attribute",
            "claims": [
                {
                    "claim_id": "claim-1",
                    "candidates": [
                        {
                            **relation,
                            "inherited_from_claim_id": None,
                        }
                    ],
                },
                {
                    "claim_id": "claim-2",
                    "candidates": [
                        {
                            **relation,
                            "inherited_from_claim_id": "claim-1",
                        }
                    ],
                },
                {
                    "claim_id": "claim-3",
                    "candidates": [
                        {
                            **relation,
                            "inherited_from_claim_id": "claim-2",
                        }
                    ],
                },
            ],
        }
    )

    result = asyncio.run(
        attribute_claims(
            claims,
            blocks=blocks,
            notes=ledger.notes,
            model_client=model,
        )
    )

    assert _by_claim(result)["claim-2"].status == (
        AttributionStatus.CANDIDATE_SOURCES
    )
    third = _by_claim(result)["claim-3"]
    assert third.status == AttributionStatus.CANDIDATE_SOURCES_WITH_ERRORS
    assert third.errors[0].code == "inheritance_origin_not_direct"
    assert len(third.candidates) == 1
    assert third.candidates[0].resolution == SourceResolution.UNRESOLVED


def test_repeated_pair_without_origin_is_direct_for_every_claim() -> None:
    blocks = (_block("block-1", 0),)
    claims = (
        _claim("claim-1", "block-1", "First claim."),
        _claim("claim-2", "block-1", "Second claim."),
    )
    ledger = ResearchLedger()
    note = _ledger_note(
        ledger,
        item_id="item-1",
        finding="Finding.",
        quote="Exact.",
        source_text="Exact.",
        url="https://source.example/page",
    )
    repeated = {
        "note_id": note.note_id,
        "source_id": note.source_id,
        "inherited_from_claim_id": None,
    }
    model = ScriptedAttributionModel(
        {
            "action": "attribute",
            "claims": [
                {"claim_id": "claim-1", "candidates": [repeated]},
                {"claim_id": "claim-2", "candidates": [repeated]},
            ],
        }
    )

    result = asyncio.run(
        attribute_claims(
            claims,
            blocks=blocks,
            notes=ledger.notes,
            model_client=model,
        )
    )

    assert all(
        attribution.status == AttributionStatus.CANDIDATE_SOURCES
        for attribution in result.attributions
    )
    assert all(
        attribution.candidates[0].resolution == SourceResolution.DIRECT
        for attribution in result.attributions
    )


def test_omitted_claim_is_attribution_error_while_explicit_empty_is_legal() -> None:
    blocks = (_block("block-1", 0),)
    claims = (
        _claim("claim-1", "block-1", "First claim."),
        _claim("claim-2", "block-1", "Second claim."),
    )
    model = ScriptedAttributionModel(
        {
            "action": "attribute",
            "claims": [{"claim_id": "claim-1", "candidates": []}],
        }
    )

    result = asyncio.run(
        attribute_claims(
            claims,
            blocks=blocks,
            notes=(),
            model_client=model,
        )
    )

    by_claim = _by_claim(result)
    assert by_claim["claim-1"].status == (
        AttributionStatus.NO_CANDIDATE_SOURCE
    )
    assert by_claim["claim-2"].status == AttributionStatus.ATTRIBUTION_ERROR
    assert by_claim["claim-2"].errors[0].code == (
        "missing_claim_attribution"
    )


def test_non_external_claims_never_enter_model_attribution_scope() -> None:
    blocks = (
        _block("block-1", 0),
        _block("block-2", 1),
        _block("block-3", 2),
    )
    external = _claim("claim-external", "block-1", "External fact.")
    internal = _claim(
        "claim-internal",
        "block-2",
        "Internal heading.",
        citation_requirement=CitationRequirement.INTERNAL,
    )
    none = _claim(
        "claim-none",
        "block-3",
        "Non-factual transition.",
        citation_requirement=CitationRequirement.NONE,
    )
    ledger = ResearchLedger()
    note = _ledger_note(
        ledger,
        item_id="item-1",
        finding="External finding.",
        quote="Exact.",
        source_text="Exact.",
        url="https://source.example/page",
    )
    model = ScriptedAttributionModel(
        {
            "action": "attribute",
            "claims": [
                {
                    "claim_id": external.claim_id,
                    "candidates": [
                        {
                            "note_id": note.note_id,
                            "source_id": note.source_id,
                            "inherited_from_claim_id": None,
                        }
                    ],
                }
            ],
        }
    )

    result = asyncio.run(
        attribute_claims(
            (external, internal, none),
            blocks=blocks,
            notes=ledger.notes,
            model_client=model,
        )
    )

    assert len(model.prompts) == 1
    assert external.claim_id in model.prompts[0]
    assert internal.claim_id not in model.prompts[0]
    assert none.claim_id not in model.prompts[0]
    by_claim = _by_claim(result)
    assert by_claim[external.claim_id].status == (
        AttributionStatus.CANDIDATE_SOURCES
    )
    for claim in (internal, none):
        attribution = by_claim[claim.claim_id]
        assert attribution.status == AttributionStatus.NO_CANDIDATE_SOURCE
        assert attribution.candidates == ()
        assert attribution.errors == ()
        assert attribution.claim.source_resolution == (
            SourceResolution.UNRESOLVED
        )


def test_all_non_external_claims_skip_the_attribution_model() -> None:
    blocks = (_block("block-1", 0), _block("block-2", 1))
    claims = (
        _claim(
            "claim-internal",
            "block-1",
            "Internal heading.",
            citation_requirement=CitationRequirement.INTERNAL,
        ),
        _claim(
            "claim-none",
            "block-2",
            "Non-factual transition.",
            citation_requirement=CitationRequirement.NONE,
        ),
    )
    model = ScriptedAttributionModel()

    result = asyncio.run(
        attribute_claims(
            claims,
            blocks=blocks,
            notes=(),
            model_client=model,
        )
    )

    assert model.prompts == []
    assert result.usage == ()
    assert result.stop_reason == AttributionStopReason.COMPLETED
    assert all(
        attribution.status == AttributionStatus.NO_CANDIDATE_SOURCE
        and attribution.candidates == ()
        and attribution.errors == ()
        for attribution in result.attributions
    )
