from __future__ import annotations

import math

from open_deep_research.harness.claims import (
    AtomicClaim,
    CitationRequirement,
    ClaimNormalizationStatus,
    SourceResolution,
    parse_markdown_blocks,
)
from open_deep_research.harness.concentration import (
    audit_domain_proxy_concentration,
)
from open_deep_research.harness.notes import (
    NoteLocationStatus,
    QuoteSpan,
    ResearchNote,
    source_id_for_url,
)
from open_deep_research.harness.reconcile import (
    ChecklistCoverageDisposition,
    ChecklistCoverageRecord,
    ChecklistCoverageReference,
    ChecklistCoverageSummary,
    ChecklistReportReconciliation,
    CoverageAssessmentStatus,
)
from open_deep_research.harness.render import render_verified_report
from open_deep_research.harness.truth_conditions import (
    ElementAssessmentExecutionStatus,
    ElementVerificationVerdict,
    ElementizationProposal,
    ElementizationReview,
    ElementizationSemanticStatus,
    aggregate_truth_condition_claim,
    build_truth_condition_registry,
)
from open_deep_research.harness.verify import (
    ClaimEvidenceState,
    ClaimVerification,
    VerificationRecordStatus,
    VerificationResult,
    VerificationVerdict,
    VerifiedElementRelation,
    VerifiedSourceRelation,
    build_claim_verification,
)


def _claim(
    draft: str,
    blocks,
    claim_id: str,
    anchor: str,
) -> AtomicClaim:
    start = draft.index(anchor)
    block = next(block for block in blocks if anchor in block.text)
    return AtomicClaim(
        claim_id=claim_id,
        block_id=block.block_id,
        selected_text=anchor,
        claim_text=anchor,
        anchor_text=anchor,
        start_char=start,
        end_char=start + len(anchor),
        citation_requirement=CitationRequirement.EXTERNAL,
        source_resolution=SourceResolution.DIRECT,
        normalization_status=ClaimNormalizationStatus.LOCATED,
    )


def _relation(
    claim_id: str,
    *,
    url: str,
    proxy: str,
    source_number: int,
) -> VerifiedSourceRelation:
    source_id = source_id_for_url(url)
    return VerifiedSourceRelation(
        claim_id=claim_id,
        source_id=source_id,
        url=url,
        publisher_domain_proxy=proxy,
        candidate_note_ids=(f"note-{source_number}",),
        candidate_source_ids=(source_id,),
        status=VerificationRecordStatus.COMPLETED,
        semantic_verdict=VerificationVerdict.SUPPORTS,
        model_quote=f"Evidence {source_number}",
        source_quote=f"Evidence {source_number}",
        span=QuoteSpan(
            start_char=source_number * 10,
            end_char=source_number * 10 + 10,
        ),
        location_status=NoteLocationStatus.LOCATABLE,
        is_formal_supporting_evidence=True,
    )


def _verified(
    claim: AtomicClaim,
    *relations: VerifiedSourceRelation,
) -> ClaimVerification:
    proxies = tuple(
        sorted({relation.publisher_domain_proxy for relation in relations})
    )
    return ClaimVerification(
        claim=claim,
        state=(
            ClaimEvidenceState.SUPPORTED_MULTIPLE_DOMAIN_PROXIES
            if len(proxies) > 1
            else ClaimEvidenceState.SUPPORTED_SINGLE_DOMAIN_PROXY
        ),
        corroboration_target=2,
        relations=tuple(relations),
        formal_supporting_evidence_count=len(relations),
        publisher_domain_proxy_count=len(proxies),
        publisher_domain_proxies=proxies,
    )


def _reference(claim: AtomicClaim) -> ChecklistCoverageReference:
    assert claim.anchor_text is not None
    assert claim.start_char is not None
    assert claim.end_char is not None
    return ChecklistCoverageReference(
        claim_id=claim.claim_id,
        block_id=claim.block_id,
        anchor_text=claim.anchor_text,
        start_char=claim.start_char,
        end_char=claim.end_char,
    )


def _reconciliation(
    first: AtomicClaim,
    second: AtomicClaim,
    third: AtomicClaim,
) -> ChecklistReportReconciliation:
    records = (
        ChecklistCoverageRecord(
            item_id="item-1",
            question="First question?",
            proposed_disposition=ChecklistCoverageDisposition.COVERED,
            disposition=ChecklistCoverageDisposition.COVERED,
            rationale="Both relevant claims answer it.",
            references=(_reference(first), _reference(second)),
            proposed_claim_ids=(first.claim_id, second.claim_id),
            assessment_status=CoverageAssessmentStatus.COMPLETED,
        ),
        ChecklistCoverageRecord(
            item_id="item-2",
            question="Second question?",
            proposed_disposition=(
                ChecklistCoverageDisposition.PARTIALLY_COVERED
            ),
            disposition=ChecklistCoverageDisposition.PARTIALLY_COVERED,
            rationale="One relevant claim addresses part of it.",
            references=(_reference(third),),
            proposed_claim_ids=(third.claim_id,),
            assessment_status=CoverageAssessmentStatus.COMPLETED,
        ),
        ChecklistCoverageRecord(
            item_id="item-3",
            question="Third question?",
            proposed_disposition=ChecklistCoverageDisposition.NOT_COVERED,
            disposition=ChecklistCoverageDisposition.NOT_COVERED,
            rationale="The report does not address it.",
            assessment_status=CoverageAssessmentStatus.COMPLETED,
        ),
    )
    return ChecklistReportReconciliation(
        records=records,
        summary=ChecklistCoverageSummary(
            total_items=3,
            assessed_items=3,
            covered_items=1,
            partially_covered_items=1,
            not_covered_items=1,
            assessment_failed_items=0,
            covered_rate=1 / 3,
            partially_covered_item_ids=("item-2",),
            not_covered_item_ids=("item-3",),
        ),
    )


def test_concentration_uses_formal_claim_source_relations_without_score() -> None:
    draft = (
        "# Report\n\n## Alpha\n\nAlpha one. Alpha two.\n\n"
        "## Beta\n\nBeta fact."
    )
    blocks = parse_markdown_blocks(draft)
    first = _claim(draft, blocks, "claim-1", "Alpha one.")
    second = _claim(draft, blocks, "claim-2", "Alpha two.")
    third = _claim(draft, blocks, "claim-3", "Beta fact.")
    first_a = _relation(
        first.claim_id,
        url="https://a.example/one",
        proxy="a.example",
        source_number=1,
    )
    second_a = _relation(
        second.claim_id,
        url="https://a.example/two",
        proxy="a.example",
        source_number=2,
    )
    second_b = _relation(
        second.claim_id,
        url="https://b.example/two",
        proxy="b.example",
        source_number=3,
    )
    third_b = _relation(
        third.claim_id,
        url="https://b.example/three",
        proxy="b.example",
        source_number=4,
    )
    verification = VerificationResult(
        claims=(
            _verified(first, first_a),
            _verified(second, second_a, second_b),
            _verified(third, third_b),
        )
    )

    audit = audit_domain_proxy_concentration(
        verification,
        blocks=blocks,
        reconciliation=_reconciliation(first, second, third),
        source_cache={
            first_a.url: "Evidence 1",
            second_a.url: "Evidence 2",
            second_b.url: "Evidence 3",
            third_b.url: "Evidence 4",
        },
        notes=(),
    )

    assert audit.counting_unit == "formal_claim_source_support_relation"
    assert audit.method == "publisher_domain_proxy"
    assert audit.is_organization_independence_determination is False
    assert audit.is_viewpoint_diversity_determination is False
    assert audit.overall.formal_support_relation_count == 4
    assert audit.overall.publisher_domain_proxy_count == 2
    assert [
        (
            row.publisher_domain_proxy,
            row.formal_support_relation_count,
        )
        for row in audit.overall.publisher_domain_proxy_distribution
    ] == [("a.example", 2), ("b.example", 2)]
    assert audit.overall.largest_publisher_domain_proxy_share == 0.5
    assert audit.overall.raw_hhi == 0.5
    assert audit.overall.effective_publisher_domain_proxy_count == 2.0
    assert "score" not in audit.model_dump_json()
    assert "threshold" not in audit.model_dump_json()


def test_sections_and_checklist_items_expose_monopoly_and_unused_reads() -> None:
    draft = (
        "# Report\n\n## Alpha\n\nAlpha one. Alpha two.\n\n"
        "## Beta\n\nBeta fact."
    )
    blocks = parse_markdown_blocks(draft)
    first = _claim(draft, blocks, "claim-1", "Alpha one.")
    second = _claim(draft, blocks, "claim-2", "Alpha two.")
    third = _claim(draft, blocks, "claim-3", "Beta fact.")
    source_a = "https://a.example/article"
    source_b = "https://b.example/article"
    source_c = "https://c.example/article"
    first_a = _relation(
        first.claim_id,
        url=source_a,
        proxy="a.example",
        source_number=1,
    )
    second_b = _relation(
        second.claim_id,
        url=source_b,
        proxy="b.example",
        source_number=2,
    )
    third_b = second_b.model_copy(update={"claim_id": third.claim_id})
    verification = VerificationResult(
        claims=(
            _verified(first, first_a),
            _verified(second, second_b),
            _verified(third, third_b),
        )
    )
    unused_note = ResearchNote(
        note_id="note-unused",
        item_id="item-1",
        source_id=source_id_for_url(source_c),
        finding="Relevant material was collected but never used.",
        model_quote="Unused exact quote.",
        source_quote="Unused exact quote.",
        url=source_c,
        publisher="c.example",
        span=QuoteSpan(start_char=0, end_char=19),
        location_status=NoteLocationStatus.LOCATABLE,
    )

    audit = audit_domain_proxy_concentration(
        verification,
        blocks=blocks,
        reconciliation=_reconciliation(first, second, third),
        source_cache={
            source_a: "Evidence 1",
            source_b: "Evidence 2",
            source_c: "Unused exact quote.",
        },
        notes=(unused_note,),
    )

    beta = next(
        section
        for section in audit.sections
        if section.section_path == ("Report", "Beta")
    )
    assert beta.is_single_publisher_domain_proxy_monopoly is True
    assert beta.monopoly_publisher_domain_proxy == "b.example"
    assert beta.unit_id in audit.single_publisher_monopoly_section_ids
    assert {
        source.url for source in beta.read_but_unused_sources
    } == {source_a, source_c}

    first_item = next(
        item
        for item in audit.checklist_items
        if item.checklist_item_id == "item-1"
    )
    assert first_item.distribution.formal_support_relation_count == 2
    assert {
        row.publisher_domain_proxy
        for row in first_item.distribution.publisher_domain_proxy_distribution
    } == {"a.example", "b.example"}
    unused_c = next(
        source
        for source in first_item.read_but_unused_sources
        if source.url == source_c
    )
    assert unused_c.total_note_count == 1
    assert unused_c.notes_for_checklist_item == 1

    uncovered = next(
        item
        for item in audit.checklist_items
        if item.checklist_item_id == "item-3"
    )
    assert uncovered.claim_ids == ()
    assert uncovered.distribution.formal_support_relation_count == 0
    assert {source.url for source in uncovered.read_but_unused_sources} == {
        source_a,
        source_b,
        source_c,
    }


def test_duplicate_claim_source_relation_is_audited_not_double_counted() -> None:
    draft = "# Report\n\nOne fact."
    blocks = parse_markdown_blocks(draft)
    claim = _claim(draft, blocks, "claim-1", "One fact.")
    relation = _relation(
        claim.claim_id,
        url="https://one.example/article",
        proxy="one.example",
        source_number=1,
    )
    verification = VerificationResult(
        claims=(_verified(claim, relation, relation),)
    )
    reconciliation = ChecklistReportReconciliation(
        records=(
            ChecklistCoverageRecord(
                item_id="item-1",
                question="Question?",
                proposed_disposition=ChecklistCoverageDisposition.COVERED,
                disposition=ChecklistCoverageDisposition.COVERED,
                rationale="The claim answers it.",
                references=(_reference(claim),),
                proposed_claim_ids=(claim.claim_id,),
                assessment_status=CoverageAssessmentStatus.COMPLETED,
            ),
        ),
        summary=ChecklistCoverageSummary(
            total_items=1,
            assessed_items=1,
            covered_items=1,
            partially_covered_items=0,
            not_covered_items=0,
            assessment_failed_items=0,
            covered_rate=1.0,
        ),
    )

    audit = audit_domain_proxy_concentration(
        verification,
        blocks=blocks,
        reconciliation=reconciliation,
        source_cache={relation.url: "Evidence 1"},
        notes=(),
    )

    assert audit.overall.formal_support_relation_count == 1
    assert audit.diagnostics == (
        "duplicate_formal_claim_source_relation:claim-1:"
        f"{relation.source_id}",
    )


def test_renderer_exposes_only_reader_facing_maximum_share() -> None:
    draft = "# Report\n\nOne fact. Two facts."
    blocks = parse_markdown_blocks(draft)
    first = _claim(draft, blocks, "claim-1", "One fact.")
    second = _claim(draft, blocks, "claim-2", "Two facts.")
    first_relation = _relation(
        first.claim_id,
        url="https://large.example/one",
        proxy="large.example",
        source_number=1,
    )
    second_relation = _relation(
        second.claim_id,
        url="https://small.example/two",
        proxy="small.example",
        source_number=2,
    )
    verification = VerificationResult(
        claims=(
            _verified(first, first_relation),
            _verified(second, second_relation),
        )
    )
    reconciliation = _reconciliation(first, second, second)
    audit = audit_domain_proxy_concentration(
        verification,
        blocks=blocks,
        reconciliation=reconciliation,
        source_cache={
            first_relation.url: "Evidence 1",
            second_relation.url: "Evidence 2",
        },
        notes=(),
    )

    rendered = render_verified_report(
        draft,
        verification,
        domain_proxy_concentration=audit,
    )

    assert rendered.domain_proxy_concentration_line is not None
    assert "最大域名代理 large.example" in (
        rendered.domain_proxy_concentration_line
    )
    assert "50.0%（1/2）" in rendered.domain_proxy_concentration_line
    assert "域名仅作发布方代理" in rendered.domain_proxy_concentration_line
    assert rendered.domain_proxy_concentration_line in rendered.markdown
    assert "HHI" not in rendered.markdown
    assert "有效发布方" not in rendered.markdown
    assert math.isclose(audit.overall.raw_hhi, 0.5)


def test_concentration_excludes_element_only_support_from_whole_claim_count() -> None:
    draft = "# Report\n\nAlpha acquired Beta for $2 billion."
    blocks = parse_markdown_blocks(draft)
    claim = _claim(
        draft,
        blocks,
        "claim-distributed",
        "Alpha acquired Beta for $2 billion.",
    )

    registry = build_truth_condition_registry(
        {claim.claim_id: claim.claim_text},
        proposals=(
            ElementizationProposal(
                claim_id=claim.claim_id,
                elements=("Alpha acquired Beta.", "The price was $2 billion."),
                rationale="proposal",
            ),
        ),
        reviews=(
            ElementizationReview(
                claim_id=claim.claim_id,
                semantic_status=ElementizationSemanticStatus.COMPLETE,
                elements=("Alpha acquired Beta.", "The price was $2 billion."),
                rationale="independent review",
            ),
        ),
    )
    entry = registry.entries[0]
    first_element, second_element = entry.elements

    def element_relation(
        source_id: str,
        element,
        *,
        supports: bool,
    ) -> VerifiedElementRelation:
        quote = element.text if supports else None
        return VerifiedElementRelation(
            claim_id=claim.claim_id,
            element_id=element.element_id,
            element_text=element.text,
            source_id=source_id,
            status=ElementAssessmentExecutionStatus.COMPLETE,
            semantic_verdict=(
                ElementVerificationVerdict.SUPPORTS
                if supports
                else ElementVerificationVerdict.DOES_NOT_SUPPORT
            ),
            source_quote=quote,
            span=(
                QuoteSpan(start_char=0, end_char=len(quote))
                if quote is not None
                else None
            ),
            location_status=(
                NoteLocationStatus.LOCATABLE if supports else None
            ),
            is_formal_supporting_evidence=supports,
        )

    def partial_relation(
        source_id: str,
        proxy: str,
        *,
        supports_first: bool,
    ) -> VerifiedSourceRelation:
        return VerifiedSourceRelation(
            claim_id=claim.claim_id,
            source_id=source_id,
            url=f"https://{proxy}/report",
            publisher_domain_proxy=proxy,
            candidate_note_ids=(f"note-{source_id}",),
            candidate_source_ids=(source_id,),
            status=VerificationRecordStatus.COMPLETED,
            semantic_verdict=VerificationVerdict.NOT_ENOUGH_INFORMATION,
            element_relations=(
                element_relation(
                    source_id,
                    first_element,
                    supports=supports_first,
                ),
                element_relation(
                    source_id,
                    second_element,
                    supports=not supports_first,
                ),
            ),
        )

    # Relation source IDs are valid audit identities but deliberately differ
    # from the cache's URL-derived IDs.  Used/unused classification must still
    # follow the canonical URL identity.
    first_source_id = "external-source-first"
    second_source_id = "historical-source-second"
    relations = (
        partial_relation(
            first_source_id,
            "first.example",
            supports_first=True,
        ),
        partial_relation(
            second_source_id,
            "second.example",
            supports_first=False,
        ),
    )
    aggregate = aggregate_truth_condition_claim(
        entry,
        tuple(
            element.as_assessment()
            for relation in relations
            for element in relation.element_relations
        ),
        expected_source_ids=(first_source_id, second_source_id),
    )
    verification = VerificationResult(
        claims=(
            build_claim_verification(
                claim,
                relations,
                required_sources=2,
                truth_condition_aggregate=aggregate,
            ),
        )
    )
    reconciliation = ChecklistReportReconciliation(
        records=(),
        summary=ChecklistCoverageSummary(
            total_items=0,
            assessed_items=0,
            covered_items=0,
            partially_covered_items=0,
            not_covered_items=0,
            assessment_failed_items=0,
            covered_rate=0.0,
        ),
    )

    audit = audit_domain_proxy_concentration(
        verification,
        blocks=blocks,
        reconciliation=reconciliation,
        source_cache={
            relation.url: "Element-level evidence."
            for relation in relations
        },
        notes=(),
    )

    assert audit.overall.formal_support_relation_count == 0
    assert audit.overall.publisher_domain_proxy_count == 0
    assert audit.element_only_support is not None
    assert audit.element_only_support.claim_source_relation_count == 2
    assert audit.element_only_support.source_ids == tuple(
        sorted((first_source_id, second_source_id))
    )
    assert audit.element_only_support.publisher_domain_proxies == (
        "first.example",
        "second.example",
    )
    assert audit.sections[0].element_only_support == audit.element_only_support
    assert audit.sections[0].read_but_unused_sources == ()
    assert verification.claims[0].state is (
        ClaimEvidenceState.SUPPORTED_DISTRIBUTED_ELEMENT_EVIDENCE
    )
