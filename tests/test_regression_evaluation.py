from open_deep_research.graphrag.evaluation.regression import (
    FactCheck,
    RegressionCase,
    RegressionSuiteResult,
    evaluate_case,
    render_regression_summary,
)
from open_deep_research.graphrag.runtime import (
    GraphResearchResult,
    GraphResearchUsage,
    ResearchRoundTrace,
)
from open_deep_research.graphrag.schemas import EvidencePack, EvidencePackItem


def result() -> GraphResearchResult:
    pack = EvidencePack(
        topic="FTX",
        coverage_ratio=0.5,
        items=[
            EvidencePackItem(
                slot_id="what.core_event",
                conclusion="FTX filed for Chapter 11 bankruptcy on November 11, 2022.",
                confidence=0.8,
                provenance_episode_ids=["episode-1"],
                source_urls=["https://a.example"],
                source_count=1,
                caveats=["single source; no claim-level corroboration"],
            ),
            EvidencePackItem(
                slot_id="who.primary_actor",
                conclusion="Sam Bankman-Fried founded FTX.",
                confidence=0.8,
                provenance_episode_ids=["episode-2", "episode-3"],
                source_urls=["https://b.example", "https://c.example"],
                source_count=2,
                claim_corroborated=True,
                support_requirement_met=True,
                caveats=[],
            ),
        ],
        provenance=["episode-1", "episode-2"],
    )
    return GraphResearchResult(
        topic="FTX",
        research_id="run-test",
        stop_reason="coverage_reached",
        stop_detail="done",
        coverage_ratio=0.5,
        evidence_pack=pack,
        report="report",
        rounds=[
            ResearchRoundTrace(
                round_number=1,
                slot_id="what.core_event",
                query="ftx bankruptcy",
                succeeded=True,
                facts_written=1,
                coverage_after=0.5,
            )
        ],
        fact_count=2,
        source_count=2,
        dated_fact_count=1,
        usage=GraphResearchUsage(
            llm_calls=2,
            search_calls=1,
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            extraction_rows=3,
            grounding_rejections=1,
            elapsed_seconds=4.2,
        ),
    )


def case() -> RegressionCase:
    return RegressionCase(
        case_id="ftx",
        topic="FTX",
        checks=[
            FactCheck(
                check_id="correct_year",
                description="correct year",
                patterns=[r"FTX", r"bankrupt", r"2022"],
            ),
            FactCheck(
                check_id="wrong_year_absent",
                description="wrong year absent",
                patterns=[r"FTX", r"bankrupt", r"2026"],
                should_exist=False,
            ),
            FactCheck(
                check_id="actor_in_slot",
                description="actor in WHO",
                patterns=[r"Sam Bankman-Fried"],
                slot_ids=["who.primary_actor"],
            ),
        ],
    )


def test_case_metrics_keep_quality_dimensions_separate() -> None:
    evaluated = evaluate_case(
        result(),
        case(),
        slot_relevance=0.75,
        judged_item_count=2,
    )

    assert evaluated.factual_check_pass_rate == 1.0
    assert evaluated.citation_coverage == 1.0
    assert evaluated.model_judged_slot_relevance == 0.75
    assert evaluated.claim_corroboration_rate == 0.5
    assert evaluated.multi_source_slot_rate == 0.5
    assert evaluated.cross_corroborated_slot_rate == 0.5
    assert evaluated.round_success_rate == 1.0
    assert evaluated.grounding_rejection_rate == 1 / 3
    assert evaluated.dated_fact_rate == 0.5
    assert evaluated.chat_provider_cost_usd is None


def test_expected_fact_missing_fails_without_affecting_forbidden_check() -> None:
    missing = RegressionCase(
        case_id="missing",
        topic="FTX",
        checks=[
            FactCheck(
                check_id="missing",
                description="not present",
                patterns=[r"Bahamas"],
            ),
            FactCheck(
                check_id="forbidden",
                description="still absent",
                patterns=[r"2026"],
                should_exist=False,
            ),
        ],
    )

    evaluated = evaluate_case(result(), missing)

    assert [check.passed for check in evaluated.fact_checks] == [False, True]
    assert evaluated.factual_check_pass_rate == 0.5


def test_a_declared_explanation_check_can_span_multiple_causal_facts() -> None:
    run = result()
    run.evidence_pack.items = [
        EvidencePackItem(
            slot_id="why.trigger",
            conclusion="Higher interest rates reduced asset values.",
            confidence=0.8,
            provenance_episode_ids=["ep-rate"],
        ),
        EvidencePackItem(
            slot_id="how.mechanism",
            conclusion="SVB sold securities at a $1.8 billion loss.",
            confidence=0.8,
            provenance_episode_ids=["ep-loss"],
        ),
    ]
    causal = RegressionCase(
        case_id="causal",
        topic="SVB",
        checks=[
            FactCheck(
                check_id="rates_and_securities",
                description="both causal components",
                patterns=[r"interest rate", r"securit.*loss|loss.*securit"],
                slot_ids=["why.trigger", "how.mechanism"],
                patterns_may_span_facts=True,
            )
        ],
    )

    evaluated = evaluate_case(run, causal)

    assert evaluated.fact_checks[0].passed is True
    assert len(evaluated.fact_checks[0].matching_facts) == 2


def test_duplicate_queries_are_reported() -> None:
    run = result()
    run.rounds.append(
        ResearchRoundTrace(
            round_number=2,
            slot_id="when.event_time",
            query="FTX BANKRUPTCY",
            succeeded=False,
            facts_written=0,
            coverage_after=0.5,
        )
    )

    assert evaluate_case(run, case()).duplicate_query_count == 1


def test_markdown_summary_labels_model_judgement() -> None:
    evaluated = evaluate_case(result(), case(), slot_relevance=0.5, judged_item_count=2)

    rendered = render_regression_summary(RegressionSuiteResult(cases=[evaluated]))

    assert "Slot relevance*" in rendered
    assert "Claim corroboration" in rendered
    assert "model-judged" in rendered
    assert "Fixed checks" in rendered
