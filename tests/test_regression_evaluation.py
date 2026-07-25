from open_deep_research.graphrag.evaluation.regression import (
    DEFAULT_REGRESSION_CASES,
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


def quake_case() -> RegressionCase:
    return next(
        case
        for case in DEFAULT_REGRESSION_CASES
        if case.case_id == "turkiye_quake_2023"
    )


def quake_result(*conclusions: tuple[str, str]) -> GraphResearchResult:
    run = result()
    run.topic = quake_case().topic
    run.evidence_pack = EvidencePack(
        topic=run.topic,
        coverage_ratio=1.0,
        items=[
            EvidencePackItem(
                slot_id=slot_id,
                conclusion=conclusion,
                confidence=0.9,
                provenance_episode_ids=[f"episode-{index}"],
            )
            for index, (slot_id, conclusion) in enumerate(conclusions)
        ],
    )
    return run


def test_turkiye_quake_holdout_has_six_auditable_checks() -> None:
    holdout = quake_case()

    assert "Türkiye–Syria earthquakes" in holdout.topic
    assert "February 6, 2023" in holdout.topic
    assert "Mw 7.8" in holdout.topic
    assert len(holdout.checks) == 6
    assert len({check.check_id for check in holdout.checks}) == 6
    assert sum(not check.should_exist for check in holdout.checks) == 1
    assert {
        check.check_id
        for check in holdout.checks
        if check.slot_ids
    } == {"turkiye_quake_date"}


def test_turkiye_quake_patterns_match_cross_language_facts() -> None:
    run = quake_result(
        ("when.event_time", "主震发生于2023年2月6日。"),
        ("what.core_event", "The mainshock had a moment magnitude of Mw 7.8."),
        ("where.event_location", "土耳其南部受到地震影响。"),
        ("who.affected_parties", "Northern Syria was also severely affected."),
        (
            "what.scale",
            "More than 59,000 people were killed across the affected region.",
        ),
        (
            "how.sequence",
            "International search-and-rescue teams joined the response.",
        ),
    )

    evaluated = evaluate_case(run, quake_case())

    assert evaluated.factual_check_pass_rate == 1.0
    assert all(check.passed for check in evaluated.fact_checks)


def test_turkiye_quake_patterns_reject_wrong_facts() -> None:
    run = quake_result(
        ("when.event_time", "The earthquake struck on February 6, 2024."),
        ("what.core_event", "The mainshock had a magnitude of Mw 7.5."),
        ("where.event_location", "Türkiye was affected."),
        ("what.scale", "Nine people were killed."),
    )

    evaluated = evaluate_case(run, quake_case())
    by_check = {check.check_id: check for check in evaluated.fact_checks}

    assert by_check["turkiye_quake_date"].passed is False
    assert by_check["turkiye_quake_magnitude"].passed is False
    assert by_check["turkiye_quake_affected_countries"].passed is False
    assert by_check["turkiye_quake_death_toll"].passed is False
    assert by_check["turkiye_quake_no_wrong_year"].passed is False
