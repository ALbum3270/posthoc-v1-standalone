import asyncio
from types import SimpleNamespace

from open_deep_research.graphrag.control.stopping import StopReason, StoppingConfig
from open_deep_research.graphrag.ontology import OntologySlot
from open_deep_research.graphrag.runtime import (
    EXTRACTION_SYSTEM_PROMPT,
    GraphResearchRunner,
    GraphResearchSettings,
    GraphResearchUsage,
    SUPPORT_EXTRACTION_SYSTEM_PROMPT,
    _finalize_loop_stop,
)
from open_deep_research.graphrag.schemas import (
    EntityRef,
    ExtractedTriple,
    RelevanceStatus,
    SlotApplicabilityStatus,
    SourceDocument,
)


SLOT = OntologySlot(
    slot_id="what.scale",
    dimension="WHAT",
    label="Scale",
    question="What was the financial scale?",
)


def response(content: str, *, cost: float | None = None):
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    if cost is not None:
        usage["cost"] = cost
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage,
    )


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeLLM:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


class FakeTavily:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []

    async def search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"results": self.results}


def runner(*responses, settings=None, tavily=None) -> GraphResearchRunner:
    return GraphResearchRunner(
        graphiti=SimpleNamespace(),
        llm=FakeLLM(responses),
        tavily=tavily or FakeTavily(),
        settings=settings
        or GraphResearchSettings(
            model="test/model",
            enable_relevance_gate=False,
        ),
    )


def test_usage_collects_tokens_and_only_reports_real_provider_cost() -> None:
    usage = GraphResearchUsage()

    usage.observe_llm_response(response("one", cost=0.012))
    usage.observe_llm_response(response("two"))

    assert usage.llm_calls == 2
    assert usage.prompt_tokens == 20
    assert usage.completion_tokens == 10
    assert usage.total_tokens == 30
    assert usage.chat_provider_cost_usd == 0.012
    assert usage.provider_cost_reported is True


def test_final_allowed_round_reports_coverage_before_round_cap() -> None:
    settings = StoppingConfig(coverage_target=1.0, max_rounds=1)

    completed = _finalize_loop_stop(
        None,
        coverage_ratio=1.0,
        settings=settings,
    )
    exhausted = _finalize_loop_stop(
        None,
        coverage_ratio=0.5,
        settings=settings,
    )

    assert completed.reason is StopReason.COVERAGE_REACHED
    assert exhausted.reason is StopReason.MAX_ROUNDS


def test_extraction_accepts_only_exactly_quoted_rows() -> None:
    source = SourceDocument(
        document_id="doc",
        title="Source",
        url="https://example.com",
        content="The filing says FTX owed $8 billion to customers.",
    )
    payload = (
        '{"triples":['
        '{"subject":"FTX","predicate":"owed","object":"$8 billion",'
        '"quote":"The filing says FTX owed $8 billion to customers."},'
        '{"subject":"FTX","predicate":"owed","object":"$9 billion",'
        '"quote":"The filing says FTX owed $8 billion to customers."}'
        "]}"
    )
    active = runner(response(payload))

    triples = asyncio.run(active.extract(document=source, slot=SLOT))

    assert len(triples) == 1
    assert triples[0].source_span.quote == source.content
    assert active.usage.extraction_rows == 2
    assert active.usage.grounding_rejections == 1


def test_extraction_prompts_require_complete_self_contained_sentences() -> None:
    for prompt in (EXTRACTION_SYSTEM_PROMPT, SUPPORT_EXTRACTION_SYSTEM_PROMPT):
        assert "complete, self-contained sentence" in prompt
        assert "explicitly names the triple subject" in prompt
        assert "mid-word" in prompt


def test_query_generator_deterministically_breaks_an_exact_repeat() -> None:
    active = runner(response("same query"))

    query = asyncio.run(
        active.generate_query(
            topic="topic",
            slot=SLOT,
            previous_queries=["same query"],
        )
    )

    assert query != "same query"
    assert query.startswith("same query primary source evidence attempt")


def test_causal_query_requests_both_underlying_and_proximate_causes() -> None:
    causal = OntologySlot(
        slot_id="why.trigger",
        dimension="WHY",
        label="Trigger",
        question="What caused the event?",
    )
    active = runner(response("specific causal query"))

    asyncio.run(
        active.generate_query(
            topic="SVB collapse",
            slot=causal,
            previous_queries=[],
        )
    )

    prompt = active.llm.chat.completions.calls[0]["messages"][1]["content"]
    assert "proximate trigger" in prompt
    assert "longer-term underlying causal mechanism" in prompt


def test_tavily_boundary_limits_an_ordinary_query_without_splitting_phrase() -> None:
    tavily = FakeTavily()
    active = runner(tavily=tavily)
    query = '"Bank of New York Mellon" ' + " ".join(
        f"ordinary-{index}" for index in range(80)
    )

    asyncio.run(active.search(query=query, exclude_urls=[]))

    sent_query = tavily.calls[0][0][0]
    assert len(sent_query) <= 400
    assert sent_query.startswith('"Bank of New York Mellon" ')
    assert sent_query.split()[-1] in query.split()


def test_tavily_boundary_limits_a_support_query_without_splitting_phrase() -> None:
    tavily = FakeTavily()
    active = runner(tavily=tavily)
    query = '"the filing reported exactly eight billion dollars" ' + " ".join(
        f"support-{index}" for index in range(80)
    )

    asyncio.run(active.search(query=query, exclude_urls=[]))

    sent_query = tavily.calls[0][0][0]
    assert len(sent_query) <= 400
    assert sent_query.startswith(
        '"the filing reported exactly eight billion dollars" '
    )
    assert sent_query.split()[-1] in query.split()


def test_search_rejects_ambiguous_name_results_without_topic_anchors() -> None:
    tavily = FakeTavily(
        [
            {
                "title": "Sam definition and meaning",
                "url": "https://dictionary.example/sam",
                "content": "Sam is a given name.",
                "raw_content": "Sam is a common given name and abbreviation.",
            },
            {
                "title": "Sam Bankman-Fried and the collapse of FTX",
                "url": "https://news.example/ftx",
                "content": "A report about the 2022 collapse.",
                "raw_content": (
                    "Sam Bankman-Fried founded FTX, which collapsed in 2022."
                ),
            },
        ]
    )
    active = runner(tavily=tavily)

    documents = asyncio.run(
        active.search(
            query='"Sam Bankman-Fried" FTX collapse 2022',
            exclude_urls=[],
        )
    )

    assert [document.url for document in documents] == [
        "https://news.example/ftx"
    ]
    assert active.usage.search_results_rejected == 1


def test_search_requires_the_full_multiword_entity_anchor() -> None:
    tavily = FakeTavily(
        [
            {
                "title": "Silicon - chemical element",
                "url": "https://chemistry.example/silicon",
                "content": "Silicon is used in electronics and was studied in 2023.",
                "raw_content": (
                    "Silicon is a chemical element used in electronics. "
                    "This reference page was updated in March 2023."
                ),
            },
            {
                "title": "Silicon Valley Bank collapse",
                "url": "https://finance.example/svb",
                "content": "Silicon Valley Bank was closed in March 2023.",
                "raw_content": (
                    "Silicon Valley Bank failed and was closed in March 2023."
                ),
            },
        ]
    )
    active = runner(tavily=tavily)

    documents = asyncio.run(
        active.search(
            query='"Silicon Valley Bank" collapse March 2023',
            exclude_urls=[],
        )
    )

    assert [document.url for document in documents] == [
        "https://finance.example/svb"
    ]
    assert active.usage.search_results_rejected == 1


def test_search_does_not_combine_adjacent_entities_into_a_fake_anchor() -> None:
    tavily = FakeTavily(
        [
            {
                "title": "Sam Bankman-Fried founded Alameda Research",
                "url": "https://news.example/alameda",
                "content": "Sam Bankman-Fried founded Alameda Research.",
                "raw_content": (
                    "Sam Bankman-Fried founded Alameda Research before the "
                    "collapse."
                ),
            },
        ]
    )
    active = runner(tavily=tavily)

    documents = asyncio.run(
        active.search(
            query="Sam Bankman-Fried Alameda Research collapse",
            exclude_urls=[],
        )
    )

    assert [document.url for document in documents] == [
        "https://news.example/alameda"
    ]
    assert active.usage.search_results_rejected == 0


def test_search_does_not_weaken_a_long_quoted_institution_name() -> None:
    tavily = FakeTavily(
        [
            {
                "title": "New York liquidity overview",
                "url": "https://example.com/new-york",
                "content": "New York liquidity conditions improved.",
                "raw_content": "New York liquidity conditions improved.",
            },
            {
                "title": "Bank of New York Mellon liquidity report",
                "url": "https://example.com/bnym",
                "content": "Bank of New York Mellon reported stronger liquidity.",
                "raw_content": (
                    "Bank of New York Mellon reported stronger liquidity."
                ),
            },
        ]
    )
    active = runner(tavily=tavily)

    documents = asyncio.run(
        active.search(
            query='"Bank of New York Mellon" liquidity',
            exclude_urls=[],
        )
    )

    assert [document.url for document in documents] == [
        "https://example.com/bnym"
    ]
    assert active.usage.search_results_rejected == 1


def test_search_preserves_a_quoted_chinese_person_name_as_an_anchor() -> None:
    tavily = FakeTavily(
        [
            {
                "title": "阿里巴巴公司动态",
                "url": "https://example.com/alibaba",
                "content": "阿里巴巴发布公司动态。",
                "raw_content": "阿里巴巴发布公司动态。",
            },
            {
                "title": "马云谈阿里巴巴",
                "url": "https://example.com/jack-ma",
                "content": "马云谈到阿里巴巴的发展。",
                "raw_content": "马云谈到阿里巴巴的发展。",
            },
        ]
    )
    active = runner(tavily=tavily)

    documents = asyncio.run(
        active.search(
            query='"马云" 阿里巴巴',
            exclude_urls=[],
        )
    )

    assert [document.url for document in documents] == [
        "https://example.com/jack-ma"
    ]
    assert active.usage.search_results_rejected == 1


def test_critical_slot_keeps_one_central_claim_for_independent_support() -> None:
    critical = SLOT.model_copy(update={"required_source_count": 2})
    source = SourceDocument(
        document_id="doc",
        title="Source",
        content="FTX owed $8 billion. FTX had more than one million creditors.",
    )
    payload = (
        '{"triples":['
        '{"subject":"FTX","predicate":"owed","object":"$8 billion",'
        '"quote":"FTX owed $8 billion."},'
        '{"subject":"FTX","predicate":"had","object":"more than one million creditors",'
        '"quote":"FTX had more than one million creditors."}'
        "]}"
    )
    active = runner(response(payload))

    triples = asyncio.run(active.extract(document=source, slot=critical))

    assert len(triples) == 1
    assert triples[0].predicate == "owed"
    user_prompt = active.llm.chat.completions.calls[0]["messages"][1]["content"]
    assert "at most 1 central, atomic claim" in user_prompt


def test_trigger_slot_can_keep_one_underlying_cause_and_one_immediate_trigger() -> None:
    trigger = OntologySlot(
        slot_id="why.trigger",
        dimension="WHY",
        label="Trigger",
        question="What caused the event?",
        required_source_count=2,
        max_initial_claims=2,
    )
    source = SourceDocument(
        document_id="doc",
        title="Source",
        content=(
            "Rising rates reduced the value of SVB's securities. "
            "A failed capital raise triggered withdrawals."
        ),
    )
    payload = (
        '{"triples":['
        '{"subject":"rising rates","predicate":"reduced","object":"security values",'
        '"quote":"Rising rates reduced the value of SVB\\u0027s securities."},'
        '{"subject":"a failed capital raise","predicate":"triggered",'
        '"object":"withdrawals",'
        '"quote":"A failed capital raise triggered withdrawals."},'
        '{"subject":"SVB","predicate":"was","object":"a bank",'
        '"quote":"Rising rates reduced the value of SVB\\u0027s securities."}'
        "]}"
    )
    active = runner(response(payload))

    triples = asyncio.run(active.extract(document=source, slot=trigger))

    assert len(triples) == 2
    user_prompt = active.llm.chat.completions.calls[0]["messages"][1]["content"]
    assert "one for an underlying condition" in user_prompt
    assert "one for the immediate trigger" in user_prompt


def test_extraction_prompt_requires_a_verbatim_quote() -> None:
    active = runner(response('{"triples":[]}'))
    source = SourceDocument(document_id="doc", title="Source", content="body")

    asyncio.run(active.extract(document=source, slot=SLOT))

    system_prompt = active.llm.chat.completions.calls[0]["messages"][0]["content"]
    assert "quote is mandatory" in system_prompt.lower()
    assert "verbatim" in system_prompt.lower()


def test_relevance_gate_rejects_a_grounded_but_off_topic_quote() -> None:
    source = SourceDocument(
        document_id="doc",
        title="Source",
        url="https://example.com",
        content="Silicon is a chemical element with the symbol Si.",
    )
    extracted = (
        '{"triples":[{"subject":"Silicon","predicate":"is",'
        '"object":"a chemical element",'
        '"quote":"Silicon is a chemical element with the symbol Si."}]}'
    )
    verdict = (
        '{"decisions":[{"index":0,"status":"rejected","confidence":0.99,'
        '"reason":"chemical element is unrelated to the bank"}]}'
    )
    active = runner(
        response(extracted),
        response(verdict),
        settings=GraphResearchSettings(
            model="test/model",
            enable_relevance_gate=True,
        ),
    )

    triples = asyncio.run(
        active.extract(
            document=source,
            slot=SLOT,
            topic="Silicon Valley Bank collapse",
        )
    )

    assert triples == []
    assert active.usage.relevance_rejected == 1
    assert active.relevance_audit[0].status is RelevanceStatus.REJECTED


def test_low_confidence_rejection_is_retained_as_uncertain() -> None:
    source = SourceDocument(
        document_id="doc",
        title="Source",
        content="FTX filed for bankruptcy.",
    )
    extracted = (
        '{"triples":[{"subject":"FTX","predicate":"filed for",'
        '"object":"bankruptcy","quote":"FTX filed for bankruptcy."}]}'
    )
    verdict = (
        '{"decisions":[{"index":0,"status":"rejected","confidence":0.55,'
        '"reason":"ambiguous"}]}'
    )
    active = runner(
        response(extracted),
        response(verdict),
        settings=GraphResearchSettings(
            model="test/model",
            enable_relevance_gate=True,
            relevance_reject_threshold=0.8,
        ),
    )

    triples = asyncio.run(
        active.extract(document=source, slot=SLOT, topic="FTX collapse")
    )

    assert len(triples) == 1
    assert triples[0].relevance_status is RelevanceStatus.UNCERTAIN


def test_support_extraction_must_copy_an_existing_structured_claim() -> None:
    target = ExtractedTriple(
        slot_id=SLOT.slot_id,
        subject=EntityRef(name="FTX"),
        predicate="filed for",
        object="Chapter 11",
        source_document_id="first",
    )
    source = SourceDocument(
        document_id="second",
        title="Independent",
        content="An independent filing confirms FTX filed for Chapter 11.",
    )
    payload = (
        '{"triples":[{"subject":"FTX","predicate":"filed for",'
        '"object":"Chapter 11",'
        '"quote":"An independent filing confirms FTX filed for Chapter 11."},'
        '{"subject":"FTX","predicate":"had","object":"customers",'
        '"quote":"An independent filing confirms FTX filed for Chapter 11."}]}'
    )
    active = runner(response(payload))

    triples = asyncio.run(
        active.extract_support(
            document=source,
            slot=SLOT,
            targets=[target],
            topic="FTX collapse",
        )
    )

    assert len(triples) == 1
    assert active._triple_key(triples[0]) == active._triple_key(target)
    assert active.usage.support_rows_rejected == 1
    system_prompt = active.llm.chat.completions.calls[0]["messages"][0]["content"]
    assert "json" in system_prompt.casefold()


def test_conditional_slot_can_be_audited_as_not_applicable() -> None:
    conditional = OntologySlot(
        slot_id="where.asset_flow",
        dimension="WHERE",
        label="Asset flow",
        question="Where did assets move?",
        applicability="conditional",
    )
    verdict = (
        '{"slots":[{"slot_id":"where.asset_flow",'
        '"status":"not_applicable","confidence":0.95,'
        '"reason":"an IT outage has no asset movement"}]}'
    )
    active = runner(
        response(verdict),
        settings=GraphResearchSettings(model="test/model"),
    )

    decisions = asyncio.run(
        active.classify_slot_applicability(
            topic="CrowdStrike IT outage",
            slots=[conditional],
        )
    )

    assert (
        decisions["where.asset_flow"].status
        is SlotApplicabilityStatus.NOT_APPLICABLE
    )
    assert active.usage.not_applicable_slots == 1
