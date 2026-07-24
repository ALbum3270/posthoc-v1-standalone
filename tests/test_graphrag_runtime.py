import asyncio
from types import SimpleNamespace

from open_deep_research.graphrag.ontology import OntologySlot
from open_deep_research.graphrag.runtime import (
    GraphResearchRunner,
    GraphResearchSettings,
    GraphResearchUsage,
)
from open_deep_research.graphrag.schemas import SourceDocument


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
    async def search(self, *args, **kwargs):
        return {"results": []}


def runner(*responses) -> GraphResearchRunner:
    return GraphResearchRunner(
        graphiti=SimpleNamespace(),
        llm=FakeLLM(responses),
        tavily=FakeTavily(),
        settings=GraphResearchSettings(model="test/model"),
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


def test_extraction_prompt_requires_a_verbatim_quote() -> None:
    active = runner(response('{"triples":[]}'))
    source = SourceDocument(document_id="doc", title="Source", content="body")

    asyncio.run(active.extract(document=source, slot=SLOT))

    system_prompt = active.llm.chat.completions.calls[0]["messages"][0]["content"]
    assert "quote is mandatory" in system_prompt.lower()
    assert "verbatim" in system_prompt.lower()
