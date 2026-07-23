from types import SimpleNamespace

from open_deep_research.graphrag.extraction.targeted_extractor import (
    TargetedExtractionConfig,
    TargetedExtractor,
)
from open_deep_research.graphrag.schemas import SourceDocument, SourceType


class FakeStructuredModel:
    def __init__(self, response):
        self.response = response
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        return self.response


def test_targeted_extractor_maps_structured_output_to_claims():
    fake_response = SimpleNamespace(
        claims=[
            SimpleNamespace(
                slot_id="who.primary_actor",
                claim_text="The primary actor is ACME Corp.",
                subject_name="ACME Corp",
                subject_type="Organization",
                predicate="is_primary_actor",
                object_value="investigation",
                object_type="Literal",
                confidence=0.92,
                rationale="Explicitly stated in the notice.",
                quote="ACME Corp led the operation",
                start_char=15,
                end_char=42,
            ),
            SimpleNamespace(
                slot_id="when.event_time",
                claim_text="The event happened on 2026-03-01.",
                subject_name="Event",
                subject_type="Event",
                predicate="occurred_on",
                object_value="2026-03-01",
                object_type="Literal",
                confidence=0.89,
                rationale=None,
                quote=None,
                start_char=60,
                end_char=70,
            ),
        ]
    )
    model = FakeStructuredModel(fake_response)
    extractor = TargetedExtractor(
        TargetedExtractionConfig(max_chars=80),
        model=model,
    )
    document = SourceDocument(
        document_id="doc-1",
        title="Official Filing",
        source_type=SourceType.OFFICIAL,
        content="A" * 200,
        url="https://example.com/filing",
    )

    claims = __import__("asyncio").run(
        extractor.extract(
            topic="ACME investigation",
            document=document,
            pending_questions=["who.primary_actor", "when.event_time"],
        )
    )

    assert len(claims) == 2
    assert claims[0].slot_id == "who.primary_actor"
    assert claims[0].triples[0].subject.entity_type == "Organization"
    assert claims[1].triples[0].object == "2026-03-01"
    assert len(model.messages[1].content.split('"""')[1]) == 80


def test_targeted_extractor_ignores_unrequested_slots_and_unknown_questions():
    fake_response = SimpleNamespace(
        claims=[
            SimpleNamespace(
                slot_id="unknown.slot",
                claim_text="Ignore this.",
                subject_name="X",
                subject_type="Entity",
                predicate="ignored",
                object_value="Y",
                object_type="Entity",
                confidence=0.5,
                rationale=None,
                quote=None,
                start_char=None,
                end_char=None,
            )
        ]
    )
    model = FakeStructuredModel(fake_response)
    extractor = TargetedExtractor(TargetedExtractionConfig(), model=model)
    document = SourceDocument(
        document_id="doc-2",
        title="News",
        source_type=SourceType.NEWS,
        content="content",
    )

    claims = __import__("asyncio").run(
        extractor.extract(
            topic="topic",
            document=document,
            pending_questions=["Who is the main actor or primary subject involved?", "not-a-slot"],
        )
    )

    assert claims == []
