from open_deep_research.graphrag.ontology import (
    INVESTIGATION_SCHEMA,
    OntologySlot,
    compute_coverage_ratio,
    extend_ontology,
    get_open_slots,
)
from open_deep_research.graphrag.schemas import (
    ClaimVerificationResult,
    EvidencePack,
    ExtractedClaim,
    SourceDocument,
    SourceType,
    VerificationStatus,
)


def test_compute_coverage_ratio_uses_slot_ids():
    filled = {
        "who.primary_actor",
        "what.core_event",
        "when.event_time",
    }

    ratio = compute_coverage_ratio(filled, INVESTIGATION_SCHEMA)

    assert ratio == 3 / 17


def test_get_open_slots_orders_by_priority():
    open_slots = get_open_slots({"what.core_event"}, INVESTIGATION_SCHEMA)

    assert open_slots[0].slot_id == "who.primary_actor"
    assert all(slot.slot_id != "what.core_event" for slot in open_slots)


def test_extend_ontology_appends_dynamic_slot_without_mutating_default():
    custom_slot = OntologySlot(
        slot_id="why.hidden_incentive",
        dimension="WHY",
        label="Hidden Incentive",
        question="What hidden incentive emerged during research?",
        priority=72,
        dynamic=True,
    )

    extended = extend_ontology([custom_slot], INVESTIGATION_SCHEMA)

    assert any(slot.slot_id == custom_slot.slot_id for slot in extended["WHY"])
    assert all(slot.slot_id != custom_slot.slot_id for slot in INVESTIGATION_SCHEMA["WHY"])


def test_graphrag_schemas_validate_and_expose_graph_write_decision():
    document = SourceDocument(
        document_id="doc-1",
        title="Official Notice",
        source_type=SourceType.OFFICIAL,
        content="Regulator confirms timeline and scope.",
    )
    claim = ExtractedClaim(
        claim_id="claim-1",
        slot_id="when.event_time",
        text="The event occurred on March 1.",
        source_document_id=document.document_id,
        confidence=0.84,
    )
    verdict = ClaimVerificationResult(
        claim_id=claim.claim_id,
        status=VerificationStatus.PASSED,
        confidence_score=0.81,
        truth_score=0.88,
    )
    pack = EvidencePack(topic="Example topic")

    assert claim.source_document_id == "doc-1"
    assert verdict.allows_graph_write is True
    assert pack.coverage_ratio == 0.0
