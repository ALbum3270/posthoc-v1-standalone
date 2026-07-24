"""Ontology definitions and helpers for graph-driven research coverage."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OntologySlot(BaseModel):
    """A single research slot that can be filled by evidence from the graph."""

    model_config = ConfigDict(extra="forbid")

    slot_id: str
    dimension: Literal["WHO", "WHAT", "WHEN", "WHERE", "WHY", "HOW"]
    label: str
    question: str
    priority: int = Field(default=50, ge=0, le=100)
    dynamic: bool = False
    aliases: list[str] = Field(default_factory=list)
    applicability: Literal["always", "conditional"] = "always"
    required_source_count: int = Field(default=1, ge=1, le=5)
    max_initial_claims: int | None = Field(default=None, ge=1, le=5)


INVESTIGATION_SCHEMA: dict[str, tuple[OntologySlot, ...]] = {
    "WHO": (
        OntologySlot(
            slot_id="who.primary_actor",
            dimension="WHO",
            label="Primary Actor",
            question=(
                "Which person, organization, or system is the central subject "
                "of the event, excluding later responders, buyers, and merely "
                "affected parties?"
            ),
            priority=100,
            aliases=["main actor", "primary subject"],
            required_source_count=2,
        ),
        OntologySlot(
            slot_id="who.affected_parties",
            dimension="WHO",
            label="Affected Parties",
            question="Who is affected, targeted, or impacted?",
            priority=80,
            aliases=["victims", "impacted groups"],
        ),
        OntologySlot(
            slot_id="who.related_organizations",
            dimension="WHO",
            label="Related Organizations",
            question="Which organizations, subsidiaries, or partners are involved?",
            priority=70,
        ),
        OntologySlot(
            slot_id="who.regulators",
            dimension="WHO",
            label="Regulators",
            question="Which regulators, investigators, or authorities are involved?",
            priority=60,
            applicability="conditional",
        ),
    ),
    "WHAT": (
        OntologySlot(
            slot_id="what.core_event",
            dimension="WHAT",
            label="Core Event",
            question="What exactly happened?",
            priority=100,
            required_source_count=2,
        ),
        OntologySlot(
            slot_id="what.scale",
            dimension="WHAT",
            label="Scale",
            question="What is the scale, size, or material impact of the event?",
            priority=80,
            required_source_count=2,
        ),
        OntologySlot(
            slot_id="what.products",
            dimension="WHAT",
            label="Products",
            question="Which products, services, or assets are involved?",
            priority=65,
            applicability="conditional",
        ),
    ),
    "WHEN": (
        OntologySlot(
            slot_id="when.event_time",
            dimension="WHEN",
            label="Event Time",
            question="When did the key event occur?",
            priority=90,
            required_source_count=2,
        ),
        OntologySlot(
            slot_id="when.discovery_time",
            dimension="WHEN",
            label="Discovery Time",
            question="When was the event discovered or disclosed?",
            priority=75,
            applicability="conditional",
        ),
        OntologySlot(
            slot_id="when.intervention_time",
            dimension="WHEN",
            label="Intervention Time",
            question="When did an official intervention or response occur?",
            priority=60,
            applicability="conditional",
        ),
    ),
    "WHERE": (
        OntologySlot(
            slot_id="where.jurisdiction",
            dimension="WHERE",
            label="Jurisdiction",
            question="Where is the primary legal or operational jurisdiction?",
            priority=75,
            applicability="conditional",
        ),
        OntologySlot(
            slot_id="where.asset_flow",
            dimension="WHERE",
            label="Asset Flow",
            question="Where did money, assets, or activity move?",
            priority=65,
            applicability="conditional",
        ),
        OntologySlot(
            slot_id="where.event_location",
            dimension="WHERE",
            label="Event Location",
            question="Where did the event happen or become visible?",
            priority=55,
        ),
    ),
    "WHY": (
        OntologySlot(
            slot_id="why.motivation",
            dimension="WHY",
            label="Motivation",
            question="Why did the actor behave this way or pursue this action?",
            priority=60,
            applicability="conditional",
        ),
        OntologySlot(
            slot_id="why.trigger",
            dimension="WHY",
            label="Trigger",
            question=(
                "What immediate trigger and underlying conditions caused or "
                "catalyzed the event?"
            ),
            priority=55,
            required_source_count=2,
            max_initial_claims=2,
        ),
    ),
    "HOW": (
        OntologySlot(
            slot_id="how.mechanism",
            dimension="HOW",
            label="Mechanism",
            question=(
                "What causal mechanism made the event happen, or how was it "
                "executed?"
            ),
            priority=70,
            required_source_count=2,
        ),
        OntologySlot(
            slot_id="how.sequence",
            dimension="HOW",
            label="Sequence",
            question="How did the activity unfold step by step?",
            priority=55,
        ),
    ),
}


def iter_slots(
    schema: dict[str, tuple[OntologySlot, ...]] | None = None,
) -> list[OntologySlot]:
    """Return a flat list of ontology slots."""

    active_schema = schema or INVESTIGATION_SCHEMA
    return [slot for slots in active_schema.values() for slot in slots]


def get_slot(
    slot_id: str,
    schema: dict[str, tuple[OntologySlot, ...]] | None = None,
) -> OntologySlot | None:
    """Look up one ontology slot by id."""

    for slot in iter_slots(schema):
        if slot.slot_id == slot_id:
            return slot
    return None


def compute_coverage_ratio(
    filled_slot_ids: Iterable[str],
    schema: dict[str, tuple[OntologySlot, ...]] | None = None,
) -> float:
    """Compute ontology coverage as filled slots over total slots."""

    slots = iter_slots(schema)
    if not slots:
        return 0.0

    filled = {slot_id for slot_id in filled_slot_ids}
    total_slots = len(slots)
    filled_count = sum(1 for slot in slots if slot.slot_id in filled)
    return filled_count / total_slots


def get_open_slots(
    filled_slot_ids: Iterable[str],
    schema: dict[str, tuple[OntologySlot, ...]] | None = None,
) -> list[OntologySlot]:
    """Return unfilled slots ordered by highest priority first."""

    filled = {slot_id for slot_id in filled_slot_ids}
    open_slots = [
        slot for slot in iter_slots(schema) if slot.slot_id not in filled
    ]
    return sorted(open_slots, key=lambda slot: (-slot.priority, slot.slot_id))


def extend_ontology(
    additions: Iterable[OntologySlot],
    schema: dict[str, tuple[OntologySlot, ...]] | None = None,
) -> dict[str, tuple[OntologySlot, ...]]:
    """Return a copied schema with new slots appended by dimension.

    Existing slot ids are preserved and will not be duplicated.
    """

    base_schema = deepcopy(schema or INVESTIGATION_SCHEMA)
    seen_slot_ids = {slot.slot_id for slot in iter_slots(base_schema)}

    for slot in additions:
        if slot.slot_id in seen_slot_ids:
            continue
        base_schema.setdefault(slot.dimension, tuple())
        base_schema[slot.dimension] = (*base_schema[slot.dimension], slot)
        seen_slot_ids.add(slot.slot_id)

    return base_schema
