"""Model-generated, membership-frozen research checklists."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Mapping
from enum import Enum
from typing import Any, Protocol

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from open_deep_research.harness.jsonio import loads_lenient


class ChecklistStatus(str, Enum):
    """Lifecycle states for one research question."""

    UNEXPLORED = "unexplored"
    HAS_MATERIAL = "has_material"
    SETTLED = "settled"
    EXHAUSTED_NOT_FOUND = "exhausted_not_found"

    @property
    def is_terminal(self) -> bool:
        """Return whether no more collection is required for this state."""

        return self in {
            ChecklistStatus.SETTLED,
            ChecklistStatus.EXHAUSTED_NOT_FOUND,
        }

    @property
    def is_failure(self) -> bool:
        """Return whether the state represents a failed checklist item."""

        # Exhausting a reasonable search without finding material is an honest,
        # completed outcome, not a failed run.
        return False


class ChecklistDimension(str, Enum):
    """The topic-neutral 5W1H dimensions used to structure a checklist."""

    WHO = "who"
    WHAT = "what"
    WHEN = "when"
    WHERE = "where"
    WHY = "why"
    HOW = "how"


class ChecklistItem(BaseModel):
    """One immutable question in a research checklist."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1)
    dimension: ChecklistDimension
    question: str = Field(min_length=1)
    priority: int = Field(ge=1)
    corroboration_target: int = Field(
        ge=1,
        le=2,
        validation_alias=AliasChoices(
            "corroboration_target",
            "required_source_count",
        ),
    )
    status: ChecklistStatus = ChecklistStatus.UNEXPLORED

    @property
    def required_source_count(self) -> int:
        """Read legacy code without emitting the obsolete audit field."""

        return self.corroboration_target

    @property
    def is_complete(self) -> bool:
        """Return whether the item is in either legitimate terminal state."""

        return self.status.is_terminal

    @property
    def is_failed(self) -> bool:
        """Return whether the item is a failure."""

        return self.status.is_failure


class ChecklistAuditSink(Protocol):
    """The ledger operation required by checklist mutations."""

    def record_checklist_change(
        self,
        *,
        event: str,
        item_id: str,
        accepted: bool,
        reason: str,
        from_status: str | None = None,
        to_status: str | None = None,
        settlement_evidence: Mapping[str, Any] | None = None,
    ) -> Any:
        """Record one accepted or rejected membership/status request."""


class ResearchChecklist(BaseModel):
    """A checklist whose initial membership cannot be deleted or rewritten."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    topic: str = Field(min_length=1)
    items: tuple[ChecklistItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _item_ids_are_unique(self) -> ResearchChecklist:
        ids = [item.item_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("checklist item_id values must be unique")
        return self

    @property
    def is_complete(self) -> bool:
        """Return whether every item has reached a legitimate terminal state."""

        return all(item.is_complete for item in self.items)

    @property
    def has_failures(self) -> bool:
        """Return whether any checklist item represents failure."""

        return any(item.is_failed for item in self.items)

    def get(self, item_id: str) -> ChecklistItem:
        """Return an item by id."""

        for item in self.items:
            if item.item_id == item_id:
                return item
        raise KeyError(item_id)

    def append_item(
        self,
        item: ChecklistItem,
        *,
        reason: str,
        ledger: ChecklistAuditSink,
    ) -> ResearchChecklist:
        """Return a checklist with one audited, reasoned addition."""

        normalized_reason = reason.strip()
        if not normalized_reason:
            raise ValueError("appending a checklist item requires a reason")
        if any(existing.item_id == item.item_id for existing in self.items):
            raise ValueError(f"duplicate checklist item_id: {item.item_id}")

        ledger.record_checklist_change(
            event="append",
            item_id=item.item_id,
            accepted=True,
            reason=normalized_reason,
            to_status=item.status.value,
        )
        return self.model_copy(update={"items": (*self.items, item)})

    def request_delete(
        self,
        item_id: str,
        *,
        reason: str,
        ledger: ChecklistAuditSink,
    ) -> bool:
        """Reject and audit every deletion request after generation."""

        item = next(
            (candidate for candidate in self.items if candidate.item_id == item_id),
            None,
        )
        ledger.record_checklist_change(
            event="delete",
            item_id=item_id,
            accepted=False,
            reason=reason.strip(),
            from_status=item.status.value if item is not None else None,
        )
        return False

    def set_status(
        self,
        item_id: str,
        status: ChecklistStatus | str,
        *,
        reason: str,
        ledger: ChecklistAuditSink,
        settlement_evidence: Mapping[str, Any] | None = None,
    ) -> ResearchChecklist:
        """Return a copy with one audited status update."""

        current = self.get(item_id)
        target_status = ChecklistStatus(status)
        replacement = current.model_copy(update={"status": target_status})
        items = tuple(
            replacement if item.item_id == item_id else item for item in self.items
        )
        audit_fields: dict[str, Any] = {
            "event": "status_change",
            "item_id": item_id,
            "accepted": True,
            "reason": reason.strip(),
            "from_status": current.status.value,
            "to_status": target_status.value,
        }
        if settlement_evidence is not None:
            audit_fields["settlement_evidence"] = settlement_evidence
        ledger.record_checklist_change(
            **audit_fields,
        )
        return ResearchChecklist(topic=self.topic, items=items)


class ChecklistModelClient(Protocol):
    """Injected model boundary used only for initial checklist generation."""

    def generate(self, prompt: str) -> str | Mapping[str, Any] | Awaitable[Any]:
        """Return the checklist JSON payload for a prompt."""


CHECKLIST_PROMPT = """\
Create a research checklist for the supplied topic.

Use all six topic-neutral dimensions: who, what, when, where, why, and how.
Write concrete questions that together support a thorough investigation.
Set priority to a positive integer where a smaller number means earlier work.
Set corroboration_target to 1 or 2. It is a later evidence-gap resource
priority signal, not a truth threshold or a report-quality label.
Return JSON only, in this shape:
{{"items":[{{"item_id":"unique-id","dimension":"who|what|when|where|why|how",\
"question":"...","priority":1,"corroboration_target":1}}]}}

Topic:
{topic}
"""


def build_checklist_prompt(topic: str) -> str:
    """Build the topic-neutral checklist-generation prompt."""

    normalized_topic = topic.strip()
    if not normalized_topic:
        raise ValueError("topic must not be blank")
    return CHECKLIST_PROMPT.format(topic=normalized_topic)


def _decode_model_payload(payload: str | Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        return payload
    if not isinstance(payload, str):
        content = getattr(payload, "content", None)
        if isinstance(content, str):
            payload = content
        else:
            raise TypeError("checklist model must return JSON text or a mapping")
    decoded = loads_lenient(payload)
    if not isinstance(decoded, Mapping):
        raise ValueError("checklist model response must be a JSON object")
    return decoded


async def generate_checklist(
    topic: str,
    *,
    model_client: ChecklistModelClient,
) -> ResearchChecklist:
    """Ask an injected model for a checklist and freeze the parsed result."""

    normalized_topic = topic.strip()
    prompt = build_checklist_prompt(normalized_topic)
    response = model_client.generate(prompt)
    if inspect.isawaitable(response):
        response = await response
    payload = _decode_model_payload(response)
    return ResearchChecklist(topic=normalized_topic, items=payload.get("items", ()))
