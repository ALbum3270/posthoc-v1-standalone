"""Model-directed collection loop with deterministic budgets and bookkeeping."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from open_deep_research.harness.checklist import (
    ChecklistStatus,
    ResearchChecklist,
)
from open_deep_research.harness.jsonio import loads_lenient
from open_deep_research.harness.ledger import (
    ExhaustionAttemptSnapshot,
    ResearchLedger,
    SettlementEvidence,
)
from open_deep_research.harness.notes import (
    NoteLocationStatus,
    QuoteFailureReason,
    ResearchNote,
    create_note,
)
from open_deep_research.harness.tools import (
    SearchResult,
    TavilyClient,
    read,
    search,
)


class StopReason(str, Enum):
    """Mutually exclusive reasons why collection stopped."""

    ALL_ITEMS_TERMINAL = "all_items_terminal"
    MODEL_STOP_WITH_OPEN_ITEMS = "model_stop_with_open_items"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MALFORMED_ACTION_LIMIT = "malformed_action_limit"


class LoopBudget(BaseModel):
    """Hard limits enforced by the collection loop."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_rounds: int = Field(default=25, ge=0)
    max_tokens: int = Field(default=100_000, ge=0)
    max_cost_usd: float = Field(default=10.0, ge=0.0)
    writing_token_reserve: int = Field(default=0, ge=0)
    writing_cost_reserve_usd: float = Field(default=0.0, ge=0.0)
    max_consecutive_malformed_actions: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _writing_reserve_fits_total_budget(self) -> LoopBudget:
        if self.writing_token_reserve > self.max_tokens:
            raise ValueError("writing_token_reserve must not exceed max_tokens")
        if self.writing_cost_reserve_usd > self.max_cost_usd:
            raise ValueError(
                "writing_cost_reserve_usd must not exceed max_cost_usd"
            )
        return self

    @property
    def collection_token_limit(self) -> int:
        """Tokens collection may use without consuming the writing reserve."""

        return self.max_tokens - self.writing_token_reserve

    @property
    def collection_cost_limit_usd(self) -> float:
        """Cost collection may use without consuming the writing reserve."""

        return self.max_cost_usd - self.writing_cost_reserve_usd


class LoopSettings(BaseModel):
    """Budget for source text the decision model asked to see.

    Whether a source body enters the decision context is the model's call, made
    by returning a ``recall`` action. Code only caps how much can be injected,
    which is a budget concern rather than a judgement about relevance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_source_char_limit: int = Field(default=100_000, ge=0)
    note_page_size: int = Field(default=8, ge=1, le=50)
    max_recalled_notes: int = Field(default=8, ge=1, le=50)
    # This is a capacity bound on cross-item discovery, not an evidence-quality
    # threshold. Keep it fixed during the first comparison run.
    max_cross_item_seeds: int = Field(default=3, ge=0, le=20)


class ModelEnvelope(BaseModel):
    """One model output plus its measured usage."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    content: Any
    token_count: int = Field(ge=0)
    cost_usd: float = Field(ge=0.0)


class _ItemAction(BaseModel):
    """Shared fields for actions attributed to one checklist item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1)

    @field_validator("item_id")
    @classmethod
    def _item_id_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("item_id must not be blank")
        return normalized


class SearchAction(_ItemAction):
    """Ask the search tool for candidate sources."""

    action: Literal["search"]
    query: str = Field(min_length=1)

    @field_validator("query")
    @classmethod
    def _query_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


class ReadAction(_ItemAction):
    """Read one source, then ask the note model to inspect it."""

    action: Literal["read"]
    url: str = Field(min_length=1)

    @field_validator("url")
    @classmethod
    def _url_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("url must not be blank")
        return normalized


class ReanalyzeAction(_ItemAction):
    """Explicitly rerun note extraction over an already cached source."""

    action: Literal["reanalyze"]
    url: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("url", "reason")
    @classmethod
    def _required_text_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("reanalyze url and reason must not be blank")
        return normalized


class CandidateDismissal(BaseModel):
    """One explicit, reasoned rejection of a surfaced candidate URL."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("url", "reason")
    @classmethod
    def _required_text_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("candidate url and reason must not be blank")
        return normalized


class DismissCandidatesAction(_ItemAction):
    """Explicitly reject pending candidates without judging them in code."""

    action: Literal["dismiss_candidates"]
    candidates: tuple[CandidateDismissal, ...] = Field(
        min_length=1,
        max_length=50,
    )

    @model_validator(mode="after")
    def _candidate_urls_are_unique(self) -> DismissCandidatesAction:
        urls = [candidate.url for candidate in self.candidates]
        if len(urls) != len(set(urls)):
            raise ValueError("dismiss_candidates must not repeat a URL")
        return self


class RecallAction(_ItemAction):
    """Ask for an already-read source body in the next decision context.

    Note summaries are the default view because they are cheap, not because the
    model is untrusted with the original text. When summaries are not enough the
    model recalls the source itself; code only enforces the character budget.
    """

    action: Literal["recall"]
    url: str = Field(min_length=1)

    @field_validator("url")
    @classmethod
    def _url_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("url must not be blank")
        return normalized


class InspectNotesAction(_ItemAction):
    """Page through compact note summaries for one checklist item."""

    action: Literal["inspect_notes"]
    cursor: str | None = None

    @field_validator("cursor")
    @classmethod
    def _cursor_is_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("cursor must be null or a non-blank string")
        return normalized


class RecallNotesAction(_ItemAction):
    """Recall full details for explicitly selected stable note IDs."""

    action: Literal["recall_notes"]
    note_ids: tuple[str, ...] = Field(min_length=1, max_length=50)

    @field_validator("note_ids")
    @classmethod
    def _note_ids_are_unique_and_nonblank(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized = tuple(note_id.strip() for note_id in value)
        if any(not note_id for note_id in normalized):
            raise ValueError("note_ids must not contain blank values")
        if len(normalized) != len(set(normalized)):
            raise ValueError("note_ids must be unique")
        return normalized


class SettleAction(_ItemAction):
    """Mark one checklist item as sufficiently researched."""

    action: Literal["settle"]


class MarkExhaustedAction(_ItemAction):
    """Mark an honestly searched but unanswered item as complete."""

    action: Literal["mark_exhausted"]
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def _reason_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("reason must not be blank")
        return normalized


class StopAction(BaseModel):
    """Request collection to stop without claiming successful completion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["stop"]


LoopAction = Annotated[
    SearchAction
    | ReadAction
    | ReanalyzeAction
    | DismissCandidatesAction
    | RecallAction
    | InspectNotesAction
    | RecallNotesAction
    | SettleAction
    | MarkExhaustedAction
    | StopAction,
    Field(discriminator="action"),
]
_ACTION_ADAPTER = TypeAdapter(LoopAction)

PrimaryAction = Annotated[
    SearchAction
    | ReadAction
    | ReanalyzeAction
    | DismissCandidatesAction
    | RecallAction
    | InspectNotesAction
    | RecallNotesAction
    | StopAction,
    Field(discriminator="action"),
]
_PRIMARY_ACTION_ADAPTER = TypeAdapter(PrimaryAction)


class StatusUpdate(BaseModel):
    """One independently reasoned terminal-state judgement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1)
    status: Literal["settled", "exhausted_not_found"]
    reason: str = Field(min_length=1)

    @field_validator("item_id", "reason")
    @classmethod
    def _required_text_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("status update item_id and reason must not be blank")
        return normalized


class DecisionTurn(BaseModel):
    """Any number of terminal updates plus at most one next action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status_updates: tuple[StatusUpdate, ...] = ()
    action: PrimaryAction | None = None

    @model_validator(mode="after")
    def _item_updates_are_unique(self) -> DecisionTurn:
        item_ids = [update.item_id for update in self.status_updates]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("status_updates must not repeat an item_id")
        if not self.status_updates and self.action is None:
            raise ValueError("a decision needs a status update or an action")
        return self


@dataclass(frozen=True)
class _DecisionParse:
    """Executable pieces plus independently rejected model output."""

    turn: DecisionTurn | None
    error: str | None = None
    rejected_status_updates: tuple[dict[str, Any], ...] = ()
    rejected_action: dict[str, Any] | None = None


class NoteDraft(BaseModel):
    """A note proposed by the note model for mechanical grounding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1)
    finding: str = Field(min_length=1)
    quote: str

    @field_validator("item_id", "finding")
    @classmethod
    def _required_text_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("required note text must not be blank")
        return normalized


@dataclass(frozen=True)
class _NoteParse:
    """Independently parsed active notes and bounded cross-item seeds."""

    active_notes: tuple[NoteDraft, ...] = ()
    cross_item_seeds: tuple[NoteDraft, ...] = ()
    active_errors: tuple[str, ...] = ()
    cross_errors: tuple[str, ...] = ()
    error: str | None = None


class LoopModelClient(Protocol):
    """Injected decision or note-generation model boundary."""

    def generate(self, prompt: str) -> Any | Awaitable[Any]:
        """Return a usage envelope."""


class LoopResult(BaseModel):
    """Final deterministic collection state."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    checklist: ResearchChecklist
    ledger: ResearchLedger
    stop_reason: StopReason
    stop_detail: str = ""
    open_item_ids: tuple[str, ...] = ()
    rounds_executed: int = Field(ge=0)
    consecutive_collection_failures: dict[str, int]

    @property
    def consecutive_failures(self) -> dict[str, int]:
        """Backward-compatible access to the renamed collection-failure memory."""

        return self.consecutive_collection_failures

    @property
    def is_success(self) -> bool:
        """Only terminal checklist completion counts as success."""

        return self.stop_reason is StopReason.ALL_ITEMS_TERMINAL


DECISION_PROMPT = """\
Choose terminal status updates and at most one next action for this research run.

Return this JSON shape:
{{"status_updates":[
  {{"item_id":"...","status":"settled","reason":"..."}},
  {{"item_id":"...","status":"exhausted_not_found","reason":"..."}}
],"action":{{"action":"search|read|reanalyze|dismiss_candidates|recall|stop", ...}}}}

status_updates may contain any number of distinct checklist items. Give every
update its own reason. status_updates accepts terminal judgements only:
"settled" or "exhausted_not_found". Never put "unexplored" or "has_material"
in status_updates; the system maintains those non-terminal states.

"not_attempted" and "attempted_no_result" are different. Use
"exhausted_not_found" only after at least one real search, read, or reanalyze
attempt attributed to that item. A zero-result search or a tool error is still
an attempt; an action rejected before the tool runs is not. Code enforces this
mechanically and rejects only the ineligible status entry.

The optional action is one of:
{{"action":"search","item_id":"...","query":"..."}}
{{"action":"read","item_id":"...","url":"..."}}
{{"action":"reanalyze","item_id":"...","url":"...","reason":"..."}}
{{"action":"dismiss_candidates","item_id":"...","candidates":[
  {{"url":"...","reason":"..."}}
]}}
{{"action":"recall","item_id":"...","url":"..."}}
{{"action":"inspect_notes","item_id":"...","cursor":null}}
{{"action":"recall_notes","item_id":"...","note_ids":["note-000001"]}}
{{"action":"stop"}}

The action item_id identifies the checklist item this round is working on.

search_candidates holds every url search has surfaced so far, with its snippet
and whether you already read it. Searching only adds candidates; reading one is
what produces notes. Prefer reading a promising unread candidate over searching
again for the same thing.

candidate_work shows, per checklist item, which surfaced URLs remain unread,
which were read, which failed mechanically and are now unreadable, and which
you explicitly dismissed with reasons. Never retry a URL listed as unreadable
for that item; choose another pending URL or search again after pending URLs are
resolved. If an item has pending unread candidates, another search for that
same item is rejected. Read them or explicitly dismiss them first. Dismissal is
not a claim that a source is bad; it records your reason for not using that
candidate.

read fetches and analyzes a URL only once. Reading a cached URL returns its
existing note IDs without rerunning note extraction. Use reanalyze, with a
reason, when another extraction pass over cached text is warranted.

You normally see note summaries rather than source bodies. When a summary is not
enough, recall an already-read url and its full text joins the next round's
state. Recall as often as you need; the oldest recalled sources drop out first
once the character budget is reached.

The default state contains only a compact note index. Use inspect_notes to page
through one item's note summaries. Use the returned next_cursor to continue.
Use recall_notes when you need full quote and span details for selected note IDs.

Return JSON only.

Current collection state:
{state}
"""


NOTE_PROMPT = """\
Read the single source below once and perform two distinct tasks in order.

1. active_notes: Extract evidence for the active checklist item. Every entry's
item_id must equal the active item_id. This remains the complete active-item
extraction pass.
2. cross_item_seeds: After the active pass, scan for other non-terminal
checklist items that this source directly answers. Select at most
{max_cross_item_seeds} different items and return at most one seed note for
each. A seed only makes useful cross-item material visible; it is not a full
extraction for that item. Return an empty list when no other item is directly
answered.

The cross-item limit is a fixed output-capacity bound, not a quality threshold.
Only item_id values listed under eligible_cross_item_targets may appear in
cross_item_seeds. Do not put the active item there.

Each quote must be one continuous passage copied verbatim from the source:
- Do not use an ellipsis ("..." or "…").
- Do not join text from separate passages or paragraphs.
- Do not reorder words or clauses.
- Do not change wording, whitespace, capitalization, or punctuation.
- If one finding needs two non-contiguous passages, return two separate notes,
  each with its own continuous verbatim quote.

Purely structural example:
Source text contains "First statement." and later "Second statement."
Valid: two notes, one quoting "First statement." and one quoting
"Second statement."
Invalid: one note quoting "First statement. ... Second statement."

Return JSON only as
{{"active_notes":[
  {{"item_id":"...","finding":"...","quote":"exact source text"}}
],"cross_item_seeds":[
  {{"item_id":"...","finding":"...","quote":"exact source text"}}
]}}.
Returning two empty lists is valid when the source answers nothing.

Active item:
{active_item}

Eligible cross-item targets:
{eligible_cross_item_targets}

Source URL:
{url}

Source text:
{source_text}
"""


_MAX_CANDIDATES_IN_CONTEXT = 40
_CANDIDATE_SNIPPET_CHARS = 400


def _candidate_state(
    candidates: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose the urls search surfaced so the model can choose what to read.

    Without this the model searches, never sees the hits, and searches again --
    the zero-entropy loop this project already measured once.
    """

    recent = list(candidates.values())[-_MAX_CANDIDATES_IN_CONTEXT:]
    recent.reverse()
    return [
        {
            "url": candidate["url"],
            "title": candidate["title"],
            "snippet": candidate["snippet"],
            "score": candidate["score"],
            "read": candidate["read"],
            "pending_for_item_ids": sorted(
                item_id
                for item_id, work in candidate.get("_item_work", {}).items()
                if work["pending"]
            ),
            "unreadable_for_item_ids": sorted(
                item_id
                for item_id, work in candidate.get("_item_work", {}).items()
                if work.get("unreadable_error") is not None
            ),
        }
        for candidate in recent
    ]


def _remember_candidates(
    candidates: dict[str, dict[str, Any]],
    results: Sequence[SearchResult],
    *,
    item_id: str,
) -> list[str]:
    newly_pending: list[str] = []
    for result in results:
        existing = candidates.get(result.url)
        if existing is None:
            existing = {
                "url": result.url,
                "title": result.title,
                "snippet": result.snippet[:_CANDIDATE_SNIPPET_CHARS],
                "score": result.score,
                "read": False,
                "_item_work": {},
            }
            candidates[result.url] = existing
        work = existing["_item_work"].get(item_id)
        if work is None:
            pending = not existing["read"]
            existing["_item_work"][item_id] = {
                "pending": pending,
                "read": existing["read"],
                "dismiss_reason": None,
                "unreadable_error": None,
            }
            if pending:
                newly_pending.append(result.url)
    return newly_pending


def _pending_candidate_urls(
    candidates: Mapping[str, dict[str, Any]],
    item_id: str,
) -> list[str]:
    return [
        url
        for url, candidate in candidates.items()
        if (
            item_id in candidate.get("_item_work", {})
            and candidate["_item_work"][item_id]["pending"]
            and not candidate["read"]
        )
    ]


def _mark_candidate_read(
    candidates: dict[str, dict[str, Any]],
    *,
    item_id: str,
    url: str,
) -> None:
    candidate = candidates.get(url)
    if candidate is None:
        candidate = {
            "url": url,
            "title": "",
            "snippet": "",
            "score": None,
            "read": False,
            "_item_work": {},
        }
        candidates[url] = candidate
    candidate["read"] = True
    work = candidate["_item_work"].setdefault(
        item_id,
        {
            "pending": False,
            "read": False,
            "dismiss_reason": None,
            "unreadable_error": None,
        },
    )
    work["read"] = True
    work["pending"] = False
    work["unreadable_error"] = None
    # A successful read resolves this URL for every item whose search surfaced
    # it. It does not claim the source was useful for any of those items.
    for linked_work in candidate["_item_work"].values():
        linked_work["read"] = True
        linked_work["pending"] = False
        linked_work["unreadable_error"] = None


def _mark_candidate_unreadable(
    candidates: dict[str, dict[str, Any]],
    *,
    item_id: str,
    url: str,
    error: str,
) -> None:
    """Consume one failed candidate without inventing a semantic dismissal."""

    candidate = candidates.get(url)
    if candidate is None:
        candidate = {
            "url": url,
            "title": "",
            "snippet": "",
            "score": None,
            "read": False,
            "_item_work": {},
        }
        candidates[url] = candidate
    work = candidate["_item_work"].setdefault(
        item_id,
        {
            "pending": False,
            "read": False,
            "dismiss_reason": None,
            "unreadable_error": None,
        },
    )
    work["pending"] = False
    work["read"] = False
    work["unreadable_error"] = error


def _candidate_unreadable_error(
    candidates: Mapping[str, dict[str, Any]],
    *,
    item_id: str,
    url: str,
) -> str | None:
    candidate = candidates.get(url)
    if candidate is None:
        return None
    work = candidate.get("_item_work", {}).get(item_id)
    if work is None:
        return None
    error = work.get("unreadable_error")
    return error if isinstance(error, str) and error else None


def _dismiss_candidate(
    candidates: dict[str, dict[str, Any]],
    *,
    item_id: str,
    url: str,
    reason: str,
) -> str | None:
    candidate = candidates.get(url)
    if candidate is None:
        return f"URL {url!r} was not surfaced by search"
    work = candidate["_item_work"].get(item_id)
    if work is None:
        return f"URL {url!r} was not surfaced for item {item_id!r}"
    if not work["pending"] or candidate["read"]:
        return f"URL {url!r} is not pending for item {item_id!r}"
    work["pending"] = False
    work["dismiss_reason"] = reason
    work["unreadable_error"] = None
    return None


def _candidate_work_state(
    candidates: Mapping[str, dict[str, Any]],
    item_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Expose mechanical candidate dispositions without ranking source quality."""

    state: dict[str, dict[str, Any]] = {}
    for item_id in item_ids:
        read_urls: list[str] = []
        pending_urls: list[str] = []
        dismissed: list[dict[str, str]] = []
        unreadable: list[dict[str, str]] = []
        for url, candidate in candidates.items():
            work = candidate.get("_item_work", {}).get(item_id)
            if work is None:
                continue
            if work["read"]:
                read_urls.append(url)
            if work["pending"] and not candidate["read"]:
                pending_urls.append(url)
            if work["dismiss_reason"] is not None:
                dismissed.append(
                    {
                        "url": url,
                        "reason": work["dismiss_reason"],
                    }
                )
            if work.get("unreadable_error") is not None:
                unreadable.append(
                    {
                        "url": url,
                        "error": work["unreadable_error"],
                    }
                )
        state[item_id] = {
            "read_count": len(read_urls),
            "read_urls": read_urls,
            "dismissed_count": len(dismissed),
            "dismissed_candidates": dismissed,
            "unreadable_count": len(unreadable),
            "unreadable_candidates": unreadable,
            "pending_unread_count": len(pending_urls),
            "pending_unread_urls": pending_urls,
            "candidates_pending": bool(pending_urls),
        }
    return state


def _new_acquisition_attempt_state(
    item_ids: Sequence[str],
) -> dict[str, dict[str, int]]:
    return {
        item_id: {
            "search_attempts": 0,
            "search_successes": 0,
            "search_errors": 0,
            "read_attempts": 0,
            "read_successes": 0,
            "read_errors": 0,
            "reanalyze_attempts": 0,
            "reanalyze_successes": 0,
            "reanalyze_errors": 0,
        }
        for item_id in item_ids
    }


def _record_acquisition_attempt(
    attempts: dict[str, dict[str, int]],
    *,
    item_id: str,
    action: Literal["search", "read", "reanalyze"],
    succeeded: bool,
) -> None:
    state = attempts[item_id]
    state[f"{action}_attempts"] += 1
    state[f"{action}_{'successes' if succeeded else 'errors'}"] += 1


def _acquisition_attempt_state(
    attempts: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, Any]]:
    """Expose attempts, not semantic source-quality judgements."""

    return {
        item_id: {
            **dict(values),
            "qualifying_attempts": (
                values["search_attempts"]
                + values["read_attempts"]
                + values["reanalyze_attempts"]
            ),
            "attempt_status": (
                "attempted"
                if (
                    values["search_attempts"]
                    + values["read_attempts"]
                    + values["reanalyze_attempts"]
                )
                else "not_attempted"
            ),
        }
        for item_id, values in attempts.items()
    }


def _note_summary(note: ResearchNote) -> dict[str, Any]:
    return {
        "note_id": note.note_id,
        "source_id": note.source_id,
        "item_id": note.item_id,
        "finding": note.finding,
        "model_quote": note.model_quote,
        "source_quote": note.source_quote,
        "url": note.url,
        "publisher": note.publisher,
        "location_status": note.location_status.value,
        "failure_reason": (
            note.failure_reason.value
            if note.failure_reason is not None
            else None
        ),
        "located_fragment_count": note.located_fragment_count,
    }


def _inspect_note_summary(note: ResearchNote) -> dict[str, Any]:
    """Return finding-level detail without injecting either quote body."""

    return {
        "note_id": note.note_id,
        "source_id": note.source_id,
        "item_id": note.item_id,
        "finding": note.finding,
        "url": note.url,
        "publisher": note.publisher,
        "location_status": note.location_status.value,
        "failure_reason": (
            note.failure_reason.value
            if note.failure_reason is not None
            else None
        ),
    }


def quote_quality_metrics(notes: Sequence[ResearchNote]) -> dict[str, Any]:
    """Return non-overlapping grounding rates over the supplied note set."""

    total = len(notes)
    strict = sum(
        note.location_status is NoteLocationStatus.LOCATABLE for note in notes
    )
    repaired = sum(
        note.location_status is NoteLocationStatus.REPAIRED_LOCATABLE
        for note in notes
    )
    composite = sum(
        note.failure_reason is QuoteFailureReason.NONCONTIGUOUS_COMPOSITE
        for note in notes
    )

    def rate(count: int) -> float:
        return round(count / total, 6) if total else 0.0

    return {
        "note_count": total,
        "strict_locatable_count": strict,
        "strict_locatable_rate": rate(strict),
        "repaired_locatable_count": repaired,
        "format_repair_rate": rate(repaired),
        "usable_source_span_count": strict + repaired,
        "usable_source_span_rate": rate(strict + repaired),
        "noncontiguous_composite_count": composite,
        "noncontiguous_composite_rate": rate(composite),
    }


def _compact_note_index(
    checklist: ResearchChecklist,
    notes: Sequence[ResearchNote],
) -> dict[str, Any]:
    """Summarize durable notes without turning the ledger into a prompt buffer."""

    by_item: list[dict[str, Any]] = []
    for item in checklist.items:
        item_notes = [note for note in notes if note.item_id == item.item_id]
        by_item.append(
            {
                "item_id": item.item_id,
                "note_count": len(item_notes),
                "publisher_count": len(
                    {note.publisher for note in item_notes}
                ),
                "source_count": len({note.source_id for note in item_notes}),
                "quote_quality": quote_quality_metrics(item_notes),
                "can_inspect": bool(item_notes),
            }
        )

    source_groups: dict[str, dict[str, Any]] = {}
    for note in notes:
        group = source_groups.setdefault(
            note.source_id,
            {
                "source_id": note.source_id,
                "url": note.url,
                "publisher": note.publisher,
                "note_count": 0,
                "item_ids": set(),
            },
        )
        group["note_count"] += 1
        group["item_ids"].add(note.item_id)
    by_source = [
        {
            **group,
            "item_ids": sorted(group["item_ids"]),
        }
        for group in sorted(
            source_groups.values(),
            key=lambda value: (value["publisher"], value["url"]),
        )
    ]
    return {
        "note_count": len(notes),
        "by_item": by_item,
        "by_source": by_source,
    }


def _inspect_notes_page(
    ledger: ResearchLedger,
    *,
    item_id: str,
    cursor: str | None,
    page_size: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return one deterministic page or a memory-action error."""

    if cursor is None:
        offset = 0
    elif cursor.isdigit():
        offset = int(cursor)
    else:
        return None, "cursor must be a non-negative decimal offset"

    item_notes = [note for note in ledger.notes if note.item_id == item_id]
    if offset > len(item_notes):
        return None, (
            f"cursor {offset} is past the item note count {len(item_notes)}"
        )
    page_notes = item_notes[offset : offset + page_size]
    next_offset = offset + len(page_notes)
    next_cursor = str(next_offset) if next_offset < len(item_notes) else None
    return {
        "item_id": item_id,
        "cursor": cursor,
        "notes": [_inspect_note_summary(note) for note in page_notes],
        "returned_count": len(page_notes),
        "total_count": len(item_notes),
        "next_cursor": next_cursor,
    }, None


def _recall_note_details(
    ledger: ResearchLedger,
    note_ids: Sequence[str],
    *,
    max_notes: int,
) -> tuple[list[dict[str, Any]], list[str], bool]:
    requested = list(note_ids)
    truncated = len(requested) > max_notes
    allowed = requested[:max_notes]
    notes_by_id = {
        note.note_id: note
        for note in ledger.notes
        if note.note_id is not None
    }
    found = [
        _note_summary(notes_by_id[note_id])
        for note_id in allowed
        if note_id in notes_by_id
    ]
    missing = [note_id for note_id in allowed if note_id not in notes_by_id]
    return found, missing, truncated


def _checklist_state(
    checklist: ResearchChecklist,
    collection_failures: Mapping[str, int],
) -> list[dict[str, Any]]:
    return [
        {
            "item_id": item.item_id,
            "dimension": item.dimension.value,
            "question": item.question,
            "priority": item.priority,
            "status": item.status.value,
            "consecutive_collection_failures": collection_failures.get(
                item.item_id, 0
            ),
        }
        for item in checklist.items
    ]


def _source_injection(
    ledger: ResearchLedger | None,
    settings: LoopSettings,
    recalled_urls: Sequence[str] = (),
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Inject the sources the model recalled, newest first, within budget."""

    audit: dict[str, Any] = {
        "recalled_urls": list(recalled_urls),
        "char_limit": settings.decision_source_char_limit,
        "injected_chars": 0,
        "injected_sources": [],
        "omitted_urls": [],
        "truncated": False,
    }
    if not recalled_urls or ledger is None:
        return [], audit

    urls = [url for url in recalled_urls if url in ledger.source_cache]
    remaining = settings.decision_source_char_limit
    sources: list[dict[str, str]] = []
    for url in urls:
        source_text = ledger.source_cache[url]
        injected_text = source_text[:remaining] if remaining > 0 else ""
        injected_chars = len(injected_text)
        was_truncated = injected_chars < len(source_text)
        if injected_chars:
            sources.append({"url": url, "content": injected_text})
            audit["injected_sources"].append(
                {
                    "url": url,
                    "source_chars": len(source_text),
                    "injected_chars": injected_chars,
                    "truncated": was_truncated,
                }
            )
            remaining -= injected_chars
        else:
            audit["omitted_urls"].append(url)
        if was_truncated:
            audit["truncated"] = True

    audit["injected_chars"] = sum(
        source["injected_chars"] for source in audit["injected_sources"]
    )
    return sources, audit


def _build_decision_prompt(
    checklist: ResearchChecklist,
    notes: list[ResearchNote],
    consecutive_collection_failures: Mapping[str, int],
    *,
    ledger: ResearchLedger | None,
    settings: LoopSettings,
    budget: LoopBudget,
    rounds_completed: int,
    tokens_used: int,
    cost_used_usd: float,
    recalled_urls: Sequence[str] = (),
    candidates: Mapping[str, dict[str, Any]] | None = None,
    inspected_note_page: Mapping[str, Any] | None = None,
    recalled_notes: Sequence[Mapping[str, Any]] = (),
    acquisition_attempts: Mapping[str, Mapping[str, int]] | None = None,
) -> tuple[str, dict[str, Any]]:
    sources, injection_audit = _source_injection(ledger, settings, recalled_urls)

    attempt_state = acquisition_attempts or _new_acquisition_attempt_state(
        [item.item_id for item in checklist.items]
    )
    state: dict[str, Any] = {
        "topic": checklist.topic,
        "checklist": _checklist_state(
            checklist, consecutive_collection_failures
        ),
        "budget": _budget_state(
            budget,
            rounds_completed=rounds_completed,
            tokens_used=tokens_used,
            cost_used_usd=cost_used_usd,
        ),
        "search_candidates": _candidate_state(candidates or {}),
        "candidate_work": _candidate_work_state(
            candidates or {},
            [item.item_id for item in checklist.items],
        ),
        "acquisition_attempts": _acquisition_attempt_state(attempt_state),
        "note_index": _compact_note_index(checklist, notes),
        "quote_quality": quote_quality_metrics(notes),
    }
    if inspected_note_page is not None:
        state["inspected_note_page"] = inspected_note_page
    if recalled_notes:
        state["recalled_notes"] = list(recalled_notes)
    if sources:
        state["recalled_sources"] = sources
    prompt = DECISION_PROMPT.format(
        state=json.dumps(state, ensure_ascii=False, sort_keys=True),
    )
    return prompt, injection_audit


def build_decision_prompt(
    checklist: ResearchChecklist,
    notes: list[ResearchNote],
    consecutive_collection_failures: Mapping[str, int],
    *,
    ledger: ResearchLedger | None = None,
    settings: LoopSettings | None = None,
    budget: LoopBudget | None = None,
    rounds_completed: int = 0,
    tokens_used: int = 0,
    cost_used_usd: float = 0.0,
    recalled_urls: Sequence[str] = (),
    candidates: Mapping[str, dict[str, Any]] | None = None,
    inspected_note_page: Mapping[str, Any] | None = None,
    recalled_notes: Sequence[Mapping[str, Any]] = (),
    acquisition_attempts: Mapping[str, Mapping[str, int]] | None = None,
) -> str:
    """Build a decision prompt holding only the sources the model recalled."""

    prompt, _ = _build_decision_prompt(
        checklist,
        notes,
        consecutive_collection_failures,
        ledger=ledger,
        settings=settings or LoopSettings(),
        budget=budget or LoopBudget(),
        rounds_completed=rounds_completed,
        tokens_used=tokens_used,
        cost_used_usd=cost_used_usd,
        recalled_urls=recalled_urls,
        candidates=candidates,
        inspected_note_page=inspected_note_page,
        recalled_notes=recalled_notes,
        acquisition_attempts=acquisition_attempts,
    )
    return prompt


def build_note_prompt(
    checklist: ResearchChecklist,
    *,
    active_item_id: str,
    url: str,
    source_text: str,
    max_cross_item_seeds: int = 3,
) -> str:
    """Build the isolated, single-source note extraction prompt."""

    active_item = checklist.get(active_item_id)
    if max_cross_item_seeds < 0:
        raise ValueError("max_cross_item_seeds must be non-negative")
    eligible_cross_item_targets = [
        {
            "item_id": item.item_id,
            "question": item.question,
        }
        for item in checklist.items
        if item.item_id != active_item_id and not item.is_complete
    ]
    return NOTE_PROMPT.format(
        active_item=json.dumps(
            {
                "item_id": active_item.item_id,
                "question": active_item.question,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        max_cross_item_seeds=max_cross_item_seeds,
        eligible_cross_item_targets=json.dumps(
            eligible_cross_item_targets,
            ensure_ascii=False,
            sort_keys=True,
        ),
        url=url,
        source_text=source_text,
    )


async def _generate(client: LoopModelClient, prompt: str) -> Any:
    response = client.generate(prompt)
    if inspect.isawaitable(response):
        response = await response
    return response


def _best_effort_usage(response: Any) -> tuple[int, float]:
    if not isinstance(response, Mapping):
        return 0, 0.0
    token_count = response.get("token_count", 0)
    cost_usd = response.get("cost_usd", 0.0)
    safe_tokens = (
        token_count
        if isinstance(token_count, int)
        and not isinstance(token_count, bool)
        and token_count >= 0
        else 0
    )
    safe_cost = (
        float(cost_usd)
        if isinstance(cost_usd, (int, float))
        and not isinstance(cost_usd, bool)
        and cost_usd >= 0
        else 0.0
    )
    return safe_tokens, safe_cost


def _parse_envelope(response: Any) -> tuple[ModelEnvelope | None, str | None]:
    try:
        return ModelEnvelope.model_validate(response), None
    except ValidationError as exc:
        return None, f"invalid usage envelope: {exc.errors(include_url=False)}"


def _decode_json_content(content: Any) -> Any:
    if isinstance(content, str):
        return loads_lenient(content)
    return content


def _rejected_update(
    index: int | None,
    raw: Any,
    error: str,
) -> dict[str, Any]:
    return {
        "index": index,
        "raw": raw,
        "error": error,
    }


def _parse_decision(content: Any) -> _DecisionParse:
    try:
        decoded = _decode_json_content(content)
        if isinstance(decoded, Mapping) and (
            "status_updates" in decoded
            or isinstance(decoded.get("action"), Mapping)
        ):
            rejected_updates: list[dict[str, Any]] = []
            valid_updates: list[StatusUpdate] = []
            seen_item_ids: set[str] = set()
            raw_updates = decoded.get("status_updates", ())
            if not isinstance(raw_updates, (list, tuple)):
                rejected_updates.append(
                    _rejected_update(
                        None,
                        raw_updates,
                        "status_updates must be an array",
                    )
                )
                raw_updates = ()
            for index, raw_update in enumerate(raw_updates):
                try:
                    update = StatusUpdate.model_validate(raw_update)
                except (TypeError, ValidationError, ValueError) as exc:
                    rejected_updates.append(
                        _rejected_update(index, raw_update, str(exc))
                    )
                    continue
                if update.item_id in seen_item_ids:
                    rejected_updates.append(
                        _rejected_update(
                            index,
                            raw_update,
                            f"duplicate item_id {update.item_id!r}",
                        )
                    )
                    continue
                seen_item_ids.add(update.item_id)
                valid_updates.append(update)

            action: PrimaryAction | None = None
            rejected_action: dict[str, Any] | None = None
            raw_action = decoded.get("action")
            if raw_action is not None:
                try:
                    action = _PRIMARY_ACTION_ADAPTER.validate_python(raw_action)
                except (TypeError, ValidationError, ValueError) as exc:
                    rejected_action = {
                        "raw": raw_action,
                        "error": str(exc),
                    }

            if valid_updates or action is not None:
                return _DecisionParse(
                    turn=DecisionTurn(
                        status_updates=tuple(valid_updates),
                        action=action,
                    ),
                    rejected_status_updates=tuple(rejected_updates),
                    rejected_action=rejected_action,
                )
            return _DecisionParse(
                turn=None,
                error="malformed action: no executable turn components",
                rejected_status_updates=tuple(rejected_updates),
                rejected_action=rejected_action,
            )

        # Preserve compatibility with already recorded scripted runs and simple
        # clients that still emit the original one-action protocol.
        legacy = _ACTION_ADAPTER.validate_python(decoded)
        if isinstance(legacy, SettleAction):
            return _DecisionParse(
                turn=DecisionTurn(
                    status_updates=(
                        StatusUpdate(
                            item_id=legacy.item_id,
                            status="settled",
                            reason="decision model settled the item",
                        ),
                    ),
                ),
            )
        if isinstance(legacy, MarkExhaustedAction):
            return _DecisionParse(
                turn=DecisionTurn(
                    status_updates=(
                        StatusUpdate(
                            item_id=legacy.item_id,
                            status="exhausted_not_found",
                            reason=legacy.reason,
                        ),
                    ),
                ),
            )
        return _DecisionParse(turn=DecisionTurn(action=legacy))
    except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as exc:
        return _DecisionParse(
            turn=None,
            error=f"malformed action: {exc}",
        )


def _prepare_decision(
    parsed: _DecisionParse,
    *,
    checklist: ResearchChecklist,
    ledger: ResearchLedger,
) -> tuple[
    DecisionTurn | None,
    str | None,
    list[dict[str, Any]],
    dict[str, Any] | None,
]:
    """Reject invalid components independently and retain executable pieces."""

    rejected_updates = list(parsed.rejected_status_updates)
    rejected_action = parsed.rejected_action
    valid_updates: list[StatusUpdate] = []
    action: PrimaryAction | None = None

    if parsed.turn is not None:
        for update in parsed.turn.status_updates:
            try:
                checklist.get(update.item_id)
            except KeyError:
                rejected_updates.append(
                    _rejected_update(
                        None,
                        update.model_dump(mode="json"),
                        f"unknown item_id {update.item_id!r}",
                    )
                )
            else:
                valid_updates.append(update)

        action = parsed.turn.action
        if isinstance(action, _ItemAction):
            try:
                checklist.get(action.item_id)
            except KeyError:
                rejected_action = {
                    "raw": action.model_dump(mode="json"),
                    "error": f"unknown item_id {action.item_id!r}",
                }
                action = None

    rejection_audit: list[dict[str, Any]] = []
    for rejection in rejected_updates:
        raw = rejection.get("raw")
        raw_mapping = raw if isinstance(raw, Mapping) else {}
        raw_item_id = raw_mapping.get("item_id")
        item_id = (
            raw_item_id.strip()
            if isinstance(raw_item_id, str) and raw_item_id.strip()
            else "<missing>"
        )
        raw_status = raw_mapping.get("status")
        to_status = raw_status if isinstance(raw_status, str) else None
        try:
            from_status = checklist.get(item_id).status.value
        except KeyError:
            from_status = None
        reason = f"rejected model status update: {rejection['error']}"
        ledger.record_checklist_change(
            event="status_update",
            item_id=item_id,
            accepted=False,
            reason=reason,
            from_status=from_status,
            to_status=to_status,
        )
        rejection_audit.append(
            {
                **rejection,
                "item_id": item_id,
                "status": to_status,
            }
        )

    if valid_updates or action is not None:
        return (
            DecisionTurn(
                status_updates=tuple(valid_updates),
                action=action,
            ),
            None,
            rejection_audit,
            rejected_action,
        )
    return (
        None,
        parsed.error or "malformed action: no executable turn components",
        rejection_audit,
        rejected_action,
    )


def _parse_note_channel(
    value: Any,
    *,
    channel: str,
) -> tuple[tuple[NoteDraft, ...], tuple[str, ...]]:
    if not isinstance(value, list):
        return (), (f"{channel} must be a JSON array",)
    drafts: list[NoteDraft] = []
    errors: list[str] = []
    for index, raw in enumerate(value):
        try:
            drafts.append(NoteDraft.model_validate(raw))
        except ValidationError as exc:
            errors.append(f"{channel}[{index}] invalid: {exc}")
    return tuple(drafts), tuple(errors)


def _parse_notes(content: Any) -> _NoteParse:
    """Parse both channels independently so one malformed entry stays local."""

    try:
        decoded = _decode_json_content(content)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return _NoteParse(error=f"malformed note output: {exc}")
    if not isinstance(decoded, Mapping):
        return _NoteParse(
            error="malformed note output: top-level value must be an object"
        )
    active, active_errors = _parse_note_channel(
        decoded.get("active_notes"),
        channel="active_notes",
    )
    cross, cross_errors = _parse_note_channel(
        decoded.get("cross_item_seeds"),
        channel="cross_item_seeds",
    )
    unexpected = sorted(
        str(key)
        for key in decoded
        if key not in {"active_notes", "cross_item_seeds"}
    )
    if unexpected:
        active_errors = (
            *active_errors,
            f"unexpected top-level fields: {unexpected}",
        )
    return _NoteParse(
        active_notes=active,
        cross_item_seeds=cross,
        active_errors=active_errors,
        cross_errors=cross_errors,
    )


def _open_item_ids(checklist: ResearchChecklist) -> tuple[str, ...]:
    return tuple(item.item_id for item in checklist.items if not item.is_complete)


def _budget_state(
    budget: LoopBudget,
    *,
    rounds_completed: int,
    tokens_used: int,
    cost_used_usd: float,
) -> dict[str, Any]:
    """Expose collection headroom and the protected writing allocation."""

    return {
        "remaining_rounds": max(0, budget.max_rounds - rounds_completed),
        "remaining_collection_tokens": max(
            0, budget.collection_token_limit - tokens_used
        ),
        "remaining_collection_cost_usd": max(
            0.0, budget.collection_cost_limit_usd - cost_used_usd
        ),
        "writing_reserve": {
            "tokens": budget.writing_token_reserve,
            "cost_usd": budget.writing_cost_reserve_usd,
        },
    }


def _budget_limit_reached(
    budget: LoopBudget,
    *,
    rounds: int,
    tokens: int,
    cost_usd: float,
) -> str | None:
    if rounds >= budget.max_rounds:
        return "rounds"
    if tokens >= budget.collection_token_limit:
        return "tokens"
    if cost_usd >= budget.collection_cost_limit_usd:
        return "cost"
    return None


def _settlement_evidence(
    ledger: ResearchLedger,
    item_id: str,
) -> SettlementEvidence:
    """Snapshot strict/repaired evidence without changing grounding policy."""

    strict = 0
    repaired = 0
    publishers: set[str] = set()
    for note in ledger.notes:
        if note.item_id != item_id:
            continue
        status = note.location_status.value
        if status == "locatable":
            strict += 1
        elif status == "repaired_locatable":
            # The repair status is introduced in step 5. String comparison
            # keeps this audit forward-compatible without implementing repair
            # or weakening strict grounding in this step.
            repaired += 1
        else:
            continue
        publishers.add(note.publisher)
    ordered_publishers = tuple(sorted(publishers))
    return SettlementEvidence(
        strict_locatable_notes=strict,
        repaired_locatable_notes=repaired,
        located_notes=strict + repaired,
        publisher_count=len(ordered_publishers),
        publishers=ordered_publishers,
    )


def _exhaustion_attempt_snapshot(
    *,
    ledger: ResearchLedger,
    item_id: str,
    acquisition_attempts: Mapping[str, Mapping[str, int]],
    candidates: Mapping[str, dict[str, Any]],
) -> ExhaustionAttemptSnapshot:
    """Freeze only the information state visible at terminal judgement time."""

    attempts = acquisition_attempts[item_id]
    candidate_work = _candidate_work_state(candidates, (item_id,))[item_id]
    surfaced_urls = tuple(
        url
        for url, candidate in candidates.items()
        if item_id in candidate.get("_item_work", {})
    )
    return ExhaustionAttemptSnapshot(
        **dict(attempts),
        surfaced_candidate_urls=surfaced_urls,
        read_urls=tuple(candidate_work["read_urls"]),
        dismissed_candidates=tuple(candidate_work["dismissed_candidates"]),
        unreadable_candidates=tuple(candidate_work["unreadable_candidates"]),
        pending_unread_urls=tuple(candidate_work["pending_unread_urls"]),
        note_count=sum(note.item_id == item_id for note in ledger.notes),
    )


def _collection_integrity_signals(ledger: ResearchLedger) -> list[str]:
    """Return compact immutable collection warnings for stop details."""

    signals: list[str] = []
    rejected = ledger.rejected_exhausted_without_collection_attempt_item_ids
    if rejected:
        signals.append(
            "rejected_exhausted_without_collection_attempt="
            f"{len(rejected)} ({', '.join(rejected)})"
        )
    accepted = ledger.accepted_exhausted_without_collection_attempt_item_ids
    if accepted:
        signals.append(
            "accepted_exhausted_without_collection_attempt="
            f"{len(accepted)} ({', '.join(accepted)})"
        )
    unknown = ledger.accepted_exhausted_attempt_unknown_legacy_item_ids
    if unknown:
        signals.append(
            "accepted_exhausted_attempt_unknown_legacy="
            f"{len(unknown)} ({', '.join(unknown)})"
        )
    unread = ledger.exhausted_with_unread_candidates_item_ids
    if unread:
        signals.append(
            "exhausted_with_unread_candidates="
            f"{len(unread)} ({', '.join(unread)})"
        )
    return signals


def _with_collection_integrity_signals(
    detail: str,
    ledger: ResearchLedger,
) -> str:
    signals = _collection_integrity_signals(ledger)
    return f"{detail}; {'; '.join(signals)}" if signals else detail


def _terminal_stop_detail(ledger: ResearchLedger) -> str:
    item_ids = ledger.settled_without_located_evidence_item_ids
    detail = "all checklist items reached a terminal state"
    if item_ids:
        detail += (
            "; settled_without_located_evidence="
            f"{len(item_ids)} ({', '.join(item_ids)})"
        )
    return _with_collection_integrity_signals(detail, ledger)


def _audit_summary(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _record_preflight_stop(
    ledger: ResearchLedger,
    *,
    stop_reason: StopReason,
    stop_detail: str,
    open_item_ids: tuple[str, ...],
) -> None:
    next_round = max((record.round_number for record in ledger.rounds), default=0) + 1
    ledger.record_round(
        round_number=next_round,
        action="stop",
        result_summary=_audit_summary(
            {
                "stop_reason": stop_reason.value,
                "stop_detail": stop_detail,
                "open_item_ids": list(open_item_ids),
            }
        ),
    )


def _result(
    *,
    checklist: ResearchChecklist,
    ledger: ResearchLedger,
    stop_reason: StopReason,
    stop_detail: str,
    rounds_executed: int,
    collection_failures: Mapping[str, int],
) -> LoopResult:
    return LoopResult(
        checklist=checklist,
        ledger=ledger,
        stop_reason=stop_reason,
        stop_detail=stop_detail,
        open_item_ids=_open_item_ids(checklist),
        rounds_executed=rounds_executed,
        consecutive_collection_failures=dict(collection_failures),
    )


def _apply_status_updates(
    checklist: ResearchChecklist,
    *,
    updates: Sequence[StatusUpdate],
    ledger: ResearchLedger,
    acquisition_attempts: Mapping[str, Mapping[str, int]],
    candidates: Mapping[str, dict[str, Any]],
) -> tuple[
    ResearchChecklist,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Apply independently reasoned model judgements with settle-time evidence."""

    current = checklist
    audit: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for update in updates:
        if update.status == "settled":
            evidence = _settlement_evidence(ledger, update.item_id)
            current = current.set_status(
                update.item_id,
                ChecklistStatus.SETTLED,
                reason=update.reason,
                ledger=ledger,
                settlement_evidence=evidence.model_dump(mode="json"),
            )
            audit.append(
                {
                    "item_id": update.item_id,
                    "status": update.status,
                    "reason": update.reason,
                    "settlement_evidence": evidence.model_dump(mode="json"),
                }
            )
        else:
            snapshot = _exhaustion_attempt_snapshot(
                ledger=ledger,
                item_id=update.item_id,
                acquisition_attempts=acquisition_attempts,
                candidates=candidates,
            )
            snapshot_payload = snapshot.model_dump(mode="json")
            if not snapshot.has_qualifying_attempt:
                error = (
                    "exhausted_not_found requires a prior real search, read, "
                    "or reanalyze attempt attributed to this item"
                )
                ledger.record_checklist_change(
                    event="status_change",
                    item_id=update.item_id,
                    accepted=False,
                    reason=update.reason,
                    from_status=current.get(update.item_id).status.value,
                    to_status=ChecklistStatus.EXHAUSTED_NOT_FOUND.value,
                    exhaustion_attempts=snapshot_payload,
                )
                rejected.append(
                    {
                        "item_id": update.item_id,
                        "status": update.status,
                        "reason": update.reason,
                        "error": error,
                        "rejection_reason": (
                            "no_prior_collection_attempt"
                        ),
                        "exhaustion_attempts": snapshot_payload,
                    }
                )
                continue
            current = current.set_status(
                update.item_id,
                ChecklistStatus.EXHAUSTED_NOT_FOUND,
                reason=update.reason,
                ledger=ledger,
                exhaustion_attempts=snapshot_payload,
            )
            audit.append(
                {
                    "item_id": update.item_id,
                    "status": update.status,
                    "reason": update.reason,
                    "exhaustion_attempts": snapshot_payload,
                    "exhausted_with_unread_candidates": bool(
                        snapshot.pending_unread_urls
                    ),
                }
            )
    return current, audit, rejected


def _note_location_counts(notes: Sequence[ResearchNote]) -> dict[str, int]:
    """Return stable per-channel quote-location counts for one note pass."""

    return {
        "strict_locatable": sum(
            note.location_status is NoteLocationStatus.LOCATABLE for note in notes
        ),
        "repaired_locatable": sum(
            note.location_status is NoteLocationStatus.REPAIRED_LOCATABLE
            for note in notes
        ),
        "unlocatable": sum(
            note.location_status is NoteLocationStatus.UNLOCATABLE for note in notes
        ),
    }


async def _extract_notes(
    checklist: ResearchChecklist,
    *,
    ledger: ResearchLedger,
    note_model: LoopModelClient,
    active_item_id: str,
    url: str,
    source_text: str,
    max_cross_item_seeds: int,
) -> tuple[ResearchChecklist, int, float, dict[str, Any]]:
    """Run one explicit note pass and retain every mechanically checked draft."""

    note_response: Any = None
    note_call_error: str | None = None
    try:
        note_response = await _generate(
            note_model,
            build_note_prompt(
                checklist,
                active_item_id=active_item_id,
                url=url,
                source_text=source_text,
                max_cross_item_seeds=max_cross_item_seeds,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - auditable model turn
        note_call_error = f"note model error: {exc}"

    note_envelope, note_envelope_error = _parse_envelope(note_response)
    note_tokens, note_cost = _best_effort_usage(note_response)
    if note_envelope is not None:
        note_tokens = note_envelope.token_count
        note_cost = note_envelope.cost_usd

    parsed = _NoteParse(error=note_call_error or note_envelope_error)
    if parsed.error is None and note_envelope is not None:
        parsed = _parse_notes(note_envelope.content)

    current = checklist
    created: list[ResearchNote] = []
    active_created: list[ResearchNote] = []
    cross_created: list[ResearchNote] = []
    active_errors = list(parsed.active_errors)
    cross_errors = list(parsed.cross_errors)

    def retain(draft: NoteDraft) -> tuple[ResearchNote | None, str | None]:
        try:
            current.get(draft.item_id)
            note = create_note(
                item_id=draft.item_id,
                finding=draft.finding,
                quote=draft.quote,
                url=url,
                source_text=source_text,
            )
        except Exception as exc:  # noqa: BLE001 - preserve the rest of the batch
            return None, str(exc)
        note = ledger.add_note(note)
        created.append(note)
        return note, None

    for index, draft in enumerate(parsed.active_notes):
        if draft.item_id != active_item_id:
            active_errors.append(
                f"active_notes[{index}] item_id must equal active item "
                f"{active_item_id!r}, got {draft.item_id!r}"
            )
            continue
        note, error = retain(draft)
        if note is None:
            active_errors.append(
                f"active_notes[{index}] could not be retained: {error}"
            )
            continue
        active_created.append(note)

    cross_item_ids: set[str] = set()
    for index, draft in enumerate(parsed.cross_item_seeds):
        prefix = f"cross_item_seeds[{index}]"
        if draft.item_id == active_item_id:
            cross_errors.append(f"{prefix} must not target the active item")
            continue
        try:
            item = current.get(draft.item_id)
        except Exception as exc:  # noqa: BLE001 - preserve valid siblings
            cross_errors.append(f"{prefix} unknown item_id: {exc}")
            continue
        if item.is_complete:
            cross_errors.append(
                f"{prefix} targets terminal item {draft.item_id!r}"
            )
            continue
        if draft.item_id in cross_item_ids:
            cross_errors.append(
                f"{prefix} repeats cross-item seed {draft.item_id!r}"
            )
            continue
        if len(cross_item_ids) >= max_cross_item_seeds:
            cross_errors.append(
                f"{prefix} exceeds cross-item seed capacity "
                f"{max_cross_item_seeds}"
            )
            continue
        cross_item_ids.add(draft.item_id)
        note, error = retain(draft)
        if note is None:
            cross_errors.append(f"{prefix} could not be retained: {error}")
            continue
        cross_created.append(note)

    for note in created:
        item = current.get(note.item_id)
        if item.status is ChecklistStatus.UNEXPLORED:
            current = current.set_status(
                note.item_id,
                ChecklistStatus.HAS_MATERIAL,
                reason=f"note collected from {url}",
                ledger=ledger,
            )

    active_note_count = len(active_created)
    summary = {
        "source_chars": len(source_text),
        "notes_created": len(created),
        "active_item_notes": active_note_count,
        "active_notes_proposed": len(parsed.active_notes),
        "active_notes_created": len(active_created),
        "active_note_location_counts": _note_location_counts(active_created),
        "cross_item_seeds_proposed": len(parsed.cross_item_seeds),
        "cross_item_seeds_created": len(cross_created),
        "cross_item_seed_location_counts": _note_location_counts(cross_created),
        "cross_item_seed_item_ids": [note.item_id for note in cross_created],
        "cross_item_seed_capacity": max_cross_item_seeds,
        "note_item_ids": [note.item_id for note in created],
        "note_output_error": parsed.error,
        "active_note_errors": active_errors,
        "cross_item_seed_errors": cross_errors,
        "note_creation_errors": [*active_errors, *cross_errors],
        "note_model_called": True,
    }
    return current, note_tokens, note_cost, summary


def _cached_note_summary(
    ledger: ResearchLedger,
    url: str,
) -> dict[str, Any]:
    grouped = ledger.note_ids_for_url(url)
    by_item = {
        item_id: {"count": len(note_ids), "note_ids": list(note_ids)}
        for item_id, note_ids in sorted(grouped.items())
    }
    return {
        "existing_note_ids": [
            note_id for details in by_item.values() for note_id in details["note_ids"]
        ],
        "existing_notes_by_item": by_item,
    }


async def run_research_loop(
    checklist: ResearchChecklist,
    *,
    ledger: ResearchLedger,
    decision_model: LoopModelClient,
    note_model: LoopModelClient,
    tavily_client: TavilyClient,
    budget: LoopBudget | None = None,
    settings: LoopSettings | None = None,
    max_search_results: int = 5,
) -> LoopResult:
    """Run collection until completion, explicit stop, or a hard limit."""

    limits = budget or LoopBudget()
    context_settings = settings or LoopSettings()
    current = checklist
    collection_failures = {item.item_id: 0 for item in current.items}
    rounds_executed = 0
    total_tokens = ledger.total_tokens
    total_cost = ledger.total_cost_usd

    open_ids = _open_item_ids(current)
    if not open_ids:
        detail = _terminal_stop_detail(ledger)
        _record_preflight_stop(
            ledger,
            stop_reason=StopReason.ALL_ITEMS_TERMINAL,
            stop_detail=detail,
            open_item_ids=(),
        )
        return _result(
            checklist=current,
            ledger=ledger,
            stop_reason=StopReason.ALL_ITEMS_TERMINAL,
            stop_detail=detail,
            rounds_executed=0,
            collection_failures=collection_failures,
        )

    initial_limit = _budget_limit_reached(
        limits,
        rounds=0,
        tokens=total_tokens,
        cost_usd=total_cost,
    )
    if initial_limit is not None:
        detail = f"{initial_limit} budget was exhausted before the first round"
        _record_preflight_stop(
            ledger,
            stop_reason=StopReason.BUDGET_EXHAUSTED,
            stop_detail=detail,
            open_item_ids=open_ids,
        )
        return _result(
            checklist=current,
            ledger=ledger,
            stop_reason=StopReason.BUDGET_EXHAUSTED,
            stop_detail=detail,
            rounds_executed=0,
            collection_failures=collection_failures,
        )

    consecutive_malformed = 0
    recalled: list[str] = []
    inspected_note_page: dict[str, Any] | None = None
    recalled_note_details: list[dict[str, Any]] = []
    candidates: dict[str, dict[str, Any]] = {}
    acquisition_attempts = _new_acquisition_attempt_state(
        [item.item_id for item in current.items]
    )
    while True:
        budget_before_decision = _budget_state(
            limits,
            rounds_completed=rounds_executed,
            tokens_used=total_tokens,
            cost_used_usd=total_cost,
        )
        prompt, context_audit = _build_decision_prompt(
            current,
            ledger.notes,
            collection_failures,
            ledger=ledger,
            settings=context_settings,
            budget=limits,
            rounds_completed=rounds_executed,
            tokens_used=total_tokens,
            cost_used_usd=total_cost,
            recalled_urls=recalled,
            candidates=candidates,
            inspected_note_page=inspected_note_page,
            recalled_notes=recalled_note_details,
            acquisition_attempts=acquisition_attempts,
        )
        rounds_executed += 1
        response: Any = None
        decision_error: str | None = None
        try:
            response = await _generate(decision_model, prompt)
        except Exception as exc:  # noqa: BLE001 - provider errors are auditable turns
            decision_error = f"decision model error: {exc}"

        envelope, envelope_error = _parse_envelope(response)
        decision_tokens, decision_cost = _best_effort_usage(response)
        if envelope is not None:
            decision_tokens = envelope.token_count
            decision_cost = envelope.cost_usd
        total_tokens += decision_tokens
        total_cost += decision_cost

        turn: DecisionTurn | None = None
        rejected_status_updates: list[dict[str, Any]] = []
        rejected_action: dict[str, Any] | None = None
        action_error = decision_error or envelope_error
        if action_error is None and envelope is not None:
            parsed = _parse_decision(envelope.content)
            (
                turn,
                action_error,
                rejected_status_updates,
                rejected_action,
            ) = _prepare_decision(
                parsed,
                checklist=current,
                ledger=ledger,
            )

        summary: dict[str, Any] = {}
        query: str | None = None
        url: str | None = None
        action_name = "invalid_action"
        round_tokens = decision_tokens
        round_cost = decision_cost
        stop_reason: StopReason | None = None
        stop_detail = ""
        status_audit: list[dict[str, Any]] = []
        candidate_audit_item_ids: set[str] = set()
        if turn is not None:
            candidate_audit_item_ids.update(
                update.item_id for update in turn.status_updates
            )
            if isinstance(turn.action, _ItemAction):
                candidate_audit_item_ids.add(turn.action.item_id)

        decision_budget_limit = _budget_limit_reached(
            limits,
            rounds=rounds_executed - 1,
            tokens=total_tokens,
            cost_usd=total_cost,
        )
        if decision_budget_limit is not None:
            action_name = (
                turn.action.action
                if turn is not None and turn.action is not None
                else action_name
            )
            summary = {
                "action_skipped": True,
                "error": action_error,
                "budget_limit": decision_budget_limit,
            }
            stop_reason = StopReason.BUDGET_EXHAUSTED
            stop_detail = (
                f"{decision_budget_limit} budget reached by decision model usage"
            )
        elif turn is None:
            consecutive_malformed += 1
            summary = {
                "error": action_error or "malformed action",
                "raw_content": (
                    response.get("content")
                    if isinstance(response, Mapping)
                    else response
                ),
                "consecutive_malformed_actions": consecutive_malformed,
            }
            if (
                consecutive_malformed
                >= limits.max_consecutive_malformed_actions
            ):
                stop_reason = StopReason.MALFORMED_ACTION_LIMIT
                stop_detail = (
                    f"{consecutive_malformed} consecutive malformed actions"
                )
        else:
            consecutive_malformed = 0
            action = turn.action
            if action is not None:
                action_name = action.action
            elif len(turn.status_updates) == 1:
                action_name = (
                    "settle"
                    if turn.status_updates[0].status == "settled"
                    else "mark_exhausted"
                )
            else:
                action_name = "status_updates"

            current, status_audit, rejected_exhaustions = (
                _apply_status_updates(
                current,
                updates=turn.status_updates,
                ledger=ledger,
                acquisition_attempts=acquisition_attempts,
                candidates=candidates,
                )
            )
            rejected_status_updates.extend(rejected_exhaustions)
            if status_audit:
                summary["status_updates"] = status_audit

            if current.is_complete:
                if action is not None:
                    summary["action_skipped"] = True
                    summary["action_skip_reason"] = (
                        "status updates made every checklist item terminal"
                    )
                stop_reason = StopReason.ALL_ITEMS_TERMINAL
                stop_detail = _terminal_stop_detail(ledger)

            elif isinstance(action, SearchAction):
                query = action.query
                pending_urls = _pending_candidate_urls(
                    candidates,
                    action.item_id,
                )
                if pending_urls:
                    summary = {
                        "action_rejected": True,
                        "rejection_reason": "candidates_pending",
                        "pending_unread_count": len(pending_urls),
                        "pending_unread_urls": pending_urls,
                    }
                else:
                    try:
                        results = await search(
                            action.query,
                            tavily_client=tavily_client,
                            max_results=max_search_results,
                        )
                        _record_acquisition_attempt(
                            acquisition_attempts,
                            item_id=action.item_id,
                            action="search",
                            succeeded=True,
                        )
                        newly_pending = _remember_candidates(
                            candidates,
                            results,
                            item_id=action.item_id,
                        )
                        summary = {
                            "result_count": len(results),
                            "results": [
                                result.model_dump(mode="json")
                                for result in results
                            ],
                            "new_pending_unread_count": len(newly_pending),
                            "new_pending_unread_urls": newly_pending,
                        }
                    except Exception as exc:  # noqa: BLE001 - tool failure is one turn
                        _record_acquisition_attempt(
                            acquisition_attempts,
                            item_id=action.item_id,
                            action="search",
                            succeeded=False,
                        )
                        summary = {
                            "error": f"search failed: {exc}",
                            "result_count": 0,
                        }
                collection_failures[action.item_id] = (
                    collection_failures.get(action.item_id, 0) + 1
                )

            elif isinstance(action, ReadAction):
                url = action.url
                source_text = ledger.get_source(action.url)
                cache_hit = source_text is not None
                if cache_hit:
                    _record_acquisition_attempt(
                        acquisition_attempts,
                        item_id=action.item_id,
                        action="read",
                        succeeded=True,
                    )
                    _mark_candidate_read(
                        candidates,
                        item_id=action.item_id,
                        url=action.url,
                    )
                    summary = {
                        "cache_hit": True,
                        "source_chars": len(source_text),
                        "notes_created": 0,
                        "active_item_notes": 0,
                        "note_model_called": False,
                        **_cached_note_summary(ledger, action.url),
                    }
                    collection_failures[action.item_id] = (
                        collection_failures.get(action.item_id, 0) + 1
                    )
                else:
                    prior_error = _candidate_unreadable_error(
                        candidates,
                        item_id=action.item_id,
                        url=action.url,
                    )
                    if prior_error is not None:
                        summary = {
                            "action_rejected": True,
                            "rejection_reason": "candidate_unreadable",
                            "previous_read_error": prior_error,
                            "notes_created": 0,
                            "note_model_called": False,
                        }
                        collection_failures[action.item_id] = (
                            collection_failures.get(action.item_id, 0) + 1
                        )
                    else:
                        try:
                            source_text = await read(
                                action.url,
                                tavily_client=tavily_client,
                            )
                            ledger.cache_source(action.url, source_text)
                            _record_acquisition_attempt(
                                acquisition_attempts,
                                item_id=action.item_id,
                                action="read",
                                succeeded=True,
                            )
                        except Exception as exc:  # noqa: BLE001 - tool failure is one turn
                            error = f"read failed: {exc}"
                            _record_acquisition_attempt(
                                acquisition_attempts,
                                item_id=action.item_id,
                                action="read",
                                succeeded=False,
                            )
                            _mark_candidate_unreadable(
                                candidates,
                                item_id=action.item_id,
                                url=action.url,
                                error=error,
                            )
                            summary = {
                                "cache_hit": False,
                                "error": error,
                                "candidate_marked_unreadable": True,
                                "notes_created": 0,
                                "note_model_called": False,
                            }
                            collection_failures[action.item_id] = (
                                collection_failures.get(action.item_id, 0) + 1
                            )

                    if source_text is not None:
                        _mark_candidate_read(
                            candidates,
                            item_id=action.item_id,
                            url=action.url,
                        )
                        current, note_tokens, note_cost, note_summary = (
                            await _extract_notes(
                                current,
                                ledger=ledger,
                                note_model=note_model,
                                active_item_id=action.item_id,
                                url=action.url,
                                source_text=source_text,
                                max_cross_item_seeds=(
                                    context_settings.max_cross_item_seeds
                                ),
                            )
                        )
                        summary = {"cache_hit": False, **note_summary}
                        if note_summary["active_item_notes"]:
                            collection_failures[action.item_id] = 0
                        else:
                            collection_failures[action.item_id] = (
                                collection_failures.get(action.item_id, 0) + 1
                            )
                        total_tokens += note_tokens
                        total_cost += note_cost
                        round_tokens += note_tokens
                        round_cost += note_cost

            elif isinstance(action, DismissCandidatesAction):
                dismissed: list[dict[str, str]] = []
                rejected_dismissals: list[dict[str, str]] = []
                for candidate in action.candidates:
                    dismissal_error = _dismiss_candidate(
                        candidates,
                        item_id=action.item_id,
                        url=candidate.url,
                        reason=candidate.reason,
                    )
                    if dismissal_error is None:
                        dismissed.append(candidate.model_dump(mode="json"))
                    else:
                        rejected_dismissals.append(
                            {
                                **candidate.model_dump(mode="json"),
                                "error": dismissal_error,
                            }
                        )
                remaining_pending = _pending_candidate_urls(
                    candidates,
                    action.item_id,
                )
                summary = {
                    "dismissed_candidates": dismissed,
                    "dismissed_count": len(dismissed),
                    "rejected_dismissals": rejected_dismissals,
                    "remaining_pending_unread_count": len(remaining_pending),
                    "remaining_pending_unread_urls": remaining_pending,
                }

            elif isinstance(action, ReanalyzeAction):
                url = action.url
                source_text = ledger.get_source(action.url)
                if source_text is None:
                    summary = {
                        "reason": action.reason,
                        "error": "reanalyze requires a URL in the source cache",
                        "notes_created": 0,
                        "note_model_called": False,
                    }
                    collection_failures[action.item_id] = (
                        collection_failures.get(action.item_id, 0) + 1
                    )
                else:
                    _mark_candidate_read(
                        candidates,
                        item_id=action.item_id,
                        url=action.url,
                    )
                    current, note_tokens, note_cost, note_summary = (
                        await _extract_notes(
                            current,
                            ledger=ledger,
                            note_model=note_model,
                            active_item_id=action.item_id,
                            url=action.url,
                            source_text=source_text,
                            max_cross_item_seeds=(
                                context_settings.max_cross_item_seeds
                            ),
                        )
                    )
                    summary = {
                        "cache_hit": True,
                        "reanalyze_reason": action.reason,
                        **note_summary,
                    }
                    _record_acquisition_attempt(
                        acquisition_attempts,
                        item_id=action.item_id,
                        action="reanalyze",
                        succeeded=not bool(note_summary["note_output_error"]),
                    )
                    if note_summary["active_item_notes"]:
                        collection_failures[action.item_id] = 0
                    else:
                        collection_failures[action.item_id] = (
                            collection_failures.get(action.item_id, 0) + 1
                        )
                    total_tokens += note_tokens
                    total_cost += note_cost
                    round_tokens += note_tokens
                    round_cost += note_cost

            elif isinstance(action, RecallAction):
                url = action.url
                if action.url in ledger.source_cache:
                    # Newest first, so the budget drops the stalest recall.
                    if action.url in recalled:
                        recalled.remove(action.url)
                    recalled.insert(0, action.url)
                    summary = {
                        "url": action.url,
                        "recalled": True,
                        "source_chars": len(ledger.source_cache[action.url]),
                        "recalled_urls": list(recalled),
                    }
                else:
                    # A recall for an unread url is an honest miss, not a crash:
                    # record it and let the model pick again next round. Recall
                    # is a memory action, so it cannot change collection failure
                    # memory even when it misses.
                    summary = {
                        "url": action.url,
                        "recalled": False,
                        "detail": "url is not in the source cache",
                    }

            elif isinstance(action, InspectNotesAction):
                page, page_error = _inspect_notes_page(
                    ledger,
                    item_id=action.item_id,
                    cursor=action.cursor,
                    page_size=context_settings.note_page_size,
                )
                if page is None:
                    summary = {
                        "item_id": action.item_id,
                        "cursor": action.cursor,
                        "inspected": False,
                        "detail": page_error,
                    }
                else:
                    inspected_note_page = page
                    summary = {
                        "item_id": action.item_id,
                        "cursor": action.cursor,
                        "inspected": True,
                        "returned_note_ids": [
                            note["note_id"] for note in page["notes"]
                        ],
                        "returned_count": page["returned_count"],
                        "total_count": page["total_count"],
                        "next_cursor": page["next_cursor"],
                    }

            elif isinstance(action, RecallNotesAction):
                details, missing_ids, truncated = _recall_note_details(
                    ledger,
                    action.note_ids,
                    max_notes=context_settings.max_recalled_notes,
                )
                recalled_note_details = details
                summary = {
                    "requested_note_ids": list(action.note_ids),
                    "recalled_note_ids": [
                        note["note_id"] for note in details
                    ],
                    "missing_note_ids": missing_ids,
                    "truncated": truncated,
                    "max_recalled_notes": context_settings.max_recalled_notes,
                }

            elif isinstance(action, StopAction):
                open_ids = _open_item_ids(current)
                summary = {"open_item_ids": list(open_ids)}
                stop_reason = StopReason.MODEL_STOP_WITH_OPEN_ITEMS
                stop_detail = "decision model requested stop while items remained open"

        if status_audit:
            summary["status_updates"] = status_audit
        if rejected_status_updates:
            summary["rejected_status_updates"] = rejected_status_updates
        if rejected_action is not None:
            summary["rejected_action"] = rejected_action
        if candidate_audit_item_ids:
            summary["candidate_work"] = _candidate_work_state(
                candidates,
                sorted(candidate_audit_item_ids),
            )
        summary["decision_context"] = context_audit
        summary["budget_before_decision"] = budget_before_decision
        summary["acquisition_attempts"] = _acquisition_attempt_state(
            acquisition_attempts
        )

        if stop_reason is None and current.is_complete:
            stop_reason = StopReason.ALL_ITEMS_TERMINAL
            stop_detail = _terminal_stop_detail(ledger)

        if stop_reason is None:
            limit = _budget_limit_reached(
                limits,
                rounds=rounds_executed,
                tokens=total_tokens,
                cost_usd=total_cost,
            )
            if limit is not None:
                stop_reason = StopReason.BUDGET_EXHAUSTED
                stop_detail = f"{limit} budget exhausted"
                summary["budget_limit"] = limit

        if stop_reason is not None:
            if stop_reason is not StopReason.ALL_ITEMS_TERMINAL:
                stop_detail = _with_collection_integrity_signals(
                    stop_detail,
                    ledger,
                )
            summary["stop_reason"] = stop_reason.value
            summary["stop_detail"] = stop_detail
            summary["open_item_ids"] = list(_open_item_ids(current))

        ledger.record_round(
            round_number=rounds_executed,
            action=action_name,
            item_id=(
                turn.action.item_id
                if turn is not None and isinstance(turn.action, _ItemAction)
                else None
            ),
            query=query,
            url=url,
            result_summary=_audit_summary(summary),
            token_count=round_tokens,
            cost_usd=round_cost,
        )

        if stop_reason is not None:
            return _result(
                checklist=current,
                ledger=ledger,
                stop_reason=stop_reason,
                stop_detail=stop_detail,
                rounds_executed=rounds_executed,
                collection_failures=collection_failures,
            )
