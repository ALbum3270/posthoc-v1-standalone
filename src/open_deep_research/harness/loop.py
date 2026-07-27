"""Model-directed collection loop with deterministic budgets and bookkeeping."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Mapping, Sequence
from enum import Enum
from typing import Annotated, Any, Literal, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
)

from open_deep_research.harness.checklist import (
    ChecklistStatus,
    ResearchChecklist,
)
from open_deep_research.harness.jsonio import loads_lenient
from open_deep_research.harness.ledger import ResearchLedger
from open_deep_research.harness.notes import ResearchNote, create_note
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
    max_consecutive_malformed_actions: int = Field(default=3, ge=1)


class LoopSettings(BaseModel):
    """Budget for source text the decision model asked to see.

    Whether a source body enters the decision context is the model's call, made
    by returning a ``recall`` action. Code only caps how much can be injected,
    which is a budget concern rather than a judgement about relevance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_source_char_limit: int = Field(default=100_000, ge=0)


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
    | RecallAction
    | SettleAction
    | MarkExhaustedAction
    | StopAction,
    Field(discriminator="action"),
]
_ACTION_ADAPTER = TypeAdapter(LoopAction)


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


class NoteBatch(BaseModel):
    """The note model may legitimately return an empty batch."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    notes: tuple[NoteDraft, ...] = ()


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
    consecutive_failures: dict[str, int]

    @property
    def is_success(self) -> bool:
        """Only terminal checklist completion counts as success."""

        return self.stop_reason is StopReason.ALL_ITEMS_TERMINAL


DECISION_PROMPT = """\
Choose exactly one next action for this research run.

You may return:
{{"action":"search","item_id":"...","query":"..."}}
{{"action":"read","item_id":"...","url":"..."}}
{{"action":"recall","item_id":"...","url":"..."}}
{{"action":"settle","item_id":"..."}}
{{"action":"mark_exhausted","item_id":"...","reason":"..."}}
{{"action":"stop"}}

The action item_id identifies the checklist item this round is working on.

search_candidates holds every url search has surfaced so far, with its snippet
and whether you already read it. Searching only adds candidates; reading one is
what produces notes. Prefer reading a promising unread candidate over searching
again for the same thing.

You normally see note summaries rather than source bodies. When a summary is not
enough, recall an already-read url and its full text joins the next round's
state. Recall as often as you need; the oldest recalled sources drop out first
once the character budget is reached.

Return JSON only.

Current collection state:
{state}
"""


NOTE_PROMPT = """\
Read the single source below and extract every useful note that answers any
checklist item. A note may target an item other than the action that led here.
Copy each quote verbatim from the source without rewriting it. Return JSON only
as {{"notes":[{{"item_id":"...","finding":"...","quote":"exact source text"}}]}}.
Returning {{"notes":[]}} is valid when the source answers nothing.

Checklist:
{checklist}

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
    return recent


def _remember_candidates(
    candidates: dict[str, dict[str, Any]],
    results: Sequence[SearchResult],
) -> None:
    for result in results:
        existing = candidates.get(result.url)
        if existing is not None:
            continue
        candidates[result.url] = {
            "url": result.url,
            "title": result.title,
            "snippet": result.snippet[:_CANDIDATE_SNIPPET_CHARS],
            "score": result.score,
            "read": False,
        }


def _note_summary(note: ResearchNote) -> dict[str, Any]:
    return {
        "item_id": note.item_id,
        "finding": note.finding,
        "quote": note.quote,
        "url": note.url,
        "publisher": note.publisher,
        "location_status": note.location_status.value,
    }


def _checklist_state(
    checklist: ResearchChecklist,
    failures: Mapping[str, int],
) -> list[dict[str, Any]]:
    return [
        {
            "item_id": item.item_id,
            "dimension": item.dimension.value,
            "question": item.question,
            "priority": item.priority,
            "required_source_count": item.required_source_count,
            "status": item.status.value,
            "consecutive_failures": failures.get(item.item_id, 0),
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
    consecutive_failures: Mapping[str, int],
    *,
    ledger: ResearchLedger | None,
    settings: LoopSettings,
    recalled_urls: Sequence[str] = (),
    candidates: Mapping[str, dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    sources, injection_audit = _source_injection(ledger, settings, recalled_urls)

    state: dict[str, Any] = {
        "topic": checklist.topic,
        "checklist": _checklist_state(checklist, consecutive_failures),
        "search_candidates": _candidate_state(candidates or {}),
        "note_summaries": [_note_summary(note) for note in notes],
    }
    if sources:
        state["recalled_sources"] = sources
    prompt = DECISION_PROMPT.format(
        state=json.dumps(state, ensure_ascii=False, sort_keys=True),
    )
    return prompt, injection_audit


def build_decision_prompt(
    checklist: ResearchChecklist,
    notes: list[ResearchNote],
    consecutive_failures: Mapping[str, int],
    *,
    ledger: ResearchLedger | None = None,
    settings: LoopSettings | None = None,
    recalled_urls: Sequence[str] = (),
    candidates: Mapping[str, dict[str, Any]] | None = None,
) -> str:
    """Build a decision prompt holding only the sources the model recalled."""

    prompt, _ = _build_decision_prompt(
        checklist,
        notes,
        consecutive_failures,
        ledger=ledger,
        settings=settings or LoopSettings(),
        recalled_urls=recalled_urls,
        candidates=candidates,
    )
    return prompt


def build_note_prompt(
    checklist: ResearchChecklist,
    *,
    url: str,
    source_text: str,
) -> str:
    """Build the isolated, single-source note extraction prompt."""

    items = [
        {
            "item_id": item.item_id,
            "question": item.question,
            "status": item.status.value,
        }
        for item in checklist.items
    ]
    return NOTE_PROMPT.format(
        checklist=json.dumps(items, ensure_ascii=False, sort_keys=True),
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


def _parse_action(content: Any) -> tuple[LoopAction | None, str | None]:
    try:
        decoded = _decode_json_content(content)
        return _ACTION_ADAPTER.validate_python(decoded), None
    except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as exc:
        return None, f"malformed action: {exc}"


def _parse_notes(content: Any) -> tuple[NoteBatch | None, str | None]:
    try:
        decoded = _decode_json_content(content)
        if isinstance(decoded, list):
            decoded = {"notes": decoded}
        return NoteBatch.model_validate(decoded), None
    except (json.JSONDecodeError, TypeError, ValidationError, ValueError) as exc:
        return None, f"malformed note output: {exc}"


def _open_item_ids(checklist: ResearchChecklist) -> tuple[str, ...]:
    return tuple(item.item_id for item in checklist.items if not item.is_complete)


def _budget_limit_reached(
    budget: LoopBudget,
    *,
    rounds: int,
    tokens: int,
    cost_usd: float,
) -> str | None:
    if rounds >= budget.max_rounds:
        return "rounds"
    if tokens >= budget.max_tokens:
        return "tokens"
    if cost_usd >= budget.max_cost_usd:
        return "cost"
    return None


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
    failures: Mapping[str, int],
) -> LoopResult:
    return LoopResult(
        checklist=checklist,
        ledger=ledger,
        stop_reason=stop_reason,
        stop_detail=stop_detail,
        open_item_ids=_open_item_ids(checklist),
        rounds_executed=rounds_executed,
        consecutive_failures=dict(failures),
    )


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
    failures = {item.item_id: 0 for item in current.items}
    rounds_executed = 0
    total_tokens = ledger.total_tokens
    total_cost = ledger.total_cost_usd

    open_ids = _open_item_ids(current)
    if not open_ids:
        _record_preflight_stop(
            ledger,
            stop_reason=StopReason.ALL_ITEMS_TERMINAL,
            stop_detail="all checklist items were already terminal",
            open_item_ids=(),
        )
        return _result(
            checklist=current,
            ledger=ledger,
            stop_reason=StopReason.ALL_ITEMS_TERMINAL,
            stop_detail="all checklist items were already terminal",
            rounds_executed=0,
            failures=failures,
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
            failures=failures,
        )

    consecutive_malformed = 0
    recalled: list[str] = []
    candidates: dict[str, dict[str, Any]] = {}
    while True:
        rounds_executed += 1
        prompt, context_audit = _build_decision_prompt(
            current,
            ledger.notes,
            failures,
            ledger=ledger,
            settings=context_settings,
            recalled_urls=recalled,
            candidates=candidates,
        )
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

        action: LoopAction | None = None
        action_error = decision_error or envelope_error
        if action_error is None and envelope is not None:
            action, action_error = _parse_action(envelope.content)
        if isinstance(action, _ItemAction):
            try:
                current.get(action.item_id)
            except KeyError:
                action_error = f"malformed action: unknown item_id {action.item_id!r}"
                action = None

        summary: dict[str, Any] = {}
        query: str | None = None
        url: str | None = None
        action_name = "invalid_action"
        round_tokens = decision_tokens
        round_cost = decision_cost
        stop_reason: StopReason | None = None
        stop_detail = ""

        decision_budget_limit = _budget_limit_reached(
            limits,
            rounds=rounds_executed - 1,
            tokens=total_tokens,
            cost_usd=total_cost,
        )
        if decision_budget_limit is not None:
            action_name = action.action if action is not None else action_name
            summary = {
                "action_skipped": True,
                "error": action_error,
                "budget_limit": decision_budget_limit,
            }
            stop_reason = StopReason.BUDGET_EXHAUSTED
            stop_detail = (
                f"{decision_budget_limit} budget reached by decision model usage"
            )
        elif action is None:
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
            if isinstance(response, Mapping):
                candidate_content = response.get("content")
                if isinstance(candidate_content, Mapping):
                    candidate_item_id = candidate_content.get("item_id")
                    if candidate_item_id in failures:
                        failures[str(candidate_item_id)] += 1
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
            action_name = action.action
            if isinstance(action, SearchAction):
                query = action.query
                try:
                    results = await search(
                        action.query,
                        tavily_client=tavily_client,
                        max_results=max_search_results,
                    )
                    _remember_candidates(candidates, results)
                    summary = {
                        "result_count": len(results),
                        "results": [
                            result.model_dump(mode="json") for result in results
                        ],
                    }
                except Exception as exc:  # noqa: BLE001 - tool failure is one turn
                    summary = {"error": f"search failed: {exc}", "result_count": 0}
                failures[action.item_id] = failures.get(action.item_id, 0) + 1

            elif isinstance(action, ReadAction):
                url = action.url
                candidate = candidates.setdefault(
                    action.url,
                    {
                        "url": action.url,
                        "title": "",
                        "snippet": "",
                        "score": None,
                        "read": False,
                    },
                )
                candidate["read"] = True
                source_text = ledger.get_source(action.url)
                cache_hit = source_text is not None
                if source_text is None:
                    try:
                        source_text = await read(
                            action.url,
                            tavily_client=tavily_client,
                        )
                        ledger.cache_source(action.url, source_text)
                    except Exception as exc:  # noqa: BLE001 - tool failure is one turn
                        summary = {
                            "cache_hit": False,
                            "error": f"read failed: {exc}",
                            "notes_created": 0,
                        }
                        failures[action.item_id] = failures.get(action.item_id, 0) + 1

                if source_text is not None:
                    note_response: Any = None
                    note_call_error: str | None = None
                    try:
                        note_response = await _generate(
                            note_model,
                            build_note_prompt(
                                current,
                                url=action.url,
                                source_text=source_text,
                            ),
                        )
                    except Exception as exc:  # noqa: BLE001 - auditable model turn
                        note_call_error = f"note model error: {exc}"

                    note_envelope, note_envelope_error = _parse_envelope(note_response)
                    note_tokens, note_cost = _best_effort_usage(note_response)
                    if note_envelope is not None:
                        note_tokens = note_envelope.token_count
                        note_cost = note_envelope.cost_usd
                    total_tokens += note_tokens
                    total_cost += note_cost
                    round_tokens += note_tokens
                    round_cost += note_cost

                    batch: NoteBatch | None = None
                    note_error = note_call_error or note_envelope_error
                    if note_error is None and note_envelope is not None:
                        batch, note_error = _parse_notes(note_envelope.content)

                    created: list[ResearchNote] = []
                    creation_errors: list[str] = []
                    for draft in batch.notes if batch is not None else ():
                        try:
                            note = create_note(
                                item_id=draft.item_id,
                                finding=draft.finding,
                                quote=draft.quote,
                                url=action.url,
                                source_text=source_text,
                            )
                        except Exception as exc:  # noqa: BLE001 - preserve the batch
                            creation_errors.append(str(exc))
                            continue
                        ledger.add_note(note)
                        created.append(note)
                        if draft.item_id in failures:
                            item = current.get(draft.item_id)
                            if item.status is ChecklistStatus.UNEXPLORED:
                                current = current.set_status(
                                    draft.item_id,
                                    ChecklistStatus.HAS_MATERIAL,
                                    reason=f"note collected from {action.url}",
                                    ledger=ledger,
                                )

                    active_note_count = sum(
                        note.item_id == action.item_id for note in created
                    )
                    if active_note_count:
                        failures[action.item_id] = 0
                    else:
                        failures[action.item_id] = (
                            failures.get(action.item_id, 0) + 1
                        )
                    summary = {
                        "cache_hit": cache_hit,
                        "source_chars": len(source_text),
                        "notes_created": len(created),
                        "active_item_notes": active_note_count,
                        "note_item_ids": [note.item_id for note in created],
                        "note_output_error": note_error,
                        "note_creation_errors": creation_errors,
                    }

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
                    # record it and let the model pick again next round.
                    failures[action.item_id] = failures.get(action.item_id, 0) + 1
                    summary = {
                        "url": action.url,
                        "recalled": False,
                        "detail": "url is not in the source cache",
                    }

            elif isinstance(action, SettleAction):
                current = current.set_status(
                    action.item_id,
                    ChecklistStatus.SETTLED,
                    reason="decision model settled the item",
                    ledger=ledger,
                )
                summary = {"item_id": action.item_id, "status": "settled"}

            elif isinstance(action, MarkExhaustedAction):
                current = current.set_status(
                    action.item_id,
                    ChecklistStatus.EXHAUSTED_NOT_FOUND,
                    reason=action.reason,
                    ledger=ledger,
                )
                summary = {
                    "item_id": action.item_id,
                    "status": "exhausted_not_found",
                    "reason": action.reason,
                }

            elif isinstance(action, StopAction):
                open_ids = _open_item_ids(current)
                summary = {"open_item_ids": list(open_ids)}
                stop_reason = StopReason.MODEL_STOP_WITH_OPEN_ITEMS
                stop_detail = "decision model requested stop while items remained open"

        summary["decision_context"] = context_audit

        if stop_reason is None and current.is_complete:
            stop_reason = StopReason.ALL_ITEMS_TERMINAL
            stop_detail = "all checklist items reached a terminal state"

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
            summary["stop_reason"] = stop_reason.value
            summary["stop_detail"] = stop_detail
            summary["open_item_ids"] = list(_open_item_ids(current))

        ledger.record_round(
            round_number=rounds_executed,
            action=action_name,
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
                failures=failures,
            )
