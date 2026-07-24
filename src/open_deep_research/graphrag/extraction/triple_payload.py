"""Parse whatever shape a model returns triples in.

Observed against ``openai/gpt-4.1-mini`` on 2026-07-24, same task, prompt wording
the only variable:

    {"triples": [ {...}, {...} ]}      # short prompt
    {"subject": ..., "predicate": ..., "object": ...}   # long prompt, ONE bare triple
    ```json\\n[ {...} ]```             # no response_format

The bare-object case is the dangerous one. A parser that only looks for a list
under a known key reads it as "no facts found" -- indistinguishable from a
passage that genuinely had nothing, so a live run reports clean failures while
silently discarding every extraction. That is exactly what happened.

Root cause of the variance was a self-contradictory prompt: ``json_object`` mode
forces a top-level object while the instructions asked for an array. Fixing the
prompt narrows the distribution but does not bound it, so parsing stays liberal.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

_TRIPLE_KEYS = ("subject", "predicate", "object")
_OPTIONAL_TEXT_KEYS = ("quote",)
# Keys a model plausibly nests the list under.
_LIST_KEYS = ("triples", "facts", "results", "items", "data", "output", "extractions")


def _strip_fence(text: str) -> str:
    """Remove a markdown code fence if the model wrapped its JSON in one."""

    return _FENCE.sub("", text.strip())


def _is_triple(candidate: Any) -> bool:
    return isinstance(candidate, dict) and all(
        str(candidate.get(key) or "").strip() for key in _TRIPLE_KEYS
    )


def parse_triple_payload(raw: str | None) -> list[dict[str, str]]:
    """Extract triple dicts from a model response, whatever shape it arrived in.

    Returns a list of ``{"subject", "predicate", "object"}`` dicts with values
    stripped. Anything unparseable, or any entry missing a field, is dropped --
    a malformed triple is not worth guessing at when it is destined for a graph
    that is meant to be trustworthy.
    """

    if not raw:
        return []

    try:
        payload = json.loads(_strip_fence(raw))
    except json.JSONDecodeError:
        return []

    rows: list[Any]
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        if _is_triple(payload):
            # One bare triple, unwrapped.
            rows = [payload]
        else:
            rows = []
            for key in _LIST_KEYS:
                value = payload.get(key)
                if isinstance(value, list):
                    rows = value
                    break
            else:
                # Unknown key: accept the single list of dicts if there is
                # exactly one, rather than losing the extraction to naming.
                lists = [v for v in payload.values() if isinstance(v, list)]
                if len(lists) == 1:
                    rows = lists[0]
    else:
        return []

    triples: list[dict[str, str]] = []
    for row in rows:
        if not _is_triple(row):
            continue
        parsed = {key: str(row[key]).strip() for key in _TRIPLE_KEYS}
        for key in _OPTIONAL_TEXT_KEYS:
            value = row.get(key)
            if value is not None and str(value).strip():
                parsed[key] = str(value).strip()
        triples.append(parsed)
    return triples
