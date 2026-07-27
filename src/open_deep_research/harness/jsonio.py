"""Lenient JSON decoding for model output.

Models asked for "JSON only" still wrap it in a markdown fence or prepend a
sentence. Rejecting that output would discard a decision the model made
correctly, so decoding is tolerant of the packaging while staying strict about
the payload: nothing is invented, and an undecodable response still raises.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(
    r"^\s*```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n(?P<body>.*?)\r?\n?[ \t]*```\s*$",
    re.DOTALL,
)


def _strip_fence(text: str) -> str:
    match = _FENCE.match(text)
    return match.group("body") if match else text


def _bracketed_span(text: str) -> str | None:
    """Return the outermost JSON object or array substring, if any."""

    starts = [index for index in (text.find("{"), text.find("[")) if index != -1]
    if not starts:
        return None
    start = min(starts)
    closer = "}" if text[start] == "{" else "]"
    end = text.rfind(closer)
    if end <= start:
        return None
    return text[start : end + 1]


def loads_lenient(text: str) -> Any:
    """Decode JSON that may be fenced or surrounded by prose.

    Raises:
        json.JSONDecodeError: when no JSON value can be decoded.
    """

    candidates = [text, _strip_fence(text)]
    span = _bracketed_span(_strip_fence(text))
    if span is not None:
        candidates.append(span)

    error: json.JSONDecodeError | None = None
    for candidate in candidates:
        stripped = candidate.strip()
        if not stripped:
            continue
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            error = exc

    raise error or json.JSONDecodeError("Expecting value", text, 0)
