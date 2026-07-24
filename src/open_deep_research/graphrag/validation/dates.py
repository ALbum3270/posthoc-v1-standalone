"""Deterministic date extraction used to gate ``EntityEdge.valid_at``.

Why this exists (SESSION_HANDOFF §3.12): when a source triple is temporally
underspecified -- "FTX announced bankruptcy in mid-November", no year -- Graphiti's
edge date resolution is unconstrained. It fills the year from either
``reference_time`` (producing 2026 in the observed run) or from model prior
(producing 2023). Neither year appears in the source. The day part resolved
correctly both times; only the year was invented.

The rule enforced here is mechanical, so it needs no threshold calibration
(SESSION_HANDOFF §2.8, reversed half):

    A date may be written to the graph only if the source text explicitly
    supports it. No explicit date -> ``valid_at`` stays None.

An absent ``valid_at`` is a correct answer, not a degraded one. ``expired_at IS
NULL`` still marks the edge active (§2.3), so a dateless fact is fully usable;
a fact carrying a fabricated year is not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone

# Month names accepted in either direction. Index 0 is a placeholder so that
# MONTHS.index(name) == month number.
_MONTH_NAMES = (
    "",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_MONTH_BY_NAME: dict[str, int] = {}
for _i, _n in enumerate(_MONTH_NAMES):
    if _n:
        _MONTH_BY_NAME[_n] = _i
        _MONTH_BY_NAME[_n[:3]] = _i

# Years outside this band are almost certainly not calendar years (version
# numbers, quantities, identifiers).
_YEAR_MIN = 1900
_YEAR_MAX = 2100

_MONTH_ALT = "|".join(sorted(_MONTH_BY_NAME, key=len, reverse=True))

# 2022-11-11 / 2022/11/11
_RE_ISO = re.compile(r"\b(?P<y>\d{4})[-/](?P<m>\d{1,2})[-/](?P<d>\d{1,2})\b")
# 11 November 2022
_RE_DMY = re.compile(
    rf"\b(?P<d>\d{{1,2}})\s+(?P<mon>{_MONTH_ALT})\.?,?\s+(?P<y>\d{{4}})\b", re.I
)
# November 11, 2022
_RE_MDY = re.compile(
    rf"\b(?P<mon>{_MONTH_ALT})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y>\d{{4}})\b",
    re.I,
)
# November 2022  (no day)
_RE_MY = re.compile(rf"\b(?P<mon>{_MONTH_ALT})\.?,?\s+(?P<y>\d{{4}})\b", re.I)
# Chinese: 2022年11月11日 / 2022年11月
_RE_CJK = re.compile(r"(?P<y>\d{4})\s*年\s*(?:(?P<m>\d{1,2})\s*月\s*(?:(?P<d>\d{1,2})\s*日)?)?")


@dataclass(frozen=True)
class DateEvidence:
    """One date the source text explicitly supports.

    ``precision`` records how much the text actually said, so callers can tell a
    stated "November 2022" (day defaulted to 1) from a stated "11 November 2022".
    """

    value: date
    precision: str  # "day" | "month" | "year"
    matched_text: str

    @property
    def has_month(self) -> bool:
        return self.precision in {"day", "month"}


def _safe_date(year: int, month: int, day: int) -> date | None:
    """Build a date, rejecting out-of-band years and impossible calendar values."""

    if not (_YEAR_MIN <= year <= _YEAR_MAX):
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _consume(text: str, spans: list[tuple[int, int]], start: int, end: int) -> bool:
    """Claim a span of the text, refusing overlaps with already-matched spans.

    Patterns run most-specific first, so an earlier (more precise) match wins and
    a later, looser pattern cannot re-report the same characters at lower
    precision -- "11 November 2022" must not also yield a bare "November 2022".
    """

    for s, e in spans:
        if start < e and s < end:
            return False
    spans.append((start, end))
    return True


def extract_explicit_dates(text: str) -> list[DateEvidence]:
    """Return every date the text states outright, most precise patterns first.

    Only explicit calendar references count. Relative or partial expressions --
    "mid-November", "last year", "recently" -- deliberately yield nothing: they
    are exactly the inputs that let an LLM invent a year.
    """

    if not text:
        return []

    found: list[DateEvidence] = []
    spans: list[tuple[int, int]] = []

    for match in _RE_ISO.finditer(text):
        value = _safe_date(int(match["y"]), int(match["m"]), int(match["d"]))
        if value and _consume(text, spans, *match.span()):
            found.append(DateEvidence(value, "day", match.group(0)))

    for pattern in (_RE_DMY, _RE_MDY):
        for match in pattern.finditer(text):
            month = _MONTH_BY_NAME.get(match["mon"].lower())
            value = _safe_date(int(match["y"]), month or 0, int(match["d"]))
            if value and _consume(text, spans, *match.span()):
                found.append(DateEvidence(value, "day", match.group(0)))

    for match in _RE_CJK.finditer(text):
        month = int(match["m"]) if match["m"] else 1
        day = int(match["d"]) if match["d"] else 1
        value = _safe_date(int(match["y"]), month, day)
        if value and _consume(text, spans, *match.span()):
            precision = "day" if match["d"] else ("month" if match["m"] else "year")
            found.append(DateEvidence(value, precision, match.group(0)))

    for match in _RE_MY.finditer(text):
        month = _MONTH_BY_NAME.get(match["mon"].lower())
        value = _safe_date(int(match["y"]), month or 0, 1)
        if value and _consume(text, spans, *match.span()):
            found.append(DateEvidence(value, "month", match.group(0)))

    # Bare years last: only the ones no richer pattern already claimed.
    for match in re.finditer(r"\b(\d{4})\b", text):
        year = int(match.group(1))
        value = _safe_date(year, 1, 1)
        if value and _consume(text, spans, *match.span()):
            found.append(DateEvidence(value, "year", match.group(0)))

    return sorted(found, key=lambda ev: (ev.value, ev.precision))


def resolve_valid_at(
    fact_text: str,
    *,
    require_month: bool = True,
    published_at: datetime | None = None,
) -> datetime | None:
    """Decide the ``valid_at`` an edge is allowed to carry.

    Returns None whenever the text does not state a date precise enough to
    stand on its own. That is the whole point: a missing date is honest, an
    inferred one is not.

    ``published_at`` is accepted but deliberately unused as a fallback -- it is
    the correct ``reference_time`` for Graphiti's own extraction (§3.12 fix 1),
    not a licence to date a fact the source never dated. It stays in the
    signature so callers do not reach for it themselves.
    """

    candidates = extract_explicit_dates(fact_text)
    if not candidates:
        return None

    if require_month:
        candidates = [ev for ev in candidates if ev.has_month]
        if not candidates:
            return None

    # Most precise wins; ties break to the earliest date so a fact spanning a
    # range is anchored at its start.
    best = min(candidates, key=lambda ev: ({"day": 0, "month": 1, "year": 2}[ev.precision], ev.value))
    return datetime(best.value.year, best.value.month, best.value.day, tzinfo=timezone.utc)


def stated_years(text: str) -> set[int]:
    """Years the text explicitly contains, for the subset assertion (§3.12).

    The acceptance rule is that years appearing on written edges must be a
    subset of the years appearing in the source triples.
    """

    return {ev.value.year for ev in extract_explicit_dates(text)}
