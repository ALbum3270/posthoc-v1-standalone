"""Code-owned necessary checks for numeric evidence alignment.

The verifier model decides whether a source semantically bears on a claim.  A
located quote alone, however, cannot establish that the numeric value in the
claim is the value in that quote.  This module checks a deliberately narrow,
mechanical necessary condition for formal support: when both texts expose
comparable, normalized numeric surfaces, every numeric surface in the claim
must occur in the selected quote.

It is not a general arithmetic or fact-checking system.  In particular, it
does not infer quantities written only in words, determine the semantic role
of a number, or decide whether a number is current.  Those remain model and
research questions.  The check exists to prevent readily detectable errors
such as treating ``$900 million`` as support for ``9000 万美元``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum


class NumericConsistencyStatus(str, Enum):
    """Mechanical numeric-alignment result for one located verifier quote."""

    NOT_APPLICABLE = "not_applicable"
    ALIGNED = "aligned"
    MISMATCH = "mismatch"
    SOURCE_VALUES_NOT_RECOGNIZED = "source_values_not_recognized"


class NumericSurfaceKind(str, Enum):
    """Kinds whose values can be compared without domain-specific knowledge."""

    CURRENCY = "currency"
    PERCENT = "percent"
    DATE = "date"
    MAGNITUDE = "magnitude"


@dataclass(frozen=True)
class NumericSurface:
    """One exact numeric surface normalized into an inclusive interval."""

    kind: NumericSurfaceKind
    lower: Decimal
    upper: Decimal
    raw: str
    currency: str | None = None

    def display(self) -> str:
        """Render a stable, audit-friendly representation without rounding."""

        interval = (
            str(self.lower)
            if self.lower == self.upper
            else f"{self.lower}..{self.upper}"
        )
        suffix = f" {self.currency}" if self.currency is not None else ""
        return f"{self.kind.value}({interval}{suffix}; {self.raw!r})"


@dataclass(frozen=True)
class NumericConsistencyAssessment:
    """Non-semantic numeric comparison with a concise audit detail."""

    status: NumericConsistencyStatus
    detail: str | None = None


_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
_CN_UNIT = r"(?:万亿|亿|万|千)"
_EN_UNIT = r"(?:thousand|million|billion|trillion|bn|b|m|k)"
_UNIT = rf"(?:{_CN_UNIT}|{_EN_UNIT})"
_CURRENCY_PREFIX = r"(?:US\s?\$|USD|\$|€|£|¥)"
_CURRENCY_SUFFIX = (
    r"(?:美元|美金|人民币|(?:U\.?S\.?\s*)?dollars?|USD|EUR|euros?|"
    r"GBP|pounds?)"
)
_RANGE_SEPARATOR = r"(?:-|–|—|~|～|至|到|to)"

_DATE_PATTERNS = (
    re.compile(
        r"(?P<year>\d{4})年\s*(?P<month>\d{1,2})月\s*(?P<day>\d{1,2})日?"
    ),
    re.compile(r"(?P<year>\d{4})[-/.](?P<month>\d{1,2})[-/.](?P<day>\d{1,2})"),
)
_CURRENCY_RANGE = re.compile(
    rf"(?P<first>{_NUMBER})\s*(?P<unit_first>{_UNIT})?\s*"
    rf"{_RANGE_SEPARATOR}\s*(?P<second>{_NUMBER})\s*"
    rf"(?P<unit_second>{_UNIT})?\s*(?P<currency>{_CURRENCY_SUFFIX})",
    re.IGNORECASE,
)
_CURRENCY_PREFIX_SINGLE = re.compile(
    rf"(?P<prefix>{_CURRENCY_PREFIX})\s*(?P<value>{_NUMBER})\s*"
    rf"(?P<unit>{_UNIT})?",
    re.IGNORECASE,
)
_CURRENCY_SUFFIX_SINGLE = re.compile(
    rf"(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})?\s*"
    rf"(?P<currency>{_CURRENCY_SUFFIX})",
    re.IGNORECASE,
)
_PERCENT_RANGE = re.compile(
    rf"(?P<first>{_NUMBER})\s*%\s*{_RANGE_SEPARATOR}\s*"
    rf"(?P<second>{_NUMBER})\s*%?"
)
_PERCENT_SINGLE = re.compile(rf"(?P<value>{_NUMBER})\s*%")
_PERCENT_CN = re.compile(rf"百分之\s*(?P<value>{_NUMBER})")
_MAGNITUDE_SINGLE = re.compile(
    rf"(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})\b", re.IGNORECASE
)

_UNIT_MULTIPLIERS: dict[str, Decimal] = {
    "千": Decimal("1000"),
    "万": Decimal("10000"),
    "亿": Decimal("100000000"),
    "万亿": Decimal("1000000000000"),
    "thousand": Decimal("1000"),
    "k": Decimal("1000"),
    "million": Decimal("1000000"),
    "m": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "bn": Decimal("1000000000"),
    "b": Decimal("1000000000"),
    "trillion": Decimal("1000000000000"),
}


def _decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None


def _multiplier(unit: str | None) -> Decimal:
    return _UNIT_MULTIPLIERS.get((unit or "").casefold(), Decimal("1"))


def _currency_code(marker: str | None) -> str | None:
    if marker is None:
        return None
    normalized = marker.casefold().replace(" ", "")
    if (
        "$" in normalized
        or "usd" in normalized
        or "dollar" in normalized
        or "美元" in normalized
        or "美金" in normalized
    ):
        return "USD"
    if "€" in normalized or "eur" in normalized or "euro" in normalized:
        return "EUR"
    if "£" in normalized or "gbp" in normalized or "pound" in normalized:
        return "GBP"
    if "¥" in normalized or "人民币" in normalized:
        return "CNY"
    return None


def _in_used_span(start: int, end: int, used: list[tuple[int, int]]) -> bool:
    return any(start < used_end and end > used_start for used_start, used_end in used)


def _append_surface(
    surfaces: list[NumericSurface],
    used: list[tuple[int, int]],
    *,
    match: re.Match[str],
    kind: NumericSurfaceKind,
    lower: Decimal,
    upper: Decimal | None = None,
    currency: str | None = None,
) -> None:
    if _in_used_span(match.start(), match.end(), used):
        return
    surfaces.append(
        NumericSurface(
            kind=kind,
            lower=min(lower, upper if upper is not None else lower),
            upper=max(lower, upper if upper is not None else lower),
            raw=match.group(0),
            currency=currency,
        )
    )
    used.append((match.start(), match.end()))


def extract_numeric_surfaces(text: str) -> tuple[NumericSurface, ...]:
    """Extract only unambiguous, byte-visible numeric expressions from text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    surfaces: list[NumericSurface] = []
    used: list[tuple[int, int]] = []

    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            year = _decimal(match.group("year"))
            month = _decimal(match.group("month"))
            day = _decimal(match.group("day"))
            if year is None or month is None or day is None:
                continue
            if not (Decimal("1") <= month <= Decimal("12")) or not (
                Decimal("1") <= day <= Decimal("31")
            ):
                continue
            encoded = year * Decimal("10000") + month * Decimal("100") + day
            _append_surface(
                surfaces,
                used,
                match=match,
                kind=NumericSurfaceKind.DATE,
                lower=encoded,
            )

    for match in _CURRENCY_RANGE.finditer(text):
        first = _decimal(match.group("first"))
        second = _decimal(match.group("second"))
        if first is None or second is None:
            continue
        shared_unit = match.group("unit_second") or match.group("unit_first")
        _append_surface(
            surfaces,
            used,
            match=match,
            kind=NumericSurfaceKind.CURRENCY,
            lower=first * _multiplier(shared_unit),
            upper=second * _multiplier(shared_unit),
            currency=_currency_code(match.group("currency")),
        )

    for match in _CURRENCY_PREFIX_SINGLE.finditer(text):
        value = _decimal(match.group("value"))
        if value is None:
            continue
        _append_surface(
            surfaces,
            used,
            match=match,
            kind=NumericSurfaceKind.CURRENCY,
            lower=value * _multiplier(match.group("unit")),
            currency=_currency_code(match.group("prefix")),
        )

    for match in _CURRENCY_SUFFIX_SINGLE.finditer(text):
        value = _decimal(match.group("value"))
        if value is None:
            continue
        _append_surface(
            surfaces,
            used,
            match=match,
            kind=NumericSurfaceKind.CURRENCY,
            lower=value * _multiplier(match.group("unit")),
            currency=_currency_code(match.group("currency")),
        )

    for match in _PERCENT_RANGE.finditer(text):
        first = _decimal(match.group("first"))
        second = _decimal(match.group("second"))
        if first is None or second is None:
            continue
        _append_surface(
            surfaces,
            used,
            match=match,
            kind=NumericSurfaceKind.PERCENT,
            lower=first,
            upper=second,
        )

    for pattern in (_PERCENT_SINGLE, _PERCENT_CN):
        for match in pattern.finditer(text):
            value = _decimal(match.group("value"))
            if value is None:
                continue
            _append_surface(
                surfaces,
                used,
                match=match,
                kind=NumericSurfaceKind.PERCENT,
                lower=value,
            )

    for match in _MAGNITUDE_SINGLE.finditer(text):
        value = _decimal(match.group("value"))
        if value is None:
            continue
        _append_surface(
            surfaces,
            used,
            match=match,
            kind=NumericSurfaceKind.MAGNITUDE,
            lower=value * _multiplier(match.group("unit")),
        )

    return tuple(surfaces)


def _comparable(left: NumericSurface, right: NumericSurface) -> bool:
    if left.kind is not right.kind:
        return False
    if left.kind is NumericSurfaceKind.CURRENCY:
        return left.currency is None or right.currency is None or left.currency == right.currency
    return True


def _overlap(left: NumericSurface, right: NumericSurface) -> bool:
    return left.lower <= right.upper and right.lower <= left.upper


def assess_numeric_consistency(
    claim_text: str,
    source_quote: str,
) -> NumericConsistencyAssessment:
    """Check claim values against a semantically selected source quote.

    A mismatch is emitted only when the quote exposes comparable numeric
    surfaces and none overlaps a claim surface.  If the source expresses the
    relevant number only in words, this check abstains rather than inventing a
    mismatch; semantic support remains the verifier model's responsibility.
    """

    claim_values = extract_numeric_surfaces(claim_text)
    if not claim_values:
        return NumericConsistencyAssessment(
            status=NumericConsistencyStatus.NOT_APPLICABLE
        )
    source_values = extract_numeric_surfaces(source_quote)
    unmatched: list[NumericSurface] = []
    source_lacked_comparable_value = False
    for claim_value in claim_values:
        comparable_values = [
            source_value
            for source_value in source_values
            if _comparable(claim_value, source_value)
        ]
        if not comparable_values:
            source_lacked_comparable_value = True
            continue
        if not any(_overlap(claim_value, source_value) for source_value in comparable_values):
            unmatched.append(claim_value)

    if unmatched:
        detail = "; ".join(
            (
                "claim numeric surfaces do not occur in selected quote: "
                + ", ".join(value.display() for value in unmatched)
                + "; quote numeric surfaces: "
                + ", ".join(value.display() for value in source_values)
            ).splitlines()
        )
        return NumericConsistencyAssessment(
            status=NumericConsistencyStatus.MISMATCH,
            detail=detail,
        )
    if source_lacked_comparable_value:
        return NumericConsistencyAssessment(
            status=NumericConsistencyStatus.SOURCE_VALUES_NOT_RECOGNIZED,
            detail=(
                "claim has a numeric surface but selected quote has no "
                "code-recognized comparable numeric surface"
            ),
        )
    return NumericConsistencyAssessment(status=NumericConsistencyStatus.ALIGNED)
