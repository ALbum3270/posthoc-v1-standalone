"""Tests for the deterministic date guard (SESSION_HANDOFF §3.12).

The fixtures below are not invented. Every string in ``REAL_FACTS_FROM_GRAPH``
was read back out of the local Neo4j after the 2026-07-24 V1 baseline run, so
these tests pin the exact behaviour that produced -- and would have prevented --
the observed 2023/2026 contamination.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from open_deep_research.graphrag.validation.dates import (
    extract_explicit_dates,
    resolve_valid_at,
    stated_years,
)


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


# (source text, expected valid_at) -- read out of the graph on 2026-07-24.
REAL_FACTS_FROM_GRAPH = [
    (
        "On 11 November 2022, the cryptocurrency exchange FTX filed for Chapter 11 bankruptcy",
        _utc(2022, 11, 11),
    ),
    ("FTX filed for bankruptcy protection in Delaware on 11 November 2022.", _utc(2022, 11, 11)),
    ("FTX and Alameda Research filed for bankruptcy together on November 11, 2022", _utc(2022, 11, 11)),
    ("The FTX bankruptcy began in November 2022", _utc(2022, 11, 1)),
    ("Bankman-Fried was convicted related to the FTX bankruptcy in November 2023", _utc(2023, 11, 1)),
    ("日本金融厅 ordered FTX Japan to suspend its operations on 2022-11-11", _utc(2022, 11, 11)),
    ("SBF posted a tweet to explain the situation of FTX on 2022-11-09 at midnight", _utc(2022, 11, 9)),
]


@pytest.mark.parametrize("text,expected", REAL_FACTS_FROM_GRAPH)
def test_explicit_dates_are_preserved(text: str, expected: datetime) -> None:
    """Facts that state a date keep it. The guard must not be blanket-restrictive."""

    assert resolve_valid_at(text) == expected


def test_the_actual_bug_yields_no_date() -> None:
    """The exact triple that caused the contamination.

    v1's extractor produced this with no year; Graphiti then filled the year from
    reference_time (2026) in one episode and from model prior (2023) in another.
    Under the guard it gets no date at all, which is the correct answer.
    """

    assert resolve_valid_at("FTX announced bankruptcy in mid-November") is None


@pytest.mark.parametrize(
    "text",
    [
        "FTX announced bankruptcy in mid-November",
        "the collapse happened last year",
        "FTX faces a liquidity gap of up to 8 billion USD",
        "Alameda Research holds 6.112 billion USD in FTT tokens",
        "SBF's net worth dropped from 16 billion USD to 1 USD",
        "",
    ],
)
def test_underspecified_or_dateless_text_yields_no_date(text: str) -> None:
    """No explicit calendar reference -> no valid_at. Large numbers are not years."""

    assert resolve_valid_at(text) is None


def test_bare_year_does_not_become_a_dated_edge() -> None:
    """A stated year with no month is still underspecified for valid_at.

    "founded FTX in 2019" would have to invent a month to become a date. The year
    is real, so it is reported by stated_years() and belongs in edge attributes --
    but it must not masquerade as a calendar date on the edge.
    """

    text = "Sam Bankman-Fried founded FTX in 2019"
    assert resolve_valid_at(text) is None
    assert stated_years(text) == {2019}


def test_chinese_dates_are_understood() -> None:
    assert resolve_valid_at("FTX 于 2022年11月11日 申请破产保护") == _utc(2022, 11, 11)
    assert resolve_valid_at("事件发生在 2022年11月") == _utc(2022, 11, 1)


def test_precise_match_is_not_double_reported_at_lower_precision() -> None:
    """"11 November 2022" must yield one day-precision date, not also "November 2022"."""

    found = extract_explicit_dates("FTX filed on 11 November 2022.")
    assert [(ev.value.isoformat(), ev.precision) for ev in found] == [("2022-11-11", "day")]


def test_earliest_and_most_precise_wins_when_several_dates_appear() -> None:
    text = "FTX filed on 11 November 2022 and Bankman-Fried was convicted in November 2023"
    assert resolve_valid_at(text) == _utc(2022, 11, 11)
    assert stated_years(text) == {2022, 2023}


def test_published_at_is_never_used_to_date_an_undated_fact() -> None:
    """reference_time fixes Graphiti's own extraction; it is not a dating fallback.

    Passing it must not resurrect the exact failure mode the guard exists to stop.
    """

    assert (
        resolve_valid_at(
            "FTX announced bankruptcy in mid-November",
            published_at=_utc(2022, 11, 20),
        )
        is None
    )


def test_out_of_band_years_are_not_dates() -> None:
    """Version numbers and quantities must not be read as calendar years."""

    assert stated_years("build 1234 shipped") == set()
    assert resolve_valid_at("port 8080 was open in 1899") is None


def test_subset_assertion_holds_for_the_guard() -> None:
    """The §3.12 acceptance rule, stated as a property of the guard itself.

    Any year the guard is willing to write must be a year the source stated.
    """

    for text, _ in REAL_FACTS_FROM_GRAPH:
        resolved = resolve_valid_at(text)
        assert resolved is not None
        assert resolved.year in stated_years(text)
