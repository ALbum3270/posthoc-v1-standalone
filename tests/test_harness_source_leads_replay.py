from pathlib import Path

import pytest

from scripts.replay_harness_source_leads import replay_source_leads


_FINANCE_11_AUDIT = Path(
    "/data/Langgraph/open_deep_research-main/"
    "harness_runs/finance-11/audit.json"
)
_INVESTOPEDIA_URL = (
    "https://www.investopedia.com/what-went-wrong-with-ftx-6828447"
)


def test_finance_11_replay_extracts_real_leads_without_inventing_links():
    if not _FINANCE_11_AUDIT.is_file():
        pytest.skip("finance-11 historical audit is not present in this checkout")
    before = _FINANCE_11_AUDIT.read_bytes()

    result = replay_source_leads(_FINANCE_11_AUDIT)

    assert result["summary"] == {
        "cached_source_count": 7,
        "source_with_explicit_url_count": 0,
        "explicit_url_count": 0,
        "doi_count": 0,
        "numbered_entry_count": 52,
        "bibliographic_candidate_count": 40,
        "locator_entry_count": 3,
    }
    by_url = {source["url"]: source for source in result["sources"]}
    entries = {
        entry["number"]: entry
        for entry in by_url[_INVESTOPEDIA_URL]["numbered_entries"]
    }
    assert entries[6]["source_label_candidate"] == "X"
    assert entries[6]["document_title_candidate"] == (
        "@cz\\_binance, 11:09 a.m. Nov. 8, 2022."
    )
    assert entries[7]["source_label_candidate"] == "The Wall Street Journal"
    assert entries[7]["document_title_candidate"] == (
        "Binance Walks Away from Deal to Rescue FTX."
    )
    assert entries[12]["source_label_candidate"] == (
        "U.S. Bankruptcy Court for the District of Delaware, via CourtListener"
    )
    assert entries[12]["document_title_candidate"] == (
        "Voluntary Petition for Non-Individuals Filing for Bankruptcy: "
        "FTX Crypto Services Ltd."
    )
    assert entries[36]["locators"] == [
        {"kind": "docket_number", "value": "docket #26030"},
        {
            "kind": "related_document_numbers",
            "value": "related document(s)19139, 22165",
        },
    ]
    assert entries[37]["locators"] == [
        {"kind": "docket_number", "value": "Docket #14301"}
    ]
    assert result["source_audit_unchanged"] is True
    routes = {
        route["claim_id"]: route
        for route in result["routing_replay"]["routes"]
    }
    assert {
        claim_id: route["query_route"]
        for claim_id, route in routes.items()
    } == {
        "claim-0035": "source_chain",
        "claim-0036": "source_chain",
        "claim-0040": "source_chain",
        "claim-0047": "direct_search_fallback",
        "claim-0056": "source_chain",
    }
    assert routes["claim-0040"]["source_document_hint"].endswith(
        "Docket #14301, click link. Downloads a document."
    )
    assert "FTX Recovery Trust" in routes["claim-0056"][
        "source_document_hint"
    ]
    assert routes["claim-0047"]["source_document_hint"] is None
    assert result["routing_replay"]["network_calls"] == 0
    assert _FINANCE_11_AUDIT.read_bytes() == before
