import asyncio
from pathlib import Path

import pytest

from scripts.replay_harness_recovery import replay_finance_recovery


_FINANCE_11_AUDIT = Path(
    "/data/Langgraph/open_deep_research-main/"
    "harness_runs/finance-11/audit.json"
)


def test_finance_11_registry_replays_recovery_without_network_or_mutation():
    """Exercise the real mixed-outcome registry when local run data exists."""

    if not _FINANCE_11_AUDIT.is_file():
        pytest.skip("finance-11 historical audit is not present in this checkout")
    before = _FINANCE_11_AUDIT.read_bytes()

    result = asyncio.run(replay_finance_recovery(_FINANCE_11_AUDIT))

    assert result["source_registry"] == (
        "posthoc_evidence.pre_edit_evidence"
    )
    assert result["source_registry_profile"] == {
        "claim_count": 81,
        "citation_requirement_counts": {"external": 81},
        "source_cache_count": 7,
        "note_count": 33,
        "observed_internal_recovery_claim_ids": [],
    }
    assert result["triage"]["expected_research_more_matches"] is True
    assert result["triage"]["expected_edit_directly_present"] is True
    assert result["triage"]["research_more_claim_ids"] == [
        "claim-0035",
        "claim-0036",
        "claim-0040",
        "claim-0047",
        "claim-0056",
    ]
    assert {
        claim_id: result["triage"]["decisions"][claim_id]
        for claim_id in ("claim-0053", "claim-0068")
    } == {
        "claim-0053": "edit_directly",
        "claim-0068": "edit_directly",
    }

    internal = result["internal_boundary"]
    assert internal["observed_in_source_registry"] is False
    assert internal["result"] == "not_observable_in_finance_11"
    assert internal["probe"]["passed"] is True
    assert internal["probe"]["triage_status"] == "no_targets"
    assert internal["probe"]["triage_model_calls"] == 0
    assert internal["probe"]["inapplicable_claims"][0]["claim_id"] == (
        "claim-0033"
    )

    empty = result["empty_retrieval"]
    assert empty["stop_reason"] == "no_information_yield"
    assert empty["new_completed_relation_count"] == 0
    assert empty["attempted_claim_ids"] == [
        "claim-0035",
        "claim-0036",
        "claim-0040",
        "claim-0047",
        "claim-0056",
    ]
    assert empty["frozen_equals_triage_research_more"] is True
    assert empty["gap_targets_equal_frozen"] is True
    assert empty["boundary_call_counts"] == {
        "gap_model": 1,
        "note_model": 0,
        "attribution_model": 0,
        "verification_model": 0,
        "tavily": 1,
    }
    empty_attempts = {
        attempt["claim_id"]: attempt for attempt in empty["attempts"]
    }
    assert {
        claim_id: attempt["query_route"]
        for claim_id, attempt in empty_attempts.items()
    } == {
        "claim-0035": "source_chain",
        "claim-0036": "source_chain",
        "claim-0040": "source_chain",
        "claim-0047": "direct_search_fallback",
        "claim-0056": "source_chain",
    }
    assert all(
        empty_attempts[claim_id]["selected_source_lead_id"] is not None
        for claim_id in (
            "claim-0035",
            "claim-0036",
            "claim-0040",
            "claim-0056",
        )
    )
    assert empty_attempts["claim-0047"]["selected_source_lead_id"] is None
    assert empty_attempts["claim-0047"]["source_chain_access"] == (
        "not_applicable_direct_search"
    )
    assert all(
        empty_attempts[claim_id]["source_chain_access"]
        == "lead_search_no_result"
        for claim_id in (
            "claim-0035",
            "claim-0036",
            "claim-0040",
            "claim-0056",
        )
    )

    mixed = result["mixed_verdict_retrieval"]
    assert mixed["new_completed_relation_count"] == 3
    assert mixed["new_completed_verdict_counts"] == {
        "supports": 1,
        "does_not_support": 1,
        "contradicts": 0,
        "not_enough_information": 1,
    }
    assert mixed["frozen_equals_triage_research_more"] is True
    assert mixed["gap_targets_equal_frozen"] is True
    assert mixed["boundary_call_counts"] == {
        "gap_model": 1,
        "note_model": 0,
        "attribution_model": 0,
        "verification_model": 1,
        "tavily": 0,
    }
    by_claim = {
        attempt["claim_id"]: attempt
        for attempt in mixed["attempts"]
    }
    assert by_claim["claim-0035"]["new_completed_verdict_counts"] == {
        "supports": 1
    }
    assert by_claim["claim-0036"]["new_completed_verdict_counts"] == {
        "does_not_support": 1
    }
    assert by_claim["claim-0040"]["new_completed_verdict_counts"] == {
        "not_enough_information": 1
    }

    assert result["source_audit_unchanged"] is True
    assert _FINANCE_11_AUDIT.read_bytes() == before
