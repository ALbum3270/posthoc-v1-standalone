"""Evidence assembly and report rendering from the graph."""

from open_deep_research.graphrag.reporting.evidence_pack import (
    FactRecord,
    build_evidence_pack,
    detect_conflicts,
    fetch_facts,
)
from open_deep_research.graphrag.reporting.report import render_report

__all__ = [
    "FactRecord",
    "build_evidence_pack",
    "detect_conflicts",
    "fetch_facts",
    "render_report",
]
