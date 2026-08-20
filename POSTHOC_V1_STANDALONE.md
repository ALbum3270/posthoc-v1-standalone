# posthoc-v1 standalone branch

This branch contains only the model-directed `posthoc-v1` research harness.
It intentionally excludes the experimental evidence-v2, evidence-v2.1, and
evidence-v2.2 planners, controllers, section workspaces, applicability stages,
ablation tooling, and evaluation fixtures.

The executable entry point is `run_harness.py`. Its pipeline is:

1. model-authored checklist;
2. model-directed search, read, recall, and settlement loop;
3. one whole-report draft;
4. claim decomposition, independent truth-condition review, checklist
   reconciliation, attribution, and element-level verification;
5. bounded evidence-gap, disagreement, and recovery passes;
6. optional audited editing followed by a hash/range-bound transaction review
   and complete changed-region re-audit before any bytes commit;
7. deterministic report, source companion, and audit publication.

The default reader artifact uses `clean-reader-v2`: `report.md` contains the
narrative, compact citations, and only material unresolved warnings;
`sources.md` carries full quotes and mechanical audit summaries; `audit.json`
retains the complete machine state. Historical presentation remains available
with `--reader-report-style audit-annotated-v1`.

The semantic choices remain model-owned. Code enforces only protocol identity,
source/span fidelity, resource limits, recoverable failure records, and artifact
integrity. The normative post-draft rules are frozen in
`src/open_deep_research/harness/POSTHOC_EVIDENCE_CONTRACT.md`.

Run the offline suite with:

```bash
PYTHONPATH=src python -m pytest -q tests/
```

The branch starts from the last pre-v2 architecture commit and carries only
provider-independent reliability changes relevant to this pipeline. In
addition to charged-usage preservation and recoverable provider failures, the
live path now closes the truth-condition denominator, keeps semantic verdicts
separate from execution completeness, preserves element intent through
recovery, performs one bounded model-selected retry for oversized evidence
spans, and rolls unsafe editorial proposals back while retaining their audit.
