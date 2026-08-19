# posthoc-v1 standalone branch

This branch contains only the model-directed `posthoc-v1` research harness.
It intentionally excludes the experimental evidence-v2, evidence-v2.1, and
evidence-v2.2 planners, controllers, section workspaces, applicability stages,
ablation tooling, and evaluation fixtures.

The executable entry point is `run_harness.py`. Its pipeline is:

1. model-authored checklist;
2. model-directed search, read, recall, and settlement loop;
3. one whole-report draft;
4. claim decomposition, checklist reconciliation, attribution, and verification;
5. bounded evidence-gap, disagreement, and recovery passes;
6. optional audited editing followed by a complete re-audit when bytes change;
7. deterministic report, source companion, and audit publication.

The semantic choices remain model-owned. Code enforces only protocol identity,
source/span fidelity, resource limits, recoverable failure records, and artifact
integrity. The normative post-draft rules are frozen in
`src/open_deep_research/harness/POSTHOC_EVIDENCE_CONTRACT.md`.

Run the offline suite with:

```bash
PYTHONPATH=src python -m pytest -q tests/
```

The branch starts from the last pre-v2 architecture commit and carries only
three later provider-independent reliability fixes relevant to v1: charged
usage preservation on model failure, one bounded model-selected retry for an
oversized evidence span, and recoverable rejection of unusable search-result
URLs.
