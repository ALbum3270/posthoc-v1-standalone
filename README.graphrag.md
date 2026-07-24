# Verified GraphRAG research

This repository has two research engines:

- `RESEARCH_MODE=legacy` — the upstream notes-based workflow and compatibility default.
- `RESEARCH_MODE=graphrag` — the graph-first workflow implemented in this branch.

V1 under `v1/` is frozen as an experimental baseline. It is not the production
path.

## What the graph-first path guarantees

The shared runtime executes:

```text
classify conditional ontology slots as optional / not applicable
  → ontology gap from Neo4j
  → query with failure memory
  → Tavily search
  → topic and full-name anchor filtering
  → relevant-passage selection
  → slot-directed extraction
  → exact source-quote and numeric checks
  → pre-write semantic relevance gate
  → verbatim Graphiti write
  → targeted independent support for high-impact claims
  → EvidencePack
  → report
```

Facts are written only when the extractor supplies a quote that can be located
in the selected source text. The graph stores that source passage verbatim; it
does not send the triple through Graphiti for a second LLM extraction.

Source grounding is not treated as universal truth. A grounded candidate must
also pass the slot-relevance gate before it is written. High-impact facts keep
their supporting episode, URL, publisher identity, and quote arrays on the same
edge. A support-only round can attach provenance to an existing exact
subject/predicate/object claim, but cannot create a nearby replacement claim.

Coverage, slot-level source breadth, exact-claim corroboration, and
high-impact support are separate metrics.

## Configuration

Start from `.env.example`. GraphRAG mode requires:

```dotenv
RESEARCH_MODE=graphrag

OPENAI_API_KEY=...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=openai/gpt-4.1-mini
TAVILY_API_KEY=...

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
GRAPH_GROUP_ID=neo4j

GRAPHRAG_MIN_SOURCES_PER_CLAIM=2
GRAPHRAG_ENABLE_RELEVANCE_GATE=true
GRAPHRAG_RELEVANCE_REJECT_THRESHOLD=0.8
GRAPHRAG_ENABLE_SLOT_APPLICABILITY=true
GRAPHRAG_APPLICABILITY_THRESHOLD=0.8
```

The main LangGraph entry remains
`src/open_deep_research/deep_researcher.py:deep_researcher`. In GraphRAG mode,
`write_research_brief` routes to the verified engine and returns the
EvidencePack report directly. Legacy mode continues to route to the original
supervisor/researcher workflow.

## Running one investigation

```bash
python scripts/run_graphrag_research.py \
  "CrowdStrike global IT outage on July 19, 2024" \
  --max-rounds 24 \
  --coverage-target 1.0 \
  --min-sources-per-claim 2
```

The command writes:

- `graphrag_report_<research_id>.md`
- `graphrag_run_<research_id>.json`

The JSON audit includes every query, coverage transition, applicability
decision, pre-write relevance decision, support target edge, rejected search
result count, API token count, provider-reported chat cost, grounding rejection
count, and elapsed time. Embedding calls made inside Graphiti are not included
in the reported chat cost.

## Running the fixed regression suite

The suite uses three stable historical incidents: FTX (2022), CrowdStrike
(2024), and Silicon Valley Bank (2023).

```bash
python scripts/run_graphrag_regression.py \
  --max-rounds 24 \
  --coverage-target 1.0 \
  --min-sources-per-claim 2 \
  --strict
```

This spends real Tavily, chat-model, and embedding calls. Results are written
under `regression_results/<timestamp>/`.

To recompute deterministic checks after changing only evaluation rules:

```bash
python scripts/reevaluate_graphrag_results.py \
  regression_results/<timestamp>
```

That command makes no network calls.

## 2026-07-24 measured history

The first committed M4 run is in `regression_results/m4_20260724/`.

| Case | Graph coverage | Fixed checks | Citation coverage | Model-judged slot relevance | Sources | Chat cost |
|---|---:|---:|---:|---:|---:|---:|
| FTX | 100% | 100% | 100% | 100% | 7 | $0.0130 |
| CrowdStrike | 100% | 100% | 100% | 79% | 11 | $0.0104 |
| SVB | 100% | 67% | 100% | 91% | 6 | $0.0116 |

Across the suite:

- 59,482 chat tokens and $0.0350 provider-reported chat cost;
- 105 source-grounded facts;
- zero duplicate queries;
- every report item carries an Episode citation;
- no slot had facts from two independent sources, so slot-level source breadth was 0%.

The SVB run missed an explicit March 10 closure statement and the
interest-rate/securities-loss explanation. It also placed unrelated facts about
the chemical element silicon into the affected-parties slot. The CrowdStrike
run included bibliography titles and weakly related material in several slots.

Therefore `coverage=100%` means only “each slot has at least one persisted
source-grounded fact.” It does not mean the report is complete or every fact is
relevant.

### M4 metric correction

`m4_verify_20260724` reported “94% cross-corroborated slots.” That name was
wrong. The implementation only established that two sources contributed
possibly different facts to the same slot. It did not establish that two
independent publishers supported the same structured fact.

M5 split that measurement into:

- `multi_source_slot_rate`: source breadth within a slot;
- `claim_corroboration_rate`: exact claims with at least two publisher identities;
- `high_impact_support_rate`: high-impact claims meeting their configured source requirement.

The diagnostic M5 run is committed in
`regression_results/m5_initial_audit_20260724/`:

| Case | Coverage | Fixed checks | Citations | Slot relevance* | Exact-claim corroboration | High-impact support |
|---|---:|---:|---:|---:|---:|---:|
| FTX | 100% | 100% | 100% | 100% | 2% | 6% |
| CrowdStrike | 100% | 100% | 100% | 100% | 17% | 67% |
| SVB | 100% | 83% | 100% | 100% | 4% | 14% |

\* Model-judged after the run; the separate pre-write relevance gate is
serialized in each run audit.

This run is an audit baseline, not final M5 acceptance. It falsified the old
94% interpretation and drove the later targeted-support, entity-lookup, and
causal-query fixes. The current code passes 255 local tests plus compilation
and diff checks. Final acceptance still requires a fresh SVB run followed by
the full three-case live suite with the post-audit code.
