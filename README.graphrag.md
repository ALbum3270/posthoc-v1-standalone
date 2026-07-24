# Verified GraphRAG research

This repository has two research engines:

- `RESEARCH_MODE=legacy` — the upstream notes-based workflow and compatibility default.
- `RESEARCH_MODE=graphrag` — the graph-first workflow implemented in this branch.

V1 under `v1/` is frozen as an experimental baseline. It is not the production
path.

## What the graph-first path guarantees

The shared runtime executes:

```text
ontology gap from Neo4j
  → query with failure memory
  → Tavily search
  → relevant-passage selection
  → slot-directed extraction
  → exact source-quote and numeric checks
  → verbatim Graphiti write
  → EvidencePack
  → report
```

Facts are written only when the extractor supplies a quote that can be located
in the selected source text. The graph stores that source passage verbatim; it
does not send the triple through Graphiti for a second LLM extraction.

This proves source grounding, not universal truth. A real source passage can
still be irrelevant to the assigned ontology slot. The regression suite
measures that separately.

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
  --coverage-target 1.0
```

The command writes:

- `graphrag_report_<research_id>.md`
- `graphrag_run_<research_id>.json`

The JSON audit includes every query, coverage transition, API token count,
provider-reported chat cost, grounding rejection count, and elapsed time.
Embedding calls made inside Graphiti are not included in the reported chat
cost.

## Running the fixed regression suite

The suite uses three stable historical incidents: FTX (2022), CrowdStrike
(2024), and Silicon Valley Bank (2023).

```bash
python scripts/run_graphrag_regression.py \
  --max-rounds 24 \
  --coverage-target 1.0
```

This spends real Tavily, chat-model, and embedding calls. Results are written
under `regression_results/<timestamp>/`.

To recompute deterministic checks after changing only evaluation rules:

```bash
python scripts/reevaluate_graphrag_results.py \
  regression_results/<timestamp>
```

That command makes no network calls.

## 2026-07-24 measured baseline

The committed run is in `regression_results/m4_20260724/`.

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
- no slot had two independent sources, so cross-corroboration was 0%.

The SVB run missed an explicit March 10 closure statement and the
interest-rate/securities-loss explanation. It also placed unrelated facts about
the chemical element silicon into the affected-parties slot. The CrowdStrike
run included bibliography titles and weakly related material in several slots.

Therefore `coverage=100%` means only “each slot has at least one persisted
source-grounded fact.” It does not mean the report is complete or every fact is
relevant.

## Current next quality work

The next work is not another loop rewrite. It is:

1. reject source-grounded passages that do not answer their assigned slot;
2. require two independent sources for high-impact conclusions;
3. distinguish article body from navigation and bibliography text;
4. improve document-level temporal context without inventing dates.
