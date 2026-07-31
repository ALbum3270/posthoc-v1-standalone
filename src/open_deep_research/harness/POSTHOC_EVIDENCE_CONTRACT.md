# Harness post-hoc evidence contract

Version: `posthoc-evidence-v1`

This contract governs the report-writing and evidence-verification stages that
follow collection. It is intentionally independent of any research topic.
Changes to a frozen rule require a new contract version and re-evaluation of
every frozen report fixture.

## One report, with a mandatory evidence companion

Each complete run is a three-file bundle: one reader-facing report,
`<run_id>.md`; its full verbatim evidence companion,
`<run_id>.sources.md`; and its audit JSON. The sources file is not a second
report variant for automated judging. Human reviewers read the same annotated
report that readers receive and may follow any footnote to the companion.

The canonical draft is retained inside the audit solely to reproduce
`anchor_text` locations and diagnose deterministic rendering. It is not
emitted as another report and is not a review surface.

Evidence labels, the evidence summary, and a compact local definition for
every mechanically generated footnote must remain visible in the report.
Full, untruncated verbatim quotes may live in the mandatory companion but
may not be removed or weakened merely to improve aesthetics or reviewer
scores. This rule protects readers from presentation pressure that would
otherwise hide uncertainty.

## Post-hoc attribution

The writing model produces a canonical narrative draft without citations,
footnotes, source URLs, evidence states, or note/source handles. Attribution is
entirely post-hoc; the writer never owns identifiers or candidate-source links.

After drafting, the pipeline:

1. selects externally verifiable assertions;
2. decontextualizes each assertion into a self-contained `claim_text`;
3. locates an exact contiguous `anchor_text` in the canonical draft;
4. links candidate notes and identified sources;
5. verifies each claim against cached source text; and
6. deterministically renders evidence labels and footnotes into the one report.

The claim registry, not rendered Markdown, is the source of truth for
verification results and report-level metrics.

## Frozen atomic-claim granularity

`atomic-v1` defines one claim as one independently truth-valued event or state.
If either coordinated clause could be true while the other is false, the
clauses are separate claims. All truth-conditional entities, times, places,
quantities, negation, modality, and attribution qualifiers remain in the
claim.

`claim_text` is self-contained and may differ from the report wording.
`anchor_text` is a verbatim, contiguous substring of the canonical draft.
The extraction model selects report segment IDs; code resolves the range,
copies the authoritative report bytes, and verifies that it stays inside the
selected Markdown block. Invalid pointers are retained as
`normalization_failed`; they are never clamped, repaired, or silently
discarded.

Granularity is not a score-tuning parameter. Changing it requires a new
granularity contract version and re-scoring all historical fixtures.

## Source fidelity

Writing, verification, and footnote rendering consume `source_quote`, never
`model_quote`. The verifier selects a continuous source segment range; code
copies `source_quote` from the hash-bound cached source and owns its offsets.
Invalid, reversed, oversized, or unknown ranges cannot become supporting
evidence. Historical diagnostic fragments, paraphrases, and unlocatable model
quotes likewise cannot become supporting evidence.

The verifier may judge a source relation as `supports`, `does_not_support`,
`contradicts`, or `not_enough_information`. Execution errors are recorded
separately from semantic verdicts. Missing evidence is not contradiction.

## Verification batching

Claims are grouped by evidence URL. One model call receives one cached source
text and no more than 20 claims. Larger groups are split deterministically by
`claim_id`. Each claim receives an independent result; malformed or omitted
entries are retried individually without discarding valid siblings.

The first implementation uses the full cached source text. It does not add
top-k chunking, embedding retrieval, or excerpt selection. Any later source
text reduction is a separate frozen-fixture A/B experiment. The experiment
must hold verifier, prompt, claims, and batch size constant and report F1,
false-support rate, omissions, tokens, and cost before the optimization can
be adopted.

Independent-publisher counts use normalized publisher domains only as a
reproducible proxy. This is not a strict editorial-independence determination:
common ownership, syndicated or republished material, and one brand operating
multiple domains remain unresolved and visible as audit limitations. No
domain or organization wordlist is used to guess those relationships.

## Evidence states and FEVER compatibility

Fine-grained evidence states remain authoritative. A deterministic coarse
mapping is additionally exposed:

- supported by one publisher or corroborated across publishers: `SUPPORTED`;
- an explicit verifier result of `contradicts`: `REFUTED`;
- conflicting support and contradiction: `CONFLICTING`;
- cited sources that do not support the claim, or no candidate source:
  `NOT_ENOUGH_INFO`;
- normalization failure or verification not run: unmapped.

`REFUTED` is never inferred from failure to find support.

## Checklist/report reconciliation

After the canonical draft and claim registry are frozen, a read-only model
judges whether each frozen checklist item is `covered`, `partially_covered`, or
`not_covered`. The model must identify existing claim IDs and give a reason;
code resolves every accepted claim ID to its block, verbatim anchor, and
character bounds. A covered or partially covered judgement without at least
one mechanically valid report location is rejected and audited rather than
accepted on the model's assertion alone.

Reconciliation is observability only. Its result cannot rewrite the report or
checklist, trigger another writing pass, suppress artifact output, or alter
claim attribution and verification. Invalid or missing model output remains a
separate assessment failure and is never silently reported as `not_covered`.

## Underspecified evaluative claims

After decomposition freezes the claim registry, a separate non-gating pass may
diagnose retained external claims whose comparison scope, evaluation
criterion, or temporal scope is not explicit. The categories may overlap and
are advisory metadata only. They cannot change a claim's selection
disposition, `citation_requirement`, evidence state, rendering, or membership
in any denominator.

The diagnostic result contains no replacement claim collection. It records
the frozen registry hash and external-claim denominator before and after the
pass; both must remain identical. Every external claim receives an assessment.
Missing, duplicate, malformed, or failed model output remains
`diagnostic_failed` and is never converted to `not_underspecified`.

## Reader-visible rendering

The report starts with an evidence summary that separately counts claims with
zero, one, or multiple supporting publisher-domain proxies. A fixed legend
states that a footnote without an additional inline status label means one
publisher supplied a locatable supporting quote. This rule is invariant across
reports and never changes in response to label prevalence. Multi-publisher
support and every exceptional evidence state remain visible at the claim
anchor. One-publisher support is a factual evidence state, not a failed
threshold. `corroboration_target` is a gap-round resource priority signal; the
renderer never reads it. `corroborated` remains reserved for support from at
least two publisher-domain proxies.

Footnote identifiers are assigned in deterministic anchor, claim, and source
order. A compact report definition identifies the publisher-domain proxy,
semantic relation, original URL, and a stable link to the companion evidence
anchor. That companion entry contains the full, untruncated, mechanically
located `source_quote`; machine-only source IDs, character offsets, claim IDs,
and repair details remain in the audit. Every report marker has exactly one
local definition, every definition maps to exactly one companion anchor, and
there are no orphan or duplicate definitions or anchors. Inspected but
non-supporting sources remain in the audit instead of masquerading as ordinary
supporting footnotes. Conflicting sources are displayed only with their
supporting or contradicting relation made explicit.

The companion carries the run ID and links back to the report. Its SHA-256 is
written into both report and audit. Artifact publication stages all three
files, publishes sources first and the digest-bearing report second, and
publishes the audit last as the completion marker. An in-process publication
failure rolls back already published siblings; after an abrupt process crash,
a missing audit marker means the run is incomplete.

## Metrics and external proxy anchors

At minimum the audit records numerators and denominators for:

- `attribution_coverage`;
- `fully_grounded_claim_rate`;
- `citation_support_precision`;
- one-publisher support, multi-publisher support, zero-support, corroborated,
  conflicting, no-candidate, unverified, and normalization failure counts.

The initial attribution-coverage floor is 75% and the stable target is 90%.
The initial fully-grounded-claim target is 75% and the mature target is 90%.
These are external proxy anchors, not same-denominator gold standards.
Coverage and citation accuracy must never be multiplied to manufacture a
joint threshold.

Collection success and evidence quality are separate signals:
`is_success` describes honest checklist termination, while
`evidence_gate_passed` describes post-hoc report evidence.

## Budget and incomplete verification

Collection, writing, claim processing, and verification have separately
reserved usage. Admission estimates are calibrated from observed model usage.
If the remaining budget cannot admit every verification call, unprocessed
claims remain in the registry as `verification_not_run` and are visibly
marked; they are not dropped.

## Frozen real-run fixture

`tests/fixtures/harness_posthoc_b1407b.json` is the `posthoc-evidence-v1`
offline fixture. It preserves the fifth run's report, checklist terminal
state, usage summary, all 32 notes, and all four complete cached source texts.
It also preserves the legacy report's duplicate-footnote defect and the
`atomic-v1` selection, decontextualization, and decomposition examples.
