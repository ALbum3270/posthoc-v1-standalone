#!/usr/bin/env python
"""Run the standalone model-directed research harness."""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tavily import AsyncTavilyClient

from open_deep_research.harness.loop import LoopBudget, LoopSettings
from open_deep_research.harness.budget import (
    RunCostBudget,
    RunCostCapReached,
)
from open_deep_research.harness.disagreement import (
    DisagreementBudget,
    PosthocRetrievalBudget,
)
from open_deep_research.harness.evidence_gap import EvidenceGapBudget
from open_deep_research.harness.runner import HarnessRunResult, run_harness

_OPENROUTER_PROXY = "http://127.0.0.1:7890"


@dataclass
class UsageCalibration:
    """Observed prompt-character to provider-usage calibration."""

    prompt_chars: int = 0
    token_count: int = 0
    cost_usd: float = 0.0

    def observe(
        self,
        *,
        prompt_chars: int,
        token_count: int,
        cost_usd: float,
    ) -> None:
        self.prompt_chars += max(0, prompt_chars)
        self.token_count += max(0, token_count)
        self.cost_usd += max(0.0, cost_usd)

    def estimate_tokens(self, prompt: str) -> int:
        if self.prompt_chars <= 0:
            raise RuntimeError("no observed usage is available for admission")
        return math.ceil(
            len(prompt) * self.token_count / self.prompt_chars
        )

    def estimate_cost_usd(self, prompt: str) -> float:
        if self.prompt_chars <= 0:
            raise RuntimeError("no observed usage is available for admission")
        return len(prompt) * self.cost_usd / self.prompt_chars


class OpenAIEnvelopeModel:
    """Adapt Chat Completions to the harness usage-envelope contract."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        *,
        json_mode: bool = True,
        calibration: UsageCalibration | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.json_mode = json_mode
        self.calibration = calibration or UsageCalibration()
        self.last_usage = {"token_count": 0, "cost_usd": 0.0}

    async def generate(self, prompt: str) -> dict[str, object]:
        # Roles that must return JSON ask the provider to enforce it; the report
        # writer must not, because its output is markdown.
        if self.json_mode:
            # Some OpenAI-compatible providers reject json_object unless the
            # message itself contains the literal word "json". Keep this
            # adapter-level invariant even if a future role prompt omits it.
            if "json" not in prompt.casefold():
                prompt = "Return one JSON object.\n\n" + prompt
            extra: dict[str, object] = {
                "response_format": {"type": "json_object"}
            }
        else:
            extra = {}
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **extra,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("chat completion returned no text content")

        usage_data = (
            response.usage.model_dump()
            if response.usage is not None
            else {}
        )
        token_count = usage_data.get("total_tokens", 0)
        cost_usd = usage_data.get(
            "cost_usd",
            usage_data.get("cost", 0.0),
        )
        self.last_usage = {
            "token_count": int(token_count or 0),
            "cost_usd": float(cost_usd or 0.0),
        }
        self.calibration.observe(
            prompt_chars=len(prompt),
            token_count=int(token_count or 0),
            cost_usd=float(cost_usd or 0.0),
        )
        return {"content": content, **self.last_usage}

    def estimate_tokens(self, prompt: str) -> int:
        """Estimate from observed real provider usage, never a fixed ratio."""

        return self.calibration.estimate_tokens(prompt)

    def estimate_cost_usd(self, prompt: str) -> float:
        """Estimate from observed real provider cost, never a fixed ratio."""

        return self.calibration.estimate_cost_usd(prompt)


class ChecklistOpenAIModel:
    """Expose plain checklist content while retaining its measured usage."""

    def __init__(self, envelope_model: OpenAIEnvelopeModel) -> None:
        self.envelope_model = envelope_model
        self.last_usage = {"token_count": 0, "cost_usd": 0.0}

    async def generate(self, prompt: str) -> str:
        response = await self.envelope_model.generate(prompt)
        self.last_usage = {
            "token_count": response["token_count"],
            "cost_usd": response["cost_usd"],
        }
        return str(response["content"])

    def estimate_cost_usd(self, prompt: str) -> float:
        return self.envelope_model.estimate_cost_usd(prompt)


@dataclass(frozen=True)
class LiveClients:
    """Owned provider clients and separately configured model roles."""

    openai: AsyncOpenAI
    tavily: AsyncTavilyClient
    checklist_model: ChecklistOpenAIModel
    decision_model: OpenAIEnvelopeModel
    note_model: OpenAIEnvelopeModel
    write_model: OpenAIEnvelopeModel
    claim_model: OpenAIEnvelopeModel
    reconciliation_model: OpenAIEnvelopeModel
    attribution_model: OpenAIEnvelopeModel
    verification_model: OpenAIEnvelopeModel
    editor_model: OpenAIEnvelopeModel
    recovery_model: OpenAIEnvelopeModel
    decision_model_name: str
    note_model_name: str
    claim_model_name: str
    reconciliation_model_name: str
    attribution_model_name: str
    verification_model_name: str
    editor_model_name: str
    recovery_model_name: str

    async def close(self) -> None:
        await self.openai.close()
        await self.tavily.close()


def _model_name(role: str, default: str) -> str:
    value = os.environ.get(f"HARNESS_{role.upper()}_MODEL", default).strip()
    if not value:
        raise RuntimeError(f"HARNESS_{role.upper()}_MODEL must not be blank")
    return value


def configure_openrouter_proxy(base_url: str | None) -> None:
    """Supply the known local proxy only when OpenRouter has no proxy configured."""

    hostname = urlparse(base_url or "").hostname or ""
    if hostname.lower() != "openrouter.ai":
        return
    os.environ.setdefault("https_proxy", _OPENROUTER_PROXY)
    os.environ.setdefault("HTTPS_PROXY", os.environ["https_proxy"])


def build_live_clients() -> LiveClients:
    """Construct real provider clients from the loaded environment."""

    openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    tavily_api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    missing = [
        name
        for name, value in (
            ("OPENAI_API_KEY", openai_api_key),
            ("TAVILY_API_KEY", tavily_api_key),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "missing required configuration: " + ", ".join(missing)
        )

    base_url = os.environ.get("OPENAI_BASE_URL") or None
    configure_openrouter_proxy(base_url)
    default_model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini").strip()
    if not default_model:
        raise RuntimeError("OPENAI_MODEL must not be blank")
    decision_model_name = _model_name("decision", default_model)
    note_model_name = _model_name("note", default_model)
    claim_model_name = _model_name("claim", decision_model_name)
    attribution_model_name = _model_name(
        "attribution", decision_model_name
    )
    reconciliation_model_name = _model_name(
        "reconciliation", attribution_model_name
    )
    verification_model_name = _model_name("verification", default_model)
    editor_model_name = _model_name("editor", claim_model_name)
    # Recovery triage decides whether an audited content exception is central
    # enough to justify one more bounded research attempt. Keep it separately
    # configurable, while defaulting to the existing editorial judgement tier.
    recovery_model_name = _model_name("recovery", editor_model_name)

    openai = AsyncOpenAI(api_key=openai_api_key, base_url=base_url)
    tavily = AsyncTavilyClient(api_key=tavily_api_key)
    checklist_envelope = OpenAIEnvelopeModel(
        openai,
        decision_model_name,
        calibration=UsageCalibration(),
    )
    return LiveClients(
        openai=openai,
        tavily=tavily,
        checklist_model=ChecklistOpenAIModel(checklist_envelope),
        decision_model=OpenAIEnvelopeModel(
            openai,
            decision_model_name,
            calibration=UsageCalibration(),
        ),
        note_model=OpenAIEnvelopeModel(
            openai,
            note_model_name,
            calibration=UsageCalibration(),
        ),
        write_model=OpenAIEnvelopeModel(
            openai,
            decision_model_name,
            json_mode=False,
            calibration=UsageCalibration(),
        ),
        claim_model=OpenAIEnvelopeModel(
            openai,
            claim_model_name,
            calibration=UsageCalibration(),
        ),
        reconciliation_model=OpenAIEnvelopeModel(
            openai,
            reconciliation_model_name,
            calibration=UsageCalibration(),
        ),
        attribution_model=OpenAIEnvelopeModel(
            openai,
            attribution_model_name,
            calibration=UsageCalibration(),
        ),
        verification_model=OpenAIEnvelopeModel(
            openai,
            verification_model_name,
            calibration=UsageCalibration(),
        ),
        editor_model=OpenAIEnvelopeModel(
            openai,
            editor_model_name,
            calibration=UsageCalibration(),
        ),
        recovery_model=OpenAIEnvelopeModel(
            openai,
            recovery_model_name,
            calibration=UsageCalibration(),
        ),
        decision_model_name=decision_model_name,
        note_model_name=note_model_name,
        claim_model_name=claim_model_name,
        reconciliation_model_name=reconciliation_model_name,
        attribution_model_name=attribution_model_name,
        verification_model_name=verification_model_name,
        editor_model_name=editor_model_name,
        recovery_model_name=recovery_model_name,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without loading configuration."""

    parser = argparse.ArgumentParser(
        description="Run the standalone model-directed research harness.",
    )
    parser.add_argument("topic", help="research topic")
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help=(
            "optional collection round cap. A fixed turn count is not an "
            "independent resource: a cheap link inspection and an expensive "
            "read spend the same one round, so a default here stops research "
            "with cost still unspent. Without it the cost cap is the finite "
            "boundary"
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "optional cumulative collection token cap. Without it, the "
            "recoverable round and cost caps bound collection; this is not a "
            "per-call input-size guard"
        ),
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=10.0,
        help=(
            "absolute run cost cap enforced by pre-call admission. Not a "
            "runaway-only guard: it sits close enough to normal run cost that "
            "it can and does interrupt legitimate work. Provider cost is known "
            "only after a call, so one admitted call may overshoot"
        ),
    )
    parser.add_argument(
        "--cost-objective-usd",
        type=float,
        default=None,
        help=(
            "advisory per-report cost target. Never blocks a call and never "
            "becomes a stop reason; exceeding it is recorded as an audit "
            "event only. No default, because no measurement so far shows any "
            "particular figure to be a defensible target"
        ),
    )
    parser.add_argument(
        "--collection-max-cost-usd",
        type=float,
        help=(
            "optional collection-only sub-cap; defaults to the run-level "
            "ceiling and cannot enlarge it"
        ),
    )
    parser.add_argument(
        "--evidence-tail-cost-reserve-usd",
        type=float,
        default=0.0,
        help=(
            "initial estimated reserve for claim decomposition, attribution, "
            "initial verification, checklist reconciliation, and deterministic "
            "rendering, plus one audit-after-edit pass and a full re-audit when "
            "the draft changes. It is recalculated from observed work units and is not "
            "a provider-side guarantee; no dollar amount is inferred by default"
        ),
    )
    parser.add_argument(
        "--verification-cost-reserve-usd",
        type=float,
        default=0.0,
        help=(
            "deprecated verifier-only reserve retained for historical command "
            "lines; use --evidence-tail-cost-reserve-usd"
        ),
    )
    parser.add_argument(
        "--writing-token-reserve",
        type=int,
        default=0,
        help="protect this many tokens from collection for report writing",
    )
    parser.add_argument(
        "--writing-cost-reserve-usd",
        type=float,
        default=0.0,
        help="protect this much cost from collection for report writing",
    )
    parser.add_argument(
        "--max-malformed-actions",
        type=int,
        default=3,
        help="stop after this many consecutive malformed decision actions",
    )
    parser.add_argument(
        "--source-char-limit",
        type=int,
        default=100_000,
        help=(
            "budget for source text the model recalls into a decision prompt; "
            "whether to recall at all is the model's decision, not a flag"
        ),
    )
    parser.add_argument(
        "--note-page-size",
        type=int,
        default=8,
        help="number of note summaries returned by inspect_notes",
    )
    parser.add_argument(
        "--source-link-page-size",
        type=int,
        default=64,
        help=(
            "number of mechanically captured link identities returned by "
            "inspect_source_links"
        ),
    )
    parser.add_argument(
        "--provider-timeout-seconds",
        type=float,
        default=60.0,
        help=(
            "local deadline for each Tavily search or extraction call "
            "(1-60 seconds); failures remain auditable and degradable"
        ),
    )
    parser.add_argument(
        "--max-recalled-notes",
        type=int,
        default=8,
        help="maximum full notes injected by one recall_notes action",
    )
    parser.add_argument(
        "--corroboration-target",
        "--verification-required-sources",
        dest="corroboration_target",
        type=int,
        choices=(1, 2),
        default=2,
        help=(
            "gap-round resource target for publisher corroboration; the "
            "legacy --verification-required-sources spelling is accepted"
        ),
    )
    parser.add_argument("--evidence-gap-max-tokens", type=int, default=60_000)
    parser.add_argument(
        "--evidence-gap-max-cost-usd",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--evidence-gap-max-searches",
        type=int,
        default=6,
        help=(
            "maximum searches in the bounded evidence-gap pass; targets "
            "without an allocated route remain explicitly deferred"
        ),
    )
    parser.add_argument("--evidence-gap-max-reads", type=int, default=3)
    parser.add_argument(
        "--evidence-recovery-max-tokens",
        type=int,
        default=40_000,
        help=(
            "independent token cap for the one bounded evidence-recovery "
            "pass; its target claim set is frozen before retrieval and the "
            "pass never starts an automatic second round"
        ),
    )
    parser.add_argument(
        "--evidence-recovery-max-cost-usd",
        type=float,
        default=0.08,
        help=(
            "independent cost cap for recovery retrieval and re-verification; "
            "all recovery calls also remain inside --max-cost-usd"
        ),
    )
    parser.add_argument(
        "--evidence-recovery-max-searches",
        type=int,
        default=3,
        help=(
            "maximum searches in the single recovery pass; an upper bound, "
            "not a target"
        ),
    )
    parser.add_argument(
        "--evidence-recovery-max-reads",
        type=int,
        default=3,
        help=(
            "maximum source reads in the single recovery pass; an upper "
            "bound, not a target"
        ),
    )
    parser.add_argument(
        "--posthoc-retrieval-max-tokens",
        type=int,
        default=110_000,
        help=(
            "shared token cap across evidence-gap and disagreement passes; "
            "the default is the sum of both independent pass caps, so "
            "reserving disagreement capacity does not reduce evidence-gap "
            "capacity"
        ),
    )
    parser.add_argument(
        "--posthoc-retrieval-max-cost-usd",
        type=float,
        default=0.16,
        help=(
            "shared cost cap across evidence-gap and disagreement passes; "
            "the default is the sum of both independent pass caps, so "
            "reserving disagreement capacity does not reduce evidence-gap "
            "capacity"
        ),
    )
    parser.add_argument(
        "--disagreement-max-tokens",
        type=int,
        default=50_000,
    )
    parser.add_argument(
        "--disagreement-max-cost-usd",
        type=float,
        default=0.06,
    )
    parser.add_argument("--disagreement-max-claims", type=int, default=6)
    parser.add_argument("--disagreement-max-searches", type=int, default=3)
    parser.add_argument("--disagreement-max-reads", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("harness_runs"),
    )
    parser.add_argument("--run-id")
    return parser


async def _run(args: argparse.Namespace) -> HarnessRunResult:
    clients = build_live_clients()
    try:
        return await run_harness(
            args.topic,
            checklist_model=clients.checklist_model,
            decision_model=clients.decision_model,
            note_model=clients.note_model,
            write_model=clients.write_model,
            claim_model=clients.claim_model,
            reconciliation_model=clients.reconciliation_model,
            attribution_model=clients.attribution_model,
            verification_model=clients.verification_model,
            editor_model=clients.editor_model,
            recovery_model=clients.recovery_model,
            tavily_client=clients.tavily,
            budget=LoopBudget(
                max_rounds=args.max_rounds,
                max_tokens=args.max_tokens,
                max_cost_usd=(
                    args.collection_max_cost_usd
                    if args.collection_max_cost_usd is not None
                    else args.max_cost_usd
                ),
                writing_token_reserve=args.writing_token_reserve,
                writing_cost_reserve_usd=args.writing_cost_reserve_usd,
                max_consecutive_malformed_actions=args.max_malformed_actions,
            ),
            loop_settings=LoopSettings(
                decision_source_char_limit=args.source_char_limit,
                note_page_size=args.note_page_size,
                source_link_page_size=args.source_link_page_size,
                provider_timeout_seconds=args.provider_timeout_seconds,
                max_recalled_notes=args.max_recalled_notes,
            ),
            output_dir=args.output_dir,
            run_id=args.run_id,
            corroboration_target_for_external_claims=(
                args.corroboration_target
            ),
            evidence_gap_budget=EvidenceGapBudget(
                max_tokens=args.evidence_gap_max_tokens,
                max_cost_usd=args.evidence_gap_max_cost_usd,
                max_search_queries=args.evidence_gap_max_searches,
                max_reads=args.evidence_gap_max_reads,
                provider_timeout_seconds=args.provider_timeout_seconds,
            ),
            evidence_recovery_budget=EvidenceGapBudget(
                max_tokens=args.evidence_recovery_max_tokens,
                max_cost_usd=args.evidence_recovery_max_cost_usd,
                max_search_queries=args.evidence_recovery_max_searches,
                max_reads=args.evidence_recovery_max_reads,
                provider_timeout_seconds=args.provider_timeout_seconds,
            ),
            disagreement_budget=DisagreementBudget(
                max_tokens=args.disagreement_max_tokens,
                max_cost_usd=args.disagreement_max_cost_usd,
                max_selected_claims=args.disagreement_max_claims,
                max_search_queries=args.disagreement_max_searches,
                max_reads=args.disagreement_max_reads,
                provider_timeout_seconds=args.provider_timeout_seconds,
            ),
            posthoc_retrieval_budget=PosthocRetrievalBudget(
                max_tokens=args.posthoc_retrieval_max_tokens,
                max_cost_usd=args.posthoc_retrieval_max_cost_usd,
            ),
            run_cost_budget=RunCostBudget(
                max_cost_usd=args.max_cost_usd,
                evidence_tail_reserve_usd=(
                    args.evidence_tail_cost_reserve_usd
                ),
                verification_reserve_usd=(
                    args.verification_cost_reserve_usd
                ),
                cost_objective_usd=args.cost_objective_usd,
            ),
            model_names={
                "decision": clients.decision_model_name,
                "note": clients.note_model_name,
                "writing": clients.decision_model_name,
                "claim": clients.claim_model_name,
                "reconciliation": clients.reconciliation_model_name,
                "attribution": clients.attribution_model_name,
                "verification": clients.verification_model_name,
                "editorial": clients.editor_model_name,
                "recovery": clients.recovery_model_name,
            },
        )
    finally:
        await clients.close()


def main() -> int:
    """Parse arguments, load repository-local configuration and run."""

    args = build_parser().parse_args()
    load_dotenv(Path(__file__).resolve().parent / ".env")
    try:
        result = asyncio.run(_run(args))
    except RunCostCapReached as error:
        # A cap hit is an expected outcome, not a crash. Print where the money
        # went instead of a traceback, so the next decision -- raise the cap,
        # or fix whatever was burning it -- has something to stand on.
        print(error.report(), file=sys.stderr)
        return 2
    print(result.report_path)
    print(result.sources_path)
    print(result.audit_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
