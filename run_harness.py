#!/usr/bin/env python
"""Run the standalone model-directed research harness."""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tavily import AsyncTavilyClient

from open_deep_research.harness.loop import LoopBudget, LoopSettings
from open_deep_research.harness.runner import HarnessRunResult, run_harness

_OPENROUTER_PROXY = "http://127.0.0.1:7890"


class OpenAIEnvelopeModel:
    """Adapt Chat Completions to the harness usage-envelope contract."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        *,
        json_mode: bool = True,
    ) -> None:
        self.client = client
        self.model = model
        self.json_mode = json_mode
        self.last_usage = {"token_count": 0, "cost_usd": 0.0}

    async def generate(self, prompt: str) -> dict[str, object]:
        # Roles that must return JSON ask the provider to enforce it; the report
        # writer must not, because its output is markdown.
        extra: dict[str, object] = (
            {"response_format": {"type": "json_object"}} if self.json_mode else {}
        )
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
        return {"content": content, **self.last_usage}


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
    attribution_model: OpenAIEnvelopeModel
    verification_model: OpenAIEnvelopeModel
    decision_model_name: str
    note_model_name: str
    claim_model_name: str
    attribution_model_name: str
    verification_model_name: str

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
    verification_model_name = _model_name("verification", default_model)

    openai = AsyncOpenAI(api_key=openai_api_key, base_url=base_url)
    tavily = AsyncTavilyClient(api_key=tavily_api_key)
    checklist_envelope = OpenAIEnvelopeModel(openai, decision_model_name)
    return LiveClients(
        openai=openai,
        tavily=tavily,
        checklist_model=ChecklistOpenAIModel(checklist_envelope),
        decision_model=OpenAIEnvelopeModel(openai, decision_model_name),
        note_model=OpenAIEnvelopeModel(openai, note_model_name),
        write_model=OpenAIEnvelopeModel(
            openai, decision_model_name, json_mode=False
        ),
        claim_model=OpenAIEnvelopeModel(openai, claim_model_name),
        attribution_model=OpenAIEnvelopeModel(
            openai, attribution_model_name
        ),
        verification_model=OpenAIEnvelopeModel(
            openai, verification_model_name
        ),
        decision_model_name=decision_model_name,
        note_model_name=note_model_name,
        claim_model_name=claim_model_name,
        attribution_model_name=attribution_model_name,
        verification_model_name=verification_model_name,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser without loading configuration."""

    parser = argparse.ArgumentParser(
        description="Run the standalone model-directed research harness.",
    )
    parser.add_argument("topic", help="research topic")
    parser.add_argument("--max-rounds", type=int, default=25)
    parser.add_argument("--max-tokens", type=int, default=100_000)
    parser.add_argument("--max-cost-usd", type=float, default=10.0)
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
        "--max-recalled-notes",
        type=int,
        default=8,
        help="maximum full notes injected by one recall_notes action",
    )
    parser.add_argument(
        "--verification-required-sources",
        type=int,
        choices=(1, 2),
        default=2,
        help=(
            "independent-source proxy requirement for each externally "
            "verifiable report claim"
        ),
    )
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
            attribution_model=clients.attribution_model,
            verification_model=clients.verification_model,
            tavily_client=clients.tavily,
            budget=LoopBudget(
                max_rounds=args.max_rounds,
                max_tokens=args.max_tokens,
                max_cost_usd=args.max_cost_usd,
                writing_token_reserve=args.writing_token_reserve,
                writing_cost_reserve_usd=args.writing_cost_reserve_usd,
                max_consecutive_malformed_actions=args.max_malformed_actions,
            ),
            loop_settings=LoopSettings(
                decision_source_char_limit=args.source_char_limit,
                note_page_size=args.note_page_size,
                max_recalled_notes=args.max_recalled_notes,
            ),
            output_dir=args.output_dir,
            run_id=args.run_id,
            verification_required_independent_sources=(
                args.verification_required_sources
            ),
            model_names={
                "decision": clients.decision_model_name,
                "note": clients.note_model_name,
                "writing": clients.decision_model_name,
                "claim": clients.claim_model_name,
                "attribution": clients.attribution_model_name,
                "verification": clients.verification_model_name,
            },
        )
    finally:
        await clients.close()


def main() -> int:
    """Parse arguments, load repository-local configuration and run."""

    args = build_parser().parse_args()
    load_dotenv(Path(__file__).resolve().parent / ".env")
    result = asyncio.run(_run(args))
    print(result.report_path)
    print(result.audit_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
