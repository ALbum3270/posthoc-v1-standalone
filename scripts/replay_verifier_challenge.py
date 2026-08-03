"""Re-verify one run's frozen claim/source pairs with a different model.

Everything except the model is held fixed: the same claims, the same cached
source bytes, and the production verification path -- `verify_attributions`
builds the prompt, resolves spans and parses results, so this measures the
model rather than a reimplementation of the verifier.

The output is a prediction file scored offline against the frozen challenge
set. It never writes into a run bundle and never touches the frozen cases.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import AsyncOpenAI  # noqa: E402

from open_deep_research.harness.attribution import ClaimAttribution  # noqa: E402
from open_deep_research.harness.verify import (  # noqa: E402
    VerificationVerdict,
    verify_attributions,
)
from open_deep_research.harness.verifier_quality import (  # noqa: E402
    VerifierPrediction,
)
from run_harness import OpenAIEnvelopeModel, configure_openrouter_proxy  # noqa: E402


class _JsonClient:
    """Adapt the envelope model to the verifier's client protocol."""

    def __init__(self, model: OpenAIEnvelopeModel) -> None:
        self._model = model
        self.token_count = 0
        self.cost_usd = 0.0

    async def generate(self, prompt: str) -> dict[str, object]:
        # The verifier validates a full usage envelope, not a bare string; a
        # string here is rejected as ``verification_model_error`` and every
        # verdict silently becomes null.
        envelope = await self._model.generate(prompt)
        self.token_count += int(envelope["token_count"])
        self.cost_usd += float(envelope["cost_usd"])
        return envelope


def _view(audit: dict, view: str) -> dict:
    posthoc = audit.get("posthoc_evidence") or {}
    if view == "pre_edit":
        return posthoc.get("pre_edit_evidence") or {}
    return posthoc


async def _run(args: argparse.Namespace) -> int:
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    packet = json.loads(args.fixture.read_text(encoding="utf-8"))
    run_id = str(audit["run_id"])

    evidence = _view(audit, args.view)
    raw = (evidence.get("attribution") or {}).get("attributions") or ()
    attributions = tuple(ClaimAttribution.model_validate(a) for a in raw)
    source_cache = dict((audit.get("ledger") or {}).get("source_cache") or {})
    if not attributions:
        raise SystemExit(f"no attributions in view {args.view}")

    configure_openrouter_proxy(os.environ.get("OPENAI_BASE_URL") or None)
    openai = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )
    client = _JsonClient(
        OpenAIEnvelopeModel(openai, args.model, json_mode=True)
    )

    result = await verify_attributions(
        attributions,
        source_cache=source_cache,
        model_client=client,
    )

    wanted = {case["case_id"] for case in packet["cases"]}
    predictions: list[dict[str, object]] = []
    for claim_result in result.claims:
        for index, relation in enumerate(claim_result.relations):
            case_id = (
                f"{run_id}:{args.view}:{claim_result.claim.claim_id}:"
                f"{relation.source_id}:{index}"
            )
            if case_id not in wanted:
                continue
            verdict = relation.semantic_verdict
            if verdict is None:
                continue
            predictions.append(
                VerifierPrediction(
                    case_id=case_id,
                    verdict=VerificationVerdict(verdict),
                    explanation=str(relation.explanation or ""),
                ).model_dump(mode="json")
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "system_id": args.model,
                "source_run_id": run_id,
                "audit_view": args.view,
                "cases_sha256": packet["cases_sha256"],
                "token_count": client.token_count,
                "cost_usd": round(client.cost_usd, 6),
                "predictions": predictions,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    matched = len(predictions)
    print(f"model      = {args.model}")
    print(f"predictions= {matched}/{len(wanted)} frozen cases")
    print(f"tokens     = {client.token_count:,}  cost = ${client.cost_usd:.4f}")
    print(f"wrote      = {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--view", choices=("post_edit", "pre_edit"), default="post_edit"
    )
    parser.add_argument("--output", type=Path, required=True)
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
