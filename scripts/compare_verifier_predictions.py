"""Put several models' verdicts on the frozen verifier cases side by side.

Two layers, deliberately kept apart:

* the official metric, which scores only against human-reviewed gold and
  therefore reports null while the reference labels remain provisional;
* a provisional comparison table, which is useful for reading disagreement but
  is explicitly not an accuracy measurement.

Collapsing the two is the mistake this file exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_deep_research.harness.verifier_quality import (  # noqa: E402
    FrozenVerifierCase,
    VerifierGold,
    VerifierGoldStatus,
    VerifierPrediction,
    evaluate_verifier_challenge,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    packet = json.loads(args.fixture.read_text(encoding="utf-8"))
    cases = tuple(FrozenVerifierCase.model_validate(c) for c in packet["cases"])
    gold = tuple(VerifierGold.model_validate(g) for g in packet["gold"])
    by_case = {c.case_id: c for c in cases}
    gold_by_case = {g.case_id: g for g in gold}

    runs: dict[str, dict[str, str]] = {}
    for path in args.prediction:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["cases_sha256"] != packet["cases_sha256"]:
            raise SystemExit(f"{path} was produced against different cases")
        runs[str(payload["system_id"])] = {
            p["case_id"]: p["verdict"] for p in payload["predictions"]
        }
        metrics = evaluate_verifier_challenge(
            cases=cases,
            gold=gold,
            predictions=tuple(
                VerifierPrediction.model_validate(p)
                for p in payload["predictions"]
            ),
            system_id=str(payload["system_id"]),
        )
        print(
            f"官方指标 {payload['system_id']}: "
            f"reviewed={len(metrics.reviewed_case_ids)} "
            f"pending={len(metrics.pending_case_ids)} "
            f"accuracy={metrics.accuracy}"
        )

    print(
        "\n（accuracy 为 None 是正确的：参考标签仍是 provisional_model_review，"
        "不是人审 gold。下表只是分歧视图，不是准确率。）\n"
    )

    labelled = [
        g
        for g in gold
        if g.review_status is VerifierGoldStatus.PROVISIONAL_MODEL_REVIEW
    ]
    names = list(runs)
    header = f"{'断言（截断）':<34}{'原判':<22}" + "".join(
        f"{n.split('/')[-1]:<22}" for n in names
    ) + "provisional 参考"
    print(header)
    print("-" * len(header))
    agree = {n: 0 for n in names}
    for g in labelled:
        case = by_case[g.case_id]
        row = f"{case.claim_text[:30]:<34}{case.original_verdict.value:<22}"
        for n in names:
            v = runs[n].get(g.case_id, "-")
            row += f"{v:<22}"
            if v == g.verdict.value:
                agree[n] += 1
        row += g.verdict.value
        print(row)
    print()
    total = len(labelled)
    original_agree = sum(
        1 for g in labelled if by_case[g.case_id].original_verdict == g.verdict
    )
    print(f"与 provisional 参考一致（{total} 条）：")
    print(f"  finance-24 原判（gpt-4.1-mini）: {original_agree}/{total}")
    for n in names:
        print(f"  {n}: {agree[n]}/{total}")

    print("\n全部 44 条上各模型的判定分布：")
    from collections import Counter

    print(
        f"  原判: "
        f"{dict(Counter(c.original_verdict.value for c in cases))}"
    )
    for n in names:
        print(f"  {n}: {dict(Counter(runs[n].values()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
