"""
V1 最小闭环主循环。

验证核心问题："图谱空洞能否指导搜索词生成，并让调查收敛？"

用法:
    python -m v1.run "特斯拉 2024 年交付量下滑"
    python -m v1.run "FTX 暴雷事件" --max-rounds 8 --neo4j-uri bolt://localhost:7687

环境变量（必须）:
    OPENAI_API_KEY
    TAVILY_API_KEY

环境变量（可选）:
    OPENAI_BASE_URL   默认 https://api.openai.com/v1
    OPENAI_MODEL      默认 gpt-4o-mini

环境变量（连接 Neo4j，不填则尝试默认值）:
    NEO4J_URI      默认 bolt://localhost:7687
    NEO4J_USER     默认 neo4j
    NEO4J_PASSWORD 默认 password
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

# 确保 open_deep_research-main 目录在 sys.path
_BASE = Path(__file__).resolve().parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

from dotenv import load_dotenv
load_dotenv(_BASE / ".env")  # 加载 .env（如果存在）

from v1.ontology import SLOTS, all_filled, coverage_pct, get_empty_slots
from v1.supervisor import pick_next_slot_and_query
from v1.searcher import search as tavily_search
from v1.extractor import extract_triples
from v1.graph_writer import write_slot_episode
from v1.reporter import EvidenceRef, build_report


MAX_ROUNDS_DEFAULT = len(SLOTS) + 8
SEARCH_RESULTS_PER_ROUND = 5   # 多取结果，给 URL 去重和失败退避留余量
PAGES_TO_EXTRACT_PER_ROUND = 3
MAX_CHARS_PER_PAGE = 2000      # 每页送入抽取器的相关文本预算

# graphiti 0.28.2 会把 group_id 当 Neo4j 数据库名（graphiti.py:884-888：
# group_id != driver._database 时 driver.clone(database=group_id)）。
# Neo4j Community 只有 neo4j / system 两个库，传随机 research_id 会
# DatabaseNotFound。取值等于 driver 默认库名即可绕开 clone 分支。
# research_id 写入 Episode 名称/来源描述；本轮上下文和报告均按本次
# Episode UUID 列表隔离，不再用 group_id 充当研究会话 ID。
GRAPH_GROUP_ID = "neo4j"


async def run(topic: str, max_rounds: int = MAX_ROUNDS_DEFAULT) -> None:
    """主研究循环。"""
    from graphiti_core import Graphiti

    # ── 连接 Graphiti ─────────────────────────────────────────────────────────
    neo4j_uri  = os.environ.get("NEO4J_URI",      "bolt://localhost:7687")
    neo4j_user = os.environ.get("NEO4J_USER",     "neo4j")
    neo4j_pwd  = os.environ.get("NEO4J_PASSWORD", "password")

    gdb = Graphiti(uri=neo4j_uri, user=neo4j_user, password=neo4j_pwd)
    await gdb.build_indices_and_constraints()

    research_id = uuid.uuid4().hex[:12]
    filled_ids: set[str] = set()
    failure_counts: dict[str, int] = {}
    previous_queries: dict[str, list[str]] = {}
    seen_urls: dict[str, set[str]] = {}
    evidence_by_slot: dict[str, list[EvidenceRef]] = {}
    research_episode_ids: list[str] = []

    print(f"\n{'='*60}")
    print(f"🔍 研究课题  : {topic}")
    print(f"🆔 研究 ID   : {research_id}")
    print(f"🔄 最大轮次  : {max_rounds}")
    print(f"{'='*60}\n")

    # ── 主循环 ────────────────────────────────────────────────────────────────
    for round_num in range(1, max_rounds + 1):
        if all_filled(filled_ids):
            print(f"\n✅ 所有槽位已填充，提前终止（第 {round_num - 1} 轮）\n")
            break

        empties = get_empty_slots(filled_ids)
        print(f"── 第 {round_num} 轮 ── 待填槽位 {len(empties)} 个 "
              f"（覆盖率 {coverage_pct(filled_ids):.0f}%）")

        # Step 1: Supervisor 选槽位 + 生成搜索词
        slot, query = await pick_next_slot_and_query(
            topic,
            filled_ids,
            failure_counts=failure_counts,
            previous_queries=previous_queries,
        )
        slot_id = slot["slot_id"]
        previous_queries.setdefault(slot_id, []).append(query)
        print(f"   🎯 目标槽位: [{slot['dimension']}] {slot['label']}")
        print(f"   🔎 搜索词  : {query}")

        # Step 2: 搜索
        results = await tavily_search(
            query,
            max_results=SEARCH_RESULTS_PER_ROUND,
            max_chars=MAX_CHARS_PER_PAGE,
            focus=slot["question"],
            exclude_urls=seen_urls.get(slot_id, set()),
        )
        print(f"   📄 搜索结果: {len(results)} 条")

        # Step 3: 抽取 + 写入
        slot_written = False
        for res in results[:PAGES_TO_EXTRACT_PER_ROUND]:
            seen_urls.setdefault(slot_id, set()).add(res.url)
            if not res.raw_text:
                continue

            triples = await extract_triples(res.raw_text, slot["question"])
            if not triples:
                print(f"      ↳ {res.url[:60]}… — 无相关内容")
                continue

            print(f"      ↳ {res.url[:60]}… — 抽到 {len(triples)} 条三元组")
            for t in triples:
                print(f"         • {t['subject']} | {t['predicate']} | {t['object']}")

            # Step 4: 写 Graphiti
            episode_uuid = await write_slot_episode(
                gdb,
                group_id=GRAPH_GROUP_ID,
                research_id=research_id,
                slot_id=slot_id,
                slot_label=slot["label"],
                triples=triples,
                source_url=res.url,
                source_title=res.title,
                previous_episode_uuids=research_episode_ids,
            )
            if not episode_uuid:
                print(f"      ↳ {res.url[:60]}… — Graphiti 未返回 Episode UUID")
                continue

            evidence_by_slot.setdefault(slot_id, []).append(
                EvidenceRef(
                    episode_uuid=episode_uuid,
                    title=res.title,
                    url=res.url,
                )
            )
            research_episode_ids.append(episode_uuid)
            filled_ids.add(slot_id)
            slot_written = True
            break  # 该槽位已写入，进入下一轮

        if not slot_written:
            failure_counts[slot_id] = failure_counts.get(slot_id, 0) + 1
            print(
                f"   ⚠️  本轮未能填充 [{slot['label']}]；"
                "下一轮优先调查其他未尝试槽位"
            )

        print()

    # ── 报告合成 ──────────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"📊 研究结束 — 覆盖率: {coverage_pct(filled_ids):.0f}% "
          f"({len(filled_ids)}/{len(SLOTS)} 槽位)")
    print(f"{'='*60}\n")
    print("生成报告中...\n")

    report = await build_report(
        gdb,
        topic=topic,
        evidence_by_slot=evidence_by_slot,
    )

    print(report)

    # 保存报告到文件
    report_path = _BASE / f"v1_report_{research_id}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n📁 报告已保存: {report_path}")

    await gdb.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="V1 最小闭环: 图谱空洞驱动搜索"
    )
    parser.add_argument("topic", help="研究课题，例如: '特斯拉2024交付量'")
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS_DEFAULT,
                        help=f"最大迭代轮次（默认 {MAX_ROUNDS_DEFAULT}）")
    args = parser.parse_args()

    asyncio.run(run(args.topic, max_rounds=args.max_rounds))


if __name__ == "__main__":
    main()
