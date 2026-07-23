"""
最小本体定义 — 5W1H 硬编码槽位，不做动态扩展。

每个槽位是一个 dict：
  {
    "slot_id":   唯一标识，形如 "who.primary_actor"
    "dimension": 所属维度 (WHO / WHAT / WHEN / WHERE / WHY / HOW)
    "label":     可读名称
    "question":  发给 LLM 的调查问题
    "priority":  0-100，数值越高越优先填充
  }
"""

from __future__ import annotations

SLOTS: list[dict] = [
    # ── WHO ──────────────────────────────────────────────────────────────────
    {"slot_id": "who.primary_actor",     "dimension": "WHO",   "priority": 100,
     "label": "主要行为人",
     "question": "Who is the main actor, organization, or primary subject directly involved in this event?"},
    {"slot_id": "who.affected_parties",  "dimension": "WHO",   "priority": 80,
     "label": "受影响方",
     "question": "Who is affected, targeted, harmed, or impacted by this event?"},
    {"slot_id": "who.related_orgs",      "dimension": "WHO",   "priority": 65,
     "label": "相关机构",
     "question": "Which organizations, subsidiaries, or partners are connected to this event?"},
    {"slot_id": "who.regulators",        "dimension": "WHO",   "priority": 55,
     "label": "监管/执法方",
     "question": "Which regulators, investigators, government bodies, or authorities are involved?"},

    # ── WHAT ─────────────────────────────────────────────────────────────────
    {"slot_id": "what.core_event",       "dimension": "WHAT",  "priority": 100,
     "label": "核心事件",
     "question": "What exactly happened? Describe the core event or action."},
    {"slot_id": "what.scale",            "dimension": "WHAT",  "priority": 85,
     "label": "规模/量化数据",
     "question": "What is the scale, financial amount, quantity, or material impact?"},
    {"slot_id": "what.products",         "dimension": "WHAT",  "priority": 65,
     "label": "涉及产品/服务",
     "question": "What products, services, technologies, or assets are directly involved?"},

    # ── WHEN ─────────────────────────────────────────────────────────────────
    {"slot_id": "when.event_time",       "dimension": "WHEN",  "priority": 90,
     "label": "事发时间",
     "question": "When did the key event or action occur? Provide specific dates or time ranges."},
    {"slot_id": "when.discovery_time",   "dimension": "WHEN",  "priority": 70,
     "label": "暴露/发现时间",
     "question": "When was the event first discovered, disclosed, or made public?"},
    {"slot_id": "when.response_time",    "dimension": "WHEN",  "priority": 55,
     "label": "官方介入时间",
     "question": "When did official intervention, response, or regulatory action begin?"},

    # ── WHERE ────────────────────────────────────────────────────────────────
    {"slot_id": "where.jurisdiction",    "dimension": "WHERE", "priority": 75,
     "label": "司法/运营地",
     "question": "Where is the primary legal jurisdiction, country, or operational location?"},
    {"slot_id": "where.asset_flow",      "dimension": "WHERE", "priority": 65,
     "label": "资金/资产流向",
     "question": "Where did money, assets, data, or key activity flow to or from?"},

    # ── WHY ──────────────────────────────────────────────────────────────────
    {"slot_id": "why.motivation",        "dimension": "WHY",   "priority": 70,
     "label": "动机",
     "question": "Why did the actor behave this way? What was the motivation or stated reason?"},
    {"slot_id": "why.trigger",           "dimension": "WHY",   "priority": 55,
     "label": "导火索",
     "question": "What was the immediate trigger, catalyst, or precipitating condition?"},

    # ── HOW ──────────────────────────────────────────────────────────────────
    {"slot_id": "how.mechanism",         "dimension": "HOW",   "priority": 80,
     "label": "手法/机制",
     "question": "How was the event carried out? What method, mechanism, or process was used?"},
    {"slot_id": "how.sequence",          "dimension": "HOW",   "priority": 55,
     "label": "时序步骤",
     "question": "How did the activity unfold step by step? What was the sequence of actions?"},
]

# 按优先级降序预排好，运行时直接取第一个未填的
SLOTS_SORTED = sorted(SLOTS, key=lambda s: -s["priority"])


def get_empty_slots(filled_ids: set[str]) -> list[dict]:
    """返回尚未填充的槽位，按优先级降序排列。"""
    return [s for s in SLOTS_SORTED if s["slot_id"] not in filled_ids]


def all_filled(filled_ids: set[str]) -> bool:
    """所有槽位是否都已有内容。"""
    return all(s["slot_id"] in filled_ids for s in SLOTS)


def coverage_pct(filled_ids: set[str]) -> float:
    """已填充槽位占总槽位的百分比。"""
    return len(filled_ids) / len(SLOTS) * 100
