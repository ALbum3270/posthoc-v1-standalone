# V1 最小闭环 — 快速上手

验证一件事：**图谱空洞能否指导搜索词生成，并让调查收敛？**

## 系统边界

| 有 | 没有 |
|---|---|
| 5W1H 硬编码槽位驱动搜索 | CurioCat 质检 |
| 小模型三元组抽取 | 冲突检测 / flush 屏障 |
| Graphiti 写入 + 检索 | 动态 Ontology |
| Markdown 报告输出 | Provenance UUID 链 |
| 最多 10 轮自动停止 | 冲突裁决 |

## 环境配置

```bash
# 1. 在 open_deep_research-main/ 目录下创建 .env
cp .env.example .env
# 填入必要的 key：
#   OPENAI_API_KEY=sk-...
#   TAVILY_API_KEY=tvly-...

# 2. 确保 Neo4j 正在运行（默认端口 7687）
# Docker 快速启动：
docker compose up -d neo4j   # 使用根目录的 docker-compose.yml

# 可选：覆盖默认连接
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password
```

## 运行

```bash
# 在 open_deep_research-main/ 目录下执行
cd open_deep_research-main

# 基本用法
python -m v1.run "FTX 暴雷事件"

# 自定义最大轮次
python -m v1.run "特斯拉 2024 年交付量" --max-rounds 5

# 英文课题
python -m v1.run "OpenAI GPT-4o release" --max-rounds 8
```

## 输出解读

```
── 第 1 轮 ── 待填槽位 16 个 （覆盖率 0%）
   🎯 目标槽位: [WHO] 主要行为人
   🔎 搜索词  : FTX founder Sam Bankman-Fried role collapse
   📄 搜索结果: 3 条
      ↳ https://... — 抽到 3 条三元组
         • Sam Bankman-Fried | founded | FTX exchange
         • ...
```

报告自动保存到 `v1_report_<research_id>.md`。

## 文件结构

```
v1/
├── __init__.py
├── ontology.py        # 5W1H 硬编码槽位（16 个）
├── supervisor.py      # 槽位选择 + 搜索词生成
├── searcher.py        # Tavily 搜索封装
├── extractor.py       # GPT-4o-mini 三元组抽取
├── graph_writer.py    # Graphiti Episode 写入
├── reporter.py        # 从图谱读取并生成 Markdown 报告
├── run.py             # 主循环入口
└── README.md          # 本文件
```

## 依赖

已在 `pyproject.toml` 中声明（主项目的 dependencies 均可用）：
- `openai`（直接调用，不走 langchain）
- `tavily-python`
- `graphiti-core`（从 `graphiti-main/` 目录安装）
- `python-dotenv`
