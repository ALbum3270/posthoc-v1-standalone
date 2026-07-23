"""Environment capability probe.

Checks every external dependency the GraphRAG loop needs, independently, so one
failure does not mask the others. Prints capability results only, never secrets.

Run:  python probe_env.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ENV_PATH = Path(r"c:\Users\Lenovo\Desktop\Langgraph\open_deep_research-main\.env")

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""), flush=True)


def load_env() -> None:
    from dotenv import load_dotenv

    load_dotenv(ENV_PATH, override=True)


# --------------------------------------------------------------------------
# 1. Neo4j: connectivity, edition, database list
# --------------------------------------------------------------------------
async def probe_neo4j() -> None:
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not uri:
        record("neo4j.config", SKIP, "NEO4J_URI unset")
        return

    scheme = uri.split("://", 1)[0]
    record("neo4j.config", PASS, f"scheme={scheme}")

    try:
        from neo4j import AsyncGraphDatabase
    except ImportError as exc:
        record("neo4j.driver_import", FAIL, str(exc))
        return

    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        # Connectivity
        try:
            records, _, _ = await driver.execute_query("RETURN 1 AS ok")
            record("neo4j.connect", PASS, f"RETURN 1 -> {records[0]['ok']}")
        except Exception as exc:
            record("neo4j.connect", FAIL, f"{type(exc).__name__}: {exc}")
            return

        # Edition -- decides whether multi-database (and therefore Graphiti's
        # group_id -> database mapping) is even possible.
        try:
            records, _, _ = await driver.execute_query(
                "CALL dbms.components() YIELD name, versions, edition "
                "RETURN name, versions, edition"
            )
            row = records[0]
            record(
                "neo4j.edition",
                PASS,
                f"{row['name']} {row['versions'][0]} edition={row['edition']}",
            )
        except Exception as exc:
            record("neo4j.edition", FAIL, f"{type(exc).__name__}: {exc}")

        # Database list -- Community exposes exactly one user database.
        try:
            records, _, _ = await driver.execute_query("SHOW DATABASES")
            names = sorted({r["name"] for r in records})
            record("neo4j.databases", PASS, f"{names}")
        except Exception as exc:
            record("neo4j.databases", SKIP, f"{type(exc).__name__}: {exc}")
    finally:
        await driver.close()


# --------------------------------------------------------------------------
# 2-4. OpenAI-compatible endpoint capabilities
# --------------------------------------------------------------------------
def _client():
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )


# Resolved in main(), after load_env() -- reading these at import time would miss
# whatever the .env file sets.
CHAT_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-small"


def resolve_models() -> None:
    """Pick the models to probe: explicit override > .env > graphiti's defaults."""
    global CHAT_MODEL, EMBED_MODEL
    CHAT_MODEL = (
        os.getenv("PROBE_CHAT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"
    )
    EMBED_MODEL = os.getenv("PROBE_EMBED_MODEL") or "text-embedding-3-small"


async def probe_list_models() -> None:
    """Enumerate what this key actually grants, so model choice is not guesswork."""
    try:
        client = _client()
        page = await client.models.list()
        ids = sorted(m.id for m in page.data)
        chat_like = [i for i in ids if "embed" not in i.lower()]
        embed_like = [i for i in ids if "embed" in i.lower()]
        record("openai.models_list", PASS, f"{len(ids)} models")
        print(f"       chat-ish  ({len(chat_like)}): {chat_like}")
        print(f"       embedding ({len(embed_like)}): {embed_like}")
    except Exception as exc:
        record("openai.models_list", SKIP, f"{type(exc).__name__}: {exc}")


async def probe_chat_completions() -> None:
    """Baseline: does the endpoint proxy plain chat completions at all?"""
    try:
        client = _client()
        resp = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            max_tokens=5,
        )
        record("openai.chat_completions", PASS, repr(resp.choices[0].message.content))
    except Exception as exc:
        record("openai.chat_completions", FAIL, f"{type(exc).__name__}: {exc}")


async def probe_responses_parse() -> None:
    """The path graphiti's DEFAULT OpenAIClient uses (openai_client.py:99).

    If this fails, Graphiti must be constructed with OpenAIGenericClient instead.
    """
    from pydantic import BaseModel

    class Answer(BaseModel):
        city: str
        country: str

    try:
        client = _client()
        resp = await client.responses.parse(
            model=CHAT_MODEL,
            input=[{"role": "user", "content": "Capital of Japan? Fill the schema."}],
            text_format=Answer,
        )
        record("openai.responses_parse", PASS, repr(resp.output_parsed))
    except Exception as exc:
        record("openai.responses_parse", FAIL, f"{type(exc).__name__}: {exc}")


async def probe_json_object() -> None:
    """The path graphiti's OpenAIGenericClient uses (openai_generic_client.py:111)."""
    try:
        client = _client()
        resp = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": 'Return JSON {"city": ..., "country": ...} for Japan\'s capital.',
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=60,
        )
        record("openai.json_object", PASS, repr(resp.choices[0].message.content))
    except Exception as exc:
        record("openai.json_object", FAIL, f"{type(exc).__name__}: {exc}")


async def probe_embeddings() -> None:
    """Silent-failure guard: no embeddings -> hybrid search returns nothing."""
    try:
        client = _client()
        resp = await client.embeddings.create(model=EMBED_MODEL, input=["hello world"])
        dim = len(resp.data[0].embedding)
        record("openai.embeddings", PASS, f"model={EMBED_MODEL} dim={dim}")
    except Exception as exc:
        record("openai.embeddings", FAIL, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
async def main() -> int:
    load_env()
    resolve_models()
    print(f"python: {sys.version.split()[0]}")
    print(f"chat model: {CHAT_MODEL}   embed model: {EMBED_MODEL}\n")

    await probe_neo4j()
    print()
    await probe_list_models()
    await probe_chat_completions()
    await probe_responses_parse()
    await probe_json_object()
    await probe_embeddings()

    print("\n--- summary ---")
    failed = [name for name, status, _ in results if status == FAIL]
    for name, status, _ in results:
        print(f"{status:4}  {name}")
    if failed:
        print(f"\n{len(failed)} check(s) failed: {', '.join(failed)}")
    else:
        print("\nall checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
