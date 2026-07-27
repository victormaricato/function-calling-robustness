"""Async OpenRouter runner.

Iterates over (task, level, model) cells, makes the OpenRouter call with the
perturbed inventory, and records: tool calls, visible text, reasoning text
(when exposed), latency, token usage. Writes one JSON line per cell.
Resumable — rerunning skips successful cells and retries failed ones.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

from .models import ModelSpec
from .perturbations import PerturbedTask, perturb
from .settings import load_openrouter_key

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

SYSTEM_PROMPT = (
    "You are an assistant that selects and invokes the appropriate tool to answer the user's "
    "question. Always emit a tool call when a suitable tool is available. If no tool fits, "
    "explain why in plain text instead."
)


def _env_set(name: str) -> set[str]:
    raw = os.environ.get(name, "")
    return {part.strip() for part in raw.split(",") if part.strip()}


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=load_openrouter_key(), base_url=OPENROUTER_BASE, timeout=120)


async def call_one(
    client: AsyncOpenAI,
    spec: ModelSpec,
    pt: PerturbedTask,
    max_attempts: int = 5,
) -> dict:
    """Make a single tool-use call with retry on transient errors. Return a result dict."""
    # The system prompt embeds the skill-style guidance that names the OBSOLETE tool:
    # the documentation surrounding the model still references an identifier the live
    # inventory no longer matches.
    system_text = SYSTEM_PROMPT + "\n\n" + pt.skill_instruction
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": pt.instruction},
    ]
    tools = pt.inventory
    # We allow fallbacks to keep throughput; we record which provider served each call
    extra_body: dict = {"provider": {"allow_fallbacks": True}}
    if spec.extra_body:
        # merge nested 'reasoning' etc.
        for k, v in spec.extra_body.items():
            if k in extra_body and isinstance(extra_body[k], dict) and isinstance(v, dict):
                extra_body[k].update(v)
            else:
                extra_body[k] = v

    t0 = time.time()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    last_err = None
    resp = None
    attempts = 0
    for attempt in range(max_attempts):
        attempts = attempt + 1
        try:
            resp = await client.chat.completions.create(
                model=spec.slug,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0,
                max_tokens=1024,
                extra_body=extra_body,
            )
            # OpenRouter sometimes returns an embedded error inside a 200 with choices=None
            if not getattr(resp, "choices", None):
                err = getattr(resp, "error", None) or {}
                err_msg = err.get("message", str(resp)) if isinstance(err, dict) else str(err)
                raise RuntimeError(f"OpenRouter returned no choices: {err_msg}")
            break
        except Exception as e:
            msg = str(e)
            last_err = f"{type(e).__name__}: {msg}"
            transient = (
                ("429" in msg)
                or ("503" in msg)
                or ("502" in msg)
                or ("timeout" in msg.lower())
                or ("temporarily" in msg.lower())
                or ("rate" in msg.lower())
                or ("no choices" in msg.lower())
            )
            if not transient or attempt == max_attempts - 1:
                return {
                    "ok": False,
                    "error": last_err,
                    "latency_s": time.time() - t0,
                    "ts": ts,
                    "attempts": attempts,
                }
            await asyncio.sleep(min(2**attempt, 16) + (0.1 * attempt))

    msg = resp.choices[0].message
    tool_calls = []
    for tc in msg.tool_calls or []:
        tool_calls.append(
            {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
        )
    # OpenRouter sometimes exposes reasoning trace at msg.reasoning or .reasoning_details
    reasoning = getattr(msg, "reasoning", None)
    if reasoning is None and hasattr(msg, "model_extra"):
        reasoning = (msg.model_extra or {}).get("reasoning")

    usage = resp.usage
    # OpenRouter reports the serving provider as a top-level `provider` field on the
    # response; the OpenAI SDK surfaces unknown fields via model_extra. `resp.model`
    # is only the slug echoed back — record it separately as response_model.
    served_by = getattr(resp, "provider", None)
    if served_by is None and hasattr(resp, "model_extra"):
        served_by = (resp.model_extra or {}).get("provider")
    return {
        "ok": True,
        "tool_calls": tool_calls,
        "text": msg.content or "",
        "reasoning": reasoning,
        "finish_reason": resp.choices[0].finish_reason,
        "latency_s": time.time() - t0,
        "ts": ts,
        "attempts": attempts,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "provider": served_by,
        "response_model": resp.model,
    }


async def run_block(
    tasks: list[dict],
    levels: list[str],
    models: list[ModelSpec],
    out_path: Path,
    inventory_size: int = 16,
    concurrency: int = 8,
    seed: int | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    excluded_slugs = _env_set("EXCLUDE_MODEL_SLUGS")
    excluded_nicks = _env_set("EXCLUDE_MODEL_NICKS")
    if excluded_slugs or excluded_nicks:
        before = len(models)
        models = [
            m for m in models if m.slug not in excluded_slugs and m.nickname not in excluded_nicks
        ]
        print(
            f"  model filter: {before} -> {len(models)} "
            f"(excluded slugs={sorted(excluded_slugs)}, nicks={sorted(excluded_nicks)})"
        )
    # Resume cache: skip cells that completed successfully on a previous run.
    # Cells that failed (ok=False) are retried — common for transient OpenRouter
    # errors (429, 503, provider rotation). To force a full re-run, delete the
    # output JSONL.
    done: set[tuple] = set()
    retried = 0
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["task_id"], r["level"], r["model_nick"])
            if r.get("ok"):
                done.add(key)
            else:
                retried += 1

    cells = [
        (t, lvl, m)
        for t in tasks
        for lvl in levels
        for m in models
        if (t["id"], lvl, m.nickname) not in done
    ]
    print(
        f"  cells to run: {len(cells)} "
        f"(skipping {len(done)} cached successes, retrying {retried} prior failures)"
    )

    sem = asyncio.Semaphore(concurrency)
    client = _client()
    f = out_path.open("a")

    async def one(t: dict, lvl: str, m: ModelSpec) -> None:
        pt = perturb(t, lvl, inventory_size)
        async with sem:
            r = await call_one(client, m, pt)
        inv_hash = hashlib.sha1(
            json.dumps(pt.inventory, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:12]
        rec = {
            "task_id": t["id"],
            "bfcl_category": t.get("_bfcl_category", "unknown"),
            "level": lvl,
            "model_nick": m.nickname,
            "model_slug": m.slug,
            "is_reasoning": m.is_reasoning,
            "pair_id": m.pair_id,
            "effort": m.effort,
            "inventory_size": inventory_size,
            "actual_inventory_size": len(pt.inventory),
            "seed": seed,
            "inventory_hash": inv_hash,
            "skill_prose": pt.skill_instruction,
            "target_name": pt.target_name,
            "obsolete_name": pt.obsolete_name,
            "perturb_meta": pt.meta or None,
            **r,
        }
        line = json.dumps(rec, ensure_ascii=False)
        f.write(line + "\n")
        f.flush()

    # Schedule all coroutines and let the semaphore meter throughput. Continuous
    # flow keeps the pool saturated; chunked gather wastes slots waiting on slow tails.
    coros = [one(t, lvl, m) for (t, lvl, m) in cells]
    completed = {"n": 0}

    async def report(coro):
        await coro
        completed["n"] += 1
        n = completed["n"]
        if n % 200 == 0 or n == len(coros):
            print(f"  [{n}/{len(cells)}] {n / max(1, time.time() - t_start):.1f} cells/s avg")

    t_start = time.time()
    await asyncio.gather(*(report(c) for c in coros), return_exceptions=False)
    f.close()
