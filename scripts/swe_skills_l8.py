"""Ecological L8 replay on SWE-Skills-Bench (rebuttal follow-up).

SWE-Skills-Bench (GeniusHTX/SWE-Skills-Bench, MIT) ships 49 real agent skills:
an agent-facing task prompt plus a Claude-style skill document whose frontmatter
names the skill identifier. We cast each skill as a routable capability tool and
replay the directive-override probe with *real* prose:

  L0E (control)  — one invocation tool per skill: real identifier + real frontmatter
                   description; the skill document (which cites the identifier) is in
                   the system prompt; gold = that tool.
  L8E (override) — tool A keeps the real identifier but a generic-superset
                   description (the platform rebound the skill; docs lag); tool B is
                   org-renamed and carries the real description, which specifically
                   matches the task prompt; gold = B. The system prompt carries the
                   real skill document, so the directive is *incidental* (a name in
                   authentic prose), not a purpose-built instruction.

Decoys are the other skills' real (name, description) pairs — a semantically crowded,
same-domain manifest. Selection-only scoring (the invocation tools take one `task`
argument by construction).

Usage:
  uv run --with datasets python scripts/swe_skills_l8.py [--models N1 N2 ...]
      [--out results/block_e8_swe_skills.jsonl] [--concurrency 16]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

from stale_tools.harness.models import by_nickname
from stale_tools.harness.perturbations import (
    _directive_decoy_description,
    _org_prefix_rename,
    _sanitize_name,
    _seed,
)
from stale_tools.harness.runner import SYSTEM_PROMPT, _client, call_one
from stale_tools.harness.perturbations import PerturbedTask

DEFAULT_MODELS = [
    "GPT-5.6-Sol",
    "GPT-5.6-Luna",
    "Opus-5",
    "Sonnet-5",
    "Gemini-3.6-Flash",
    "GPT-5",
    "Gemini-3.1-Pro",
    "DeepSeek-V4-Pro",
    "DeepSeek-R1",
    "GLM-5.2",
    "Qwen3.6-35B-A3B",
]

SKILL_DOC_MAX_CHARS = 4000
INVENTORY_SIZE = 12


def _tool(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": _sanitize_name(name),
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The task to hand to this skill."}
                },
                "required": ["task"],
            },
        },
    }


def build_cells(rows: list[dict]) -> list[PerturbedTask]:
    cells: list[PerturbedTask] = []
    for i, r in enumerate(rows):
        sid = _sanitize_name(r["skill_id"])
        desc = r["description"].strip().rstrip(".") + "."
        skill_doc = r["skill_document"][:SKILL_DOC_MAX_CHARS]
        # Deterministic decoys: the next skills in a seeded rotation.
        start = _seed(sid) % len(rows)
        decoys = []
        j = 0
        while len(decoys) < INVENTORY_SIZE:  # take extras; trimmed below per level
            cand = rows[(start + j) % len(rows)]
            j += 1
            if cand["skill_id"] == r["skill_id"]:
                continue
            decoys.append(_tool(cand["skill_id"], cand["description"].strip().rstrip(".") + "."))

        # L0E: real identifier + real description agree.
        inv0 = [_tool(sid, desc)] + decoys[: INVENTORY_SIZE - 1]
        cells.append(
            PerturbedTask(
                f"swe_{sid}", "L0E", sid, sid, inv0, r["task_prompt"], skill_doc, False, {}
            )
        )

        # L8E: directive tool (real name, generic desc) vs description-match sibling.
        new_name = _org_prefix_rename(sid, sid + "_match")
        tool_a = _tool(sid, _directive_decoy_description(sid))
        tool_b = _tool(new_name, desc)
        inv8 = [tool_a, tool_b] + decoys[: INVENTORY_SIZE - 2]
        cells.append(
            PerturbedTask(
                f"swe_{sid}",
                "L8E",
                _sanitize_name(new_name),
                sid,
                inv8,
                r["task_prompt"],
                skill_doc,
                False,
                {},
            )
        )
    return cells


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--out", type=Path, default=Path("results/block_e8_swe_skills.jsonl"))
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    rows = list(load_dataset("GeniusHTX/SWE-Skills-Bench")["train"])
    cells = build_cells(rows)
    models = [by_nickname(n) for n in args.models]

    done: set[tuple] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("ok"):
                done.add((rec["task_id"], rec["level"], rec["model_nick"]))

    todo = [(pt, m) for pt in cells for m in models if (pt.task_id, pt.level, m.nickname) not in done]
    print(f"{len(todo)} cells to run ({len(done)} cached)")

    sem = asyncio.Semaphore(args.concurrency)
    client = _client()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    f = args.out.open("a")

    async def one(pt: PerturbedTask, m) -> None:
        async with sem:
            r = await call_one(client, m, pt)
        inv_hash = hashlib.sha1(
            json.dumps(pt.inventory, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:12]
        rec = {
            "task_id": pt.task_id,
            "bfcl_category": "swe_skills",
            "level": pt.level,
            "model_nick": m.nickname,
            "model_slug": m.slug,
            "is_reasoning": m.is_reasoning,
            "pair_id": m.pair_id,
            "effort": m.effort,
            "inventory_size": INVENTORY_SIZE,
            "actual_inventory_size": len(pt.inventory),
            "seed": None,
            "inventory_hash": inv_hash,
            "skill_prose": pt.skill_instruction[:500],
            "target_name": pt.target_name,
            "obsolete_name": pt.obsolete_name,
            "perturb_meta": {"source": "SWE-Skills-Bench", "ts_built": datetime.now(timezone.utc).isoformat(timespec="seconds")},
            **r,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()

    n = 0
    for chunk_start in range(0, len(todo), 200):
        await asyncio.gather(*(one(pt, m) for pt, m in todo[chunk_start : chunk_start + 200]))
        n = min(chunk_start + 200, len(todo))
        print(f"  [{n}/{len(todo)}]")
    f.close()

    # Selection-only summary
    from collections import defaultdict

    agg = defaultdict(lambda: defaultdict(lambda: {"n": 0, "corr": 0, "dirf": 0, "abst": 0}))
    for line in args.out.read_text().splitlines():
        rec = json.loads(line)
        if not rec.get("ok"):
            continue
        called = (rec.get("tool_calls") or [{}])[0].get("name")
        a = agg[rec["level"]][rec["model_nick"]]
        a["n"] += 1
        a["corr"] += called == rec["target_name"]
        a["dirf"] += rec["level"] == "L8E" and called == rec["obsolete_name"]
        a["abst"] += not rec.get("tool_calls")
    for lvl in sorted(agg):
        print(f"\n{lvl}: model correct% dirf% abstain%")
        tot = {"n": 0, "corr": 0, "dirf": 0, "abst": 0}
        for mn in sorted(agg[lvl]):
            a = agg[lvl][mn]
            for k in tot:
                tot[k] += a[k]
            print(
                f"  {mn:18s} {100 * a['corr'] / a['n']:6.1f} {100 * a['dirf'] / a['n']:6.1f} "
                f"{100 * a['abst'] / a['n']:6.1f}"
            )
        print(
            f"  {'POOLED':18s} {100 * tot['corr'] / tot['n']:6.1f} {100 * tot['dirf'] / tot['n']:6.1f} "
            f"{100 * tot['abst'] / tot['n']:6.1f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
