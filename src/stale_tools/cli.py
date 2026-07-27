"""``stale-tools run <block>`` — execute one experiment block.

Each invocation writes one JSON line per (task, level, model) cell into
``$STALE_TOOLS_RESULTS_DIR/<block>.jsonl``. Runs are resumable: rerunning
skips cells whose previous attempt succeeded and retries cells whose
previous attempt failed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .harness.apibank_tasks import stratified_sample as apibank_stratified
from .harness.models import (
    EFFORT_SWEEP,
    SIZE_LADDER,
    SOTA,
    SOTA_V2,
    by_nickname,
    core_12,
    scale_8,
    schema_pilot,
)
from .harness.perturbations import (
    CROWDING_LEVELS,
    DIRECTIVE_STRENGTH_LEVELS,
    LEVELS,
    LINT_WARN_LEVELS,
    MATCHED_DESC_LEVELS,
    SCHEMA_DRIFT_LEVELS,
)
from .harness.runner import run_block
from .harness.settings import results_dir
from .harness.tasks import stratified_sample as bfcl_stratified

ALL_LEVELS = list(LEVELS)
ANCHOR_LEVELS = ["L0", "L3", "L6", "L7"]
INFORMATIVE_LEVELS = ["L0", "L3", "L6"]
DIRECTIVE_LEVELS = ["L0", "L8"]  # baseline + directive override only
L8_ABLATION_LEVELS = ["L8N", "L8I"]  # no-skill and intent-only skill ablations
SCALE_LEVELS = ["L0", "L4", "L6", "L8"]  # inventory-scale sweep (E4)


def _probe_models() -> list:
    return [
        by_nickname("Opus-4.7"),
        by_nickname("GPT-5.1"),
        by_nickname("Gemini-3.1-Pro"),
        by_nickname("DeepSeek-V4-Pro"),
    ]


def _post_cutoff_tasks() -> list[dict]:
    pc = Path(__file__).resolve().parents[2] / "data" / "post_cutoff_tasks.json"
    raw = json.loads(pc.read_text())
    out = []
    for r in raw:
        gold_name = next(iter(r["gold"][0].keys()))
        out.append(
            {
                "id": r["id"],
                "_bfcl_category": "post_cutoff",
                "function": r["function"],
                "_gold_target": gold_name,
                "_gold_target_idx": next(
                    (i for i, f in enumerate(r["function"]) if f["name"] == gold_name), 0
                ),
                "question": [[{"role": "user", "content": r["user_query"]}]],
            }
        )
    return out


# Block specifications: each maps a name to (task-builder, level subset, model set, output stem).
BLOCKS: dict[str, dict] = {
    "breadth": dict(
        tasks=lambda seed: bfcl_stratified(75, 75, 25, seed=seed),
        levels=ALL_LEVELS,
        models=lambda: SOTA_V2,
        out="block_a_sota",
        concurrency=40,
    ),
    "effort": dict(
        tasks=lambda seed: bfcl_stratified(75, 50, 25, seed=seed),
        levels=ALL_LEVELS,
        models=lambda: EFFORT_SWEEP,
        out="block_b_effort",
        concurrency=24,
    ),
    "apibank": dict(
        tasks=lambda seed: apibank_stratified(n=150, seed=seed),
        levels=ANCHOR_LEVELS,
        models=lambda: SOTA,
        out="block_c_apibank",
        concurrency=24,
    ),
    "size-ladder": dict(
        tasks=lambda seed: bfcl_stratified(40, 25, 10, seed=seed),
        levels=INFORMATIVE_LEVELS,
        models=lambda: SIZE_LADDER,
        out="block_s_size",
        concurrency=24,
    ),
    "post-cutoff": dict(
        tasks=lambda seed: _post_cutoff_tasks(),
        levels=INFORMATIVE_LEVELS,
        models=_probe_models,
        out="post_cutoff_holdout",
        concurrency=12,
    ),
    "apibank-pilot": dict(
        tasks=lambda seed: apibank_stratified(n=30, seed=seed),
        levels=["L1", "L2", "L4", "L5"],
        models=lambda: SOTA,
        out="apibank_full_levels_pilot",
        concurrency=24,
    ),
    "directive": dict(
        tasks=lambda seed: bfcl_stratified(75, 75, 25, seed=seed),
        levels=DIRECTIVE_LEVELS,
        models=lambda: SOTA_V2,
        out="block_d_directive",
        concurrency=40,
    ),
    "l8-ablation": dict(
        tasks=lambda seed: bfcl_stratified(75, 75, 25, seed=seed),
        levels=L8_ABLATION_LEVELS,
        models=lambda: SOTA_V2,
        out="block_d_l8_ablation",
        concurrency=40,
    ),
    # ── Rebuttal blocks ────────────────────────────────────────────────────
    "directive-strength": dict(  # E1 — U8pb Q1: how strong must the skill prose be?
        tasks=lambda seed: bfcl_stratified(75, 75, 25, seed=seed),
        levels=list(DIRECTIVE_STRENGTH_LEVELS),
        models=core_12,
        out="block_e1_directive_strength",
        concurrency=40,
    ),
    "matched-desc": dict(  # E2 — U8pb Q3: remove the description-quality gap at L8
        tasks=lambda seed: bfcl_stratified(75, 75, 25, seed=seed),
        levels=list(MATCHED_DESC_LEVELS),
        models=core_12,
        out="block_e2_matched_desc",
        concurrency=40,
    ),
    "lint-warn": dict(  # E3 (online half) — xMJB Q4: does a lint warning recover routing?
        tasks=lambda seed: bfcl_stratified(75, 75, 25, seed=seed),
        levels=list(LINT_WARN_LEVELS),
        models=core_12,
        out="block_e3_lint_warn",
        concurrency=40,
    ),
    "crowding": dict(  # E4 (crowding half) — xMJB Q2: near-duplicate siblings at 12 tools
        tasks=lambda seed: bfcl_stratified(75, 75, 25, seed=seed),
        levels=list(CROWDING_LEVELS),
        models=scale_8,
        out="block_e4_crowded",
        concurrency=40,
    ),
    "schema-drift": dict(  # E5 — xMJB Q1: parameter-schema evolution pilot
        tasks=lambda seed: bfcl_stratified(75, 75, 25, seed=seed),
        levels=list(SCHEMA_DRIFT_LEVELS),
        models=schema_pilot,
        out="block_e5_schema_drift",
        concurrency=40,
    ),
}


async def _run_one(name: str, args: argparse.Namespace) -> None:
    spec = BLOCKS[name]
    tasks = spec["tasks"](args.seed)
    models = spec["models"]()
    print(
        f"{name}: {len(tasks)} tasks × {len(spec['levels'])} levels × {len(models)} models "
        f"= {len(tasks) * len(spec['levels']) * len(models)} cells"
    )
    await run_block(
        tasks=tasks,
        levels=spec["levels"],
        models=models,
        out_path=results_dir() / f"{spec['out']}.jsonl",
        inventory_size=args.inventory_size,
        concurrency=args.concurrency or spec["concurrency"],
        seed=args.seed,
    )


async def cmd_run(args: argparse.Namespace) -> None:
    await _run_one(args.block, args)


async def cmd_inventory_sensitivity(args: argparse.Namespace) -> None:
    """Sweep the inventory-size factor across probe sizes on a fixed task subset.

    Writes to ``block_inv{size}_v2.jsonl``: the original ``block_inv{size}`` files
    were produced when padding was capped by the 15-entry decoy pool, so sizes
    above ~16 silently ran at 17-19 tools. The v2 runs use the BFCL-pool padding
    and record ``actual_inventory_size`` per cell.
    """
    tasks = bfcl_stratified(40, 25, 10, seed=args.seed)
    probe = _probe_models()
    for inv in args.sizes:
        print(
            f"inventory-sensitivity size={inv}: {len(tasks)} tasks × {len(INFORMATIVE_LEVELS)} "
            f"levels × {len(probe)} models = {len(tasks) * len(INFORMATIVE_LEVELS) * len(probe)} cells"
        )
        await run_block(
            tasks=tasks,
            levels=INFORMATIVE_LEVELS,
            models=probe,
            out_path=results_dir() / f"block_inv{inv}_v2.jsonl",
            inventory_size=inv,
            concurrency=args.concurrency,
            seed=args.seed,
        )


async def cmd_inventory_scale(args: argparse.Namespace) -> None:
    """E4 — inventory-scale sweep: {12, 50, 100} tools × {L0, L4, L6, L8} × 8 models."""
    tasks = bfcl_stratified(90, 80, 30, seed=args.seed)
    models = scale_8()
    for inv in args.sizes:
        print(
            f"inventory-scale size={inv}: {len(tasks)} tasks × {len(SCALE_LEVELS)} "
            f"levels × {len(models)} models = {len(tasks) * len(SCALE_LEVELS) * len(models)} cells"
        )
        await run_block(
            tasks=tasks,
            levels=SCALE_LEVELS,
            models=models,
            out_path=results_dir() / f"block_e4_inv{inv}.jsonl",
            inventory_size=inv,
            concurrency=args.concurrency,
            seed=args.seed,
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stale-tools",
        description=(
            "Execute one experiment block of the stale-tools probe. "
            "Records land in $STALE_TOOLS_RESULTS_DIR/<block>.jsonl (default ./results)."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one experiment block")
    run.add_argument(
        "block",
        choices=sorted(BLOCKS.keys()),
        help="which block to run",
    )
    run.add_argument("--seed", type=int, default=2026)
    run.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="max concurrent OpenRouter requests (0 = block default)",
    )
    run.add_argument("--inventory-size", type=int, default=12)
    run.set_defaults(func=cmd_run)

    inv = sub.add_parser(
        "inventory-sensitivity",
        help="inventory-size sensitivity sweep across two probe sizes",
    )
    inv.add_argument("--seed", type=int, default=2026)
    inv.add_argument("--sizes", type=int, nargs="+", default=[8, 24])
    inv.add_argument("--concurrency", type=int, default=24)
    inv.set_defaults(func=cmd_inventory_sensitivity)

    scale = sub.add_parser(
        "inventory-scale",
        help="E4: inventory-scale sweep (12/50/100 tools) over L0/L4/L6/L8 on the 8-model arm",
    )
    scale.add_argument("--seed", type=int, default=2026)
    scale.add_argument("--sizes", type=int, nargs="+", default=[12, 50, 100])
    scale.add_argument("--concurrency", type=int, default=24)
    scale.set_defaults(func=cmd_inventory_scale)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    asyncio.run(args.func(args))


if __name__ == "__main__":
    main()
