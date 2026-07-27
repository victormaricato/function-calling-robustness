"""Recompute the paper's headline numbers from the shipped per-cell records.

Every perturbation is deterministic from (task_id, seed), so each recorded cell can
be reconstructed exactly: we rebuild the PerturbedTask the model saw, re-apply the
programmatic judge to the recorded structured response, and aggregate. No API access
is needed; this audits the numbers, it does not re-query models.

Covers the BFCL-based blocks (breadth, directive-override, L8 ablations, and the
rebuttal blocks E1--E5 once present). APIBank blocks additionally require the
HuggingFace-cached dataset (see README) and are skipped when absent.

Usage:
  uv run python analysis/recompute.py [--results results] [--seed 2026]

Expected headline values (paper, blocks as of submission):
  block_d_directive     L0 selection 97.2% | L8 selection 0.2% | L8 directive-followed 96.0%
  block_d_l8_ablation   L8N correct 26.3% | L8I correct 41.5%
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from stale_tools.harness import tasks as T
from stale_tools.harness.judge import classify
from stale_tools.harness.perturbations import perturb

BLOCKS = [
    "block_a_sota",
    "block_d_directive",
    "block_d_l8_ablation",
    "block_e1_directive_strength",
    "block_e2_matched_desc",
    "block_e3_lint_warn",
    "block_e4_crowded",
    "block_e5_schema_drift",
]


def load_task_index() -> tuple[dict, dict]:
    answers = T.all_answers()
    idx: dict[str, dict] = {}
    for loader, cat in (
        (T.load_simple, "simple"),
        (T.load_multiple, "multiple"),
        (T.load_parallel, "parallel"),
    ):
        for r in loader():
            r = dict(r)
            r["_bfcl_category"] = cat
            gold = answers.get(r["id"], [])
            if gold:
                gname = next(iter(gold[0].keys()))
                r["_gold_target"] = gname
                r["_gold_target_idx"] = next(
                    (i for i, f in enumerate(r["function"]) if f["name"] == gname), 0
                )
            else:
                r["_gold_target"] = r["function"][0]["name"]
                r["_gold_target_idx"] = 0
            idx[r["id"]] = r
    return idx, answers


def dedupe(path: Path) -> list[dict]:
    """Latest-successful-wins per (task, level, model): failed rows are retried on
    resume, so files may contain multiple rows per key."""
    rows: dict[tuple, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["task_id"], r["level"], r["model_nick"])
        if key not in rows or r.get("ok"):
            rows[key] = r
    return list(rows.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    idx, answers = load_task_index()

    for block in BLOCKS:
        path = args.results / f"{block}.jsonl"
        if not path.exists():
            continue
        agg: dict[str, dict] = defaultdict(
            lambda: {
                "n": 0,
                "name_correct": 0,
                "strict_correct": 0,
                "directive": 0,
                "abstain": 0,
                "schema_valid": 0,
            }
        )
        skipped = 0
        for r in dedupe(path):
            if not r.get("ok"):
                skipped += 1
                continue
            task = idx.get(r["task_id"])
            if task is None:
                skipped += 1
                continue
            pt = perturb(task, r["level"], r.get("inventory_size", 12))
            res = classify(pt, r.get("tool_calls"), r.get("text"), answers.get(r["task_id"], []))
            a = agg[r["level"]]
            a["n"] += 1
            # The paper's headline "selection accuracy" is name-level (did the model
            # route to the right tool); strict additionally requires admissible args.
            a["name_correct"] += res["called_name"] == pt.target_name
            a["strict_correct"] += res["selection_correct"]
            a["directive"] += res["directive_followed"]
            a["abstain"] += res["code"] == "abstain"
            a["schema_valid"] += bool(res.get("schema_valid"))
        print(f"\n{block}  (skipped {skipped} failed/unmatched rows)")
        print(
            f"  {'level':6s} {'n':>6s} {'select%':>8s} {'strict%':>8s} "
            f"{'dir-followed%':>14s} {'abstain%':>9s}"
        )
        for lvl in sorted(agg):
            a = agg[lvl]
            extra = ""
            if lvl.startswith("L9"):
                extra = f"  schema_valid%={100 * a['schema_valid'] / a['n']:.1f}"
            print(
                f"  {lvl:6s} {a['n']:>6d} {100 * a['name_correct'] / a['n']:>8.1f} "
                f"{100 * a['strict_correct'] / a['n']:>8.1f} "
                f"{100 * a['directive'] / a['n']:>14.1f} {100 * a['abstain'] / a['n']:>9.1f}{extra}"
            )


if __name__ == "__main__":
    main()
