"""Per-model rebuttal tables for the E-blocks (markdown output).

For each block present in results/, prints a per-model x per-level table of
selection accuracy (name-level), directive-followed rate, and abstain rate,
plus pooled rows. Reuses the deterministic rebuild-and-rejudge machinery from
recompute.py, so numbers are auditable from the shipped records alone.

Usage:
  uv run python analysis/rebuttal_tables.py [--results results] [--blocks e1 e2 ...]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from recompute import dedupe, load_task_index  # noqa: E402

from stale_tools.harness.judge import classify  # noqa: E402
from stale_tools.harness.perturbations import perturb  # noqa: E402

BLOCK_FILES = {
    "e1": "block_e1_directive_strength",
    "e2": "block_e2_matched_desc",
    "e3": "block_e3_lint_warn",
    "e4c": "block_e4_crowded",
    "e5": "block_e5_schema_drift",
    "d": "block_d_directive",
    "abl": "block_d_l8_ablation",
    "inv12": "block_e4_inv12",
    "inv50": "block_e4_inv50",
    "inv100": "block_e4_inv100",
    "inv8v2": "block_inv8_v2",
    "inv24v2": "block_inv24_v2",
}


def table(path: Path, idx: dict, answers: dict) -> None:
    per: dict[tuple, dict] = defaultdict(
        lambda: {"n": 0, "sel": 0, "dirf": 0, "abst": 0, "schema": 0}
    )
    levels: set[str] = set()
    for r in dedupe(path):
        if not r.get("ok") or r["task_id"] not in idx:
            continue
        pt = perturb(idx[r["task_id"]], r["level"], r.get("inventory_size", 12))
        res = classify(pt, r.get("tool_calls"), r.get("text"), answers.get(r["task_id"], []))
        a = per[(r["model_nick"], r["level"])]
        levels.add(r["level"])
        a["n"] += 1
        a["sel"] += res["called_name"] == pt.target_name
        a["dirf"] += res["directive_followed"]
        a["abst"] += res["code"] == "abstain"
        a["schema"] += bool(res.get("schema_valid"))
    if not per:
        print(f"  (no scored rows in {path.name})")
        return
    lvls = sorted(levels)
    is_l9 = any(lv.startswith("L9") for lv in lvls)
    metric = "schema-valid%" if is_l9 else "dir-followed%"
    print(f"\n### {path.stem}  — select% / {metric} per (model, level)\n")
    header = "| model | " + " | ".join(lvls) + " |"
    print(header)
    print("|" + "---|" * (len(lvls) + 1))
    models = sorted({m for (m, _) in per})
    pooled: dict[str, dict] = defaultdict(lambda: {"n": 0, "sel": 0, "x": 0})
    for m in models:
        cells = []
        for lv in lvls:
            a = per.get((m, lv))
            if not a or not a["n"]:
                cells.append("-")
                continue
            x = a["schema"] if is_l9 else a["dirf"]
            cells.append(f"{100 * a['sel'] / a['n']:.1f} / {100 * x / a['n']:.1f}")
            p = pooled[lv]
            p["n"] += a["n"]
            p["sel"] += a["sel"]
            p["x"] += x
        print(f"| {m} | " + " | ".join(cells) + " |")
    cells = []
    for lv in lvls:
        p = pooled[lv]
        cells.append(f"**{100 * p['sel'] / p['n']:.1f} / {100 * p['x'] / p['n']:.1f}**" if p["n"] else "-")
    print("| **pooled** | " + " | ".join(cells) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--blocks", nargs="+", default=list(BLOCK_FILES))
    args = ap.parse_args()
    idx, answers = load_task_index()
    for b in args.blocks:
        path = args.results / f"{BLOCK_FILES.get(b, b)}.jsonl"
        if path.exists():
            table(path, idx, answers)


if __name__ == "__main__":
    main()
