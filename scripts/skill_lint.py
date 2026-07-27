"""Skill-manifest lint (rebuttal E3, offline half).

A platform-side mitigation: statically cross-check the tool reference inside a
skill document against the live manifest, with no model in the loop. Three rules:

  A missing-identifier    — the referenced tool name is absent from the manifest
                            (fires under renames/removal: L1, L2, L3, L7)
  B deprecated-reference  — the referenced tool carries a deprecation marker (L6)
  C description-mismatch  — the referenced tool exists, but its live description no
                            longer covers the purpose stated in the skill prose
                            (fires under description drift L5 and directive override L8)

Rule C uses word containment: the fraction of content words from the skill's stated
purpose that appear in the live description. No model, no embeddings — the lint must
be cheap enough to run on every skill x manifest pair at deploy time.

The lint is evaluated over the exact perturbed configurations of the experiment
blocks (same stratified sample, same seed), reporting per-level flag rates:
recall at perturbed levels, false-positive rate at L0/L4 (semantics-preserving).

Usage:
  uv run python scripts/skill_lint.py [--seed 2026] [--out results/lint_report.json]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from stale_tools.harness.perturbations import PerturbedTask, perturb
from stale_tools.harness.tasks import stratified_sample

LINT_LEVELS = [
    "L0",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "L7",
    "L8",
    "L8S1",
    "L8S2",
    "L8S4",
    "L8M",
    "L8I",
    "L9A",
    "L9B",
    "L9C",
]

_BACKTICK_NAME = re.compile(r"`([A-Za-z0-9_\-.]+)`")
_JSON_NAME = re.compile(r'"name"\s*:\s*"([A-Za-z0-9_\-.]+)"')
_PURPOSE = re.compile(r"Its purpose: (.+?)\.(?:\s|$)")
_STOPWORDS = frozenset(
    "a an the of to for in on and or with by from is are be this that it its as when".split()
)
CONTAINMENT_THRESHOLD = 0.5


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS}


def lint(skill_prose: str, inventory: list[dict]) -> dict:
    """Return {flagged, rules} for one skill x manifest pair."""
    m = _BACKTICK_NAME.search(skill_prose) or _JSON_NAME.search(skill_prose)
    if not m:
        return {"flagged": False, "rules": [], "referenced": None}
    ref = m.group(1)
    by_name = {t["function"]["name"]: t["function"] for t in inventory if t.get("function")}
    rules: list[str] = []
    if ref not in by_name:
        rules.append("A-missing-identifier")
    else:
        desc = by_name[ref].get("description", "") or ""
        if desc.lstrip().startswith("[DEPRECATED"):
            rules.append("B-deprecated-reference")
        pm = _PURPOSE.search(skill_prose)
        if pm:
            purpose_words = _content_words(pm.group(1))
            if purpose_words:
                containment = len(purpose_words & _content_words(desc)) / len(purpose_words)
                if containment < CONTAINMENT_THRESHOLD:
                    rules.append("C-description-mismatch")
    return {"flagged": bool(rules), "rules": rules, "referenced": ref}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, default=Path("results/lint_report.json"))
    args = ap.parse_args()

    tasks = stratified_sample(75, 75, 25, seed=args.seed)
    per_level: dict[str, dict] = {}
    for lvl in LINT_LEVELS:
        n = flagged = 0
        rule_counts: dict[str, int] = defaultdict(int)
        for t in tasks:
            pt: PerturbedTask = perturb(t, lvl, inventory_size=12)
            if not pt.skill_instruction:
                continue
            r = lint(pt.skill_instruction, pt.inventory)
            n += 1
            flagged += r["flagged"]
            for rule in r["rules"]:
                rule_counts[rule] += 1
        per_level[lvl] = {
            "n": n,
            "flag_rate": round(flagged / n, 4) if n else None,
            "rules": dict(sorted(rule_counts.items())),
        }

    # Semantics-preserving levels: a flag is a false positive. L4's paraphrase and
    # L0's identity manifest should sail through; every identifier/description/
    # marker drift level should be caught.
    should_not_flag = {"L0", "L4", "L8I", "L9A", "L9B", "L9C"}
    should_flag = {"L1", "L2", "L3", "L5", "L6", "L7", "L8", "L8S1", "L8S2", "L8S4", "L8M"}
    fp = [per_level[lvl]["flag_rate"] for lvl in should_not_flag & set(per_level)]
    rec = [per_level[lvl]["flag_rate"] for lvl in should_flag & set(per_level)]
    summary = {
        "false_positive_rate_semantics_preserving": round(sum(fp) / len(fp), 4),
        "recall_drifted_levels": round(sum(rec) / len(rec), 4),
        "threshold": CONTAINMENT_THRESHOLD,
        "note": (
            "L9* (schema drift) is out of scope for a prose-level lint: the skill "
            "prose names no parameters, so schema evolution needs a schema-diff lint."
        ),
    }
    report = {"seed": args.seed, "n_tasks": len(tasks), "summary": summary, "per_level": per_level}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"lint report -> {args.out}")
    print(f"{'level':8s} {'n':>4s} {'flag%':>7s}  rules")
    for lvl, r in per_level.items():
        pct = f"{100 * r['flag_rate']:.1f}" if r["flag_rate"] is not None else "-"
        print(f"{lvl:8s} {r['n']:>4d} {pct:>7s}  {r['rules']}")
    print(f"\nsummary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
