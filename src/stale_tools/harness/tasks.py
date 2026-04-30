"""BFCL v4 task loader. Stratified sampling helpers."""

from __future__ import annotations

import json
from pathlib import Path
from random import Random

DATA = Path(__file__).resolve().parents[3] / "data" / "bfcl"


def _load(filename: str) -> list[dict]:
    return [json.loads(l) for l in (DATA / filename).read_text().splitlines() if l.strip()]


def load_simple() -> list[dict]:
    return _load("BFCL_v4_simple_python.json")


def load_multiple() -> list[dict]:
    return _load("BFCL_v4_multiple.json")


def load_parallel() -> list[dict]:
    return _load("BFCL_v4_parallel.json")


def load_answers(filename: str) -> dict[str, list[dict]]:
    rows = _load("answer_" + filename)
    return {r["id"]: r["ground_truth"] for r in rows}


def stratified_sample(
    n_simple: int, n_multiple: int, n_parallel: int, seed: int = 42
) -> list[dict]:
    """Return a stratified sample of BFCL tasks.

    Each task carries:
      - `_bfcl_category` ("simple" / "multiple" / "parallel")
      - `_gold_target` (the canonical function name in the gold answer)
      - `_gold_target_idx` (index of that function in `task['function']`)

    For BFCL `multiple`, the gold target is not always at index 0; for `simple_python`
    there is only one function, so index 0 is correct by construction.
    """
    rng = Random(seed)
    answers = all_answers()
    out: list[dict] = []
    for fn, cat, k in [
        (load_simple, "simple", n_simple),
        (load_multiple, "multiple", n_multiple),
        (load_parallel, "parallel", n_parallel),
    ]:
        rows = fn()
        rng.shuffle(rows)
        for r in rows[:k]:
            r = dict(r)
            r["_bfcl_category"] = cat
            gold = answers.get(r["id"], [])
            if gold:
                first_call = gold[0]
                gold_name = next(iter(first_call.keys()))
                r["_gold_target"] = gold_name
                r["_gold_target_idx"] = next(
                    (i for i, f in enumerate(r["function"]) if f["name"] == gold_name),
                    0,
                )
            else:
                r["_gold_target"] = r["function"][0]["name"]
                r["_gold_target_idx"] = 0
            out.append(r)
    return out


def all_answers() -> dict[str, list[dict]]:
    """Combine answers from all three categories, keyed by task id."""
    out: dict[str, list[dict]] = {}
    for fn in ("BFCL_v4_simple_python.json", "BFCL_v4_multiple.json", "BFCL_v4_parallel.json"):
        out.update(load_answers(fn))
    return out
