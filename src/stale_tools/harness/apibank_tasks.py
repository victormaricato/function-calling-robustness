"""API-Bank task loader. Converts API-Bank level-1 single-call tasks into the
same `task` dict shape that `perturb()` expects (BFCL-compatible), so the
existing perturbation/runner pipeline works unchanged.

API-Bank source schema (per row):
  - file: str
  - id: int
  - instruction: str (contains API descriptions as embedded JSON blocks)
  - input: str (multi-turn dialogue: "User: ... AI: ... User: ...")
  - expected_output: str ("API-Request: [ToolName(arg='val', ...)]")

Target shape (BFCL-style):
  {"id": "apibank_...", "question": [[{role, content}]], "function": [{name, description, parameters}, ...]}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from random import Random

CACHE = Path.home() / ".cache/huggingface/hub/datasets--liminghao1630--API-Bank"
LEVEL1 = next(CACHE.rglob("level-1-api.json"))


_API_DESC_RE = re.compile(r'\{"name":[^\n]+\}')
_GOLD_RE = re.compile(r"\[(\w+)\((.*?)\)\]\s*$", re.DOTALL)


def _parse_apis_from_instruction(instruction: str) -> list[dict]:
    """Each instruction includes one or more JSON-encoded API descriptions on
    separate lines. Parse them out into BFCL-style function dicts."""
    fns: list[dict] = []
    for m in _API_DESC_RE.finditer(instruction):
        try:
            api = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        params = api.get("input_parameters", {}) or {}
        properties: dict = {}
        required: list[str] = []
        for pname, pinfo in params.items():
            ptype = (pinfo.get("type") or "string").lower()
            ptype = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}.get(
                ptype, ptype
            )
            properties[pname] = {"type": ptype, "description": pinfo.get("description", "")}
            required.append(pname)
        fns.append(
            {
                "name": api["name"],
                "description": api.get("description", ""),
                "parameters": {"type": "object", "properties": properties, "required": required},
            }
        )
    return fns


def _parse_gold(expected_output: str) -> tuple[str, dict] | None:
    """Extract gold tool name and arguments from `API-Request: [Name(k='v', ...)]`."""
    m = _GOLD_RE.search(expected_output.replace("API-Request: ", "").strip())
    if not m:
        return None
    name = m.group(1)
    args_str = m.group(2)
    args: dict = {}
    # Lightweight key='value' parser. Values can be quoted strings, numbers, or bools.
    for kv in re.finditer(r"(\w+)\s*=\s*('([^']*)'|\"([^\"]*)\"|([^,\)]+))", args_str):
        k = kv.group(1)
        v = (
            kv.group(3)
            if kv.group(3) is not None
            else (kv.group(4) if kv.group(4) is not None else kv.group(5))
        )
        v = v.strip()
        args[k] = v
    return name, args


def load_tasks() -> list[dict]:
    """Return API-Bank level-1 tasks in BFCL-compatible shape."""
    raw = json.load(LEVEL1.open())
    out: list[dict] = []
    for r in raw:
        gold = _parse_gold(r["expected_output"])
        if gold is None:
            continue
        gold_name, _gold_args = gold
        fns = _parse_apis_from_instruction(r["instruction"])
        if not fns:
            continue
        target_idx = next((i for i, f in enumerate(fns) if f["name"] == gold_name), None)
        if target_idx is None:
            continue
        # Use the dialogue input as the user message.
        # API-Bank reuses ids across files, so namespace by file stem to avoid collisions.
        file_stem = Path(r["file"]).stem.replace(".", "_")[:60]
        task_id = f"apibank_{file_stem}_{r['id']:04d}"
        out.append(
            {
                "id": task_id,
                "question": [[{"role": "user", "content": r["input"]}]],
                "function": fns,
                "_apibank_category": "level-1",
                "_gold_target": gold_name,
                "_gold_target_idx": target_idx,
            }
        )
    return out


def stratified_sample(n: int, seed: int = 42) -> list[dict]:
    """Random sample of n API-Bank tasks. Filters tasks whose gold tool has
    1-3 args (matches BFCL `simple` shape — easier for the codebook)."""
    rng = Random(seed)
    pool = [
        t
        for t in load_tasks()
        if 1 <= len(t["function"][t["_gold_target_idx"]]["parameters"]["properties"]) <= 5
    ]
    rng.shuffle(pool)
    return pool[:n]


if __name__ == "__main__":
    tasks = stratified_sample(30)
    print(f"Sampled {len(tasks)} API-Bank tasks")
    for t in tasks[:3]:
        print(f"  {t['id']:30s} target={t['_gold_target']:25s} #fns={len(t['function'])}")
