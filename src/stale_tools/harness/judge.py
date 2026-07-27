"""Programmatic scorer + behavioural-codebook classifier."""

from __future__ import annotations

import json
from typing import Any

from .perturbations import PerturbedTask


def _names_in_inventory(inventory: list[dict]) -> set[str]:
    return {t["function"]["name"] for t in inventory if t.get("function")}


def _live_target_params(pt: PerturbedTask) -> dict:
    """The target tool's parameter schema as the model actually saw it."""
    for t in pt.inventory:
        fn = t.get("function") or {}
        if fn.get("name") == pt.target_name:
            return fn.get("parameters") or {}
    return {}


def _schema_conformance(pt: PerturbedTask, args: dict) -> tuple[bool, bool]:
    """Score a call's arguments against the LIVE schema (schema-drift levels L9*).

    Returns (schema_valid, stale_args):
      schema_valid — every live-required param present, every arg name declared in
        the live properties, and a retyped param (L9C) passed with the new type.
      stale_args — the call used the pre-drift schema: the renamed-away param name
        (L9A) or a missing newly-required param (L9B).
    """
    params = _live_target_params(pt)
    props = params.get("properties") or {}
    required = params.get("required") or []
    meta = pt.meta or {}
    schema_valid = all(r in args for r in required)
    if props:
        schema_valid = schema_valid and all(k in props for k in args)
    if meta.get("schema_mode") == "retype" and meta.get("retyped") in args:
        schema_valid = schema_valid and isinstance(args[meta["retyped"]], str)
    stale = False
    if meta.get("renamed_from"):
        stale = meta["renamed_from"] in args
    if meta.get("added_required"):
        stale = stale or meta["added_required"] not in args
    return schema_valid, stale


def _gold_for_live_schema(pt: PerturbedTask, gold: list[dict]) -> list[dict]:
    """Remap gold arg names onto the live (drifted) schema so arg_f1 stays meaningful.

    Only L9A changes arg names; L9B adds a param gold doesn't know about (scored via
    schema_valid, not arg_f1) and L9C only changes a declared type (loose string
    comparison in _arg_match already tolerates "5" vs 5).
    """
    meta = pt.meta or {}
    if meta.get("schema_mode") != "rename" or not meta.get("renamed_from"):
        return gold
    out = []
    for g in gold:
        fn_name = next(iter(g))
        inner = dict(g[fn_name])
        if meta["renamed_from"] in inner:
            inner[meta["renamed_to"]] = inner.pop(meta["renamed_from"])
        out.append({fn_name: inner})
    return out


def _arg_match(pred_args: dict, gold_call: dict) -> tuple[bool, float]:
    """Return (all_required_match, per_key_f1).

    `gold_call` is BFCL's ground-truth dict for one call, e.g.
        {"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}
    The dict has a single key (the gold function name); its value is a dict of
    arg-name -> list of acceptable values (with "" meaning the arg is optional).
    """
    fn_name = next(iter(gold_call))
    gold_args = gold_call[fn_name]

    # Required args have no empty string in their accepted-values list
    required = {k for k, vs in gold_args.items() if "" not in vs}
    matched = 0
    n_keys = max(len(gold_args), 1)
    for k, accepted in gold_args.items():
        if k not in pred_args:
            if "" in accepted:  # optional, missing is OK
                matched += 1
            continue
        v = pred_args[k]
        # accepted is a list of allowed values (any types). Loose comparison.
        ok = False
        for a in accepted:
            try:
                if a == v or str(a) == str(v):
                    ok = True
                    break
                if (
                    isinstance(a, (int, float))
                    and isinstance(v, (int, float))
                    and float(a) == float(v)
                ):
                    ok = True
                    break
            except Exception:
                continue
        if ok:
            matched += 1

    all_req_match = all(
        any(
            (k in pred_args and (pred_args[k] == a or str(pred_args[k]) == str(a)))
            for a in gold_args[k]
        )
        for k in required
    )
    return all_req_match, matched / n_keys


def classify(
    pt: PerturbedTask,
    response_tool_calls: list[dict] | None,
    response_text: str | None,
    gold: list[dict],
) -> dict[str, Any]:
    """Apply the four-class behavioural codebook + selection accuracy.

    `response_tool_calls`: list of {"name": str, "arguments": dict} or None.
    `response_text`: visible assistant text content or None.

    Returns a dict with keys:
      called_name, code (correct/wrong-neighbour/hallucinated/abstain),
      selection_correct (bool), arg_f1 (float), detection (bool),
      hallucinated_obsolete (bool), directive_followed (bool).

    `directive_followed` is True when the model called the skill-directed name
    while that name is present in the inventory — only meaningful at L_8, where
    the directive points to a tool whose description does not specifically match
    the query. At all other levels obsolete_name is either equal to the target
    or absent from the inventory, so this flag stays False there.
    """
    inv_names = _names_in_inventory(pt.inventory)
    detection = _detected(pt, response_text)

    if not response_tool_calls:
        return {
            "called_name": None,
            "code": "abstain",
            "selection_correct": False,
            "arg_f1": 0.0,
            "detection": detection,
            "hallucinated_obsolete": False,
            "directive_followed": False,
        }

    call = response_tool_calls[0]  # primary call (BFCL parallel evaluated separately if needed)
    name = call.get("name", "")
    args = call.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}

    # 1) Hallucinated identifier (not in inventory)
    if name not in inv_names:
        is_obsolete = pt.obsolete_name is not None and name == pt.obsolete_name
        return {
            "called_name": name,
            "code": "hallucinated",
            "selection_correct": False,
            "arg_f1": 0.0,
            "detection": detection,
            "hallucinated_obsolete": is_obsolete,
            "directive_followed": False,
        }

    # 2) In inventory but not the target → wrong neighbour
    if name != pt.target_name:
        directive_followed = (
            pt.obsolete_name is not None
            and name == pt.obsolete_name
            and pt.obsolete_name in inv_names
            and pt.obsolete_name != pt.target_name
        )
        return {
            "called_name": name,
            "code": "wrong-neighbour",
            "selection_correct": False,
            "arg_f1": 0.0,
            "detection": detection,
            "hallucinated_obsolete": False,
            "directive_followed": directive_followed,
        }

    # 3) Correct name — check arguments against any gold candidate
    # Gold is a list of acceptable function-call dicts (BFCL allows multiple gold answers)
    best_f1 = 0.0
    any_required_match = False
    for g in _gold_for_live_schema(pt, gold):
        # gold dict's key is the ORIGINAL canonical function name; we replace it with the
        # perturbed target name for comparison, since the args dict structure is what matters
        renamed = {pt.target_name: next(iter(g.values()))}
        ok, f1 = _arg_match(args, renamed)
        any_required_match = any_required_match or ok
        best_f1 = max(best_f1, f1)

    out = {
        "called_name": name,
        "code": "correct" if any_required_match else "wrong-neighbour",
        "selection_correct": any_required_match,
        "arg_f1": best_f1,
        "detection": detection,
        "hallucinated_obsolete": False,
        "directive_followed": False,
    }
    if pt.level.startswith("L9"):
        schema_valid, stale_args = _schema_conformance(pt, args)
        out["schema_valid"] = schema_valid
        out["stale_args"] = stale_args
    return out


def _detected(pt: PerturbedTask, text: str | None) -> bool:
    """Did the model surface awareness of the perturbation in its visible response?"""
    if not text:
        return False
    lower = text.lower()
    triggers = [
        "deprecated",
        "renamed",
        "no longer available",
        "obsolete",
        "this tool does not exist",
        "could not find the tool",
        "i cannot find a tool",
        "i do not see a tool",
        "did you mean",
        "i am unable to find",
    ]
    return any(t in lower for t in triggers)
