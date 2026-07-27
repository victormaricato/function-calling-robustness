"""Perturbation operators P_ell mapping a clean tool inventory to a perturbed one.

Each level is deterministic given a seed and the clean inventory. We implement
all 9 levels: L0 (control), L1 (surface re-casing), L2 (synonym rename),
L3 (org-prefix rename), L4 (description rewrite), L5 (description drift),
L6 (deprecation marker), L7 (removal + replacement), L8 (directive override
— skill prescribes a tool whose description does not specifically match the
query, while a sibling tool's description does).

A "tool inventory" is a list of OpenAI-style tool definitions:
  [{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}, ...]

For each task, exactly one tool is the *target* (the gold call). Perturbations
modify the target's identifier and/or description, and may introduce decoy tools
to fill the inventory.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass, field
from random import Random

# Synonym pool for L_2 — semantically related renames per common verb roots.
# The mapping is deterministic and conservative (verb-preserving).
SYNONYM_BANK: dict[str, list[str]] = {
    "calculate": ["compute", "evaluate", "derive"],
    "compute": ["calculate", "evaluate", "derive"],
    "get": ["fetch", "retrieve", "lookup"],
    "fetch": ["get", "retrieve", "lookup"],
    "find": ["lookup", "search_for", "locate"],
    "search": ["query", "lookup", "find"],
    "run": ["execute", "invoke", "perform"],
    "execute": ["run", "invoke", "perform"],
    "send": ["dispatch", "submit", "post"],
    "create": ["build", "make", "construct"],
    "build": ["create", "make", "construct"],
    "predict": ["forecast", "estimate", "infer"],
    "convert": ["transform", "translate", "map"],
    "translate": ["convert", "transform", "render"],
    "list": ["enumerate", "show", "display"],
    "show": ["display", "render", "list"],
    "filter": ["select", "subset", "narrow"],
    "sort": ["order", "rank", "arrange"],
    "update": ["modify", "change", "edit"],
    "delete": ["remove", "erase", "drop"],
    "check": ["verify", "validate", "confirm"],
    "is": ["check_is", "verify_is", "test_is"],
    "process": ["handle", "transform", "consume"],
    "analyze": ["examine", "inspect", "study"],
}

ORG_PREFIXES = ["acme", "umbra", "globex", "wayne", "stark", "wonka", "initech", "soylent"]

DECOY_TOOLS: list[dict] = [
    {
        "name": "send_email",
        "description": "Send an email message.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "schedule_meeting",
        "description": "Schedule a calendar meeting.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "translate_text",
        "description": "Translate text between languages.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount between currencies.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "summarize_document",
        "description": "Summarize a long document.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "search_news",
        "description": "Search recent news articles.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "transcribe_audio",
        "description": "Transcribe an audio file to text.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "encode_image",
        "description": "Encode an image to base64.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "list_files",
        "description": "List files in a directory.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_user_profile",
        "description": "Get the profile of a user.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "post_to_slack",
        "description": "Post a message to a Slack channel.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "create_jira_ticket",
        "description": "Create a Jira ticket.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "lookup_address",
        "description": "Look up the address of a place.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "compute_distance",
        "description": "Compute the distance between two locations.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]


@dataclass
class PerturbedTask:
    task_id: str
    level: str
    target_name: str  # the name the model SHOULD call (under this perturbation)
    obsolete_name: str | None  # the name in the (stale) skill instruction (None if same as target)
    inventory: list[dict]  # OpenAI-style tool definitions, the FULL list shown to the model
    instruction: str  # user message
    skill_instruction: str  # system-prompt-style guidance that references obsolete_name (the
    # documentation drift the probe simulates: a skill document hardcoded an old identifier)
    deprecation_marker: bool  # True only at L_6
    meta: dict = field(default_factory=dict)  # operator-specific details (e.g. schema-drift edits)


def _short_desc(description: str) -> str:
    short = description.strip().rstrip(".")
    if len(short) > 120:
        short = short[:120] + "..."
    return short


def _skill_instruction(original_name: str, original_description: str) -> str:
    """Skill-document-style guidance pinned to the ORIGINAL (pre-rename) tool name.

    This simulates the real-world scenario: a skill or system prompt was written when the
    tool was called `original_name`. After perturbation, the inventory may no longer contain
    that name — the model has to bridge the gap.
    """
    short = _short_desc(original_description)
    return (
        f"When responding to the user, you may use the `{original_name}` tool. "
        f"Its purpose: {short}. Prefer this tool when applicable."
    )


# ─── Directive-strength variants of the L8 skill prose (rebuttal block E1) ──
# S3 (the "prefer" strength) is the default `_skill_instruction` above; the
# variants below bracket it on both sides of the directive-strength axis.


def _weak_skill_instruction(original_name: str, original_description: str) -> str:
    """S1 — weak mention: declarative reference, no imperative routing language."""
    short = _short_desc(original_description)
    return (
        f"For reference: a `{original_name}` tool exists in this workspace. Its purpose: {short}."
    )


def _example_skill_instruction(original_name: str) -> str:
    """S2 — example-only: a stale worked example naming the tool, no instruction at all."""
    return (
        "Example of a previously successful interaction for this kind of request:\n"
        f'  tool_call: {{"name": "{original_name}", "arguments": {{...}}}}'
    )


def _hard_skill_instruction(original_name: str, original_description: str) -> str:
    """S4 — hard routing rule: mandatory language plus an explicit prohibition."""
    short = _short_desc(original_description)
    return (
        f"Always route these requests to the `{original_name}` tool. Its purpose: {short}. "
        "Do not use any other tool for this."
    )


# L8W — the platform-side mitigation probe: the stale directive stays in place,
# but an automated skill-lint warning is appended to it.
_LINT_WARNING = (
    "\n\nNote: an automated skill lint flagged the tool reference above as potentially "
    "stale: the named tool may have been renamed, deprecated, or replaced in the live "
    "manifest. Verify the reference against the live tool descriptions before calling, "
    "and prefer the tool whose description matches the user's request."
)


def _seed(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


def _synonym_rename(name: str) -> str | None:
    """Pick a synonym-rename by replacing the leading verb root."""
    parts = re.split(r"[_\-]", name)
    head = parts[0].lower() if parts else ""
    if head not in SYNONYM_BANK:
        return None
    pool = SYNONYM_BANK[head]
    pick = pool[_seed(name) % len(pool)]
    return "_".join([pick] + parts[1:])


def _surface_recase(name: str, seed_str: str) -> str:
    """L1: case/separator mangling. Identifier preserved up to surface form.

    Rotates through a small set of surface variants so different tasks land on
    different shapes (PascalCase, kebab-case, ALLCAPS, etc.)."""
    parts = re.split(r"[_\-]", name)
    pick = _seed(seed_str + "case") % 4
    if pick == 0:
        return "".join(p.capitalize() for p in parts)  # PascalCase
    if pick == 1:
        return "-".join(parts)  # kebab-case
    if pick == 2:
        return "_".join(p.upper() for p in parts)  # SCREAMING_SNAKE
    return parts[0] + "".join(p.capitalize() for p in parts[1:])  # camelCase


_DESC_PARAPHRASE_PREFIXES = (
    "This function is used to ",
    "Use this to ",
    "Helper that will ",
    "Utility to ",
    "When invoked, this will ",
)


def _paraphrase_description(desc: str, seed_str: str) -> str:
    """L4: description rewrite, semantics preserved.

    Deterministic verb-preserving paraphrase: lowercase the first letter,
    drop a trailing period, and prepend one of a small set of stock framings.
    Resulting text retains the same content words and intent, but the surface
    string changes meaningfully (Levenshtein > 8 on any non-trivial description).
    """
    body = desc.strip().rstrip(".")
    if not body:
        return desc
    body = body[0].lower() + body[1:] if body[0].isupper() else body
    pref = _DESC_PARAPHRASE_PREFIXES[_seed(seed_str + "para") % len(_DESC_PARAPHRASE_PREFIXES)]
    return pref + body + "."


_DRIFT_TEMPLATES = (
    "Performs an internal bookkeeping task on the user's account.",
    "Returns metadata for a stored configuration entry.",
    "Logs a diagnostic event for the host process.",
    "Updates a cached value in a key-value store.",
    "Prepares an outbound notification for delivery.",
    "Resolves an opaque identifier into a display label.",
)


def _drift_description(seed_str: str) -> str:
    """L5: description drift. Replace the description with a plausible-but-unrelated
    summary that no longer matches the function's true semantics. The identifier is
    kept (the "true" tool is still there by name), but description-routing should
    fail at this level."""
    return _DRIFT_TEMPLATES[_seed(seed_str + "drift") % len(_DRIFT_TEMPLATES)]


_DIRECTIVE_DECOY_TEMPLATES = (
    "Run a query against the data warehouse and return rows.",
    "Execute a generic platform action chosen by the caller.",
    "Look up a record in the system by free-form input.",
    "Process the user's request by dispatching to a backend service.",
    "Perform a generic capability call defined by the platform manifest.",
    "Handle a system operation selected by the caller's intent string.",
)


def _directive_decoy_description(seed_str: str) -> str:
    """L8: a generic-but-plausibly-superset description for the skill-directed tool.

    The directive tool's description is broader than the user's request, so the
    description channel does not specifically endorse it. A sibling tool, with the
    original (specifically matching) description and a renamed identifier, is
    available alongside. The contrast tests directive-following vs description-
    matching when the two channels disagree.
    """
    return _DIRECTIVE_DECOY_TEMPLATES[
        _seed(seed_str + "directive") % len(_DIRECTIVE_DECOY_TEMPLATES)
    ]


def _org_prefix_rename(name: str, seed_str: str) -> str:
    """Generic org-prefixed name unrelated to the original verb."""
    pref = ORG_PREFIXES[_seed(seed_str) % len(ORG_PREFIXES)]
    # Drop the verb head, replace with a generic 'process' / 'handler' / 'op'.
    bodies = ["call", "op", "fn", "handler", "process"]
    body = bodies[_seed(seed_str + "b") % len(bodies)]
    suffix = format(_seed(seed_str + "s") % 9999, "04d")
    return f"{pref}_{body}_{suffix}"


_TYPE_MAP = {
    "dict": "object",
    "list": "array",
    "tuple": "array",
    "any": "string",  # OpenAI/Gemini reject "any"; coarse approximation is "string"
    "Dict": "object",
    "List": "array",
    "Tuple": "array",
    "Any": "string",
    "float": "number",
    "long": "integer",
    "str": "string",
    "int": "integer",
    "bool": "boolean",
    "Str": "string",
    "Int": "integer",
    "Bool": "boolean",
    "Float": "number",
}


_VALID_JSON_TYPES = {"string", "integer", "number", "boolean", "object", "array", "null"}


def _normalize_schema(node):
    """Recursively normalize a JSON-Schema-ish dict to OpenAI/Vertex-compatible types.

    Some BFCL/APIBank tasks emit Python-style annotations like `list(str)` that
    OpenAI's strict schema rejects. We map known aliases via _TYPE_MAP and
    fall back to `string` for anything else, so the runner never produces an
    invalid tool schema.
    """
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                if v in _TYPE_MAP:
                    out[k] = _TYPE_MAP[v]
                elif v in _VALID_JSON_TYPES:
                    out[k] = v
                elif v.startswith(("list", "List", "tuple", "Tuple")):
                    out[k] = "array"
                elif v.startswith(("dict", "Dict")):
                    out[k] = "object"
                else:
                    out[k] = "string"
            else:
                out[k] = _normalize_schema(v)
        # If we just produced an array without an `items` schema, OpenAI's strict
        # mode rejects it; fill in a permissive items={"type": "string"} default.
        if out.get("type") == "array" and "items" not in out:
            out["items"] = {"type": "string"}
        return out
    if isinstance(node, list):
        return [_normalize_schema(x) for x in node]
    return node


_NAME_RE = re.compile(r"[^A-Za-z0-9_-]")


def _sanitize_name(name: str) -> str:
    """Conform identifier to OpenAI's tool-name pattern '^[a-zA-Z0-9_-]+$'.

    Some BFCL tool names contain dots (e.g. `religion.history_info`), and L1's
    case-mangle / L2's verb-prefix fallback both preserve the dot, which then
    fails OpenAI's tool-schema validation with HTTP 400. Replace any non-allowed
    char with `_`. Idempotent for already-valid names.
    """
    return _NAME_RE.sub("_", name)


def _to_oai(fn: dict) -> dict:
    """Wrap a raw function definition into OpenAI tools schema, normalising types."""
    fn = dict(fn)
    if "name" in fn:
        fn["name"] = _sanitize_name(fn["name"])
    if "parameters" in fn:
        fn["parameters"] = _normalize_schema(fn["parameters"])
    return {"type": "function", "function": fn}


_BFCL_POOL: list[dict] | None = None


def _bfcl_decoy_pool() -> list[dict]:
    """All unique BFCL v4 function definitions, sorted by name for determinism.

    Used to pad inventories past the 16-tool capacity of the generic DECOY_TOOLS
    list with *real* tool definitions (name + specific description + schema), so
    large-inventory runs (50/100 tools) face realistic neighbours rather than
    synthetic filler.
    """
    global _BFCL_POOL
    if _BFCL_POOL is None:
        from .tasks import load_multiple, load_parallel, load_simple

        seen: dict[str, dict] = {}
        for loader in (load_simple, load_multiple, load_parallel):
            for row in loader():
                for fn in row.get("function", []):
                    name = _sanitize_name(fn.get("name", ""))
                    if name and name not in seen:
                        f = dict(fn)
                        f["name"] = name
                        seen[name] = f
        _BFCL_POOL = sorted(seen.values(), key=lambda f: f["name"])
    return _BFCL_POOL


def _pad_inventory(
    target_fn: dict, target_size: int, seed: int, exclude: set[str] | frozenset = frozenset()
) -> list[dict]:
    """Return a list of OpenAI tool defs of length target_size, with the target tool included.

    Up to 16 tools the behaviour (and the resulting inventory bytes) is identical to the
    original DECOY_TOOLS-based padding, preserving comparability with previously recorded
    runs. Beyond that, additional decoys are drawn deterministically from the full BFCL
    function pool, skipping any name in `exclude` (names already present in the inventory).
    """
    out = [_to_oai(target_fn)]
    decoys = list(DECOY_TOOLS)
    # deterministic shuffle
    for i in range(len(decoys) - 1):
        j = (seed + i * 7) % (len(decoys) - i)
        decoys[i], decoys[i + j] = decoys[i + j], decoys[i]
    out.extend(_to_oai(d) for d in decoys[: target_size - 1])
    if len(out) < target_size:
        taken = {t["function"]["name"] for t in out} | set(exclude)
        pool = list(_bfcl_decoy_pool())
        Random(seed).shuffle(pool)
        for f in pool:
            if len(out) >= target_size:
                break
            if f["name"] in taken:
                continue
            taken.add(f["name"])
            out.append(_to_oai(f))
    return out


def _inv_names(inv: list[dict]) -> set[str]:
    return {t["function"]["name"] for t in inv if t.get("function")}


def _matched_decoy_description(seed_str: str, original_desc: str, exclude_name: str) -> str:
    """L8M: a *specific* description borrowed from a different real BFCL function.

    The matched-description counterfactual removes the description-quality gap of L8:
    instead of a generic-superset sentence, the directive-named tool carries a concrete,
    domain-specific description (of some other capability), length-matched to the gold
    tool's description as closely as the pool allows.
    """
    pool = _bfcl_decoy_pool()
    n = len(pool)
    start = _seed(seed_str + "match") % n
    cands = [pool[(start + i) % n] for i in range(40)]
    cands = [
        c
        for c in cands
        if c["name"] != exclude_name and c.get("description") and c["description"] != original_desc
    ]
    best = min(cands, key=lambda c: abs(len(c["description"]) - len(original_desc)))
    return best["description"]


def _sibling_variants(fn: dict, seed_str: str, k: int = 3, org_style: bool = False) -> list[dict]:
    """Near-duplicate siblings of `fn` for the semantic-crowding condition.

    Each sibling keeps the function's schema, takes a distinct name variant, and a
    paraphrased (semantics-preserving) description — the crowding is in the description
    channel, where several tools now plausibly match the user's request.
    """
    base_name = fn["name"]
    desc = fn.get("description", "")
    out: list[dict] = []
    used = {base_name}
    for i in range(k):
        if org_style:
            cand = _org_prefix_rename(base_name, seed_str + f"sib{i}")
        elif i == 0:
            cand = _surface_recase(base_name, seed_str + "sib0")
        elif i == 1:
            cand = _synonym_rename(base_name) or "do_" + base_name
        else:
            cand = base_name + "_v2"
        if cand in used:
            cand = f"{base_name}_alt{i}"
        used.add(cand)
        s = dict(fn)
        s["name"] = cand
        s["description"] = _paraphrase_description(desc, seed_str + f"sib{i}")
        out.append(s)
    return out


# ─── Schema-drift operators (rebuttal block E5): L9A / L9B / L9C ────────────
# The identifier and description are untouched; the *parameters* of the live tool
# have evolved while the skill documentation (and the gold answer) remain stale.

_PARAM_RENAME_SUFFIX = "_value"


def _perturb_schema(fn: dict, mode: str) -> tuple[dict, dict]:
    """Return (new_fn, meta) where new_fn carries a drifted parameter schema.

    Modes:
      rename       — L9A: one required parameter renamed (stale arg names no longer valid)
      add_required — L9B: a new required parameter added to the live schema
      retype       — L9C: one parameter's declared type changed to string
    Meta records exactly what changed so the judge can score schema conformance.
    """
    fn = copy.deepcopy(fn)
    params = fn.get("parameters") or {}
    props = params.get("properties") or {}
    required = list(params.get("required") or [])
    meta: dict = {"schema_mode": mode}

    if mode == "rename":
        pick = required[0] if required else (sorted(props)[0] if props else None)
        if pick is None:
            return fn, meta
        new = pick + _PARAM_RENAME_SUFFIX if not pick.endswith(_PARAM_RENAME_SUFFIX) else pick + "2"
        props[new] = props.pop(pick)
        params["required"] = [new if r == pick else r for r in required]
        meta.update({"renamed_from": pick, "renamed_to": new})
    elif mode == "add_required":
        newp = "request_context"
        props[newp] = {
            "type": "string",
            "description": "Opaque routing context string required by the platform for this call.",
        }
        params["required"] = required + [newp]
        meta.update({"added_required": newp})
    elif mode == "retype":
        pick = None
        for cand in required or sorted(props):
            t = (props.get(cand) or {}).get("type")
            if t in ("integer", "number", "boolean", "int", "float", "bool"):
                pick = cand
                break
        if pick is None:
            if not (required or props):
                return fn, meta
            pick = required[0] if required else sorted(props)[0]
        entry = dict(props.get(pick) or {})
        old_t = entry.get("type", "string")
        entry["type"] = "string"
        entry["description"] = (
            (entry.get("description") or "").rstrip(". ") + ". Pass the value as a string."
        ).strip()
        props[pick] = entry
        meta.update({"retyped": pick, "from_type": old_t, "to_type": "string"})
    else:
        raise ValueError(f"Unknown schema mode: {mode}")

    params["properties"] = props
    fn["parameters"] = params
    return fn, meta


def perturb(task: dict, level: str, inventory_size: int = 16) -> PerturbedTask:
    """Apply perturbation P_level to a BFCL task and return a PerturbedTask.

    `task` shape (BFCL v4):
      {"id": "...", "question": [[{role, content}]], "function": [{name, description, parameters}, ...]}

    For multi-tool tasks (BFCL `multiple` or `parallel`) we treat the FIRST function as
    the target by convention. For BFCL `simple` there is exactly one function.
    """
    seed = _seed(task["id"])
    # The target is the function whose name appears in the gold answer (set by tasks.py).
    # For BFCL `simple_python` this is index 0 by construction; for `multiple` it can be anywhere.
    target_idx = task.get("_gold_target_idx", 0)
    target = dict(task["function"][target_idx])  # shallow copy
    # Sanitize source identifiers up-front so downstream PerturbedTask.target_name
    # always matches what the model sees in the inventory (OpenAI rejects dots).
    target["name"] = _sanitize_name(target["name"])
    original_name = target["name"]
    instruction = task["question"][0][0]["content"]

    # Other functions in the inventory are kept as-is (un-perturbed); they are the real
    # neighbours the model could mis-select.
    other_fns = [dict(f) for i, f in enumerate(task["function"]) if i != target_idx]

    if level == "L0":
        inventory = [_to_oai(target)] + [_to_oai(f) for f in other_fns]
        inventory += _pad_inventory(target, inventory_size, seed, _inv_names(inventory))[1:]
        inventory = inventory[:inventory_size]
        skill = _skill_instruction(original_name, target.get("description", ""))
        return PerturbedTask(
            task["id"], "L0", original_name, original_name, inventory, instruction, skill, False
        )

    if level == "L1":
        new_name = _surface_recase(original_name, task["id"])
        if new_name == original_name:
            new_name = original_name.upper()
        target["name"] = new_name
        inv = [_to_oai(target)] + [_to_oai(f) for f in other_fns]
        inv += _pad_inventory(target, inventory_size, seed, _inv_names(inv))[1:]
        skill = _skill_instruction(original_name, target.get("description", ""))
        return PerturbedTask(
            task["id"],
            "L1",
            new_name,
            original_name,
            inv[:inventory_size],
            instruction,
            skill,
            False,
        )

    if level == "L2":
        new_name = _synonym_rename(original_name)
        if new_name is None or new_name == original_name:
            # Fall back: prepend a synonym verb generically
            new_name = "do_" + original_name
        # Replace the target's name in the inventory
        target["name"] = new_name
        inv = [_to_oai(target)] + [_to_oai(f) for f in other_fns]
        inv += _pad_inventory(target, inventory_size, seed, _inv_names(inv))[1:]
        # Instruction still references the OLD name (implicitly, via natural language —
        # the user message in BFCL doesn't usually mention the function name explicitly,
        # but we add a system hint that names the obsolete tool to make the perturbation tangible).
        instr = instruction
        skill = _skill_instruction(original_name, target.get("description", ""))
        return PerturbedTask(
            task["id"], "L2", new_name, original_name, inv[:inventory_size], instr, skill, False
        )

    if level == "L3":
        new_name = _org_prefix_rename(original_name, task["id"])
        target["name"] = new_name
        inv = [_to_oai(target)] + [_to_oai(f) for f in other_fns]
        inv += _pad_inventory(target, inventory_size, seed, _inv_names(inv))[1:]
        skill = _skill_instruction(original_name, target.get("description", ""))
        return PerturbedTask(
            task["id"],
            "L3",
            new_name,
            original_name,
            inv[:inventory_size],
            instruction,
            skill,
            False,
        )

    if level == "L4":
        # Identifier preserved; description paraphrased (semantics preserved).
        original_desc = target.get("description", "")
        target["description"] = _paraphrase_description(original_desc, task["id"])
        inv = [_to_oai(target)] + [_to_oai(f) for f in other_fns]
        inv += _pad_inventory(target, inventory_size, seed, _inv_names(inv))[1:]
        skill = _skill_instruction(original_name, original_desc)
        return PerturbedTask(
            task["id"],
            "L4",
            original_name,
            original_name,
            inv[:inventory_size],
            instruction,
            skill,
            False,
        )

    if level == "L5":
        # Identifier preserved; description drifted (semantics broken).
        original_desc = target.get("description", "")
        target["description"] = _drift_description(task["id"])
        inv = [_to_oai(target)] + [_to_oai(f) for f in other_fns]
        inv += _pad_inventory(target, inventory_size, seed, _inv_names(inv))[1:]
        skill = _skill_instruction(original_name, original_desc)
        return PerturbedTask(
            task["id"],
            "L5",
            original_name,
            original_name,
            inv[:inventory_size],
            instruction,
            skill,
            False,
        )

    if level == "L6":
        new_name = _org_prefix_rename(original_name, task["id"])
        # Both names appear: obsolete (with deprecation marker) and the new target.
        obsolete_fn = dict(target)
        obsolete_fn["name"] = original_name
        obsolete_fn["description"] = f"[DEPRECATED — use `{new_name}` instead] " + obsolete_fn.get(
            "description", ""
        )
        target["name"] = new_name
        inv = [_to_oai(obsolete_fn), _to_oai(target)] + [_to_oai(f) for f in other_fns]
        inv += _pad_inventory(target, inventory_size, seed, _inv_names(inv))[1:]
        skill = _skill_instruction(original_name, target.get("description", ""))
        return PerturbedTask(
            task["id"],
            "L6",
            new_name,
            original_name,
            inv[:inventory_size],
            instruction,
            skill,
            True,
        )

    if level == "L7":
        # Old name absent; new name present. (Same as L3 for the inventory; we keep them
        # separated because in real deployments, between L3 and L7 the system prompt may or
        # may not still reference the old name. In this study, instructions are identical
        # across L0-L7 since BFCL prompts don't textually name the tool.)
        new_name = _org_prefix_rename(original_name, task["id"] + "_v2")
        target["name"] = new_name
        inv = [_to_oai(target)] + [_to_oai(f) for f in other_fns]
        inv += _pad_inventory(target, inventory_size, seed, _inv_names(inv))[1:]
        skill = _skill_instruction(original_name, target.get("description", ""))
        return PerturbedTask(
            task["id"],
            "L7",
            new_name,
            original_name,
            inv[:inventory_size],
            instruction,
            skill,
            False,
        )

    if level in ("L8", "L8N", "L8I", "L8S1", "L8S2", "L8S4", "L8W", "L8M", "L8C"):
        # Directive override: two candidates compete inside the inventory.
        #   Tool A (directive-named): identifier is `original_name` so the skill
        #     instruction citing it lands; description is a generic-superset
        #     sentence that does NOT specifically match the user's query.
        #   Tool B (description-match): renamed identifier, original description
        #     preserved so the description channel specifically endorses it.
        # Skill-prose ablations vary the prose only:
        #   L8   — default directive: "you may use X ... prefer this tool" (S3 strength).
        #   L8N  — no-skill control: empty skill_instruction.
        #   L8I  — intent-only skill: prose describes the user-intent class but
        #          names no tool, so the description channel does the routing.
        #   L8S1 — weak mention (declarative reference, no imperative).
        #   L8S2 — example-only (a stale worked example naming X, no instruction).
        #   L8S4 — hard routing rule ("always use X ... do not use any other tool").
        #   L8W  — default directive + an automated skill-lint staleness warning.
        # Inventory ablations vary tool A / the neighbourhood instead:
        #   L8M  — matched-description counterfactual: tool A carries a specific,
        #          length-matched (borrowed) description instead of a generic one.
        #   L8C  — semantic crowding: near-duplicate siblings of tool B compete
        #          inside the description channel.
        original_desc = target.get("description", "")
        tool_a = dict(target)
        tool_a["name"] = original_name
        if level == "L8M":
            tool_a["description"] = _matched_decoy_description(
                task["id"], original_desc, original_name
            )
        else:
            tool_a["description"] = _directive_decoy_description(task["id"])
        new_name = _org_prefix_rename(original_name, task["id"] + "_match")
        if new_name == original_name:
            new_name = new_name + "_v2"
        tool_b = dict(target)
        tool_b["name"] = new_name
        tool_b["description"] = original_desc
        front = [tool_a, tool_b]
        meta: dict = {}
        if level == "L8C":
            siblings = _sibling_variants(tool_b, task["id"], k=3, org_style=True)
            front += siblings
            meta["sibling_names"] = [s["name"] for s in siblings]
        if level == "L8M":
            meta["decoy_description"] = tool_a["description"]
        inv = [_to_oai(f) for f in front] + [_to_oai(f) for f in other_fns]
        inv += _pad_inventory(target, inventory_size, seed, _inv_names(inv))[1:]
        if level == "L8N":
            skill = ""
        elif level == "L8I":
            skill = _intent_skill_instruction()
        elif level == "L8S1":
            skill = _weak_skill_instruction(original_name, original_desc)
        elif level == "L8S2":
            skill = _example_skill_instruction(original_name)
        elif level == "L8S4":
            skill = _hard_skill_instruction(original_name, original_desc)
        elif level == "L8W":
            skill = _skill_instruction(original_name, original_desc) + _LINT_WARNING
        else:  # L8, L8M, L8C — the default directive prose
            skill = _skill_instruction(original_name, original_desc)
        return PerturbedTask(
            task["id"],
            level,
            new_name,
            original_name,
            inv[:inventory_size],
            instruction,
            skill,
            False,
            meta,
        )

    if level == "L0C":
        # Semantic-crowding control: the un-perturbed target competes against
        # near-duplicate siblings (variant names, paraphrased descriptions).
        siblings = _sibling_variants(target, task["id"], k=3)
        inv = [_to_oai(target)] + [_to_oai(s) for s in siblings] + [_to_oai(f) for f in other_fns]
        inv += _pad_inventory(target, inventory_size, seed, _inv_names(inv))[1:]
        skill = _skill_instruction(original_name, target.get("description", ""))
        return PerturbedTask(
            task["id"],
            "L0C",
            original_name,
            original_name,
            inv[:inventory_size],
            instruction,
            skill,
            False,
            {"sibling_names": [s["name"] for s in siblings]},
        )

    if level in ("L9A", "L9B", "L9C"):
        # Schema drift: identifier and description unchanged; the live parameter
        # schema has evolved past the (stale) skill documentation and gold answer.
        mode = {"L9A": "rename", "L9B": "add_required", "L9C": "retype"}[level]
        drifted, meta = _perturb_schema(target, mode)
        inv = [_to_oai(drifted)] + [_to_oai(f) for f in other_fns]
        inv += _pad_inventory(drifted, inventory_size, seed, _inv_names(inv))[1:]
        skill = _skill_instruction(original_name, target.get("description", ""))
        return PerturbedTask(
            task["id"],
            level,
            original_name,
            original_name,
            inv[:inventory_size],
            instruction,
            skill,
            False,
            meta,
        )

    raise ValueError(f"Unsupported level: {level}")


def _intent_skill_instruction() -> str:
    """Skill prose that describes the user-intent contract without naming any tool.

    Used by the L8I ablation to test whether removing identifier-pinning prose
    while preserving skill-style guidance recovers description-channel routing.
    """
    return (
        "When the user's request requires calling a tool, select the most "
        "appropriate one from the available inventory based on each tool's "
        "description and how specifically it matches the request."
    )


LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8")

# Rebuttal-block level groups (all reuse the L8 A/B construction unless noted).
DIRECTIVE_STRENGTH_LEVELS = ("L8S1", "L8S2", "L8S4")  # S3 == L8 itself
SCHEMA_DRIFT_LEVELS = ("L9A", "L9B", "L9C")
CROWDING_LEVELS = ("L0C", "L8C")
MATCHED_DESC_LEVELS = ("L8M",)
LINT_WARN_LEVELS = ("L8W",)
