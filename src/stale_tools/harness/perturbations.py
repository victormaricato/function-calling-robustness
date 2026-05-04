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

import hashlib
import re
from dataclasses import dataclass

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


def _skill_instruction(original_name: str, original_description: str) -> str:
    """Skill-document-style guidance pinned to the ORIGINAL (pre-rename) tool name.

    This simulates the real-world scenario: a skill or system prompt was written when the
    tool was called `original_name`. After perturbation, the inventory may no longer contain
    that name — the model has to bridge the gap.
    """
    short = original_description.strip().rstrip(".")
    if len(short) > 120:
        short = short[:120] + "..."
    return (
        f"When responding to the user, you may use the `{original_name}` tool. "
        f"Its purpose: {short}. Prefer this tool when applicable."
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
    return _DIRECTIVE_DECOY_TEMPLATES[_seed(seed_str + "directive") % len(_DIRECTIVE_DECOY_TEMPLATES)]


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


def _pad_inventory(target_fn: dict, target_size: int, seed: int) -> list[dict]:
    """Return a list of OpenAI tool defs of length target_size, with the target tool included."""
    out = [_to_oai(target_fn)]
    decoys = list(DECOY_TOOLS)
    # deterministic shuffle
    for i in range(len(decoys) - 1):
        j = (seed + i * 7) % (len(decoys) - i)
        decoys[i], decoys[i + j] = decoys[i + j], decoys[i]
    out.extend(_to_oai(d) for d in decoys[: target_size - 1])
    return out


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
        inventory += _pad_inventory(target, inventory_size, seed)[1:]
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
        inv += _pad_inventory(target, inventory_size, seed)[1:]
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
        inv += _pad_inventory(target, inventory_size, seed)[1:]
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
        inv += _pad_inventory(target, inventory_size, seed)[1:]
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
        inv += _pad_inventory(target, inventory_size, seed)[1:]
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
        inv += _pad_inventory(target, inventory_size, seed)[1:]
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
        inv += _pad_inventory(target, inventory_size, seed)[1:]
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
        inv += _pad_inventory(target, inventory_size, seed)[1:]
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

    if level in ("L8", "L8N", "L8I"):
        # Directive override: two candidates compete inside the inventory.
        #   Tool A (directive-named): identifier is `original_name` so the skill
        #     instruction citing it lands; description is a generic-superset
        #     sentence that does NOT specifically match the user's query.
        #   Tool B (description-match): renamed identifier, original description
        #     preserved so the description channel specifically endorses it.
        # L8 ablations vary the skill prose only:
        #   L8  — current default: skill names original_name (the directive).
        #   L8N — no-skill control: empty skill_instruction.
        #   L8I — intent-only skill: prose describes the user-intent class but
        #         names no tool, so the description channel does the routing.
        original_desc = target.get("description", "")
        tool_a = dict(target)
        tool_a["name"] = original_name
        tool_a["description"] = _directive_decoy_description(task["id"])
        new_name = _org_prefix_rename(original_name, task["id"] + "_match")
        if new_name == original_name:
            new_name = new_name + "_v2"
        tool_b = dict(target)
        tool_b["name"] = new_name
        tool_b["description"] = original_desc
        inv = [_to_oai(tool_a), _to_oai(tool_b)] + [_to_oai(f) for f in other_fns]
        inv += _pad_inventory(target, inventory_size, seed)[1:]
        if level == "L8":
            skill = _skill_instruction(original_name, original_desc)
        elif level == "L8N":
            skill = ""
        else:  # L8I
            skill = _intent_skill_instruction()
        return PerturbedTask(
            task["id"],
            level,
            new_name,
            original_name,
            inv[:inventory_size],
            instruction,
            skill,
            False,
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
