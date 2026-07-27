"""Smoke tests for the perturbation operator and the model registry."""

import json

from stale_tools.harness.judge import classify
from stale_tools.harness.models import (
    ALL_MODELS,
    EFFORT_SWEEP,
    NEW_SOTA,
    SIZE_LADDER,
    SOTA,
    by_nickname,
    core_12,
    scale_8,
    schema_pilot,
)
from stale_tools.harness.perturbations import (
    CROWDING_LEVELS,
    DECOY_TOOLS,
    DIRECTIVE_STRENGTH_LEVELS,
    LEVELS,
    SCHEMA_DRIFT_LEVELS,
    perturb,
)


def _toy_task() -> dict:
    return {
        "id": "toy_0",
        "_bfcl_category": "simple",
        "_gold_target": "get_weather",
        "_gold_target_idx": 0,
        "function": [
            {
                "name": "get_weather",
                "description": "Return the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        "question": [[{"role": "user", "content": "What's the weather in Berlin right now?"}]],
    }


def test_levels_cover_the_taxonomy() -> None:
    assert tuple(LEVELS) == ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8")


def test_perturb_runs_at_every_level() -> None:
    task = _toy_task()
    for lvl in LEVELS:
        pt = perturb(task, lvl, inventory_size=8)
        assert pt.level == lvl
        assert len(pt.inventory) == 8
        assert pt.target_name
        assert pt.obsolete_name


def test_l0_preserves_identifier() -> None:
    task = _toy_task()
    pt = perturb(task, "L0", inventory_size=8)
    assert pt.target_name == "get_weather"


def test_l8_directive_override() -> None:
    """L8 must keep the directive name in the inventory but route the gold answer
    to a sibling whose description matches the query, with the skill citing the
    directive name."""
    task = _toy_task()
    pt = perturb(task, "L8", inventory_size=8)
    inv_names = [t["function"]["name"] for t in pt.inventory]
    # The skill-directed name (original) is in the inventory.
    assert pt.obsolete_name == "get_weather"
    assert "get_weather" in inv_names
    # The gold target is a different name from the skill-directed name.
    assert pt.target_name != pt.obsolete_name
    assert pt.target_name in inv_names
    # The directive tool's description has been replaced with a generic one;
    # the gold tool still carries the original description.
    by_name = {t["function"]["name"]: t["function"] for t in pt.inventory}
    assert by_name[pt.obsolete_name]["description"] != "Return the current weather for a city."
    assert by_name[pt.target_name]["description"] == "Return the current weather for a city."
    # The skill instruction names the directive tool.
    assert "`get_weather`" in pt.skill_instruction


def test_model_registry_consistency() -> None:
    assert len(SOTA) >= 12
    assert any(m.nickname == "Llama-3.3-70B" for m in SOTA)
    assert {m.effort for m in EFFORT_SWEEP if m.effort} == {"off", "low", "med", "high"}
    assert all(m.is_reasoning is False for m in SIZE_LADDER)
    assert by_nickname("Opus-4.7").slug == "anthropic/claude-opus-4.7"
    assert all(m in ALL_MODELS for m in SOTA + EFFORT_SWEEP + SIZE_LADDER)


def _math_task() -> dict:
    """Toy task whose name does not collide with any DECOY_TOOLS entry and whose
    schema has a numeric required param (needed by the L9C retype test)."""
    return {
        "id": "toy_math_0",
        "_bfcl_category": "simple",
        "_gold_target": "calculate_area",
        "_gold_target_idx": 0,
        "function": [
            {
                "name": "calculate_area",
                "description": "Calculate the area of a rectangle from width and height.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                    },
                    "required": ["width", "height"],
                },
            }
        ],
        "question": [[{"role": "user", "content": "What is the area of a 3 by 5 rectangle?"}]],
    }


def test_rebuttal_roster_lists() -> None:
    assert len(NEW_SOTA) == 7
    assert len(core_12()) == 12
    assert len(schema_pilot()) == 4
    assert len(scale_8()) == 8
    nicks = [m.nickname for m in ALL_MODELS]
    assert len(nicks) == len(set(nicks)), "duplicate nicknames break by_nickname/resume keys"
    assert by_nickname("Opus-5").slug == "anthropic/claude-opus-5"
    assert by_nickname("GPT-5.6-Sol").slug == "openai/gpt-5.6-sol"


def test_legacy_padding_unchanged_at_small_sizes() -> None:
    """At <=16 tools, padding must draw only from DECOY_TOOLS so inventories stay
    byte-comparable with previously recorded runs."""
    decoy_names = {d["name"] for d in DECOY_TOOLS}
    task = _math_task()
    for lvl in LEVELS:
        pt = perturb(task, lvl, inventory_size=12)
        assert len(pt.inventory) == 12
        extras = (
            {t["function"]["name"] for t in pt.inventory}
            - decoy_names
            - {pt.target_name, pt.obsolete_name, "calculate_area"}
        )
        assert not extras, f"{lvl}: unexpected non-legacy decoys {extras}"


def test_large_inventories_reach_requested_size() -> None:
    task = _math_task()
    for size in (24, 50, 100):
        pt = perturb(task, "L8", inventory_size=size)
        names = [t["function"]["name"] for t in pt.inventory]
        assert len(names) == size, f"requested {size}, got {len(names)}"
        assert len(set(names)) == size, "duplicate tool names in inventory"
        assert pt.target_name in names and pt.obsolete_name in names


def test_perturb_is_deterministic() -> None:
    task = _math_task()
    for lvl in (
        list(LEVELS)
        + list(DIRECTIVE_STRENGTH_LEVELS)
        + list(SCHEMA_DRIFT_LEVELS)
        + list(CROWDING_LEVELS)
        + ["L8M", "L8W", "L8N", "L8I"]
    ):
        a = perturb(task, lvl, inventory_size=50)
        b = perturb(task, lvl, inventory_size=50)
        assert json.dumps(a.inventory, sort_keys=True) == json.dumps(b.inventory, sort_keys=True)
        assert a.skill_instruction == b.skill_instruction


def test_directive_strength_variants() -> None:
    task = _math_task()
    base = perturb(task, "L8", inventory_size=12)
    s1 = perturb(task, "L8S1", inventory_size=12)
    s2 = perturb(task, "L8S2", inventory_size=12)
    s4 = perturb(task, "L8S4", inventory_size=12)
    # Same A/B inventory construction as L8 — only the skill prose varies.
    for v in (s1, s2, s4):
        assert v.target_name == base.target_name
        assert v.obsolete_name == base.obsolete_name
        assert json.dumps(v.inventory, sort_keys=True) == json.dumps(base.inventory, sort_keys=True)
    assert "For reference" in s1.skill_instruction and "Prefer" not in s1.skill_instruction
    assert '"name": "calculate_area"' in s2.skill_instruction
    assert "Always route" in s4.skill_instruction
    assert "Do not use any other tool" in s4.skill_instruction


def test_l8w_appends_lint_warning() -> None:
    task = _math_task()
    base = perturb(task, "L8", inventory_size=12)
    warn = perturb(task, "L8W", inventory_size=12)
    assert warn.skill_instruction.startswith(base.skill_instruction)
    assert "automated skill lint" in warn.skill_instruction
    assert json.dumps(warn.inventory, sort_keys=True) == json.dumps(base.inventory, sort_keys=True)


def test_l8m_matched_description_is_specific() -> None:
    task = _math_task()
    base = perturb(task, "L8", inventory_size=12)
    matched = perturb(task, "L8M", inventory_size=12)
    by_name = {t["function"]["name"]: t["function"] for t in matched.inventory}
    a_desc = by_name[matched.obsolete_name]["description"]
    base_by_name = {t["function"]["name"]: t["function"] for t in base.inventory}
    assert a_desc != base_by_name[base.obsolete_name]["description"]
    assert a_desc != "Calculate the area of a rectangle from width and height."
    assert matched.meta.get("decoy_description") == a_desc


def test_crowding_levels_inject_siblings() -> None:
    task = _math_task()
    for lvl in CROWDING_LEVELS:
        pt = perturb(task, lvl, inventory_size=12)
        names = {t["function"]["name"] for t in pt.inventory}
        sibs = pt.meta.get("sibling_names", [])
        assert len(sibs) == 3
        assert set(sibs) <= names
        assert pt.target_name in names
        assert pt.target_name not in sibs


def test_l9a_param_rename() -> None:
    pt = perturb(_math_task(), "L9A", inventory_size=12)
    assert pt.target_name == "calculate_area"
    params = next(
        t["function"]["parameters"]
        for t in pt.inventory
        if t["function"]["name"] == "calculate_area"
    )
    assert "width_value" in params["properties"] and "width" not in params["properties"]
    assert "width_value" in params["required"]
    assert pt.meta == {
        "schema_mode": "rename",
        "renamed_from": "width",
        "renamed_to": "width_value",
    }


def test_l9b_added_required_param() -> None:
    pt = perturb(_math_task(), "L9B", inventory_size=12)
    params = next(
        t["function"]["parameters"]
        for t in pt.inventory
        if t["function"]["name"] == "calculate_area"
    )
    assert "request_context" in params["properties"]
    assert "request_context" in params["required"]
    assert pt.meta.get("added_required") == "request_context"


def test_l9c_retype_numeric_to_string() -> None:
    pt = perturb(_math_task(), "L9C", inventory_size=12)
    params = next(
        t["function"]["parameters"]
        for t in pt.inventory
        if t["function"]["name"] == "calculate_area"
    )
    assert params["properties"]["width"]["type"] == "string"
    assert pt.meta.get("retyped") == "width" and pt.meta.get("from_type") == "integer"


_GOLD = [{"calculate_area": {"width": [3], "height": [5]}}]


def test_judge_l9a_scores_against_live_schema() -> None:
    pt = perturb(_math_task(), "L9A", inventory_size=12)
    # Model adapted to the drifted schema: new param name, correct values.
    adapted = classify(
        pt, [{"name": "calculate_area", "arguments": {"width_value": 3, "height": 5}}], "", _GOLD
    )
    assert adapted["selection_correct"] and adapted["schema_valid"] and not adapted["stale_args"]
    # Model used the stale (pre-drift) param name.
    stale = classify(
        pt, [{"name": "calculate_area", "arguments": {"width": 3, "height": 5}}], "", _GOLD
    )
    assert stale["stale_args"] and not stale["schema_valid"]


def test_judge_l9b_requires_new_param() -> None:
    pt = perturb(_math_task(), "L9B", inventory_size=12)
    missing = classify(
        pt, [{"name": "calculate_area", "arguments": {"width": 3, "height": 5}}], "", _GOLD
    )
    assert missing["stale_args"] and not missing["schema_valid"]
    present = classify(
        pt,
        [
            {
                "name": "calculate_area",
                "arguments": {"width": 3, "height": 5, "request_context": "x"},
            }
        ],
        "",
        _GOLD,
    )
    assert present["schema_valid"] and not present["stale_args"]
