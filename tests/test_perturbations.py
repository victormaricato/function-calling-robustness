"""Smoke tests for the perturbation operator and the model registry."""

from stale_tools.harness.models import ALL_MODELS, EFFORT_SWEEP, SIZE_LADDER, SOTA, by_nickname
from stale_tools.harness.perturbations import LEVELS, perturb


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
