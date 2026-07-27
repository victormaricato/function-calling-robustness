"""OpenRouter model registry.

Three named lists:

  SOTA          — frontier models from multiple providers, used for the breadth sweep.
  EFFORT_SWEEP  — a subset of base models replicated across off/low/med/high
                  reasoning-effort settings on the same weights.
  SIZE_LADDER   — within-architecture pairs (Qwen 2.5, Llama 3.1) to isolate
                  the model-size factor from training and reasoning differences.

Reasoning is provider-specific:
  - OpenAI o-series and GPT-5.* take ``reasoning_effort`` ∈ {low, medium, high}.
  - Anthropic / Google / DeepSeek take a token budget via OpenRouter's
    normalised ``reasoning.max_tokens`` field.

Both forms are mapped onto OpenRouter's normalised ``reasoning`` payload.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    nickname: str
    slug: str
    is_reasoning: bool
    pair_id: str
    extra_body: dict | None = None  # OpenRouter `reasoning` / `provider` overrides
    effort: str | None = None  # 'off' | 'low' | 'med' | 'high' (for Block B)


# ─── Block A: SOTA breadth (replaces the v1 matched-pair list) ──────────────
SOTA: list[ModelSpec] = [
    # Anthropic — within-family scale ladder
    ModelSpec(
        "Opus-4.7",
        "anthropic/claude-opus-4.7",
        True,
        "anthropic",
        extra_body={"reasoning": {"max_tokens": 4000}},
    ),
    ModelSpec(
        "Sonnet-4.6",
        "anthropic/claude-sonnet-4.6",
        True,
        "anthropic",
        extra_body={"reasoning": {"max_tokens": 4000}},
    ),
    ModelSpec(
        "Haiku-4.5",
        "anthropic/claude-haiku-4.5",
        True,
        "anthropic",
        extra_body={"reasoning": {"max_tokens": 4000}},
    ),
    # OpenAI — current generation + the o-series reasoning anchor
    ModelSpec(
        "GPT-5.5", "openai/gpt-5.5", True, "openai", extra_body={"reasoning": {"effort": "medium"}}
    ),
    ModelSpec(
        "GPT-5.1", "openai/gpt-5.1", True, "openai", extra_body={"reasoning": {"effort": "medium"}}
    ),
    ModelSpec(
        "GPT-5", "openai/gpt-5", True, "openai", extra_body={"reasoning": {"effort": "medium"}}
    ),
    ModelSpec("o3", "openai/o3", True, "openai", extra_body={"reasoning": {"effort": "medium"}}),
    # Google — Pro vs Flash, both reasoning-capable
    ModelSpec(
        "Gemini-3.1-Pro",
        "google/gemini-3.1-pro-preview",
        True,
        "google",
        extra_body={"reasoning": {"max_tokens": 4000}},
    ),
    ModelSpec(
        "Gemini-3.1-Flash",
        "google/gemini-3.1-flash-lite-preview",
        True,
        "google",
        extra_body={"reasoning": {"max_tokens": 2000}},
    ),
    # DeepSeek — V4 Pro is the cheap-but-strong default; R1 is the dedicated reasoner
    ModelSpec(
        "DeepSeek-V4-Pro",
        "deepseek/deepseek-v4-pro",
        True,
        "deepseek",
        extra_body={"reasoning": {"max_tokens": 4000}},
    ),
    ModelSpec("DeepSeek-R1", "deepseek/deepseek-r1-0528", True, "deepseek"),
    # Moonshot — K2.6 is reasoning-only on OpenRouter
    ModelSpec("Kimi-K2.6", "moonshotai/kimi-k2.6", True, "moonshot"),
    # Meta — non-reasoning baseline (controls for "reasoning helps" vs "newer = better").
    # Llama 4 Maverick on OpenRouter has no tool-use endpoints; 3.3-70B is the current
    # Meta non-reasoning flagship that does.
    ModelSpec("Llama-3.3-70B", "meta-llama/llama-3.3-70b-instruct", False, "meta"),
    # Mistral — second non-reasoning anchor, MoE architecture
    ModelSpec("Mixtral-8x22B", "mistralai/mixtral-8x22b-instruct", False, "mistral"),
]


# ─── Rebuttal roster refresh (July 2026): newest generation per provider ────
# Additions only — SOTA above stays frozen so previously recorded blocks remain
# exactly reproducible; SOTA_V2 = SOTA + NEW_SOTA drives the refreshed sweeps.
NEW_SOTA: list[ModelSpec] = [
    ModelSpec(
        "GPT-5.6-Sol",
        "openai/gpt-5.6-sol",
        True,
        "openai",
        extra_body={"reasoning": {"effort": "medium"}},
    ),
    ModelSpec(
        "GPT-5.6-Luna",
        "openai/gpt-5.6-luna",
        True,
        "openai",
        extra_body={"reasoning": {"effort": "medium"}},
    ),
    ModelSpec(
        "Opus-5",
        "anthropic/claude-opus-5",
        True,
        "anthropic",
        extra_body={"reasoning": {"max_tokens": 4000}},
    ),
    ModelSpec(
        "Sonnet-5",
        "anthropic/claude-sonnet-5",
        True,
        "anthropic",
        extra_body={"reasoning": {"max_tokens": 4000}},
    ),
    ModelSpec(
        "Gemini-3.6-Flash",
        "google/gemini-3.6-flash",
        True,
        "google",
        extra_body={"reasoning": {"max_tokens": 2000}},
    ),
    ModelSpec("GLM-5.2", "z-ai/glm-5.2", True, "zai"),
    ModelSpec("Qwen3.6-35B-A3B", "qwen/qwen3.6-35b-a3b", True, "qwen"),
]

SOTA_V2: list[ModelSpec] = SOTA + NEW_SOTA


# ─── Block S (size ladder): isolate model-size effect from architecture/training ──
# Qwen 2.5 spans 7B and 72B at the same instruct post-training; Llama provides a
# parallel 8B vs 70B ladder so the size signal is replicated across two families.
# All four are non-reasoning so reasoning-vs-scale confounding is removed.
SIZE_LADDER: list[ModelSpec] = [
    # Llama 3.x ladder spans 1B → 70B across two sub-versions trained from
    # the same base lineage (3.2 1B/3B distilled from 3.1 teachers).
    ModelSpec("Llama-3.2-1B", "meta-llama/llama-3.2-1b-instruct", False, "llama-ladder"),
    ModelSpec("Llama-3.2-3B", "meta-llama/llama-3.2-3b-instruct", False, "llama-ladder"),
    ModelSpec("Llama-3.1-8B", "meta-llama/llama-3.1-8b-instruct", False, "llama-ladder"),
    ModelSpec("Llama-3.1-70B", "meta-llama/llama-3.1-70b-instruct", False, "llama-ladder"),
    # Qwen 3 dense ladder spans 8B → 32B at the same instruct post-training.
    ModelSpec("Qwen3-8B", "qwen/qwen3-8b", False, "qwen-ladder"),
    ModelSpec("Qwen3-14B", "qwen/qwen3-14b", False, "qwen-ladder"),
    ModelSpec("Qwen3-32B", "qwen/qwen3-32b", False, "qwen-ladder"),
]


# ─── Block B: reasoning-effort curve ────────────────────────────────────────
# 4 models × 4 efforts. Within-family Anthropic (Opus vs Sonnet) plus one
# OpenAI (effort enum) and one Google (budget tokens).

# Budget-token mapping for max_tokens-style providers (Anthropic, Google, DeepSeek)
_BUDGET_FOR_EFFORT = {
    "off": None,
    "low": 1024,
    "med": 4096,
    "high": 16384,
}

# OpenAI effort enum mapping
_OPENAI_EFFORT = {
    "off": None,
    "low": "low",
    "med": "medium",
    "high": "high",
}


def _budget_specs(base_nick: str, slug: str, pair_id: str) -> list[ModelSpec]:
    out = []
    for eff, budget in _BUDGET_FOR_EFFORT.items():
        nick = f"{base_nick}-{eff}"
        if budget is None:
            extra = None
            is_reasoning = False
        else:
            extra = {"reasoning": {"max_tokens": budget}}
            is_reasoning = True
        out.append(ModelSpec(nick, slug, is_reasoning, pair_id, extra_body=extra, effort=eff))
    return out


def _openai_effort_specs(base_nick: str, slug: str, pair_id: str) -> list[ModelSpec]:
    out = []
    for eff, val in _OPENAI_EFFORT.items():
        nick = f"{base_nick}-{eff}"
        if val is None:
            # GPT-5.1 with no reasoning param defaults to internal minimum;
            # we explicitly request 'minimal' to mark this as the off-effort baseline.
            extra = {"reasoning": {"effort": "minimal"}}
            is_reasoning = False
        else:
            extra = {"reasoning": {"effort": val}}
            is_reasoning = True
        out.append(ModelSpec(nick, slug, is_reasoning, pair_id, extra_body=extra, effort=eff))
    return out


EFFORT_SWEEP: list[ModelSpec] = (
    _budget_specs("Opus-4.7", "anthropic/claude-opus-4.7", "opus_eff")
    + _budget_specs("Sonnet-4.6", "anthropic/claude-sonnet-4.6", "sonnet_eff")
    + _openai_effort_specs("GPT-5.1", "openai/gpt-5.1", "gpt51_eff")
    + _openai_effort_specs("GPT-5.5", "openai/gpt-5.5", "gpt55_eff")
    + _budget_specs("Gemini-3.1-Pro", "google/gemini-3.1-pro-preview", "gemini_eff")
)


ALL_MODELS = SOTA + NEW_SOTA + EFFORT_SWEEP + SIZE_LADDER


def _pick(*nicks: str) -> list[ModelSpec]:
    return [by_nickname(n) for n in nicks]


def core_12() -> list[ModelSpec]:
    """Roster for the rebuttal experiment blocks: the 7 refresh models plus 5
    continuity anchors from the original SOTA list (incl. DeepSeek-R1, the sole
    L8 directive holdout)."""
    return list(NEW_SOTA) + _pick(
        "GPT-5", "Opus-4.7", "Gemini-3.1-Pro", "DeepSeek-V4-Pro", "DeepSeek-R1"
    )


def schema_pilot() -> list[ModelSpec]:
    """4-model pilot arm for the schema-drift block (E5)."""
    return _pick("GPT-5.6-Sol", "Sonnet-5", "DeepSeek-V4-Pro", "Qwen3.6-35B-A3B")


def scale_8() -> list[ModelSpec]:
    """8-model arm for the inventory-scale/crowding block (E4)."""
    return _pick(
        "GPT-5.6-Sol",
        "Sonnet-5",
        "Gemini-3.6-Flash",
        "GPT-5",
        "Opus-4.7",
        "DeepSeek-V4-Pro",
        "GLM-5.2",
        "Qwen3.6-35B-A3B",
    )


def by_nickname(nick: str) -> ModelSpec:
    for m in ALL_MODELS:
        if m.nickname == nick:
            return m
    raise KeyError(nick)
