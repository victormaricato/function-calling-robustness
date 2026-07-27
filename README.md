# stale-tools

> Code and per-cell measurement records accompanying
> **"Function-Calling LLMs Under Stale Tool Documentation."**
> Anonymous submission to NeurIPS 2026, currently under double-blind review.

A controlled probe that injects identifier, description, deprecation, and
directive-override drift into existing function-calling tasks at nine graded
severity levels and scores the model's structured tool call against a
four-class behavioural codebook
(*correct*, *wrong-neighbour*, *hallucinated*, *abstain*).

The package ships the experiment runner, the per-cell records collected
across every block, and an audit script (`analysis/recompute.py`) that
reproduces the paper's headline numbers from the shipped records alone —
every perturbation is deterministic from `(task_id, seed)`, so each recorded
cell can be reconstructed exactly and re-judged without any API access.

## Requirements

* Python 3.10+ (3.12 recommended; pinned in `.python-version`)
* [uv](https://docs.astral.sh/uv/)
* An OpenRouter API key

## Setup

```bash
uv sync --dev
export OPENROUTER_API_KEY=sk-or-v1-...
```

## Run an experiment block

```bash
uv run stale-tools run breadth          # main BFCL v4 sweep
uv run stale-tools run --help           # all available blocks and flags
```

Available blocks (`uv run stale-tools run <block>`):

| Block | What it runs |
|---|---|
| `breadth` | BFCL v4 sweep across the nine $L_0$--$L_8$ levels and the full SOTA model list |
| `effort` | Reasoning-effort sweep on five base models at four effort settings |
| `apibank` | APIBank replication at the four anchor levels |
| `size-ladder` | Within-architecture model-size comparison (Qwen 3 dense, Llama 3.x) |
| `post-cutoff` | Held-out hand-built tasks naming post-cutoff tools |
| `apibank-pilot` | APIBank at the levels the main replication skips |
| `directive` | Directive-override block: $L_0$ vs $L_8$ on the BFCL breadth pool, full SOTA list |
| `l8-ablation` | Skill-prose ablations of $L_8$: no-skill ($L_{8N}$) and intent-only ($L_{8I}$) |
| `directive-strength` | $L_8$ skill-prose strength variants: weak mention, example-only, hard routing |
| `matched-desc` | $L_8$ matched-description counterfactual (removes the description-quality gap) |
| `lint-warn` | $L_8$ plus an automated skill-lint staleness warning in the prompt |
| `crowding` | Semantic crowding: near-duplicate siblings at $L_0$/$L_8$ |
| `schema-drift` | Parameter-schema evolution pilot: renamed / newly-required / retyped params |

Plus two sweep commands: `uv run stale-tools inventory-sensitivity --sizes 8 24`
(probe-model subset; writes `block_inv{size}_v2.jsonl` with per-cell
`actual_inventory_size` recorded) and `uv run stale-tools inventory-scale`
(12/50/100-tool manifests padded from the full BFCL function pool).

There is also an offline skill-lint evaluation that needs no API access:

```bash
uv run python scripts/skill_lint.py     # -> results/lint_report.json
```

## Audit the paper's numbers (no API access needed)

```bash
uv run python analysis/recompute.py
```

Rebuilds each recorded cell deterministically, re-applies the programmatic
judge to the recorded structured responses, and prints per-level selection
accuracy, directive-followed, and abstain rates for every BFCL-based block
present in `results/`.

The runner is **resumable**: rerunning skips cells that already completed
successfully and retries cells whose previous attempt failed (transient
provider 429s, 503s, timeouts). To force a full re-run, delete the relevant
`results/*.jsonl` first.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | *(required)* | OpenRouter API key |
| `STALE_TOOLS_RESULTS_DIR` | `./results` | Destination directory for JSONL output |
| `EXCLUDE_MODEL_SLUGS` | empty | Comma-separated slugs to skip on this run |
| `EXCLUDE_MODEL_NICKS` | empty | Comma-separated nicknames to skip on this run |

## APIBank data

The APIBank loader reads `level-1-api.json` from the local HuggingFace cache.
Pull the dataset once before running APIBank-related blocks:

```bash
uv pip install huggingface_hub
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('liminghao1630/API-Bank', repo_type='dataset')"
```

## Citation

While the paper is under double-blind review at NeurIPS 2026, please cite as:

```bibtex
@unpublished{anonymous2026staletools,
  title  = {Function-Calling {LLMs} Under Stale Tool Documentation},
  author = {Anonymous},
  year   = {2026},
  note   = {Submitted to NeurIPS 2026; under double-blind review.
            Code and per-cell records:
            \url{https://anonymous.4open.science/r/function-calling-robustness-EC4B/}}
}
```

Final author and venue metadata will be added once the review process closes.
