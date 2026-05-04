# stale-tools

> Code and per-cell measurement records accompanying
> **"Function-Calling LLMs Under Stale Tool Documentation."**
> Anonymous submission to NeurIPS 2026, currently under double-blind review.

A controlled probe that injects identifier, description, deprecation, and
directive-override drift into existing function-calling tasks at nine graded
severity levels and scores the model's structured tool call against a
four-class behavioural codebook
(*correct*, *wrong-neighbour*, *hallucinated*, *abstain*).

The package ships the experiment runner and the per-cell records collected
across every block; analysis and figure generation are deliberately left out
of scope.

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
| `breadth` | BFCL v4 sweep across all eight $L_0$--$L_7$ levels and the full SOTA model list |
| `effort` | Reasoning-effort sweep on five base models at four effort settings |
| `apibank` | APIBank replication at the four anchor levels |
| `size-ladder` | Within-architecture model-size comparison (Qwen 2.5, Llama 3.1) |
| `post-cutoff` | Held-out hand-built tasks naming post-cutoff tools |
| `apibank-pilot` | APIBank at the levels the main replication skips |
| `directive` | Directive-override block: $L_0$ vs $L_8$ on the BFCL breadth pool, full SOTA list |

Plus a separate `uv run stale-tools inventory-sensitivity --sizes 8 24`
that sweeps two inventory sizes on a probe-model subset.

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
