# stale-tools

A controlled probe that injects identifier and description drift into existing
function-calling tasks at eight graded severity levels and scores the model's
structured tool call against a four-class behavioural codebook
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
| `breadth` | BFCL v4 sweep across all eight levels and the full SOTA model list |
| `effort` | Reasoning-effort sweep on five base models at four effort settings |
| `apibank` | APIBank replication at the four anchor levels |
| `size-ladder` | Within-architecture model-size comparison (Qwen 2.5, Llama 3.1) |
| `post-cutoff` | Held-out hand-built tasks naming post-cutoff tools |
| `apibank-pilot` | APIBank at the levels the main replication skips |

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

## Layout

```
src/stale_tools/
├── harness/
│   ├── perturbations.py    eight-level operator P_l on a tool inventory
│   ├── tasks.py            BFCL v4 task loader + stratified sampler
│   ├── apibank_tasks.py    APIBank task loader (HuggingFace cache)
│   ├── models.py           SOTA + reasoning-effort + size-ladder registries
│   ├── runner.py           async OpenRouter runner, resumable
│   ├── judge.py            programmatic four-class codebook scorer
│   └── settings.py         env-var configuration
└── cli.py                  ``stale-tools`` argparse entry point

data/                       BFCL v4 task pools and the post-cutoff tasks
results/                    bundled per-cell JSONL records
scripts/smoke_test_models.py  one-call sanity check across SOTA + effort variants
tests/                      smoke tests for the perturbation operator and registry
```

## Citation

Anonymous double-blind submission; citation metadata will be added after review.
