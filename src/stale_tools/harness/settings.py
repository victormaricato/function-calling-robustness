"""Runtime settings for the probe.

Reads from environment variables only. Set ``OPENROUTER_API_KEY`` (and optionally
``STALE_TOOLS_RESULTS_DIR``) before running any block.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it in your shell, e.g.\n"
            "  export OPENROUTER_API_KEY=sk-or-v1-..."
        )
    return key


def results_dir() -> Path:
    """Where the runner writes JSONL output files. Defaults to ``./results``."""
    return Path(os.environ.get("STALE_TOOLS_RESULTS_DIR", "results")).resolve()
