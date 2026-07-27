"""ToolScope-style retrieval filter over the 100-tool L8 manifests (rebuttal E6).

Reimplements the retrieval half of ToolScope (Liu et al., 2025; no public code — the
same-named GitHub repo is a different paper) per their reported best configuration:
dense retrieval with thenlper/gte-large embeddings (alpha=1.0, dense-only), then
cross-encoder/ms-marco-MiniLM-L6-v2 reranking of the top-50, keeping top-k.

Offline question answered here: does context-aware filtering keep the gold
(description-match) tool in the model's context at L8, and does it also keep the
directive-named tool? If both survive, curation cannot resolve the stale-directive
conflict — the routing decision still happens downstream. The online half (running
models on the filtered inventory) reuses the harness.

Usage:
  uv run --with sentence-transformers python scripts/toolscope_filter.py \
      [--inventory-size 100] [--k 5 10 20] [--seed 2026] [--out results/toolscope_retention.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer, util

from stale_tools.harness.perturbations import perturb
from stale_tools.harness.tasks import stratified_sample


def tool_text(t: dict) -> str:
    fn = t["function"]
    return f"{fn['name']}: {fn.get('description', '')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory-size", type=int, default=100)
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--rerank-pool", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path("results/toolscope_retention.json"))
    args = ap.parse_args()

    tasks = stratified_sample(90, 80, 30, seed=args.seed)
    embedder = SentenceTransformer("thenlper/gte-large")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

    emb_cache: dict[str, object] = {}

    def embed(texts: list[str]):
        missing = [t for t in texts if t not in emb_cache]
        if missing:
            vecs = embedder.encode(missing, normalize_embeddings=True, show_progress_bar=False)
            for t, v in zip(missing, vecs):
                emb_cache[t] = v
        return [emb_cache[t] for t in texts]

    ks = sorted(args.k)
    counts = {k: {"gold": 0, "directive": 0, "both": 0} for k in ks}
    n = 0
    for task in tasks:
        pt = perturb(task, "L8", inventory_size=args.inventory_size)
        texts = [tool_text(t) for t in pt.inventory]
        names = [t["function"]["name"] for t in pt.inventory]
        tool_vecs = embed(texts)
        q_vec = embedder.encode([pt.instruction], normalize_embeddings=True)[0]
        sims = [float(util.cos_sim(q_vec, v)) for v in tool_vecs]
        pool_idx = sorted(range(len(sims)), key=lambda i: -sims[i])[: args.rerank_pool]
        scores = reranker.predict([(pt.instruction, texts[i]) for i in pool_idx])
        ranked = [pool_idx[i] for i in sorted(range(len(pool_idx)), key=lambda i: -float(scores[i]))]
        n += 1
        for k in ks:
            top = {names[i] for i in ranked[:k]}
            g = pt.target_name in top
            d = pt.obsolete_name in top
            counts[k]["gold"] += g
            counts[k]["directive"] += d
            counts[k]["both"] += g and d
        if n % 25 == 0:
            print(f"  {n}/{len(tasks)}")

    report = {
        "config": {
            "inventory_size": args.inventory_size,
            "seed": args.seed,
            "embedder": "thenlper/gte-large",
            "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "rerank_pool": args.rerank_pool,
            "n_tasks": n,
        },
        "retention": {
            str(k): {m: round(c / n, 4) for m, c in counts[k].items()} for k in ks
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
