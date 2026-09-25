"""Measure end-to-end latency per query for the small and large model.

Sets the cost ratio c_l / c_s of the replication the way paper Section 6.1
and Appendix B.3 set theirs: mean end-to-end latency (prompt processing plus
generation) over 100 queries, one query at a time (batch size 1), for each
model on the same machine. Costs are then normalized to c_s = 1 and
c_l = mean_large / mean_small.

Each query is timed from the rendered prompt to the decoded answer:
tokenization, greedy generation (no per-token statistics, so only the
serving work is timed) and detokenization, with a device synchronization
before the clock stops. No key/value or prefix cache is reused between
queries (the paper's "cold cache"). A few untimed warm-up queries, on
sentences outside the timed set, run first so one-time kernel compilation
is not charged to the first timed query; ``--warmup 0`` disables them.
Timing uses the default attention kernel of transformers (normally PyTorch's
fused ``sdpa``): at batch size 1 there is no padding, so the MPS padding
problem that makes ``generate.py`` use eager attention does not arise.

The same seeded sample of sentences is timed for both models. The output
JSON keeps every per-query latency so the summary can be recomputed.

Example
-------
::

    python benchmarks/conll2003/latency.py \\
        --small Qwen/Qwen2.5-1.5B-Instruct --large Qwen/Qwen2.5-7B-Instruct \\
        --out benchmarks/conll2003/runs/full/latency.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import conll  # noqa: E402
from backends import make_backend, now_utc  # noqa: E402


def pick_queries(
    pool: Sequence[conll.Sentence], n_queries: int, n_warmup: int, seed: int
) -> Dict[str, List[conll.Sentence]]:
    """Seeded, disjoint warm-up and timed samples from the pool.

    Raises
    ------
    ValueError
        If the pool is smaller than ``n_queries + n_warmup``.
    """
    need = n_queries + n_warmup
    if need > len(pool):
        raise ValueError(
            f"need {need} sentences for {n_warmup} warm-up + {n_queries} timed queries, pool has {len(pool)}"
        )
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    return {
        "warmup": [pool[i] for i in order[:n_warmup]],
        "timed": [pool[i] for i in order[n_warmup:need]],
    }


def summarize(ms: Sequence[float]) -> Dict[str, float]:
    """Mean, sample standard deviation, median, min and max of per-query latencies (ms)."""
    return {
        "mean_ms": float(statistics.fmean(ms)),
        "std_ms": float(statistics.stdev(ms)) if len(ms) > 1 else 0.0,
        "median_ms": float(statistics.median(ms)),
        "min_ms": float(min(ms)),
        "max_ms": float(max(ms)),
    }


def time_model(
    args: argparse.Namespace,
    model_id: str,
    revision: Optional[str],
    queries: Dict[str, List[conll.Sentence]],
) -> Dict[str, Any]:
    """Load one model, run the warm-up, then time each query at batch size 1."""
    backend = make_backend(
        args.backend,
        model_id,
        revision,
        args.device,
        args.dtype,
        args.max_new_tokens,
        args.attn_implementation,
    )
    try:
        for s in queries["warmup"]:
            backend.generate([backend.render(conll.build_messages(s.text))], collect_stats=False)
        backend.synchronize()
        per_query: List[Dict[str, Any]] = []
        for s in queries["timed"]:
            messages = conll.build_messages(s.text)
            t0 = time.perf_counter()
            g = backend.generate([backend.render(messages)], collect_stats=False)[0]
            backend.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0
            per_query.append(
                {
                    "id": s.id,
                    "latency_ms": ms,
                    "n_tokens": g.n_tokens,
                    "prompt_tokens": g.prompt_tokens,
                    "stop_reason": g.stop_reason,
                }
            )
            print(
                f"[latency] {model_id} {len(per_query)}/{len(queries['timed'])} {ms:.1f} ms",
                file=sys.stderr,
                flush=True,
            )
        lat = [q["latency_ms"] for q in per_query]
        return {
            "model": model_id,
            "revision": revision,
            "backend_info": backend.describe(),
            **summarize(lat),
            "mean_new_tokens": float(statistics.fmean(q["n_tokens"] for q in per_query)),
            "per_query": per_query,
        }
    finally:
        backend.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--small", required=True, help="small model id")
    p.add_argument("--small-revision", default=None)
    p.add_argument("--large", required=True, help="large model id")
    p.add_argument("--large-revision", default=None)
    p.add_argument("--out", required=True, type=Path, help="output JSON path")
    p.add_argument(
        "--n-queries", type=int, default=100, help="timed queries per model (paper: 100)"
    )
    p.add_argument("--warmup", type=int, default=3, help="untimed warm-up queries per model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default="transformers", choices=["transformers", "vllm"])
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--attn-implementation",
        default="default",
        help="attention kernel for timing; 'default' is the transformers default (normally sdpa), "
        "safe here because a batch of one has no padding",
    )
    p.add_argument("--splits", nargs="+", default=["validation", "test"])
    p.add_argument("--dataset-id", default=conll.DATASET_ID)
    p.add_argument("--dataset-revision", default=conll.DATASET_REVISION)
    args = p.parse_args(argv)
    if args.n_queries < 2:
        p.error("--n-queries must be at least 2")
    if args.warmup < 0:
        p.error("--warmup must be non-negative")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    pool = conll.load_pool(args.splits, args.dataset_id, args.dataset_revision)
    queries = pick_queries(pool, args.n_queries, args.warmup, args.seed)
    started = now_utc()
    small = time_model(args, args.small, args.small_revision, queries)
    large = time_model(args, args.large, args.large_revision, queries)
    result = {
        "format": "ucci-latency",
        "version": 1,
        "protocol": (
            "end-to-end latency per query (tokenize, greedy generate, detokenize), batch size 1, "
            "no cache reuse between queries, device synchronized before timing stops; "
            "paper Section 6.1 and Appendix B.3"
        ),
        "started_at": started,
        "finished_at": now_utc(),
        "n_queries": args.n_queries,
        "warmup": args.warmup,
        "seed": args.seed,
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "source_splits": list(args.splits),
        "small": small,
        "large": large,
        "c_small": 1.0,
        "c_large": large["mean_ms"] / small["mean_ms"],
        "cost_ratio": large["mean_ms"] / small["mean_ms"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"[latency] small {small['mean_ms']:.1f} ms, large {large['mean_ms']:.1f} ms, "
        f"ratio c_l/c_s = {result['cost_ratio']:.3f} -> {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
