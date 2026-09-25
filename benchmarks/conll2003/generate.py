"""Run one model over CoNLL-2003 and log one JSONL record per sentence.

This is step 0 of the replication proposed in paper Section 7 ("replicate the
method on a public NER benchmark such as CoNLL-2003 and on a different
small/large model family"). It follows the protocol of paper Section 6.1:
JSON entity extraction with identical prompts for both models, greedy
decoding (Appendix B.2: temperature 0, at most 256 new tokens), and per-query
logging of everything the analysis needs.

Each record holds the sentence id and source split, the model id, the raw
output, the parsed and gold entities, the exact-match event (e(x) = 1 minus
it, paper Section 4.2), entity-level tp/fp/fn for micro-F1, u(x) (paper
Eq. 4, over content tokens, end-of-sequence token excluded), the mean
full-vocabulary entropy and mean maximum probability (signals for the
baselines and ablations), the number of content tokens and an amortized
latency. A ``<out>.meta.json`` file records the dataset revision, model
revision, dtype, device, batch size, prompt hash and library versions.

The run is deterministic for a fixed configuration and resumable: records
already in ``--out`` are kept, and batches are formed from the full sentence
list the same way on every run, so a resumed run computes each remaining
batch exactly as an uninterrupted run would.

Example
-------
::

    python benchmarks/conll2003/generate.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --out benchmarks/conll2003/runs/full/small.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import conll  # noqa: E402
from backends import Generation, make_backend, now_utc  # noqa: E402

#: Settings that must match between a run and the run it resumes.
RESUME_KEYS = (
    "model",
    "revision",
    "backend",
    "dtype",
    "batch_size",
    "max_new_tokens",
    "dataset_id",
    "dataset_revision",
    "source_splits",
    "limit",
    "sample_seed",
    "prompt_version",
    "prompt_sha256",
)


def mean_or_none(xs: Sequence[float]) -> Optional[float]:
    """Arithmetic mean, None for an empty sequence."""
    return float(sum(xs) / len(xs)) if xs else None


def u_from_generation(g: Generation) -> Optional[float]:
    """u(x) of paper Eq. 4 via the package's own implementation, None for an empty generation."""
    if g.n_tokens == 0:
        return None
    from ucci import token_margin_uncertainty

    return float(token_margin_uncertainty(list(zip(g.p1, g.p2))))


def make_record(
    sent: conll.Sentence,
    g: Generation,
    model_id: str,
    latency_ms: float,
    batch_ms: float,
    batch_size: int,
) -> Dict[str, Any]:
    """The JSONL record for one sentence (see the module docstring)."""
    parse_ok, schema_ok, pred = conll.parse_entities(g.text)
    sc = conll.score_entities(pred, sent.gold, parse_ok)
    return {
        "id": sent.id,
        "source_split": sent.source_split,
        "model": model_id,
        "sentence": sent.text,
        "raw_output": g.text,
        "parse_ok": parse_ok,
        "schema_ok": schema_ok,
        "pred": pred,
        "gold": sent.gold,
        "exact_match": sc.exact_match,
        "tp": sc.tp,
        "fp": sc.fp,
        "fn": sc.fn,
        "per_type": sc.per_type,
        "u": u_from_generation(g),
        "entropy": mean_or_none(g.entropy),
        "max_prob": mean_or_none(g.p1),
        "n_tokens": g.n_tokens,
        "stop_reason": g.stop_reason,
        "argmax_mismatch": g.argmax_mismatch,
        "prompt_tokens": g.prompt_tokens,
        "latency_ms": latency_ms,
        "batch_ms": batch_ms,
        "batch_size": batch_size,
    }


def read_done_ids(path: Path) -> Set[str]:
    """Ids already logged in ``path``; a torn last line from a crash is truncated away."""
    done: Set[str] = set()
    if not path.exists():
        return done
    good_bytes = 0
    with path.open("rb") as fh:
        for raw in fh:
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                break
            if not raw.endswith(b"\n"):
                break
            done.add(str(rec["id"]))
            good_bytes += len(raw)
    if good_bytes != path.stat().st_size:
        with path.open("r+b") as fh:
            fh.truncate(good_bytes)
        print(f"[generate] truncated a partial trailing line in {path}", file=sys.stderr)
    return done


def check_resume(meta_path: Path, meta: Dict[str, Any], force: bool) -> Dict[str, Any]:
    """Load existing metadata and refuse to resume under different settings."""
    if not meta_path.exists():
        return {}
    old = json.loads(meta_path.read_text())
    diffs = {k: (old.get(k), meta.get(k)) for k in RESUME_KEYS if old.get(k) != meta.get(k)}
    if diffs and not force:
        lines = "\n".join(f"  {k}: logged {a!r}, now {b!r}" for k, (a, b) in diffs.items())
        raise SystemExit(
            f"refusing to resume {meta_path.name}: settings differ from the logged run\n{lines}\n"
            "use a new --out, or --force-resume to append anyway"
        )
    return old


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, help="Hugging Face model id or local path")
    p.add_argument("--revision", default=None, help="model commit (recommended: pin it)")
    p.add_argument("--out", required=True, type=Path, help="output JSONL path")
    p.add_argument("--backend", default="transformers", choices=["transformers", "vllm"])
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu (transformers backend)")
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=256, help="paper Appendix B.2 uses 256")
    p.add_argument(
        "--splits",
        nargs="+",
        default=["validation", "test"],
        help="CoNLL-2003 splits to pool (default: validation test)",
    )
    p.add_argument("--dataset-id", default=conll.DATASET_ID)
    p.add_argument("--dataset-revision", default=conll.DATASET_REVISION)
    p.add_argument("--limit", type=int, default=None, help="seeded subset of this many sentences")
    p.add_argument("--sample-seed", type=int, default=0, help="seed for --limit")
    p.add_argument(
        "--save-token-stats",
        action="store_true",
        help="also write per-token p1, p2, entropy to <out>.tokens.jsonl",
    )
    p.add_argument(
        "--force-resume", action="store_true", help="append even if the logged settings differ"
    )
    p.add_argument(
        "--attn-implementation",
        default="auto",
        help="transformers attention kernel; auto = eager on MPS (see backends.py), default elsewhere",
    )
    p.add_argument(
        "--adapter-check-batches",
        type=int,
        default=1,
        help="recompute the signals of this many batches with the package adapter "
        "(ucci.integrations.transformers or .vllm) and record the largest differences "
        "in the metadata (0 disables)",
    )
    args = p.parse_args(argv)
    if args.batch_size < 1:
        p.error("--batch-size must be at least 1")
    if args.max_new_tokens < 1:
        p.error("--max-new-tokens must be at least 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    tok_path = out.with_suffix(".tokens.jsonl")

    pool = conll.load_pool(args.splits, args.dataset_id, args.dataset_revision)
    sents = conll.sample_pool(pool, args.limit, args.sample_seed)
    print(
        f"[generate] {len(sents)} sentences from {args.dataset_id}@{args.dataset_revision[:12]} "
        f"splits={args.splits}",
        file=sys.stderr,
    )

    t_load = time.perf_counter()
    backend = make_backend(
        args.backend,
        args.model,
        args.revision,
        args.device,
        args.dtype,
        args.max_new_tokens,
        args.attn_implementation,
    )
    load_s = time.perf_counter() - t_load
    desc = backend.describe()

    prompts = [backend.render(conll.build_messages(s.text)) for s in sents]
    meta: Dict[str, Any] = {
        "format": "ucci-conll2003-generation",
        "version": 1,
        "model": args.model,
        "revision": args.revision,
        "backend": args.backend,
        "dtype": desc["dtype"],
        "device": desc["device"],
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "source_splits": list(args.splits),
        "limit": args.limit,
        "sample_seed": args.sample_seed,
        "n_sentences": len(sents),
        "prompt_version": conll.PROMPT_VERSION,
        "prompt_sha256": conll.prompt_sha256(),
        "prompt_template": conll.PROMPT_TEMPLATE,
        "rendered_prompt_example": prompts[0] if prompts else None,
        "u_token_convention": "content tokens only; stop token and padding excluded",
        "backend_info": desc,
    }
    old = check_resume(meta_path, meta, args.force_resume)
    sessions: List[Dict[str, Any]] = list(old.get("sessions", []))
    session: Dict[str, Any] = {"started_at": now_utc(), "model_load_s": round(load_s, 3)}
    sessions.append(session)
    meta["sessions"] = sessions
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    done = read_done_ids(out)
    lengths = [backend.prompt_length(p) for p in prompts]
    # Longest prompts first (padding waste is lowest and memory peaks early); ties by pool order.
    order = sorted(range(len(sents)), key=lambda i: (-lengths[i], i))
    batches = [order[i : i + args.batch_size] for i in range(0, len(order), args.batch_size)]
    todo = [b for b in batches if any(sents[i].id not in done for i in b)]
    print(
        f"[generate] {len(done)} already logged, {len(todo)}/{len(batches)} batches to run "
        f"(model load {load_s:.1f}s, device={desc['device']}, dtype={desc['dtype']})",
        file=sys.stderr,
    )

    n_new = 0
    mismatches = 0
    checks: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    with (
        out.open("a", encoding="utf-8") as fh,
        (
            tok_path.open("a", encoding="utf-8") if args.save_token_stats else open(os.devnull, "w")
        ) as tfh,
    ):
        for bi, batch in enumerate(todo):
            t0 = time.perf_counter()
            check = bi < args.adapter_check_batches
            gens = backend.generate([prompts[i] for i in batch], cross_check=check)
            backend.synchronize()
            if check and getattr(backend, "last_cross_check", None):
                checks.append(backend.last_cross_check)
            batch_ms = (time.perf_counter() - t0) * 1000.0
            per_ms = batch_ms / len(batch)
            for i, g in zip(batch, gens):
                sid = sents[i].id
                if sid in done:
                    continue
                if args.save_token_stats:
                    tfh.write(
                        json.dumps({"id": sid, "p1": g.p1, "p2": g.p2, "entropy": g.entropy}) + "\n"
                    )
                rec = make_record(sents[i], g, args.model, per_ms, batch_ms, len(batch))
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done.add(sid)
                n_new += 1
                mismatches += g.argmax_mismatch
            fh.flush()
            tfh.flush()
            el = time.perf_counter() - t_start
            rate = n_new / el if el > 0 else 0.0
            left = sum(1 for b in todo[bi + 1 :] for i in b if sents[i].id not in done)
            eta = left / rate if rate > 0 else float("nan")
            print(
                f"[generate] batch {bi + 1}/{len(todo)}  +{len(batch)}  total {len(done)}/{len(sents)}  "
                f"{rate:.2f} sent/s  eta {eta / 60:.1f} min",
                file=sys.stderr,
                flush=True,
            )

    elapsed = time.perf_counter() - t_start
    session.update(
        {
            "finished_at": now_utc(),
            "n_generated": n_new,
            "elapsed_s": round(elapsed, 3),
            "sentences_per_s": round(n_new / elapsed, 4) if elapsed > 0 else None,
            "argmax_mismatch_tokens": mismatches,
        }
    )
    if checks:
        session["adapter_cross_check"] = {
            "batches": len(checks),
            "tokens": sum(c["tokens"] for c in checks),
            "max_abs_diff_u": max(c["max_abs_diff_u"] for c in checks),
            "max_abs_diff_mean_entropy": max(c["max_abs_diff_mean_entropy"] for c in checks),
            "max_abs_diff_mean_max_prob": max(c["max_abs_diff_mean_max_prob"] for c in checks),
        }
        if session["adapter_cross_check"]["max_abs_diff_u"] > 1e-4:
            print(
                "[generate] WARNING: u(x) differs from the ucci.integrations adapter by more than 1e-4",
                file=sys.stderr,
            )
    meta["n_logged"] = len(done)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    backend.close()
    if mismatches:
        print(
            f"[generate] WARNING: {mismatches} content tokens were not the arg-max of the logits; "
            "decoding was not plain greedy",
            file=sys.stderr,
        )
    print(f"[generate] wrote {n_new} records to {out} in {elapsed:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
