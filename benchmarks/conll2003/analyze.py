"""Analyze a small/large CoNLL-2003 run and write results.json, results.md and figures.

Joins the two generation logs written by ``generate.py``, splits the pooled
sentences 30 / 20 / 50 into calibration, validation and test sets with a
fixed seed, and runs the evaluation protocol of paper Section 6.1 for UCCI
and the paper's baselines (see ``analysis.py``). Costs are normalized to the
small model: c_s = 1 and c_l = the latency ratio measured by ``latency.py``
(paper Section 6.1 and Appendix B.3).

Outputs, in ``--out-dir``:

* ``results.json``: every number, with its configuration and input hashes;
* ``results.md``: the same numbers as Markdown tables with 95% bootstrap
  intervals (1000 resamples over test sentences by default);
* ``reliability_test.png`` and ``reliability_cal.png``: reliability
  diagrams of raw u(x) and the isotonic p_hat (paper Figure 1);
* ``pareto_test.png``: cost against micro-F1 on the test split (paper
  Figure 2);
* ``joined.jsonl``: one record per sentence in the package's traffic
  format, with the split, so ``ucci fit`` / ``ucci evaluate`` can be run on
  it directly.

Example
-------
::

    python benchmarks/conll2003/analyze.py \\
        --small benchmarks/conll2003/runs/full/small.jsonl \\
        --large benchmarks/conll2003/runs/full/large.jsonl \\
        --latency benchmarks/conll2003/runs/full/latency.json \\
        --out-dir benchmarks/conll2003/runs/full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import analysis  # noqa: E402

import ucci  # noqa: E402


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def display_path(path: Path, out_dir: Path) -> str:
    """``path`` relative to ``out_dir`` (so results never embed a home directory)."""
    try:
        return os.path.relpath(Path(path).resolve(), Path(out_dir).resolve())
    except ValueError:  # different drives on Windows
        return Path(path).name


def _f(x: Optional[float], digits: int = 3, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def _ci(ci: Optional[Sequence[float]], digits: int = 3, signed: bool = False) -> str:
    if not ci:
        return ""
    return f" [{_f(ci[0], digits, signed)}, {_f(ci[1], digits, signed)}]"


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100.0 * x:.1f}%"


def method_table(
    rows: Sequence[Mapping[str, Any]], with_target: bool, groups: Sequence[str]
) -> List[str]:
    """Markdown rows: method, cost, micro-F1, escalation rate, delta vs target, saving (with CIs)."""
    head = "| Method | Cost [95% CI] | Micro-F1 [95% CI] | Escalated [95% CI] |"
    sep = "|---|---:|---:|---:|"
    if with_target:
        head += " Delta F1 vs target [95% CI] |"
        sep += "---:|"
    head += " Saving vs large-only |"
    sep += "---:|"
    out = [head, sep]
    for r in rows:
        if r["group"] not in groups:
            continue
        name = r["method"]
        if "cost" not in r:
            if r.get("status") == "failed":
                first = "not run: " + str(r.get("error", "")).split(":")[0]
            else:
                first = "infeasible on validation"
            cells = [first, "", ""] + ([""] if with_target else []) + [""]
            out.append(f"| {name} | " + " | ".join(cells) + " |")
            continue
        ci = r.get("ci95", {})
        line = (
            f"| {name} | {_f(r['cost'], 2)}{_ci(ci.get('cost'), 2)} "
            f"| {_f(r['micro_f1'])}{_ci(ci.get('micro_f1'))} "
            f"| {_pct(r['escalation_rate'])}{_ci([100 * v for v in ci['escalation_rate']], 1) if ci.get('escalation_rate') else ''} |"
        )
        if with_target:
            line += f" {_f(r.get('delta_vs_target'), 3, True)}{_ci(ci.get('delta_vs_target'), 3, True)} |"
        line += f" {_pct(r.get('saving_vs_large'))} |"
        out.append(line)
    return out


def paired_table(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """Markdown rows of paired bootstrap differences against UCCI."""
    out = [
        "| Method | Cost minus UCCI [95% CI] | Micro-F1 minus UCCI [95% CI] |",
        "|---|---:|---:|",
    ]
    for r in rows:
        p = r.get("paired_vs_UCCI")
        if not p:
            continue
        out.append(
            f"| {r['method']} | {_f(p['cost_diff'], 3, True)}{_ci(p['cost_diff_ci'], 3, True)} "
            f"| {_f(p['f1_diff'], 3, True)}{_ci(p['f1_diff_ci'], 3, True)} |"
        )
    return out


def _oracle_line(o: Optional[Mapping[str, Any]]) -> List[str]:
    """One sentence on the label-dependent oracle (extension, analysis only)."""
    if not o:
        return []
    if "cost" not in o:
        return ["Oracle (extension, analysis only): infeasible on the test split.", ""]
    return [
        f"Oracle (extension, analysis only; {o['note']}): cost {_f(o['cost'], 2)}, "
        f"micro-F1 {_f(o['micro_f1'])}, escalated {_pct(o['escalation_rate'])}.",
        "",
    ]


def render_markdown(res: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    """The results as Markdown. Every number is read from ``res``."""
    L: List[str] = []
    cost = res["costs"]
    tgt = res["target"]
    sm = res["single_model_test"]
    L += [
        "# UCCI on CoNLL-2003: results",
        "",
        "Produced by `benchmarks/conll2003/analyze.py` from the generation logs listed below. "
        "Do not edit by hand; re-run the script.",
        "",
    ]
    L += [
        "## Setup",
        "",
        f"- Small model: `{meta['small_model']}`; large model: `{meta['large_model']}`.",
        f"- Sentences: {res['n_sentences']} (split {res['split']['sizes']['cal']} / "
        f"{res['split']['sizes']['val']} / {res['split']['sizes']['test']} calibration / validation / test, seed {res['split']['seed']}).",
        f"- Costs: c_s = {_f(cost['c_small'], 2)}, c_l = {_f(cost['c_large'], 3)} ({meta['cost_source']}).",
        f"- F1 target tau = {_f(tgt['tau'], 4)} ({tgt['rule']}; validation small-only {_f(tgt['val_small_f1'], 4)}, "
        f"large-only {_f(tgt['val_large_f1'], 4)}).",
        f"- Matched cost budget = {_f(res['budget']['budget'], 3)} ({res['budget']['rule']}).",
        f"- Intervals: {res['bootstrap']['method']}, {res['bootstrap']['n_boot']} resamples, 95%.",
        "",
    ]
    L += [
        "## Single models on the test split",
        "",
        "| Model | Micro-F1 | Exact match | Cost |",
        "|---|---:|---:|---:|",
        f"| Small | {_f(sm['small']['micro_f1'])} | {_pct(sm['small']['exact_match_rate'])} | {_f(sm['small']['cost'], 2)} |",
        f"| Large | {_f(sm['large']['micro_f1'])} | {_pct(sm['large']['exact_match_rate'])} | {_f(sm['large']['cost'], 2)} |",
        "",
    ]
    L += [
        "## Routing at the F1 target (Table 2, top block)",
        "",
        "Each method selects its operating point on the validation split (cheapest with validation "
        "micro-F1 >= tau) and routes every test sentence end to end.",
        "",
    ]
    L += method_table(res["at_target"], True, ("table2",)) + [""]
    L += _oracle_line(res.get("oracle_at_target"))
    L += (
        ["Paired bootstrap differences against UCCI (same resamples):", ""]
        + paired_table([r for r in res["at_target"] if r["group"] == "table2"])
        + [""]
    )
    L += [
        "## Routing at the matched cost budget (Table 2, bottom block)",
        "",
        "Each method selects the operating point with the highest validation micro-F1 whose "
        "validation cost is within the budget.",
        "",
    ]
    L += method_table(res["at_budget"], True, ("table2",)) + [""]
    L += _oracle_line(res.get("oracle_at_budget"))
    L += ["## Ablations and extensions at the F1 target (Section 6.3, Appendix B.4)", ""]
    L += method_table(res["at_target"], True, ("ablation", "extension")) + [""]
    cal = res["calibration"]
    L += [
        "## Calibration of the error forecast (Figure 1, Section 6.2)",
        "",
        "ECE of each forecast of e(x) (small model wrong, exact match). The calibration-split numbers "
        "are in-sample for the fitted maps (the paper reports its Figure 1 on the calibration set); "
        "the test-split numbers are out of sample.",
        "",
        "| Split | Forecast | ECE, 10 equal-width bins [95% CI] | ECE, deciles [95% CI] | Brier |",
        "|---|---|---:|---:|---:|",
    ]
    for s in ("cal", "test"):
        for key, label in (
            ("raw_u", "raw u(x)"),
            ("temperature_scaling", "temperature scaling"),
            ("isotonic", "isotonic (UCCI)"),
        ):
            if key not in cal[s]:
                continue
            b = cal[s][key]
            L.append(
                f"| {s} | {label} | {_f(b['ece_uniform'])}{_ci(b['ece_uniform_ci95'])} "
                f"| {_f(b['ece_quantile'])}{_ci(b['ece_quantile_ci95'])} | {_f(b['brier'])} |"
            )
    L += [
        "",
        f"Isotonic map: {cal['n_knots']} knots. Fitted temperature: {_f(cal.get('temperature'), 4)}.",
        "",
    ]
    a = res.get("assumption_ii") or {}
    if a:
        L += [
            "## Theorem 1, assumption (ii) (Section 6.3)",
            "",
            "At the UCCI operating point on the test split:",
            "",
            "| Quantity | Micro-F1 |",
            "|---|---:|",
            f"| Large model on escalated sentences (n = {a['n_escalated']}) | {_f(a.get('large_f1_escalated'))} |",
            f"| Large model on all test sentences | {_f(a.get('large_f1_all_test'))} |",
            f"| Small model on escalated sentences | {_f(a.get('small_f1_escalated'))} |",
            f"| Small model on kept sentences | {_f(a.get('small_f1_kept'))} |",
            "",
            f"Gap (all minus escalated, large model): {_f(a.get('gap'), 3, True)}.",
            "",
        ]
    pe = res["per_entity_test"]
    cols = [k for k in ("small", "large", "ucci_routed") if k in pe]
    L += [
        "## Per-entity micro-F1 on the test split (Table 4 analogue)",
        "",
        "| Entity type | " + " | ".join(c.replace("_", " ") for c in cols) + " |",
        "|---|" + "---:|" * len(cols),
    ]
    for t in pe["small"]:
        L.append(f"| {t} | " + " | ".join(_f(pe[c][t]) for c in cols) + " |")
    L += [""]
    if res.get("cost_ratio_sensitivity"):
        L += [
            "## Cost-ratio sensitivity of the UCCI routing (Table 3 analogue)",
            "",
            "The same test routing re-costed at other large/small cost ratios. The first row is the ratio "
            "measured on this machine; 3.02 is the paper's measured H100 ratio and 5 and 10 are the paper's "
            "hypothetical ratios.",
            "",
            "| c_l / c_s | Source | UCCI cost | Saving vs large-only | theta re-selected at this ratio is unchanged |",
            "|---:|---|---:|---:|---|",
        ]
        for r in res["cost_ratio_sensitivity"]:
            src = "measured" if r.get("measured") else "hypothetical"
            L.append(
                f"| {_f(r['cost_ratio'], 2)} | {src} | {_f(r['ucci_cost'], 2)} | {_pct(r['saving_vs_large'])} "
                f"| {'yes' if r['same_theta'] else 'no'} |"
            )
        L += [""]
    L += [
        "## Per-split summary statistics",
        "",
        "| Split | n | Words (mean / median) | With >= 1 entity | Entities (mean / max) | Small F1 | Small EM | Large F1 | Large EM | Mean u |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in ("cal", "val", "test", "all"):
        p = res["per_split_summary"][s]
        L.append(
            f"| {s} | {p['n']} | {_f(p['words_mean'], 1)} / {_f(p['words_median'], 0)} | {_pct(p['frac_with_entity'])} "
            f"| {_f(p['entities_mean'], 2)} / {p['entities_max']} | {_f(p['small']['micro_f1'])} | {_pct(p['small']['exact_match_rate'])} "
            f"| {_f(p['large']['micro_f1'])} | {_pct(p['large']['exact_match_rate'])} | {_f(p['u_mean'])} |"
        )
    chk = res["checks"]
    L += [
        "",
        "## Checks",
        "",
        f"- Logged scores that differ from re-scoring the raw outputs: small {chk['rescore_mismatches']['small']}, "
        f"large {chk['rescore_mismatches']['large']}.",
        f"- Empty small-model generations (u undefined, routed as u = 1): {chk['empty_small_generations']}.",
        f"- Generated tokens that were not the arg-max of the logits (0 means plain greedy decoding): "
        f"small {chk['argmax_mismatch_tokens'].get('small', 0)}, large {chk['argmax_mismatch_tokens'].get('large', 0)}.",
        f"- Outputs that hit the 256-token limit: small {_pct(res['per_split_summary']['all']['small']['hit_max_new_tokens_rate'])}, "
        f"large {_pct(res['per_split_summary']['all']['large']['hit_max_new_tokens_rate'])}.",
        "- Agreement with `ucci.baselines.compare_routers` on the Table 2 methods (max absolute difference): "
        f"cost {_f(max((x['cost_abs_diff'] for x in chk['compare_routers_agreement']), default=0.0), 2)}, "
        f"F1 {_f(max((x['f1_abs_diff'] for x in chk['compare_routers_agreement']), default=0.0), 2)}.",
    ]
    failed = [
        (blk, r)
        for blk in ("at_target", "at_budget")
        for r in res[blk]
        if r.get("status") == "failed"
    ]
    for blk, r in failed:
        L.append(
            f"- {r['method']} ({blk.replace('_', ' ')}) raised an error and is reported as not run: {r['error']}"
        )
    if res["calibration"].get("temperature_error"):
        L.append(
            f"- Temperature scaling for the calibration table failed: {res['calibration']['temperature_error']}"
        )
    L += [""]
    L += ["## Inputs", "", "Paths are relative to this file.", ""]
    for k, v in meta["inputs"].items():
        L.append(f"- `{k}`: `{v['path']}` (sha256 `{v['sha256'][:16]}...`)")
    L.append("")
    return "\n".join(L)


def make_figures(res: Mapping[str, Any], out_dir: Path, meta: Mapping[str, Any]) -> List[str]:
    """Write the reliability diagrams and the Pareto figure; returns the file names."""
    try:
        from ucci import plotting
    except ImportError as exc:  # pragma: no cover - matplotlib is optional
        print(f"[analyze] skipping figures: {exc}", file=sys.stderr)
        return []
    written: List[str] = []
    figs = res["_figures"]
    for s, label in (("test", "test split"), ("cal", "calibration split")):
        d = figs["reliability"][s]
        fig = plotting.reliability_diagram(
            {"raw u(x)": d["u"], "isotonic p_hat (UCCI)": d["p_hat"]},
            d["e"],
            title=f"CoNLL-2003, {label}: P(small model wrong)",
        )
        name = f"reliability_{s}.png"
        fig.savefig(out_dir / name, dpi=200)
        written.append(name)
    curves = figs["curves"]
    points = dict(figs["points"])
    fig = plotting.pareto_plot(
        {
            "UCCI (isotonic p_hat)": curves["UCCI (isotonic p_hat, 0.005 grid)"],
            "Raw u threshold": curves["Raw u"],
            "Entropy threshold": curves["Entropy"],
        },
        {
            k: v
            for k, v in points.items()
            if k in ("UCCI", "Small-only", "Large-only", "Oracle (analysis only)")
        },
        target=figs["tau"],
        title=f"CoNLL-2003 test split: {meta['small_short']} to {meta['large_short']}",
        ylabel="Micro-F1",
    )
    fig.savefig(out_dir / "pareto_test.png", dpi=200)
    written.append("pareto_test.png")
    return written


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--small", required=True, type=Path, help="small-model JSONL log")
    p.add_argument("--large", required=True, type=Path, help="large-model JSONL log")
    p.add_argument("--latency", type=Path, default=None, help="latency.json from latency.py")
    p.add_argument("--cost-ratio", type=float, default=None, help="c_l / c_s; overrides --latency")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0, help="split seed")
    p.add_argument("--cal-frac", type=float, default=0.3)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument(
        "--target-frac",
        type=float,
        default=0.75,
        help="tau = small F1 + frac x (large F1 - small F1) on validation",
    )
    p.add_argument(
        "--target-f1", type=float, default=None, help="fixed tau; overrides --target-frac"
    )
    p.add_argument(
        "--budget-frac", type=float, default=0.5, help="budget = c_s + frac x (c_l - c_s)"
    )
    p.add_argument(
        "--budget", type=float, default=None, help="fixed cost budget; overrides --budget-frac"
    )
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--no-ablations", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)
    if args.cost_ratio is None and args.latency is None:
        p.error("pass --latency (from latency.py) or --cost-ratio")
    if args.cost_ratio is not None and not args.cost_ratio > 1.0:
        p.error("--cost-ratio must be above 1 (Theorem 1, assumption (i))")
    if args.n_boot < 1:
        p.error("--n-boot must be at least 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    small = analysis.read_jsonl(args.small)
    large = analysis.read_jsonl(args.large)
    inputs: Dict[str, Dict[str, str]] = {
        "small": {
            "path": display_path(args.small, args.out_dir),
            "sha256": sha256_file(args.small),
        },
        "large": {
            "path": display_path(args.large, args.out_dir),
            "sha256": sha256_file(args.large),
        },
    }
    if args.cost_ratio is not None:
        c_small, c_large = 1.0, float(args.cost_ratio)
        cost_source = "--cost-ratio"
    else:
        lat = json.loads(Path(args.latency).read_text())
        c_small, c_large = analysis.cost_from_latency(lat)
        cost_source = (
            f"measured by latency.py: {lat['small']['mean_ms']:.1f} ms vs {lat['large']['mean_ms']:.1f} ms "
            f"mean over {lat.get('n_queries', '?')} queries, batch size 1"
        )
        inputs["latency"] = {
            "path": display_path(args.latency, args.out_dir),
            "sha256": sha256_file(args.latency),
        }

    cfg = analysis.AnalysisConfig(
        cal_frac=args.cal_frac,
        val_frac=args.val_frac,
        seed=args.seed,
        target_frac=args.target_frac,
        target_f1=args.target_f1,
        budget_frac=args.budget_frac,
        budget=args.budget,
        n_boot=args.n_boot,
        boot_seed=args.boot_seed,
        include_ablations=not args.no_ablations,
    )
    res = analysis.analyze(small, large, c_small, c_large, cfg)
    small_model = str(small[0].get("model", "small")) if small else "small"
    large_model = str(large[0].get("model", "large")) if large else "large"
    meta = {
        "small_model": small_model,
        "large_model": large_model,
        "small_short": small_model.split("/")[-1],
        "large_short": large_model.split("/")[-1],
        "cost_source": cost_source,
        "inputs": inputs,
    }
    out = analysis.to_json(res)
    out["provenance"] = {
        **meta,
        "config": {
            k: (display_path(v, args.out_dir) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "ucci_version": ucci.__version__,
        "numpy_version": np.__version__,
        "python": platform.python_version(),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures = [] if args.no_figures else make_figures(res, args.out_dir, meta)
    out["figures"] = figures
    (args.out_dir / "results.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    (args.out_dir / "results.md").write_text(render_markdown(out, meta))
    with (args.out_dir / "joined.jsonl").open("w", encoding="utf-8") as fh:
        for rec in analysis.iter_joined_records(res):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    t = next((r for r in out["at_target"] if r["method"] == "UCCI"), None)
    if t is not None and "cost" in t:
        print(
            f"[analyze] UCCI at tau={out['target']['tau']:.4f}: cost {t['cost']:.3f}, micro-F1 {t['micro_f1']:.4f}, "
            f"escalated {t['escalation_rate']:.1%}",
            file=sys.stderr,
        )
    print(
        f"[analyze] wrote results.json, results.md, joined.jsonl{', ' + ', '.join(figures) if figures else ''} "
        f"to {args.out_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
