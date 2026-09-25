#!/usr/bin/env python3
"""Generate the golden test vectors shared by the Python package and the Rust crate.

Usage::

    python tools/make_golden.py           # (re)write tests/golden/*.json
    python tools/make_golden.py --check   # exit 1 if a file is missing or stale

Each golden file lists cases. A case names one function of the Python core
(``src/ucci``), the inputs it was called with, and either the outputs it
returned (``expected``) or the error it raised (``error``). Two test suites
replay every case:

* ``tests/test_golden.py`` against the current Python core, so any change in
  the core's behaviour fails until the goldens are regenerated on purpose;
* ``rust/tests/golden.rs`` against the Rust crate, which must agree to an
  absolute tolerance of 1e-12 (``tolerance`` in each file).

Output is deterministic: fixed seeds, sorted keys, floats in Python's
shortest round-trip form (``repr``). JSON has no NaN or infinity, so
non-finite floats are written as the strings ``"nan"``, ``"inf"`` and
``"-inf"``; readers map them back.

The script also writes ``rust/examples/router.json``, a router saved by the
Python package that ``cargo run --example route`` loads.

The file ``tests/golden/rust_written_routers.json`` is not written here: the
Rust test suite writes it (``UCCI_BLESS=1 cargo test --test golden``) and
``tests/test_golden.py`` checks that the Python package reads every router
the Rust crate writes.

The functions ``build_cases`` and ``compute`` are imported by
``tests/test_golden.py``; keep them free of side effects.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

# Always test the source tree, never an installed copy of the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

import ucci
from ucci import io as ucci_io
from ucci import metrics, policy, signal
from ucci.calibration import IsotonicCalibrator, pav
from ucci.router import UCCIRouter

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "tests" / "golden"
SCHEMA = "ucci-golden/1"
TOLERANCE = 1e-12

Case = dict[str, Any]

# --------------------------------------------------------------------------
# JSON encoding of floats (non-finite values as strings)
# --------------------------------------------------------------------------

_SPECIAL = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def enc(obj: Any) -> Any:
    """Plain-JSON form of ``obj``: numpy types unwrapped, non-finite floats as text."""
    if isinstance(obj, dict):
        return {str(k): enc(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [enc(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [enc(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if math.isnan(f):
            return "nan"
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    return obj


def dec(obj: Any) -> Any:
    """Inverse of :func:`enc` for inputs: "nan", "inf" and "-inf" become floats."""
    if isinstance(obj, dict):
        return {k: dec(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [dec(v) for v in obj]
    if isinstance(obj, str) and obj in _SPECIAL:
        return _SPECIAL[obj]
    return obj


def fl(values: Any) -> list[float]:
    """A list of Python floats."""
    return [float(v) for v in np.asarray(values, dtype=np.float64).ravel()]


# --------------------------------------------------------------------------
# compute(): one call into the Python core
# --------------------------------------------------------------------------


def _choice(c: policy.ThresholdChoice) -> dict[str, float]:
    return {
        "theta": c.theta,
        "cost": c.cost,
        "accuracy": c.accuracy,
        "escalation_rate": c.escalation_rate,
    }


def _grid(inp: dict[str, Any]) -> Any:
    return policy.DEFAULT_GRID if inp.get("grid") is None else inp["grid"]


def _costs(inp: dict[str, Any]) -> dict[str, Any]:
    return {
        "c_small": inp["c_small"],
        "c_large": inp["c_large"],
        "cost_model": inp["cost_model"],
    }


def _sweep_scores(inp: dict[str, Any]) -> Any:
    """Frontier arrays for the error details of an infeasible selection."""
    if "small_counts" in inp:
        metric = metrics.routed_micro_f1(
            inp["small_counts"], inp["large_counts"], inp.get("zero_division", 0.0)
        )
        return policy.pareto_frontier(
            inp["p_hat"], None, None, grid=_grid(inp), metric=metric, **_costs(inp)
        )
    return policy.pareto_frontier(
        inp["p_hat"],
        inp["small_score"],
        inp["large_score"],
        grid=_grid(inp),
        **_costs(inp),
    )


def _metric(inp: dict[str, Any]) -> Any:
    if "small_counts" not in inp:
        return None
    return metrics.routed_micro_f1(
        inp["small_counts"], inp["large_counts"], inp.get("zero_division", 0.0)
    )


def _reject_constant(name: str) -> float:
    raise ValueError(f"router file contains {name}, which is not valid JSON")


def _router_fields(router: UCCIRouter, doc: dict[str, Any]) -> dict[str, Any]:
    assert router.calibrator.x_ is not None and router.calibrator.y_ is not None
    return {
        "x": fl(router.calibrator.x_),
        "y": fl(router.calibrator.y_),
        "theta": router.theta,
        "c_small": router.c_small,
        "c_large": router.c_large,
        "cost_model": router.cost_model,
        "tau": router.tau,
        "grid_step": router.grid_step,
        "created_by": doc["created_by"],
    }


def _compute(fn: str, inp: dict[str, Any]) -> dict[str, Any]:
    """Run ``fn`` of the Python core on ``inp`` and return its outputs."""
    # ---------------------------------------------------------------- signal
    if fn == "margins_from_top2":
        return {"margins": signal.margins_from_top2(inp["pairs"])}
    if fn == "token_margin_uncertainty":
        return {"u": signal.token_margin_uncertainty(inp["pairs"])}
    if fn == "uncertainty_from_margins":
        return {"u": signal.uncertainty_from_margins(inp["margins"])}
    if fn == "uncertainty_from_probs":
        return {"u": signal.uncertainty_from_probs(inp["p1"], inp["p2"])}
    if fn == "uncertainty_from_logprobs":
        return {"u": signal.uncertainty_from_logprobs(inp["lp1"], inp["lp2"])}
    if fn == "top2_from_logprobs":
        return {"pairs": [list(p) for p in signal.top2_from_logprobs(inp["per_token"])]}
    if fn == "uncertainty_from_top_logprobs":
        return {"u": signal.token_margin_uncertainty(signal.top2_from_logprobs(inp["per_token"]))}
    if fn == "from_openai_logprobs":
        return {"u": signal.from_openai_logprobs(inp["content"])}
    # ----------------------------------------------------------- calibration
    if fn == "pav":
        return {"fit": fl(pav(inp["y"], inp["w"]))}
    if fn == "calibrator_fit":
        cal = IsotonicCalibrator().fit(inp["u"], inp["e"], inp["sample_weight"])
        return {
            "x": fl(cal.x_),
            "y": fl(cal.y_),
            "n_samples": cal.n_samples_,
            "predict": fl(cal.predict(np.asarray(inp["query"], dtype=np.float64))),
        }
    if fn == "calibrator_from_knots":
        cal = IsotonicCalibrator.from_dict({"x": inp["x"], "y": inp["y"]})
        return {"predict": fl(cal.predict(np.asarray(inp["query"], dtype=np.float64)))}
    if fn == "calibrator_predict":
        cal = IsotonicCalibrator.from_dict({"x": inp["x"], "y": inp["y"]})
        return {"p_hat": cal.predict(float(inp["u"]))}
    # ---------------------------------------------------------------- policy
    if fn == "default_grid":
        return {"grid": fl(policy.DEFAULT_GRID)}
    if fn == "make_grid":
        return {"grid": fl(policy.make_grid(inp["step"]))}
    if fn == "escalate":
        mask = policy.escalate(np.asarray(inp["p_hat"], dtype=np.float64), inp["theta"])
        return {"mask": [bool(v) for v in mask]}
    if fn == "escalate_scalar":
        return {"escalate": bool(policy.escalate(float(inp["p_hat"]), inp["theta"]))}
    if fn == "policy_cost":
        return {"cost": policy.policy_cost(inp["esc"], **_costs(inp))}
    if fn == "policy_accuracy":
        return {
            "accuracy": policy.policy_accuracy(inp["esc"], inp["small_score"], inp["large_score"])
        }
    if fn == "select_threshold":
        metric = _metric(inp)
        c = policy.select_threshold(
            inp["p_hat"],
            inp.get("small_score"),
            inp.get("large_score"),
            inp["tau"],
            grid=_grid(inp),
            metric=metric,
            **_costs(inp),
        )
        return _choice(c)
    if fn == "select_threshold_for_budget":
        metric = _metric(inp)
        c = policy.select_threshold_for_budget(
            inp["p_hat"],
            inp.get("small_score"),
            inp.get("large_score"),
            inp["budget"],
            grid=_grid(inp),
            metric=metric,
            **_costs(inp),
        )
        return _choice(c)
    if fn == "pareto_frontier":
        metric = _metric(inp)
        f = policy.pareto_frontier(
            inp["p_hat"],
            inp.get("small_score"),
            inp.get("large_score"),
            grid=_grid(inp),
            metric=metric,
            **_costs(inp),
        )
        return f.to_dict()
    if fn == "evaluate":
        metric = _metric(inp)
        c = policy.evaluate(
            inp["p_hat"],
            inp.get("small_score"),
            inp.get("large_score"),
            inp["theta"],
            metric=metric,
            **_costs(inp),
        )
        return _choice(c)
    # --------------------------------------------------------------- metrics
    if fn == "ece":
        return {
            "ece": metrics.ece(
                inp["p"], inp["y"], inp["n_bins"], inp["strategy"], inp["sample_weight"]
            )
        }
    if fn == "reliability_table":
        rows = metrics.reliability_table(
            inp["p"], inp["y"], inp["n_bins"], inp["strategy"], inp["sample_weight"]
        )
        return {"rows": [r._asdict() for r in rows]}
    if fn == "brier_score":
        return {"brier": metrics.brier_score(inp["p"], inp["y"], inp["sample_weight"])}
    if fn == "micro_f1":
        return {"f1": metrics.micro_f1(inp["tp"], inp["fp"], inp["fn"], inp["zero_division"])}
    if fn == "routed_micro_f1":
        f1 = metrics.routed_micro_f1(inp["small_counts"], inp["large_counts"], inp["zero_division"])
        return {"scores": [f1(np.asarray(m, dtype=bool)) for m in inp["masks"]]}
    # ---------------------------------------------------------------- router
    if fn == "router_load":
        data = json.loads(inp["json_text"], parse_constant=_reject_constant)
        doc = ucci_io.validate_router_dict(data)
        router = UCCIRouter.from_dict(data)
        out = _router_fields(router, doc)
        out["p_hat"] = [router.error_probability(float(q)) for q in inp["queries"]]
        out["escalate"] = [bool(router.escalate(float(q))) for q in inp["queries"]]
        return out
    if fn == "router_fit":
        router = UCCIRouter(inp["c_small"], inp["c_large"], inp["cost_model"], inp["grid_step"])
        router.calibrate(inp["u_cal"], inp["e_cal"], inp.get("w_cal"))
        if inp.get("tau") is not None:
            choice = router.choose_threshold(
                inp["u_val"], inp["small_val"], inp["large_val"], inp["tau"]
            )
        else:
            choice = router.choose_threshold_for_budget(
                inp["u_val"], inp["small_val"], inp["large_val"], inp["budget"]
            )
        test = router.evaluate(inp["u_test"], inp["small_test"], inp["large_test"])
        doc = ucci_io.validate_router_dict(router.to_dict())
        out = _router_fields(router, doc)
        out.pop("created_by")
        out["choice"] = _choice(choice)
        out["test"] = _choice(test)
        out["n_samples"] = router.calibrator.n_samples_
        return out
    raise KeyError(f"unknown golden function {fn!r}")


def compute(fn: str, inp: dict[str, Any]) -> dict[str, Any]:
    """Outputs of ``fn`` on ``inp`` as ``{"expected": ...}`` or ``{"error": ...}``.

    ``inp`` uses the JSON encoding of :func:`enc` (it is decoded here), and
    the result is JSON-ready. Errors record the exception type and message;
    an infeasible threshold selection also records the best accuracy (or the
    cheapest cost) on the grid and its threshold, which the Rust error
    carries too.
    """
    raw = dec(inp)
    try:
        return {"expected": enc(_compute(fn, raw))}
    except (ValueError, RuntimeError) as exc:
        err: dict[str, Any] = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, policy.InfeasibleTargetError) and fn == "router_fit":
            err["kind"] = "infeasible" if raw.get("tau") is not None else "over_budget"
        elif isinstance(exc, policy.InfeasibleTargetError):
            f = _sweep_scores(raw)
            if fn == "select_threshold":
                j = int(np.argmax(f.accuracy))
                err.update(
                    kind="infeasible",
                    best_accuracy=float(f.accuracy[j]),
                    best_theta=float(f.theta[j]),
                )
            else:
                j = int(np.argmin(f.cost))
                err.update(
                    kind="over_budget",
                    min_cost=float(f.cost[j]),
                    cheapest_theta=float(f.theta[j]),
                )
        return {"error": enc(err)}


# --------------------------------------------------------------------------
# Case builders
# --------------------------------------------------------------------------


class CaseList(list[Case]):
    """Cases of one golden file plus the named datasets they share.

    A case whose input holds ``"$data": name`` takes every field of
    ``data[name]`` as input too (fields given in the case win). Sharing keeps
    the files small when many cases run on the same arrays.
    """

    def __init__(self) -> None:
        super().__init__()
        self.data: dict[str, dict[str, Any]] = {}


def _case(
    cases: CaseList,
    cid: str,
    func: str,
    /,
    *,
    data: tuple[str, dict[str, Any]] | None = None,
    **inp: Any,
) -> None:
    if any(c["id"] == cid for c in cases):
        raise ValueError(f"duplicate case id {cid!r}")
    payload = enc(inp)
    if data is not None:
        name, fields = data
        encoded = enc(fields)
        if cases.data.setdefault(name, encoded) != encoded:
            raise ValueError(f"dataset {name!r} registered twice with different content")
        payload["$data"] = name
    cases.append({"id": cid, "fn": func, "input": payload})


def resolve(inp: dict[str, Any], data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Full input of a case: its shared dataset (if any) merged with its own fields."""
    if "$data" not in inp:
        return inp
    merged = dict(data[inp["$data"]])
    merged.update({k: v for k, v in inp.items() if k != "$data"})
    return merged


def _probs_pairs(rng: np.random.Generator, n: int) -> list[list[float]]:
    """Top-2 probability pairs as a softmax over 5 logits would give them."""
    logits = rng.normal(0.0, 2.5, size=(n, 5))
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    p.sort(axis=1)
    return [[float(a), float(b)] for a, b in zip(p[:, -1], p[:, -2])]


def signal_cases() -> CaseList:
    rng = np.random.default_rng(101)
    c = CaseList()
    basic = [[0.9, 0.05], [0.6, 0.3], [0.5, 0.5]]
    _case(c, "pairs_paper_example", "token_margin_uncertainty", pairs=basic)
    _case(c, "margins_paper_example", "margins_from_top2", pairs=basic)
    _case(c, "pairs_swapped", "token_margin_uncertainty", pairs=[[0.1, 0.8], [0.3, 0.6]])
    _case(
        c,
        "margins_swapped",
        "margins_from_top2",
        pairs=[[0.1, 0.8], [0.3, 0.6], [0.2, 0.2]],
    )
    _case(c, "pairs_all_certain", "token_margin_uncertainty", pairs=[[1.0, 0.0]] * 4)
    _case(c, "pairs_all_tied", "token_margin_uncertainty", pairs=[[0.4, 0.4]] * 4)
    _case(c, "pairs_one_token", "token_margin_uncertainty", pairs=[[0.73, 0.21]])
    _case(
        c,
        "pairs_above_one_within_tolerance",
        "margins_from_top2",
        pairs=[[1.0 + 5e-10, 0.0], [0.7, 0.2]],
    )
    for n in (7, 8, 9, 127, 128, 129, 300, 1000):
        _case(
            c,
            f"pairs_random_{n}",
            "token_margin_uncertainty",
            pairs=_probs_pairs(rng, n),
        )
    _case(c, "margins_random_200", "margins_from_top2", pairs=_probs_pairs(rng, 200))
    _case(c, "pairs_error_empty", "token_margin_uncertainty", pairs=[])
    _case(
        c,
        "pairs_error_above_one",
        "token_margin_uncertainty",
        pairs=[[0.5, 0.2], [1.1, 0.0]],
    )
    _case(
        c,
        "pairs_error_above_tolerance",
        "token_margin_uncertainty",
        pairs=[[1.0 + 2e-9, 0.0]],
    )
    _case(c, "pairs_error_negative", "token_margin_uncertainty", pairs=[[-0.1, 0.5]])
    _case(c, "pairs_error_nan", "margins_from_top2", pairs=[[math.nan, 0.5]])
    _case(c, "pairs_error_nan_second", "margins_from_top2", pairs=[[0.5, math.nan]])

    _case(c, "margins_paper", "uncertainty_from_margins", margins=[0.85, 0.3, 0.0])
    _case(c, "margins_random_333", "uncertainty_from_margins", margins=fl(rng.random(333)))
    _case(c, "margins_bounds", "uncertainty_from_margins", margins=[0.0, 1.0, 1.0])
    _case(c, "margins_error_empty", "uncertainty_from_margins", margins=[])
    _case(c, "margins_error_above_one", "uncertainty_from_margins", margins=[0.5, 1.5])
    _case(c, "margins_error_negative", "uncertainty_from_margins", margins=[-0.1])
    _case(c, "margins_error_nan", "uncertainty_from_margins", margins=[0.2, math.nan])

    pairs = np.asarray(_probs_pairs(rng, 50))
    _case(
        c,
        "probs_random_50",
        "uncertainty_from_probs",
        p1=fl(pairs[:, 0]),
        p2=fl(pairs[:, 1]),
    )
    swap = rng.random(50) < 0.5
    p1 = np.where(swap, pairs[:, 1], pairs[:, 0])
    p2 = np.where(swap, pairs[:, 0], pairs[:, 1])
    _case(c, "probs_random_50_swapped", "uncertainty_from_probs", p1=fl(p1), p2=fl(p2))
    _case(
        c,
        "probs_tolerance",
        "uncertainty_from_probs",
        p1=[1.0 + 1e-9, 0.5],
        p2=[0.0, 0.25],
    )
    _case(c, "probs_error_mismatch", "uncertainty_from_probs", p1=[0.5, 0.4], p2=[0.2])
    _case(c, "probs_error_empty", "uncertainty_from_probs", p1=[], p2=[])
    _case(
        c,
        "probs_error_nan",
        "uncertainty_from_probs",
        p1=[0.5, math.nan],
        p2=[0.2, 0.1],
    )
    _case(c, "probs_error_above_one", "uncertainty_from_probs", p1=[1.5], p2=[0.2])
    _case(c, "probs_error_negative", "uncertainty_from_probs", p1=[0.5], p2=[-0.2])

    lp = np.log(np.asarray(_probs_pairs(rng, 64)))
    _case(
        c,
        "logprobs_random_64",
        "uncertainty_from_logprobs",
        lp1=fl(lp[:, 0]),
        lp2=fl(lp[:, 1]),
    )
    _case(
        c,
        "logprobs_with_zero_probability",
        "uncertainty_from_logprobs",
        lp1=[0.0, -0.1, -0.7],
        lp2=[-math.inf, -2.5, -0.7],
    )
    _case(
        c,
        "logprobs_at_max",
        "uncertainty_from_logprobs",
        lp1=[math.log1p(1e-9)],
        lp2=[-3.0],
    )
    _case(
        c,
        "logprobs_error_above_max",
        "uncertainty_from_logprobs",
        lp1=[1e-9],
        lp2=[-3.0],
    )
    _case(c, "logprobs_error_positive", "uncertainty_from_logprobs", lp1=[0.5], lp2=[-3.0])
    _case(
        c,
        "logprobs_error_plus_inf",
        "uncertainty_from_logprobs",
        lp1=[math.inf],
        lp2=[-3.0],
    )
    _case(
        c,
        "logprobs_error_nan",
        "uncertainty_from_logprobs",
        lp1=[-0.1, math.nan],
        lp2=[-1.0, -2.0],
    )
    _case(
        c,
        "logprobs_error_mismatch",
        "uncertainty_from_logprobs",
        lp1=[-0.1],
        lp2=[-1.0, -2.0],
    )
    _case(c, "logprobs_error_empty", "uncertainty_from_logprobs", lp1=[], lp2=[])

    per_token: list[list[float]] = []
    for k in [2, 3, 5, 2, 4, 6, 3, 2]:
        logits = rng.normal(0.0, 2.0, size=12)
        lse = float(np.log(np.exp(logits).sum()))
        cand = sorted((logits - lse).tolist(), reverse=True)[:k]
        order = rng.permutation(k)
        per_token.append([cand[i] for i in order])
    per_token.append([-0.5, -0.5, -3.0])
    per_token.append([-math.inf, 0.0])
    per_token.append([-math.inf, -math.inf, -0.2])
    _case(c, "top2_candidates_mixed", "top2_from_logprobs", per_token=per_token)
    _case(c, "top_logprobs_mixed", "uncertainty_from_top_logprobs", per_token=per_token)
    _case(
        c,
        "top2_error_one_candidate",
        "top2_from_logprobs",
        per_token=[[-0.1, -2.0], [-0.3]],
    )
    _case(c, "top2_error_no_candidates", "top2_from_logprobs", per_token=[[]])
    _case(c, "top2_error_nan", "top2_from_logprobs", per_token=[[-0.1, math.nan, -2.0]])
    _case(
        c,
        "top2_error_nan_not_top",
        "top2_from_logprobs",
        per_token=[[-0.1, -0.2, math.nan]],
    )
    _case(c, "top2_error_positive", "top2_from_logprobs", per_token=[[0.5, -1.0]])
    _case(c, "top_logprobs_error_empty", "uncertainty_from_top_logprobs", per_token=[])

    content = [
        {
            "token": f"t{i}",
            "logprob": cands[0],
            "top_logprobs": [{"token": f"c{j}", "logprob": lpv} for j, lpv in enumerate(cands)],
        }
        for i, cands in enumerate(per_token[:8])
    ]
    _case(c, "openai_content", "from_openai_logprobs", content=content)
    _case(
        c,
        "openai_error_missing_top_logprobs",
        "from_openai_logprobs",
        content=[{"token": "a", "logprob": -0.1, "top_logprobs": None}],
    )
    _case(
        c,
        "openai_error_one_candidate",
        "from_openai_logprobs",
        content=[
            {
                "token": "a",
                "logprob": -0.1,
                "top_logprobs": [{"token": "a", "logprob": -0.1}],
            }
        ],
    )
    _case(c, "openai_error_empty", "from_openai_logprobs", content=[])
    return c


def calibration_cases() -> CaseList:
    rng = np.random.default_rng(202)
    c = CaseList()
    _case(c, "pav_known", "pav", y=[1.0, 3.0, 2.0, 4.0], w=None)
    _case(c, "pav_decreasing", "pav", y=[4.0, 3.0, 2.0, 1.0], w=None)
    _case(c, "pav_constant", "pav", y=[0.3] * 5, w=None)
    _case(c, "pav_single", "pav", y=[0.7], w=None)
    _case(c, "pav_late_violator", "pav", y=[1.0, 2.0, 3.0, 0.0], w=None)
    _case(
        c,
        "pav_weighted_dyadic",
        "pav",
        y=[3.0, 1.0, 2.0, 0.5, 4.0],
        w=[3.0, 1.0, 0.5, 2.25, 1.0],
    )
    _case(c, "pav_random_300", "pav", y=fl(rng.random(300)), w=None)
    _case(
        c,
        "pav_random_weighted_300",
        "pav",
        y=fl(rng.random(300)),
        w=fl(rng.random(300) + 0.05),
    )
    _case(
        c,
        "pav_binary_500",
        "pav",
        y=fl((rng.random(500) < np.linspace(0.1, 0.9, 500)) * 1.0),
        w=None,
    )
    _case(c, "pav_error_empty", "pav", y=[], w=None)
    _case(c, "pav_error_zero_weight", "pav", y=[1.0, 2.0], w=[1.0, 0.0])
    _case(c, "pav_error_negative_weight", "pav", y=[1.0, 2.0], w=[1.0, -1.0])
    _case(c, "pav_error_nan", "pav", y=[1.0, math.nan], w=None)
    _case(c, "pav_error_inf_weight", "pav", y=[1.0, 2.0], w=[1.0, math.inf])
    _case(c, "pav_error_length", "pav", y=[1.0, 2.0], w=[1.0])

    query = [
        -1.0,
        -1e-300,
        0.0,
        0.001,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.33,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        0.999,
        1.0,
        1.5,
        1e300,
    ]
    _case(
        c,
        "fit_docstring",
        "calibrator_fit",
        u=[0.1, 0.2, 0.3, 0.4],
        e=[0, 1, 0, 1],
        sample_weight=None,
        query=query,
    )
    _case(
        c,
        "fit_trim_example",
        "calibrator_fit",
        u=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        e=[0, 1, 0, 0, 1, 1],
        sample_weight=None,
        query=query,
    )
    _case(
        c,
        "fit_ties",
        "calibrator_fit",
        u=[0.5, 0.1, 0.5, 0.9, 0.5, 0.1],
        e=[1, 0, 0, 1, 1, 1],
        sample_weight=None,
        query=query,
    )
    u = rng.random(2000)
    e = (rng.random(2000) < u**2).astype(float)
    _case(
        c,
        "fit_random_2000",
        "calibrator_fit",
        u=fl(u),
        e=fl(e),
        sample_weight=None,
        query=query + fl(rng.random(40)),
    )
    u = np.round(rng.random(800), 2)
    e = (rng.random(800) < 0.2 + 0.6 * u).astype(float)
    _case(
        c,
        "fit_rounded_ties_800",
        "calibrator_fit",
        u=fl(u),
        e=fl(e),
        sample_weight=None,
        query=query + fl(np.unique(u)[:30]),
    )
    u = rng.random(300)
    e = (rng.random(300) < u).astype(float)
    w = rng.random(300) * 3
    w[rng.random(300) < 0.2] = 0.0
    _case(
        c,
        "fit_weights_with_zeros",
        "calibrator_fit",
        u=fl(u),
        e=fl(e),
        sample_weight=fl(w),
        query=query,
    )
    _case(
        c,
        "fit_zero_weight_drops_point",
        "calibrator_fit",
        u=[0.0, 0.5, 1.0],
        e=[1, 0, 1],
        sample_weight=[0.0, 1.0, 1.0],
        query=query,
    )
    u = np.round(rng.random(400), 1)
    e = (rng.random(400) < 0.5).astype(float)
    wd = rng.integers(1, 9, 400) / 4.0
    _case(
        c,
        "fit_dyadic_weights_ties",
        "calibrator_fit",
        u=fl(u),
        e=fl(e),
        sample_weight=fl(wd),
        query=query,
    )
    nxt = float(np.nextafter(0.5, 1.0))
    nxt2 = float(np.nextafter(nxt, 1.0))
    _case(
        c,
        "fit_near_ties",
        "calibrator_fit",
        u=[0.1, 0.5, nxt, nxt2, 0.5 + 2e-15, 0.9],
        e=[0, 1, 0, 1, 0, 1],
        sample_weight=None,
        query=[*query, 0.5, nxt, nxt2],
    )
    _case(
        c,
        "fit_signed_zero",
        "calibrator_fit",
        u=[0.0, -0.0, 0.5, 1.0],
        e=[1, 0, 0, 1],
        sample_weight=None,
        query=query,
    )
    _case(
        c,
        "fit_all_correct",
        "calibrator_fit",
        u=fl(rng.random(50)),
        e=[0.0] * 50,
        sample_weight=None,
        query=query,
    )
    _case(
        c,
        "fit_all_wrong",
        "calibrator_fit",
        u=fl(rng.random(50)),
        e=[1.0] * 50,
        sample_weight=None,
        query=query,
    )
    _case(
        c,
        "fit_single_value",
        "calibrator_fit",
        u=[0.3, 0.3, 0.3],
        e=[1, 0, 0],
        sample_weight=None,
        query=query,
    )
    _case(
        c,
        "fit_single_point",
        "calibrator_fit",
        u=[0.42],
        e=[1],
        sample_weight=None,
        query=query,
    )
    _case(
        c,
        "fit_soft_labels",
        "calibrator_fit",
        u=fl(rng.random(100)),
        e=fl(rng.random(100)),
        sample_weight=None,
        query=query,
    )
    _case(
        c,
        "fit_u_outside_unit_interval",
        "calibrator_fit",
        u=[-2.0, -0.5, 0.5, 3.0, 7.5],
        e=[0, 1, 0, 1, 1],
        sample_weight=None,
        query=[*query, -3.0, 8.0],
    )
    _case(c, "fit_error_empty", "calibrator_fit", u=[], e=[], sample_weight=None, query=[])
    _case(
        c,
        "fit_error_length",
        "calibrator_fit",
        u=[0.1, 0.2],
        e=[1],
        sample_weight=None,
        query=[],
    )
    _case(
        c,
        "fit_error_nan_u",
        "calibrator_fit",
        u=[0.1, math.nan],
        e=[1, 0],
        sample_weight=None,
        query=[],
    )
    _case(
        c,
        "fit_error_inf_u",
        "calibrator_fit",
        u=[0.1, math.inf],
        e=[1, 0],
        sample_weight=None,
        query=[],
    )
    _case(
        c,
        "fit_error_label_above_one",
        "calibrator_fit",
        u=[0.1, 0.2],
        e=[1, 2],
        sample_weight=None,
        query=[],
    )
    _case(
        c,
        "fit_error_label_nan",
        "calibrator_fit",
        u=[0.1, 0.2],
        e=[1, math.nan],
        sample_weight=None,
        query=[],
    )
    _case(
        c,
        "fit_error_negative_weight",
        "calibrator_fit",
        u=[0.1, 0.2],
        e=[1, 0],
        sample_weight=[1.0, -1.0],
        query=[],
    )
    _case(
        c,
        "fit_error_zero_weights",
        "calibrator_fit",
        u=[0.1, 0.2],
        e=[1, 0],
        sample_weight=[0.0, 0.0],
        query=[],
    )
    _case(
        c,
        "fit_error_weight_length",
        "calibrator_fit",
        u=[0.1, 0.2],
        e=[1, 0],
        sample_weight=[1.0],
        query=[],
    )
    _case(
        c,
        "fit_error_nan_query",
        "calibrator_fit",
        u=[0.1, 0.2],
        e=[1, 0],
        sample_weight=None,
        query=[0.1, math.nan],
    )

    _case(
        c,
        "knots_three",
        "calibrator_from_knots",
        x=[0.1, 0.4, 0.8],
        y=[0.05, 0.2, 0.7],
        query=query,
    )
    _case(c, "knots_single", "calibrator_from_knots", x=[0.3], y=[0.25], query=query)
    _case(
        c,
        "knots_flat_steps",
        "calibrator_from_knots",
        x=[0.0, 0.2, 0.2000001, 0.6, 1.0],
        y=[0.0, 0.0, 0.5, 0.5, 1.0],
        query=[*query, 0.20000005],
    )
    _case(
        c,
        "knots_error_not_increasing",
        "calibrator_from_knots",
        x=[0.2, 0.1],
        y=[0.1, 0.2],
        query=[],
    )
    _case(
        c,
        "knots_error_duplicate_x",
        "calibrator_from_knots",
        x=[0.1, 0.1],
        y=[0.1, 0.2],
        query=[],
    )
    _case(
        c,
        "knots_error_decreasing_y",
        "calibrator_from_knots",
        x=[0.1, 0.2],
        y=[0.3, 0.2],
        query=[],
    )
    _case(
        c,
        "knots_error_y_above_one",
        "calibrator_from_knots",
        x=[0.1, 0.2],
        y=[0.3, 1.2],
        query=[],
    )
    _case(
        c,
        "knots_error_y_negative",
        "calibrator_from_knots",
        x=[0.1, 0.2],
        y=[-0.1, 0.2],
        query=[],
    )
    _case(
        c,
        "knots_error_length",
        "calibrator_from_knots",
        x=[0.1, 0.2],
        y=[0.3],
        query=[],
    )
    _case(c, "knots_error_empty", "calibrator_from_knots", x=[], y=[], query=[])
    _case(
        c,
        "knots_error_nan",
        "calibrator_from_knots",
        x=[0.1, math.nan],
        y=[0.3, 0.4],
        query=[],
    )
    _case(
        c,
        "predict_at_knot",
        "calibrator_predict",
        x=[0.1, 0.4, 0.8],
        y=[0.05, 0.2, 0.7],
        u=0.4,
    )
    _case(
        c,
        "predict_error_nan",
        "calibrator_predict",
        x=[0.1, 0.4],
        y=[0.05, 0.2],
        u=math.nan,
    )
    _case(
        c,
        "predict_error_inf",
        "calibrator_predict",
        x=[0.1, 0.4],
        y=[0.05, 0.2],
        u=math.inf,
    )
    _case(
        c,
        "predict_error_nan_single_knot",
        "calibrator_predict",
        x=[0.1],
        y=[0.05],
        u=math.nan,
    )
    return c


def _synthetic_scores(
    rng: np.random.Generator, n: int, frac: bool = False
) -> dict[str, list[float]]:
    """p_hat plus 0/1 (or fractional) scores of a small and a large model."""
    p_hat = rng.random(n)
    small = (rng.random(n) >= p_hat).astype(float)
    large = (rng.random(n) >= 0.08).astype(float)
    if frac:
        levels = np.array([0.0, 0.25, 0.5, 2.0 / 3.0, 0.8, 1.0])
        small = np.where(small > 0, 1.0, levels[rng.integers(0, 5, n)])
        large = np.where(large > 0, 1.0, levels[rng.integers(0, 5, n)])
    return {"p_hat": fl(p_hat), "small_score": fl(small), "large_score": fl(large)}


def _counts(rng: np.random.Generator, n: int, p_hat: np.ndarray) -> dict[str, Any]:
    """Per-query (tp, fp, fn) entity counts: the small model errs more at high p_hat."""
    gold = rng.integers(0, 4, n)
    s_tp = np.where(rng.random(n) < p_hat, np.maximum(gold - 1, 0), gold)
    s_fp = (rng.random(n) < 0.5 * p_hat).astype(int)
    l_tp = np.where(rng.random(n) < 0.1, np.maximum(gold - 1, 0), gold)
    l_fp = (rng.random(n) < 0.05).astype(int)
    small = np.stack([s_tp, s_fp, gold - s_tp], axis=1).astype(float)
    large = np.stack([l_tp, l_fp, gold - l_tp], axis=1).astype(float)
    return {"small_counts": small.tolist(), "large_counts": large.tolist()}


#: The paper's normalized costs and cost model (Section 6.1).
PAPER: dict[str, Any] = {"c_small": 1.0, "c_large": 3.02, "cost_model": "routing"}


def policy_cases() -> CaseList:
    rng = np.random.default_rng(303)
    c = CaseList()
    _case(c, "default_grid", "default_grid")
    for step in (0.005, 0.01, 0.02, 0.05, 0.1, 0.125, 0.25, 0.5, 1.0, 1.0 / 3.0, 0.001):
        _case(c, f"make_grid_{step!r}", "make_grid", step=step)
    for step in (0.0, -0.1, 1.5, 0.3, 0.4, math.nan):
        _case(c, f"make_grid_error_{step!r}", "make_grid", step=step)

    _case(
        c,
        "escalate_strict",
        "escalate",
        p_hat=[0.1, 0.3, 0.30000000000000004, 0.9, 0.3],
        theta=0.3,
    )
    _case(c, "escalate_outside_unit", "escalate", p_hat=[-0.5, 1.5], theta=0.5)
    _case(c, "escalate_error_nan", "escalate", p_hat=[0.1, math.nan], theta=0.3)
    _case(c, "escalate_error_theta_nan", "escalate", p_hat=[0.1], theta=math.nan)
    _case(c, "escalate_scalar_equal", "escalate_scalar", p_hat=0.3, theta=0.3)
    _case(c, "escalate_scalar_above", "escalate_scalar", p_hat=0.31, theta=0.3)
    _case(c, "escalate_scalar_error_inf", "escalate_scalar", p_hat=math.inf, theta=0.3)

    mask = [True, False, False, True, True, False, False]
    for model in ("routing", "sequential"):
        _case(
            c,
            f"cost_{model}",
            "policy_cost",
            esc=mask,
            c_small=1.0,
            c_large=3.02,
            cost_model=model,
        )
        _case(
            c,
            f"cost_{model}_ratio5",
            "policy_cost",
            esc=mask,
            c_small=1.0,
            c_large=5.0,
            cost_model=model,
        )
    _case(
        c,
        "cost_table3_ratio10",
        "policy_cost",
        esc=[True] * 1069 + [False] * 931,
        c_small=1.0,
        c_large=10.0,
        cost_model="routing",
    )
    _case(c, "cost_all_kept", "policy_cost", esc=[False] * 3, **PAPER)
    _case(c, "cost_error_empty", "policy_cost", esc=[], **PAPER)
    _case(
        c,
        "cost_error_zero_cost",
        "policy_cost",
        esc=[True],
        c_small=0.0,
        c_large=3.02,
        cost_model="routing",
    )
    _case(
        c,
        "cost_error_negative_cost",
        "policy_cost",
        esc=[True],
        c_small=1.0,
        c_large=-2.0,
        cost_model="routing",
    )
    d = _synthetic_scores(rng, 300, frac=True)
    esc = (rng.random(300) < 0.4).tolist()
    _case(
        c,
        "accuracy_fractional_300",
        "policy_accuracy",
        esc=esc,
        small_score=d["small_score"],
        large_score=d["large_score"],
    )
    _case(
        c,
        "accuracy_error_length",
        "policy_accuracy",
        esc=[True, False],
        small_score=[1.0],
        large_score=[1.0, 0.0],
    )
    _case(
        c,
        "accuracy_error_nan",
        "policy_accuracy",
        esc=[True, False],
        small_score=[1.0, math.nan],
        large_score=[1.0, 0.0],
    )

    _case(
        c,
        "select_docstring",
        "select_threshold",
        p_hat=[0.1, 0.2, 0.6, 0.9],
        small_score=[1, 1, 0, 0],
        large_score=[1, 1, 1, 1],
        tau=1.0,
        grid=None,
        **PAPER,
    )
    d500 = _synthetic_scores(rng, 500)
    # 0.912 = 456 / 500 is the best accuracy on the grid: feasibility at equality.
    for tau in (0.5, 0.8, 0.85, 0.88, 0.9, 0.912):
        for model in ("routing", "sequential"):
            _case(
                c,
                f"select_500_tau{tau}_{model}",
                "select_threshold",
                tau=tau,
                grid=None,
                c_small=1.0,
                c_large=3.02,
                cost_model=model,
                data=("d500", d500),
            )
    d400 = _synthetic_scores(rng, 400, frac=True)
    for tau in (0.7, 0.85, 0.9):
        _case(
            c,
            f"select_fractional_400_tau{tau}",
            "select_threshold",
            tau=tau,
            grid=None,
            **PAPER,
            data=("d400", d400),
        )
    _case(
        c,
        "select_error_infeasible",
        "select_threshold",
        tau=0.999,
        grid=None,
        **PAPER,
        data=("d500", d500),
    )
    _case(
        c,
        "select_error_infeasible_above_one",
        "select_threshold",
        tau=1.01,
        grid=None,
        **PAPER,
        data=("d500", d500),
    )
    _case(
        c,
        "select_tau_zero",
        "select_threshold",
        tau=0.0,
        grid=None,
        **PAPER,
        data=("d500", d500),
    )
    _case(
        c,
        "select_tau_negative",
        "select_threshold",
        tau=-1.0,
        grid=None,
        **PAPER,
        data=("d500", d500),
    )
    grid = [0.9, 0.1, 0.5, 0.5, 0.0, 0.25, 1.0, 0.75]
    _case(
        c,
        "select_custom_grid",
        "select_threshold",
        tau=0.85,
        grid=grid,
        **PAPER,
        data=("d500", d500),
    )
    g = np.asarray(policy.DEFAULT_GRID)
    on_grid = g[rng.integers(0, g.size, 300)]
    tied = {
        "p_hat": fl(on_grid),
        "small_score": fl((rng.random(300) >= on_grid).astype(float)),
        "large_score": fl((rng.random(300) >= 0.05).astype(float)),
    }
    for tau in (0.8, 0.9):
        _case(
            c,
            f"select_ties_on_grid_tau{tau}",
            "select_threshold",
            tau=tau,
            grid=None,
            **PAPER,
            data=("tied", tied),
        )
    _case(
        c,
        "select_single_query",
        "select_threshold",
        p_hat=[0.4],
        small_score=[0.0],
        large_score=[1.0],
        tau=1.0,
        grid=None,
        **PAPER,
    )
    _case(
        c,
        "select_p_hat_outside_unit",
        "select_threshold",
        p_hat=[-0.2, 0.3, 1.7],
        small_score=[1, 1, 0],
        large_score=[1, 1, 1],
        tau=1.0,
        grid=None,
        **PAPER,
    )
    d2000 = _synthetic_scores(rng, 2000, frac=True)
    _case(
        c,
        "select_fractional_2000",
        "select_threshold",
        tau=0.9,
        grid=None,
        **PAPER,
        data=("d2000", d2000),
    )
    _case(
        c,
        "select_error_costs_unordered",
        "select_threshold",
        tau=0.8,
        grid=None,
        c_small=2.0,
        c_large=1.0,
        cost_model="routing",
        data=("d500", d500),
    )
    _case(
        c,
        "select_sequential_costs_unordered",
        "select_threshold",
        tau=0.8,
        grid=None,
        c_small=2.0,
        c_large=1.0,
        cost_model="sequential",
        data=("d500", d500),
    )
    _case(
        c,
        "select_error_tau_nan",
        "select_threshold",
        tau=math.nan,
        grid=None,
        **PAPER,
        data=("d500", d500),
    )
    _case(
        c,
        "select_error_grid_outside",
        "select_threshold",
        tau=0.8,
        grid=[0.1, 1.5],
        **PAPER,
        data=("d500", d500),
    )
    _case(
        c,
        "select_error_grid_empty",
        "select_threshold",
        tau=0.8,
        grid=[],
        **PAPER,
        data=("d500", d500),
    )
    _case(
        c,
        "select_error_length",
        "select_threshold",
        p_hat=[0.1, 0.2],
        small_score=[1.0],
        large_score=[1.0, 1.0],
        tau=0.5,
        grid=None,
        **PAPER,
    )
    _case(
        c,
        "select_error_nan_p_hat",
        "select_threshold",
        p_hat=[0.1, math.nan],
        small_score=[1.0, 1.0],
        large_score=[1.0, 1.0],
        tau=0.5,
        grid=None,
        **PAPER,
    )

    for budget in (1.0, 1.5, 2.0, 2.5, 3.02, 10.0):
        _case(
            c,
            f"budget_500_{budget}",
            "select_threshold_for_budget",
            budget=budget,
            grid=None,
            **PAPER,
            data=("d500", d500),
        )
    _case(
        c,
        "budget_fractional_400",
        "select_threshold_for_budget",
        budget=2.0,
        grid=None,
        **PAPER,
        data=("d400", d400),
    )
    _case(
        c,
        "budget_sequential",
        "select_threshold_for_budget",
        budget=2.5,
        grid=None,
        c_small=1.0,
        c_large=3.02,
        cost_model="sequential",
        data=("d500", d500),
    )
    _case(
        c,
        "budget_error_below_min",
        "select_threshold_for_budget",
        budget=0.5,
        grid=None,
        **PAPER,
        data=("d500", d500),
    )
    _case(
        c,
        "budget_error_nan",
        "select_threshold_for_budget",
        budget=math.nan,
        grid=None,
        **PAPER,
        data=("d500", d500),
    )
    _case(
        c,
        "budget_exact_cost",
        "select_threshold_for_budget",
        p_hat=[0.1, 0.2, 0.6, 0.9],
        small_score=[1, 1, 0, 0],
        large_score=[1, 1, 1, 1],
        budget=2.0,
        grid=None,
        c_small=1.0,
        c_large=3.0,
        cost_model="routing",
    )

    _case(c, "frontier_500", "pareto_frontier", grid=None, **PAPER, data=("d500", d500))
    _case(
        c,
        "frontier_fractional_custom_grid",
        "pareto_frontier",
        grid=grid,
        **PAPER,
        data=("d400", d400),
    )
    _case(
        c,
        "frontier_unordered_costs",
        "pareto_frontier",
        grid=None,
        c_small=3.0,
        c_large=1.0,
        cost_model="routing",
        data=("d500", d500),
    )
    _case(
        c,
        "frontier_sequential",
        "pareto_frontier",
        grid=None,
        c_small=1.0,
        c_large=3.02,
        cost_model="sequential",
        data=("d400", d400),
    )

    for theta in (-0.5, 0.0, 0.3, 0.535, 1.0, 2.0):
        _case(
            c,
            f"evaluate_fractional_{theta}",
            "evaluate",
            theta=theta,
            **PAPER,
            data=("d400", d400),
        )
    _case(
        c,
        "evaluate_sequential",
        "evaluate",
        theta=0.4,
        c_small=1.0,
        c_large=5.0,
        cost_model="sequential",
        data=("d500", d500),
    )
    _case(
        c,
        "evaluate_error_theta_nan",
        "evaluate",
        theta=math.nan,
        **PAPER,
        data=("d500", d500),
    )
    _case(
        c,
        "evaluate_error_costs",
        "evaluate",
        theta=0.5,
        c_small=-1.0,
        c_large=3.0,
        cost_model="routing",
        data=("d500", d500),
    )

    p_hat = rng.random(250)
    counts = {"p_hat": fl(p_hat), **_counts(rng, 250, p_hat)}
    for tau in (0.8, 0.9, 0.94):
        _case(
            c,
            f"select_micro_f1_tau{tau}",
            "select_threshold",
            tau=tau,
            grid=None,
            zero_division=0.0,
            data=("counts", counts),
            **PAPER,
        )
    _case(
        c,
        "select_micro_f1_error_infeasible",
        "select_threshold",
        tau=0.9999,
        grid=None,
        zero_division=0.0,
        data=("counts", counts),
        **PAPER,
    )
    _case(
        c,
        "budget_micro_f1",
        "select_threshold_for_budget",
        budget=2.0,
        grid=None,
        zero_division=0.0,
        data=("counts", counts),
        **PAPER,
    )
    _case(
        c,
        "frontier_micro_f1",
        "pareto_frontier",
        grid=None,
        zero_division=0.0,
        data=("counts", counts),
        **PAPER,
    )
    _case(
        c,
        "evaluate_micro_f1",
        "evaluate",
        theta=0.42,
        zero_division=0.0,
        data=("counts", counts),
        **PAPER,
    )
    return c


def metrics_cases() -> CaseList:
    rng = np.random.default_rng(404)
    c = CaseList()
    p = rng.random(400)
    y = (rng.random(400) < p**1.3).astype(float)
    py = {"p": fl(p), "y": fl(y)}
    for strategy in ("uniform", "quantile"):
        for n_bins in (1, 10, 15) if strategy == "uniform" else (5, 10):
            _case(
                c,
                f"ece_{strategy}_{n_bins}",
                "ece",
                data=("py", py),
                n_bins=n_bins,
                strategy=strategy,
                sample_weight=None,
            )
        _case(
            c,
            f"table_{strategy}_10",
            "reliability_table",
            data=("py", py),
            n_bins=10,
            strategy=strategy,
            sample_weight=None,
        )
    pe = np.round(rng.random(300), 1)
    ye = (rng.random(300) < pe).astype(float)
    edges = {"p": fl(pe), "y": fl(ye)}
    for strategy in ("uniform", "quantile"):
        _case(
            c,
            f"ece_on_edges_{strategy}",
            "ece",
            data=("edges", edges),
            n_bins=10,
            strategy=strategy,
            sample_weight=None,
        )
        _case(
            c,
            f"table_on_edges_{strategy}",
            "reliability_table",
            p=fl(pe),
            y=fl(ye),
            n_bins=10,
            strategy=strategy,
            sample_weight=None,
        )
    w = rng.random(400) * 2
    w[rng.random(400) < 0.1] = 0.0
    pyw = {"p": fl(p), "y": fl(y), "sample_weight": fl(w)}
    for strategy in ("uniform", "quantile"):
        _case(
            c,
            f"ece_weighted_{strategy}",
            "ece",
            data=("pyw", pyw),
            n_bins=10,
            strategy=strategy,
        )
        _case(
            c,
            f"table_weighted_{strategy}",
            "reliability_table",
            data=("pyw", pyw),
            n_bins=10,
            strategy=strategy,
        )
    _case(
        c,
        "ece_docstring",
        "ece",
        p=[0.25] * 4,
        y=[1, 0, 0, 0],
        n_bins=10,
        strategy="uniform",
        sample_weight=None,
    )
    _case(
        c,
        "table_docstring",
        "reliability_table",
        p=[0.05, 0.15, 0.15, 0.95],
        y=[0, 0, 1, 1],
        n_bins=10,
        strategy="uniform",
        sample_weight=None,
    )
    _case(
        c,
        "table_constant_quantile",
        "reliability_table",
        p=[0.3] * 5,
        y=[1, 0, 0, 0, 0],
        n_bins=10,
        strategy="quantile",
        sample_weight=None,
    )
    _case(
        c,
        "ece_constant_quantile",
        "ece",
        p=[0.3] * 5,
        y=[1, 0, 0, 0, 0],
        n_bins=10,
        strategy="quantile",
        sample_weight=None,
    )
    _case(
        c,
        "table_endpoints",
        "reliability_table",
        p=[0.0, 0.0, 1.0, 1.0, 0.5],
        y=[0, 1, 1, 1, 0],
        n_bins=10,
        strategy="uniform",
        sample_weight=None,
    )
    _case(
        c,
        "table_soft_outcomes",
        "reliability_table",
        p=fl(rng.random(60)),
        y=fl(rng.random(60)),
        n_bins=4,
        strategy="quantile",
        sample_weight=None,
    )
    _case(
        c,
        "ece_error_empty",
        "ece",
        p=[],
        y=[],
        n_bins=10,
        strategy="uniform",
        sample_weight=None,
    )
    _case(
        c,
        "ece_error_p_above_one",
        "ece",
        p=[1.5],
        y=[1],
        n_bins=10,
        strategy="uniform",
        sample_weight=None,
    )
    _case(
        c,
        "ece_error_y_above_one",
        "ece",
        p=[0.5],
        y=[2],
        n_bins=10,
        strategy="uniform",
        sample_weight=None,
    )
    _case(
        c,
        "ece_error_nan",
        "ece",
        p=[math.nan],
        y=[1],
        n_bins=10,
        strategy="uniform",
        sample_weight=None,
    )
    _case(
        c,
        "ece_error_zero_bins",
        "ece",
        p=[0.5],
        y=[1],
        n_bins=0,
        strategy="uniform",
        sample_weight=None,
    )
    _case(
        c,
        "ece_error_length",
        "ece",
        p=[0.5, 0.2],
        y=[1],
        n_bins=10,
        strategy="uniform",
        sample_weight=None,
    )
    _case(
        c,
        "ece_error_zero_weights",
        "ece",
        p=[0.5, 0.2],
        y=[1, 0],
        n_bins=10,
        strategy="uniform",
        sample_weight=[0.0, 0.0],
    )
    _case(
        c,
        "ece_error_negative_weight",
        "ece",
        p=[0.5, 0.2],
        y=[1, 0],
        n_bins=10,
        strategy="uniform",
        sample_weight=[1.0, -1.0],
    )

    _case(c, "brier_random", "brier_score", data=("py", py), sample_weight=None)
    _case(c, "brier_weighted", "brier_score", data=("pyw", pyw))
    _case(
        c,
        "brier_simple",
        "brier_score",
        p=[0.0, 1.0, 0.5],
        y=[0, 0, 1],
        sample_weight=None,
    )
    _case(c, "brier_error_y", "brier_score", p=[0.5], y=[-1], sample_weight=None)

    _case(c, "micro_f1_docstring", "micro_f1", tp=3.0, fp=1.0, fn=1.0, zero_division=0.0)
    _case(c, "micro_f1_empty_zero", "micro_f1", tp=0.0, fp=0.0, fn=0.0, zero_division=0.0)
    _case(c, "micro_f1_empty_one", "micro_f1", tp=0.0, fp=0.0, fn=0.0, zero_division=1.0)
    _case(c, "micro_f1_thirds", "micro_f1", tp=1.0, fp=1.0, fn=2.0, zero_division=0.0)
    _case(
        c,
        "micro_f1_error_negative",
        "micro_f1",
        tp=-1.0,
        fp=0.0,
        fn=0.0,
        zero_division=0.0,
    )
    counts = _counts(rng, 40, rng.random(40))
    masks = [[False] * 40, [True] * 40] + [(rng.random(40) < q).tolist() for q in (0.2, 0.5, 0.8)]
    _case(
        c,
        "routed_micro_f1_masks",
        "routed_micro_f1",
        zero_division=0.0,
        masks=masks,
        data=("counts", counts),
    )
    _case(
        c,
        "routed_micro_f1_docstring",
        "routed_micro_f1",
        small_counts=[[1, 0, 1], [2, 0, 0]],
        large_counts=[[2, 0, 0], [2, 0, 0]],
        zero_division=0.0,
        masks=[[False, False], [True, False], [False, True], [True, True]],
    )
    _case(
        c,
        "routed_micro_f1_all_empty",
        "routed_micro_f1",
        small_counts=[[0, 0, 0]],
        large_counts=[[0, 0, 0]],
        zero_division=1.0,
        masks=[[False], [True]],
    )
    return c


def _fit_router_doc(
    seed: int,
    cost_model: str,
    c_large: float,
    grid_step: float,
    tau: float | None,
    budget: float | None,
) -> str:
    """JSON text of a router fitted and saved by the Python package."""
    rng = np.random.default_rng(seed)
    n = 1200
    u = rng.random(n)
    small = (rng.random(n) >= u**1.5).astype(float)
    large = (rng.random(n) >= 0.05).astype(float)
    router = UCCIRouter(1.0, c_large, cost_model, grid_step).calibrate(u[:400], 1.0 - small[:400])
    if tau is not None:
        router.choose_threshold(u[400:800], small[400:800], large[400:800], tau)
    else:
        assert budget is not None
        router.choose_threshold_for_budget(u[400:800], small[400:800], large[400:800], budget)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "router.json"
        router.save(path)
        return path.read_text(encoding="utf-8")


def router_cases() -> CaseList:
    rng = np.random.default_rng(505)
    c = CaseList()
    queries = [
        -1.0,
        0.0,
        0.01,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.99,
        1.0,
        2.0,
    ]
    queries += fl(rng.random(20))
    saved = {
        "saved_routing_tau": _fit_router_doc(1, "routing", 3.02, 0.005, 0.85, None),
        "saved_sequential_step001": _fit_router_doc(2, "sequential", 5.0, 0.01, 0.8, None),
        "saved_budget": _fit_router_doc(3, "routing", 3.02, 0.005, None, 2.0),
    }
    for cid, text in saved.items():
        _case(c, cid, "router_load", json_text=text, queries=queries)

    base = json.loads(saved["saved_routing_tau"])

    def variant(cid: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        doc = json.loads(json.dumps(base))
        mutate(doc)
        _case(c, cid, "router_load", json_text=json.dumps(doc), queries=queries[:8])

    def set_(key: str, value: Any) -> Callable[[dict[str, Any]], None]:
        def f(d: dict[str, Any]) -> None:
            d[key] = value

        return f

    def drop(key: str) -> Callable[[dict[str, Any]], None]:
        def f(d: dict[str, Any]) -> None:
            d.pop(key, None)

        return f

    def cal(x: Any, y: Any) -> Callable[[dict[str, Any]], None]:
        return set_("calibrator", {"x": x, "y": y})

    def drop_optional(d: dict[str, Any]) -> None:
        for k in ("tau", "grid_step", "created_by"):
            d.pop(k)

    variant("minimal_no_optional_fields", drop_optional)
    variant(
        "optional_fields_null",
        lambda d: d.update(tau=None, grid_step=None, created_by=None),
    )
    variant(
        "unknown_keys_ignored",
        lambda d: d.update(
            future={"nested": [1, 2, {"a": None}]},
            notes="written by hand",
            calibrator={**d["calibrator"], "method": "isotonic"},
        ),
    )
    variant("integer_numbers", lambda d: d.update(theta=0, c_small=1, c_large=3))
    variant("created_by_not_string", set_("created_by", 7))
    variant("single_knot", cal([0.4], [0.3]))
    variant("theta_outside_unit_interval", set_("theta", 1.5))
    variant("grid_step_tenth", set_("grid_step", 0.1))
    variant("error_format", set_("format", "ucci"))
    variant("error_format_missing", drop("format"))
    variant("error_version_2", set_("version", 2))
    variant("error_version_float", set_("version", 1.0))
    variant("error_version_string", set_("version", "1"))
    variant("error_version_bool", set_("version", True))
    variant("error_version_missing", drop("version"))
    variant("error_calibrator_list", set_("calibrator", [1, 2]))
    variant("error_calibrator_missing_y", set_("calibrator", {"x": [0.1]}))
    variant("error_calibrator_length", cal([0.1, 0.2], [0.1]))
    variant("error_calibrator_empty", cal([], []))
    variant("error_x_decreasing", cal([0.2, 0.1], [0.1, 0.2]))
    variant("error_x_duplicate", cal([0.1, 0.1], [0.1, 0.2]))
    variant("error_y_decreasing", cal([0.1, 0.2], [0.3, 0.2]))
    variant("error_y_above_one", cal([0.1, 0.2], [0.3, 1.2]))
    variant("error_y_negative", cal([0.1, 0.2], [-0.1, 0.2]))
    variant("error_x_bool", cal([0.1, True], [0.1, 0.2]))
    variant("error_x_string", cal("0.1", [0.1]))
    variant("error_theta_missing", drop("theta"))
    variant("error_theta_string", set_("theta", "0.3"))
    variant("error_theta_null", set_("theta", None))
    variant("error_theta_bool", set_("theta", False))
    variant("error_c_small_missing", drop("c_small"))
    variant("error_c_small_zero", set_("c_small", 0.0))
    variant("error_c_large_negative", set_("c_large", -1.0))
    variant("error_cost_model", set_("cost_model", "latency"))
    variant("error_cost_model_missing", drop("cost_model"))
    variant("error_tau_string", set_("tau", "high"))
    variant("error_grid_step_zero", set_("grid_step", 0.0))
    variant("error_grid_step_above_one", set_("grid_step", 1.5))
    variant("error_grid_step_not_divisor", set_("grid_step", 0.3))
    _case(c, "error_not_object", "router_load", json_text="[1, 2, 3]", queries=[])
    _case(c, "error_not_json", "router_load", json_text="{not json", queries=[])
    nan_text = saved["saved_routing_tau"].replace('"theta": ', '"theta": NaN, "x_theta": ', 1)
    _case(c, "error_nan_literal", "router_load", json_text=nan_text, queries=[])
    return c


def pipeline_cases() -> CaseList:
    rng = np.random.default_rng(606)
    c = CaseList()
    n = 2000
    u = rng.random(n)
    small = (rng.random(n) >= u**2).astype(float)
    large = (rng.random(n) >= 0.04).astype(float)
    cal, val = slice(0, 600), slice(600, 1000)
    test = slice(1000, n)
    split = {
        "u_cal": fl(u[cal]),
        "e_cal": fl(1.0 - small[cal]),
        "u_val": fl(u[val]),
        "small_val": fl(small[val]),
        "large_val": fl(large[val]),
        "u_test": fl(u[test]),
        "small_test": fl(small[test]),
        "large_test": fl(large[test]),
    }
    _case(
        c,
        "protocol_paper_costs_tau_0.9",
        "router_fit",
        tau=0.9,
        budget=None,
        c_small=1.0,
        c_large=3.02,
        cost_model="routing",
        grid_step=0.005,
        data=("split", split),
    )
    _case(
        c,
        "protocol_sequential_tau_0.85_step_0.01",
        "router_fit",
        tau=0.85,
        budget=None,
        c_small=1.0,
        c_large=5.0,
        cost_model="sequential",
        grid_step=0.01,
        data=("split", split),
    )
    _case(
        c,
        "protocol_budget_2.0",
        "router_fit",
        tau=None,
        budget=2.0,
        c_small=1.0,
        c_large=3.02,
        cost_model="routing",
        grid_step=0.005,
        data=("split", split),
    )
    w = np.round(rng.random(600) * 4) / 2
    _case(
        c,
        "protocol_weighted_calibration",
        "router_fit",
        tau=0.9,
        budget=None,
        c_small=1.0,
        c_large=3.02,
        cost_model="routing",
        grid_step=0.005,
        w_cal=fl(w),
        data=("split", split),
    )
    _case(
        c,
        "protocol_error_infeasible",
        "router_fit",
        tau=0.999,
        budget=None,
        c_small=1.0,
        c_large=3.02,
        cost_model="routing",
        grid_step=0.005,
        data=("split", split),
    )
    return c


FILES: dict[str, Callable[[], CaseList]] = {
    "signal.json": signal_cases,
    "calibration.json": calibration_cases,
    "policy.json": policy_cases,
    "metrics.json": metrics_cases,
    "router.json": router_cases,
    "pipeline.json": pipeline_cases,
}

NOTES = {
    "signal.json": "Token-margin uncertainty u(x), paper Section 4.1, Eq. 4.",
    "calibration.json": "Weighted PAV and the isotonic calibrator, paper Section 4.2.",
    "policy.json": "Threshold policy and selection, paper Section 4.3 (Eqs. 6 and 7), "
    "Table 2 (budget form) and Table 3 (cost models).",
    "metrics.json": "ECE, reliability tables, Brier score, micro-F1; paper Section 6.",
    "router.json": "The ucci-router JSON format: files saved by the Python package, "
    "valid variants and documents that must be rejected.",
    "pipeline.json": "The three-step protocol of paper Section 6.1 on synthetic data.",
}


def build_cases(name: str) -> CaseList:
    """The cases (inputs only) of one golden file."""
    return FILES[name]()


def build_file(name: str) -> dict[str, Any]:
    """One golden file: header plus every case with its computed outputs.

    A case id contains "error" exactly when the Python core raises on it; this
    guards against a typo in a case silently turning it into an error case.
    """
    built = build_cases(name)
    cases = []
    for case in built:
        out = dict(case)
        out.update(compute(case["fn"], resolve(case["input"], built.data)))
        if ("error" in out) != ("error" in case["id"]):
            detail = out.get("error", {}).get("message", "no error")
            raise AssertionError(f"{name}: case {case['id']!r} gave {detail!r}")
        cases.append(out)
    return {
        "schema": SCHEMA,
        "generator": "tools/make_golden.py",
        "module": name[: -len(".json")],
        "note": NOTES[name],
        "tolerance": TOLERANCE,
        "data": built.data,
        "cases": cases,
    }


def dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"


#: Router file shipped with the Rust crate's example (``cargo run --example route``).
EXAMPLE_ROUTER = ROOT / "rust" / "examples" / "router.json"


def example_router_text() -> str:
    """A router saved by the Python package: synthetic data, tau = 0.9, paper costs."""
    return _fit_router_doc(7, "routing", 3.02, 0.005, 0.9, None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if any golden file is missing or differs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=GOLDEN_DIR,
        help="output directory (default: tests/golden)",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name in FILES:
        doc = build_file(name)
        outputs.append((args.out / name, dumps(doc), len(doc["cases"])))
    outputs.append((EXAMPLE_ROUTER, example_router_text(), 0))
    stale = []
    for path, text, n in outputs:
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(str(path))
            continue
        path.write_text(text, encoding="utf-8")
        shown = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        print(f"wrote {shown}" + (f" ({n} cases)" if n else ""))
    if stale:
        print("stale golden files: " + ", ".join(stale), file=sys.stderr)
        return 1
    print(f"ucci {ucci.__version__}, numpy {np.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
