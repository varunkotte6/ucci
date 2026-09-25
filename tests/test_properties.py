"""Property-based tests over the public API of ucci (hypothesis).

Each test states a fact that must hold for every input: a mathematical
property of the paper's method (Varun Kotte, "UCCI: Calibrated Uncertainty
for Cost-Optimal LLM Cascade Routing", arXiv:2605.18796 v1) or a documented
contract of a public function. Every property runs on at least 300
generated examples with no deadline.

Covered:

* PAV and :class:`ucci.IsotonicCalibrator` (Section 4.2): monotone output,
  idempotence, weighted block means, the optimality conditions of isotonic
  regression, permutation invariance, zero-weight and replication
  invariance, predictions within the fitted range, and parity with
  scikit-learn's ``IsotonicRegression(out_of_bounds="clip")`` including ties,
  near-ties and weights.
* The signal u(x) (Section 4.1, Eq. 4): range, pair-order invariance,
  agreement of the probability, log-probability, batch and payload paths,
  and agreement of the serving adapters with the core.
* The policy (Section 4.3, Eq. 6 and 7; Section 5, Theorem 1):
  :func:`ucci.select_threshold` is feasible, minimal-cost and follows its
  tie-break rule against a brute-force search over the same grid; the
  budget form is feasible and maximal; cost is monotone in theta; the
  chosen theta does not depend on the costs; and Theorem 1 holds against a
  brute-force search over all escalation subsets of small validation sets.
* Metrics: ECE range and exact zero on calibrated forecasts, reliability
  counts, weights as replication, bootstrap intervals, micro-F1.
* The router JSON round trip, bit for bit.
* Baselines (Section 6.1, Appendix B.4): split conformal against the order
  statistic definition, the raw-score routers against exhaustive search,
  the oracle against every escalation subset, temperature scaling and Platt
  against their likelihood optimum, and :func:`ucci.baselines.compare_routers`
  against running each router by hand.
* The ``ucci`` CLI (fit, evaluate, route, report) against the Python API on
  random logged traffic, including the documented random split rule.
* Extensions (not in the paper): the online recalibrating router and the
  cascade helper.

Findings from this suite that the code does not yet satisfy are noted where
a property had to be relaxed .

The module is skipped when hypothesis is not installed.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import tempfile
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, assume, event, given, settings
from hypothesis import strategies as st

from ucci import (
    DEFAULT_GRID,
    InfeasibleTargetError,
    IsotonicCalibrator,
    UCCIRouter,
    batch_uncertainty,
    bootstrap_ci,
    brier_score,
    ece,
    escalate,
    evaluate,
    from_openai_logprobs,
    from_vllm_logprobs,
    make_grid,
    margins_from_top2,
    micro_f1,
    pareto_frontier,
    pav,
    policy_accuracy,
    policy_cost,
    reliability_table,
    routed_micro_f1,
    select_threshold,
    select_threshold_for_budget,
    token_margin_uncertainty,
    top2_from_logprobs,
    uncertainty_from_logprobs,
    uncertainty_from_margins,
    uncertainty_from_probs,
)
from ucci import io as ucci_io

# Every property: at least 300 examples, no per-example deadline.
PROPERTY = settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

UNIT = st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)
#: Multiples of 1/64: sums of up to a few hundred of them are exact in float64,
#: so accuracies computed along different summation paths are bit-identical.
DYADIC = st.integers(0, 64).map(lambda k: k / 64.0)
COARSE = st.integers(0, 20).map(lambda k: k / 20.0)
#: Values within scikit-learn's tie tolerance (1e-15) of each other.
NEAR_TIE = st.one_of(
    st.integers(0, 12).map(lambda k: 0.5 + k * 2.0**-53),
    st.integers(0, 12).map(lambda k: k * 2.5e-16),
    st.integers(0, 12).map(lambda k: 1.0 - k * 2.0**-53),
)
U_VALUE = st.one_of(UNIT, COARSE, NEAR_TIE)
U_NO_NEAR_TIES = st.one_of(UNIT, COARSE)
LABEL = st.one_of(st.sampled_from([0.0, 1.0]), UNIT)
WEIGHT = st.one_of(
    st.floats(1e-3, 1e3, allow_nan=False, allow_infinity=False), st.integers(1, 5).map(float)
)
COST_MODEL = st.sampled_from(["routing", "sequential"])
BINARY = st.sampled_from([0.0, 1.0])


@st.composite
def calibration_data(
    draw: Any, max_n: int = 60, u_values: Any = U_VALUE, weighted: bool | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(u, e, w) with ties, near-ties, soft labels and optional weights."""
    n = draw(st.integers(1, max_n))
    u = np.array(draw(st.lists(u_values, min_size=n, max_size=n)), dtype=np.float64)
    e = np.array(draw(st.lists(LABEL, min_size=n, max_size=n)), dtype=np.float64)
    use_w = draw(st.booleans()) if weighted is None else weighted
    w = np.array(draw(st.lists(WEIGHT, min_size=n, max_size=n))) if use_w else None
    return u, e, w


@st.composite
def pav_data(draw: Any, max_n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """(y, w) for PAV: arbitrary finite targets and strictly positive weights."""
    n = draw(st.integers(1, max_n))
    y = np.array(
        draw(
            st.lists(
                st.one_of(
                    st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                    st.integers(-5, 5).map(float),
                ),
                min_size=n,
                max_size=n,
            )
        )
    )
    w = np.array(draw(st.lists(WEIGHT, min_size=n, max_size=n)))
    return y, w


@st.composite
def routing_problem(
    draw: Any, max_n: int = 80, scores: Any = BINARY
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validation p_hat (with ties and grid hits), small and large scores."""
    n = draw(st.integers(1, max_n))
    p_val = st.one_of(UNIT, COARSE, st.integers(0, 200).map(lambda k: DEFAULT_GRID[k]))
    p = np.array(draw(st.lists(p_val, min_size=n, max_size=n)), dtype=np.float64)
    s = np.array(draw(st.lists(scores, min_size=n, max_size=n)), dtype=np.float64)
    lg = np.array(draw(st.lists(scores, min_size=n, max_size=n)), dtype=np.float64)
    return p, s, lg


@st.composite
def threshold_grid(draw: Any) -> np.ndarray:
    """A grid in [0, 1]: the default, a coarse one, or a random unsorted one with repeats."""
    kind = draw(st.sampled_from(["default", "step", "random"]))
    if kind == "default":
        return np.asarray(DEFAULT_GRID)
    if kind == "step":
        return make_grid(draw(st.sampled_from([0.5, 0.25, 0.1, 0.05, 0.01])))
    vals = draw(st.lists(st.one_of(UNIT, COARSE), min_size=1, max_size=30))
    return np.array(vals + vals[: len(vals) // 3], dtype=np.float64)


@st.composite
def costs(draw: Any) -> tuple[float, float]:
    """(c_small, c_large) with c_large > c_small by a non-degenerate margin."""
    cs = draw(st.sampled_from([1.0, 0.5, 2.0, 0.125]))
    ratio = draw(st.sampled_from([3.02, 1.5, 5.0, 10.0, 1.125, 64.0]))
    return cs, cs * ratio


def _cost(rate: float, cs: float, cl: float, model: str) -> float:
    """The documented cost formula of :func:`ucci.policy_cost`."""
    return cs * (1.0 - rate) + cl * rate if model == "routing" else cs + cl * rate


def _grid_rows(
    p: np.ndarray,
    s: np.ndarray,
    lg: np.ndarray,
    grid: np.ndarray,
    cs: float,
    cl: float,
    model: str,
) -> list[tuple[float, float, float, float]]:
    """(theta, cost, accuracy, rate) of pi_theta for every distinct grid value, by brute force."""
    rows = []
    n = p.size
    for theta in np.unique(grid):
        mask = p > theta
        k = int(mask.sum())
        acc = (float(s[~mask].sum()) + float(lg[mask].sum())) / n
        rows.append((float(theta), _cost(k / n, cs, cl, model), acc, k / n))
    return rows


# ---------------------------------------------------------------------------
# PAV (Section 4.2)
# ---------------------------------------------------------------------------


def _runs(f: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs [a, b) of exactly equal values."""
    out, a = [], 0
    for i in range(1, f.size + 1):
        if i == f.size or f[i] != f[a]:
            out.append((a, i))
            a = i
    return out


@PROPERTY
@given(data=pav_data())
def test_pav_monotone_idempotent_and_within_range(data: Any) -> None:
    y, w = data
    f = pav(y, w)
    assert f.shape == y.shape
    assert bool(np.all(np.diff(f) >= 0.0)), "PAV output must be non-decreasing"
    np.testing.assert_array_equal(pav(f, w), f)  # a monotone sequence is a fixed point
    tol = 1e-12 * max(1.0, float(np.abs(y).max()))
    assert f.min() >= y.min() - tol and f.max() <= y.max() + tol
    # The weighted mean is preserved (Barlow et al., 1972).
    assert float(np.sum(w * f)) == pytest.approx(float(np.sum(w * y)), rel=1e-12, abs=tol * y.size)


@PROPERTY
@given(y=st.lists(st.floats(-1e3, 1e3, allow_nan=False), min_size=1, max_size=60))
def test_pav_leaves_sorted_input_unchanged(y: list[float]) -> None:
    ys = np.sort(np.array(y))
    np.testing.assert_array_equal(pav(ys), ys)


@PROPERTY
@given(data=pav_data())
def test_pav_runs_are_weighted_block_means(data: Any) -> None:
    y, w = data
    f = pav(y, w)
    scale = max(1.0, float(np.abs(y).max()))
    for a, b in _runs(f):
        mean = float(np.sum(w[a:b] * y[a:b]) / np.sum(w[a:b]))
        assert f[a] == pytest.approx(mean, rel=1e-12, abs=1e-12 * scale)


@PROPERTY
@given(data=pav_data(), seed=st.integers(0, 2**32 - 1))
def test_pav_satisfies_isotonic_optimality_conditions(data: Any, seed: int) -> None:
    """KKT conditions: blocks cannot be split, and <y - f, g - f>_w <= 0 for monotone g."""
    y, w = data
    f = pav(y, w)
    scale = max(1.0, float(np.abs(y).max()))
    tol = 1e-9 * scale
    for a, b in _runs(f):
        cw = np.cumsum(w[a:b])
        cwy = np.cumsum(w[a:b] * y[a:b])
        prefix_means = cwy[:-1] / cw[:-1]
        # Every proper prefix of a block has mean >= the block mean; otherwise
        # splitting the block would lower the weighted squared error.
        assert bool(np.all(prefix_means >= f[a] - tol))
    rng = np.random.default_rng(seed)
    for _ in range(5):
        g = np.sort(rng.normal(scale=scale, size=y.size))
        inner = float(np.sum(w * (y - f) * (g - f)))
        assert inner <= 1e-9 * scale * scale * max(1.0, float(w.sum()))


@PROPERTY
@given(data=pav_data())
def test_pav_matches_sklearn_isotonic_regression(data: Any) -> None:
    sk = pytest.importorskip("sklearn.isotonic")
    y, w = data
    y0, w0 = y.copy(), w.copy()
    ours = pav(y, w)
    np.testing.assert_array_equal(y, y0)  # pav never modifies its inputs
    np.testing.assert_array_equal(w, w0)
    # Copies: scikit-learn < 1.0 overwrites sample_weight in place.
    ref = sk.isotonic_regression(y.copy(), sample_weight=w.copy())
    scale = max(1.0, float(np.abs(y).max()))
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-11 * scale)


@PROPERTY
@given(
    data=pav_data(),
    a=st.floats(1e-3, 1e3),
    b=st.floats(-1e3, 1e3),
    c=st.floats(1e-3, 1e3),
)
def test_pav_affine_equivariance_and_weight_scale_invariance(
    data: Any, a: float, b: float, c: float
) -> None:
    y, w = data
    f = pav(y, w)
    scale = max(1.0, float(np.abs(y).max()))
    np.testing.assert_allclose(
        pav(a * y + b, w), a * f + b, rtol=1e-9, atol=1e-9 * (a * scale + abs(b))
    )
    np.testing.assert_allclose(pav(y, c * w), f, rtol=1e-9, atol=1e-9 * scale)


# ---------------------------------------------------------------------------
# IsotonicCalibrator (Section 4.2)
# ---------------------------------------------------------------------------


@PROPERTY
@given(data=calibration_data())
def test_calibrator_knots_and_predictions_structure(data: Any) -> None:
    u, e, w = data
    snapshot = (u.copy(), e.copy(), None if w is None else w.copy())
    cal = IsotonicCalibrator().fit(u, e, sample_weight=w)
    np.testing.assert_array_equal(u, snapshot[0])  # fit never modifies its inputs
    np.testing.assert_array_equal(e, snapshot[1])
    if w is not None:
        np.testing.assert_array_equal(w, snapshot[2])
    x, y = cal.x_, cal.y_
    assert x is not None and y is not None
    assert x.size == y.size >= 1
    assert bool(np.all(np.diff(x) > 0.0)) and bool(np.all(np.diff(y) >= 0.0))
    assert set(x.tolist()) <= set(u.tolist()), "knots are calibration u values"
    assert x[0] == u.min() and x[-1] <= u.max()
    assert y.min() >= e.min() - 1e-12 and y.max() <= e.max() + 1e-12
    assert y.min() >= 0.0 and y.max() <= 1.0
    assert cal.n_samples_ == u.size
    # Predictions at the knots are the knot values, exactly.
    np.testing.assert_array_equal(cal.predict(x), y)
    # Clipped outside the calibration range, never NaN.
    assert cal.predict(float(x[0]) - 1.0) == y[0]
    assert cal.predict(float(x[-1]) + 1.0) == y[-1]
    q = np.sort(np.concatenate([np.linspace(-0.5, 1.5, 101), u, np.nextafter(x, np.inf)]))
    pq = cal.predict(q)
    assert bool(np.all(np.isfinite(pq)))
    assert bool(np.all(np.diff(pq) >= 0.0)), "g must be non-decreasing in u"
    assert pq.min() >= y[0] and pq.max() <= y[-1]
    # Scalar in, float out; array in, same shape out.
    assert isinstance(cal.predict(float(u[0])), float)
    assert cal.predict(u.reshape(-1, 1)).shape == (u.size, 1)


@PROPERTY
@given(data=calibration_data(), perm_seed=st.integers(0, 2**32 - 1))
def test_calibrator_is_permutation_invariant(data: Any, perm_seed: int) -> None:
    u, e, w = data
    perm = np.random.default_rng(perm_seed).permutation(u.size)
    a = IsotonicCalibrator().fit(u, e, sample_weight=w)
    b = IsotonicCalibrator().fit(u[perm], e[perm], sample_weight=None if w is None else w[perm])
    np.testing.assert_array_equal(a.x_, b.x_)
    if w is None:
        np.testing.assert_array_equal(a.y_, b.y_)
    else:
        np.testing.assert_allclose(np.asarray(a.y_), np.asarray(b.y_), rtol=0, atol=1e-12)


@PROPERTY
@given(
    data=calibration_data(weighted=True),
    extra=st.lists(st.tuples(U_VALUE, LABEL), min_size=1, max_size=10),
)
def test_calibrator_ignores_zero_weight_points(data: Any, extra: list[tuple[float, float]]) -> None:
    u, e, w = data
    eu = np.array([t[0] for t in extra])
    ee = np.array([t[1] for t in extra])
    base = IsotonicCalibrator().fit(u, e, sample_weight=w)
    padded = IsotonicCalibrator().fit(
        np.concatenate([eu, u]),
        np.concatenate([ee, e]),
        sample_weight=np.concatenate([np.zeros(eu.size), w]),
    )
    np.testing.assert_array_equal(padded.x_, base.x_)
    np.testing.assert_array_equal(padded.y_, base.y_)
    assert padded.n_samples_ == base.n_samples_


@PROPERTY
@given(data=calibration_data(weighted=False), reps=st.lists(st.integers(1, 4), min_size=60))
def test_calibrator_integer_weights_equal_replication(data: Any, reps: list[int]) -> None:
    u, e, _ = data
    r = np.array(reps[: u.size])
    weighted = IsotonicCalibrator().fit(u, e, sample_weight=r.astype(float))
    replicated = IsotonicCalibrator().fit(np.repeat(u, r), np.repeat(e, r))
    np.testing.assert_array_equal(weighted.x_, replicated.x_)
    np.testing.assert_allclose(
        np.asarray(weighted.y_), np.asarray(replicated.y_), rtol=0, atol=1e-12
    )


def _sklearn_iso(u: np.ndarray, e: np.ndarray, w: np.ndarray | None) -> Any:
    sk = pytest.importorskip("sklearn.isotonic")
    return sk.IsotonicRegression(out_of_bounds="clip").fit(
        u.copy(), e.copy(), sample_weight=None if w is None else w.copy()
    )


@PROPERTY
@given(data=calibration_data())
def test_calibrator_predictions_match_sklearn_clip(data: Any) -> None:
    u, e, w = data
    ours = IsotonicCalibrator().fit(u, e, sample_weight=w)
    ref = _sklearn_iso(u, e, w)
    assert ours.x_ is not None
    q = np.concatenate(
        [
            np.linspace(-0.5, 1.5, 81),
            u,
            ours.x_,
            np.nextafter(ours.x_, np.inf),
            np.nextafter(ours.x_, -np.inf),
        ]
    )
    np.testing.assert_allclose(ours.predict(q), ref.predict(q), rtol=0, atol=1e-12)


@PROPERTY
@given(data=calibration_data())
def test_calibrator_knots_match_sklearn_thresholds(data: Any) -> None:
    """``x_`` and ``y_`` are documented as scikit-learn's X_thresholds_ and y_thresholds_."""
    u, e, w = data
    ours = IsotonicCalibrator().fit(u, e, sample_weight=w)
    ref = _sklearn_iso(u, e, w)
    np.testing.assert_array_equal(ours.x_, ref.X_thresholds_)
    np.testing.assert_allclose(np.asarray(ours.y_), ref.y_thresholds_, rtol=0, atol=1e-12)


@PROPERTY
@given(data=calibration_data(u_values=U_NO_NEAR_TIES))
def test_calibrator_is_calibrated_in_sample(data: Any) -> None:
    """On its own data, each level set of g has weighted error rate equal to its value.

    Points closer than 1e-15 are pooled at the first value of their group (as
    in scikit-learn) and then predicted by interpolation, so such near-ties are
    excluded here; the sklearn parity tests cover them.
    """
    u, e, w = data
    distinct = np.unique(u)
    assume(distinct.size < 2 or float(np.diff(distinct).min()) >= 1e-12)
    ww = np.ones(u.size) if w is None else w
    cal = IsotonicCalibrator().fit(u, e, sample_weight=w)
    p = cal.predict(u)
    for v in np.unique(p):
        sel = p == v
        assert float(np.sum(ww[sel] * e[sel]) / np.sum(ww[sel])) == pytest.approx(
            float(v), abs=1e-12
        )


@PROPERTY
@given(data=calibration_data())
def test_calibrator_dict_and_json_round_trip_exact(data: Any) -> None:
    u, e, w = data
    cal = IsotonicCalibrator().fit(u, e, sample_weight=w)
    back = IsotonicCalibrator.from_dict(json.loads(json.dumps(cal.to_dict())))
    np.testing.assert_array_equal(back.x_, cal.x_)
    np.testing.assert_array_equal(back.y_, cal.y_)
    q = np.concatenate([np.linspace(-0.5, 1.5, 101), u])
    np.testing.assert_array_equal(back.predict(q), cal.predict(q))


# ---------------------------------------------------------------------------
# Signal u(x) (Section 4.1, Eq. 4)
# ---------------------------------------------------------------------------


@st.composite
def top2_pairs(draw: Any, max_t: int = 40, positive: bool = False) -> list[tuple[float, float]]:
    """Per-token (p1, p2) with p1 >= p2 and p1 + p2 <= 1 (one distribution)."""
    lo = 1e-300 if positive else 0.0
    t = draw(st.integers(1, max_t))
    out = []
    for _ in range(t):
        p1 = draw(st.one_of(st.floats(lo, 1.0), st.sampled_from([1.0, 0.5])))
        p2 = draw(st.floats(lo, max(lo, min(p1, 1.0 - p1))))
        out.append((p1, min(p2, p1)))
    return out


@PROPERTY
@given(pairs=top2_pairs(), flips=st.lists(st.booleans(), min_size=40, max_size=40))
def test_u_in_unit_interval_and_invariant_to_pair_order(
    pairs: list[tuple[float, float]], flips: list[bool]
) -> None:
    u = token_margin_uncertainty(pairs)
    assert 0.0 <= u <= 1.0
    swapped = [(b, a) if f else (a, b) for (a, b), f in zip(pairs, flips)]
    assert token_margin_uncertainty(swapped) == u
    p1 = np.array([a for a, _ in swapped])
    p2 = np.array([b for _, b in swapped])
    assert uncertainty_from_probs(p1, p2) == u
    assert uncertainty_from_probs(p2, p1) == u
    assert uncertainty_from_margins(margins_from_top2(pairs)) == u
    m = np.array(margins_from_top2(pairs))
    assert bool(np.all((m >= 0.0) & (m <= 1.0)))
    assert u == pytest.approx(1.0 - float(np.mean(m)), abs=0)


@PROPERTY
@given(t=st.integers(1, 30), p=st.floats(0.0, 0.5))
def test_u_extremes(t: int, p: float) -> None:
    assert token_margin_uncertainty([(1.0, 0.0)] * t) == 0.0
    assert token_margin_uncertainty([(p, p)] * t) == 1.0


@PROPERTY
@given(pairs=top2_pairs(positive=True))
def test_u_logprob_path_agrees_with_probability_path(pairs: list[tuple[float, float]]) -> None:
    p1 = np.array([a for a, _ in pairs])
    p2 = np.array([b for _, b in pairs])
    lp1, lp2 = np.log(p1), np.log(p2)
    u_lp = uncertainty_from_logprobs(lp1, lp2)
    assert u_lp == uncertainty_from_probs(np.exp(lp1), np.exp(lp2))
    assert u_lp == pytest.approx(uncertainty_from_probs(p1, p2), abs=1e-12)
    assert batch_uncertainty([lp1], [lp2], logprobs=True)[0] == pytest.approx(u_lp, abs=1e-15)


@st.composite
def candidate_lists(draw: Any, max_t: int = 20) -> list[list[float]]:
    """Per-token top-k candidate log-probs (k >= 2) of one distribution, shuffled."""
    t = draw(st.integers(1, max_t))
    out = []
    for _ in range(t):
        k = draw(st.integers(2, 5))
        raw = draw(st.lists(st.floats(1e-6, 1.0), min_size=k + 1, max_size=k + 1))
        probs = np.array(raw) / float(np.sum(raw))  # a full distribution over k + 1 tokens
        cands = [float(np.log(v)) for v in probs[:k]]
        order = draw(st.permutations(list(range(k))))
        out.append([min(cands[i], 0.0) for i in order])
    return out


@PROPERTY
@given(cands=candidate_lists())
def test_top2_from_logprobs_is_order_free_and_matches_sorted_top2(cands: list[list[float]]) -> None:
    top2 = top2_from_logprobs(cands)
    for (p1, p2), c in zip(top2, cands):
        s = sorted(c, reverse=True)
        assert (p1, p2) == (math.exp(s[0]), math.exp(s[1]))
    assert top2_from_logprobs([sorted(c) for c in cands]) == top2


@st.composite
def ragged_batch(draw: Any) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n = draw(st.integers(1, 12))
    a, b = [], []
    for _ in range(n):
        pairs = draw(top2_pairs(max_t=15, positive=True))
        a.append(np.array([x for x, _ in pairs]))
        b.append(np.array([y for _, y in pairs]))
    return a, b


@PROPERTY
@given(batch=ragged_batch(), junk=st.sampled_from([np.nan, -7.0, 3.0]))
def test_batch_uncertainty_ragged_padded_and_single_agree(batch: Any, junk: float) -> None:
    a, b = batch
    ragged = batch_uncertainty(a, b)
    lens = np.array([x.size for x in a])
    t_max = int(lens.max())
    pa = np.full((len(a), t_max), junk)
    pb = np.full((len(a), t_max), junk)
    for i, (x, y) in enumerate(zip(a, b)):
        pa[i, : x.size] = x
        pb[i, : y.size] = y
    np.testing.assert_array_equal(batch_uncertainty(pa, pb, lengths=lens), ragged)
    single = np.array([uncertainty_from_probs(x, y) for x, y in zip(a, b)])
    np.testing.assert_allclose(ragged, single, rtol=0, atol=1e-15)
    lp = batch_uncertainty([np.log(x) for x in a], [np.log(y) for y in b], logprobs=True)
    np.testing.assert_allclose(lp, ragged, rtol=0, atol=1e-12)
    # Eq. 4 over a concatenation is the token-weighted mean of the parts.
    whole = uncertainty_from_probs(np.concatenate(a), np.concatenate(b))
    assert whole == pytest.approx(float(np.sum(ragged * lens) / lens.sum()), abs=1e-12)


@PROPERTY
@given(
    batch=ragged_batch(),
    empties=st.lists(st.booleans(), min_size=12, max_size=12),
    fill=UNIT,
)
def test_batch_uncertainty_empty_value_marks_only_empty_rows(
    batch: Any, empties: list[bool], fill: float
) -> None:
    """Extension (not in the paper): rows with no content tokens get empty_value."""
    a, b = batch
    lens = np.array([0 if emp else x.size for x, emp in zip(a, empties)])
    t_max = max(1, int(max(x.size for x in a)))
    pa = np.full((len(a), t_max), np.nan)
    pb = np.full((len(a), t_max), np.nan)
    for i, (x, y) in enumerate(zip(a, b)):
        pa[i, : x.size] = x
        pb[i, : y.size] = y
    got = batch_uncertainty(pa, pb, lengths=lens, empty_value=fill)
    for i, (x, y) in enumerate(zip(a, b)):
        if lens[i] == 0:
            assert got[i] == fill
        else:
            assert got[i] == pytest.approx(uncertainty_from_probs(x, y), abs=1e-15)
    if (lens == 0).any():
        with pytest.raises(ValueError):
            batch_uncertainty(pa, pb, lengths=lens)


# ---------------------------------------------------------------------------
# Serving adapters: payload paths, streaming, stop tokens, ablation signals
# ---------------------------------------------------------------------------


class _Obj:
    """Attribute-style payload object (like the OpenAI SDK's pydantic models)."""

    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


@PROPERTY
@given(cands=candidate_lists(), as_objects=st.booleans())
def test_payload_paths_and_adapters_agree_with_core(
    cands: list[list[float]], as_objects: bool
) -> None:
    expected = token_margin_uncertainty(top2_from_logprobs(cands))
    wrap = (lambda **kw: _Obj(**kw)) if as_objects else (lambda **kw: dict(**kw))
    content = [
        wrap(
            token=f"t{t}",
            logprob=max(c),
            top_logprobs=[wrap(token=f"c{j}", logprob=lp) for j, lp in enumerate(c)],
        )
        for t, c in enumerate(cands)
    ]
    assert from_openai_logprobs(content) == expected
    vllm_positions = [
        {j: (_Obj(logprob=lp) if as_objects else lp) for j, lp in enumerate(c)} for c in cands
    ]
    assert from_vllm_logprobs(vllm_positions) == expected
    from ucci.integrations import openai as oa
    from ucci.integrations import vllm as vl

    assert oa.u_from_chat_completion(content) == expected
    response = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "length",
                "logprobs": {"content": content},
            }
        ]
    }
    assert oa.u_from_chat_completion(response, drop_stop_token=True) == expected
    assert vl.u_from_vllm(vllm_positions) == expected
    sig = vl.signals_from_vllm(vllm_positions)
    assert sig.n_tokens == len(cands)
    assert sig.margins == tuple(margins_from_top2(top2_from_logprobs(cands)))


@PROPERTY
@given(
    cands=candidate_lists(),
    cuts=st.lists(st.integers(0, 20), max_size=6),
    finish=st.sampled_from(["stop", "length"]),
)
def test_adapters_streaming_and_stop_token_agree_with_core(
    cands: list[list[float]], cuts: list[int], finish: str
) -> None:
    from ucci.integrations import llamacpp as lc
    from ucci.integrations import openai as oa

    entries = [
        {
            "token": f"t{t}",
            "logprob": max(c),
            "top_logprobs": [{"token": f"c{j}", "logprob": lp} for j, lp in enumerate(c)],
        }
        for t, c in enumerate(cands)
    ]
    full = token_margin_uncertainty(top2_from_logprobs(cands))
    bounds = sorted({0, len(entries), *(c for c in cuts if c <= len(entries))})
    chunks: list[dict[str, Any]] = [
        {"choices": [{"index": 0, "logprobs": {"content": entries[a:b]}}]}
        for a, b in zip(bounds, bounds[1:])
    ]
    chunks.append({"choices": [{"index": 0, "finish_reason": finish, "logprobs": None}]})
    chunks.append({"choices": [], "usage": {"completion_tokens": len(entries)}})
    assert oa.u_from_chat_completion_chunks(chunks) == full
    response = {
        "choices": [{"index": 0, "finish_reason": finish, "logprobs": {"content": entries}}]
    }
    llama = {
        "completion_probabilities": entries,
        "stop_type": "eos" if finish == "stop" else "limit",
    }
    events = [{"completion_probabilities": entries[a:b]} for a, b in zip(bounds, bounds[1:])] + [
        {"stop_type": llama["stop_type"]}
    ]
    if finish == "stop" and len(cands) >= 2:
        # The terminating token is excluded (token convention of docs/paper_mapping.md).
        head = token_margin_uncertainty(top2_from_logprobs(cands[:-1]))
        assert oa.u_from_chat_completion(response, drop_stop_token=True) == head
        assert oa.u_from_chat_completion_chunks(chunks, drop_stop_token=True) == head
        assert lc.u_from_llamacpp(llama) == head
        assert lc.u_from_llamacpp_stream(events) == head
    elif finish == "length":
        assert oa.u_from_chat_completion(response, drop_stop_token=True) == full
        assert lc.u_from_llamacpp(llama) == full
        assert lc.u_from_llamacpp_stream(events) == full
    legacy = [
        {
            "content": f"t{t}",
            "probs": [{"tok_str": f"c{j}", "prob": math.exp(lp)} for j, lp in enumerate(c)],
        }
        for t, c in enumerate(cands)
    ]
    probs = [(math.exp(max(c)), math.exp(sorted(c)[-2])) for c in cands]
    assert lc.u_from_llamacpp(legacy, drop_stop_token=False) == pytest.approx(
        token_margin_uncertainty(probs), abs=1e-12
    )


@PROPERTY
@given(cands=candidate_lists())
def test_adapter_ablation_signals_match_baselines(cands: list[list[float]]) -> None:
    """TokenSignals entropy / max-prob agree with ucci.baselines where documented."""
    from ucci.baselines import mean_max_prob, mean_token_entropy
    from ucci.integrations import openai as oa

    content = [
        {"token": "x", "logprob": max(c), "top_logprobs": [{"logprob": lp} for lp in c]}
        for c in cands
    ]
    renorm = oa.signals_from_chat_completion(content, renormalize_entropy=True)
    assert renorm.entropy_support == "top_k_renormalized"
    assert renorm.mean_entropy == pytest.approx(mean_token_entropy(cands), rel=1e-12, abs=1e-12)
    truncated = oa.signals_from_chat_completion(content)
    assert truncated.mean_max_prob == pytest.approx(mean_max_prob(cands), rel=1e-15, abs=1e-15)
    assert truncated.u == from_openai_logprobs(content)
    assert truncated.n_non_greedy == 0
    assert truncated.top_k == min(len(c) for c in cands)


# ---------------------------------------------------------------------------
# Policy (Section 4.3, Eq. 6 and 7; Section 5, Theorem 1)
# ---------------------------------------------------------------------------


@PROPERTY
@given(p=st.one_of(UNIT, COARSE), theta=st.one_of(UNIT, COARSE))
def test_escalate_is_strict(p: float, theta: float) -> None:
    assert escalate(p, theta) == (p > theta)
    assert escalate(theta, theta) is False
    assert escalate(float(np.nextafter(theta, 2.0)), theta) is True


@PROPERTY
@given(
    prob=routing_problem(scores=DYADIC),
    grid=threshold_grid(),
    cost=costs(),
    model=COST_MODEL,
    data=st.data(),
)
def test_select_threshold_matches_brute_force(
    prob: Any, grid: np.ndarray, cost: tuple[float, float], model: str, data: Any
) -> None:
    """Feasible, minimal cost, then highest accuracy, then largest theta (Eq. 7)."""
    p, s, lg = prob
    cs, cl = cost
    rows = _grid_rows(p, s, lg, grid, cs, cl, model)
    tau = data.draw(st.one_of(st.sampled_from([r[2] for r in rows]), UNIT, DYADIC), label="tau")
    feasible = [r for r in rows if r[2] >= tau]
    if not feasible:
        event("infeasible target")
        with pytest.raises(InfeasibleTargetError):
            select_threshold(p, s, lg, tau, cs, cl, grid, cost_model=model)
        return
    want = min(feasible, key=lambda r: (r[1], -r[2], -r[0]))
    got = select_threshold(p, s, lg, tau, cs, cl, grid, cost_model=model)
    assert (got.theta, got.cost, got.accuracy, got.escalation_rate) == want
    assert got.accuracy >= tau
    assert got.theta in set(np.unique(grid).tolist())
    # The chosen operating point is on the efficient frontier (Figure 2).
    front = pareto_frontier(p, s, lg, cs, cl, grid, cost_model=model)
    assert bool(front.efficient[int(np.flatnonzero(front.theta == got.theta)[0])])

    # A corpus metric equal to the mean score gives the same choice.
    def metric(mask: np.ndarray) -> float:
        return (float(s[~mask].sum()) + float(lg[mask].sum())) / int(p.size)

    assert select_threshold(p, None, None, tau, cs, cl, grid, metric, model) == got
    # Re-evaluating the chosen theta on the same split reproduces it (Section 6.1, step 3).
    again = evaluate(p, s, lg, got.theta, cs, cl, cost_model=model)
    assert again == got


@PROPERTY
@given(prob=routing_problem(scores=UNIT), cost=costs(), model=COST_MODEL, data=st.data())
def test_select_threshold_float_scores_feasible_and_consistent_with_evaluate(
    prob: Any, cost: tuple[float, float], model: str, data: Any
) -> None:
    """With arbitrary float scores the choice is feasible and minimal-cost on the frontier."""
    p, s, lg = prob
    cs, cl = cost
    front = pareto_frontier(p, s, lg, cs, cl, cost_model=model)
    tau = data.draw(st.one_of(st.sampled_from(front.accuracy.tolist()), UNIT), label="tau")
    try:
        got = select_threshold(p, s, lg, tau, cs, cl, cost_model=model)
    except InfeasibleTargetError:
        assert float(front.accuracy.max()) < tau
        return
    assert got.accuracy >= tau
    feas = front.accuracy >= tau
    assert got.cost == float(front.cost[feas].min())
    again = evaluate(p, s, lg, got.theta, cs, cl, cost_model=model)
    assert again.cost == got.cost and again.escalation_rate == got.escalation_rate
    # The sweep and evaluate() sum the scores in different orders, so with
    # non-dyadic float scores the two accuracies can differ in the last bits
    # (bit-equality fails, and the chosen theta
    # can then re-evaluate a few ULP below tau). Only round-off is allowed.
    assert again.accuracy == pytest.approx(got.accuracy, rel=1e-14, abs=1e-15)


@PROPERTY
@given(
    prob=routing_problem(scores=DYADIC),
    grid=threshold_grid(),
    cost=costs(),
    model=COST_MODEL,
    data=st.data(),
)
def test_select_threshold_for_budget_matches_brute_force(
    prob: Any, grid: np.ndarray, cost: tuple[float, float], model: str, data: Any
) -> None:
    """Budget form (Table 2, bottom block): cost <= B, max accuracy, lower cost, larger theta."""
    p, s, lg = prob
    cs, cl = cost
    rows = _grid_rows(p, s, lg, grid, cs, cl, model)
    hi = cs + cl
    budget = data.draw(
        st.one_of(st.sampled_from([r[1] for r in rows]), st.floats(cs * 0.5, hi)), label="budget"
    )
    feasible = [r for r in rows if r[1] <= budget]
    if not feasible:
        event("infeasible target")
        with pytest.raises(InfeasibleTargetError):
            select_threshold_for_budget(p, s, lg, budget, cs, cl, grid, cost_model=model)
        return
    want = max(feasible, key=lambda r: (r[2], -r[1], r[0]))
    got = select_threshold_for_budget(p, s, lg, budget, cs, cl, grid, cost_model=model)
    assert (got.theta, got.cost, got.accuracy, got.escalation_rate) == want
    assert got.cost <= budget
    front = pareto_frontier(p, s, lg, cs, cl, grid, cost_model=model)
    assert bool(front.efficient[int(np.flatnonzero(front.theta == got.theta)[0])])


@PROPERTY
@given(prob=routing_problem(scores=UNIT), grid=threshold_grid(), cost=costs(), model=COST_MODEL)
def test_frontier_cost_and_rate_monotone_and_efficient_set_exact(
    prob: Any, grid: np.ndarray, cost: tuple[float, float], model: str
) -> None:
    p, s, lg = prob
    cs, cl = cost
    front = pareto_frontier(p, s, lg, cs, cl, grid, cost_model=model)
    np.testing.assert_array_equal(front.theta, np.unique(grid))
    assert bool(np.all(np.diff(front.escalation_rate) <= 0.0))
    assert bool(np.all(np.diff(front.cost) <= 0.0)), "cost must not increase with theta"
    for j in range(front.theta.size):
        esc = p > front.theta[j]
        assert front.cost[j] == policy_cost(esc, cs, cl, model)
        assert front.escalation_rate[j] == esc.mean()
    # Efficient points: not weakly dominated with one strict improvement.
    c, a = front.cost, front.accuracy
    for j in range(c.size):
        dominated = bool(np.any((c <= c[j]) & (a >= a[j]) & ((c < c[j]) | (a > a[j]))))
        assert bool(front.efficient[j]) == (not dominated)


@PROPERTY
@given(
    prob=routing_problem(scores=DYADIC),
    cost_a=costs(),
    cost_b=costs(),
    model_a=COST_MODEL,
    model_b=COST_MODEL,
    data=st.data(),
)
def test_selected_theta_does_not_depend_on_costs(
    prob: Any,
    cost_a: tuple[float, float],
    cost_b: tuple[float, float],
    model_a: str,
    model_b: str,
    data: Any,
) -> None:
    """Every escalation adds the same marginal cost, so theta* is cost-free (Section 4.3)."""
    p, s, lg = prob
    front = pareto_frontier(p, s, lg)
    tau = data.draw(st.one_of(st.sampled_from(front.accuracy.tolist()), DYADIC), label="tau")
    assume(float(front.accuracy.max()) >= tau)
    a = select_threshold(p, s, lg, tau, *cost_a, cost_model=model_a)
    b = select_threshold(p, s, lg, tau, *cost_b, cost_model=model_b)
    assert (a.theta, a.accuracy, a.escalation_rate) == (b.theta, b.accuracy, b.escalation_rate)


@PROPERTY
@given(
    k=st.lists(st.integers(0, 32), min_size=1, max_size=9),
    alpha_k=st.integers(0, 32),
    data=st.data(),
)
def test_theorem1_threshold_policy_is_cost_optimal_among_all_subsets(
    k: list[int], alpha_k: int, data: Any
) -> None:
    """Theorem 1 (Section 5, Appendix A.1) against every escalation subset.

    With calibrated p_hat, keeping x has expected accuracy 1 - p_hat(x) and
    escalating it has the fixed accuracy alpha_large (assumption (ii)).
    Among all 2^n escalation sets meeting tau, the cheapest has size k*.
    The threshold policy selected by :func:`ucci.select_threshold` escalates
    exactly k* queries whenever the top-k* queries by p_hat are a union of
    level sets of p_hat, and never fewer than k* (tie-breaking on level
    sets is the only gap the theorem allows).
    """
    n = len(k)
    p = np.array(k, dtype=np.float64) / 32.0
    alpha = alpha_k / 32.0
    small = 1.0 - p
    large = np.full(n, alpha)
    fp = [Fraction(v) for v in k]
    total_keep = sum(Fraction(32) - v for v in fp)

    def exp_acc(subset: tuple[int, ...]) -> Fraction:
        gain = sum(Fraction(alpha_k) - (Fraction(32) - fp[i]) for i in subset)
        return Fraction(total_keep + gain) / (32 * n)

    accs = {size: [exp_acc(sub) for sub in combinations(range(n), size)] for size in range(n + 1)}
    options = sorted({float(a) for v in accs.values() for a in v})
    tau = data.draw(
        st.one_of(
            st.sampled_from(options),
            st.sampled_from(options).map(lambda v: float(np.nextafter(v, 2.0))),
            st.sampled_from(options).map(lambda v: float(np.nextafter(v, -1.0))),
            UNIT,
        ),
        label="tau",
    )
    # The library compares float accuracies: an exact (dyadic) score sum divided
    # by n, which is the correctly rounded value float(exact accuracy).
    feasible_sizes = [size for size, v in accs.items() if any(float(a) >= tau for a in v)]
    grid = np.unique(np.concatenate([p, [0.0, 1.0]]))
    if not feasible_sizes:
        event("infeasible target")
        with pytest.raises(InfeasibleTargetError):
            select_threshold(p, small, large, tau, grid=grid)
        return
    k_star = min(feasible_sizes)
    # Greedy optimality behind Theorem 1: top-k by p_hat maximizes expected accuracy.
    order = np.argsort(-p, kind="stable")
    for size in range(n + 1):
        assert exp_acc(tuple(order[:size].tolist())) == max(accs[size])
    # A threshold policy meets tau whenever any escalation set does: the most
    # accurate set, {x : p_hat(x) > 1 - alpha_large}, is itself a threshold set.
    got = select_threshold(p, small, large, tau, grid=grid)
    n_esc = round(got.escalation_rate * n)
    assert n_esc >= k_star
    esc = p > got.theta
    assert float(exp_acc(tuple(np.flatnonzero(esc).tolist()))) >= tau
    sp = np.sort(p)[::-1]
    splits_level_set = 0 < k_star < n and sp[k_star - 1] == sp[k_star]
    if not splits_level_set:
        assert n_esc == k_star


@PROPERTY
@given(prob=routing_problem(scores=DYADIC), cost=costs(), model=COST_MODEL)
def test_policy_cost_and_accuracy_formulas(
    prob: Any, cost: tuple[float, float], model: str
) -> None:
    p, s, lg = prob
    cs, cl = cost
    mask = p > 0.5
    r = int(mask.sum()) / p.size
    assert policy_cost(mask, cs, cl, model) == _cost(r, cs, cl, model)
    assert policy_cost(mask.astype(int), cs, cl, model) == policy_cost(mask, cs, cl, model)
    assert policy_accuracy(mask, s, lg) == float(np.where(mask, lg, s).sum()) / p.size


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@st.composite
def forecasts(draw: Any, max_n: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    n = draw(st.integers(1, max_n))
    p = np.array(draw(st.lists(st.one_of(UNIT, COARSE), min_size=n, max_size=n)))
    y = np.array(draw(st.lists(LABEL, min_size=n, max_size=n)))
    w = (
        np.array(draw(st.lists(st.one_of(WEIGHT, st.just(0.0)), min_size=n, max_size=n)))
        if draw(st.booleans())
        else None
    )
    if w is not None and not w.sum() > 0:
        w[0] = 1.0
    return p, y, w


@PROPERTY
@given(
    fc=forecasts(),
    n_bins=st.integers(1, 20),
    strategy=st.sampled_from(["uniform", "quantile"]),
)
def test_ece_and_reliability_table_invariants(fc: Any, n_bins: int, strategy: str) -> None:
    p, y, w = fc
    rows = reliability_table(p, y, n_bins=n_bins, strategy=strategy, sample_weight=w)
    kept = p.size if w is None else int(np.count_nonzero(w > 0))
    assert sum(r.count for r in rows) == kept
    assert len(rows) <= n_bins
    for r in rows:
        assert r.count >= 1
        assert r.bin_lower <= r.bin_upper
        assert r.bin_lower - 1e-12 <= r.mean_forecast <= r.bin_upper + 1e-12
        assert 0.0 <= r.observed_frequency <= 1.0
    for a, b in zip(rows, rows[1:]):
        assert a.bin_upper <= b.bin_lower
    value = ece(p, y, n_bins=n_bins, strategy=strategy, sample_weight=w)
    assert 0.0 <= value <= 1.0
    ww = np.ones(p.size) if w is None else w
    if w is None:
        by_rows = sum(r.count * abs(r.mean_forecast - r.observed_frequency) for r in rows) / kept
        assert value == pytest.approx(by_rows, abs=1e-12)
    assert brier_score(p, y, w) == pytest.approx(
        float(np.sum(ww * (p - y) ** 2) / np.sum(ww)), abs=1e-12
    )


@PROPERTY
@given(
    groups=st.lists(
        st.integers(1, 12).flatmap(lambda c: st.tuples(st.just(c), st.integers(0, c))),
        min_size=1,
        max_size=12,
    ),
    n_bins=st.integers(1, 15),
    strategy=st.sampled_from(["uniform", "quantile"]),
)
def test_ece_is_zero_for_perfectly_calibrated_forecasts(
    groups: list[tuple[int, int]], n_bins: int, strategy: str
) -> None:
    """Forecast a/c for a group of c queries with a errors: every bin is calibrated."""
    p_parts, y_parts = [], []
    for c, a in groups:
        p_parts.append(np.full(c, a / c))
        y_parts.append(np.array([1.0] * a + [0.0] * (c - a)))
    p, y = np.concatenate(p_parts), np.concatenate(y_parts)
    assert ece(p, y, n_bins=n_bins, strategy=strategy) == pytest.approx(0.0, abs=1e-12)
    for r in reliability_table(p, y, n_bins=n_bins, strategy=strategy):
        assert r.mean_forecast == pytest.approx(r.observed_frequency, abs=1e-12)


@PROPERTY
@given(fc=forecasts(), reps=st.lists(st.integers(1, 4), min_size=60), n_bins=st.integers(1, 15))
def test_ece_integer_weights_equal_replication(fc: Any, reps: list[int], n_bins: int) -> None:
    p, y, _ = fc
    r = np.array(reps[: p.size])
    weighted = ece(p, y, n_bins=n_bins, sample_weight=r.astype(float))
    replicated = ece(np.repeat(p, r), np.repeat(y, r), n_bins=n_bins)
    assert weighted == pytest.approx(replicated, abs=1e-12)


@PROPERTY
@given(
    x=st.lists(st.floats(-1e3, 1e3), min_size=1, max_size=50),
    n_boot=st.integers(1, 60),
    alpha=st.floats(0.001, 0.999),
    seed=st.integers(0, 2**31 - 1),
)
def test_bootstrap_ci_ordered_bounded_and_reproducible(
    x: list[float], n_boot: int, alpha: float, seed: int
) -> None:
    xs = np.array(x)

    def stat(idx: np.ndarray) -> float:
        return float(xs[idx].mean())

    lo, hi = bootstrap_ci(stat, xs.size, n_boot=n_boot, alpha=alpha, seed=seed)
    assert lo <= hi
    assert xs.min() - 1e-9 <= lo and hi <= xs.max() + 1e-9
    assert bootstrap_ci(stat, xs.size, n_boot=n_boot, alpha=alpha, seed=seed) == (lo, hi)
    const = bootstrap_ci(lambda idx: 0.25, xs.size, n_boot=n_boot, alpha=alpha, seed=seed)
    assert const == (0.25, 0.25)


@PROPERTY
@given(
    counts=st.lists(st.tuples(*(st.integers(0, 6) for _ in range(6))), min_size=1, max_size=30),
    mask_bits=st.lists(st.booleans(), min_size=30, max_size=30),
)
def test_micro_f1_and_routed_micro_f1(counts: list[tuple[int, ...]], mask_bits: list[bool]) -> None:
    arr = np.array(counts, dtype=float)
    small, large = arr[:, :3], arr[:, 3:]
    mask = np.array(mask_bits[: arr.shape[0]])
    f1 = routed_micro_f1(small, large)
    routed = np.where(mask[:, None], large, small).sum(axis=0)
    want = micro_f1(*routed)
    assert f1(mask) == want
    assert 0.0 <= want <= 1.0
    t, fp, fn = routed
    if 2 * t + fp + fn > 0:
        assert want == pytest.approx(2 * t / (2 * t + fp + fn), abs=1e-15)
    assert f1(np.zeros(mask.size, bool)) == micro_f1(small[:, 0], small[:, 1], small[:, 2])
    assert f1(np.ones(mask.size, bool)) == micro_f1(large[:, 0], large[:, 1], large[:, 2])


# ---------------------------------------------------------------------------
# Router and router JSON
# ---------------------------------------------------------------------------


@PROPERTY
@given(
    cal=calibration_data(max_n=40),
    val=routing_problem(max_n=40, scores=DYADIC),
    cost=costs(),
    model=COST_MODEL,
    step=st.sampled_from([0.005, 0.01, 0.1, 0.25]),
    mode=st.sampled_from(["tau", "budget", "manual"]),
    data=st.data(),
)
def test_router_json_round_trip_is_exact(
    cal: Any,
    val: Any,
    cost: tuple[float, float],
    model: str,
    step: float,
    mode: str,
    data: Any,
) -> None:
    u, e, w = cal
    uv, s, lg = val
    router = UCCIRouter(*cost, cost_model=model, grid_step=step).calibrate(u, e, w)
    pv = router.error_probability(uv)
    front = pareto_frontier(pv, s, lg, *cost, make_grid(step), cost_model=model)
    try:
        if mode == "tau":
            tau = data.draw(st.sampled_from(front.accuracy.tolist()), label="tau")
            choice = router.choose_threshold(uv, s, lg, tau)
            assert choice == select_threshold(pv, s, lg, tau, *cost, make_grid(step), None, model)
        elif mode == "budget":
            budget = data.draw(st.sampled_from(front.cost.tolist()), label="budget")
            router.choose_threshold_for_budget(uv, s, lg, budget)
        else:
            router.theta = data.draw(st.floats(-2.0, 2.0), label="theta")
    except InfeasibleTargetError:  # pragma: no cover - targets are drawn from the frontier
        pytest.fail("a target taken from the frontier must be feasible")
    doc = router.to_dict()
    text = ucci_io.dumps_router_dict(ucci_io.validate_router_dict(doc))
    back = UCCIRouter.from_dict(json.loads(text))
    assert back.to_dict() == doc
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "router.json"
        router.save(path)
        loaded = UCCIRouter.load(path)
    assert loaded.to_dict() == doc
    x = np.asarray(router.calibrator.x_)
    q = np.concatenate(
        [
            np.linspace(-0.5, 1.5, 101),
            u,
            x,
            np.nextafter(x, np.inf),
            np.nextafter(x, -np.inf),
        ]
    )
    np.testing.assert_array_equal(loaded.error_probability(q), router.error_probability(q))
    np.testing.assert_array_equal(loaded.escalate(q), router.escalate(q))
    res = loaded.route(q)
    np.testing.assert_array_equal(res.escalate, escalate(res.p_hat, loaded.theta))
    np.testing.assert_array_equal(res.p_hat, loaded.calibrator.predict(q))
    assert loaded.evaluate(uv, s, lg) == evaluate(pv, s, lg, router.theta, *cost, cost_model=model)


# ---------------------------------------------------------------------------
# Baselines (Section 6.1): split conformal, raw-score thresholds, oracle
# ---------------------------------------------------------------------------


def _conformal_quantile(correct: np.ndarray, delta: float) -> float:
    """Smallest order statistic s_(j) with j / (n + 1) >= 1 - delta, else +inf."""
    s = np.sort(correct)
    n = s.size
    target = 1 - Fraction(repr(float(delta)))
    for j in range(1, n + 1):
        if Fraction(j, n + 1) >= target:
            return float(s[j - 1])
    return math.inf


@st.composite
def conformal_problem(draw: Any) -> Any:
    n = draw(st.integers(1, 40))
    u = np.array(draw(st.lists(st.one_of(UNIT, COARSE), min_size=n, max_size=n)))
    e = np.array(draw(st.lists(st.sampled_from([0.0, 1.0]), min_size=n, max_size=n)))
    if not (e == 0.0).any():
        e[draw(st.integers(0, n - 1))] = 0.0
    return u, e


@PROPERTY
@given(
    cal=conformal_problem(),
    deltas=st.lists(
        st.one_of(
            st.integers(1, 199).map(lambda k: k * 0.005),
            st.floats(1e-6, 1 - 1e-6),
        ),
        min_size=1,
        max_size=12,
    ),
)
def test_split_conformal_quantile_is_the_order_statistic(cal: Any, deltas: list[float]) -> None:
    from ucci.baselines import SplitConformalRouter

    u, e = cal
    router = SplitConformalRouter().calibrate(u, e)
    correct = u[e == 0.0]
    n = correct.size
    for d in deltas:
        q = router.quantile(d)
        assert q == _conformal_quantile(correct, d)
        # k / (n + 1) >= 1 - delta holds exactly; as floats the two sides can
        # differ by one rounding.
        bound = router.keep_probability_bound(d)
        assert Fraction(bound) >= 1 - Fraction(repr(float(d))) - Fraction(1, 2**52)
        if math.isfinite(q):
            covered = int(np.count_nonzero(correct <= q))
            assert Fraction(covered, n) >= 1 - Fraction(repr(float(d)))


@PROPERTY
@given(
    cal=conformal_problem(),
    val=routing_problem(max_n=40, scores=DYADIC),
    cost=costs(),
    model=COST_MODEL,
    data=st.data(),
)
def test_split_conformal_choose_threshold_matches_brute_force(
    cal: Any, val: Any, cost: tuple[float, float], model: str, data: Any
) -> None:
    from ucci.baselines import DEFAULT_DELTA_GRID, SplitConformalRouter

    u, e = cal
    sv, s, lg = val
    router = SplitConformalRouter(*cost, cost_model=model).calibrate(u, e)
    correct = u[e == 0.0]
    n = sv.size
    rows = []
    for d in DEFAULT_DELTA_GRID:
        q = _conformal_quantile(correct, float(d))
        mask = sv > q
        k = int(mask.sum())
        acc = (float(s[~mask].sum()) + float(lg[mask].sum())) / n
        rows.append((q, _cost(k / n, *cost, model), acc, float(d)))
    tau = data.draw(st.one_of(st.sampled_from([r[2] for r in rows]), DYADIC), label="tau")
    feasible = [r for r in rows if r[2] >= tau]
    if not feasible:
        event("infeasible target")
        with pytest.raises(InfeasibleTargetError):
            router.choose_threshold(sv, s, lg, tau)
        return
    best = min(feasible, key=lambda r: (r[1], -r[2], -r[0]))
    got = router.choose_threshold(sv, s, lg, tau)
    assert (got.theta, got.cost, got.accuracy) == best[:3]
    assert router.alpha_ == best[0]
    assert router.delta_ == min(r[3] for r in rows if r[0] == best[0])
    np.testing.assert_array_equal(router.escalate(sv), sv > best[0])


@PROPERTY
@given(
    val=routing_problem(max_n=40, scores=DYADIC),
    cost=costs(),
    model=COST_MODEL,
    direction=st.sampled_from(["uncertainty", "confidence"]),
    data=st.data(),
)
def test_raw_threshold_router_search_is_exhaustive(
    val: Any, cost: tuple[float, float], model: str, direction: str, data: Any
) -> None:
    """Entropy / FrugalGPT-style routers: the exact search equals trying every threshold."""
    from ucci.baselines import EntropyThresholdRouter, FrugalGPTStyleRouter

    scores, s, lg = val
    n = scores.size
    cls = EntropyThresholdRouter if direction == "uncertainty" else FrugalGPTStyleRouter
    router = cls(*cost, cost_model=model)
    sign = 1.0 if direction == "uncertainty" else -1.0
    rows = []
    for t in [-math.inf, *np.unique(scores).tolist(), math.inf]:
        mask = scores > t if direction == "uncertainty" else scores < t
        k = int(mask.sum())
        acc = (float(s[~mask].sum()) + float(lg[mask].sum())) / n
        rows.append((t, _cost(k / n, *cost, model), acc))
    tau = data.draw(st.one_of(st.sampled_from([r[2] for r in rows]), DYADIC), label="tau")
    feasible = [r for r in rows if r[2] >= tau]
    if not feasible:
        event("infeasible target")
        with pytest.raises(InfeasibleTargetError):
            router.choose_threshold(scores, s, lg, tau)
        return
    best = min(feasible, key=lambda r: (r[1], -r[2], -sign * r[0]))
    got = router.choose_threshold(scores, s, lg, tau)
    assert (got.theta, got.cost, got.accuracy) == best
    mask = router.escalate(scores)
    assert policy_cost(mask, *cost, model) == got.cost
    assert policy_accuracy(mask, s, lg) == got.accuracy


@PROPERTY
@given(
    scores=st.lists(st.tuples(DYADIC, DYADIC), min_size=1, max_size=9),
    cost=costs(),
    model=COST_MODEL,
    data=st.data(),
)
def test_oracle_is_the_exact_optimum_over_all_subsets(
    scores: list[tuple[float, float]], cost: tuple[float, float], model: str, data: Any
) -> None:
    """With tau: minimum cost of any escalation set; with a budget: maximum accuracy."""
    from ucci.baselines import Oracle

    s = np.array([a for a, _ in scores])
    lg = np.array([b for _, b in scores])
    n = s.size
    subsets = []
    for size in range(n + 1):
        for sub in combinations(range(n), size):
            mask = np.zeros(n, dtype=bool)
            mask[list(sub)] = True
            acc = (float(s[~mask].sum()) + float(lg[mask].sum())) / n
            subsets.append((size, _cost(size / n, *cost, model), acc))
    oracle = Oracle(*cost, cost_model=model)
    tau = data.draw(st.one_of(st.sampled_from([a for _, _, a in subsets]), DYADIC), label="tau")
    feasible = [c for _, c, a in subsets if a >= tau]
    if feasible:
        got = oracle.evaluate(s, lg, tau)
        assert got.accuracy >= tau
        assert got.cost == min(feasible)
    else:
        with pytest.raises(InfeasibleTargetError):
            oracle.evaluate(s, lg, tau)
    budget = data.draw(st.sampled_from([c for _, c, _ in subsets]), label="budget")
    within = [a for _, c, a in subsets if c <= budget]
    got = oracle.evaluate(s, lg, budget=budget)
    assert got.cost <= budget
    assert got.accuracy == max(within)
    best = oracle.evaluate(s, lg)
    assert best.accuracy == max(a for _, _, a in subsets)


# ---------------------------------------------------------------------------
# Calibration ablation (Appendix B.4) and the comparison protocol (Section 6.1)
# ---------------------------------------------------------------------------


@PROPERTY
@given(data=calibration_data(u_values=st.one_of(UNIT, COARSE)))
def test_temperature_scaling_is_monotone_and_minimizes_nll(data: Any) -> None:
    import warnings

    from ucci.baselines import TemperatureScalingCalibrator

    u, e, w = data
    cal = TemperatureScalingCalibrator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cal.fit(u, e, sample_weight=w)
    temp = cal.temperature_
    assert temp is not None and cal.t_min <= temp <= cal.t_max
    q = np.linspace(0.0, 1.0, 101)
    pq = cal.predict(q)
    assert bool(np.all(np.diff(pq) >= 0.0)) and pq.min() >= 0.0 and pq.max() <= 1.0
    assert cal.predict(0.5) == 0.5
    ww = np.ones(u.size) if w is None else w
    z = cal._logit(u)

    def nll(s: float) -> float:
        x = s * z
        return float(np.sum(ww * (np.logaddexp(0.0, x) - e * x)) / np.sum(ww))

    s_star = 1.0 / temp
    best = nll(s_star)
    assert best == pytest.approx(cal.nll_, rel=1e-12, abs=1e-15)
    for s in np.geomspace(1.0 / cal.t_max, 1.0 / cal.t_min, 61):
        assert best <= nll(float(s)) + 1e-12 * max(1.0, abs(best))


@PROPERTY
@given(
    data=calibration_data(u_values=st.one_of(UNIT, COARSE)),
    l2=st.sampled_from([0.0, 0.0, 0.1, 1.0]),
    smoothing=st.booleans(),
)
def test_platt_fit_is_the_penalized_likelihood_optimum(
    data: Any, l2: float, smoothing: bool
) -> None:
    """When PlattCalibrator.fit succeeds, no (a, b) has a lower objective."""
    scipy_opt = pytest.importorskip("scipy.optimize")
    from ucci.baselines import PlattCalibrator

    u, e, w = data
    # u values a few ULP apart (for example 0, 1e-38 and 1e-15) can make the
    # optimum lie at a slope near 1e16, where the Newton decrement is already
    # below the stopping rule; fit then returns a non-optimal (a, b) without
    # raising. Such near-ties are excluded.
    distinct = np.unique(u)
    assume(distinct.size < 2 or float(np.diff(distinct).min()) >= 1e-9)
    if smoothing:
        e = (e >= 0.5).astype(float)
    cal = PlattCalibrator(l2=l2, target_smoothing=smoothing)
    try:
        cal.fit(u, e, sample_weight=w)
    except ValueError as exc:
        assert (
            "not identifiable" in str(exc)
            or "infinite" in str(exc)
            or "separates" in str(exc)
            or "did not converge" in str(exc)
        ), str(exc)
        return
    ww = np.ones(u.size) if w is None else w
    if smoothing:
        n_pos, n_neg = float(ww[e == 1.0].sum()), float(ww[e == 0.0].sum())
        t = np.where(e == 1.0, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    else:
        t = e

    def objective(ab: np.ndarray) -> float:
        x = ab[0] * u + ab[1]
        return float(np.sum(ww * (np.logaddexp(0.0, x) - t * x)) + 0.5 * l2 * ab[0] ** 2)

    ours = objective(np.array([cal.a_, cal.b_]))
    ref = scipy_opt.minimize(
        objective,
        np.array([cal.a_, cal.b_]),
        method="Nelder-Mead",
        options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 4000},
    )
    assert ours <= float(ref.fun) + 1e-9 * max(1.0, abs(ours))
    pq = cal.predict(np.linspace(0.0, 1.0, 51))
    assert bool(np.all(np.diff(pq) >= 0.0)) or bool(np.all(np.diff(pq) <= 0.0))


@PROPERTY
@given(
    cal=calibration_data(max_n=40),
    val=routing_problem(max_n=40, scores=DYADIC),
    cost=costs(),
    model=COST_MODEL,
    data=st.data(),
)
def test_calibrated_threshold_router_with_isotonic_is_ucci(
    cal: Any, val: Any, cost: tuple[float, float], model: str, data: Any
) -> None:
    """Appendix B.4 protocol: with the isotonic calibrator it is UCCIRouter itself."""
    from ucci.baselines import CalibratedThresholdRouter

    u, e, w = cal
    uv, s, lg = val
    ours = CalibratedThresholdRouter(IsotonicCalibrator(), *cost, cost_model=model)
    ours.calibrate(u, e, w)
    core = UCCIRouter(*cost, cost_model=model).calibrate(u, e, w)
    front = pareto_frontier(core.error_probability(uv), s, lg, *cost, cost_model=model)
    if data.draw(st.booleans(), label="budget form"):
        budget = data.draw(st.sampled_from(front.cost.tolist()), label="budget")
        a = ours.choose_threshold(uv, s, lg, budget=budget)
        b = core.choose_threshold_for_budget(uv, s, lg, budget)
    else:
        tau = data.draw(st.sampled_from(front.accuracy.tolist()), label="tau")
        a = ours.choose_threshold(uv, s, lg, tau)
        b = core.choose_threshold(uv, s, lg, tau)
    assert a == b
    np.testing.assert_array_equal(ours.escalate(uv), core.escalate(uv))


@st.composite
def split_data(draw: Any, n_min: int = 1, n_max: int = 40) -> Any:
    """A SplitData with u, entropy and confidence signals and 0/1 scores."""
    from ucci.baselines import SplitData

    n = draw(st.integers(n_min, n_max))
    u = np.array(draw(st.lists(st.one_of(UNIT, COARSE), min_size=n, max_size=n)))
    ent = np.array(draw(st.lists(st.floats(0.0, 5.0), min_size=n, max_size=n)))
    conf = np.array(draw(st.lists(UNIT, min_size=n, max_size=n)))
    s = np.array(draw(st.lists(BINARY, min_size=n, max_size=n)))
    lg = np.array(draw(st.lists(BINARY, min_size=n, max_size=n)))
    return SplitData({"u": u, "entropy": ent, "confidence": conf}, s, lg)


@PROPERTY
@given(
    cal=split_data(),
    val=split_data(),
    test=split_data(),
    tau=st.one_of(DYADIC, UNIT),
    cost=costs(),
    model=COST_MODEL,
)
def test_compare_routers_rows_match_manual_protocol_and_oracle_bounds_them(
    cal: Any, val: Any, test: Any, tau: float, cost: tuple[float, float], model: str
) -> None:
    """Section 6.1 protocol: each row equals running that router by hand; the oracle is a floor."""
    from ucci.baselines import compare_routers

    # Split conformal needs one small-correct calibration query; without one,
    # compare_routers raises instead of reporting that row as infeasible
    #, so such splits are excluded here.
    assume(bool((np.asarray(cal.small_score) == 1.0).any()))
    rows = {
        r.method: r
        for r in compare_routers(
            cal, val, test, tau=tau, c_small=cost[0], c_large=cost[1], cost_model=model
        )
    }
    s_c = np.asarray(cal.small_score)
    u_v, s_v, l_v = val.signals["u"], np.asarray(val.small_score), np.asarray(val.large_score)
    u_t, s_t, l_t = test.signals["u"], np.asarray(test.small_score), np.asarray(test.large_score)
    router = UCCIRouter(*cost, cost_model=model).calibrate(
        cal.signals["u"], (s_c < 1.0).astype(float)
    )
    row = rows["UCCI"]
    try:
        choice = router.choose_threshold(u_v, s_v, l_v, tau)
    except InfeasibleTargetError:
        assert not row.feasible_on_val
    else:
        assert row.feasible_on_val
        assert row.threshold == choice.theta
        assert (row.val_cost, row.val_accuracy) == (choice.cost, choice.accuracy)
        res = router.evaluate(u_t, s_t, l_t)
        assert (row.cost, row.accuracy, row.escalation_rate) == (
            res.cost,
            res.accuracy,
            res.escalation_rate,
        )
    oracle = rows["Oracle"]
    if oracle.feasible_on_val:
        for r in rows.values():
            if math.isfinite(r.accuracy) and r.accuracy >= tau:
                assert oracle.cost <= r.cost, r.method
    assert rows["Large-only"].cost == policy_cost(np.ones(u_t.size, bool), *cost, model)
    assert rows["Small-only"].accuracy == float(s_t.mean())


# ---------------------------------------------------------------------------
# CLI (ucci fit, evaluate, route, report) against the Python API
# ---------------------------------------------------------------------------


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    from ucci.cli import main

    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(args)
    return code, out.getvalue(), err.getvalue()


@st.composite
def logged_traffic(draw: Any, labelled: bool = True) -> list[dict[str, Any]]:
    n = draw(st.integers(4, 60))
    recs = []
    for i in range(n):
        rec: dict[str, Any] = {
            "id": f"q{i}",
            "u": draw(st.one_of(UNIT, COARSE)),
            "small_correct": draw(st.sampled_from([0, 1])),
            "large_correct": draw(st.sampled_from([0, 1])),
        }
        if labelled:
            rec["split"] = draw(st.sampled_from(["cal", "val", "test"]))
        recs.append(rec)
    if labelled:
        recs[0]["split"], recs[1]["split"] = "cal", "val"
    return recs


def _write_jsonl(path: Path, recs: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


def _arrays(recs: list[dict[str, Any]], idx: np.ndarray) -> tuple[np.ndarray, ...]:
    u = np.array([recs[i]["u"] for i in idx], dtype=np.float64)
    sc = np.array([recs[i]["small_correct"] for i in idx], dtype=np.float64)
    lc = np.array([recs[i]["large_correct"] for i in idx], dtype=np.float64)
    return u, sc, lc


@PROPERTY
@given(
    recs=logged_traffic(),
    tau=st.one_of(DYADIC, UNIT),
    model=COST_MODEL,
    budget_mode=st.booleans(),
    budget=st.floats(1.0, 4.02),
)
def test_cli_fit_evaluate_route_match_python_api(
    recs: list[dict[str, Any]], tau: float, model: str, budget_mode: bool, budget: float
) -> None:
    idx = {k: np.array([i for i, r in enumerate(recs) if r["split"] == k]) for k in ("cal", "val")}
    idx["test"] = np.array([i for i, r in enumerate(recs) if r["split"] == "test"], dtype=int)
    u_c, s_c, _ = _arrays(recs, idx["cal"])
    u_v, s_v, l_v = _arrays(recs, idx["val"])
    api = UCCIRouter(cost_model=model).calibrate(u_c, 1.0 - s_c)
    objective = ["--budget", repr(budget)] if budget_mode else ["--tau", repr(tau)]
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "traffic.jsonl"
        out = Path(tmp) / "router.json"
        _write_jsonl(data, recs)
        code, stdout, stderr = _run_cli(
            [
                "fit",
                "--data",
                str(data),
                "--out",
                str(out),
                "--cost-model",
                model,
                "--json",
                *objective,
            ]
        )
        try:
            if budget_mode:
                choice = api.choose_threshold_for_budget(u_v, s_v, l_v, budget)
            else:
                choice = api.choose_threshold(u_v, s_v, l_v, tau)
        except InfeasibleTargetError:
            assert code == 4, stderr
            assert not out.exists()
            return
        assert code == 0, stderr
        doc = json.loads(out.read_text(encoding="utf-8"))
        summary = json.loads(stdout)
        want = api.to_dict()
        for key in ("calibrator", "theta", "c_small", "c_large", "cost_model", "tau", "grid_step"):
            assert doc[key] == want[key], key
        assert summary["theta"] == choice.theta
        assert summary["validation"]["cost"] == choice.cost
        assert summary["validation"]["accuracy"] == choice.accuracy
        assert summary["validation"]["escalation_rate"] == choice.escalation_rate
        assert summary["split"]["sizes"] == {k: int(v.size) for k, v in idx.items()}

        # Step 3 of Section 6.1 on the test split, through the CLI and the API.
        if idx["test"].size:
            code, stdout, stderr = _run_cli(
                [
                    "evaluate",
                    "--router",
                    str(out),
                    "--data",
                    str(data),
                    "--split",
                    "test",
                    "--bootstrap",
                    "25",
                    "--seed",
                    "3",
                    "--json",
                ]
            )
            assert code == 0, stderr
            ev = json.loads(stdout)
            u_t, s_t, l_t = _arrays(recs, idx["test"])
            res = api.evaluate(u_t, s_t, l_t)
            assert ev["n"] == idx["test"].size
            assert ev["ucci"]["cost"] == res.cost
            assert ev["ucci"]["accuracy"] == res.accuracy
            assert ev["ucci"]["escalation_rate"] == res.escalation_rate
            esc = api.escalate(u_t)

            def cost_stat(i: np.ndarray) -> float:
                return policy_cost(esc[i], api.c_small, api.c_large, model)

            ci = bootstrap_ci(cost_stat, u_t.size, n_boot=25, alpha=1.0 - 0.95, seed=3)
            assert ev["bootstrap"]["ci"]["cost"] == [ci.low, ci.high]

        # Eq. 6 on every record through `ucci route`.
        code, stdout, stderr = _run_cli(
            ["route", "--router", str(out), "--data", str(data), "--json"]
        )
        assert code == 0, stderr
        routed = json.loads(stdout)
        u_all = np.array([r["u"] for r in recs])
        decisions = api.route(u_all)
        assert [q["p_hat"] for q in routed["queries"]] == decisions.p_hat.tolist()
        assert [q["escalate"] for q in routed["queries"]] == decisions.escalate.tolist()
        assert [q["id"] for q in routed["queries"]] == [r["id"] for r in recs]


@st.composite
def unlabelled_ids(draw: Any, n: int) -> list[Any]:
    """Record ids: unique strings, repeated strings, integers, or missing (None)."""
    kind = draw(st.sampled_from(["unique", "repeated", "int", "missing"]))
    if kind == "unique":
        return [f"q{i}" for i in range(n)]
    if kind == "repeated":
        ids: list[Any] = draw(
            st.lists(st.sampled_from(["a", "b", "c", "7"]), min_size=n, max_size=n)
        )
        return ids
    if kind == "int":
        ints: list[Any] = draw(st.lists(st.integers(0, 9), min_size=n, max_size=n))
        return ints
    maybe: list[Any] = draw(
        st.lists(st.one_of(st.none(), st.sampled_from(["3", "x"])), min_size=n, max_size=n)
    )
    return maybe


def _write_records(path: Path, recs: list[dict[str, Any]], fmt: str) -> None:
    """Write records as JSON Lines, a JSON array, CSV or TSV (missing id = empty cell)."""
    if fmt == "jsonl":
        path.write_text(
            "".join(json.dumps({k: v for k, v in r.items() if v is not None}) + "\n" for r in recs),
            encoding="utf-8",
        )
    elif fmt == "json":
        body = [{k: v for k, v in r.items() if v is not None} for r in recs]
        path.write_text(json.dumps(body), encoding="utf-8")
    else:
        sep = "," if fmt == "csv" else "\t"
        cols = ["id", "u", "small_correct", "large_correct"]
        lines = [sep.join(cols)] + [
            sep.join("" if r[c] is None else repr(r[c]) if c == "u" else str(r[c]) for c in cols)
            for r in recs
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@PROPERTY
@given(
    recs=logged_traffic(labelled=False),
    seed=st.integers(0, 50),
    fmt=st.sampled_from(["jsonl", "json", "csv", "tsv"]),
    data=st.data(),
)
def test_cli_random_split_follows_documented_rule(
    recs: list[dict[str, Any]], seed: int, fmt: str, data: Any
) -> None:
    """Records ordered by SHA-256 of "<seed>:<id>" (ties by position), cut at floor(f * n + 0.5).

    A missing id is the record's 0-based index. Every input format gives the
    same split and the same router.
    """
    n = len(recs)
    for r, rid in zip(recs, data.draw(unlabelled_ids(n), label="ids")):
        r["id"] = rid
    ids = [str(i) if r["id"] is None else str(r["id"]) for i, r in enumerate(recs)]
    keys = [hashlib.sha256(f"{seed}:{rid}".encode()).hexdigest() for rid in ids]
    order = sorted(range(n), key=lambda i: (keys[i], i))
    n_cal = min(math.floor(0.3 * n + 0.5), n)
    n_val = min(math.floor(0.2 * n + 0.5), n - n_cal)
    cal = np.array(sorted(order[:n_cal]))
    val = np.array(sorted(order[n_cal : n_cal + n_val]))
    test = np.array(sorted(order[n_cal + n_val :]), dtype=int)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"traffic.{fmt}"
        _write_records(path, recs, fmt)
        out = Path(tmp) / "router.json"
        code, stdout, stderr = _run_cli(
            [
                *("fit", "--data", str(path), "--out", str(out)),
                *("--tau", "0", "--seed", str(seed), "--json"),
            ]
        )
        assert code == 0, stderr
        summary = json.loads(stdout)
        assert summary["split"]["sizes"] == {"cal": cal.size, "val": val.size, "test": test.size}
        u_c, s_c, _ = _arrays(recs, cal)
        u_v, s_v, l_v = _arrays(recs, val)
        api = UCCIRouter().calibrate(u_c, 1.0 - s_c)
        choice = api.choose_threshold(u_v, s_v, l_v, 0.0)
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["calibrator"] == api.to_dict()["calibrator"]
        assert doc["theta"] == choice.theta
        if test.size:
            code, stdout, stderr = _run_cli(
                [
                    *("evaluate", "--router", str(out), "--data", str(path)),
                    *("--split", "test", "--bootstrap", "0", "--json"),
                ]
            )
            assert code == 0, stderr
            ev = json.loads(stdout)
            u_t, s_t, l_t = _arrays(recs, test)
            res = api.evaluate(u_t, s_t, l_t)
            assert ev["n"] == test.size
            assert (ev["ucci"]["cost"], ev["ucci"]["accuracy"]) == (res.cost, res.accuracy)


@PROPERTY
@given(
    recs=logged_traffic(),
    tau=st.one_of(DYADIC, UNIT),
    bins=st.integers(1, 12),
    strategy=st.sampled_from(["uniform", "quantile"]),
)
def test_cli_validation_numbers_and_report_match_python_api(
    recs: list[dict[str, Any]], tau: float, bins: int, strategy: str
) -> None:
    """`evaluate --split val` reproduces the fit summary; `report` equals ucci.ece."""
    cal_idx = np.array([i for i, r in enumerate(recs) if r["split"] == "cal"])
    val_idx = np.array([i for i, r in enumerate(recs) if r["split"] == "val"])
    u_c, s_c, _ = _arrays(recs, cal_idx)
    u_v, s_v, l_v = _arrays(recs, val_idx)
    api = UCCIRouter().calibrate(u_c, 1.0 - s_c)
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "traffic.jsonl"
        out = Path(tmp) / "router.json"
        _write_jsonl(data, recs)
        code, stdout, stderr = _run_cli(
            ["fit", "--data", str(data), "--out", str(out), "--tau", repr(tau), "--json"]
        )
        if code == 4:
            with pytest.raises(InfeasibleTargetError):
                api.choose_threshold(u_v, s_v, l_v, tau)
            return
        assert code == 0, stderr
        fit = json.loads(stdout)
        code, stdout, stderr = _run_cli(
            [
                *("evaluate", "--router", str(out), "--data", str(data), "--split", "val"),
                *("--bootstrap", "0", "--json"),
            ]
        )
        assert code == 0, stderr
        ev = json.loads(stdout)
        for key in ("cost", "accuracy", "escalation_rate", "savings_vs_large"):
            assert ev["ucci"][key] == fit["validation"][key], key
        assert ev["ucci"]["accuracy_minus_tau"] >= 0.0
        code, stdout, stderr = _run_cli(
            [
                *("report", "--router", str(out), "--data", str(data), "--split", "val"),
                *("--bins", str(bins), "--strategy", strategy, "--bootstrap", "0", "--json"),
            ]
        )
        assert code == 0, stderr
        rep = json.loads(stdout)
    api.choose_threshold(u_v, s_v, l_v, tau)
    e_v = 1.0 - s_v
    assert rep["raw"]["ece"] == ece(u_v, e_v, n_bins=bins, strategy=strategy)
    p_v = api.error_probability(u_v)
    assert rep["calibrated"]["ece"] == ece(p_v, e_v, n_bins=bins, strategy=strategy)
    rows = reliability_table(p_v, e_v, n_bins=bins, strategy=strategy)
    assert [r["count"] for r in rep["calibrated"]["reliability"]] == [r.count for r in rows]


@PROPERTY
@given(
    recs=logged_traffic(),
    lat=st.lists(st.tuples(st.floats(1.0, 500.0), st.floats(1.0, 500.0)), min_size=60),
    model=COST_MODEL,
)
def test_cli_cost_from_latency_uses_cal_and_val_means(
    recs: list[dict[str, Any]], lat: list[tuple[float, float]], model: str
) -> None:
    """c_small = 1, c_large = mean large latency / mean small latency on cal + val (Section 6.1)."""
    for r, (ls, ll) in zip(recs, lat):
        r["latency_small_ms"], r["latency_large_ms"] = ls, ll
    fit_rows = [r for r in recs if r["split"] in ("cal", "val")]
    ratio = float(
        np.mean([r["latency_large_ms"] for r in fit_rows])
        / np.mean([r["latency_small_ms"] for r in fit_rows])
    )
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "traffic.jsonl"
        out = Path(tmp) / "router.json"
        _write_jsonl(data, recs)
        code, _stdout, stderr = _run_cli(
            [
                *("fit", "--data", str(data), "--out", str(out), "--tau", "0"),
                *("--cost-from-latency", "--cost-model", model, "--json"),
            ]
        )
        if model == "routing" and not ratio > 1.0:
            assert code == 3, stderr
            return
        assert code == 0, stderr
        doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["c_small"] == 1.0
    assert doc["c_large"] == pytest.approx(ratio, rel=1e-14)  # summation order may differ
    assert doc["fit"]["costs"]["source"] == "latency"


# ---------------------------------------------------------------------------
# Extensions (not in the paper): online recalibration and the cascade helper
# ---------------------------------------------------------------------------


@PROPERTY
@given(
    stream=st.lists(st.tuples(st.one_of(UNIT, COARSE), st.sampled_from([0.0, 1.0])), max_size=150),
    window=st.integers(1, 40),
    refit_every=st.integers(1, 30),
    min_frac=st.floats(0.0, 1.0),
    cuts=st.lists(st.integers(0, 150), max_size=12),
)
def test_recalibrating_router_batches_equal_single_labels(
    stream: list[tuple[float, float]],
    window: int,
    refit_every: int,
    min_frac: float,
    cuts: list[int],
) -> None:
    """Documented: feeding labels one at a time or in batches gives identical states."""
    from ucci.online import CalibrationMonitor, RecalibratingRouter

    min_labels = max(1, min(window, round(min_frac * window)))
    u = np.array([a for a, _ in stream], dtype=np.float64)
    e = np.array([b for _, b in stream], dtype=np.float64)
    bounds = sorted({0, u.size, *(c for c in cuts if c <= u.size)})

    batched = RecalibratingRouter(
        0.5, window=window, refit_every=refit_every, min_labels=min_labels
    )
    monitor = CalibrationMonitor(None, window=window, min_count=min(window, 1))
    for a, b in zip(bounds, bounds[1:]):
        batched.update(u[a:b], e[a:b])
        monitor.record(u[a:b], e[a:b])

    # Reference: one label at a time, refit when both conditions first hold.
    history, since, fitted_on = [], 0, None
    for i in range(u.size):
        since += 1
        lo = max(0, i + 1 - window)
        if since >= refit_every and (i + 1 - lo) >= min_labels:
            history.append(i + 1)
            fitted_on = (lo, i + 1)
            since = 0
    assert batched.refit_history == tuple(history)
    assert batched.labels_since_refit == since
    wu, we = batched.window_arrays()
    np.testing.assert_array_equal(wu, u[max(0, u.size - window) :])
    np.testing.assert_array_equal(we, e[max(0, e.size - window) :])
    mp, me = monitor.window_arrays()
    np.testing.assert_array_equal(mp, wu)
    np.testing.assert_array_equal(me, we)
    if fitted_on is None:
        assert not batched.is_calibrated
    else:
        ref = IsotonicCalibrator().fit(
            u[fitted_on[0] : fitted_on[1]], e[fitted_on[0] : fitted_on[1]]
        )
        cal = batched.calibrator
        assert isinstance(cal, IsotonicCalibrator)
        np.testing.assert_array_equal(cal.x_, ref.x_)
        np.testing.assert_array_equal(cal.y_, ref.y_)


@PROPERTY
@given(
    cal=calibration_data(max_n=30, weighted=False),
    theta=st.one_of(UNIT, COARSE),
    us=st.lists(st.one_of(UNIT, COARSE), min_size=1, max_size=25),
    shadow=st.booleans(),
)
def test_cascade_logs_route_like_the_router_and_the_cli(
    cal: Any, theta: float, us: list[float], shadow: bool
) -> None:
    """Cascade decisions are Eq. 6 of the router; logged records replay through `ucci route`."""
    from ucci.integrations.cascade import Cascade, JsonlLogger

    u, e, _ = cal
    router = UCCIRouter().calibrate(u, e)
    router.theta = theta
    calls: list[str] = []

    def small_fn(q: float) -> tuple[str, float]:
        return f"small:{q!r}", q

    def large_fn(q: float) -> str:
        calls.append("large")
        return f"large:{q!r}"

    cascade: Cascade[float, str] = Cascade(
        router, small_fn, large_fn, lambda raw: raw, shadow_large=shadow
    )
    results = cascade.map(us)
    want = router.route(np.array(us))
    assert [r.escalated for r in results] == want.escalate.tolist()
    assert [r.p_hat for r in results] == want.p_hat.tolist()
    assert [r.answer for r in results] == [
        f"large:{q!r}" if esc else f"small:{q!r}" for q, esc in zip(us, want.escalate.tolist())
    ]
    assert len(calls) == (len(us) if shadow else int(want.escalate.sum()))
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "traffic.jsonl"
        with JsonlLogger(log) as logger:
            for i, r in enumerate(results):
                logger.write(r.to_record(f"q{i}", small_correct=1, large_correct=1))
        path = Path(tmp) / "router.json"
        router.save(path)
        code, stdout, stderr = _run_cli(
            ["route", "--router", str(path), "--data", str(log), "--json"]
        )
        assert code == 0, stderr
        replay = json.loads(stdout)["queries"]
        logged = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [q["p_hat"] for q in replay] == [rec["p_hat"] for rec in logged]
    assert [q["escalate"] for q in replay] == [rec["escalated"] for rec in logged]
