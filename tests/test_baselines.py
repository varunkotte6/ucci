"""Tests for ucci.baselines: comparison methods and ablations (Section 6.1, 6.3).

Every selection result is checked against an independent brute-force search
over all thresholds (or all subsets, for the oracle) on data whose scores are
multiples of 1/4, so every sum is exact in floating point and the reference
and the implementation must agree exactly, including tie-breaking.
"""

from __future__ import annotations

import itertools
import math
import time
import warnings

import numpy as np
import pytest

from ucci import (
    DEFAULT_GRID,
    InfeasibleTargetError,
    IsotonicCalibrator,
    ThresholdChoice,
    UCCIRouter,
    policy_accuracy,
    policy_cost,
    routed_micro_f1,
    select_threshold,
    select_threshold_for_budget,
    token_margin_uncertainty,
)
from ucci.baselines import (
    DEFAULT_DELTA_GRID,
    AlwaysLarge,
    AlwaysSmall,
    CalibratedThresholdRouter,
    ComparisonRow,
    EntropyThresholdRouter,
    FrugalGPTStyleRouter,
    IdentityCalibrator,
    MethodSpec,
    Oracle,
    PlattCalibrator,
    RawThresholdRouter,
    SplitConformalRouter,
    SplitData,
    TemperatureScalingCalibrator,
    ablation_methods,
    compare_routers,
    exact_threshold_candidates,
    format_comparison,
    mean_max_prob,
    mean_token_entropy,
    table2_methods,
    token_entropies,
    token_max_probs,
)

C_S, C_L = 1.0, 3.02


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _acc(mask, small, large):
    """Mean score of the returned answers (exact for dyadic scores, small n)."""
    return float((small[~mask].sum() + large[mask].sum()) / small.size)


def _cost(k, n, c_s=C_S, c_l=C_L, cost_model="routing"):
    r = k / n
    return c_s * (1.0 - r) + c_l * r if cost_model == "routing" else c_s + c_l * r


def brute_threshold(
    scores,
    small,
    large,
    *,
    tau=None,
    budget=None,
    cands=None,
    c_s=C_S,
    c_l=C_L,
    cost_model="routing",
):
    """Independent Eq. 7 search over all thresholds for "escalate if s > t"."""
    s = np.asarray(scores, dtype=float)
    small = np.asarray(small, dtype=float)
    large = np.asarray(large, dtype=float)
    if cands is None:
        cands = [-math.inf, *sorted(set(s.tolist())), math.inf]
    n = s.size
    rows = []
    for t in cands:
        mask = s > t
        k = int(mask.sum())
        rows.append((t, k, _acc(mask, small, large), _cost(k, n, c_s, c_l, cost_model)))
    if tau is not None:
        feas = [r for r in rows if r[2] >= tau]
        if not feas:
            return None
        return min(feas, key=lambda r: (r[3], -r[2], -r[0]))
    feas = [r for r in rows if r[3] <= budget]
    if not feas:
        return None
    return min(feas, key=lambda r: (-r[2], r[3], -r[0]))


def dyadic_problem(rng, n, n_levels=6, graded=True, tie_heavy=False):
    """Random routing problem with scores in {0, 1/4, ..., 1} and tied signals."""
    if tie_heavy:
        s = rng.integers(0, n_levels, n).astype(float) / n_levels
    else:
        s = np.round(rng.random(n), 2)
    levels = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) if graded else np.array([0.0, 1.0])
    p_small_bad = np.clip(s, 0.05, 0.95)
    small = np.where(rng.random(n) < p_small_bad, levels[rng.integers(0, len(levels) - 1, n)], 1.0)
    large = np.where(rng.random(n) < 0.15, levels[rng.integers(0, len(levels), n)], 1.0)
    return s, small, large


def synthetic_split(rng, n, *, signals=True):
    """Split with a calibrated-ish u, a noisy entropy and a max-prob signal."""
    u = rng.beta(2.0, 5.0, n)
    p_wrong = np.clip(1.4 * u**1.5, 0.0, 1.0)
    small = (rng.random(n) >= p_wrong).astype(float)
    large = (rng.random(n) >= 0.05 + 0.1 * u).astype(float)
    sig = {"u": u}
    if signals:
        sig["entropy"] = 3.0 * u + rng.normal(0.0, 0.25, n)
        sig["max_prob"] = np.clip(1.0 - 0.6 * u + rng.normal(0.0, 0.05, n), 0.0, 1.0)
    return SplitData(sig, small, large)


# ---------------------------------------------------------------------------
# Signals (Section 6.3)
# ---------------------------------------------------------------------------


def test_entropy_uniform_point_mass_and_binary():
    assert mean_token_entropy([np.log(np.full(8, 1 / 8))]) == pytest.approx(math.log(8))
    assert mean_token_entropy([[0.0, -np.inf, -np.inf]]) == 0.0
    p = 0.3
    h = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    assert mean_token_entropy([[math.log(p), math.log(1 - p)]]) == pytest.approx(h, abs=1e-15)


def test_entropy_top_k_is_renormalized():
    # Top-2 of the full distribution (0.5, 0.3, 0.2): renormalized to (5/8, 3/8).
    full = np.log([0.5, 0.3, 0.2])
    top2 = full[:2]
    q = np.array([0.5, 0.3]) / 0.8
    assert mean_token_entropy([top2]) == pytest.approx(-(q * np.log(q)).sum(), abs=1e-15)
    assert mean_token_entropy([full]) == pytest.approx(-(np.exp(full) * full).sum(), abs=1e-15)


def test_entropy_is_shift_invariant_so_logits_work():
    rng = np.random.default_rng(0)
    logits = rng.normal(0.0, 3.0, (5, 50))
    logsm = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    assert np.allclose(token_entropies(logits), token_entropies(logsm), atol=1e-12)


def test_entropy_accepts_2d_ragged_and_object_arrays():
    rows = [np.log([0.6, 0.4]), np.log([0.2, 0.3, 0.5]), [0.0]]
    ragged = token_entropies(rows)
    obj = np.empty(3, dtype=object)
    obj[:] = rows
    assert np.array_equal(ragged, token_entropies(obj))
    assert ragged[2] == 0.0
    mat = np.log([[0.6, 0.4], [0.9, 0.1]])
    assert np.array_equal(token_entropies(mat), token_entropies(list(mat)))
    assert mean_token_entropy(rows) == pytest.approx(ragged.mean())


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        ([], "empty"),
        ([[]], "empty"),
        ([[math.nan, 0.0]], "NaN"),
        ([[math.inf, 0.0]], r"\+inf"),
        ([[-math.inf, -math.inf]], "positive probability"),
        (np.log([0.5, 0.5]), "2-D"),
        ([0.1, 0.2], "1-D"),
    ],
)
def test_signal_input_errors(bad, match):
    with pytest.raises(ValueError, match=match):
        mean_token_entropy(bad)
    with pytest.raises(ValueError, match=match):
        mean_max_prob(bad, renormalize=True)


def test_max_prob_exact_versus_renormalized():
    top2 = [np.log([0.5, 0.3]), np.log([0.9, 0.05])]
    assert token_max_probs(top2).tolist() == pytest.approx([0.5, 0.9], abs=1e-15)
    assert token_max_probs(top2, renormalize=True).tolist() == pytest.approx(
        [0.5 / 0.8, 0.9 / 0.95], abs=1e-15
    )
    assert mean_max_prob(top2) == pytest.approx(0.7, abs=1e-15)


def test_max_prob_rejects_logits_unless_renormalized():
    logits = [[2.0, 1.0, 0.0]]
    with pytest.raises(ValueError, match="renormalize=True"):
        mean_max_prob(logits)
    soft = np.exp([2.0, 1.0, 0.0]) / np.exp([2.0, 1.0, 0.0]).sum()
    assert mean_max_prob(logits, renormalize=True) == pytest.approx(soft[0], abs=1e-15)


def test_max_prob_and_margin_agree_on_binary_distributions():
    # With two candidates p1 + p2 = 1, so m_t = 2 p1 - 1 and u = 2 (1 - mean p1).
    rng = np.random.default_rng(1)
    p1 = rng.uniform(0.5, 1.0, 20)
    pairs = list(zip(p1, 1 - p1))
    lps = [np.log([a, b]) for a, b in pairs]
    assert token_margin_uncertainty(pairs) == pytest.approx(2 * (1 - mean_max_prob(lps)), abs=1e-12)


# ---------------------------------------------------------------------------
# Single-model anchors and the oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cost_model", ["routing", "sequential"])
def test_always_small_and_large(cost_model):
    small = np.array([1.0, 0.0, 1.0, 0.0])
    large = np.array([1.0, 1.0, 0.0, 1.0])
    x = np.zeros(4)
    lo = AlwaysSmall(C_S, C_L, cost_model).choose_threshold(x, small, large, tau=0.99)
    hi = AlwaysLarge(C_S, C_L, cost_model).choose_threshold(x, small, large, budget=1.0)
    assert (lo.theta, lo.cost, lo.accuracy, lo.escalation_rate) == (
        math.inf,
        C_S,
        0.5,
        0.0,
    )
    expected_hi = C_L if cost_model == "routing" else C_S + C_L
    assert (hi.theta, hi.cost, hi.accuracy, hi.escalation_rate) == (
        -math.inf,
        expected_hi,
        0.75,
        1.0,
    )
    assert AlwaysSmall().escalate([0.3, 5.0]).tolist() == [False, False]
    assert AlwaysLarge().escalate([0.3, 5.0]).tolist() == [True, True]
    assert AlwaysSmall().calibrate([0.1], [1]) is not None


def test_oracle_default_is_small_wrong_and_large_right():
    small = np.array([1, 0, 0, 1, 0], dtype=float)
    large = np.array([1, 1, 0, 0, 1], dtype=float)
    mask = Oracle().escalate(small, large)
    assert mask.tolist() == [False, True, False, False, True]
    res = Oracle().evaluate(small, large)
    assert res.accuracy == pytest.approx(0.8)
    assert math.isnan(res.theta)


@pytest.mark.parametrize("seed", range(25))
def test_oracle_is_the_exact_minimum_cost_and_maximum_accuracy(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 10))
    levels = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    small = levels[rng.integers(0, 5, n)]
    large = levels[rng.integers(0, 5, n)]
    subsets = []
    for bits in itertools.product([False, True], repeat=n):
        m = np.array(bits)
        subsets.append((int(m.sum()), _acc(m, small, large)))
    best_acc = max(a for _, a in subsets)
    tau = float(rng.choice([a for _, a in subsets]))
    k_min = min(k for k, a in subsets if a >= tau)
    mask = Oracle().escalate(small, large, tau=tau)
    assert int(mask.sum()) == k_min
    assert _acc(mask, small, large) >= tau
    budget = _cost(int(rng.integers(0, n + 1)), n) + 1e-9
    acc_max = max(a for k, a in subsets if _cost(k, n) <= budget)
    res = Oracle().evaluate(small, large, budget=budget)
    assert res.accuracy == acc_max
    assert res.cost <= budget
    assert Oracle().evaluate(small, large).accuracy == best_acc


def test_oracle_infeasible_and_argument_errors():
    small, large = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    with pytest.raises(InfeasibleTargetError, match="mean\\(max"):
        Oracle().escalate(small, large, tau=0.75)
    with pytest.raises(InfeasibleTargetError):
        Oracle().evaluate(small, large, budget=0.5)
    with pytest.raises(ValueError, match="only one"):
        Oracle().escalate(small, large, tau=0.5, budget=2.0)
    with pytest.raises(ValueError, match="length"):
        Oracle().escalate([1.0], [1.0, 0.0])


# ---------------------------------------------------------------------------
# Raw-score threshold routers (entropy threshold, FrugalGPT-style)
# ---------------------------------------------------------------------------


def test_exact_threshold_candidates():
    c = exact_threshold_candidates([0.3, 0.1, 0.3, 2.0])
    assert c.tolist() == [-math.inf, 0.1, 0.3, 2.0, math.inf]


def test_raw_threshold_known_answer():
    ent = np.array([0.1, 0.5, 0.9, 1.3, 2.0])
    small = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
    large = np.ones(5)
    r = EntropyThresholdRouter()
    ch = r.choose_threshold(ent, small, large, tau=0.8)
    # Escalating only 2.0 gives accuracy 0.8 at the lowest cost.
    assert ch.theta == 1.3
    assert (ch.accuracy, ch.escalation_rate) == (0.8, 0.2)
    assert ch.cost == pytest.approx(1.0 + 2.02 * 0.2)
    assert r.escalate([1.3, 1.31, -5.0, 99.0]).tolist() == [False, True, False, True]
    assert r.choose_threshold(ent, small, large, tau=1.0).theta == 0.5
    assert r.choose_threshold(ent, small, large, tau=0.6).theta == math.inf
    # If the large model misses the most uncertain query, reaching 0.8 needs
    # escalating 0.9 and 1.3 as well: the cheapest feasible threshold is 0.5.
    worse = r.choose_threshold(ent, small, np.array([0.0, 1.0, 1.0, 1.0, 0.0]), tau=0.8)
    assert (worse.theta, worse.accuracy, worse.escalation_rate) == (0.5, 0.8, 0.6)


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("cost_model", ["routing", "sequential"])
def test_raw_threshold_matches_brute_force(seed, cost_model):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 60))
    s, small, large = dyadic_problem(rng, n, graded=seed % 2 == 0, tie_heavy=seed % 3 == 0)
    accs = sorted({_acc(s > t, small, large) for t in exact_threshold_candidates(s)})
    tau = float(rng.choice(accs))
    ref = brute_threshold(s, small, large, tau=tau, cost_model=cost_model)
    ch = EntropyThresholdRouter(C_S, C_L, cost_model).choose_threshold(s, small, large, tau)
    assert (ch.theta, ch.accuracy) == (ref[0], ref[2])
    assert ch.cost == ref[3]
    k = int(rng.integers(0, n + 1))
    budget = _cost(k, n, cost_model=cost_model) + 1e-9
    ref = brute_threshold(s, small, large, budget=budget, cost_model=cost_model)
    ch = EntropyThresholdRouter(C_S, C_L, cost_model).choose_threshold(
        s, small, large, budget=budget
    )
    assert (ch.theta, ch.accuracy, ch.cost) == (ref[0], ref[2], ref[3])


@pytest.mark.parametrize("seed", range(15))
def test_confidence_direction_is_the_mirror_image(seed):
    rng = np.random.default_rng(100 + seed)
    s, small, large = dyadic_problem(rng, 40, tie_heavy=True)
    tau = float(
        np.quantile([_acc(s > t, small, large) for t in exact_threshold_candidates(s)], 0.7)
    )
    unc = EntropyThresholdRouter()
    conf = FrugalGPTStyleRouter()
    a = unc.choose_threshold(s, small, large, tau)
    b = conf.choose_threshold(-s, small, large, tau)
    assert b.theta == -a.theta
    assert (a.cost, a.accuracy) == (b.cost, b.accuracy)
    q = rng.normal(size=100)
    assert np.array_equal(unc.escalate(q), conf.escalate(-q))


def test_metric_path_matches_default_path():
    rng = np.random.default_rng(7)
    s, small, large = dyadic_problem(rng, 50, graded=True)
    tau = 0.9 * _acc(np.ones(50, bool), small, large)
    r = EntropyThresholdRouter()
    a = r.choose_threshold(s, small, large, tau)
    b = r.choose_threshold(s, small, large, tau, metric=lambda m: _acc(m, small, large))
    assert a == b


def test_corpus_metric_micro_f1():
    rng = np.random.default_rng(8)
    n = 60
    s = rng.random(n)
    small_counts = np.column_stack(
        [rng.integers(0, 3, n), rng.integers(0, 2, n), rng.integers(0, 2, n)]
    )
    large_counts = np.column_stack(
        [small_counts[:, 0] + small_counts[:, 2], np.zeros(n, int), np.zeros(n, int)]
    )
    f1 = routed_micro_f1(small_counts, large_counts)
    r = EntropyThresholdRouter()
    ch = r.choose_threshold(s, np.zeros(n), np.zeros(n), tau=0.95, metric=f1)
    assert ch.accuracy == f1(s > ch.theta)
    assert ch.accuracy >= 0.95
    feasible = [t for t in exact_threshold_candidates(s) if f1(s > t) >= 0.95]
    assert ch.theta == max(feasible)


def test_grid_candidates_reproduce_core_select_threshold():
    rng = np.random.default_rng(3)
    p = rng.random(500)
    small = (rng.random(500) > p).astype(float)
    large = np.ones(500)
    core = select_threshold(p, small, large, 0.85, C_S, C_L, DEFAULT_GRID)
    ours = RawThresholdRouter(candidates=DEFAULT_GRID).choose_threshold(p, small, large, 0.85)
    assert core == ours
    core_b = select_threshold_for_budget(p, small, large, 2.0, C_S, C_L, DEFAULT_GRID)
    ours_b = RawThresholdRouter(candidates=DEFAULT_GRID).choose_threshold(
        p, small, large, budget=2.0
    )
    assert core_b == ours_b


def test_raw_threshold_validation_reports_same_numbers_as_policy_functions():
    rng = np.random.default_rng(4)
    s, small, large = dyadic_problem(rng, 80)
    r = EntropyThresholdRouter()
    ch = r.choose_threshold(s, small, large, tau=0.8)
    mask = r.escalate(s)
    assert ch.accuracy == policy_accuracy(mask, small, large)
    assert ch.cost == policy_cost(mask, C_S, C_L)
    assert ch.escalation_rate == mask.mean()


def test_raw_threshold_infeasible_and_errors():
    s = np.array([0.1, 0.2, 0.3])
    small = np.array([0.0, 1.0, 0.0])
    large = np.array([0.0, 1.0, 1.0])
    r = EntropyThresholdRouter()
    with pytest.raises(InfeasibleTargetError, match="best accuracy any candidate"):
        r.choose_threshold(s, small, large, tau=0.9)
    with pytest.raises(InfeasibleTargetError, match="cheapest candidate costs"):
        r.choose_threshold(s, small, large, budget=0.5)
    with pytest.raises(ValueError, match="pass tau"):
        r.choose_threshold(s, small, large)
    with pytest.raises(ValueError, match="only one"):
        r.choose_threshold(s, small, large, 0.5, budget=2.0)
    with pytest.raises(ValueError, match="length mismatch"):
        r.choose_threshold(s[:2], small, large, 0.5)
    with pytest.raises(ValueError, match="NaN"):
        r.choose_threshold([0.1, math.nan, 0.3], small, large, 0.5)
    with pytest.raises(ValueError, match="Theorem 1"):
        EntropyThresholdRouter(2.0, 1.0)
    with pytest.raises(ValueError, match="direction"):
        RawThresholdRouter(direction="up")
    with pytest.raises(ValueError, match="positive"):
        EntropyThresholdRouter(0.0, 1.0)
    with pytest.raises(RuntimeError, match="choose_threshold"):
        EntropyThresholdRouter().escalate([0.1])
    with pytest.raises(ValueError, match="NaN"):
        RawThresholdRouter(candidates=[0.1, math.nan])
    # Sequential cost allows c_large <= c_small (escalation still costs more).
    EntropyThresholdRouter(2.0, 1.0, "sequential").choose_threshold(s, small, large, 0.6)


def test_raw_threshold_scales_to_large_validation_sets():
    rng = np.random.default_rng(5)
    n = 200_000
    s = rng.random(n)
    small = (rng.random(n) > s).astype(float)
    large = np.ones(n)
    start = time.perf_counter()
    ch = EntropyThresholdRouter().choose_threshold(s, small, large, tau=0.9)
    assert time.perf_counter() - start < 10.0
    assert ch.accuracy >= 0.9
    assert policy_accuracy(s > ch.theta, small, large) == ch.accuracy


# ---------------------------------------------------------------------------
# Split conformal routing
# ---------------------------------------------------------------------------


def test_conformal_quantile_formula():
    u = np.arange(1, 10) / 10.0  # 0.1 .. 0.9, all small-correct: n = 9
    r = SplitConformalRouter().calibrate(np.r_[u, 0.95], np.r_[np.zeros(9), 1.0])
    assert r.scores_.tolist() == u.tolist()
    assert r.quantile(0.2) == 0.8  # k = ceil(10 * 0.8) = 8
    assert r.quantile(0.05) == math.inf  # k = ceil(9.5) = 10 > 9
    assert r.quantile(0.5) == 0.5
    assert r.keep_probability_bound(0.2) == 0.8
    assert r.keep_probability_bound(0.05) == 1.0


def test_conformal_reads_delta_as_a_decimal():
    # With n = 9, (n + 1)(1 - 0.7) evaluates to 3.0000000000000004 in binary
    # floating point, so a naive ceiling gives k = 4; the intended k is 3.
    assert math.ceil(10 * (1 - 0.7)) == 4
    r = SplitConformalRouter().calibrate(np.arange(1, 10) / 10.0, np.zeros(9))
    assert r.quantile(0.7) == 0.3
    assert r.keep_probability_bound(0.7) == pytest.approx(0.3)


@pytest.mark.parametrize("delta", [0.05, 0.1, 0.2])
@pytest.mark.parametrize("ties", [False, True])
def test_conformal_coverage_on_exchangeable_data(delta, ties):
    rng = np.random.default_rng(int(delta * 1000) + ties)
    keep, bound = [], []
    for _ in range(1500):
        u = rng.beta(2.0, 5.0, 60)
        e = (rng.random(60) < 0.3).astype(float)
        u_new = rng.beta(2.0, 5.0, 100)  # fresh small-correct queries
        if ties:
            u, u_new = np.round(u, 1), np.round(u_new, 1)
        r = SplitConformalRouter().calibrate(u, e)
        q = r.quantile(delta)
        keep.append(np.mean(u_new <= q))
        bound.append(r.keep_probability_bound(delta))
    keep_rate, se = float(np.mean(keep)), float(np.std(keep) / math.sqrt(len(keep)))
    assert min(bound) >= 1 - delta
    assert keep_rate >= 1 - delta - 4 * se
    if not ties:
        # Continuous scores: the keep probability is exactly k / (n + 1).
        assert abs(keep_rate - float(np.mean(bound))) <= 4 * se


@pytest.mark.parametrize("seed", range(20))
def test_conformal_selection_matches_brute_force_over_deltas(seed):
    rng = np.random.default_rng(200 + seed)
    u_cal = np.round(rng.random(80), 2)
    e_cal = (rng.random(80) < u_cal).astype(float)
    s, small, large = dyadic_problem(rng, 50)
    deltas = DEFAULT_DELTA_GRID if seed % 2 else np.array([0.01, 0.1, 0.25, 0.5, 0.9])
    r = SplitConformalRouter(deltas=deltas).calibrate(u_cal, e_cal)
    qs = [r.quantile(d) for d in deltas]
    accs = sorted({_acc(s > q, small, large) for q in qs})
    tau = float(rng.choice(accs))
    ref = brute_threshold(s, small, large, tau=tau, cands=sorted(set(qs)))
    ch = r.choose_threshold(s, small, large, tau)
    assert (ch.theta, ch.accuracy, ch.cost) == (ref[0], ref[2], ref[3])
    assert r.alpha_ == ch.theta
    same = [d for d, q in zip(deltas, qs) if q == ch.theta]
    assert r.delta_ == min(same)
    assert np.array_equal(r.escalate(s), s > ch.theta)


def test_conformal_errors():
    with pytest.raises(ValueError, match="0 or 1"):
        SplitConformalRouter().calibrate([0.1, 0.2], [0.5, 0.0])
    with pytest.raises(ValueError, match="answered correctly"):
        SplitConformalRouter().calibrate([0.1, 0.2], [1.0, 1.0])
    with pytest.raises(ValueError, match="sample weights"):
        SplitConformalRouter().calibrate([0.1], [0.0], sample_weight=[1.0])
    with pytest.raises(ValueError, match="strictly between"):
        SplitConformalRouter(deltas=[0.0, 0.1])
    with pytest.raises(RuntimeError, match="calibrate"):
        SplitConformalRouter().quantile(0.1)
    r = SplitConformalRouter().calibrate([0.1, 0.2], [0.0, 0.0])
    with pytest.raises(ValueError, match="strictly between"):
        r.quantile(1.0)


# ---------------------------------------------------------------------------
# Calibrators (Appendix B.4 ablation, Platt extension)
# ---------------------------------------------------------------------------


def _temperature_nll(t, u, e, w, eps=1e-6):
    c = np.clip(u, eps, 1 - eps)
    x = (np.log(c) - np.log1p(-c)) / t
    return float(np.sum(w * (np.logaddexp(0.0, x) - e * x)) / np.sum(w))


@pytest.mark.parametrize("seed", range(6))
def test_temperature_reaches_the_dense_grid_minimum(seed):
    rng = np.random.default_rng(seed)
    n = 800
    u = rng.random(n)
    true_t = rng.uniform(0.3, 4.0)
    z = np.log(u) - np.log1p(-u)
    e = (rng.random(n) < 1 / (1 + np.exp(-z / true_t))).astype(float)
    w = rng.uniform(0.2, 2.0, n) if seed % 2 else np.ones(n)
    cal = TemperatureScalingCalibrator().fit(u, e, w)
    grid = np.logspace(-3, 3, 20001)
    nll = np.array([_temperature_nll(t, u, e, w) for t in grid])
    j = int(np.argmin(nll))
    fine = np.linspace(grid[max(j - 1, 0)], grid[min(j + 1, grid.size - 1)], 2001)
    best = min(nll[j], min(_temperature_nll(t, u, e, w) for t in fine))
    assert cal.nll_ <= best + 1e-12
    assert cal.nll_ == pytest.approx(_temperature_nll(cal.temperature_, u, e, w), abs=1e-15)
    assert not cal.at_bound_


def test_temperature_recovers_the_true_temperature():
    rng = np.random.default_rng(11)
    u = rng.random(200_000)
    z = np.log(u) - np.log1p(-u)
    e = (rng.random(u.size) < 1 / (1 + np.exp(-z / 2.5))).astype(float)
    assert TemperatureScalingCalibrator().fit(u, e).temperature_ == pytest.approx(2.5, rel=0.03)


def test_temperature_map_is_monotone_with_fixed_point_one_half():
    rng = np.random.default_rng(12)
    u = rng.random(300)
    e = (rng.random(300) < u).astype(float)
    cal = TemperatureScalingCalibrator().fit(u, e)
    grid = np.linspace(0.0, 1.0, 1001)
    p = cal.predict(grid)
    assert np.all(np.diff(p) >= 0.0)
    assert cal.predict(0.5) == pytest.approx(0.5, abs=1e-15)
    assert isinstance(cal.predict(0.2), float)
    assert cal(np.array([0.2])).shape == (1,)


def test_temperature_weights_equal_repetition():
    rng = np.random.default_rng(13)
    u = rng.random(50)
    e = (rng.random(50) < u).astype(float)
    w = rng.integers(1, 4, 50)
    a = TemperatureScalingCalibrator().fit(u, e, w)
    b = TemperatureScalingCalibrator().fit(np.repeat(u, w), np.repeat(e, w))
    assert a.temperature_ == pytest.approx(b.temperature_, rel=1e-10)


def test_temperature_at_bound_warns():
    rng = np.random.default_rng(14)
    u = rng.random(200)
    e = (u < 0.5).astype(float)  # errors fall as u rises: best T is negative
    with pytest.warns(RuntimeWarning, match="outside"):
        cal = TemperatureScalingCalibrator().fit(u, e)
    assert cal.at_bound_
    assert cal.temperature_ == pytest.approx(1e3)


def test_temperature_errors():
    with pytest.raises(RuntimeError, match="not fitted"):
        TemperatureScalingCalibrator().predict([0.2])
    with pytest.raises(ValueError, match="eps"):
        TemperatureScalingCalibrator(eps=0.6)
    with pytest.raises(ValueError, match="t_min"):
        TemperatureScalingCalibrator(t_min=2.0, t_max=1.0)
    with pytest.raises(ValueError, match="length"):
        TemperatureScalingCalibrator().fit([0.1, 0.2], [1.0])


def _platt_problem(rng, n=400):
    u = rng.random(n)
    e = (rng.random(n) < 1 / (1 + np.exp(-(4.0 * u - 2.5)))).astype(float)
    return u, e


@pytest.mark.parametrize("c", [1e10, 1.0, 0.05])
@pytest.mark.parametrize("weighted", [False, True])
def test_platt_matches_sklearn_logistic_regression(c, weighted):
    lm = pytest.importorskip("sklearn.linear_model")
    rng = np.random.default_rng(21)
    u, e = _platt_problem(rng)
    w = rng.uniform(0.1, 3.0, u.size) if weighted else None
    ours = PlattCalibrator(l2=0.0 if c == 1e10 else 1.0 / c).fit(u, e, w)
    ref = lm.LogisticRegression(C=c, tol=1e-12, max_iter=100_000).fit(
        u[:, None], e, sample_weight=w
    )
    grid = np.linspace(-0.5, 1.5, 201)
    assert np.allclose(ours.predict(grid), ref.predict_proba(grid[:, None])[:, 1], atol=1e-6)
    assert ours.a_ == pytest.approx(float(ref.coef_[0, 0]), rel=1e-5)
    assert ours.b_ == pytest.approx(float(ref.intercept_[0]), rel=1e-5, abs=1e-6)


def _platt_gradient(cal, u, t, w, l2):
    p = 1 / (1 + np.exp(-(cal.a_ * u + cal.b_)))
    return np.array([np.sum(w * (p - t) * u) + l2 * cal.a_, np.sum(w * (p - t))])


def test_platt_first_order_conditions_hold():
    rng = np.random.default_rng(22)
    u, e = _platt_problem(rng)
    w = np.ones(u.size)
    cal = PlattCalibrator().fit(u, e)
    assert np.abs(_platt_gradient(cal, u, e, w, 0.0)).max() < 1e-8
    soft = np.clip(e * 0.8 + 0.1, 0, 1)
    cal = PlattCalibrator(l2=0.5).fit(u, soft)
    assert np.abs(_platt_gradient(cal, u, soft, w, 0.5)).max() < 1e-8


def test_platt_target_smoothing_uses_platt_targets():
    rng = np.random.default_rng(23)
    u, e = _platt_problem(rng, 200)
    n_pos, n_neg = e.sum(), (1 - e).sum()
    t = np.where(e == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    a = PlattCalibrator(target_smoothing=True).fit(u, e)
    b = PlattCalibrator().fit(u, t)
    assert (a.a_, a.b_) == pytest.approx((b.a_, b.b_), rel=1e-10)


def test_platt_non_existent_fits_are_reported():
    u = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="separates"):
        PlattCalibrator().fit(u, [0, 0, 1, 1])
    with pytest.raises(ValueError, match="all labels are equal"):
        PlattCalibrator().fit(u, [1, 1, 1, 1])
    with pytest.raises(ValueError, match="constant"):
        PlattCalibrator().fit([0.2, 0.2, 0.2], [0, 1, 0])
    with pytest.raises(ValueError, match="constant"):
        PlattCalibrator(target_smoothing=True).fit([0.2, 0.2, 0.2], [0, 1, 0])
    flat = PlattCalibrator(l2=1.0).fit([0.2, 0.2, 0.2, 0.2], [0, 1, 1, 1])
    assert flat.a_ == pytest.approx(0.0, abs=1e-12)
    assert flat.predict(0.9) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="binary"):
        PlattCalibrator(target_smoothing=True).fit(u, [0, 0.5, 1, 1])
    fitted = PlattCalibrator(target_smoothing=True).fit(u, [0, 0, 1, 1])
    assert fitted.predict(0.4) > fitted.predict(0.1)
    assert PlattCalibrator(l2=1.0).fit(u, [0, 0, 1, 1]).a_ > 0
    with pytest.raises(ValueError, match="l2"):
        PlattCalibrator(l2=-1.0)
    with pytest.raises(RuntimeError, match="not fitted"):
        PlattCalibrator().predict(0.1)


def test_temperature_edge_cases():
    # u = 0.5 everywhere: the NLL does not depend on T, which stays at 1.
    cal = TemperatureScalingCalibrator().fit(np.full(10, 0.5), np.r_[np.ones(5), np.zeros(5)])
    assert cal.temperature_ == 1.0
    assert not cal.at_bound_
    # Labels that jump from 0 to 1 at u = 0.5: the NLL keeps falling as T -> 0.
    u = np.linspace(0.05, 0.95, 40)
    with pytest.warns(RuntimeWarning, match="outside"):
        cal = TemperatureScalingCalibrator(t_min=0.01).fit(u, (u > 0.5).astype(float))
    assert cal.temperature_ == pytest.approx(0.01)
    assert cal.at_bound_


@pytest.mark.parametrize(
    "labels",
    [
        [0.0, 0.0, 0.5, 1.0, 1.0],  # soft label on the cut point
        [1.0, 1.0, 0.3, 0.0, 0.0],  # the mirror image
        [0.0, 0.0, 1.0, 1.0, 1.0],  # complete separation
    ],
)
def test_platt_detects_separation_exactly(labels):
    u = np.array([0.1, 0.2, 0.25, 0.3, 0.4])
    with pytest.raises(ValueError, match="separates"):
        PlattCalibrator().fit(u, labels)
    # Quasi-separation with a tie at the boundary value.
    with pytest.raises(ValueError, match="separates"):
        PlattCalibrator().fit([0.1, 0.2, 0.2, 0.3], [0, 0, 1, 1])
    # Overlapping soft labels: the fit exists and satisfies its first-order conditions.
    soft = [0.0, 0.2, 0.5, 0.9, 1.0]
    cal = PlattCalibrator().fit(u, soft)
    assert np.abs(_platt_gradient(cal, u, np.array(soft), np.ones(5), 0.0)).max() < 1e-8
    with pytest.raises(ValueError, match="all labels are equal"):
        PlattCalibrator(l2=1.0).fit(u, np.zeros(5))
    with pytest.raises(ValueError, match="max_iter"):
        PlattCalibrator(max_iter=0)
    with pytest.raises(ValueError, match="did not converge"):
        PlattCalibrator(max_iter=1).fit(u, soft)


def test_separation_check_matches_the_definition():
    from ucci.baselines import _separable

    rng = np.random.default_rng(25)
    for _ in range(300):
        n = int(rng.integers(2, 7))
        u = rng.integers(0, 4, n).astype(float)
        t = rng.choice([0.0, 0.5, 1.0], n)
        cuts = np.r_[np.unique(u), (np.unique(u)[:-1] + np.unique(u)[1:]) / 2]
        expected = any(
            (np.all(t[u < c] == 0) and np.all(t[u > c] == 1))
            or (np.all(t[u < c] == 1) and np.all(t[u > c] == 0))
            for c in cuts
        )
        assert _separable(u, t) == expected


@pytest.mark.parametrize(
    "seed, n, slope, shift, smoothing",
    [(55, 2011, 9.0, 3.0, True), (155, 2011, 9.0, 3.0, True), (111, 400, 4.0, 2.5, False)],
)
def test_platt_converges_where_the_line_search_stalls(seed, n, slope, shift, smoothing):
    # Regression: on these ordinary, overlapping samples the Newton decrement
    # at the optimum sat above an absolute stopping rule (1e-24 * |f|) but
    # below what the objective resolves in double precision, so the line
    # search took ever smaller steps until max_iter and the fit was reported
    # as non-convergent (about 2% of random samples, with or without
    # target smoothing).
    rng = np.random.default_rng(seed)
    u = rng.beta(2.0, 4.0, n) if n == 2011 else rng.random(n)
    e = (rng.random(n) < 1 / (1 + np.exp(-(slope * u - shift)))).astype(float)
    t = e
    if smoothing:
        n_pos, n_neg = e.sum(), (1 - e).sum()
        t = np.where(e == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
    cal = PlattCalibrator(target_smoothing=smoothing).fit(u, e)
    assert cal.n_iter_ < 15
    assert np.abs(_platt_gradient(cal, u, t, np.ones(n), 0.0)).max() < 1e-8


def test_platt_non_convergence_hint_matches_the_settings():
    u = np.array([0.1, 0.2, 0.25, 0.3, 0.4])
    with pytest.raises(ValueError, match=r"max_iter=1 .*l2 > 0 or target_smoothing=True"):
        PlattCalibrator(max_iter=1).fit(u, [0.0, 0.2, 0.5, 0.9, 1.0])
    with pytest.raises(ValueError, match=r"did not converge.*; set l2 > 0, or raise max_iter"):
        PlattCalibrator(max_iter=1, target_smoothing=True).fit(u, [0, 0, 1, 1, 1])


def test_platt_handles_wide_score_ranges():
    rng = np.random.default_rng(24)
    ent = rng.uniform(0.0, 12.0, 2000)
    e = (rng.random(2000) < 1 / (1 + np.exp(-(0.8 * ent - 5.0)))).astype(float)
    cal = PlattCalibrator().fit(ent, e)
    assert cal.n_iter_ < 30
    assert cal.a_ == pytest.approx(0.8, rel=0.15)


def test_reprs_and_metric_checks():
    objs = [
        AlwaysSmall(),
        Oracle(),
        EntropyThresholdRouter(),
        FrugalGPTStyleRouter(),
        SplitConformalRouter(),
        IdentityCalibrator(),
        TemperatureScalingCalibrator(),
        PlattCalibrator(),
        CalibratedThresholdRouter(IsotonicCalibrator()),
    ]
    for obj in objs:
        assert type(obj).__name__ in repr(obj)
    with pytest.raises(ValueError, match="non-finite"):
        AlwaysSmall().choose_threshold([0.0], [1.0], [1.0], metric=lambda m: math.nan)
    with pytest.raises(ValueError, match="must return a number"):
        AlwaysSmall().choose_threshold([0.0], [1.0], [1.0], metric=lambda m: "high")
    with pytest.raises(ValueError, match="sequence"):
        mean_token_entropy(5)


def test_calibrated_router_rejects_wrong_output_shape():
    class Scalar:
        def fit(self, u, e, sample_weight=None):
            return self

        def predict(self, u):
            return np.array([0.5])

    r = CalibratedThresholdRouter(Scalar()).calibrate([0.1], [0.0])
    with pytest.raises(ValueError, match="shape"):
        r.error_probability([0.1, 0.2])


def test_ucci_row_adapter_exposes_theta():
    r = table2_methods()[0].factory(C_S, C_L, "routing")
    rng = np.random.default_rng(42)
    sp = synthetic_split(rng, 500)
    r.calibrate(sp.signal("u"), sp.error())
    ch = r.choose_threshold(sp.signal("u"), sp.small_score, sp.large_score, 0.8)
    assert r.theta == ch.theta


def test_default_delta_grid_is_read_only():
    assert DEFAULT_DELTA_GRID[0] == 0.005
    assert DEFAULT_DELTA_GRID[-1] == 0.995
    assert DEFAULT_DELTA_GRID.size == 199
    with pytest.raises(ValueError, match="read-only"):
        DEFAULT_DELTA_GRID[0] = 0.5


def test_identity_calibrator():
    cal = IdentityCalibrator().fit([0.1, 0.9], [0, 1])
    assert cal.predict([-0.5, 0.25, 1.5]).tolist() == [0.0, 0.25, 1.0]
    assert cal.predict(0.3) == 0.3
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        IdentityCalibrator().fit([0.1], [2.0])


# ---------------------------------------------------------------------------
# CalibratedThresholdRouter (the identical Eq. 6 protocol for any calibrator)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_isotonic_calibrated_router_is_ucci(seed):
    rng = np.random.default_rng(300 + seed)
    cal, val, test = (synthetic_split(rng, n) for n in (900, 600, 1500))
    ours = CalibratedThresholdRouter(IsotonicCalibrator()).calibrate(cal.signal("u"), cal.error())
    ref = UCCIRouter().calibrate(cal.signal("u"), cal.error())
    a = ours.choose_threshold(val.signal("u"), val.small_score, val.large_score, 0.85)
    b = ref.choose_threshold(val.signal("u"), val.small_score, val.large_score, 0.85)
    assert a == b
    assert np.array_equal(ours.escalate(test.signal("u")), ref.escalate(test.signal("u")))
    a = ours.choose_threshold(val.signal("u"), val.small_score, val.large_score, budget=1.9)
    b = ref.choose_threshold_for_budget(val.signal("u"), val.small_score, val.large_score, 1.9)
    assert a == b


@pytest.mark.parametrize(
    "make",
    [
        TemperatureScalingCalibrator,
        IdentityCalibrator,
        lambda: PlattCalibrator(target_smoothing=True),
    ],
)
def test_calibrated_router_runs_with_every_calibrator(make):
    rng = np.random.default_rng(31)
    cal, val = synthetic_split(rng, 800), synthetic_split(rng, 600)
    r = CalibratedThresholdRouter(make()).calibrate(cal.signal("u"), cal.error())
    ch = r.choose_threshold(val.signal("u"), val.small_score, val.large_score, 0.8)
    assert ch.theta in DEFAULT_GRID
    assert ch.accuracy >= 0.8
    p = r.error_probability(val.signal("u"))
    assert np.array_equal(r.escalate(val.signal("u")), p > ch.theta)


def test_identity_router_equals_grid_raw_threshold_on_u():
    rng = np.random.default_rng(32)
    val = synthetic_split(rng, 700)
    u = val.signal("u")
    a = CalibratedThresholdRouter(IdentityCalibrator()).calibrate(u, val.error())
    a_ch = a.choose_threshold(u, val.small_score, val.large_score, 0.8)
    b_ch = RawThresholdRouter(candidates=DEFAULT_GRID).choose_threshold(
        u, val.small_score, val.large_score, 0.8
    )
    assert a_ch == b_ch


def test_calibrated_router_rejects_invalid_calibrator_output():
    class Bad:
        def fit(self, u, e, sample_weight=None):
            return self

        def predict(self, u):
            return np.asarray(u) + 1.0

    r = CalibratedThresholdRouter(Bad()).calibrate([0.1], [0.0])
    with pytest.raises(ValueError, match="outside"):
        r.choose_threshold([0.1, 0.5], [1.0, 0.0], [1.0, 1.0], 0.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        CalibratedThresholdRouter(IdentityCalibrator(), grid=[0.5, 1.5])


def test_calibrated_router_passes_sample_weights():
    rng = np.random.default_rng(33)
    u = rng.random(100)
    e = (rng.random(100) < u).astype(float)
    w = rng.integers(1, 3, 100)
    a = CalibratedThresholdRouter(IsotonicCalibrator()).calibrate(u, e, w)
    b = IsotonicCalibrator().fit(np.repeat(u, w), np.repeat(e, w))
    assert np.allclose(a.error_probability(u), b.predict(u), atol=1e-12)


# ---------------------------------------------------------------------------
# compare_routers (the Section 6.1 protocol)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def splits():
    rng = np.random.default_rng(2026)
    return (
        synthetic_split(rng, 3000),
        synthetic_split(rng, 2000),
        synthetic_split(rng, 5000),
    )


def test_compare_routers_rows_and_determinism(splits):
    cal, val, test = splits
    rows = compare_routers(cal, val, test, tau=0.85)
    assert [r.method for r in rows] == [
        "UCCI",
        "Conformal prediction",
        "FrugalGPT-style",
        "Entropy threshold",
        "Large-only",
        "Small-only",
        "Oracle",
    ]
    assert rows == compare_routers(cal, val, test, tau=0.85)
    by = {r.method: r for r in rows}
    assert by["FrugalGPT-style"].signal == "max_prob"
    assert by["Large-only"].cost == C_L
    assert by["Small-only"].cost == C_S
    assert "delta" in by["Conformal prediction"].params
    for r in rows:
        assert isinstance(r, ComparisonRow)
        assert r.delta_vs_target == pytest.approx(r.accuracy - 0.85)
        if r.method not in ("Large-only", "Small-only", "Oracle"):
            assert r.feasible_on_val
            assert r.val_accuracy >= 0.85


def test_compare_routers_rows_equal_the_manual_pipeline(splits):
    cal, val, test = splits
    rows = {r.method: r for r in compare_routers(cal, val, test, tau=0.85)}
    ucci = UCCIRouter().calibrate(cal.signal("u"), cal.error())
    ch = ucci.choose_threshold(val.signal("u"), val.small_score, val.large_score, 0.85)
    res = ucci.evaluate(test.signal("u"), test.small_score, test.large_score)
    row = rows["UCCI"]
    assert (row.threshold, row.val_cost, row.val_accuracy) == (
        ch.theta,
        ch.cost,
        ch.accuracy,
    )
    assert (row.cost, row.accuracy, row.escalation_rate) == (
        res.cost,
        res.accuracy,
        res.escalation_rate,
    )
    ent = EntropyThresholdRouter()
    ch = ent.choose_threshold(val.signal("entropy"), val.small_score, val.large_score, 0.85)
    mask = ent.escalate(test.signal("entropy"))
    row = rows["Entropy threshold"]
    assert row.threshold == ch.theta
    assert row.cost == policy_cost(mask, C_S, C_L)
    assert row.accuracy == policy_accuracy(mask, test.small_score, test.large_score)


def test_compare_routers_oracle_bounds_every_method(splits):
    cal, val, test = splits
    rows = compare_routers(cal, val, test, tau=0.85, include_ablations=True)
    oracle = rows[-1]
    assert oracle.method == "Oracle"
    assert oracle.accuracy >= 0.85
    for r in rows[:-1]:
        if r.accuracy >= 0.85:
            assert r.cost >= oracle.cost


def test_compare_routers_budget_form(splits):
    cal, val, test = splits
    rows = compare_routers(cal, val, test, budget=2.0, tau=0.85, include_ablations=True)
    names = [r.method for r in rows]
    assert "Temperature scaling" in names
    assert "Isotonic on max prob" in names
    for r in rows:
        if r.method in ("Large-only", "Oracle"):
            continue
        assert r.val_cost <= 2.0
        assert r.feasible_on_val
        assert r.delta_vs_target == pytest.approx(r.accuracy - 0.85)
    assert {r.method: r for r in rows}["Large-only"].feasible_on_val is False
    no_tau = compare_routers(cal, val, test, budget=2.0)
    assert all(r.delta_vs_target is None for r in no_tau)
    assert "| n/a |" in format_comparison(no_tau) or no_tau[-1].method == "Oracle"
    assert format_comparison(no_tau).splitlines()[2].endswith("|  |")


def test_compare_routers_ablation_rows_match_their_routers(splits):
    cal, val, test = splits
    rows = {r.method: r for r in compare_routers(cal, val, test, tau=0.85, include_ablations=True)}
    r = CalibratedThresholdRouter(TemperatureScalingCalibrator()).calibrate(
        cal.signal("u"), cal.error()
    )
    r.choose_threshold(val.signal("u"), val.small_score, val.large_score, 0.85)
    assert rows["Temperature scaling"].params == {"temperature": r.calibrator.temperature_}
    assert rows["Temperature scaling"].threshold == r.theta
    m = CalibratedThresholdRouter(IsotonicCalibrator()).calibrate(
        1 - cal.signal("max_prob"), cal.error()
    )
    m.choose_threshold(1 - val.signal("max_prob"), val.small_score, val.large_score, 0.85)
    mask = m.escalate(1 - test.signal("max_prob"))
    assert rows["Isotonic on max prob"].accuracy == policy_accuracy(
        mask, test.small_score, test.large_score
    )


def test_compare_routers_infeasible_rows_are_nan(splits):
    cal, val, test = splits
    rows = compare_routers(cal, val, test, tau=0.999)
    for r in rows:
        if r.method in ("Large-only", "Small-only"):
            assert not math.isnan(r.cost)
            assert not r.feasible_on_val
        else:
            assert math.isnan(r.cost)
            assert not r.feasible_on_val
            assert "infeasible" in r.note
            assert math.isnan(r.delta_vs_target)
    budget_rows = compare_routers(cal, val, test, budget=0.5)
    for r in budget_rows:
        if r.method not in ("Large-only", "Small-only"):
            assert r.delta_vs_target is None
            assert "infeasible" in r.note


def test_split_data_is_not_compared_elementwise(splits):
    cal, _, _ = splits
    assert cal == cal
    assert cal != SplitData(cal.signals, cal.small_score, cal.large_score)


def test_compare_routers_signal_handling(splits):
    cal, val, test = splits
    only_u = [
        SplitData({"u": sp.signal("u")}, sp.small_score, sp.large_score) for sp in (cal, val, test)
    ]
    names = [r.method for r in compare_routers(*only_u, tau=0.85, include_oracle=False)]
    assert names == ["UCCI", "Conformal prediction", "Large-only", "Small-only"]
    with pytest.raises(ValueError, match="needs signal 'confidence'"):
        compare_routers(*only_u, tau=0.85, methods=table2_methods())
    with pytest.raises(ValueError, match="needs signal 'entropy'"):
        compare_routers(*only_u, tau=0.85, methods=table2_methods("u")[3:])
    conf = [
        SplitData(
            {**sp.signals, "confidence": 1 - sp.signal("u")},
            sp.small_score,
            sp.large_score,
        )
        for sp in (cal, val, test)
    ]
    rows = {r.method: r for r in compare_routers(*conf, tau=0.85)}
    assert rows["FrugalGPT-style"].signal == "confidence"
    with pytest.raises(KeyError, match="available"):
        cal.signal("nope")


def test_compare_routers_custom_methods_transform_and_metric(splits):
    cal, val, test = splits
    spec = MethodSpec(
        "Negated max prob",
        "max_prob",
        lambda cs, cl, cm: EntropyThresholdRouter(cs, cl, cm),
        transform=lambda x: -x,
    )
    rows = compare_routers(cal, val, test, tau=0.85, methods=[spec], include_oracle=False)
    ref = FrugalGPTStyleRouter().choose_threshold(
        val.signal("max_prob"), val.small_score, val.large_score, 0.85
    )
    assert rows[0].threshold == -ref.theta
    metric_splits = [
        SplitData(
            sp.signals,
            sp.small_score,
            sp.large_score,
            metric=(lambda sm, lg: lambda m: _acc(m, sm, lg))(
                np.asarray(sp.small_score), np.asarray(sp.large_score)
            ),
        )
        for sp in (cal, val, test)
    ]
    assert compare_routers(*metric_splits, tau=0.85) == compare_routers(cal, val, test, tau=0.85)


def test_compare_routers_uses_small_error_on_calibration(splits):
    cal, val, test = splits
    flipped = SplitData(cal.signals, cal.small_score, cal.large_score, small_error=1 - cal.error())
    a = compare_routers(cal, val, test, tau=0.85, include_oracle=False)
    b = compare_routers(flipped, val, test, tau=0.85, include_oracle=False)
    assert a[0].threshold != b[0].threshold or a[0].cost != b[0].cost


def test_compare_routers_argument_errors(splits):
    cal, val, test = splits
    with pytest.raises(ValueError, match="pass tau"):
        compare_routers(cal, val, test)
    with pytest.raises(ValueError, match="Theorem 1"):
        compare_routers(cal, val, test, tau=0.85, c_small=3.0, c_large=1.0)
    bad = SplitData(test.signals, test.small_score, np.ones(3))
    with pytest.raises(ValueError, match="length"):
        compare_routers(cal, val, bad, tau=0.85)


def test_default_graded_error_label():
    sp = SplitData({"u": [0.1, 0.2, 0.3]}, [1.0, 0.75, 0.0], [1.0, 1.0, 1.0])
    assert sp.error().tolist() == [0.0, 1.0, 1.0]


def test_format_comparison(splits):
    cal, val, test = splits
    text = format_comparison(compare_routers(cal, val, test, tau=0.85))
    lines = text.splitlines()
    assert lines[0].startswith("| Method |")
    assert len(lines) == 2 + 7
    assert "+inf" in text
    assert "-inf" in text
    assert "n/a" in text
    assert chr(0x2013) not in text
    assert chr(0x2014) not in text


def test_method_lists_are_fresh_and_labelled():
    t2 = table2_methods()
    assert [m.name for m in t2][:4] == [
        "UCCI",
        "Conformal prediction",
        "FrugalGPT-style",
        "Entropy threshold",
    ]
    abl = ablation_methods()
    assert any("extension" in m.name for m in abl)
    r1, r2 = t2[0].factory(C_S, C_L, "routing"), t2[0].factory(C_S, C_L, "routing")
    assert r1 is not r2


def test_every_router_follows_the_shared_interface():
    rng = np.random.default_rng(41)
    cal, val = synthetic_split(rng, 600), synthetic_split(rng, 400)
    routers = [spec.factory(C_S, C_L, "routing") for spec in table2_methods("max_prob")]
    routers += [spec.factory(C_S, C_L, "routing") for spec in ablation_methods()]
    for r in routers:
        r.calibrate(cal.signal("u"), cal.error())
        ch = r.choose_threshold(val.signal("u"), val.small_score, val.large_score, 0.8)
        assert isinstance(ch, ThresholdChoice)
        mask = r.escalate(val.signal("u"))
        assert mask.dtype == bool
        assert mask.shape == (400,)
        assert mask.mean() == ch.escalation_rate


def test_no_warnings_on_the_happy_path(splits):
    cal, val, test = splits
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compare_routers(cal, val, test, tau=0.85, include_ablations=True)


try:  # property tests run where hypothesis is installed (the dev extra)
    from hypothesis import given, settings
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover
    given = None


if given is not None:

    @settings(max_examples=150, deadline=None)
    @given(
        data=st.lists(
            st.tuples(
                st.integers(0, 6),
                st.sampled_from([0.0, 0.5, 1.0]),
                st.sampled_from([0.0, 0.5, 1.0]),
            ),
            min_size=1,
            max_size=25,
        ),
        q=st.floats(0.0, 1.0),
        k=st.integers(0, 25),
    )
    def test_property_raw_threshold_equals_brute_force(data, q, k):
        s = np.array([d[0] for d in data], dtype=float)
        small = np.array([d[1] for d in data])
        large = np.array([d[2] for d in data])
        accs = sorted({_acc(s > t, small, large) for t in exact_threshold_candidates(s)})
        tau = accs[min(int(q * len(accs)), len(accs) - 1)]
        ref = brute_threshold(s, small, large, tau=tau)
        ch = EntropyThresholdRouter().choose_threshold(s, small, large, tau)
        assert (ch.theta, ch.accuracy, ch.cost) == (ref[0], ref[2], ref[3])
        budget = _cost(min(k, s.size), s.size) + 1e-9
        ref = brute_threshold(s, small, large, budget=budget)
        ch = EntropyThresholdRouter().choose_threshold(s, small, large, budget=budget)
        assert (ch.theta, ch.accuracy, ch.cost) == (ref[0], ref[2], ref[3])
