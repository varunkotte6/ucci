"""Tests for ucci.policy: the threshold policy and its selection (Section 4.3)."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from ucci import (
    DEFAULT_COST_LARGE,
    DEFAULT_COST_SMALL,
    DEFAULT_GRID,
    DEFAULT_GRID_STEP,
    InfeasibleTargetError,
    ThresholdChoice,
    escalate,
    evaluate,
    make_grid,
    pareto_frontier,
    policy_accuracy,
    policy_cost,
    select_threshold,
    select_threshold_for_budget,
)


def _naive_sweep(p_hat, small, large, grid, c_s, c_l, cost_model="routing"):
    """Reference loop: route with every threshold and measure it directly."""
    out = []
    for theta in np.unique(grid):
        m = p_hat > theta
        acc = float(np.mean(np.where(m, large, small)))
        per_query = np.where(m, c_l if cost_model == "routing" else c_s + c_l, c_s)
        out.append((float(theta), float(per_query.mean()), acc, float(m.mean())))
    return out


def _naive_select(p_hat, small, large, tau, grid, c_s, c_l):
    feasible = [r for r in _naive_sweep(p_hat, small, large, grid, c_s, c_l) if r[2] >= tau]
    return min(feasible, key=lambda r: (round(r[1], 12), -r[2], -r[0]))


# ---------------------------------------------------------------- grid


def test_default_grid_is_0005_resolution_on_unit_interval():
    assert DEFAULT_GRID.shape == (201,)
    assert DEFAULT_GRID[0] == 0.0 and DEFAULT_GRID[-1] == 1.0
    np.testing.assert_allclose(np.diff(DEFAULT_GRID), 0.005, atol=1e-12)
    assert DEFAULT_GRID_STEP == 0.005
    assert not DEFAULT_GRID.flags.writeable
    with pytest.raises(ValueError):
        DEFAULT_GRID[0] = 0.5


def test_make_grid():
    np.testing.assert_array_equal(make_grid(0.005), DEFAULT_GRID)
    np.testing.assert_array_equal(make_grid(), DEFAULT_GRID)
    assert make_grid(0.1).tolist() == [round(0.1 * i, 1) for i in range(11)]
    assert make_grid(1.0).tolist() == [0.0, 1.0]
    assert make_grid(0.25).tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]


@pytest.mark.parametrize("step", [0.0, -0.1, 1.5, 0.3, float("nan"), "0.1"])
def test_make_grid_rejects_bad_steps(step):
    with pytest.raises(ValueError, match="step"):
        make_grid(step)


# --------------------------------------------------------------- policy


def test_escalate_is_strictly_greater_than_theta():
    np.testing.assert_array_equal(escalate([0.2, 0.3, 0.4], 0.3), [False, False, True])
    assert escalate(0.3, 0.3) is False
    assert escalate(0.31, 0.3) is True
    out = escalate(np.array(0.9), 0.5)
    assert isinstance(out, np.ndarray) and out.shape == ()
    assert escalate(np.ones((2, 3)), 0.5).shape == (2, 3)


def test_escalate_validates():
    with pytest.raises(ValueError, match="theta"):
        escalate([0.1], float("nan"))
    with pytest.raises(ValueError, match="p_hat"):
        escalate([0.1, float("nan")], 0.5)
    with pytest.raises(ValueError, match="theta"):
        escalate([0.1], True)


def test_policy_cost_models():
    esc = np.array([True, False, False, True])
    assert policy_cost(esc, 1.0, 3.0, "routing") == pytest.approx(2.0)
    assert policy_cost(esc, 1.0, 3.0, "sequential") == pytest.approx(2.5)
    assert policy_cost([1, 0, 0, 1], 1.0, 3.0) == pytest.approx(2.0)
    assert policy_cost(esc) == pytest.approx(0.5 * 1.0 + 0.5 * 3.02)


def test_policy_cost_endpoints_are_exact():
    assert policy_cost(np.zeros(7, bool), 1.0, 3.02) == 1.0
    assert policy_cost(np.ones(7, bool), 1.0, 3.02) == 3.02
    assert policy_cost(np.ones(7, bool), 1.0, 3.02, "sequential") == 1.0 + 3.02


@pytest.mark.parametrize(
    ("esc", "kw", "match"),
    [
        ([], {}, "empty"),
        ([0.5, 1.0], {}, "boolean"),
        ([[True, False]], {}, "1-D"),
        ([True], {"c_small": 0.0}, "c_small must be positive"),
        ([True], {"c_large": -1.0}, "c_large must be positive"),
        ([True], {"c_large": float("inf")}, "c_large must be finite"),
        ([True], {"cost_model": "parallel"}, "cost_model"),
    ],
)
def test_policy_cost_validation(esc, kw, match):
    with pytest.raises(ValueError, match=match):
        policy_cost(esc, **kw)


def test_policy_accuracy():
    esc = [True, False, True]
    assert policy_accuracy(esc, [0, 1, 0], [1, 0, 0.5]) == pytest.approx((1 + 1 + 0.5) / 3)
    with pytest.raises(ValueError, match="length mismatch"):
        policy_accuracy(esc, [0, 1], [1, 1, 1])


# --------------------------------------------------- Table 3 arithmetic


def test_table3_cost_ratio_arithmetic():
    """Table 3: one routing mask, re-costed at c_large/c_small = 3.02, 5.00, 10.00."""
    n, k = 2000, 1069  # escalation rate 0.5345, consistent with all three rows
    mask = np.zeros(n, bool)
    mask[:k] = True
    rows = [(3.02, 2.08, 0.31), (5.00, 3.14, 0.37), (10.00, 5.81, 0.42)]
    for c_large, cost, saving in rows:
        c = policy_cost(mask, 1.0, c_large)
        assert round(c, 2) == cost
        assert round(1.0 - c / c_large, 2) == saving
    # The spot checks in the build spec: 1 + 2.02 * 0.535 and 1 + 4.0 * 0.535.
    assert round(1 + 2.02 * 0.535, 2) == 2.08
    assert round(1 + 4.0 * 0.535, 2) == 3.14


def test_threshold_does_not_depend_on_cost_ratio():
    """Section 6.2: with constant marginal cost the feasible threshold is unchanged."""
    rng = np.random.default_rng(0)
    p = rng.random(3000)
    small = (rng.random(3000) > p).astype(float)
    large = (rng.random(3000) < 0.95).astype(float)
    tau = 0.85
    thetas = {
        select_threshold(p, small, large, tau, 1.0, c_l, cost_model=cm).theta
        for c_l in (1.5, 3.02, 5.0, 10.0)
        for cm in ("routing", "sequential")
    }
    assert len(thetas) == 1


# -------------------------------------------------------- select_threshold


def test_select_threshold_picks_cheapest_feasible():
    p_hat = np.array([0.1, 0.2, 0.6, 0.9])
    small = np.array([1, 1, 0, 0])
    large = np.array([1, 1, 1, 1])
    c = select_threshold(p_hat, small, large, tau=1.0, c_small=1.0, c_large=3.0)
    assert c.accuracy == 1.0 and c.cost == pytest.approx(2.0) and c.escalation_rate == 0.5
    # Largest feasible grid value below 0.6 (App. A.1's "smallest" is the argmin's
    # largest feasible theta: fewest escalations).
    assert c.theta == 0.595


def test_ties_go_to_the_larger_theta():
    c = select_threshold([0.1, 0.9], [1, 0], [1, 1], tau=1.0)
    assert c.theta == 0.895
    c = select_threshold([0.1, 0.9], [1, 1], [1, 1], tau=1.0)
    assert c.theta == 1.0 and c.escalation_rate == 0.0 and c.cost == DEFAULT_COST_SMALL


def test_p_hat_equal_to_theta_is_kept():
    c = select_threshold([0.5, 0.5], [1, 0], [1, 1], tau=1.0, grid=[0.25, 0.5, 0.75])
    assert c.theta == 0.25 and c.escalation_rate == 1.0


def test_infeasible_tau_error_message():
    with pytest.raises(InfeasibleTargetError, match=r"best accuracy on the grid is 0\.5"):
        select_threshold([0.5, 0.5], [1, 0], [0, 0], tau=0.9, c_small=1.0, c_large=2.0)
    assert issubclass(InfeasibleTargetError, ValueError)


def test_routing_needs_c_large_above_c_small():
    with pytest.raises(ValueError, match="Theorem 1 assumption"):
        select_threshold([0.5], [1], [1], tau=0.5, c_small=2.0, c_large=2.0)
    with pytest.raises(ValueError, match="Theorem 1 assumption"):
        select_threshold_for_budget([0.5], [1], [1], budget=5.0, c_small=2.0, c_large=1.0)
    # The sequential model always pays more for an escalation.
    c = select_threshold(
        [0.5], [1], [1], tau=0.5, c_small=2.0, c_large=1.0, cost_model="sequential"
    )
    assert c.theta == 1.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tau": float("nan")}, "tau"),
        ({"tau": "0.9"}, "tau"),
        ({"grid": [0.1, 1.5]}, "grid"),
        ({"grid": [-0.1, 0.5]}, "grid"),
        ({"grid": []}, "grid"),
        ({"grid": [0.1, float("nan")]}, "grid"),
        ({"cost_model": "other"}, "cost_model"),
        ({"c_small": float("nan")}, "c_small"),
    ],
)
def test_select_threshold_validation(kwargs, match):
    args = {"tau": 0.5, **kwargs}
    with pytest.raises(ValueError, match=match):
        select_threshold([0.1, 0.9], [1, 0], [1, 1], **args)


def test_select_threshold_input_validation():
    with pytest.raises(ValueError, match="length mismatch"):
        select_threshold([0.1, 0.9], [1, 0], [1], tau=0.5)
    with pytest.raises(ValueError, match="p_hat"):
        select_threshold([0.1, float("nan")], [1, 0], [1, 1], tau=0.5)
    with pytest.raises(ValueError, match="required without a metric"):
        select_threshold([0.1, 0.9], None, None, tau=0.5)
    with pytest.raises(ValueError, match="empty"):
        select_threshold([], [], [], tau=0.5)


def test_grid_is_sorted_and_deduplicated():
    p = np.array([0.1, 0.4, 0.7])
    a = select_threshold(p, [1, 0, 0], [1, 1, 1], tau=1.0, grid=[0.9, 0.05, 0.3, 0.3, 0.2])
    b = select_threshold(p, [1, 0, 0], [1, 1, 1], tau=1.0, grid=[0.05, 0.2, 0.3, 0.9])
    assert a == b and a.theta == 0.3


@pytest.mark.parametrize("seed", range(25))
def test_vectorized_sweep_matches_loop(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 500))
    p = np.round(rng.random(n), int(rng.integers(1, 4)))  # atoms and ties with the grid
    fractional = seed % 2 == 0
    if fractional:
        small = rng.random(n)
        large = rng.random(n)
    else:
        small = (rng.random(n) > p).astype(float)
        large = (rng.random(n) < 0.9).astype(float)
    grid = DEFAULT_GRID if seed % 3 else np.unique(np.round(rng.random(40), 2))
    c_l = float(rng.uniform(1.1, 10.0))
    ref = _naive_sweep(p, small, large, grid, 1.0, c_l)
    front = pareto_frontier(p, small, large, 1.0, c_l, grid)
    np.testing.assert_allclose(front.theta, [r[0] for r in ref])
    np.testing.assert_allclose(front.cost, [r[1] for r in ref], rtol=0, atol=1e-12)
    np.testing.assert_allclose(front.accuracy, [r[2] for r in ref], rtol=0, atol=1e-12)
    np.testing.assert_allclose(front.escalation_rate, [r[3] for r in ref], atol=1e-15)
    accs = sorted({r[2] for r in ref})
    for tau in accs[:: max(1, len(accs) // 5)]:
        if fractional:
            tau = tau - 1e-9  # keep tau away from floating-point ties
        got = select_threshold(p, small, large, tau, 1.0, c_l, grid)
        want = _naive_select(p, small, large, tau, grid, 1.0, c_l)
        assert got.theta == want[0]
        assert got.cost == pytest.approx(want[1], abs=1e-12)
        assert got.accuracy == pytest.approx(want[2], abs=1e-12)


def test_default_scores_exact_for_binary_labels():
    rng = np.random.default_rng(99)
    n = 15000
    p = rng.random(n)
    small = (rng.random(n) > p).astype(float)
    large = (rng.random(n) < 0.93).astype(float)
    front = pareto_frontier(p, small, large)
    for theta, acc in zip(front.theta[::10], front.accuracy[::10]):
        m = p > theta
        assert acc == (small[~m].sum() + large[m].sum()) / n  # bit-identical


# ------------------------------------------------------- custom metric


def test_custom_corpus_metric():
    p_hat = np.array([0.1, 0.9])
    calls = []

    def metric(mask):
        calls.append(mask.copy())
        return 1.0 if mask[1] else 0.5

    c = select_threshold(p_hat, None, None, tau=1.0, c_small=1.0, c_large=2.0, metric=metric)
    assert c.escalation_rate == 0.5 and c.theta == 0.895
    # Called once per distinct escalation mask (0, 1 or 2 escalations).
    assert len(calls) == 3
    assert all(m.dtype == bool and m.shape == (2,) for m in calls)


def test_custom_metric_errors():
    with pytest.raises(ValueError, match="metric returned nan"):
        select_threshold([0.5], None, None, tau=0.5, metric=lambda m: float("nan"))
    with pytest.raises(ValueError, match="must return a number"):
        select_threshold([0.5], None, None, tau=0.5, metric=lambda m: "high")
    with pytest.raises(ValueError, match="callable"):
        select_threshold([0.5], None, None, tau=0.5, metric=0.9)
    with pytest.raises(ValueError, match="callable"):
        evaluate([0.5], None, None, 0.5, metric=0.9)


# ------------------------------------------------ Theorem 1 brute force


def _expected_accuracy(p, alpha_l):
    """E[Acc] of a routing mask under Theorem 1's assumptions (ii) and (iii)."""
    return lambda esc: float(np.where(esc, alpha_l, 1.0 - p).mean())


def _all_masks(n):
    return ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(bool)


def _midpoint_grid(p):
    s = np.sort(p)
    return np.concatenate(([0.0], (s[1:] + s[:-1]) / 2, [1.0]))


@pytest.mark.parametrize("seed", range(30))
def test_theorem1_threshold_is_cost_optimal_among_all_policies(seed):
    """Among all 2^n routing subsets, the cheapest one meeting tau costs no less
    than the threshold policy that select_threshold returns (Theorem 1)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 11))
    p = rng.uniform(0.01, 0.99, n)  # calibrated p_hat, distinct values
    alpha_l = float(rng.uniform(0.8, 1.0))
    c_s, c_l = 1.0, float(rng.uniform(1.5, 6.0))
    cost_model = "routing" if seed % 2 else "sequential"
    exp_acc = _expected_accuracy(p, alpha_l)
    masks = _all_masks(n)
    accs = np.array([exp_acc(m) for m in masks])
    costs = np.array([policy_cost(m, c_s, c_l, cost_model) for m in masks])
    tau = float(rng.uniform(accs.min(), accs.max()))
    best_any = costs[accs >= tau].min()
    choice = select_threshold(p, None, None, tau, c_s, c_l, _midpoint_grid(p), exp_acc, cost_model)
    assert choice.accuracy >= tau
    assert choice.cost == pytest.approx(best_any, abs=1e-12)


def test_theorem1_tie_breaking_on_level_sets():
    """On a level set of p_hat, which members are escalated does not change the
    expected cost or accuracy (the "up to tie-breaking" clause)."""
    p = np.array([0.2, 0.5, 0.5, 0.5, 0.8])
    exp_acc = _expected_accuracy(p, 0.95)
    level = [1, 2, 3]
    results = set()
    for chosen in level:
        esc = np.zeros(5, bool)
        esc[[4, chosen]] = True
        results.add((round(exp_acc(esc), 12), policy_cost(esc, 1.0, 3.02)))
    assert len(results) == 1


@pytest.mark.parametrize("seed", range(30))
def test_budget_form_threshold_is_optimal_among_all_policies(seed):
    """Max expected accuracy subject to cost <= budget: a threshold attains the
    optimum over all 2^n routing subsets (Table 2, bottom block)."""
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(2, 11))
    p = rng.uniform(0.01, 0.99, n)
    alpha_l = float(rng.uniform(0.5, 1.0))  # may be worse than keeping some queries
    c_s, c_l = 1.0, float(rng.uniform(1.5, 6.0))
    exp_acc = _expected_accuracy(p, alpha_l)
    masks = _all_masks(n)
    accs = np.array([exp_acc(m) for m in masks])
    costs = np.array([policy_cost(m, c_s, c_l) for m in masks])
    budget = float(rng.uniform(c_s, c_l))
    best_any = accs[costs <= budget].max()
    choice = select_threshold_for_budget(
        p, None, None, budget, c_s, c_l, _midpoint_grid(p), exp_acc
    )
    assert choice.cost <= budget
    assert choice.accuracy == pytest.approx(best_any, abs=1e-12)


# ------------------------------------------------------- budget form


def test_budget_form_basic():
    p = np.array([0.1, 0.3, 0.6, 0.9])
    small = np.array([1, 1, 0, 0])
    large = np.ones(4)
    # Budget for one escalation at c_large = 3: cost 1.5.
    c = select_threshold_for_budget(p, small, large, budget=1.5, c_small=1.0, c_large=3.0)
    assert c.escalation_rate == 0.25 and c.accuracy == 0.75 and c.cost == 1.5
    assert c.theta == 0.895
    # A large budget buys the most accurate policy at the lowest cost.
    c = select_threshold_for_budget(p, small, large, budget=10.0, c_small=1.0, c_large=3.0)
    assert c.accuracy == 1.0 and c.escalation_rate == 0.5 and c.theta == 0.595


def test_budget_matches_cost_of_same_mask_exactly():
    rng = np.random.default_rng(3)
    p = rng.random(15000)
    small = (rng.random(15000) > p).astype(float)
    large = np.ones(15000)
    ref = evaluate(p, small, large, 0.4)
    c = select_threshold_for_budget(p, small, large, budget=ref.cost)
    assert c.cost <= ref.cost and c.accuracy >= ref.accuracy


def test_budget_infeasible():
    with pytest.raises(InfeasibleTargetError, match="cheapest costs 1"):
        select_threshold_for_budget([0.5], [1], [1], budget=0.5)
    with pytest.raises(ValueError, match="budget"):
        select_threshold_for_budget([0.5], [1], [1], budget=float("inf"))


# ---------------------------------------------------------- frontier


def test_pareto_frontier_efficient_mask_matches_brute_force():
    rng = np.random.default_rng(12)
    for _ in range(20):
        n = int(rng.integers(5, 300))
        p = np.round(rng.random(n), 2)
        small = rng.random(n)
        large = rng.random(n)
        f = pareto_frontier(p, small, large, grid=np.round(rng.random(60), 2))
        g = f.theta.size
        dominated = np.zeros(g, bool)
        for i in range(g):
            for j in range(g):
                better_or_equal = f.cost[j] <= f.cost[i] and f.accuracy[j] >= f.accuracy[i]
                strictly = f.cost[j] < f.cost[i] or f.accuracy[j] > f.accuracy[i]
                dominated[i] |= better_or_equal and strictly
        np.testing.assert_array_equal(f.efficient, ~dominated)
        assert np.all(np.diff(f.cost) <= 0)  # cost falls as theta rises
        assert np.all(np.diff(f.theta) > 0)


def test_pareto_frontier_to_dict_is_json_serializable():
    f = pareto_frontier([0.1, 0.9], [1, 0], [1, 1], grid=[0.0, 0.5, 1.0])
    d = json.loads(json.dumps(f.to_dict()))
    assert d["theta"] == [0.0, 0.5, 1.0]
    assert d["escalation_rate"] == [1.0, 0.5, 0.0]
    assert d["efficient"] == [False, True, True]
    assert f.cost[0] == DEFAULT_COST_LARGE


def test_pareto_frontier_accepts_any_positive_costs():
    f = pareto_frontier([0.1, 0.9], [1, 0], [1, 1], c_small=2.0, c_large=1.0)
    assert f.cost[0] == 1.0 and f.cost[-1] == 2.0


# ----------------------------------------------------------- evaluate


def test_evaluate_matches_manual_routing():
    rng = np.random.default_rng(4)
    p = rng.random(1000)
    small = rng.random(1000)
    large = rng.random(1000)
    r = evaluate(p, small, large, 0.37, 1.0, 4.0, cost_model="sequential")
    m = p > 0.37
    assert r.theta == 0.37
    assert r.escalation_rate == m.mean()
    assert r.accuracy == pytest.approx(np.where(m, large, small).mean(), abs=1e-12)
    assert r.cost == pytest.approx(1.0 + 4.0 * m.mean(), abs=1e-12)
    assert isinstance(r, ThresholdChoice)
    assert r.to_dict() == {
        "theta": r.theta,
        "cost": r.cost,
        "accuracy": r.accuracy,
        "escalation_rate": r.escalation_rate,
    }


def test_evaluate_with_metric_and_validation():
    r = evaluate([0.1, 0.9], None, None, 0.5, metric=lambda m: float(m.sum()))
    assert r.accuracy == 1.0 and r.escalation_rate == 0.5
    with pytest.raises(ValueError, match="required without a metric"):
        evaluate([0.1, 0.9], [1, 1], None, 0.5)
    with pytest.raises(ValueError, match="theta"):
        evaluate([0.1, 0.9], [1, 1], [1, 1], float("inf"))


def test_threshold_choice_is_frozen():
    c = ThresholdChoice(0.5, 2.0, 0.9, 0.5)
    with pytest.raises(AttributeError):
        c.theta = 0.1  # type: ignore[misc]


# -------------------------------------------------------- performance


def test_performance_smoke_n_200000():
    rng = np.random.default_rng(5)
    n = 200_000
    p = rng.random(n)
    small = (rng.random(n) > p).astype(float)
    large = (rng.random(n) < 0.95).astype(float)
    start = time.perf_counter()
    select_threshold(p, small, large, tau=0.9)
    select_threshold_for_budget(p, small, large, budget=2.0)
    pareto_frontier(p, small, large)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"threshold selection took {elapsed:.2f}s for n={n}"


def test_performance_million_queries():
    rng = np.random.default_rng(6)
    n = 1_000_000
    p = rng.random(n)
    small = (rng.random(n) > p).astype(float)
    large = np.ones(n)
    start = time.perf_counter()
    select_threshold(p, small, large, tau=0.9)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"select_threshold took {elapsed:.2f}s for n=1e6"
