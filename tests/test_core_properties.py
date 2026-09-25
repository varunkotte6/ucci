"""Property-based tests for the UCCI core (skipped when hypothesis is missing)."""

from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
st = pytest.importorskip("hypothesis.strategies")
hnp = pytest.importorskip("hypothesis.extra.numpy")

from hypothesis import given, settings  # noqa: E402

from ucci import (  # noqa: E402
    IsotonicCalibrator,
    UCCIRouter,
    batch_uncertainty,
    ece,
    pareto_frontier,
    pav,
    policy_cost,
    select_threshold,
    uncertainty_from_probs,
)
from ucci.policy import InfeasibleTargetError  # noqa: E402

unit = st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)
weights = st.floats(1e-3, 1e3, allow_nan=False, allow_infinity=False)
coarse_unit = st.integers(0, 20).map(lambda k: k / 20.0)  # forces ties


@st.composite
def calibration_data(draw, max_n=60):
    n = draw(st.integers(1, max_n))
    u = np.array(draw(st.lists(st.one_of(unit, coarse_unit), min_size=n, max_size=n)))
    e = np.array(
        draw(st.lists(st.one_of(unit, st.sampled_from([0.0, 1.0])), min_size=n, max_size=n))
    )
    w = np.array(draw(st.lists(weights, min_size=n, max_size=n)))
    return u, e, w


@st.composite
def routing_data(draw, max_n=80):
    n = draw(st.integers(1, max_n))
    p = np.array(draw(st.lists(st.one_of(unit, coarse_unit), min_size=n, max_size=n)))
    small = np.array(draw(st.lists(st.sampled_from([0.0, 1.0]), min_size=n, max_size=n)))
    large = np.array(draw(st.lists(st.sampled_from([0.0, 1.0]), min_size=n, max_size=n)))
    return p, small, large


@settings(max_examples=200, deadline=None)
@given(data=calibration_data())
def test_pav_is_monotone_mean_preserving_and_optimal(data):
    _, y, w = data
    fit = pav(y, w)
    assert np.all(np.diff(fit) >= -1e-12)
    assert np.average(fit, weights=w) == pytest.approx(np.average(y, weights=w), abs=1e-9)
    loss = float(np.sum(w * (y - fit) ** 2))
    rng = np.random.default_rng(0)
    for _ in range(10):
        cand = np.sort(fit + rng.normal(scale=0.05, size=fit.size))
        assert loss <= float(np.sum(w * (y - cand) ** 2)) + 1e-9


@settings(max_examples=200, deadline=None)
@given(data=calibration_data())
def test_calibrator_matches_sklearn(data):
    sk = pytest.importorskip("sklearn.isotonic")
    u, e, w = data
    ours = IsotonicCalibrator().fit(u, e, sample_weight=w)
    ref = sk.IsotonicRegression(out_of_bounds="clip").fit(u, e, sample_weight=w)
    grid = np.concatenate([np.linspace(-0.5, 1.5, 81), u])
    np.testing.assert_allclose(ours.predict(grid), ref.predict(grid), rtol=0, atol=1e-9)
    assert np.all(np.diff(ours.x_) > 0) and np.all(np.diff(ours.y_) >= 0)
    assert ours.y_.min() >= 0.0 and ours.y_.max() <= 1.0


@settings(max_examples=100, deadline=None)
@given(data=calibration_data())
def test_calibrator_json_round_trip(data):
    u, e, w = data
    cal = IsotonicCalibrator().fit(u, e, sample_weight=w)
    back = IsotonicCalibrator.from_dict(cal.to_dict())
    grid = np.linspace(-0.5, 1.5, 101)
    np.testing.assert_array_equal(back.predict(grid), cal.predict(grid))


@settings(max_examples=200, deadline=None)
@given(data=routing_data(), tau=unit)
def test_select_threshold_matches_loop(data, tau):
    p, small, large = data
    grid = np.round(np.linspace(0.0, 1.0, 21), 2)
    rows = []
    for theta in grid:
        m = p > theta
        rows.append(
            (
                float(theta),
                policy_cost(m, 1.0, 3.02),
                float(np.where(m, large, small).mean()),
            )
        )
    feasible = [r for r in rows if r[2] >= tau]
    if not feasible:
        with pytest.raises(InfeasibleTargetError):
            select_threshold(p, small, large, tau, grid=grid)
        return
    want = min(feasible, key=lambda r: (r[1], -r[2], -r[0]))
    got = select_threshold(p, small, large, tau, grid=grid)
    assert (got.theta, got.cost, got.accuracy) == want
    front = pareto_frontier(p, small, large, grid=grid)
    np.testing.assert_array_equal(front.accuracy, [r[2] for r in rows])
    np.testing.assert_array_equal(front.cost, [r[1] for r in rows])


@settings(max_examples=200, deadline=None)
@given(
    pairs=st.lists(st.tuples(unit, unit), min_size=1, max_size=40),
)
def test_uncertainty_in_unit_interval(pairs):
    p1 = np.array([max(a, b) for a, b in pairs])
    p2 = np.array([min(a, b) for a, b in pairs])
    u = uncertainty_from_probs(p1, p2)
    assert 0.0 <= u <= 1.0
    assert batch_uncertainty([p1], [p2])[0] == pytest.approx(u, abs=1e-12)


@settings(max_examples=100, deadline=None)
@given(data=calibration_data(max_n=40), theta=unit)
def test_router_save_load_preserves_decisions(tmp_path_factory, data, theta):
    u, e, w = data
    r = UCCIRouter().calibrate(u, e, sample_weight=w)
    r.theta = theta
    path = tmp_path_factory.mktemp("r") / "router.json"
    back = UCCIRouter.load(r.save(path))
    grid = np.linspace(-0.2, 1.2, 57)
    np.testing.assert_array_equal(back.escalate(grid), r.escalate(grid))


@settings(max_examples=100, deadline=None)
@given(
    p=hnp.arrays(np.float64, st.integers(1, 50), elements=unit),
    y=st.data(),
)
def test_ece_bounds(p, y):
    labels = y.draw(hnp.arrays(np.float64, p.shape, elements=st.sampled_from([0.0, 1.0])))
    for strategy in ("uniform", "quantile"):
        value = ece(p, labels, strategy=strategy)
        assert 0.0 <= value <= 1.0
