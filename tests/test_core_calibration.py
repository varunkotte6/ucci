"""Tests for ucci.calibration: isotonic calibration (paper Section 4.2)."""

from __future__ import annotations

import numpy as np
import pytest

from ucci import IsotonicCalibrator, ece, pav
from ucci.calibration import TIE_TOLERANCE

sk_isotonic = pytest.importorskip("sklearn.isotonic")


def _sklearn_fit(u, e, w=None):
    return sk_isotonic.IsotonicRegression(out_of_bounds="clip").fit(u, e, sample_weight=w)


def _draw(rng: np.random.Generator, n: int, *, decimals=None, soft=False, weights=False):
    u = rng.random(n)
    if decimals is not None:
        u = np.round(u, decimals)  # creates ties
    p = u**2
    e = rng.random(n) * p if soft else (rng.random(n) < p).astype(float)
    w = None
    if weights:
        w = rng.exponential(1.0, n)
        w[rng.random(n) < 0.1] = 0.0  # some zero weights
        w[0] = 1.0  # keep a positive total
    return u, e, w


# -------------------------------------------------------------- sklearn parity


@pytest.mark.parametrize("seed", range(40))
def test_matches_sklearn_predictions(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 400))
    decimals = [None, 1, 2, 3][seed % 4]
    u, e, w = _draw(rng, n, decimals=decimals, soft=seed % 3 == 0, weights=seed % 2 == 1)
    ours = IsotonicCalibrator().fit(u, e, sample_weight=w)
    ref = _sklearn_fit(u, e, w)
    grid = np.concatenate([np.linspace(-0.3, 1.3, 401), u, ours.x_])
    np.testing.assert_allclose(ours.predict(grid), ref.predict(grid), rtol=0, atol=1e-12)
    # Same knots as scikit-learn's X_thresholds_ / y_thresholds_.
    assert ours.x_.shape == ref.X_thresholds_.shape
    np.testing.assert_array_equal(ours.x_, ref.X_thresholds_)
    np.testing.assert_allclose(ours.y_, ref.y_thresholds_, rtol=0, atol=1e-12)


def test_matches_sklearn_with_heavy_ties_and_large_n():
    rng = np.random.default_rng(123)
    u = rng.choice(np.linspace(0, 1, 37), 20000)
    e = (rng.random(u.size) < 0.1 + 0.5 * u).astype(float)
    ours = IsotonicCalibrator().fit(u, e)
    ref = _sklearn_fit(u, e)
    grid = np.linspace(-0.1, 1.1, 1001)
    np.testing.assert_allclose(ours.predict(grid), ref.predict(grid), rtol=0, atol=1e-12)


def test_near_ties_pooled_like_sklearn():
    # Values closer than 1e-15 to their group start are pooled by scikit-learn.
    x = np.array([0.5, 0.5 + 4e-16, 0.5 + 8e-16, 0.5 + 1.2e-15, 0.6, 0.7, 0.7 + 2e-16])
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    ours = IsotonicCalibrator().fit(x, y)
    ref = _sklearn_fit(x, y)
    np.testing.assert_array_equal(ours.x_, ref.X_thresholds_)
    np.testing.assert_allclose(ours.y_, ref.y_thresholds_, atol=1e-12)
    assert np.finfo(np.float64).resolution == TIE_TOLERANCE


def test_single_point_and_constant_u():
    cal = IsotonicCalibrator().fit([0.3], [1.0])
    assert cal.predict(0.0) == 1.0 and cal.predict(0.9) == 1.0
    cal = IsotonicCalibrator().fit([0.3, 0.3, 0.3], [0, 1, 1])
    assert cal.x_.tolist() == [0.3]
    assert cal.predict([0.0, 0.3, 1.0]) == pytest.approx([2 / 3] * 3)
    ref = _sklearn_fit([0.3, 0.3, 0.3], [0, 1, 1])
    np.testing.assert_allclose(cal.y_, ref.y_thresholds_)


def test_zero_weights_are_dropped():
    rng = np.random.default_rng(5)
    u, e, _ = _draw(rng, 300, decimals=2)
    w = np.ones(u.size)
    w[::3] = 0.0
    with_zero = IsotonicCalibrator().fit(u, e, sample_weight=w)
    without = IsotonicCalibrator().fit(u[w > 0], e[w > 0])
    np.testing.assert_array_equal(with_zero.x_, without.x_)
    np.testing.assert_array_equal(with_zero.y_, without.y_)
    assert with_zero.n_samples_ == int((w > 0).sum())


def test_integer_weights_equal_repeated_points():
    rng = np.random.default_rng(6)
    u, e, _ = _draw(rng, 200)
    w = rng.integers(1, 4, u.size)
    weighted = IsotonicCalibrator().fit(u, e, sample_weight=w)
    repeated = IsotonicCalibrator().fit(np.repeat(u, w), np.repeat(e, w))
    grid = np.linspace(0, 1, 501)
    np.testing.assert_allclose(weighted.predict(grid), repeated.predict(grid), atol=1e-12)


# ------------------------------------------------------------- predictions


def test_known_fit_and_interpolation():
    cal = IsotonicCalibrator().fit([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
    assert cal.x_.tolist() == [0.1, 0.2, 0.3, 0.4]
    assert cal.y_.tolist() == [0.0, 0.5, 0.5, 1.0]
    assert cal.predict(0.15) == pytest.approx(0.25)
    assert cal.predict(0.35) == pytest.approx(0.75)


def test_clips_outside_calibration_range():
    cal = IsotonicCalibrator().fit([0.2, 0.4, 0.6], [0, 0, 1])
    assert cal.predict([0.0, 1.0, -5.0, 5.0]).tolist() == [0.0, 1.0, 0.0, 1.0]


def test_redundant_knots_trimmed():
    cal = IsotonicCalibrator().fit([0.1, 0.2, 0.3, 0.4, 0.5], [0, 0, 0, 0, 1])
    assert cal.x_.tolist() == [0.1, 0.4, 0.5]
    assert cal.y_.tolist() == [0.0, 0.0, 1.0]


def test_scalar_in_float_out_and_shapes():
    cal = IsotonicCalibrator().fit([0.1, 0.9], [0, 1])
    out = cal.predict(0.5)
    assert isinstance(out, float) and out == pytest.approx(0.5)
    assert isinstance(cal.predict(np.float64(0.5)), float)
    assert isinstance(cal.predict(1), float)
    zero_d = cal.predict(np.array(0.5))
    assert isinstance(zero_d, np.ndarray) and zero_d.shape == ()
    grid = np.linspace(0, 1, 12).reshape(3, 4)
    assert cal.predict(grid).shape == (3, 4)
    assert cal(0.5) == cal.predict(0.5)
    assert cal.predict([]).shape == (0,)


def test_predictions_are_monotone_probabilities():
    rng = np.random.default_rng(7)
    u, e, w = _draw(rng, 2000, decimals=3, weights=True)
    cal = IsotonicCalibrator().fit(u, e, sample_weight=w)
    grid = np.linspace(-1, 2, 3001)
    p = cal.predict(grid)
    assert np.all(np.diff(p) >= 0.0)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert np.all(np.diff(cal.x_) > 0)
    assert np.all(np.diff(cal.y_) >= 0)


def test_column_vector_input_accepted():
    u = np.array([[0.1], [0.5], [0.9]])
    cal = IsotonicCalibrator().fit(u, [0, 1, 1])
    assert cal.x_.tolist() == [0.1, 0.5, 0.9]


def test_calibration_reduces_ece_on_miscalibrated_signal():
    rng = np.random.default_rng(2)
    u = rng.beta(2, 5, 20000)
    true_p = u**3  # raw u overstates the error probability
    e = (rng.random(u.size) < true_p).astype(float)
    cal = IsotonicCalibrator().fit(u[:10000], e[:10000])
    raw = ece(u[10000:], e[10000:])
    calibrated = ece(cal.predict(u[10000:]), e[10000:])
    assert calibrated < 0.02 < raw


# ---------------------------------------------------------------- validation


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        IsotonicCalibrator().predict(0.5)
    with pytest.raises(RuntimeError, match="not fitted"):
        IsotonicCalibrator().to_dict()
    assert not IsotonicCalibrator().is_fitted


@pytest.mark.parametrize(
    ("u", "e", "w", "match"),
    [
        ([], [], None, "empty"),
        ([0.1, 0.2], [0], None, "length mismatch"),
        ([0.1, float("nan")], [0, 1], None, "NaN or inf"),
        ([0.1, float("inf")], [0, 1], None, "NaN or inf"),
        ([0.1, 0.2], [0, 2], None, r"e must lie in \[0, 1\]"),
        ([0.1, 0.2], [0, -1], None, r"e must lie in \[0, 1\]"),
        ([0.1, 0.2], [0, 1], [1, -1], "non-negative"),
        ([0.1, 0.2], [0, 1], [0, 0], "sums to zero"),
        ([0.1, 0.2], [0, 1], [1], "length 1, expected 2"),
        ([0.1, 0.2], [0, 1], [1, float("nan")], "NaN or inf"),
        ([[0.1, 0.2], [0.3, 0.4]], [0, 1], None, "1-D"),
        (["a", "b"], [0, 1], None, "numeric"),
    ],
)
def test_fit_validation(u, e, w, match):
    with pytest.raises(ValueError, match=match):
        IsotonicCalibrator().fit(u, e, sample_weight=w)


def test_predict_rejects_nan():
    cal = IsotonicCalibrator().fit([0.1, 0.9], [0, 1])
    with pytest.raises(ValueError, match="NaN or inf"):
        cal.predict([0.5, float("nan")])


# ------------------------------------------------------------ serialization


def test_to_dict_from_dict_round_trip_is_exact():
    rng = np.random.default_rng(8)
    u, e, _ = _draw(rng, 5000, decimals=4)
    cal = IsotonicCalibrator().fit(u, e)
    back = IsotonicCalibrator.from_dict(cal.to_dict())
    grid = rng.random(10000) * 1.4 - 0.2
    np.testing.assert_array_equal(back.predict(grid), cal.predict(grid))
    assert back.n_samples_ is None and back.is_fitted
    extra = dict(cal.to_dict(), note="ignored")
    np.testing.assert_array_equal(IsotonicCalibrator.from_dict(extra).x_, cal.x_)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"x": [0.1]}, "missing y"),
        ({"x": [0.2, 0.1], "y": [0.0, 1.0]}, "strictly increasing"),
        ({"x": [0.1, 0.1], "y": [0.0, 1.0]}, "strictly increasing"),
        ({"x": [0.1, 0.2], "y": [0.5, 0.4]}, "non-decreasing"),
        ({"x": [0.1, 0.2], "y": [0.5, 1.5]}, r"\[0, 1\]"),
        ({"x": [0.1, 0.2], "y": [0.5]}, "length mismatch"),
        ({"x": [], "y": []}, "empty"),
        ({"x": [0.1, float("nan")], "y": [0.1, 0.2]}, "NaN or inf"),
        ([0.1, 0.2], "mapping"),
    ],
)
def test_from_dict_rejects_malformed(data, match):
    with pytest.raises(ValueError, match=match):
        IsotonicCalibrator.from_dict(data)


def test_repr():
    assert repr(IsotonicCalibrator()) == "IsotonicCalibrator(unfitted)"
    cal = IsotonicCalibrator().fit([0.1, 0.5, 0.9], [0, 1, 1])
    assert repr(cal) == "IsotonicCalibrator(n_knots=3, u_range=[0.1, 0.9], p_range=[0, 1])"


# ---------------------------------------------------------------------- PAV


def test_pav_known_cases():
    assert pav([1.0, 3.0, 2.0, 4.0]).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert pav([3.0, 2.0, 1.0]).tolist() == [2.0, 2.0, 2.0]
    assert pav([1.0, 2.0, 3.0]).tolist() == [1.0, 2.0, 3.0]
    assert pav([5.0]).tolist() == [5.0]
    # Weighted: (3 * 1 + 1 * 3) / 4 = 1.5
    assert pav([3.0, 1.0], [1.0, 3.0]).tolist() == [1.5, 1.5]


def test_pav_matches_sklearn_isotonic_regression():
    rng = np.random.default_rng(9)
    for _ in range(30):
        n = int(rng.integers(1, 300))
        y = rng.normal(size=n).cumsum() * rng.choice([-1, 1])
        w = rng.exponential(1.0, n) + 1e-3
        np.testing.assert_allclose(
            pav(y, w), sk_isotonic.isotonic_regression(y, sample_weight=w), atol=1e-10
        )


def test_pav_invariants():
    rng = np.random.default_rng(10)
    for _ in range(50):
        n = int(rng.integers(1, 200))
        y = rng.random(n)
        w = rng.random(n) + 0.1
        fit = pav(y, w)
        # Monotone and mean preserving.
        assert np.all(np.diff(fit) >= 0.0)
        assert np.average(fit, weights=w) == pytest.approx(np.average(y, weights=w))
        # Each run of equal fitted values is the weighted mean of its targets.
        starts = np.flatnonzero(np.concatenate(([True], np.diff(fit) != 0)))
        ends = np.append(starts[1:], n)
        for a, b in zip(starts, ends):
            assert fit[a] == pytest.approx(np.average(y[a:b], weights=w[a:b]))
        # Idempotent, and equivariant under shifts and positive scaling.
        np.testing.assert_allclose(pav(fit, w), fit, atol=1e-12)
        np.testing.assert_allclose(pav(2.0 * y + 3.0, w), 2.0 * fit + 3.0, atol=1e-12)


def test_pav_is_the_least_squares_projection():
    rng = np.random.default_rng(11)
    for _ in range(20):
        n = int(rng.integers(2, 60))
        y = rng.random(n)
        w = rng.random(n) + 0.1
        fit = pav(y, w)
        best = float(np.sum(w * (y - fit) ** 2))
        for _ in range(50):
            cand = np.sort(fit + rng.normal(scale=0.05, size=n))  # any monotone vector
            assert best <= float(np.sum(w * (y - cand) ** 2)) + 1e-12


def test_pav_leaves_monotone_input_unchanged_exactly():
    y = np.array([0.1, 0.1, 0.30000000000000004, 0.7, 0.7, 1.0])
    np.testing.assert_array_equal(pav(y, np.array([1, 3, 2, 1, 5, 1.0])), y)


@pytest.mark.parametrize(
    ("y", "w", "match"),
    [
        ([], None, "empty"),
        ([1.0, 2.0], [1.0, 0.0], "strictly positive"),
        ([1.0, 2.0], [1.0, -1.0], "non-negative"),
        ([1.0, 2.0], [1.0], "length"),
        ([1.0, float("nan")], None, "NaN or inf"),
    ],
)
def test_pav_validation(y, w, match):
    with pytest.raises(ValueError, match=match):
        pav(y, w)


def test_fit_performance_smoke_n_200000():
    import time

    rng = np.random.default_rng(12)
    u = rng.random(200_000)
    e = (rng.random(u.size) < u**2).astype(float)
    start = time.perf_counter()
    cal = IsotonicCalibrator().fit(u, e)
    cal.predict(u)
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0, f"fit + predict took {elapsed:.2f}s for n=200000"
