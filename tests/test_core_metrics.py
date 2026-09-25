"""Tests for ucci.metrics: ECE, reliability tables, Brier, bootstrap, micro-F1."""

from __future__ import annotations

import numpy as np
import pytest

from ucci import (
    ConfidenceInterval,
    ReliabilityRow,
    RoutedMicroF1,
    bootstrap_ci,
    brier_score,
    ece,
    micro_f1,
    reliability_table,
    routed_micro_f1,
    select_threshold,
)

# ------------------------------------------------------------------ ECE


def test_ece_zero_when_forecast_equals_bin_frequency():
    p = np.array([0.25] * 4 + [0.75] * 4)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    assert ece(p, y) == pytest.approx(0.0)


def test_ece_by_hand():
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0, 1, 1, 1])
    # bin (0, 0.1]: |0.1 - 0.5| = 0.4 ; bin (0.8, 0.9]: |0.9 - 1.0| = 0.1
    assert ece(p, y) == pytest.approx(0.5 * 0.4 + 0.5 * 0.1)


def test_ece_matches_definition_on_random_data():
    rng = np.random.default_rng(0)
    for strategy in ("uniform", "quantile"):
        p = rng.random(5000)
        y = (rng.random(5000) < p**1.5).astype(float)
        rows = reliability_table(p, y, 10, strategy)
        by_hand = sum(r.count / 5000 * abs(r.mean_forecast - r.observed_frequency) for r in rows)
        assert ece(p, y, 10, strategy) == pytest.approx(by_hand, abs=1e-12)
        assert sum(r.count for r in rows) == 5000


def test_reliability_table_matches_sklearn_calibration_curve():
    calibration = pytest.importorskip("sklearn.calibration")
    rng = np.random.default_rng(1)
    p = rng.random(3000)
    y = (rng.random(3000) < p).astype(int)
    prob_true, prob_pred = calibration.calibration_curve(y, p, n_bins=10)
    rows = reliability_table(p, y, n_bins=10)
    np.testing.assert_allclose([r.observed_frequency for r in rows], prob_true, atol=1e-12)
    np.testing.assert_allclose([r.mean_forecast for r in rows], prob_pred, atol=1e-12)


def test_reliability_rows_are_named():
    rows = reliability_table([0.05, 0.15, 0.15, 0.95], [0, 0, 1, 1])
    assert all(isinstance(r, ReliabilityRow) for r in rows)
    assert [r.count for r in rows] == [1, 2, 1]
    assert rows[1] == ReliabilityRow(0.1, 0.2, 2, 0.15, 0.5)
    assert rows[1]._fields == (
        "bin_lower",
        "bin_upper",
        "count",
        "mean_forecast",
        "observed_frequency",
    )
    lo, hi, n, mp, freq = rows[0]
    assert (lo, hi, n, mp, freq) == (0.0, pytest.approx(0.1), 1, 0.05, 0.0)


def test_uniform_bins_are_right_closed():
    rows = reliability_table([0.0, 0.1, 0.1000001, 1.0], [0, 0, 1, 1], n_bins=10)
    assert [(round(r.bin_lower, 1), r.count) for r in rows] == [
        (0.0, 2),
        (0.1, 1),
        (0.9, 1),
    ]


def test_quantile_bins_are_deciles():
    p = np.arange(1, 101) / 100.0
    y = np.zeros(100)
    rows = reliability_table(p, y, n_bins=10, strategy="quantile")
    assert len(rows) == 10
    assert [r.count for r in rows] == [10] * 10


def test_quantile_bins_with_repeated_forecasts_merge():
    p = np.array([0.2] * 50 + [0.7] * 50)
    y = np.array([0] * 40 + [1] * 10 + [1] * 35 + [0] * 15)
    rows = reliability_table(p, y, n_bins=10, strategy="quantile")
    assert [(r.count, r.mean_forecast) for r in rows] == [
        (50, pytest.approx(0.2)),
        (50, pytest.approx(0.7)),
    ]
    assert ece(p, y, strategy="quantile") == pytest.approx(0.0)
    const = reliability_table([0.3] * 5, [0, 1, 0, 0, 1], strategy="quantile")
    assert len(const) == 1 and const[0][:3] == (0.3, 0.3, 5)
    assert const[0].mean_forecast == pytest.approx(0.3)
    assert const[0].observed_frequency == pytest.approx(0.4)


def test_weights_equal_repetition_and_zero_weights_drop():
    rng = np.random.default_rng(2)
    p = rng.random(300)
    y = (rng.random(300) < p).astype(float)
    w = rng.integers(0, 4, 300).astype(float)
    w[0] = 1.0
    weighted = ece(p, y, sample_weight=w)
    keep = w > 0
    repeated = ece(np.repeat(p, w.astype(int)), np.repeat(y, w.astype(int)))
    assert weighted == pytest.approx(repeated, abs=1e-12)
    rows = reliability_table(p, y, sample_weight=w)
    assert sum(r.count for r in rows) == int(keep.sum())


@pytest.mark.parametrize(
    ("p", "y", "kw", "match"),
    [
        ([0.5, 1.2], [0, 1], {}, r"p must lie in \[0, 1\]"),
        ([0.5, -0.1], [0, 1], {}, r"p must lie in \[0, 1\]"),
        ([0.5, 0.2], [0, 2], {}, r"y must lie in \[0, 1\]"),
        ([0.5, 0.2], [0], {}, "length mismatch"),
        ([], [], {}, "empty"),
        ([0.5], [1], {"n_bins": 0}, "at least 1"),
        ([0.5], [1], {"n_bins": 2.5}, "integer"),
        ([0.5], [1], {"n_bins": True}, "integer"),
        ([0.5], [1], {"strategy": "kmeans"}, "strategy"),
        ([0.5], [1], {"sample_weight": [-1.0]}, "non-negative"),
        ([0.5, float("nan")], [1, 0], {}, "NaN or inf"),
    ],
)
def test_forecast_metric_validation(p, y, kw, match):
    with pytest.raises(ValueError, match=match):
        ece(p, y, **kw)
    with pytest.raises(ValueError, match=match):
        reliability_table(p, y, **kw)


# ---------------------------------------------------------------- Brier


def test_brier_score():
    assert brier_score([0.0, 1.0], [0, 1]) == 0.0
    assert brier_score([0.5, 0.5], [0, 1]) == 0.25
    assert brier_score([0.2, 0.9], [0, 1], sample_weight=[3, 1]) == pytest.approx(
        (3 * 0.04 + 1 * 0.01) / 4
    )
    with pytest.raises(ValueError, match="p must lie"):
        brier_score([2.0], [1])


# ------------------------------------------------------------ bootstrap


def test_bootstrap_ci_is_deterministic_and_covers_the_mean():
    x = np.random.default_rng(3).normal(size=400)
    ci = bootstrap_ci(lambda idx: float(x[idx].mean()), x.size)
    assert isinstance(ci, ConfidenceInterval)
    assert ci == bootstrap_ci(lambda idx: float(x[idx].mean()), x.size)
    assert ci.low < x.mean() < ci.high
    lo, hi = ci
    assert (lo, hi) == (ci.low, ci.high)
    other = bootstrap_ci(lambda idx: float(x[idx].mean()), x.size, seed=1)
    assert other != ci
    # Rough width check: 95% CI of a mean is about 2 * 1.96 * sd / sqrt(n).
    assert 0.12 < ci.high - ci.low < 0.28


def test_bootstrap_ci_uses_n_boot_resamples_of_n_indices():
    seen = []

    def stat(idx):
        seen.append(idx)
        return float(idx.mean())

    bootstrap_ci(stat, 17, n_boot=33, alpha=0.1, seed=None)
    assert len(seen) == 33
    assert all(i.shape == (17,) and i.min() >= 0 and i.max() < 17 for i in seen)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"n": 0}, "n must be at least 1"),
        ({"n_boot": 0}, "n_boot must be at least 1"),
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.0}, "alpha"),
        ({"alpha": float("nan")}, "alpha"),
    ],
)
def test_bootstrap_ci_validation(kw, match):
    args = {"n": 10, **kw}
    with pytest.raises(ValueError, match=match):
        bootstrap_ci(lambda idx: 0.0, **args)


def test_bootstrap_ci_rejects_non_finite_statistic():
    with pytest.raises(ValueError, match="resample 0"):
        bootstrap_ci(lambda idx: float("nan"), 5)


# ------------------------------------------------------------- micro-F1


def test_micro_f1():
    assert micro_f1(2, 0, 1) == pytest.approx(0.8)
    assert micro_f1([2, 1], [0, 1], [1, 0]) == 0.75
    assert micro_f1(np.array([[1, 1], [1, 0]]), 0, 0) == 1.0
    assert micro_f1(0, 0, 0) == 0.0
    assert micro_f1(0, 0, 0, zero_division=1.0) == 1.0
    with pytest.raises(ValueError, match="non-negative"):
        micro_f1(-1, 0, 0)
    with pytest.raises(ValueError, match="NaN or inf"):
        micro_f1(float("nan"), 0, 0)


def test_routed_micro_f1():
    small = np.array([[1, 0, 1], [2, 0, 0], [0, 1, 1]])
    large = np.array([[2, 0, 0], [2, 0, 0], [1, 0, 0]])
    f1 = routed_micro_f1(small, large)
    assert isinstance(f1, RoutedMicroF1)
    assert f1([False, False, False]) == micro_f1(3, 1, 2)
    assert f1([True, False, True]) == 1.0
    assert f1(np.array([True, True, True])) == 1.0
    assert repr(f1) == "RoutedMicroF1(n=3)"
    with pytest.raises(ValueError, match=r"expected \(3,\)"):
        f1([True, False])


def test_routed_micro_f1_drives_threshold_selection():
    rng = np.random.default_rng(5)
    n = 400
    p = rng.random(n)
    gold = rng.integers(1, 4, n)
    wrong = rng.random(n) < p
    small = np.stack([np.where(wrong, gold - 1, gold), wrong.astype(int), wrong.astype(int)], 1)
    large = np.stack([gold, np.zeros(n, int), np.zeros(n, int)], 1)
    f1 = routed_micro_f1(small, large)
    c = select_threshold(p, None, None, tau=0.95, metric=f1)
    assert c.accuracy >= 0.95
    assert f1(p > c.theta) == c.accuracy


@pytest.mark.parametrize(
    ("small", "large", "match"),
    [
        ([[1, 0]], [[1, 0, 0]], r"shape \(n, 3\)"),
        (np.zeros((0, 3)), np.zeros((0, 3)), r"shape \(n, 3\)"),
        ([[1, 0, -1]], [[1, 0, 0]], "non-negative"),
        ([[1, 0, 0]], [[1, 0, 0], [1, 0, 0]], "1 rows but large_counts has 2"),
        ([[1, 0, float("inf")]], [[1, 0, 0]], "NaN or inf"),
    ],
)
def test_routed_micro_f1_validation(small, large, match):
    with pytest.raises(ValueError, match=match):
        routed_micro_f1(small, large)
