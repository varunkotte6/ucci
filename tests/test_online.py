"""Tests for ucci.online (extension, not in the paper).

The statistical tests use fixed seeds, so every assertion is deterministic;
the bounds are set a few standard errors away from the nominal values so the
suite also holds for other seeds.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from ucci import IsotonicCalibrator, UCCIRouter, ece
from ucci.online import (
    DRIFT_TESTS,
    CalibrationMonitor,
    MonitorStats,
    RecalibratingRouter,
    binomial_z_test,
    spiegelhalter_z_test,
)

# ---------------------------------------------------------------------------
# Synthetic streams
# ---------------------------------------------------------------------------


def true_error_prob(u: np.ndarray) -> np.ndarray:
    """Pre-shift P(small model wrong | u)."""
    return u**2


def shifted_error_prob(u: np.ndarray) -> np.ndarray:
    """Post-shift P(small model wrong | u): same ordering, higher error."""
    return np.minimum(1.0, 1.8 * u**2 + 0.05)


def draw(rng: np.random.Generator, n: int, shifted: bool = False) -> tuple[np.ndarray, np.ndarray]:
    u = rng.beta(2.0, 5.0, n)
    p = shifted_error_prob(u) if shifted else true_error_prob(u)
    e = (rng.random(n) < p).astype(float)
    return u, e


# ---------------------------------------------------------------------------
# Test statistics
# ---------------------------------------------------------------------------


class TestBinomialZ:
    def test_known_value(self) -> None:
        z, pv = binomial_z_test([0.5] * 100, [1] * 60 + [0] * 40)
        assert z == pytest.approx(2.0)
        assert pv == pytest.approx(math.erfc(2.0 / math.sqrt(2.0)))
        assert pv == pytest.approx(0.0455, abs=1e-4)

    def test_sign_follows_excess_errors(self) -> None:
        assert binomial_z_test([0.2] * 50, [1] * 20 + [0] * 30)[0] > 0
        assert binomial_z_test([0.2] * 50, [0] * 50)[0] < 0

    def test_symmetry_under_relabelling(self) -> None:
        rng = np.random.default_rng(0)
        p = rng.random(300)
        e = (rng.random(300) < 0.7 * p).astype(float)
        z, pv = binomial_z_test(p, e)
        z2, pv2 = binomial_z_test(1 - p, 1 - e)
        assert z2 == pytest.approx(-z)
        assert pv2 == pytest.approx(pv)

    def test_exact_match_gives_zero(self) -> None:
        assert binomial_z_test([0.25] * 8, [1, 1, 0, 0, 0, 0, 0, 0]) == (0.0, 1.0)

    def test_degenerate_forecasts(self) -> None:
        assert binomial_z_test([0.0, 1.0], [0, 1]) == (0.0, 1.0)
        z, pv = binomial_z_test([0.0, 1.0], [1, 1])
        assert z == math.inf and pv == 0.0
        z, pv = binomial_z_test([1.0, 1.0], [0, 1])
        assert z == -math.inf and pv == 0.0

    def test_empty(self) -> None:
        assert binomial_z_test([], []) == (0.0, 1.0)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            binomial_z_test([0.5, 0.5], [1])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            binomial_z_test([1.5], [1])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            binomial_z_test([0.5], [2])
        with pytest.raises(ValueError, match="NaN"):
            binomial_z_test([float("nan")], [1])


class TestSpiegelhalterZ:
    def test_known_value(self) -> None:
        # w = 1 - 2p = (0.6, -0.6); num = 0.8 * 0.6 + (-0.8) * (-0.6) = 0.96;
        # var = 2 * 0.36 * 0.16 = 0.1152; z = 0.96 / sqrt(0.1152) = 2 sqrt(2).
        z, pv = spiegelhalter_z_test([0.2, 0.8], [1, 0])
        assert z == pytest.approx(2.0 * math.sqrt(2.0))
        assert pv == pytest.approx(math.erfc(2.0))

    def test_half_forecasts_carry_no_weight(self) -> None:
        assert spiegelhalter_z_test([0.5, 0.5], [1, 1]) == (0.0, 1.0)

    def test_degenerate(self) -> None:
        assert spiegelhalter_z_test([0.0, 1.0], [0, 1]) == (0.0, 1.0)
        z, pv = spiegelhalter_z_test([0.0, 1.0], [1, 1])
        assert z == math.inf and pv == 0.0

    def test_empty_and_validation(self) -> None:
        assert spiegelhalter_z_test([], []) == (0.0, 1.0)
        with pytest.raises(ValueError, match="same length"):
            spiegelhalter_z_test([0.5], [1, 0])

    def test_detects_overconfident_forecasts_that_binomial_misses(self) -> None:
        # Forecasts are too extreme: the truth is pulled towards 0.5, the mean
        # error rate is unchanged, so only the Spiegelhalter test should fire.
        spieg_hits = binom_hits = 0
        for seed in range(40):
            rng = np.random.default_rng(seed)
            q = rng.uniform(0.05, 0.95, 2000)
            e = (rng.random(q.size) < 0.5 + 0.3 * (q - 0.5)).astype(float)
            spieg_hits += spiegelhalter_z_test(q, e)[1] < 0.01
            binom_hits += binomial_z_test(q, e)[1] < 0.01
        assert spieg_hits == 40
        assert binom_hits <= 4


@pytest.mark.parametrize("test", DRIFT_TESTS)
def test_null_pvalues_are_roughly_uniform(test: str) -> None:
    rng = np.random.default_rng(123)
    fn = binomial_z_test if test == "binomial" else spiegelhalter_z_test
    pvals = []
    for _ in range(300):
        u, e = draw(rng, 800)
        pvals.append(fn(true_error_prob(u), e)[1])
    pv = np.array(pvals)
    # Uniform(0, 1): mean 0.5, P(p < 0.1) = 0.1.
    assert abs(pv.mean() - 0.5) < 0.06
    assert 0.05 < (pv < 0.1).mean() < 0.16


# ---------------------------------------------------------------------------
# CalibrationMonitor
# ---------------------------------------------------------------------------


class TestCalibrationMonitorBasics:
    def test_empty_stats(self) -> None:
        s = CalibrationMonitor(lambda u: u, window=10, min_count=5).stats()
        assert isinstance(s, MonitorStats)
        assert s.count == 0 and s.n_seen == 0 and s.window == 10
        assert math.isnan(s.ece) and math.isnan(s.mean_predicted)
        assert math.isnan(s.observed_error_rate)
        assert (s.z, s.p_value, s.ready, s.drift) == (0.0, 1.0, False, False)

    def test_window_keeps_most_recent_in_order(self) -> None:
        mon = CalibrationMonitor(lambda u: u, window=5, min_count=1)
        mon.update([0.1, 0.2, 0.3], [0, 0, 1])
        mon.update(0.4, 1)
        mon.update([0.5, 0.6, 0.7, 0.8], [0, 1, 0, 1])
        p, e = mon.window_arrays()
        assert p.tolist() == pytest.approx([0.4, 0.5, 0.6, 0.7, 0.8])
        assert e.tolist() == [1, 0, 1, 0, 1]
        assert len(mon) == 5 and mon.n_seen == 8
        s = mon.stats()
        assert s.count == 5 and s.n_seen == 8

    @pytest.mark.parametrize("window", [1, 2, 7, 64])
    def test_ring_buffer_matches_deque(self, window: int) -> None:
        rng = np.random.default_rng(window)
        mon = CalibrationMonitor(None, window=window, min_count=1)
        ref_p: deque[float] = deque(maxlen=window)
        ref_e: deque[float] = deque(maxlen=window)
        for _ in range(60):
            n = int(rng.integers(0, 3 * window + 2))
            p = rng.random(n)
            e = (rng.random(n) < 0.5).astype(float)
            mon.record(p, e)
            ref_p.extend(p.tolist())
            ref_e.extend(e.tolist())
            got_p, got_e = mon.window_arrays()
            assert got_p.tolist() == list(ref_p)
            assert got_e.tolist() == list(ref_e)

    def test_stats_match_direct_computation(self) -> None:
        rng = np.random.default_rng(5)
        u, e = draw(rng, 3000)
        mon = CalibrationMonitor(true_error_prob, window=1000, n_bins=10, strategy="quantile")
        mon.update(u, e)
        p_w, e_w = true_error_prob(u[-1000:]), e[-1000:]
        s = mon.stats()
        assert s.count == 1000 and s.n_seen == 3000
        assert s.mean_predicted == pytest.approx(p_w.mean())
        assert s.observed_error_rate == pytest.approx(e_w.mean())
        assert s.expected_errors == pytest.approx(p_w.sum())
        assert s.observed_errors == pytest.approx(e_w.sum())
        assert s.ece == pytest.approx(ece(p_w, e_w, n_bins=10, strategy="quantile"))
        z, pv = binomial_z_test(p_w, e_w)
        assert s.z == pytest.approx(z)
        assert s.p_value == pytest.approx(pv)
        assert s.test == "binomial" and s.alpha == 0.01 and s.ready

    def test_batch_and_single_updates_agree(self) -> None:
        rng = np.random.default_rng(6)
        u, e = draw(rng, 700)
        a = CalibrationMonitor(true_error_prob, window=256)
        b = CalibrationMonitor(true_error_prob, window=256)
        a.update(u, e)
        for ui, ei in zip(u, e):
            b.update(float(ui), float(ei))
        assert a.stats() == b.stats()

    def test_as_dict(self) -> None:
        mon = CalibrationMonitor(lambda u: u, window=4, min_count=1)
        mon.update([0.1, 0.9], [0, 1])
        d = mon.stats().as_dict()
        assert set(d) == {
            "count",
            "n_seen",
            "window",
            "mean_predicted",
            "observed_error_rate",
            "expected_errors",
            "observed_errors",
            "ece",
            "test",
            "z",
            "p_value",
            "alpha",
            "ready",
            "drift",
        }
        assert d["count"] == 2

    def test_reset(self) -> None:
        mon = CalibrationMonitor(lambda u: u, window=4, min_count=1)
        mon.update([0.1, 0.9], [0, 1])
        mon.reset()
        assert len(mon) == 0 and mon.n_seen == 0 and mon.stats().count == 0

    def test_min_count_gates_the_flag(self) -> None:
        mon = CalibrationMonitor(lambda u: np.zeros_like(u) + 0.01, window=500, min_count=100)
        mon.update(np.full(99, 0.5), np.ones(99))
        s = mon.stats()
        assert s.p_value < 1e-10 and not s.ready and not s.drift
        mon.update(0.5, 1)
        s = mon.stats()
        assert s.ready and s.drift

    def test_repr(self) -> None:
        assert "window=4" in repr(CalibrationMonitor(lambda u: u, window=4, min_count=1))


class TestForecasterResolution:
    def _data(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(7)
        return draw(rng, 2000)

    def test_isotonic_calibrator(self) -> None:
        u, e = self._data()
        cal = IsotonicCalibrator().fit(u, e)
        mon = CalibrationMonitor(cal, window=100, min_count=1)
        mon.update(u[:10], e[:10])
        assert mon.window_arrays()[0] == pytest.approx(cal.predict(u[:10]))

    def test_ucci_router(self) -> None:
        u, e = self._data()
        router = UCCIRouter(c_small=1.0, c_large=3.02).calibrate(u, e)
        mon = CalibrationMonitor(router, window=100, min_count=1)
        mon.update(u[:10], e[:10])
        assert mon.window_arrays()[0] == pytest.approx(router.error_probability(u[:10]))

    def test_recalibrating_router(self) -> None:
        u, e = self._data()
        rr = RecalibratingRouter(0.3, calibrator=IsotonicCalibrator().fit(u, e))
        mon = CalibrationMonitor(rr, window=100, min_count=1)
        mon.update(u[:10], e[:10])
        assert mon.window_arrays()[0] == pytest.approx(rr.error_probability(u[:10]))

    def test_plain_callable(self) -> None:
        mon = CalibrationMonitor(lambda u: 0.5 * u, window=10, min_count=1)
        mon.update([0.2, 0.4], [0, 0])
        assert mon.window_arrays()[0].tolist() == pytest.approx([0.1, 0.2])

    def test_rejects_non_forecaster(self) -> None:
        with pytest.raises(TypeError, match="calibrator"):
            CalibrationMonitor(object(), window=10)  # type: ignore[arg-type]

    def test_bad_forecasts_are_caught(self) -> None:
        mon = CalibrationMonitor(lambda u: u * 2.0, window=10, min_count=1)
        with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
            mon.update([0.9], [1])
        mon = CalibrationMonitor(lambda u: np.array([0.1, 0.2]), window=10, min_count=1)
        with pytest.raises(ValueError, match="returned 2 values for 1"):
            mon.update([0.9], [1])

    def test_no_forecaster(self) -> None:
        mon = CalibrationMonitor(None, window=10, min_count=1)
        with pytest.raises(RuntimeError, match="record"):
            mon.update([0.1], [0])
        mon.record([0.1, 0.2], [0, 1])
        assert mon.stats().count == 2


class TestCalibrationMonitorValidation:
    @pytest.mark.parametrize(
        "kwargs, exc, match",
        [
            ({"window": 0}, ValueError, "window"),
            ({"window": 2.5}, ValueError, "window"),
            ({"window": True}, ValueError, "window"),
            ({"alpha": 0.0}, ValueError, "alpha"),
            ({"alpha": 1.0}, ValueError, "alpha"),
            ({"alpha": float("nan")}, ValueError, "alpha"),
            ({"alpha": "x"}, ValueError, "alpha"),
            ({"test": "ks"}, ValueError, "test"),
            ({"min_count": 0}, ValueError, "min_count"),
            ({"window": 10, "min_count": 11}, ValueError, "exceed"),
            ({"n_bins": 0}, ValueError, "n_bins"),
            ({"strategy": "equal"}, ValueError, "strategy"),
        ],
    )
    def test_constructor(self, kwargs: dict[str, object], exc: type[Exception], match: str) -> None:
        kw: dict[str, object] = {"window": 100, "min_count": 10}
        kw.update(kwargs)
        with pytest.raises(exc, match=match):
            CalibrationMonitor(lambda u: u, **kw)  # type: ignore[arg-type]

    def test_update_inputs(self) -> None:
        mon = CalibrationMonitor(lambda u: u, window=10, min_count=1)
        with pytest.raises(ValueError, match="same length"):
            mon.update([0.1, 0.2], [1])
        with pytest.raises(ValueError, match="NaN"):
            mon.update([float("nan")], [1])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            mon.update([0.1], [-1])
        with pytest.raises(ValueError, match="numeric"):
            mon.update(["a"], [1])
        assert mon.n_seen == 0


# ---------------------------------------------------------------------------
# Drift detection behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("test", DRIFT_TESTS)
def test_false_alarm_rate_matches_alpha(test: str) -> None:
    """With exactly calibrated forecasts the flag fires at about rate alpha."""
    alpha, n_seeds = 0.05, 400
    alarms = 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(10_000 + seed)
        mon = CalibrationMonitor(true_error_prob, window=1000, alpha=alpha, test=test)
        mon.update(*draw(rng, 1500))
        alarms += mon.stats().drift
    rate = alarms / n_seeds
    # Binomial(400, 0.05) has sd 0.011; allow about 3 sd either side.
    assert 0.018 < rate < 0.085, rate


@pytest.mark.parametrize("test", DRIFT_TESTS)
def test_quiet_on_fitted_isotonic_map_without_shift(test: str) -> None:
    alarms = 0
    for seed in range(60):
        rng = np.random.default_rng(20_000 + seed)
        cal = IsotonicCalibrator().fit(*draw(rng, 10_000))
        mon = CalibrationMonitor(cal, window=1000, alpha=0.01, test=test)
        mon.update(*draw(rng, 1000))
        alarms += mon.stats().drift
    assert alarms <= 3


@pytest.mark.parametrize("test", DRIFT_TESTS)
def test_detects_injected_shift(test: str) -> None:
    detected = 0
    delays = []
    for seed in range(50):
        rng = np.random.default_rng(30_000 + seed)
        cal = IsotonicCalibrator().fit(*draw(rng, 10_000))
        mon = CalibrationMonitor(cal, window=500, alpha=0.01, test=test)
        mon.update(*draw(rng, 500))
        u, e = draw(rng, 1000, shifted=True)
        first = None
        for i in range(0, 1000, 25):
            mon.update(u[i : i + 25], e[i : i + 25])
            if first is None and mon.stats().drift:
                first = i + 25
        if first is not None:
            detected += 1
            delays.append(first)
    assert detected == 50
    assert float(np.median(delays)) <= 300


def test_monitor_is_deterministic() -> None:
    def run() -> MonitorStats:
        rng = np.random.default_rng(99)
        mon = CalibrationMonitor(true_error_prob, window=300, test="spiegelhalter")
        for _ in range(10):
            mon.update(*draw(rng, 70))
        return mon.stats()

    assert run() == run()


# ---------------------------------------------------------------------------
# RecalibratingRouter
# ---------------------------------------------------------------------------


class TestRecalibratingRouterSchedule:
    def test_refit_points_one_label_at_a_time(self) -> None:
        rng = np.random.default_rng(1)
        u, e = draw(rng, 200)
        rr = RecalibratingRouter(0.5, window=100, refit_every=30, min_labels=50)
        counts = []
        for ui, ei in zip(u, e):
            counts.append(rr.update(ui, ei))
        # First refit waits for min_labels=50; then every 30 labels.
        assert rr.refit_history == (50, 80, 110, 140, 170, 200)
        assert sum(counts) == rr.n_refits == 6
        assert rr.labels_in_window == 100 and rr.n_labels_seen == 200
        assert rr.labels_since_refit == 0

    def test_min_labels_below_refit_every(self) -> None:
        rr = RecalibratingRouter(0.5, window=100, refit_every=40, min_labels=10)
        rr.update(np.linspace(0, 1, 130), np.zeros(130))
        assert rr.refit_history == (40, 80, 120)
        assert rr.labels_since_refit == 10

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_batches_match_single_labels(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        u, e = draw(rng, 1500)
        a = RecalibratingRouter(0.2, window=400, refit_every=70, min_labels=150)
        b = RecalibratingRouter(0.2, window=400, refit_every=70, min_labels=150)
        for ui, ei in zip(u, e):
            a.update(ui, ei)
        pos = 0
        while pos < u.size:
            n = int(rng.integers(1, 500))
            b.update(u[pos : pos + n], e[pos : pos + n])
            pos += n
        assert a.refit_history == b.refit_history
        for x, y in zip(a.window_arrays(), b.window_arrays()):
            assert np.array_equal(x, y)
        grid = np.linspace(-0.1, 1.1, 121)
        assert np.array_equal(a.error_probability(grid), b.error_probability(grid))

    def test_refit_uses_exactly_the_window(self) -> None:
        rng = np.random.default_rng(3)
        u, e = draw(rng, 1000)
        rr = RecalibratingRouter(0.2, window=300, refit_every=100, min_labels=100)
        rr.update(u, e)
        ref = IsotonicCalibrator().fit(u[-300:], e[-300:])
        grid = np.linspace(-0.1, 1.1, 241)
        assert np.array_equal(rr.error_probability(grid), ref.predict(grid))

    def test_manual_refit(self) -> None:
        rr = RecalibratingRouter(0.5, window=50, refit_every=1000, min_labels=20)
        rr.update(np.linspace(0, 1, 19), np.zeros(19))
        with pytest.raises(ValueError, match="min_labels"):
            rr.refit()
        rr.update(0.99, 1.0)
        rr.refit()
        assert rr.n_refits == 1 and rr.labels_since_refit == 0 and rr.is_calibrated

    def test_repr(self) -> None:
        assert "refit_every=5" in repr(
            RecalibratingRouter(0.5, window=10, refit_every=5, min_labels=5)
        )


class TestRecalibratingRouterRouting:
    def test_uncalibrated_router_raises_until_first_refit(self) -> None:
        rr = RecalibratingRouter(0.5, window=100, refit_every=10, min_labels=20)
        assert not rr.is_calibrated and rr.calibrator is None
        with pytest.raises(RuntimeError, match="min_labels"):
            rr.error_probability(0.3)
        rr.update(np.linspace(0, 1, 20), (np.linspace(0, 1, 20) > 0.5).astype(float))
        assert rr.is_calibrated
        assert rr.escalate([0.1, 0.9]).tolist() == [False, True]

    def test_theta_is_a_probability_threshold(self) -> None:
        cal = IsotonicCalibrator().fit([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1])
        rr = RecalibratingRouter(0.5, calibrator=cal)
        u = np.array([0.05, 0.15, 0.25, 0.35, 0.45])
        assert np.array_equal(rr.escalate(u), cal.predict(u) > 0.5)
        # Scalars and shapes behave as in UCCIRouter.
        p = rr.error_probability(0.4)
        assert isinstance(p, float) and p == float(cal.predict(0.4))
        assert rr.escalate(0.4) is True and rr.escalate(0.1) is False
        assert rr.error_probability([[0.1, 0.4]]).shape == (1, 2)
        assert rr.escalate(np.asarray(0.4)).shape == ()

    def test_route_matches_escalate(self) -> None:
        rng = np.random.default_rng(12)
        cal = IsotonicCalibrator().fit(*draw(rng, 1000))
        rr = RecalibratingRouter(0.2, calibrator=cal)
        u = rng.random(50)
        res = rr.route(u)
        assert np.array_equal(res.escalate, rr.escalate(u))
        assert np.array_equal(res.p_hat, cal.predict(u))

    def test_from_router_routes_like_router_until_refit(self) -> None:
        rng = np.random.default_rng(8)
        u_c, e_c = draw(rng, 3000)
        u_v, e_v = draw(rng, 2000)
        large = (rng.random(2000) < 0.95).astype(float)
        router = UCCIRouter(c_small=1.0, c_large=3.02).calibrate(u_c, e_c)
        router.choose_threshold(u_v, 1 - e_v, large, tau=0.9)
        before = router.calibrator.predict(np.linspace(0, 1, 11)).copy()
        rr = RecalibratingRouter.from_router(router, window=1000, refit_every=500, min_labels=500)
        assert rr.theta == router.theta
        grid = np.linspace(-0.2, 1.2, 57)
        assert np.array_equal(rr.escalate(grid), router.escalate(grid))
        rr.update(*draw(rng, 1000, shifted=True))
        assert rr.n_refits == 2
        # The input router is untouched.
        assert np.array_equal(router.calibrator.predict(np.linspace(0, 1, 11)), before)

    def test_from_router_needs_threshold(self) -> None:
        router = UCCIRouter(c_small=1.0, c_large=3.02).calibrate([0.1, 0.9], [0, 1])
        with pytest.raises(ValueError, match="choose_threshold"):
            RecalibratingRouter.from_router(router)
        with pytest.raises(TypeError, match="calibrator"):
            RecalibratingRouter.from_router(object())

    def test_initial_calibrator_is_not_modified(self) -> None:
        rng = np.random.default_rng(9)
        cal = IsotonicCalibrator().fit(*draw(rng, 500))
        grid = np.linspace(0, 1, 101)
        before = cal.predict(grid).copy()
        rr = RecalibratingRouter(0.3, calibrator=cal, window=200, refit_every=100, min_labels=100)
        rr.update(*draw(rng, 500, shifted=True))
        assert rr.calibrator is not cal
        assert np.array_equal(cal.predict(grid), before)


class TestSnapshot:
    def test_to_router_round_trips_through_the_router_format(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(13)
        u_c, e_c = draw(rng, 3000)
        u_v, e_v = draw(rng, 2000)
        router = UCCIRouter(c_small=1.0, c_large=5.0, cost_model="sequential").calibrate(u_c, e_c)
        router.choose_threshold(u_v, 1 - e_v, np.ones(u_v.size), tau=0.95)
        rr = RecalibratingRouter.from_router(router, window=800, refit_every=400, min_labels=400)
        rr.update(*draw(rng, 1200, shifted=True))
        snap = rr.to_router()
        assert (snap.c_small, snap.c_large, snap.cost_model) == (1.0, 5.0, "sequential")
        assert snap.theta == rr.theta and snap.tau is None
        grid = np.linspace(-0.1, 1.1, 97)
        assert np.array_equal(snap.escalate(grid), rr.escalate(grid))
        path = snap.save(tmp_path / "recal.json")
        loaded = UCCIRouter.load(path)
        assert np.array_equal(loaded.error_probability(grid), rr.error_probability(grid))
        # The snapshot is independent of later refits.
        before = snap.error_probability(grid).copy()
        rr.update(*draw(rng, 800))
        assert rr.n_refits == 5
        assert np.array_equal(snap.error_probability(grid), before)

    def test_to_router_cost_overrides_and_defaults(self) -> None:
        cal = IsotonicCalibrator().fit([0.1, 0.9], [0, 1])
        rr = RecalibratingRouter(0.4, calibrator=cal)
        snap = rr.to_router()
        assert (snap.c_small, snap.c_large, snap.cost_model) == (1.0, 3.02, "routing")
        snap = rr.to_router(c_small=2.0, c_large=7.0, cost_model="sequential")
        assert (snap.c_small, snap.c_large, snap.cost_model) == (2.0, 7.0, "sequential")

    def test_to_router_errors(self) -> None:
        with pytest.raises(RuntimeError, match="no calibration map"):
            RecalibratingRouter(0.5).to_router()

        class Constant:
            def predict(self, u: np.ndarray) -> np.ndarray:
                return np.full(np.shape(u), 0.3)

        rr = RecalibratingRouter(0.5, calibrator=Constant())
        assert rr.escalate(0.9) is False
        with pytest.raises(TypeError, match="IsotonicCalibrator"):
            rr.to_router()


class TestRecalibratingRouterValidation:
    @pytest.mark.parametrize("theta", [-0.1, 1.1, float("nan"), float("inf")])
    def test_theta(self, theta: float) -> None:
        with pytest.raises(ValueError, match="theta"):
            RecalibratingRouter(theta)

    @pytest.mark.parametrize(
        "kwargs, exc, match",
        [
            ({"window": 0}, ValueError, "window"),
            ({"refit_every": 0}, ValueError, "refit_every"),
            ({"min_labels": 0}, ValueError, "min_labels"),
            ({"window": 10, "min_labels": 11}, ValueError, "exceed"),
            ({"refit_every": 1.5}, ValueError, "refit_every"),
            ({"calibrator": object()}, TypeError, "predict"),
        ],
    )
    def test_constructor(self, kwargs: dict[str, object], exc: type[Exception], match: str) -> None:
        kw: dict[str, object] = {"window": 100, "refit_every": 10, "min_labels": 10}
        kw.update(kwargs)
        with pytest.raises(exc, match=match):
            RecalibratingRouter(0.5, **kw)  # type: ignore[arg-type]

    def test_update_inputs(self) -> None:
        rr = RecalibratingRouter(0.5, window=10, refit_every=5, min_labels=5)
        with pytest.raises(ValueError, match="same length"):
            rr.update([0.1, 0.2], [1])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            rr.update([0.1], [1.5])
        with pytest.raises(ValueError, match="NaN"):
            rr.update([float("inf")], [1])
        assert rr.n_labels_seen == 0 and rr.labels_in_window == 0


# ---------------------------------------------------------------------------
# Recalibration after a shift
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_recalibration_restores_calibration_after_shift(seed: int) -> None:
    """Prequential check: each label is scored on the forecast issued before it
    arrived. The static map drifts; the sliding-window refit recovers."""
    rng = np.random.default_rng(40_000 + seed)
    static = IsotonicCalibrator().fit(*draw(rng, 5000))
    rr = RecalibratingRouter(0.3, calibrator=static, window=2000, refit_every=250, min_labels=250)
    mon_static = CalibrationMonitor(static, window=1000, alpha=0.01)
    mon_recal = CalibrationMonitor(rr, window=1000, alpha=0.01)

    u, e = draw(rng, 6000, shifted=True)
    for i in range(0, u.size, 50):
        ub, eb = u[i : i + 50], e[i : i + 50]
        mon_static.update(ub, eb)
        mon_recal.update(ub, eb)  # forecast first ...
        rr.update(ub, eb)  # ... then learn from the labels

    s_static, s_recal = mon_static.stats(), mon_recal.stats()
    assert s_static.drift and s_static.ece > 0.08
    assert s_recal.ece < 0.05
    assert s_recal.ece < 0.5 * s_static.ece
    assert abs(s_recal.mean_predicted - s_recal.observed_error_rate) < 0.03
    # Theta is fixed in probability space, so higher post-shift error
    # probabilities move the implied u cutoff down: more queries escalate.
    u_new, _ = draw(rng, 5000, shifted=True)
    assert rr.escalate(u_new).mean() > (static.predict(u_new) > 0.3).mean()
