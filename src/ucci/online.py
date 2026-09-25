"""Online calibration monitoring and sliding-window recalibration.

**Extension, not in the paper.** UCCI as published fits the calibration map
once, on a held-out batch (Section 4.2, and step 1 of the evaluation protocol
in Section 6.1). The paper's Discussion and Limitations (Section 7, "Static
calibration") names the gap this module fills: in streaming deployments with
distribution shift, online or continual recalibration is needed, and the same
calibrated threshold framework applies with an adapted fitting procedure.
Nothing here is used by the paper-faithful core (``ucci.signal``,
``ucci.calibration``, ``ucci.policy``, ``ucci.router``, ``ucci.metrics``), and
nothing here is on by default.

The module provides two tools.

:class:`CalibrationMonitor`
    Keeps the most recent ``window`` pairs ``(p_hat, e)`` of forecast error
    probability and observed error event (``e = 1`` when the small model was
    wrong, as in Section 4.2) and reports windowed calibration statistics:
    ECE, mean forecast against observed error rate, and a drift flag from a
    documented two-sided hypothesis test of calibration.

:class:`RecalibratingRouter`
    Keeps theta fixed as an error-probability threshold (the policy of
    Eq. 6, escalate when ``p_hat > theta``) and refits the isotonic map of
    Section 4.2 on a sliding window of the most recent labels every
    ``refit_every`` new labels, once the window holds at least ``min_labels``.
    Because theta lives on the probability scale, the implied cutoff on the
    raw signal u(x) moves with the refitted map.

Drift tests
-----------
Both tests take the windowed forecasts ``p_1..p_n`` and outcomes
``e_1..e_n``. Under the null hypothesis that the forecasts are calibrated and
the outcomes are independent, ``e_i ~ Bernoulli(p_i)``.

``"binomial"`` (calibration in the large)
    ``O = sum e_i`` has mean ``E = sum p_i`` and variance
    ``V = sum p_i (1 - p_i)`` (a Poisson-binomial count), so
    ``z = (O - E) / sqrt(V)`` is approximately standard normal. It detects a
    shift of the overall error rate away from the forecasts, in either
    direction.

``"spiegelhalter"``
    Spiegelhalter's z statistic,
    ``z = sum (e_i - p_i)(1 - 2 p_i) / sqrt(sum (1 - 2 p_i)^2 p_i (1 - p_i))``,
    which tests the Brier score against its expectation under calibration
    (D. J. Spiegelhalter, "Probabilistic prediction in patient management and
    clinical trials", Statistics in Medicine 5(5):421-433, 1986,
    doi:10.1002/sim.4780050506). It also reacts to forecasts that become too
    extreme or too flat while the mean error rate is unchanged.

Both report a two-sided p-value ``erfc(|z| / sqrt(2))`` from the normal
approximation, which is accurate once ``V`` is moderately large (tens of
expected errors). The monitor raises the drift flag only when the window
holds at least ``min_count`` labels and the p-value is below ``alpha``.

The false-alarm rate ``alpha`` holds per check. Checking after every label
over overlapping windows makes many dependent tests; to bound the chance of
any false alarm over ``K`` checks, use ``alpha / K`` (Bonferroni) or check
once per ``window`` labels. Both tests assume the labels are a representative
sample of the traffic the forecasts are issued for. If labels arrive only for
some routing decisions (for example only for escalated queries), the window
sees a biased slice of the u(x) distribution.

Everything here is deterministic and uses numpy only.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Union, overload

import numpy as np

from .calibration import IsotonicCalibrator
from .metrics import ece as _ece
from .router import RouteResult, UCCIRouter

if TYPE_CHECKING:
    from typing import Protocol

    from numpy.typing import ArrayLike, NDArray

    FloatArray = NDArray[np.float64]
    BoolArray = NDArray[np.bool_]

    class SupportsErrorProbability(Protocol):
        """Anything with ``error_probability(u)``, such as :class:`ucci.UCCIRouter`."""

        def error_probability(self, u: Any) -> Any:
            """Return p_hat = g(u)."""

    class SupportsPredict(Protocol):
        """Anything with ``predict(u)``, such as :class:`ucci.IsotonicCalibrator`."""

        def predict(self, u: Any) -> Any:
            """Return p_hat = g(u)."""

    ForecastSource = Union[SupportsErrorProbability, SupportsPredict, Callable[[Any], Any]]

__all__ = [
    "DRIFT_TESTS",
    "CalibrationMonitor",
    "MonitorStats",
    "RecalibratingRouter",
    "binomial_z_test",
    "spiegelhalter_z_test",
]

#: Names of the drift tests accepted by :class:`CalibrationMonitor`.
DRIFT_TESTS: tuple[str, ...] = ("binomial", "spiegelhalter")

_BIN_STRATEGIES: tuple[str, ...] = ("uniform", "quantile")


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _as_1d_float(values: ArrayLike, name: str) -> FloatArray:
    """Return ``values`` as a finite 1-D float64 array (scalars become length 1)."""
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric, got {values!r}") from exc
    arr = arr.reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def _as_events(values: ArrayLike, name: str = "e") -> FloatArray:
    """Return error events as floats in [0, 1] (1 = small model wrong)."""
    arr = _as_1d_float(values, name)
    if arr.size and (arr.min() < 0.0 or arr.max() > 1.0):
        raise ValueError(
            f"{name} must lie in [0, 1] (1 = small model wrong, Section 4.2); "
            f"got values in [{arr.min():g}, {arr.max():g}]"
        )
    return arr


def _check_pair(u: FloatArray, e: FloatArray, u_name: str = "u") -> None:
    if u.shape != e.shape:
        raise ValueError(f"{u_name} and e must have the same length, got {u.size} and {e.size}")


def _check_positive_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        # ValueError for every invalid argument value, as in the core modules.
        raise ValueError(f"{name} must be an integer, got {value!r}")  # noqa: TRY004
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return int(value)


def _check_probability(value: float, name: str, *, open_interval: bool) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")  # noqa: TRY004
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if open_interval and not 0.0 < v < 1.0:
        raise ValueError(f"{name} must lie in (0, 1), got {v}")
    if not open_interval and not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {v}")
    return v


def _resolve_forecaster(source: Any) -> Callable[[FloatArray], FloatArray]:
    """Turn a calibrator, router or callable into ``u -> p_hat``.

    Resolution order: ``source.error_probability`` (routers), then
    ``source.predict`` (calibrators), then ``source`` itself if callable.
    The returned function validates that forecasts are finite, lie in
    [0, 1] and have one value per input.
    """
    method = getattr(source, "error_probability", None)
    if method is None:
        method = getattr(source, "predict", None)
    if method is None and callable(source):
        method = source
    if method is None or not callable(method):
        raise TypeError(
            "expected a calibrator (with .predict), a router (with "
            ".error_probability) or a callable u -> p_hat; got "
            f"{type(source).__name__}"
        )
    fn: Callable[[Any], Any] = method

    def forecast(u: FloatArray) -> FloatArray:
        p = np.asarray(fn(u), dtype=np.float64).reshape(-1)
        if p.shape != u.shape:
            raise ValueError(f"forecaster returned {p.size} values for {u.size} inputs")
        if not np.all(np.isfinite(p)) or (p.size and (p.min() < 0.0 or p.max() > 1.0)):
            raise ValueError("forecaster returned values outside [0, 1] or non-finite values")
        return p

    return forecast


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------


def _two_sided_p(z: float) -> float:
    """Two-sided normal p-value ``P(|Z| >= |z|)``."""
    if math.isinf(z):
        return 0.0
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def _z_from(num: float, var: float) -> float:
    """``num / sqrt(var)`` with the degenerate zero-variance case handled.

    With zero variance the null hypothesis predicts ``num == 0`` exactly, so
    any non-zero numerator is infinitely significant.
    """
    tol = 1e-12
    if var <= tol:
        if abs(num) <= 1e-9:
            return 0.0
        return math.copysign(math.inf, num)
    return float(num / math.sqrt(var))


def binomial_z_test(p_hat: ArrayLike, e: ArrayLike) -> tuple[float, float]:
    """Two-sided calibration-in-the-large test of observed vs expected errors.

    Extension, not in the paper (see the module docstring).

    Parameters
    ----------
    p_hat : array_like, shape (n,)
        Forecast error probabilities in [0, 1].
    e : array_like, shape (n,)
        Observed error events in [0, 1] (1 = small model wrong, Section 4.2).

    Returns
    -------
    z : float
        ``(sum e - sum p_hat) / sqrt(sum p_hat (1 - p_hat))``. Positive when
        the small model makes more errors than forecast. ``+-inf`` when every
        forecast is exactly 0 or 1 and the counts still disagree.
    p_value : float
        Two-sided normal-approximation p-value ``erfc(|z| / sqrt(2))``.

    Notes
    -----
    Under calibration and independence the error count is Poisson-binomial
    with mean ``sum p_hat`` and variance ``sum p_hat (1 - p_hat)``. For
    fractional ``e`` (soft error scores) the variance of the count is smaller
    than this, so the test is conservative. An empty input gives
    ``(0.0, 1.0)``.

    Examples
    --------
    >>> z, pv = binomial_z_test([0.5] * 100, [1] * 60 + [0] * 40)
    >>> round(z, 6), round(pv, 4)
    (2.0, 0.0455)
    """
    p = _as_events(p_hat, "p_hat")
    y = _as_events(e, "e")
    _check_pair(p, y, "p_hat")
    num = float(y.sum() - p.sum())
    var = float(np.sum(p * (1.0 - p)))
    z = _z_from(num, var)
    return z, _two_sided_p(z)


def spiegelhalter_z_test(p_hat: ArrayLike, e: ArrayLike) -> tuple[float, float]:
    """Spiegelhalter's two-sided z test of calibration.

    Extension, not in the paper (see the module docstring). Reference:
    D. J. Spiegelhalter, "Probabilistic prediction in patient management and
    clinical trials", Statistics in Medicine 5(5):421-433, 1986,
    doi:10.1002/sim.4780050506.

    Parameters
    ----------
    p_hat : array_like, shape (n,)
        Forecast error probabilities in [0, 1].
    e : array_like, shape (n,)
        Observed error events in [0, 1] (1 = small model wrong, Section 4.2).

    Returns
    -------
    z : float
        ``sum (e - p)(1 - 2p) / sqrt(sum (1 - 2p)^2 p (1 - p))``.
    p_value : float
        Two-sided normal-approximation p-value ``erfc(|z| / sqrt(2))``.

    Notes
    -----
    The numerator is the Brier score minus its expectation under calibration
    (times n). Forecasts at exactly 0.5 carry zero weight. When the variance
    term is zero (every forecast in {0, 0.5, 1}) the statistic is 0 if the
    numerator is 0 and ``+-inf`` otherwise. An empty input gives
    ``(0.0, 1.0)``.
    """
    p = _as_events(p_hat, "p_hat")
    y = _as_events(e, "e")
    _check_pair(p, y, "p_hat")
    w = 1.0 - 2.0 * p
    num = float(np.sum((y - p) * w))
    var = float(np.sum(w * w * p * (1.0 - p)))
    z = _z_from(num, var)
    return z, _two_sided_p(z)


_TESTS: dict[str, Callable[[ArrayLike, ArrayLike], tuple[float, float]]] = {
    "binomial": binomial_z_test,
    "spiegelhalter": spiegelhalter_z_test,
}


# ---------------------------------------------------------------------------
# Sliding window storage
# ---------------------------------------------------------------------------


class _RingBuffer:
    """Fixed-capacity FIFO of float columns, returned in arrival order."""

    def __init__(self, capacity: int, n_columns: int) -> None:
        self._data = np.zeros((n_columns, capacity), dtype=np.float64)
        self._capacity = capacity
        self._start = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        self._start = 0
        self._size = 0

    def extend(self, *columns: FloatArray) -> None:
        n = columns[0].size
        if n == 0:
            return
        cap = self._capacity
        if n >= cap:
            for row, col in enumerate(columns):
                self._data[row] = col[n - cap :]
            self._start, self._size = 0, cap
            return
        end = (self._start + self._size) % cap
        first = min(n, cap - end)
        for row, col in enumerate(columns):
            self._data[row, end : end + first] = col[:first]
            self._data[row, : n - first] = col[first:]
        overflow = max(0, self._size + n - cap)
        self._start = (self._start + overflow) % cap
        self._size = min(cap, self._size + n)

    def columns(self) -> list[FloatArray]:
        idx = (self._start + np.arange(self._size)) % self._capacity
        return [np.array(self._data[row, idx]) for row in range(self._data.shape[0])]


# ---------------------------------------------------------------------------
# CalibrationMonitor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MonitorStats:
    """Windowed calibration statistics returned by :meth:`CalibrationMonitor.stats`.

    Extension, not in the paper (see the :mod:`ucci.online` docstring).

    Attributes
    ----------
    count : int
        Labels currently in the window (at most ``window``).
    n_seen : int
        Labels received since construction or the last :meth:`~CalibrationMonitor.reset`.
    window : int
        Window capacity.
    mean_predicted : float
        Mean forecast error probability over the window (NaN when empty).
    observed_error_rate : float
        Mean observed error event over the window (NaN when empty).
    expected_errors : float
        Sum of forecasts over the window.
    observed_errors : float
        Sum of observed error events over the window.
    ece : float
        Expected calibration error of the window (:func:`ucci.ece` with the
        monitor's ``n_bins`` and ``strategy``; NaN when empty). ECE is biased
        upward on small windows.
    test : str
        Drift test used, ``"binomial"`` or ``"spiegelhalter"``.
    z : float
        Test statistic (0.0 when empty).
    p_value : float
        Two-sided p-value of the test (1.0 when empty).
    alpha : float
        Significance level of the drift flag.
    ready : bool
        True when the window holds at least ``min_count`` labels.
    drift : bool
        True when ``ready`` and ``p_value < alpha``.
    """

    count: int
    n_seen: int
    window: int
    mean_predicted: float
    observed_error_rate: float
    expected_errors: float
    observed_errors: float
    ece: float
    test: str
    z: float
    p_value: float
    alpha: float
    ready: bool
    drift: bool

    def as_dict(self) -> dict[str, Any]:
        """Return the statistics as a plain dict (JSON-serialisable except NaN)."""
        return asdict(self)


class CalibrationMonitor:
    """Sliding-window calibration monitor for a UCCI calibrator or router.

    **Extension, not in the paper.** The paper fits the calibration map once
    (Section 4.2) and names online recalibration as the natural next step
    under distribution shift (Section 7, "Static calibration"). This monitor
    tells you when that step is needed: it keeps the last ``window`` pairs of
    forecast ``p_hat = g(u(x))`` and observed error event ``e(x)`` and
    reports windowed ECE, mean forecast against observed error rate, and a
    drift flag.

    Parameters
    ----------
    forecaster : calibrator, router, callable or None
        Source of the forecasts ``p_hat``. Routers are used through
        ``error_probability(u)`` (:class:`ucci.UCCIRouter`,
        :class:`RecalibratingRouter`), calibrators through ``predict(u)``
        (:class:`ucci.IsotonicCalibrator`), anything else must be a callable
        ``u -> p_hat``. The forecaster is called at :meth:`update` time, so a
        router that changes over time is scored on the forecast it issued
        before the label arrived (prequential evaluation). Pass ``None`` to
        feed logged forecasts with :meth:`record` only.
    window : int, default 1000
        Number of most recent labels kept.
    alpha : float, default 0.01
        Two-sided significance level of the drift flag, in (0, 1).
    test : {"binomial", "spiegelhalter"}, default "binomial"
        Drift test; see :func:`binomial_z_test` and
        :func:`spiegelhalter_z_test`.
    min_count : int, default 100
        Minimum labels in the window before the drift flag can be raised.
        Must not exceed ``window``.
    n_bins : int, default 10
        Bins for the windowed ECE.
    strategy : {"uniform", "quantile"}, default "uniform"
        Binning strategy for the windowed ECE (see :func:`ucci.ece`).

    Examples
    --------
    >>> import numpy as np
    >>> from ucci import IsotonicCalibrator
    >>> from ucci.online import CalibrationMonitor
    >>> rng = np.random.default_rng(0)
    >>> u = rng.random(4000)
    >>> e = (rng.random(4000) < u).astype(float)
    >>> cal = IsotonicCalibrator().fit(u[:2000], e[:2000])
    >>> mon = CalibrationMonitor(cal, window=1000)
    >>> mon.update(u[2000:], e[2000:])
    >>> s = mon.stats()
    >>> s.count, s.drift
    (1000, False)
    """

    def __init__(
        self,
        forecaster: ForecastSource | None,
        window: int = 1000,
        *,
        alpha: float = 0.01,
        test: str = "binomial",
        min_count: int = 100,
        n_bins: int = 10,
        strategy: str = "uniform",
    ) -> None:
        self._forecast: Callable[[FloatArray], FloatArray] | None = (
            None if forecaster is None else _resolve_forecaster(forecaster)
        )
        self.forecaster = forecaster
        self.window = _check_positive_int(window, "window")
        self.alpha = _check_probability(alpha, "alpha", open_interval=True)
        if test not in _TESTS:
            raise ValueError(f"test must be one of {DRIFT_TESTS}, got {test!r}")
        self.test = test
        self.min_count = _check_positive_int(min_count, "min_count")
        if self.min_count > self.window:
            raise ValueError(f"min_count ({self.min_count}) must not exceed window ({self.window})")
        self.n_bins = _check_positive_int(n_bins, "n_bins")
        if strategy not in _BIN_STRATEGIES:
            raise ValueError(f"strategy must be one of {_BIN_STRATEGIES}, got {strategy!r}")
        self.strategy = strategy
        self._buf = _RingBuffer(self.window, 2)
        self._n_seen = 0

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def n_seen(self) -> int:
        """Labels received since construction or the last :meth:`reset`."""
        return self._n_seen

    def update(self, u: ArrayLike, e: ArrayLike) -> None:
        """Score the forecaster on new labelled queries and add them to the window.

        Parameters
        ----------
        u : float or array_like, shape (n,)
            Raw uncertainty u(x) of each query (Section 4.1, Eq. 4).
        e : float or array_like, shape (n,)
            Observed error events, 1 when the small model was wrong
            (Section 4.2). Values in [0, 1].

        Raises
        ------
        RuntimeError
            If the monitor was built with ``forecaster=None``.
        ValueError
            On mismatched lengths, non-finite values or ``e`` outside [0, 1].
        """
        if self._forecast is None:
            raise RuntimeError(
                "this monitor has no forecaster; use record(p_hat, e) to feed "
                "logged forecasts, or pass a calibrator or router"
            )
        u_arr = _as_1d_float(u, "u")
        e_arr = _as_events(e)
        _check_pair(u_arr, e_arr)
        self._push(self._forecast(u_arr), e_arr)

    def record(self, p_hat: ArrayLike, e: ArrayLike) -> None:
        """Add already computed forecasts and their outcomes to the window.

        Use this when ``p_hat`` was logged at routing time.

        Parameters
        ----------
        p_hat : float or array_like, shape (n,)
            Forecast error probabilities in [0, 1].
        e : float or array_like, shape (n,)
            Observed error events in [0, 1].
        """
        p_arr = _as_events(p_hat, "p_hat")
        e_arr = _as_events(e)
        _check_pair(p_arr, e_arr, "p_hat")
        self._push(p_arr, e_arr)

    def _push(self, p: FloatArray, e: FloatArray) -> None:
        self._buf.extend(p, e)
        self._n_seen += int(p.size)

    def window_arrays(self) -> tuple[FloatArray, FloatArray]:
        """Return copies of the windowed ``(p_hat, e)`` in arrival order."""
        p, e = self._buf.columns()
        return p, e

    def reset(self) -> None:
        """Empty the window and zero the label counter."""
        self._buf.clear()
        self._n_seen = 0

    def stats(self) -> MonitorStats:
        """Compute the windowed statistics and the drift flag.

        Returns
        -------
        MonitorStats
            See :class:`MonitorStats`. ``drift`` is True only when the window
            holds at least ``min_count`` labels and the test's two-sided
            p-value is below ``alpha``.
        """
        p, e = self._buf.columns()
        n = int(p.size)
        if n == 0:
            nan = float("nan")
            return MonitorStats(
                0,
                self._n_seen,
                self.window,
                nan,
                nan,
                0.0,
                0.0,
                nan,
                self.test,
                0.0,
                1.0,
                self.alpha,
                False,
                False,
            )
        z, p_value = _TESTS[self.test](p, e)
        ready = n >= self.min_count
        return MonitorStats(
            count=n,
            n_seen=self._n_seen,
            window=self.window,
            mean_predicted=float(p.mean()),
            observed_error_rate=float(e.mean()),
            expected_errors=float(p.sum()),
            observed_errors=float(e.sum()),
            ece=float(_ece(p, e, n_bins=self.n_bins, strategy=self.strategy)),
            test=self.test,
            z=z,
            p_value=p_value,
            alpha=self.alpha,
            ready=ready,
            drift=bool(ready and p_value < self.alpha),
        )

    def __repr__(self) -> str:
        return (
            f"CalibrationMonitor(window={self.window}, alpha={self.alpha}, "
            f"test={self.test!r}, count={len(self)}, n_seen={self._n_seen})"
        )


# ---------------------------------------------------------------------------
# RecalibratingRouter
# ---------------------------------------------------------------------------


class RecalibratingRouter:
    """Threshold router whose isotonic map is refit on a sliding window.

    **Extension, not in the paper.** The paper's router (Section 4) fits the
    isotonic map g once on a calibration batch and applies the threshold
    policy of Eq. 6, escalating when ``p_hat(x) = g(u(x)) > theta``. Section 7
    ("Static calibration") notes that streaming deployments with distribution
    shift need online or continual recalibration within the same calibrated
    threshold framework. This class does exactly that and nothing more:

    * theta stays fixed and keeps its meaning as an error-probability
      threshold, so the implied cutoff on u(x) moves as g is refit;
    * g is refit with :class:`ucci.IsotonicCalibrator` (the paper's
      calibrator) on the most recent ``window`` labels;
    * a refit happens after every ``refit_every`` new labels, and only once
      the window holds at least ``min_labels`` labels. If the window is too
      small at a scheduled refit, the refit happens at the first label that
      brings it to ``min_labels``.

    Re-selecting theta itself (Section 4.3, Eq. 7) needs large-model outcomes
    on a validation set and is out of scope here; call
    :func:`ucci.select_threshold` on fresh validation data if the accuracy
    target must be re-certified.

    Parameters
    ----------
    theta : float
        Error-probability threshold in [0, 1] (Eq. 6).
    calibrator : calibrator or None, default None
        Initial map g, used until the first refit. Anything with
        ``predict(u)``; typically a fitted :class:`ucci.IsotonicCalibrator`.
        With ``None`` the router has no map until the first refit and
        :meth:`error_probability` raises. The object is never modified.
    window : int, default 5000
        Number of most recent labels the refit uses.
    refit_every : int, default 500
        Refit after this many new labels.
    min_labels : int, default 500
        Minimum labels in the window before any refit. Must not exceed
        ``window``.

    Notes
    -----
    Labels must be representative of the traffic being routed. If only
    kept (or only escalated) queries are ever labelled, the window covers
    only part of the u(x) range and the refit map is biased there. Feeding
    labels one at a time or in batches gives identical states: refits happen
    at the same label counts either way.

    Examples
    --------
    >>> import numpy as np
    >>> from ucci import IsotonicCalibrator
    >>> from ucci.online import RecalibratingRouter
    >>> rng = np.random.default_rng(0)
    >>> u = rng.random(3000)
    >>> e = (rng.random(3000) < u).astype(float)
    >>> r = RecalibratingRouter(0.5, calibrator=IsotonicCalibrator().fit(u[:1000], e[:1000]),
    ...                         window=1000, refit_every=500, min_labels=500)
    >>> r.update(u[1000:], e[1000:])
    4
    >>> r.n_refits, r.labels_in_window
    (4, 1000)
    """

    def __init__(
        self,
        theta: float,
        *,
        calibrator: SupportsPredict | None = None,
        window: int = 5000,
        refit_every: int = 500,
        min_labels: int = 500,
    ) -> None:
        self.theta = _check_probability(theta, "theta", open_interval=False)
        self.window = _check_positive_int(window, "window")
        self.refit_every = _check_positive_int(refit_every, "refit_every")
        self.min_labels = _check_positive_int(min_labels, "min_labels")
        if self.min_labels > self.window:
            raise ValueError(
                f"min_labels ({self.min_labels}) must not exceed window ({self.window})"
            )
        if calibrator is not None and not callable(getattr(calibrator, "predict", None)):
            raise TypeError(
                f"calibrator must have a predict(u) method, got {type(calibrator).__name__}"
            )
        self._calibrator: SupportsPredict | None = calibrator
        self._buf = _RingBuffer(self.window, 2)
        self._since_refit = 0
        self._n_seen = 0
        self._refit_at: list[int] = []
        self._router_settings: dict[str, Any] = {}

    @classmethod
    def from_router(
        cls, router: Any, *, window: int = 5000, refit_every: int = 500, min_labels: int = 500
    ) -> RecalibratingRouter:
        """Start from a fitted :class:`ucci.UCCIRouter`: its map and its theta.

        Parameters
        ----------
        router : UCCIRouter
            A router with a fitted calibrator and a chosen threshold.
        window, refit_every, min_labels : int
            As in the constructor.

        Returns
        -------
        RecalibratingRouter
            A router that routes exactly like ``router`` until its first
            refit. The input router is not modified.
        """
        calibrator = getattr(router, "calibrator", None)
        if calibrator is None:
            raise TypeError("router has no calibrator attribute; pass a fitted UCCIRouter")
        try:
            theta = float(router.theta)
        except RuntimeError as exc:
            raise ValueError("router has no threshold yet; call choose_threshold() first") from exc
        out = cls(
            theta,
            calibrator=calibrator,
            window=window,
            refit_every=refit_every,
            min_labels=min_labels,
        )
        out._router_settings = {
            key: getattr(router, key)
            for key in ("c_small", "c_large", "cost_model", "grid_step")
            if hasattr(router, key)
        }
        return out

    # -- state ---------------------------------------------------------------

    @property
    def calibrator(self) -> SupportsPredict | None:
        """Current map g (None before the first refit when started without one)."""
        return self._calibrator

    @property
    def is_calibrated(self) -> bool:
        """True when a map is available for :meth:`error_probability`."""
        return self._calibrator is not None

    @property
    def n_refits(self) -> int:
        """Number of refits performed so far."""
        return len(self._refit_at)

    @property
    def refit_history(self) -> tuple[int, ...]:
        """Label counts (:attr:`n_labels_seen`) at which each refit happened."""
        return tuple(self._refit_at)

    @property
    def n_labels_seen(self) -> int:
        """Labels received since construction."""
        return self._n_seen

    @property
    def labels_in_window(self) -> int:
        """Labels currently in the sliding window."""
        return len(self._buf)

    @property
    def labels_since_refit(self) -> int:
        """Labels received since the last refit (or since construction)."""
        return self._since_refit

    def window_arrays(self) -> tuple[FloatArray, FloatArray]:
        """Return copies of the windowed ``(u, e)`` in arrival order."""
        u, e = self._buf.columns()
        return u, e

    # -- routing -------------------------------------------------------------

    def _p_hat(self, u: ArrayLike) -> FloatArray:
        if self._calibrator is None:
            raise RuntimeError(
                f"no calibration map yet: {len(self._buf)} labels in the window, "
                f"the first refit needs {self.min_labels} (min_labels); pass an "
                "initial calibrator to route before then"
            )
        u_arr = np.asarray(u, dtype=np.float64)
        p = np.asarray(self._calibrator.predict(u_arr), dtype=np.float64)
        return p.reshape(u_arr.shape)

    @overload
    def error_probability(self, u: float) -> float: ...

    @overload
    def error_probability(self, u: ArrayLike) -> FloatArray: ...

    def error_probability(self, u: ArrayLike) -> float | FloatArray:
        """Calibrated forecast ``p_hat = g(u)`` from the current map.

        Parameters
        ----------
        u : float or array_like
            Raw uncertainty u(x) (Section 4.1, Eq. 4), any shape.

        Returns
        -------
        float or numpy.ndarray of float64
            A float for a scalar input, otherwise an array of the input's
            shape, as :meth:`ucci.UCCIRouter.error_probability`.

        Raises
        ------
        RuntimeError
            Before the first refit when the router was built without a
            calibrator.
        """
        p = self._p_hat(u)
        if p.ndim == 0 and not isinstance(u, np.ndarray):
            return float(p)
        return p

    @overload
    def escalate(self, u: float) -> bool: ...

    @overload
    def escalate(self, u: ArrayLike) -> BoolArray: ...

    def escalate(self, u: ArrayLike) -> bool | BoolArray:
        """Eq. 6 with the current map: True where ``g(u) > theta`` (send to large).

        A bool for a scalar input, otherwise a mask of the input's shape.
        """
        mask = np.asarray(self._p_hat(u) > self.theta, dtype=bool)
        if mask.ndim == 0 and not isinstance(u, np.ndarray):
            return bool(mask)
        return mask

    def route(self, u: ArrayLike) -> RouteResult:
        """Routing decisions and forecasts for a batch, as :meth:`ucci.UCCIRouter.route`.

        Parameters
        ----------
        u : array_like of float
            u(x) of each query, any shape.

        Returns
        -------
        ucci.RouteResult
            ``(escalate, p_hat)``, arrays of the input's shape.
        """
        p_hat = self._p_hat(u)
        return RouteResult(escalate=np.asarray(p_hat > self.theta, dtype=bool), p_hat=p_hat)

    def to_router(
        self,
        c_small: float | None = None,
        c_large: float | None = None,
        cost_model: str | None = None,
    ) -> UCCIRouter:
        """Snapshot the current map and theta as a :class:`ucci.UCCIRouter`.

        The snapshot can be saved in the shared router JSON format
        (:meth:`ucci.UCCIRouter.save`) and served by any reader of that
        format, including the Rust crate. It has no ``tau``: a refit map no
        longer certifies the accuracy target theta was chosen for.

        Parameters
        ----------
        c_small, c_large, cost_model : optional
            Cost settings of the snapshot. Default to those of the router
            given to :meth:`from_router`, else to the paper's (1.0, 3.02,
            ``"routing"``).

        Returns
        -------
        UCCIRouter
            An independent copy; later refits do not change it.

        Raises
        ------
        RuntimeError
            If there is no map yet.
        TypeError
            If the current map is not an :class:`ucci.IsotonicCalibrator`
            (only the initial calibrator can be another type).
        """
        cal = self._calibrator
        if cal is None:
            raise RuntimeError("no calibration map yet; nothing to snapshot")
        if not isinstance(cal, IsotonicCalibrator):
            raise TypeError(
                "only an IsotonicCalibrator map can be written to the router format; "
                f"the current map is a {type(cal).__name__}"
            )
        settings = dict(self._router_settings)
        if c_small is not None:
            settings["c_small"] = c_small
        if c_large is not None:
            settings["c_large"] = c_large
        if cost_model is not None:
            settings["cost_model"] = cost_model
        router = UCCIRouter(**settings)
        router.calibrator = IsotonicCalibrator.from_dict(cal.to_dict())
        router.theta = self.theta
        return router

    # -- learning ------------------------------------------------------------

    def update(self, u: ArrayLike, e: ArrayLike) -> int:
        """Add labelled queries to the window and refit on schedule.

        Parameters
        ----------
        u : float or array_like, shape (n,)
            Raw uncertainty u(x) of each labelled query.
        e : float or array_like, shape (n,)
            Error events in [0, 1], 1 when the small model was wrong
            (Section 4.2).

        Returns
        -------
        int
            Number of refits this call performed (0 or more; a large batch can
            cross several refit points, and each is performed in order so the
            result matches feeding the labels one at a time).
        """
        u_arr = _as_1d_float(u, "u")
        e_arr = _as_events(e)
        _check_pair(u_arr, e_arr)
        refits = 0
        pos = 0
        n = int(u_arr.size)
        while pos < n:
            # Labels until the next point where a refit becomes due.
            need_new = max(1, self.refit_every - self._since_refit)
            need_min = max(1, self.min_labels - len(self._buf))
            step = min(n - pos, max(need_new, need_min))
            self._buf.extend(u_arr[pos : pos + step], e_arr[pos : pos + step])
            self._since_refit += step
            self._n_seen += step
            pos += step
            if self._since_refit >= self.refit_every and len(self._buf) >= self.min_labels:
                self._fit_window()
                refits += 1
        return refits

    def refit(self) -> None:
        """Refit the map on the current window now, regardless of the schedule.

        Raises
        ------
        ValueError
            If the window holds fewer than ``min_labels`` labels.
        """
        if len(self._buf) < self.min_labels:
            raise ValueError(
                f"cannot refit: {len(self._buf)} labels in the window, "
                f"min_labels is {self.min_labels}"
            )
        self._fit_window()

    def _fit_window(self) -> None:
        u, e = self._buf.columns()
        self._calibrator = IsotonicCalibrator().fit(u, e)
        self._since_refit = 0
        self._refit_at.append(self._n_seen)

    def __repr__(self) -> str:
        return (
            f"RecalibratingRouter(theta={self.theta}, window={self.window}, "
            f"refit_every={self.refit_every}, min_labels={self.min_labels}, "
            f"n_refits={self.n_refits}, labels_in_window={self.labels_in_window})"
        )
