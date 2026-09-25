"""Forecast-quality and evaluation metrics.

* :func:`ece` and :func:`reliability_table`: expected calibration error and
  the reliability diagram of a probability forecast (Section 6.2, Figure 1;
  the paper reports ECE 0.12 for raw u(x) and 0.03 after isotonic
  calibration on its workload).
* :func:`brier_score`: mean squared error of a probability forecast.
* :func:`bootstrap_ci`: percentile bootstrap confidence intervals over
  queries (Section 6.2; 1000 resamples by default).
* :func:`micro_f1` and :func:`routed_micro_f1`: the paper's evaluation
  metric, micro-averaged F1 over entities (Sections 3 and 4.2), for a single
  model and for the answers a routing policy returns.

For UCCI the forecast ``p`` is p_hat(x), the calibrated probability that the
small model is wrong, and the event ``y`` is e(x) (1 when it is wrong). The
paper does not state its ECE binning; the default here is 10 equal-width
bins, and ``strategy="quantile"`` gives equal-count bins (deciles with
``n_bins=10``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ._validation import (
    as_1d,
    as_float_array,
    check_finite_scalar,
    check_labels,
    check_positive_int,
    check_probabilities,
    check_same_length,
    check_weights,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ConfidenceInterval",
    "ReliabilityRow",
    "RoutedMicroF1",
    "bootstrap_ci",
    "brier_score",
    "ece",
    "micro_f1",
    "reliability_table",
    "routed_micro_f1",
]

_STRATEGIES = ("uniform", "quantile")


class ReliabilityRow(NamedTuple):
    """One non-empty bin of a reliability diagram.

    Attributes
    ----------
    bin_lower, bin_upper : float
        Bin edges. Bins are right-closed, ``(bin_lower, bin_upper]``, except
        the first, which also contains ``bin_lower``.
    count : int
        Number of forecasts in the bin (points with zero weight excluded).
    mean_forecast : float
        (Weighted) mean forecast probability in the bin.
    observed_frequency : float
        (Weighted) frequency of the event in the bin.
    """

    bin_lower: float
    bin_upper: float
    count: int  # type: ignore[assignment]  # shadows tuple.count by design
    mean_forecast: float
    observed_frequency: float


class ConfidenceInterval(NamedTuple):
    """A two-sided confidence interval; unpacks as ``(low, high)``."""

    low: float
    high: float


def _forecast_inputs(
    p: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validated forecasts in [0, 1], outcomes in [0, 1] and positive weights."""
    pp = check_probabilities(as_1d(p, "p"), "p", atol=0.0)
    yy = check_labels(as_1d(y, "y"), "y")
    n = check_same_length(p=pp, y=yy)
    w = check_weights(sample_weight, n)
    keep = w > 0.0
    if not bool(keep.all()):
        pp, yy, w = pp[keep], yy[keep], w[keep]
    return pp, yy, w


def _bin_edges(p: NDArray[np.float64], n_bins: int, strategy: str) -> NDArray[np.float64]:
    """Bin edges for a strategy already checked by the caller."""
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    return np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))


def _binned(
    p: ArrayLike,
    y: ArrayLike,
    n_bins: int,
    strategy: str,
    sample_weight: ArrayLike | None,
) -> tuple[list[ReliabilityRow], NDArray[np.float64], float]:
    """Rows of the reliability table plus per-row weights and the total weight."""
    nb = check_positive_int(n_bins, "n_bins")
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy must be 'uniform' or 'quantile', got {strategy!r}")
    pp, yy, w = _forecast_inputs(p, y, sample_weight)
    edges = _bin_edges(pp, nb, strategy)
    n_edges = edges.size
    if n_edges == 1:  # quantile bins of a constant forecast
        ids = np.zeros(pp.size, dtype=np.int64)
        lowers, uppers = edges, edges
    else:
        # Right-closed bins; a forecast equal to edges[0] lands in the first bin.
        ids = np.clip(np.searchsorted(edges, pp, side="left") - 1, 0, n_edges - 2)
        lowers, uppers = edges[:-1], edges[1:]
    n_groups = lowers.size
    counts = np.bincount(ids, minlength=n_groups)
    w_bin = np.bincount(ids, weights=w, minlength=n_groups)
    wp_bin = np.bincount(ids, weights=w * pp, minlength=n_groups)
    wy_bin = np.bincount(ids, weights=w * yy, minlength=n_groups)
    rows: list[ReliabilityRow] = []
    weights: list[float] = []
    for b in np.flatnonzero(counts):
        rows.append(
            ReliabilityRow(
                bin_lower=float(lowers[b]),
                bin_upper=float(uppers[b]),
                count=int(counts[b]),
                mean_forecast=float(wp_bin[b] / w_bin[b]),
                observed_frequency=float(wy_bin[b] / w_bin[b]),
            )
        )
        weights.append(float(w_bin[b]))
    return rows, np.asarray(weights, dtype=np.float64), float(w.sum())


def reliability_table(
    p: ArrayLike,
    y: ArrayLike,
    n_bins: int = 10,
    strategy: str = "uniform",
    sample_weight: ArrayLike | None = None,
) -> list[ReliabilityRow]:
    """Reliability diagram rows: per non-empty bin, forecast vs observed frequency.

    Parameters
    ----------
    p : array_like of float, shape (n,)
        Forecast probabilities of the event ``y = 1`` (for UCCI: p_hat, the
        probability that the small model is wrong).
    y : array_like of float, shape (n,)
        Outcomes in [0, 1] (for UCCI: e(x)).
    n_bins : int, default 10
        Number of bins.
    strategy : {"uniform", "quantile"}, default "uniform"
        Equal-width bins on [0, 1], or equal-count bins at the quantiles of
        ``p`` (duplicate edges from repeated forecasts are merged, so there
        can be fewer bins). Quantile edges ignore the weights.
    sample_weight : array_like of float, shape (n,), optional
        Non-negative weights; zero-weight points are dropped.

    Returns
    -------
    list of ReliabilityRow
        Named tuples ``(bin_lower, bin_upper, count, mean_forecast,
        observed_frequency)``, one per non-empty bin, in increasing order.

    Raises
    ------
    ValueError
        On empty or mismatched inputs, forecasts or outcomes outside
        [0, 1], invalid weights, ``n_bins < 1`` or an unknown strategy.

    Examples
    --------
    >>> rows = reliability_table([0.05, 0.15, 0.15, 0.95], [0, 0, 1, 1])
    >>> [(r.count, r.observed_frequency) for r in rows]
    [(1, 0.0), (2, 0.5), (1, 1.0)]
    """
    rows, _, _ = _binned(p, y, n_bins, strategy, sample_weight)
    return rows


def ece(
    p: ArrayLike,
    y: ArrayLike,
    n_bins: int = 10,
    strategy: str = "uniform",
    sample_weight: ArrayLike | None = None,
) -> float:
    """Expected calibration error (Naeini et al., 2015; Guo et al., 2017).

    ``ECE = sum_b (W_b / W) * |mean_forecast_b - observed_frequency_b|``,
    where ``W_b`` is the total weight in bin b (the count when unweighted).

    Parameters
    ----------
    p, y, n_bins, strategy, sample_weight
        As in :func:`reliability_table`.

    Returns
    -------
    float
        ECE in [0, 1].

    Examples
    --------
    >>> ece([0.25, 0.25, 0.25, 0.25], [1, 0, 0, 0])
    0.0
    """
    rows, weights, total = _binned(p, y, n_bins, strategy, sample_weight)
    gaps = np.array([abs(r.mean_forecast - r.observed_frequency) for r in rows])
    return float(np.sum(weights / total * gaps))


def brier_score(p: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
    """Brier score: the (weighted) mean of ``(p - y)^2``.

    Parameters
    ----------
    p : array_like of float, shape (n,)
        Forecast probabilities in [0, 1].
    y : array_like of float, shape (n,)
        Outcomes in [0, 1].
    sample_weight : array_like of float, shape (n,), optional
        Non-negative weights.

    Returns
    -------
    float
        Brier score in [0, 1]; lower is better.
    """
    pp, yy, w = _forecast_inputs(p, y, sample_weight)
    return float(np.sum(w * (pp - yy) ** 2) / np.sum(w))


def bootstrap_ci(
    stat: Callable[[NDArray[np.int64]], float],
    n: int,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap confidence interval over queries (Section 6.2).

    Draws ``n_boot`` resamples of ``n`` query indices with replacement,
    evaluates ``stat`` on each and returns the ``alpha / 2`` and
    ``1 - alpha / 2`` quantiles (linear interpolation, numpy's default).
    The paper reports 95% intervals from a bootstrap over queries
    (Section 6.2). The default of 1000 resamples is the count behind those
    intervals; arXiv v1 does not print it (see ``docs/paper_mapping.md``).

    Parameters
    ----------
    stat : callable
        ``stat(idx) -> float`` computes the statistic on the resampled query
        indices ``idx`` (an int array of length ``n``), for example the cost
        saving of the routed test queries ``idx``.
    n : int
        Number of queries.
    n_boot : int, default 1000
        Number of resamples.
    alpha : float, default 0.05
        One minus the coverage.
    seed : int or None, default 0
        Seed of ``numpy.random.default_rng``; the same seed gives the same
        interval. None draws fresh entropy.

    Returns
    -------
    ConfidenceInterval
        ``(low, high)``.

    Raises
    ------
    ValueError
        If ``n`` or ``n_boot`` is below 1, ``alpha`` is not in (0, 1), or
        ``stat`` returns a non-finite value.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(100.0)
    >>> lo, hi = bootstrap_ci(lambda idx: x[idx].mean(), 100, n_boot=200)
    >>> bool(lo < 49.5 < hi)
    True
    """
    nn = check_positive_int(n, "n")
    nb = check_positive_int(n_boot, "n_boot")
    a = check_finite_scalar(alpha, "alpha")
    if not 0.0 < a < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {a!r}")
    rng = np.random.default_rng(seed)
    values = np.empty(nb, dtype=np.float64)
    for i in range(nb):
        v = float(stat(rng.integers(0, nn, nn)))
        if not np.isfinite(v):
            raise ValueError(f"stat returned {v!r} on bootstrap resample {i}")
        values[i] = v
    low, high = np.quantile(values, [a / 2.0, 1.0 - a / 2.0])
    return ConfidenceInterval(float(low), float(high))


def micro_f1(tp: ArrayLike, fp: ArrayLike, fn: ArrayLike, zero_division: float = 0.0) -> float:
    """Micro-averaged F1 from true-positive, false-positive and false-negative counts.

    ``F1 = 2 TP / (2 TP + FP + FN)`` with ``TP``, ``FP`` and ``FN`` summed
    over all queries (and entity types). This is the paper's evaluation
    metric (Section 3 defines Acc as micro-averaged F1; Section 4.2).

    Parameters
    ----------
    tp, fp, fn : int or array_like of int
        Counts, per query or already summed. Must be non-negative.
    zero_division : float, default 0.0
        Value returned when ``2 TP + FP + FN = 0`` (no gold and no predicted
        entities); 0.0 is the value scikit-learn's ``f1_score`` returns by
        default in that case.

    Returns
    -------
    float
        Micro-F1 in [0, 1].

    Examples
    --------
    >>> micro_f1([2, 1], [0, 1], [1, 0])
    0.75
    """
    t = float(np.sum(_counts(tp, "tp")))
    f_p = float(np.sum(_counts(fp, "fp")))
    f_n = float(np.sum(_counts(fn, "fn")))
    denom = 2.0 * t + f_p + f_n
    if denom == 0.0:
        return check_finite_scalar(zero_division, "zero_division")
    return 2.0 * t / denom


def _counts(x: ArrayLike, name: str) -> NDArray[np.float64]:
    """Finite, non-negative counts of any shape."""
    arr = as_float_array(x, name)
    if (arr < 0.0).any():
        raise ValueError(f"{name} counts must be non-negative")
    return arr


class RoutedMicroF1:
    """Micro-F1 of the answers a routing mask returns (a ``metric`` callable).

    Create it with :func:`routed_micro_f1`. Calling it with an escalation
    mask sums the small model's ``(tp, fp, fn)`` over kept queries and the
    large model's over escalated ones, and returns :func:`micro_f1` of the
    totals.

    Attributes
    ----------
    small, large : numpy.ndarray of float64, shape (n, 3)
        Per-query ``(tp, fp, fn)`` counts of each model.
    zero_division : float
        See :func:`micro_f1`.
    """

    def __init__(
        self,
        small_counts: ArrayLike,
        large_counts: ArrayLike,
        zero_division: float = 0.0,
    ) -> None:
        self.small = self._check(small_counts, "small_counts")
        self.large = self._check(large_counts, "large_counts")
        if self.small.shape != self.large.shape:
            raise ValueError(
                f"small_counts has {self.small.shape[0]} rows but large_counts has "
                f"{self.large.shape[0]}"
            )
        self.zero_division = check_finite_scalar(zero_division, "zero_division")

    @staticmethod
    def _check(counts: ArrayLike, name: str) -> NDArray[np.float64]:
        arr = as_float_array(counts, name)
        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
            raise ValueError(f"{name} must have shape (n, 3) of (tp, fp, fn), got {arr.shape}")
        if (arr < 0.0).any():
            raise ValueError(f"{name} counts must be non-negative")
        return arr

    def __call__(self, esc: ArrayLike) -> float:
        mask = np.asarray(esc, dtype=bool)
        if mask.shape != (self.small.shape[0],):
            raise ValueError(f"mask has shape {mask.shape}, expected ({self.small.shape[0]},)")
        totals = self.small[~mask].sum(axis=0) + self.large[mask].sum(axis=0)
        return micro_f1(totals[0], totals[1], totals[2], self.zero_division)

    def __repr__(self) -> str:
        return f"RoutedMicroF1(n={self.small.shape[0]})"


def routed_micro_f1(
    small_counts: ArrayLike, large_counts: ArrayLike, zero_division: float = 0.0
) -> RoutedMicroF1:
    """Metric for threshold selection on micro-F1 of the routed answers.

    Pass the result as ``metric=`` to :func:`ucci.select_threshold`,
    :func:`ucci.select_threshold_for_budget`, :func:`ucci.evaluate` or the
    router methods, to select and evaluate thresholds on micro-F1 over
    entities, the Acc of Eq. 7 in the paper (Section 3), instead of on a mean
    per-query score.

    Parameters
    ----------
    small_counts, large_counts : array_like, shape (n, 3)
        Per-query ``(tp, fp, fn)`` entity counts of the small and the large
        model's output against the gold labels.
    zero_division : float, default 0.0
        See :func:`micro_f1`.

    Returns
    -------
    RoutedMicroF1
        Callable ``metric(esc_mask) -> float``.

    Raises
    ------
    ValueError
        If the count arrays are not of shape (n, 3), differ in length, or
        hold negative or non-finite values.

    Examples
    --------
    >>> small = [[1, 0, 1], [2, 0, 0]]
    >>> large = [[2, 0, 0], [2, 0, 0]]
    >>> f1 = routed_micro_f1(small, large)
    >>> round(f1([False, False]), 6), f1([True, False])
    (0.857143, 1.0)
    """
    return RoutedMicroF1(small_counts, large_counts, zero_division)
