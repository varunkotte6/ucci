"""Isotonic calibration of u(x) into an error probability (paper Section 4.2).

The calibrator learns a non-decreasing map g (Eq. 5) with

    g(u) ~= P(e(x) = 1 | u(x) = u),

where e(x) = 1 when the small model's output is wrong, on a held-out
calibration set C = {(u_i, e_i)}. The calibrated forecast used by the policy
is p_hat(x) = g(u(x)) (Section 4.3).

The paper fits g "by isotonic regression (Zadrozny & Elkan, 2002)" using
"standard open-source libraries (default settings)" (Appendix B.2); the
library was scikit-learn's ``IsotonicRegression``. This module reproduces that
estimator with numpy only, so it has no scikit-learn dependency and ports
directly to other languages:

1. points with zero sample weight are dropped;
2. points are sorted by u, and u values closer than ``1e-15`` to the first
   value of their group are pooled into one point whose label is the
   weighted mean (scikit-learn's ``_make_unique`` rule);
3. the pooled labels are fit by weighted pool-adjacent-violators (PAV),
   the least-squares non-decreasing fit (Barlow et al., 1972);
4. interior knots whose value equals both neighbours are dropped, which does
   not change the fitted function (scikit-learn's ``trim_duplicates``).

Predictions interpolate linearly between knots. Outside the calibration range
they are clipped to the end values, as with
``IsotonicRegression(out_of_bounds="clip")``. scikit-learn's default,
``out_of_bounds="nan"``, returns NaN there instead; clipping keeps p_hat
defined for every query and is the one deliberate difference (recorded in
``docs/paper_mapping.md``). Inside the range the predictions equal
scikit-learn's up to floating-point round-off, and the tests check this.

The fit costs O(n log n) for the sort plus O(n) for PAV.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from ._validation import (
    as_1d,
    as_float_array,
    check_labels,
    check_same_length,
    check_weights,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = ["IsotonicCalibrator", "pav"]

#: u values closer than this to the first value of their group are pooled,
#: as in scikit-learn (``np.finfo(np.float64).resolution``).
TIE_TOLERANCE = float(np.finfo(np.float64).resolution)


def _pav_blocks(y: NDArray[np.float64], w: NDArray[np.float64]) -> tuple[list[float], list[int]]:
    """Weighted PAV on validated inputs; returns block means and block sizes.

    Blocks are kept on a stack as (weight sum, weighted label sum). A new
    point is pooled with the block below it while that block's mean is
    strictly larger, so already monotone input passes through unchanged.
    Each point is pushed and popped at most once: O(n).
    """
    sum_w: list[float] = []
    sum_wy: list[float] = []
    means: list[float] = []
    sizes: list[int] = []
    for yi, wi in zip(y.tolist(), w.tolist()):
        s_w, s_wy, size, mean = wi, wi * yi, 1, yi
        while means and means[-1] > mean:
            s_w += sum_w.pop()
            s_wy += sum_wy.pop()
            size += sizes.pop()
            means.pop()
            mean = s_wy / s_w
        sum_w.append(s_w)
        sum_wy.append(s_wy)
        sizes.append(size)
        means.append(mean)
    return means, sizes


def pav(y: ArrayLike, w: ArrayLike | None = None) -> NDArray[np.float64]:
    """Weighted least-squares non-decreasing fit (pool-adjacent-violators).

    Solves ``min_f sum_i w_i (y_i - f_i)^2`` subject to
    ``f_1 <= f_2 <= ... <= f_n``, for ``y`` already ordered by the covariate.
    This is the isotonic regression step of Section 4.2.

    Parameters
    ----------
    y : array_like of float, shape (n,)
        Targets, ordered by the covariate (for UCCI: error labels sorted by
        u(x)).
    w : array_like of float, shape (n,), optional
        Strictly positive weights. Defaults to all ones.

    Returns
    -------
    numpy.ndarray of float64, shape (n,)
        The fitted non-decreasing values. Each maximal run of equal values is
        the weighted mean of the targets it covers.

    Raises
    ------
    ValueError
        If ``y`` is empty or non-finite, or ``w`` has the wrong length or a
        non-positive or non-finite entry.

    Examples
    --------
    >>> pav([1.0, 3.0, 2.0, 4.0]).tolist()
    [1.0, 2.5, 2.5, 4.0]
    """
    yy = as_1d(y, "y")
    ww = check_weights(w, yy.size, "w")
    if (ww <= 0.0).any():
        raise ValueError("w must be strictly positive; drop zero-weight points before calling pav")
    means, sizes = _pav_blocks(yy, ww)
    return np.repeat(np.asarray(means, dtype=np.float64), np.asarray(sizes, dtype=np.int64))


def _pool_ties(
    x: NDArray[np.float64], y: NDArray[np.float64], w: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Pool sorted ``x`` values within ``TIE_TOLERANCE`` of their group start.

    Returns the group representatives (the first x of each group), the
    weighted mean label and the total weight per group.
    """
    starts = np.flatnonzero(np.concatenate(([True], x[1:] != x[:-1])))
    xs = x[starts]
    if xs.size > 1 and float(np.min(np.diff(xs))) < TIE_TOLERANCE:
        # Rare near-ties: apply scikit-learn's greedy grouping rule over the
        # distinct values (a new group starts once x - group_start >= tol).
        keep = [0]
        group_start = float(xs[0])
        for j, value in enumerate(xs.tolist()[1:], start=1):
            if value - group_start >= TIE_TOLERANCE:
                keep.append(j)
                group_start = value
        starts = starts[np.asarray(keep, dtype=np.int64)]
        xs = x[starts]
    w_sum = np.add.reduceat(w, starts)
    y_mean = np.add.reduceat(y * w, starts) / w_sum
    return xs, y_mean, w_sum


def _check_knots(x: Any, y: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate calibration knots: x strictly increasing, y non-decreasing in [0, 1]."""
    xs = as_1d(x, "x")
    ys = as_1d(y, "y")
    check_same_length(x=xs, y=ys)
    if xs.size > 1 and not bool(np.all(np.diff(xs) > 0.0)):
        i = int(np.argmax(np.diff(xs) <= 0.0))
        raise ValueError(
            f"x must be strictly increasing; x[{i}] = {float(xs[i])!r} >= "
            f"x[{i + 1}] = {float(xs[i + 1])!r}"
        )
    if ys.size > 1 and not bool(np.all(np.diff(ys) >= 0.0)):
        i = int(np.argmax(np.diff(ys) < 0.0))
        raise ValueError(
            f"y must be non-decreasing; y[{i}] = {float(ys[i])!r} > "
            f"y[{i + 1}] = {float(ys[i + 1])!r}"
        )
    check_labels(ys, "y")
    return xs, ys


class IsotonicCalibrator:
    """Monotone map g from uncertainty u(x) to P(small model wrong) (Section 4.2).

    Attributes
    ----------
    x_ : numpy.ndarray of float64 or None
        Knot positions (u values), strictly increasing. ``x_[0]`` and
        ``x_[-1]`` bound the calibration range; predictions are clipped
        outside it. Same as scikit-learn's ``X_thresholds_``.
    y_ : numpy.ndarray of float64 or None
        Fitted error probabilities at the knots, non-decreasing, in [0, 1].
        Same as scikit-learn's ``y_thresholds_``.
    n_samples_ : int or None
        Number of calibration points with positive weight used by the last
        :meth:`fit`; None when the knots came from :meth:`from_dict`.

    Examples
    --------
    >>> cal = IsotonicCalibrator().fit([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
    >>> cal.x_.tolist(), cal.y_.tolist()
    ([0.1, 0.2, 0.3, 0.4], [0.0, 0.5, 0.5, 1.0])
    >>> cal.predict(0.25)
    0.5
    >>> cal.predict([0.0, 1.0]).tolist()
    [0.0, 1.0]
    """

    def __init__(self) -> None:
        self.x_: NDArray[np.float64] | None = None
        self.y_: NDArray[np.float64] | None = None
        self.n_samples_: int | None = None

    @property
    def is_fitted(self) -> bool:
        """True once :meth:`fit` or :meth:`from_dict` has set the knots."""
        return self.x_ is not None and self.y_ is not None

    def fit(
        self, u: ArrayLike, e: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> IsotonicCalibrator:
        """Fit g on calibration pairs ``(u_i, e_i)`` (Section 4.2, Eq. 5).

        Parameters
        ----------
        u : array_like of float, shape (n,)
            Uncertainty u(x) of each calibration query (Eq. 4).
        e : array_like of float, shape (n,)
            Error label: 1 if the small model's output was wrong, 0 if right.
            Soft labels in [0, 1] are accepted.
        sample_weight : array_like of float, shape (n,), optional
            Non-negative weights; zero-weight points are dropped. Defaults
            to all ones.

        Returns
        -------
        IsotonicCalibrator
            ``self``, fitted.

        Raises
        ------
        ValueError
            If the inputs are empty, differ in length, contain NaN or inf,
            ``e`` leaves [0, 1], or the weights are negative or sum to zero.
        """
        uu = as_1d(u, "u")
        ee = check_labels(as_1d(e, "e"), "e")
        n = check_same_length(u=uu, e=ee)
        ww = check_weights(sample_weight, n)
        positive = ww > 0.0
        if not bool(positive.all()):
            uu, ee, ww = uu[positive], ee[positive], ww[positive]

        order = np.lexsort((ee, uu))
        uu, ee, ww = uu[order], ee[order], ww[order]
        xs, y_mean, w_sum = _pool_ties(uu, ee, ww)
        means, sizes = _pav_blocks(y_mean, w_sum)
        fitted = np.repeat(np.asarray(means, dtype=np.float64), np.asarray(sizes, dtype=np.int64))

        keep = np.ones(fitted.size, dtype=bool)
        if fitted.size > 2:
            keep[1:-1] = (fitted[1:-1] != fitted[:-2]) | (fitted[1:-1] != fitted[2:])
        self.x_ = np.ascontiguousarray(xs[keep])
        self.y_ = np.ascontiguousarray(fitted[keep])
        self.n_samples_ = int(uu.size)
        return self

    def _knots(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self.x_ is None or self.y_ is None:
            raise RuntimeError("IsotonicCalibrator is not fitted; call fit(u, e) first")
        return self.x_, self.y_

    @overload
    def predict(self, u: float) -> float: ...

    @overload
    def predict(self, u: ArrayLike) -> NDArray[np.float64]: ...

    def predict(self, u: ArrayLike) -> float | NDArray[np.float64]:
        """Calibrated error probability p_hat = g(u) (Section 4.3).

        Parameters
        ----------
        u : float or array_like of float
            Uncertainty values, any shape.

        Returns
        -------
        float or numpy.ndarray of float64
            A float for a scalar input, otherwise an array of the input's
            shape. Values lie in [0, 1].

        Raises
        ------
        RuntimeError
            If the calibrator is not fitted.
        ValueError
            If ``u`` contains NaN or inf.
        """
        x, y = self._knots()
        arr = as_float_array(u, "u")
        out = np.interp(arr, x, y)
        if arr.ndim == 0 and not isinstance(u, np.ndarray):
            return float(out)
        return np.asarray(out, dtype=np.float64)

    __call__ = predict

    def to_dict(self) -> dict[str, list[float]]:
        """Knots as ``{"x": [...], "y": [...]}`` (the router JSON ``calibrator`` field).

        Floats are written with full precision, so :meth:`from_dict` restores
        the exact same function.
        """
        x, y = self._knots()
        return {"x": [float(v) for v in x], "y": [float(v) for v in y]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IsotonicCalibrator:
        """Rebuild a calibrator from :meth:`to_dict` output.

        Unknown keys are ignored.

        Raises
        ------
        ValueError
            If ``x`` or ``y`` is missing, the lengths differ or are zero,
            ``x`` is not strictly increasing, or ``y`` is not non-decreasing
            within [0, 1].
        """
        if not isinstance(data, Mapping):
            raise ValueError(f"calibrator must be a mapping, got {type(data).__name__}")
        missing = [k for k in ("x", "y") if k not in data]
        if missing:
            raise ValueError(f"calibrator is missing {', '.join(missing)}")
        x, y = _check_knots(data["x"], data["y"])
        cal = cls()
        cal.x_, cal.y_ = x.copy(), y.copy()
        return cal

    def __repr__(self) -> str:
        if self.x_ is None or self.y_ is None:
            return "IsotonicCalibrator(unfitted)"
        return (
            f"IsotonicCalibrator(n_knots={self.x_.size}, "
            f"u_range=[{self.x_[0]:.6g}, {self.x_[-1]:.6g}], "
            f"p_range=[{self.y_[0]:.6g}, {self.y_[-1]:.6g}])"
        )
