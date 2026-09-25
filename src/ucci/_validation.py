"""Input checks shared by the UCCI core modules.

Every public function in :mod:`ucci` validates its inputs through this module,
so the rules and the error messages are the same everywhere:

* arrays must be numeric, finite (no NaN or inf) and, where a sequence is
  expected, one-dimensional (a single-column 2-D array is accepted and
  flattened, as in scikit-learn);
* paired arrays must have the same length and at least one element;
* probabilities must lie in ``[0, 1]``, with ``1 + PROB_ATOL`` accepted on the
  upper side because ``exp(logprob)`` can round slightly above one;
* error labels and grid values must lie in ``[0, 1]``;
* sample weights must be finite, non-negative and have a positive total;
* costs must be finite and strictly positive;
* scalars such as ``theta``, ``tau`` and ``budget`` must be finite numbers.

Violations raise :class:`ValueError` naming the argument and the offending
value, so a bad record in a large log is easy to find.

This module is private: its names may change between releases.
"""

from __future__ import annotations

import math
import numbers
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "COST_MODELS",
    "PROB_ATOL",
    "as_1d",
    "as_float_array",
    "check_cost_model",
    "check_costs",
    "check_finite_scalar",
    "check_grid",
    "check_labels",
    "check_positive_int",
    "check_probabilities",
    "check_same_length",
    "check_weights",
]

#: Tolerance above 1 accepted for probabilities computed as ``exp(logprob)``.
PROB_ATOL = 1e-9

#: The two cost models of :func:`ucci.policy.policy_cost`.
COST_MODELS = ("routing", "sequential")


def _first_bad(arr: NDArray[Any], bad: NDArray[np.bool_]) -> str:
    """Describe the first flagged element of ``arr`` for an error message."""
    if arr.ndim == 0:
        return f"value {arr.item()!r}"
    idx = tuple(int(i) for i in np.unravel_index(int(np.argmax(bad)), bad.shape))
    where = idx[0] if len(idx) == 1 else idx
    return f"index {where} (value {arr[idx].item()!r})"


def as_float_array(x: ArrayLike, name: str, *, finite: bool = True) -> NDArray[np.float64]:
    """Convert ``x`` to a float64 array of any shape.

    Parameters
    ----------
    x : array_like
        Numbers (bool, int or float). Strings, complex numbers and ragged
        nested sequences are rejected.
    name : str
        Argument name used in error messages.
    finite : bool, default True
        Reject NaN and infinite values.

    Returns
    -------
    numpy.ndarray
        float64 array. It may share memory with ``x``; callers never modify it.

    Raises
    ------
    ValueError
        If ``x`` is not numeric or, with ``finite=True``, has NaN or inf.
    """
    try:
        raw = np.asarray(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array-like: {exc}") from exc
    kind = raw.dtype.kind
    if kind in "biuf":
        arr = raw.astype(np.float64, copy=False)
    elif kind == "O":
        if any(v is None for v in raw.flat):
            raise ValueError(f"{name} must contain only numbers (found None)")
        try:
            arr = raw.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must contain only numbers (found a non-numeric or ragged entry): {exc}"
            ) from exc
    else:
        raise ValueError(f"{name} must be numeric, got an array of dtype {raw.dtype}")
    if finite:
        bad = ~np.isfinite(arr)
        if bad.any():
            raise ValueError(f"{name} contains NaN or inf at {_first_bad(arr, bad)}")
    return arr


def as_1d(
    x: ArrayLike, name: str, *, allow_empty: bool = False, finite: bool = True
) -> NDArray[np.float64]:
    """Convert ``x`` to a one-dimensional float64 array.

    A single-column 2-D array of shape ``(n, 1)`` is flattened, as
    scikit-learn does for ``IsotonicRegression``.

    Parameters
    ----------
    x : array_like
        Sequence of numbers.
    name : str
        Argument name used in error messages.
    allow_empty : bool, default False
        Accept a zero-length sequence.
    finite : bool, default True
        Reject NaN and infinite values.

    Returns
    -------
    numpy.ndarray
        float64 array of shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``x`` is not a numeric 1-D sequence, is empty (unless
        ``allow_empty``), or contains NaN or inf (unless ``finite=False``).
    """
    arr = as_float_array(x, name, finite=finite)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        shape = "a scalar" if arr.ndim == 0 else f"shape {arr.shape}"
        raise ValueError(f"{name} must be a 1-D sequence, got {shape}")
    if arr.size == 0 and not allow_empty:
        raise ValueError(f"{name} is empty; at least one value is required")
    return arr


def check_same_length(**arrays: NDArray[Any]) -> int:
    """Check that all keyword arrays have the same length and return it.

    Raises
    ------
    ValueError
        Listing every argument with its length when they disagree.
    """
    lengths = {k: int(v.shape[0]) for k, v in arrays.items()}
    if len(set(lengths.values())) > 1:
        detail = ", ".join(f"{k} has {n}" for k, n in lengths.items())
        raise ValueError(f"length mismatch: {detail}")
    return next(iter(lengths.values()))


def check_probabilities(
    p: NDArray[np.float64], name: str, *, atol: float = PROB_ATOL
) -> NDArray[np.float64]:
    """Check that every value lies in ``[0, 1 + atol]``.

    NaN is rejected. The array is returned unchanged (not clipped); callers
    that need values capped at 1 apply ``np.minimum(p, 1.0)`` themselves.

    Raises
    ------
    ValueError
        Naming the first value outside the range.
    """
    bad = ~((p >= 0.0) & (p <= 1.0 + atol))
    if bad.any():
        raise ValueError(f"{name} must lie in [0, 1]; {_first_bad(p, bad)} does not")
    return p


def check_labels(e: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Check that error labels (0/1 or soft labels) lie in ``[0, 1]``.

    Raises
    ------
    ValueError
        Naming the first label outside ``[0, 1]``.
    """
    bad = ~((e >= 0.0) & (e <= 1.0))
    if bad.any():
        raise ValueError(f"{name} must lie in [0, 1]; {_first_bad(e, bad)} does not")
    return e


def check_weights(w: ArrayLike | None, n: int, name: str = "sample_weight") -> NDArray[np.float64]:
    """Validate sample weights for ``n`` points; ``None`` means all ones.

    Returns
    -------
    numpy.ndarray
        float64 weights of shape ``(n,)``.

    Raises
    ------
    ValueError
        If the weights are not a 1-D sequence of length ``n``, are negative,
        non-finite, or sum to zero.
    """
    if w is None:
        return np.ones(n, dtype=np.float64)
    arr = as_1d(w, name, allow_empty=True)
    if arr.shape[0] != n:
        raise ValueError(f"{name} has length {arr.shape[0]}, expected {n}")
    neg = arr < 0.0
    if neg.any():
        raise ValueError(f"{name} must be non-negative; {_first_bad(arr, neg)} is not")
    if not float(arr.sum()) > 0.0:
        raise ValueError(f"{name} sums to zero; at least one weight must be positive")
    return arr


def check_finite_scalar(value: Any, name: str) -> float:
    """Return ``value`` as a float after checking it is a finite real number.

    Booleans are rejected, since ``True`` silently reads as 1.0.

    Raises
    ------
    ValueError
        If ``value`` is not a real number or is NaN or infinite.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (numbers.Real, np.floating, np.integer)
    ):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}")
    return out


def check_costs(c_small: Any, c_large: Any) -> tuple[float, float]:
    """Validate the per-query costs of the small and large model.

    Raises
    ------
    ValueError
        If either cost is not a finite, strictly positive number.
    """
    cs = check_finite_scalar(c_small, "c_small")
    cl = check_finite_scalar(c_large, "c_large")
    if cs <= 0.0:
        raise ValueError(f"c_small must be positive, got {cs!r}")
    if cl <= 0.0:
        raise ValueError(f"c_large must be positive, got {cl!r}")
    return cs, cl


def check_cost_model(cost_model: Any) -> str:
    """Validate the cost model name (``"routing"`` or ``"sequential"``).

    Raises
    ------
    ValueError
        For any other value.
    """
    if cost_model not in COST_MODELS:
        raise ValueError(f"cost_model must be 'routing' or 'sequential', got {cost_model!r}")
    return str(cost_model)


def check_grid(grid: ArrayLike) -> NDArray[np.float64]:
    """Validate a threshold grid and return it sorted with duplicates removed.

    Raises
    ------
    ValueError
        If the grid is empty, non-numeric, non-finite, or has a value
        outside ``[0, 1]``.
    """
    g = as_1d(grid, "grid")
    bad = (g < 0.0) | (g > 1.0)
    if bad.any():
        raise ValueError(f"grid values must lie in [0, 1]; {_first_bad(g, bad)} does not")
    return np.unique(g)


def check_positive_int(value: Any, name: str) -> int:
    """Return ``value`` as an int after checking it is an integer >= 1.

    Raises
    ------
    ValueError
        If ``value`` is not an integer (booleans included) or is below 1.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (numbers.Integral, np.integer)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    out = int(value)
    if out < 1:
        raise ValueError(f"{name} must be at least 1, got {out}")
    return out
