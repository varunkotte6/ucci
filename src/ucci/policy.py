"""Threshold policy and threshold selection (paper Section 4.3, Section 5).

Policy (Eq. 6)::

    pi_theta(x) = small  if p_hat(x) <= theta
                  large  if p_hat(x) >  theta

Threshold selection (Eq. 7): on a validation set V where both models have
been run,

    theta* = argmin_theta Cost(pi_theta)  subject to  Acc(pi_theta) >= tau,

where Cost and Acc are averages over V of the actual per-query costs and the
actual scores of the answers the policy returns (no simulated routing).
:func:`select_threshold` implements Eq. 7 over a finite grid of thresholds
(:data:`DEFAULT_GRID`: theta in [0, 1] at resolution 0.005, the grid used for
the paper's threshold selection; arXiv v1 does not print the resolution, see
``docs/paper_mapping.md``). With the default metric the cost falls as theta
rises, so the argmin is the largest feasible theta (fewest escalations).

:func:`select_threshold_for_budget` solves the budget form used for the
bottom block of Table 2 (methods compared at a matched cost budget of 2.00):
the highest accuracy with Cost(pi_theta) <= budget. :func:`pareto_frontier`
returns cost and accuracy for every grid threshold (Figure 2).

Cost model (Section 3): a query answered by the small model costs c_small and
an escalated one costs c_large, so the mean cost of a policy that escalates a
fraction r of queries is::

    routing:     c_small * (1 - r) + c_large * r          (the paper's model)
    sequential:  c_small + c_large * r                    (small always runs)

The paper reports normalized costs with c_small = 1.0 and c_large = 3.02, the
measured H100 latency ratio (Section 6.1); these are the defaults here.
Table 3 re-costs one fixed routing at several cost ratios. The paper does not
print its escalation rate; a rate of r = 0.5345 is consistent with all three
rows: 1 + 2.02 * r = 2.08 at c_large = 3.02, 1 + 4.0 * r = 3.14 at
c_large = 5.00 and 1 + 9.0 * r = 5.81 at c_large = 10.00. Under either cost
model every escalation adds the same marginal cost, so the chosen theta does
not depend on c_small and c_large (as long as escalation costs more than
keeping); only the reported cost does.

Theorem 1 (Section 5, Appendix A.1): assume (i) c_large > c_small, (ii) the
large model's accuracy alpha_large does not depend on which queries are
escalated, and (iii) p_hat is calibrated. Escalating x then gains
alpha_large - 1 + p_hat(x) expected accuracy (Eq. 8) at a fixed marginal
cost, so the cheapest way to reach tau escalates queries in decreasing order
of p_hat: among policies that depend only on u(x), a threshold policy on
p_hat is cost-optimal.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from ._validation import (
    as_1d,
    as_float_array,
    check_cost_model,
    check_costs,
    check_finite_scalar,
    check_grid,
    check_same_length,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    Metric = Callable[[NDArray[np.bool_]], float]

__all__ = [
    "DEFAULT_COST_LARGE",
    "DEFAULT_COST_SMALL",
    "DEFAULT_GRID",
    "DEFAULT_GRID_STEP",
    "InfeasibleTargetError",
    "ParetoFrontier",
    "ThresholdChoice",
    "escalate",
    "evaluate",
    "make_grid",
    "pareto_frontier",
    "policy_accuracy",
    "policy_cost",
    "select_threshold",
    "select_threshold_for_budget",
]

#: Normalized small-model cost (Section 6.1).
DEFAULT_COST_SMALL = 1.0
#: Normalized large-model cost: the measured latency ratio 142.3 ms / 47.2 ms
#: (Section 6.1).
DEFAULT_COST_LARGE = 3.02
#: Resolution of the default threshold grid.
DEFAULT_GRID_STEP = 0.005
#: theta in [0, 1] at resolution 0.005 (201 values). Read-only.
DEFAULT_GRID: NDArray[np.float64] = np.round(np.linspace(0.0, 1.0, 201), 3)
DEFAULT_GRID.setflags(write=False)


class InfeasibleTargetError(ValueError):
    """No threshold on the grid meets the accuracy target or the cost budget."""


@dataclass(frozen=True)
class ThresholdChoice:
    """A threshold with the cost and accuracy it achieves on some split.

    Attributes
    ----------
    theta : float
        The threshold of the policy pi_theta (Eq. 6).
    cost : float
        Mean per-query cost under the chosen cost model.
    accuracy : float
        Mean per-query score of the returned answers, or the value of the
        custom metric.
    escalation_rate : float
        Fraction of queries sent to the large model.
    """

    theta: float
    cost: float
    accuracy: float
    escalation_rate: float

    def to_dict(self) -> dict[str, float]:
        """The four fields as a plain dict."""
        return asdict(self)


@dataclass(frozen=True)
class ParetoFrontier:
    """Cost and accuracy of pi_theta for every threshold on a grid (Figure 2).

    All arrays have one entry per grid value, in increasing theta.

    Attributes
    ----------
    theta, cost, accuracy, escalation_rate : numpy.ndarray of float64
        As in :class:`ThresholdChoice`.
    efficient : numpy.ndarray of bool
        True where no other grid point is at most as costly and at least as
        accurate with one of the two strictly better. Plot
        ``cost[efficient]`` against ``accuracy[efficient]`` for the frontier.
    """

    theta: NDArray[np.float64]
    cost: NDArray[np.float64]
    accuracy: NDArray[np.float64]
    escalation_rate: NDArray[np.float64]
    efficient: NDArray[np.bool_]

    def to_dict(self) -> dict[str, list[Any]]:
        """All arrays as lists, ready for ``json.dumps``."""
        return {
            "theta": self.theta.tolist(),
            "cost": self.cost.tolist(),
            "accuracy": self.accuracy.tolist(),
            "escalation_rate": self.escalation_rate.tolist(),
            "efficient": self.efficient.tolist(),
        }


def make_grid(step: float = DEFAULT_GRID_STEP) -> NDArray[np.float64]:
    """Evenly spaced thresholds on [0, 1], both ends included.

    Parameters
    ----------
    step : float, default 0.005
        Spacing; ``1 / step`` must be a whole number (0.005, 0.01, 0.1, ...).

    Returns
    -------
    numpy.ndarray of float64
        ``round(1 / step) + 1`` values; ``make_grid(0.005)`` equals
        :data:`DEFAULT_GRID`.

    Raises
    ------
    ValueError
        If ``step`` is not in (0, 1] or does not divide 1.
    """
    s = check_finite_scalar(step, "step")
    if not 0.0 < s <= 1.0:
        raise ValueError(f"step must lie in (0, 1], got {s!r}")
    n = round(1.0 / s)
    if abs(n * s - 1.0) > 1e-9:
        raise ValueError(f"step must divide 1 exactly (e.g. 0.005 or 0.01), got {s!r}")
    return np.round(np.linspace(0.0, 1.0, n + 1), 12)


def _as_mask(esc: ArrayLike, name: str = "esc") -> NDArray[np.bool_]:
    """A non-empty 1-D boolean escalation mask (numeric 0/1 is accepted)."""
    raw = np.asarray(esc)
    if raw.dtype == np.bool_:
        mask = raw
    else:
        vals = as_float_array(raw, name)
        if not bool(np.all((vals == 0.0) | (vals == 1.0))):
            raise ValueError(f"{name} must be boolean (or 0/1)")
        mask = vals == 1.0
    if mask.ndim != 1:
        raise ValueError(f"{name} must be a 1-D mask, got shape {mask.shape}")
    if mask.size == 0:
        raise ValueError(f"{name} is empty; at least one query is required")
    return mask


def _cost_at_rate(rate: Any, c_small: float, c_large: float, cost_model: str) -> Any:
    """Mean per-query cost when a fraction ``rate`` of queries is escalated."""
    if cost_model == "routing":
        return c_small * (1.0 - rate) + c_large * rate
    return c_small + c_large * rate


@overload
def escalate(p_hat: float, theta: float) -> bool: ...


@overload
def escalate(p_hat: ArrayLike, theta: float) -> NDArray[np.bool_]: ...


def escalate(p_hat: ArrayLike, theta: float) -> bool | NDArray[np.bool_]:
    """The routing decision of pi_theta (Eq. 6): True means escalate.

    Parameters
    ----------
    p_hat : float or array_like of float
        Calibrated error probabilities, any shape.
    theta : float
        Threshold. A query is escalated when ``p_hat > theta`` (strictly), so
        a query with ``p_hat == theta`` stays with the small model.

    Returns
    -------
    bool or numpy.ndarray of bool
        A bool for a scalar input, otherwise a mask of the input's shape.

    Raises
    ------
    ValueError
        If ``p_hat`` or ``theta`` is NaN or infinite.
    """
    t = check_finite_scalar(theta, "theta")
    arr = as_float_array(p_hat, "p_hat")
    out = arr > t
    if arr.ndim == 0 and not isinstance(p_hat, np.ndarray):
        return bool(out)
    return np.asarray(out, dtype=bool)


def policy_cost(
    esc: ArrayLike,
    c_small: float = DEFAULT_COST_SMALL,
    c_large: float = DEFAULT_COST_LARGE,
    cost_model: str = "routing",
) -> float:
    """Mean per-query cost of a routing mask (Section 3, Table 3).

    Parameters
    ----------
    esc : array_like of bool, shape (n,)
        True where the query is escalated.
    c_small, c_large : float, default 1.0 and 3.02
        Per-query cost of the small and large model (Section 6.1).
    cost_model : {"routing", "sequential"}, default "routing"
        ``"routing"`` (the paper's): kept queries cost ``c_small``, escalated
        ones ``c_large``. ``"sequential"``: the small model always runs
        first to produce u(x), so an escalated query costs
        ``c_small + c_large``.

    Returns
    -------
    float
        ``c_small * (1 - r) + c_large * r`` (routing) or
        ``c_small + c_large * r`` (sequential), with ``r`` the escalation
        rate. Every function in this package uses this exact formula, so
        equal masks give bit-identical costs.

    Raises
    ------
    ValueError
        For an empty or non-boolean mask, non-positive or non-finite costs,
        or an unknown cost model.
    """
    mask = _as_mask(esc)
    cs, cl = check_costs(c_small, c_large)
    model = check_cost_model(cost_model)
    rate = int(np.count_nonzero(mask)) / mask.size
    return float(_cost_at_rate(rate, cs, cl, model))


def policy_accuracy(esc: ArrayLike, small_score: ArrayLike, large_score: ArrayLike) -> float:
    """Mean score of the answers a routing mask returns (Acc in Eq. 7).

    Parameters
    ----------
    esc : array_like of bool, shape (n,)
        True where the query is escalated.
    small_score, large_score : array_like of float, shape (n,)
        Per-query score of each model's actual output: 0/1 correctness, or
        any per-query metric such as per-query F1. For a corpus-level metric
        (micro-F1 over entities), use ``metric=`` in :func:`select_threshold`
        with :func:`ucci.metrics.routed_micro_f1`.

    Returns
    -------
    float
        ``(sum of small_score over kept + sum of large_score over escalated) / n``.

    Raises
    ------
    ValueError
        On empty or mismatched inputs, a non-boolean mask, or NaN/inf scores.
    """
    mask = _as_mask(esc)
    s = as_1d(small_score, "small_score")
    lg = as_1d(large_score, "large_score")
    n = check_same_length(esc=mask, small_score=s, large_score=lg)
    return float((s[~mask].sum() + lg[mask].sum()) / n)


def _metric_value(metric: Metric, mask: NDArray[np.bool_], theta: float) -> float:
    """Call a custom metric on an escalation mask and check the result."""
    value = metric(mask)
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metric must return a number, got {value!r} at theta={theta!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"metric returned {out!r} at theta={theta!r}")
    return out


class _Sweep:
    """Escalation counts and accuracies of pi_theta for every grid threshold."""

    def __init__(
        self,
        p_hat: ArrayLike,
        small_score: ArrayLike | None,
        large_score: ArrayLike | None,
        grid: ArrayLike,
        metric: Metric | None,
    ) -> None:
        p = as_1d(p_hat, "p_hat")
        g = check_grid(grid)
        n = p.size
        # b_i = number of grid values strictly below p_i, so that
        # p_i > grid[j]  <=>  j < b_i  (the grid is strictly increasing).
        b = np.searchsorted(g, p, side="left")
        counts = np.bincount(b, minlength=g.size + 1)
        # k[j] = #{i : b_i > j} = number escalated at theta = grid[j].
        k = np.cumsum(counts[::-1])[::-1][1:]
        self.theta = g
        self.n = n
        self.k = k.astype(np.int64)
        self.rate = self.k / n

        if metric is None:
            if small_score is None or large_score is None:
                raise ValueError("small_score and large_score are required without a metric")
            s = as_1d(small_score, "small_score")
            lg = as_1d(large_score, "large_score")
            check_same_length(p_hat=p, small_score=s, large_score=lg)
            s_bin = np.bincount(b, weights=s, minlength=g.size + 1)
            l_bin = np.bincount(b, weights=lg, minlength=g.size + 1)
            kept_small = np.cumsum(s_bin)[:-1]  # bins 0..j are kept
            esc_large = np.cumsum(l_bin[::-1])[::-1][1:]  # bins j+1..G are escalated
            self.accuracy = np.asarray((kept_small + esc_large) / n, dtype=np.float64)
        else:
            if not callable(metric):
                raise ValueError("metric must be a callable taking the escalation mask")
            cache: dict[int, float] = {}
            acc = np.empty(g.size, dtype=np.float64)
            for j in range(g.size):
                kj = int(self.k[j])
                if kj not in cache:
                    cache[kj] = _metric_value(metric, p > g[j], float(g[j]))
                acc[j] = cache[kj]
            self.accuracy = acc

    def costs(self, c_small: float, c_large: float, cost_model: str) -> NDArray[np.float64]:
        return np.asarray(_cost_at_rate(self.rate, c_small, c_large, cost_model), dtype=np.float64)

    def choice(self, j: int, costs: NDArray[np.float64]) -> ThresholdChoice:
        return ThresholdChoice(
            theta=float(self.theta[j]),
            cost=float(costs[j]),
            accuracy=float(self.accuracy[j]),
            escalation_rate=float(self.rate[j]),
        )


def _checked_costs(
    c_small: float, c_large: float, cost_model: str, *, need_order: bool
) -> tuple[float, float, str]:
    cs, cl = check_costs(c_small, c_large)
    model = check_cost_model(cost_model)
    if need_order and model == "routing" and not cl > cs:
        raise ValueError(
            "Theorem 1 assumption (i) needs c_large > c_small under the routing cost "
            f"model; got c_small={cs!r}, c_large={cl!r}"
        )
    return cs, cl, model


def select_threshold(
    p_hat: ArrayLike,
    small_score: ArrayLike | None,
    large_score: ArrayLike | None,
    tau: float,
    c_small: float = DEFAULT_COST_SMALL,
    c_large: float = DEFAULT_COST_LARGE,
    grid: ArrayLike = DEFAULT_GRID,
    metric: Metric | None = None,
    cost_model: str = "routing",
) -> ThresholdChoice:
    """Cheapest grid threshold whose validation accuracy is at least tau (Eq. 7).

    Parameters
    ----------
    p_hat : array_like of float, shape (n,)
        Calibrated error probabilities of the validation queries.
    small_score, large_score : array_like of float, shape (n,), or None
        Per-query scores of each model's actual output on the validation
        split (both models run on it). Ignored, and may be None, when
        ``metric`` is given.
    tau : float
        Accuracy target.
    c_small, c_large : float, default 1.0 and 3.02
        Per-query costs (Section 6.1). Under the routing cost model
        ``c_large > c_small`` is required (Theorem 1, assumption (i)).
    grid : array_like of float, default DEFAULT_GRID
        Candidate thresholds in [0, 1]; sorted and de-duplicated internally.
    metric : callable, optional
        ``metric(esc_mask) -> float`` replaces the mean per-query score, for
        a corpus-level metric such as micro-F1 of the routed answers (see
        :func:`ucci.metrics.routed_micro_f1`). Called once per distinct
        escalation mask on the grid.
    cost_model : {"routing", "sequential"}, default "routing"
        See :func:`policy_cost`.

    Returns
    -------
    ThresholdChoice
        The threshold with the lowest cost among those with accuracy >= tau.
        Ties in cost go to the higher accuracy, then to the larger theta.

    Raises
    ------
    InfeasibleTargetError
        If no threshold on the grid reaches tau. The message gives the best
        accuracy on the grid. A subclass of ValueError.
    ValueError
        On invalid inputs.

    Notes
    -----
    The default metric is evaluated for all thresholds at once from per-bin
    sums (O(n log G) for n queries and G thresholds), so n = 10^6 with the
    201-point grid takes a fraction of a second. Comparisons are exact
    (``accuracy >= tau``); with 0/1 scores every accuracy is computed
    exactly as (integer count) / n.

    Examples
    --------
    >>> p_hat = [0.1, 0.2, 0.6, 0.9]
    >>> choice = select_threshold(p_hat, [1, 1, 0, 0], [1, 1, 1, 1], tau=1.0)
    >>> choice.theta, choice.accuracy, choice.escalation_rate
    (0.595, 1.0, 0.5)
    """
    t = check_finite_scalar(tau, "tau")
    cs, cl, model = _checked_costs(c_small, c_large, cost_model, need_order=True)
    sweep = _Sweep(p_hat, small_score, large_score, grid, metric)
    costs = sweep.costs(cs, cl, model)
    feasible = np.flatnonzero(sweep.accuracy >= t)
    if feasible.size == 0:
        j = int(np.argmax(sweep.accuracy))
        raise InfeasibleTargetError(
            f"no threshold on the grid reaches tau={t!r}; the best accuracy on "
            f"the grid is {sweep.accuracy[j]:.6g} (theta={sweep.theta[j]:.6g}). "
            "Lower tau or check that the large model beats the small one on "
            "this split."
        )
    idx = feasible[costs[feasible] == costs[feasible].min()]
    idx = idx[sweep.accuracy[idx] == sweep.accuracy[idx].max()]
    return sweep.choice(int(idx[-1]), costs)


def select_threshold_for_budget(
    p_hat: ArrayLike,
    small_score: ArrayLike | None,
    large_score: ArrayLike | None,
    budget: float,
    c_small: float = DEFAULT_COST_SMALL,
    c_large: float = DEFAULT_COST_LARGE,
    grid: ArrayLike = DEFAULT_GRID,
    metric: Metric | None = None,
    cost_model: str = "routing",
) -> ThresholdChoice:
    """Most accurate grid threshold whose validation cost is within a budget.

    The budget form of Eq. 7, ``argmax_theta Acc(pi_theta)`` subject to
    ``Cost(pi_theta) <= budget``, as used for the matched-cost comparison in
    the bottom block of Table 2 (budget 2.00).

    Parameters
    ----------
    p_hat, small_score, large_score, c_small, c_large, grid, metric, cost_model
        As in :func:`select_threshold`.
    budget : float
        Maximum mean per-query cost, in the units of ``c_small`` and
        ``c_large``.

    Returns
    -------
    ThresholdChoice
        The threshold with the highest accuracy among those with
        cost <= budget. Ties in accuracy go to the lower cost, then to the
        larger theta.

    Raises
    ------
    InfeasibleTargetError
        If every threshold on the grid costs more than the budget. The
        message gives the cheapest cost on the grid. A subclass of
        ValueError.
    ValueError
        On invalid inputs.
    """
    bud = check_finite_scalar(budget, "budget")
    cs, cl, model = _checked_costs(c_small, c_large, cost_model, need_order=True)
    sweep = _Sweep(p_hat, small_score, large_score, grid, metric)
    costs = sweep.costs(cs, cl, model)
    feasible = np.flatnonzero(costs <= bud)
    if feasible.size == 0:
        j = int(np.argmin(costs))
        raise InfeasibleTargetError(
            f"no threshold on the grid has cost <= budget={bud!r}; the cheapest costs "
            f"{costs[j]:.6g} (theta={sweep.theta[j]:.6g})"
        )
    idx = feasible[sweep.accuracy[feasible] == sweep.accuracy[feasible].max()]
    idx = idx[costs[idx] == costs[idx].min()]
    return sweep.choice(int(idx[-1]), costs)


def pareto_frontier(
    p_hat: ArrayLike,
    small_score: ArrayLike | None,
    large_score: ArrayLike | None,
    c_small: float = DEFAULT_COST_SMALL,
    c_large: float = DEFAULT_COST_LARGE,
    grid: ArrayLike = DEFAULT_GRID,
    metric: Metric | None = None,
    cost_model: str = "routing",
) -> ParetoFrontier:
    """Cost and accuracy of pi_theta at every grid threshold (Figure 2).

    Parameters
    ----------
    p_hat, small_score, large_score, c_small, c_large, grid, metric, cost_model
        As in :func:`select_threshold`, except that any positive costs are
        accepted (the sweep is descriptive).

    Returns
    -------
    ParetoFrontier
        Arrays over the sorted, de-duplicated grid, plus the mask of
        efficient (non-dominated) points.
    """
    cs, cl, model = _checked_costs(c_small, c_large, cost_model, need_order=False)
    sweep = _Sweep(p_hat, small_score, large_score, grid, metric)
    cost = sweep.costs(cs, cl, model)
    acc = sweep.accuracy
    unique_cost, group = np.unique(cost, return_inverse=True)
    group = group.reshape(-1)
    best_in_group = np.full(unique_cost.size, -np.inf)
    np.maximum.at(best_in_group, group, acc)
    best_cheaper = np.concatenate(([-np.inf], np.maximum.accumulate(best_in_group)[:-1]))
    efficient = (acc == best_in_group[group]) & (acc > best_cheaper[group])
    return ParetoFrontier(
        theta=sweep.theta.copy(),
        cost=cost,
        accuracy=acc.copy(),
        escalation_rate=np.asarray(sweep.rate, dtype=np.float64),
        efficient=np.asarray(efficient, dtype=bool),
    )


def evaluate(
    p_hat: ArrayLike,
    small_score: ArrayLike | None,
    large_score: ArrayLike | None,
    theta: float,
    c_small: float = DEFAULT_COST_SMALL,
    c_large: float = DEFAULT_COST_LARGE,
    metric: Metric | None = None,
    cost_model: str = "routing",
) -> ThresholdChoice:
    """Route every query with pi_theta and report the actual cost and accuracy.

    This is step 3 of the evaluation protocol (Section 6.1): on the test
    split, apply the threshold chosen on validation and accumulate the actual
    cost and the scores of the outputs actually returned.

    Parameters
    ----------
    p_hat : array_like of float, shape (n,)
        Calibrated error probabilities of the test queries.
    small_score, large_score : array_like of float, shape (n,), or None
        Per-query scores of each model's output. Ignored, and may be None,
        when ``metric`` is given.
    theta : float
        Threshold to apply.
    c_small, c_large, metric, cost_model
        As in :func:`select_threshold`, except that any positive costs are
        accepted.

    Returns
    -------
    ThresholdChoice
        ``theta`` with the cost, accuracy and escalation rate it achieves.
    """
    cs, cl, model = _checked_costs(c_small, c_large, cost_model, need_order=False)
    mask = _as_mask(escalate(as_1d(p_hat, "p_hat"), theta), "p_hat")
    if metric is None:
        if small_score is None or large_score is None:
            raise ValueError("small_score and large_score are required without a metric")
        acc = policy_accuracy(mask, small_score, large_score)
    else:
        if not callable(metric):
            raise ValueError("metric must be a callable taking the escalation mask")
        acc = _metric_value(metric, mask, float(theta))
    rate = int(np.count_nonzero(mask)) / mask.size
    return ThresholdChoice(
        theta=float(theta),
        cost=float(_cost_at_rate(rate, cs, cl, model)),
        accuracy=float(acc),
        escalation_rate=float(rate),
    )
