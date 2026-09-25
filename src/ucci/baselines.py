"""Comparison methods and ablations for cascade routing (paper Section 6.1, 6.3).

This module implements every routing method the paper compares UCCI against
(Section 6.1, "Baselines"; Table 2) and the ablations of Section 6.3 and
Appendix B.4, so the paper's comparison protocol can be run on any workload
where both models have been run on the calibration and validation queries.

Methods
-------
Table 2 comparators (Section 6.1):

* :class:`AlwaysSmall` and :class:`AlwaysLarge`: single-model inference.
* :class:`EntropyThresholdRouter`: route by uncalibrated mean token entropy.
* :class:`SplitConformalRouter`: split conformal prediction on the binary
  event "small model is correct", with raw u(x) as the nonconformity score.
* :class:`FrugalGPTStyleRouter`: a confidence threshold tuned on the
  validation set to meet the accuracy target.

Ablations (Section 6.3 and Appendix B.4):

* :class:`TemperatureScalingCalibrator`: a single-parameter monotone
  rescaling of u(x), compared against isotonic regression.
* :class:`IdentityCalibrator`: uncalibrated routing (p_hat = u).
* :func:`mean_token_entropy` and :func:`mean_max_prob`: the two alternative
  uncertainty signals (predictive entropy, max probability).
* :class:`CalibratedThresholdRouter`: the Eq. 6 policy and the Section 4.3
  threshold selection (Eq. 7) on top of any calibrator, so calibrators are
  compared under an identical protocol.

Additions that are not in the paper, each labelled as such where defined:

* :class:`Oracle`: a label-dependent lower bound on cost, for analysis only.
* :class:`PlattCalibrator`: logistic (Platt) calibration, an extension used
  by some cascade implementations.

:func:`compare_routers` runs the three-step protocol of Section 6.1 for any
set of methods: fit on the calibration split, select on the validation
split, then route every test query end to end with actual outputs and costs.

Shared conventions
------------------
Every router exposes the interface of :class:`ucci.UCCIRouter`:
``calibrate(scores, e)``, ``choose_threshold(scores, small_score,
large_score, tau)`` returning a :class:`ucci.ThresholdChoice`, and
``escalate(scores)`` returning a boolean mask (True = send to the large
model). ``e`` is the paper's error label (Section 4.2): 1 when the small
model's output is wrong. Every ``choose_threshold`` also accepts
``budget=`` for the matched-cost-budget form of Table 2 (bottom block) and
``metric=`` for corpus-level metrics such as micro-F1, with the same meaning
as in :func:`ucci.select_threshold`.

Every threshold rule here is selected by the core's own implementation of
Section 4.3 (Eq. 7), :func:`ucci.select_threshold` (accuracy target) and
:func:`ucci.select_threshold_for_budget` (cost budget), so all methods share
UCCI's selection semantics exactly:

* accuracy-target form: minimum validation cost subject to validation
  accuracy >= tau; ties in cost go to the higher accuracy, then to the larger
  threshold (fewer escalations);
* budget form: maximum validation accuracy subject to validation cost <=
  budget; ties in accuracy go to the lower cost, then to the larger
  threshold.

Thresholds on raw scores (entropy, confidence, conformal quantiles) are not
probabilities, so they are passed to the core through an exact
order-preserving relabelling of the candidate thresholds (see
:func:`exact_threshold_candidates`); the masks, costs and accuracies are
unchanged by it. Test-split numbers come from :func:`ucci.policy_cost` and
:func:`ucci.policy_accuracy` (or ``metric``) on the actual routing mask, as
in :func:`ucci.evaluate`. Infeasible targets raise
:class:`ucci.InfeasibleTargetError`.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Protocol, cast, overload

import numpy as np

from ._validation import (
    PROB_ATOL,
    as_1d,
    as_float_array,
    check_cost_model,
    check_costs,
    check_finite_scalar,
    check_grid,
    check_labels,
    check_same_length,
    check_weights,
)
from .calibration import IsotonicCalibrator
from .policy import (
    DEFAULT_COST_LARGE,
    DEFAULT_COST_SMALL,
    DEFAULT_GRID,
    InfeasibleTargetError,
    ThresholdChoice,
    escalate,
    pareto_frontier,
    policy_accuracy,
    policy_cost,
    select_threshold,
    select_threshold_for_budget,
)
from .router import UCCIRouter

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from numpy.typing import ArrayLike, NDArray

    #: A corpus-level accuracy metric evaluated on a routing mask.
    Metric = Callable[[NDArray[np.bool_]], float]

__all__ = [
    "DEFAULT_DELTA_GRID",
    "AlwaysLarge",
    "AlwaysSmall",
    "CalibratedThresholdRouter",
    "Calibrator",
    "ComparisonRow",
    "EntropyThresholdRouter",
    "FrugalGPTStyleRouter",
    "IdentityCalibrator",
    "MethodSpec",
    "Oracle",
    "PlattCalibrator",
    "RawThresholdRouter",
    "RouterLike",
    "SplitConformalRouter",
    "SplitData",
    "TemperatureScalingCalibrator",
    "ablation_methods",
    "compare_routers",
    "exact_threshold_candidates",
    "format_comparison",
    "mean_max_prob",
    "mean_token_entropy",
    "table2_methods",
    "token_entropies",
    "token_max_probs",
]

#: Miscoverage levels searched by :class:`SplitConformalRouter`: 0.005 to
#: 0.995 in steps of 0.005, the resolution of the paper's theta grid.
DEFAULT_DELTA_GRID: NDArray[np.float64] = np.round(np.arange(1, 200) * 0.005, 3)
DEFAULT_DELTA_GRID.setflags(write=False)

_EPS = float(np.finfo(np.float64).eps)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_router_costs(
    c_small: float, c_large: float, cost_model: str
) -> tuple[float, float, str]:
    """Validate costs and cost model; Theorem 1 (i) needs c_large > c_small."""
    cs, cl = check_costs(c_small, c_large)
    cm = check_cost_model(cost_model)
    if cm == "routing" and not cl > cs:
        raise ValueError(
            "Theorem 1 assumes c_large > c_small under the routing cost model; "
            f"got c_small={cs!r}, c_large={cl!r}"
        )
    return cs, cl, cm


def _objective(tau: float | None, budget: float | None) -> tuple[float | None, float | None]:
    """Validate that exactly one of ``tau`` and ``budget`` is given."""
    if tau is None and budget is None:
        raise ValueError(
            "pass tau (accuracy target, Section 4.3 Eq. 7) or budget "
            "(matched cost budget, Table 2 bottom block)"
        )
    if tau is not None and budget is not None:
        raise ValueError("pass only one of tau and budget")
    if tau is not None:
        return check_finite_scalar(tau, "tau"), None
    return None, check_finite_scalar(budget, "budget")


def _problem(
    scores: ArrayLike,
    small_score: ArrayLike,
    large_score: ArrayLike,
    name: str = "scores",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate a routing problem: three finite 1-D arrays of equal length."""
    s = as_1d(scores, name)
    sm = as_1d(small_score, "small_score")
    lg = as_1d(large_score, "large_score")
    kwargs = {name: s, "small_score": sm, "large_score": lg}
    check_same_length(**kwargs)
    return s, sm, lg


def _binary_labels(e: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """Check that labels are exactly 0 or 1."""
    bad = (e != 0.0) & (e != 1.0)
    if bad.any():
        i = int(np.argmax(bad))
        raise ValueError(f"{name} must be 0 or 1; index {i} (value {e[i]!r}) is not")
    return e


def _metric_value(metric: Metric, mask: NDArray[np.bool_]) -> float:
    """Call a user metric and check it returns a finite number."""
    raw = metric(mask)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metric must return a number, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"metric returned a non-finite value ({value!r})")
    return value


def _evaluate_mask(
    mask: NDArray[np.bool_],
    small: NDArray[np.float64],
    large: NDArray[np.float64],
    theta: float,
    c_small: float,
    c_large: float,
    cost_model: str,
    metric: Metric | None,
) -> ThresholdChoice:
    """Cost, accuracy and escalation rate of a routing mask.

    The numbers come from :func:`ucci.policy_cost` and
    :func:`ucci.policy_accuracy` (or ``metric``), exactly as in
    :func:`ucci.evaluate`.
    """
    acc = (
        _metric_value(metric, mask)
        if metric is not None
        else float(policy_accuracy(mask, small, large))
    )
    cost = float(policy_cost(mask, c_small, c_large, cost_model))
    return ThresholdChoice(
        theta=float(theta), cost=cost, accuracy=acc, escalation_rate=float(mask.mean())
    )


# ---------------------------------------------------------------------------
# Exact threshold selection on an arbitrary candidate set
# ---------------------------------------------------------------------------


def exact_threshold_candidates(scores: ArrayLike) -> NDArray[np.float64]:
    """Candidate thresholds that make the search over a raw score exact.

    Returns ``-inf``, every distinct value of ``scores`` in increasing order,
    and ``+inf``. Under the rule "escalate when score > t", each distinct
    value ``v`` produces the mask that keeps exactly the queries with score
    <= v, ``-inf`` escalates every query and ``+inf`` escalates none. These
    are all the distinct masks any threshold can produce on ``scores``, so a
    search over this set is exact rather than limited by a grid.

    Parameters
    ----------
    scores : array_like of float, shape (n,)
        Validation scores (finite).

    Returns
    -------
    numpy.ndarray of float64
        Sorted candidates, length (number of distinct scores) + 2.
    """
    s = as_1d(scores, "scores")
    return np.concatenate(([-np.inf], np.unique(s), [np.inf]))


def _select_on_candidates(
    scores: NDArray[np.float64],
    small: NDArray[np.float64],
    large: NDArray[np.float64],
    candidates: NDArray[np.float64],
    *,
    tau: float | None,
    budget: float | None,
    c_small: float,
    c_large: float,
    cost_model: str,
    metric: Metric | None,
) -> ThresholdChoice:
    """Select a threshold for the rule "escalate when score > t" (Eq. 6, 7).

    The search is delegated to :func:`ucci.select_threshold` (``tau``) or
    :func:`ucci.select_threshold_for_budget` (``budget``) through an exact
    relabelling: with sorted distinct candidates ``c_0 < ... < c_{m-1}`` and
    ``b_i`` the number of candidates strictly below ``scores[i]``, set
    ``p_i = max(b_i - 1/2, 0) / m`` and ``g_j = j / m``. Then
    ``scores[i] > c_j`` if and only if ``p_i > g_j`` (for m below 2**50), so
    every candidate produces the same mask, cost and accuracy as in the
    core, and ties are broken identically. The returned ``theta`` is mapped
    back to ``c_j``.
    """
    cands = np.unique(candidates)  # callers guarantee a non-empty, NaN-free set
    m = int(cands.size)
    below = np.searchsorted(cands, scores, side="left")
    ranks = np.maximum(below - 0.5, 0.0) / m
    grid = np.arange(m, dtype=np.float64) / m
    try:
        if tau is not None:
            choice = select_threshold(
                ranks, small, large, tau, c_small, c_large, grid, metric, cost_model
            )
        else:
            assert budget is not None
            choice = select_threshold_for_budget(
                ranks, small, large, budget, c_small, c_large, grid, metric, cost_model
            )
    except InfeasibleTargetError as exc:
        # Restate the core's message in the score's own units.
        front = pareto_frontier(ranks, small, large, c_small, c_large, grid, metric, cost_model)
        if tau is not None:
            detail = (
                f"no candidate threshold reaches tau={tau!r} on the validation set; "
                "the best accuracy any candidate achieves is "
                f"{front.accuracy.max():.6g}"
            )
        else:
            detail = (
                f"no candidate threshold has cost <= budget={budget!r} on the "
                f"validation set; the cheapest candidate costs {front.cost.min():.6g}"
            )
        raise InfeasibleTargetError(detail) from exc
    j = round(choice.theta * m)
    return dataclasses.replace(choice, theta=float(cands[j]))


# ---------------------------------------------------------------------------
# Router protocol and single-model anchors
# ---------------------------------------------------------------------------


class RouterLike(Protocol):
    """Interface shared by every router in this module (duck-typed).

    The same three steps as :class:`ucci.UCCIRouter`: ``calibrate`` on the
    calibration split, ``choose_threshold`` on the validation split,
    ``escalate`` on new queries. Methods that need no calibration accept
    ``calibrate`` as a no-op so every method runs under one protocol.
    """

    def calibrate(
        self, scores: ArrayLike, e: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> object:
        """Fit whatever the method learns from the calibration split."""
        ...

    def choose_threshold(
        self,
        scores: ArrayLike,
        small_score: ArrayLike,
        large_score: ArrayLike,
        tau: float | None = None,
        *,
        budget: float | None = None,
        metric: Metric | None = None,
    ) -> ThresholdChoice:
        """Select the operating point on the validation split."""
        ...

    def escalate(self, scores: ArrayLike) -> NDArray[np.bool_]:
        """Boolean mask, True where the query goes to the large model."""
        ...


class _CostMixin:
    """Holds and validates the cost configuration shared by all routers."""

    def __init__(
        self,
        c_small: float = DEFAULT_COST_SMALL,
        c_large: float = DEFAULT_COST_LARGE,
        cost_model: str = "routing",
    ) -> None:
        self.c_small, self.c_large, self.cost_model = _check_router_costs(
            c_small, c_large, cost_model
        )
        self.choice: ThresholdChoice | None = None

    @property
    def theta(self) -> float:
        """The selected threshold; raises until :meth:`choose_threshold` runs."""
        if self.choice is None:
            raise RuntimeError(f"{type(self).__name__}: call choose_threshold() first")
        return self.choice.theta

    def _evaluate(
        self,
        mask: NDArray[np.bool_],
        small: NDArray[np.float64],
        large: NDArray[np.float64],
        theta: float,
        metric: Metric | None,
    ) -> ThresholdChoice:
        return _evaluate_mask(
            mask,
            small,
            large,
            theta,
            self.c_small,
            self.c_large,
            self.cost_model,
            metric,
        )

    def __repr__(self) -> str:
        """Show the configuration."""
        return (
            f"{type(self).__name__}(c_small={self.c_small!r}, "
            f"c_large={self.c_large!r}, "
            f"cost_model={self.cost_model!r})"
        )


class _FixedPolicy(_CostMixin):
    """A routing policy with nothing to fit or select."""

    _escalate_all: bool = False

    def calibrate(
        self, scores: ArrayLike, e: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> _FixedPolicy:
        """No-op: a single-model policy uses no calibration data.

        Returns
        -------
        self
        """
        return self

    def choose_threshold(
        self,
        scores: ArrayLike,
        small_score: ArrayLike,
        large_score: ArrayLike,
        tau: float | None = None,
        *,
        budget: float | None = None,
        metric: Metric | None = None,
    ) -> ThresholdChoice:
        """Evaluate the fixed policy on the validation split.

        There is no threshold to choose, so ``tau`` and ``budget`` are not
        enforced; they are accepted so this class runs under the same
        protocol as the other routers. ``scores`` only sets the number of
        queries. The returned ``theta`` is ``-inf`` for always-large and
        ``+inf`` for always-small: the thresholds at which the rule "escalate
        when score > theta" reproduces the policy on any finite score.

        Returns
        -------
        ucci.ThresholdChoice
        """
        if tau is not None:
            check_finite_scalar(tau, "tau")
        if budget is not None:
            check_finite_scalar(budget, "budget")
        s, sm, lg = _problem(scores, small_score, large_score)
        self.choice = self._evaluate(self._mask(s.shape[0]), sm, lg, self._theta(), metric)
        return self.choice

    def escalate(self, scores: ArrayLike) -> NDArray[np.bool_]:
        """Constant mask with one entry per query in ``scores``."""
        return self._mask(int(as_1d(scores, "scores", allow_empty=True).shape[0]))

    def _mask(self, n: int) -> NDArray[np.bool_]:
        return np.full(n, self._escalate_all, dtype=bool)

    def _theta(self) -> float:
        return -math.inf if self._escalate_all else math.inf


class AlwaysSmall(_FixedPolicy):
    """Always answer with the small model (Table 2, "Small-only").

    Single-model inference with f_s (Section 6.1, "Always-small"). Its cost
    is ``c_small`` per query; the paper reports it as the lower-bound anchor
    that falls below the accuracy target.

    Parameters
    ----------
    c_small, c_large : float
        Per-query costs (defaults: the paper's normalized 1.0 and 3.02).
    cost_model : {"routing", "sequential"}
        As in :func:`ucci.policy_cost`.
    """

    _escalate_all = False


class AlwaysLarge(_FixedPolicy):
    """Always answer with the large model (Table 2, "Large-only").

    Single-model inference with f_l (Section 6.1, "Always-large"). The
    paper's cost savings are measured against this policy.

    Its cost is ``c_large`` per query under the routing cost model and
    ``c_small + c_large`` under the sequential one (the small model always
    runs first there).

    Parameters
    ----------
    c_small, c_large : float
        Per-query costs (defaults: the paper's normalized 1.0 and 3.02).
    cost_model : {"routing", "sequential"}
        As in :func:`ucci.policy_cost`.
    """

    _escalate_all = True


class Oracle(_CostMixin):
    """Label-dependent routing: a lower bound on cost (analysis only).

    Not in the paper. The oracle sees the correctness of both models on the
    very queries it routes, so it cannot be deployed; it bounds what any
    router could achieve on the same queries.

    * With neither ``tau`` nor ``budget``, it escalates exactly the queries
      where the large model's score beats the small model's, which for 0/1
      correctness means "small answer wrong and large answer right". This is
      the cheapest policy reaching the best accuracy any routing can reach,
      ``mean(max(small_score, large_score))``.
    * With ``tau``, it escalates the fewest queries needed to reach tau,
      taking the largest per-query gains ``large - small`` first. Every
      escalation costs the same, so this is the exact minimum cost of any
      policy, routing-based or not, that reaches tau under the mean
      per-query metric.
    * With ``budget``, it escalates the queries with the largest positive
      gains until the budget is spent: the exact maximum accuracy at that
      cost under the mean per-query metric.

    Ties between equal gains go to the lower query index, so results are
    deterministic. With a custom ``metric`` (for example corpus micro-F1)
    the same greedy order is used, and the bound is then no longer exact.

    Parameters
    ----------
    c_small, c_large : float
        Per-query costs (defaults: the paper's normalized 1.0 and 3.02).
    cost_model : {"routing", "sequential"}
        As in :func:`ucci.policy_cost`.
    """

    def escalate(
        self,
        small_score: ArrayLike,
        large_score: ArrayLike,
        tau: float | None = None,
        *,
        budget: float | None = None,
        metric: Metric | None = None,
    ) -> NDArray[np.bool_]:
        """Oracle mask from both models' per-query scores.

        Parameters
        ----------
        small_score, large_score : array_like of float, shape (n,)
            Per-query score of each model's answer (0/1 or graded).
        tau : float, optional
            Accuracy target; escalate the fewest queries that reach it.
        budget : float, optional
            Mean cost budget; escalate the most valuable queries within it.
        metric : callable, optional
            ``metric(mask) -> float`` replacing the mean per-query score.

        Returns
        -------
        numpy.ndarray of bool, shape (n,)

        Raises
        ------
        InfeasibleTargetError
            If tau exceeds what any routing achieves, or the budget is below
            ``c_small``.
        """
        return self._solve(small_score, large_score, tau, budget, metric)[0]

    def evaluate(
        self,
        small_score: ArrayLike,
        large_score: ArrayLike,
        tau: float | None = None,
        *,
        budget: float | None = None,
        metric: Metric | None = None,
    ) -> ThresholdChoice:
        """Cost, accuracy and escalation rate of the oracle mask.

        Same arguments as :meth:`escalate`. The returned ``theta`` is NaN:
        the oracle is not a threshold rule.
        """
        mask, sm, lg = self._solve(small_score, large_score, tau, budget, metric)
        self.choice = self._evaluate(mask, sm, lg, math.nan, metric)
        return self.choice

    def _solve(
        self,
        small_score: ArrayLike,
        large_score: ArrayLike,
        tau: float | None,
        budget: float | None,
        metric: Metric | None,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
        sm = as_1d(small_score, "small_score")
        lg = as_1d(large_score, "large_score")
        n = check_same_length(small_score=sm, large_score=lg)
        gain = lg - sm
        if tau is None and budget is None:
            return gain > 0.0, sm, lg
        tau_v, budget_v = _objective(tau, budget)
        # Priority n for the largest gain down to 1 for the smallest; the
        # threshold n - k escalates exactly the top-k positive gains.
        order = np.argsort(-gain, kind="mergesort")
        priority = np.empty(n, dtype=np.float64)
        priority[order] = np.arange(n, 0, -1, dtype=np.float64)
        n_pos = int((gain > 0.0).sum())
        cands = n - np.arange(n_pos + 1, dtype=np.float64)
        try:
            choice = _select_on_candidates(
                priority,
                sm,
                lg,
                cands,
                tau=tau_v,
                budget=budget_v,
                c_small=self.c_small,
                c_large=self.c_large,
                cost_model=self.cost_model,
                metric=metric,
            )
        except InfeasibleTargetError as exc:
            best = float(np.maximum(sm, lg).mean())
            raise InfeasibleTargetError(
                f"oracle: {exc}; the best accuracy any routing reaches on these "
                f"labels is mean(max(small, large)) = {best:.6g}"
            ) from exc
        return priority > choice.theta, sm, lg


# ---------------------------------------------------------------------------
# Signals for the ablations (Section 6.3)
# ---------------------------------------------------------------------------


def _logprob_rows(token_logprobs: object, name: str) -> list[NDArray[np.float64]]:
    """Split per-token log-prob vectors into validated 1-D rows."""
    rows: list[Any]
    if isinstance(token_logprobs, np.ndarray) and token_logprobs.dtype != object:
        if token_logprobs.ndim != 2:
            raise ValueError(
                f"{name} must be a 2-D array (tokens x candidates) or a sequence of "
                f"1-D log-prob vectors, got an array of shape {token_logprobs.shape}"
            )
        rows = list(token_logprobs)
    else:
        try:
            rows = list(cast("Iterable[Any]", token_logprobs))
        except TypeError as exc:
            raise ValueError(
                f"{name} must be a sequence of per-token log-prob vectors: {exc}"
            ) from exc
    if len(rows) == 0:
        raise ValueError(
            f"{name} is empty: an uncertainty signal needs at least one generated token"
        )
    out = []
    for t, row in enumerate(rows):
        lp = as_1d(row, f"{name}[{t}]", finite=False)
        if np.isnan(lp).any() or (lp == np.inf).any():
            raise ValueError(f"{name}[{t}] contains NaN or +inf")
        if not (lp > -np.inf).any():
            raise ValueError(f"{name}[{t}] has no candidate with positive probability")
        out.append(lp)
    return out


def token_entropies(
    token_logprobs: Sequence[ArrayLike] | ArrayLike,
) -> NDArray[np.float64]:
    """Per-token predictive entropy, in nats, from per-token log-prob vectors.

    At each position the given log-probabilities are renormalized with a
    softmax, ``q_i = exp(lp_i) / sum_j exp(lp_j)``, and the entropy is
    ``H_t = -sum_i q_i log q_i``. With the full next-token distribution the
    renormalization changes nothing, so ``H_t`` is the exact predictive
    entropy. With only the top-k candidates (what serving APIs return) it is
    the entropy of the renormalized top-k distribution: the probability mass
    outside the top k is ignored, so it approximates the full entropy and is
    not a bound on it. Unnormalized logits give the same result as their
    log-softmax.

    Parameters
    ----------
    token_logprobs : sequence of array_like, or 2-D array
        One vector of candidate log-probabilities per generated token (the
        token convention of :mod:`ucci.signal`: content tokens only). Rows
        may have different lengths. ``-inf`` entries (zero probability) are
        allowed; NaN and ``+inf`` are not.

    Returns
    -------
    numpy.ndarray of float64, shape (T,)
        Entropy per position, >= 0.

    Raises
    ------
    ValueError
        If there are no positions, or a position is empty, has NaN or +inf,
        or has no finite log-probability.
    """
    rows = _logprob_rows(token_logprobs, "token_logprobs")
    out = np.empty(len(rows), dtype=np.float64)
    for t, lp in enumerate(rows):
        finite = lp[lp > -np.inf]
        top = float(finite.max())
        log_z = top + math.log(float(np.exp(finite - top).sum()))
        logq = finite - log_z
        out[t] = max(0.0, -float(np.sum(np.exp(logq) * logq)))
    return out


def mean_token_entropy(token_logprobs: Sequence[ArrayLike] | ArrayLike) -> float:
    """Mean token entropy of a generation (Section 6.1, 6.3 signal).

    The paper's "Entropy threshold" baseline routes by uncalibrated mean
    token entropy, and its signal ablation compares token margin against
    predictive entropy. This returns ``(1/T) sum_t H_t`` with ``H_t`` from
    :func:`token_entropies` (natural log). Larger means more uncertain.

    With the full next-token distribution at every position the value is
    exact. With top-k log-probabilities it is computed on the renormalized
    top-k distribution at each position (see :func:`token_entropies`). The
    serving adapters in :mod:`ucci.integrations` report the truncated top-k
    sum by default (``TokenSignals.entropy_support == "top_k"``); pass
    ``renormalize_entropy=True`` there to get this definition. Both agree
    when the full vocabulary is available (the transformers adapter).

    Parameters
    ----------
    token_logprobs : sequence of array_like, or 2-D array
        One vector of candidate log-probabilities per generated token.

    Returns
    -------
    float
        Mean entropy in nats.

    Examples
    --------
    >>> import math
    >>> round(mean_token_entropy([[math.log(0.5), math.log(0.5)]]), 6)
    0.693147
    """
    return float(token_entropies(token_logprobs).mean())


def token_max_probs(
    token_logprobs: Sequence[ArrayLike] | ArrayLike, *, renormalize: bool = False
) -> NDArray[np.float64]:
    """Per-token top-1 probability ``p_{t,1}`` from per-token log-prob vectors.

    Parameters
    ----------
    token_logprobs : sequence of array_like, or 2-D array
        One vector of candidate log-probabilities per generated token.
    renormalize : bool, default False
        False: ``p_{t,1} = exp(max_i lp_i)``. This is exact whether the vector
        holds the full distribution or only the top-k candidates, because the
        top-1 log-probability a server reports is the model's own
        probability; renormalizing a top-k vector would inflate it. True:
        softmax-renormalize the given candidates first, which is what you
        want for unnormalized logits (and what a top-k-only definition of the
        signal would use).

    Returns
    -------
    numpy.ndarray of float64, shape (T,)
        Values in [0, 1].

    Raises
    ------
    ValueError
        As :func:`token_entropies`, and, with ``renormalize=False``, if a
        log-probability is above ``log(1 + 1e-9)`` (the input then looks
        like logits: pass ``renormalize=True``).
    """
    rows = _logprob_rows(token_logprobs, "token_logprobs")
    out = np.empty(len(rows), dtype=np.float64)
    limit = math.log1p(PROB_ATOL)
    for t, lp in enumerate(rows):
        finite = lp[lp > -np.inf]
        top = float(finite.max())
        if renormalize:
            out[t] = 1.0 / float(np.exp(finite - top).sum())
        else:
            if top > limit:
                raise ValueError(
                    f"token_logprobs[{t}] has a log-probability above 0 ({top!r}); "
                    "these look like unnormalized logits, pass renormalize=True"
                )
            out[t] = min(1.0, math.exp(top))
    return out


def mean_max_prob(
    token_logprobs: Sequence[ArrayLike] | ArrayLike, *, renormalize: bool = False
) -> float:
    """Mean top-1 token probability of a generation (Section 6.3 signal).

    The "max probability" signal of the paper's signal ablation:
    ``(1/T) sum_t p_{t,1}``. Larger means more confident, so use
    ``1 - mean_max_prob`` wherever an uncertainty (larger = less certain) is
    expected, for example as the input of an isotonic calibrator.

    With the full distribution the value is exact. With only top-k
    log-probabilities it is also exact by default, since the top-1
    log-probability is reported directly; pass ``renormalize=True`` to
    compute it on the renormalized top-k (or on logits) instead. See
    :func:`token_max_probs`.

    Parameters
    ----------
    token_logprobs : sequence of array_like, or 2-D array
        One vector of candidate log-probabilities per generated token.
    renormalize : bool, default False
        Softmax-renormalize each position's candidates first.

    Returns
    -------
    float
        Value in [0, 1].

    Examples
    --------
    >>> import math
    >>> round(mean_max_prob([[math.log(0.9), math.log(0.05)], [math.log(0.6)]]), 6)
    0.75
    """
    return float(token_max_probs(token_logprobs, renormalize=renormalize).mean())


# ---------------------------------------------------------------------------
# Raw-score threshold routers (Section 6.1: entropy threshold, FrugalGPT-style)
# ---------------------------------------------------------------------------


class RawThresholdRouter(_CostMixin):
    """Threshold on an uncalibrated score, tuned on validation (Section 6.1).

    The rule is Eq. 6 applied to a raw score instead of a calibrated
    probability: with ``direction="uncertainty"`` a query is escalated when
    ``score > threshold``; with ``direction="confidence"`` when
    ``score < threshold``. :meth:`choose_threshold` runs the Section 4.3
    selection (Eq. 7) on the validation split with actual outputs and costs.

    Our choice where the paper leaves the search open: the candidate
    thresholds are every distinct validation score plus the two endpoints
    (``-inf`` and ``+inf``; see :func:`exact_threshold_candidates`). That
    set produces every mask a threshold can produce on the validation data,
    so the search is exact, with no grid resolution and no need to rescale
    the score into [0, 1]. The chosen threshold is always a validation score
    or an endpoint: the endpoint that escalates nothing generalizes to
    always-small on new data and the one that escalates everything to
    always-large. Pass ``candidates`` to restrict the search (for example
    to a grid).

    Parameters
    ----------
    c_small, c_large : float
        Per-query costs (defaults: the paper's normalized 1.0 and 3.02).
        Under ``cost_model="routing"``, ``c_large > c_small`` is required
        (Theorem 1, assumption (i)).
    cost_model : {"routing", "sequential"}
        As in :func:`ucci.policy_cost`.
    direction : {"uncertainty", "confidence"}
        Whether larger scores mean less certain (escalate above the
        threshold) or more certain (escalate below it).
    candidates : array_like of float, optional
        Candidate thresholds, in the score's own units. Default: exact.

    Attributes
    ----------
    choice : ucci.ThresholdChoice or None
        Validation result of the selected threshold, in the score's units.
    """

    def __init__(
        self,
        c_small: float = DEFAULT_COST_SMALL,
        c_large: float = DEFAULT_COST_LARGE,
        cost_model: str = "routing",
        *,
        direction: str = "uncertainty",
        candidates: ArrayLike | None = None,
    ) -> None:
        super().__init__(c_small, c_large, cost_model)
        if direction not in ("uncertainty", "confidence"):
            raise ValueError(f"direction must be 'uncertainty' or 'confidence', got {direction!r}")
        self.direction = direction
        self.candidates: NDArray[np.float64] | None = None
        if candidates is not None:
            c = as_1d(candidates, "candidates", finite=False)
            if np.isnan(c).any():
                raise ValueError("candidates contain NaN")
            self.candidates = np.unique(c)

    def _sign(self) -> float:
        return 1.0 if self.direction == "uncertainty" else -1.0

    def calibrate(
        self, scores: ArrayLike, e: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> RawThresholdRouter:
        """No-op: a raw-score threshold uses no calibration data.

        The inputs are checked for shape so a misaligned call fails early.

        Returns
        -------
        self
        """
        s = as_1d(scores, "scores")
        ee = check_labels(as_1d(e, "e"), "e")
        n = check_same_length(scores=s, e=ee)
        check_weights(sample_weight, n)
        return self

    def choose_threshold(
        self,
        scores: ArrayLike,
        small_score: ArrayLike,
        large_score: ArrayLike,
        tau: float | None = None,
        *,
        budget: float | None = None,
        metric: Metric | None = None,
    ) -> ThresholdChoice:
        """Select the threshold on the validation split (Section 4.3, Eq. 7).

        Parameters
        ----------
        scores : array_like of float, shape (n,)
            Raw score of each validation query.
        small_score, large_score : array_like of float, shape (n,)
            Per-query score of each model's actual answer.
        tau : float, optional
            Accuracy target: minimum cost subject to accuracy >= tau.
        budget : float, optional
            Cost budget: maximum accuracy subject to mean cost <= budget.
        metric : callable, optional
            ``metric(mask) -> float`` over the validation queries, replacing
            the mean per-query score (for corpus micro-F1).

        Returns
        -------
        ucci.ThresholdChoice
            ``theta`` is the threshold in the score's own units.

        Raises
        ------
        InfeasibleTargetError
            If no threshold meets tau (or the budget).
        ValueError
            For malformed inputs, or when both or neither of tau and budget
            are given.
        """
        tau_v, budget_v = _objective(tau, budget)
        s, sm, lg = _problem(scores, small_score, large_score)
        sign = self._sign()
        internal = sign * s
        if self.candidates is None:
            cands = exact_threshold_candidates(internal)
        else:
            cands = sign * self.candidates
        choice = _select_on_candidates(
            internal,
            sm,
            lg,
            cands,
            tau=tau_v,
            budget=budget_v,
            c_small=self.c_small,
            c_large=self.c_large,
            cost_model=self.cost_model,
            metric=metric,
        )
        # Report the threshold in the user's units (0 * inf never occurs).
        self.choice = dataclasses.replace(choice, theta=sign * choice.theta)
        return self.choice

    def escalate(self, scores: ArrayLike) -> NDArray[np.bool_]:
        """Boolean mask, True where the query goes to the large model."""
        s = as_1d(scores, "scores", allow_empty=True)
        t = self.theta
        if self.direction == "uncertainty":
            return np.asarray(s > t, dtype=bool)
        return np.asarray(s < t, dtype=bool)

    def __repr__(self) -> str:
        """Show the configuration."""
        return (
            f"{type(self).__name__}(c_small={self.c_small!r}, "
            f"c_large={self.c_large!r}, "
            f"cost_model={self.cost_model!r}, direction={self.direction!r})"
        )


class EntropyThresholdRouter(RawThresholdRouter):
    """Route by uncalibrated mean token entropy (Section 6.1, Table 2).

    The paper's "Entropy threshold" baseline. Scores are mean token
    entropies (:func:`mean_token_entropy`); a query is escalated when its
    entropy exceeds the threshold, and the threshold is the cheapest one
    meeting tau on the validation split (exact search over all distinct
    validation entropies, see :class:`RawThresholdRouter`).

    Parameters
    ----------
    c_small, c_large : float
        Per-query costs (defaults: the paper's normalized 1.0 and 3.02).
    cost_model : {"routing", "sequential"}
        As in :func:`ucci.policy_cost`.
    candidates : array_like of float, optional
        Restrict the search to these entropy thresholds.
    """

    def __init__(
        self,
        c_small: float = DEFAULT_COST_SMALL,
        c_large: float = DEFAULT_COST_LARGE,
        cost_model: str = "routing",
        *,
        candidates: ArrayLike | None = None,
    ) -> None:
        super().__init__(
            c_small, c_large, cost_model, direction="uncertainty", candidates=candidates
        )


class FrugalGPTStyleRouter(RawThresholdRouter):
    """Confidence threshold tuned on validation (Section 6.1, "FrugalGPT-style").

    The paper's "FrugalGPT-style" baseline: tune a confidence threshold on
    the validation set to meet the accuracy target. A query keeps the small
    model's answer when its confidence is at least the threshold and is
    escalated when ``confidence < threshold``; the threshold is the cheapest
    one meeting tau on the validation split (exact search, see
    :class:`RawThresholdRouter`).

    This is the paper's FrugalGPT-style comparator, not a reimplementation
    of FrugalGPT. FrugalGPT (Chen, Zaharia and Zou, 2023, arXiv:2305.05176)
    trains a generation scoring function g(q, a) that rates the reliability
    of an answer, returns an answer when its score exceeds a per-model
    threshold, and learns the model list and thresholds by maximizing
    quality under a cost budget. This class implements only the threshold
    rule, on whatever confidence you supply, tuned as the paper describes (to
    meet the accuracy target; the budget form is also available). The paper
    does not state which confidence it used. Any per-query score where
    larger means "more likely correct" works: the output of a learned scorer
    as in FrugalGPT, the mean top-1 token probability
    (:func:`mean_max_prob`), or ``1 - u(x)``.

    Parameters
    ----------
    c_small, c_large : float
        Per-query costs (defaults: the paper's normalized 1.0 and 3.02).
    cost_model : {"routing", "sequential"}
        As in :func:`ucci.policy_cost`.
    candidates : array_like of float, optional
        Restrict the search to these confidence thresholds.
    """

    def __init__(
        self,
        c_small: float = DEFAULT_COST_SMALL,
        c_large: float = DEFAULT_COST_LARGE,
        cost_model: str = "routing",
        *,
        candidates: ArrayLike | None = None,
    ) -> None:
        super().__init__(
            c_small, c_large, cost_model, direction="confidence", candidates=candidates
        )


# ---------------------------------------------------------------------------
# Split conformal routing (Section 6.1, "Conformal prediction")
# ---------------------------------------------------------------------------


def _delta_fraction(delta: float) -> Fraction:
    """Read a miscoverage level as the shortest decimal that round-trips.

    ``Fraction(repr(0.3)) == Fraction(3, 10)``, so ``ceil((n + 1)(1 - delta))``
    gives the integer the user intended instead of one that depends on how
    0.3 rounds in binary.
    """
    return Fraction(repr(float(delta)))


class SplitConformalRouter(_CostMixin):
    """Split conformal routing on the event "small model is correct" (Section 6.1).

    The paper's "Conformal prediction" baseline: split conformal prediction
    applied to the binary event "small model is correct", with the raw
    token-margin uncertainty u(x) (not the calibrated p_hat) as the
    nonconformity score; a threshold alpha* is chosen on the validation set
    to control miscoverage, and queries with ``u(x) > alpha*`` are
    escalated.

    Precisely:

    1. :meth:`calibrate` keeps the scores ``s_1 <= ... <= s_n`` of the
       calibration queries the small model got right (``e = 0``).
    2. For a miscoverage level delta in (0, 1), the threshold is the
       conformal quantile ``q(delta) = s_(k)`` with
       ``k = ceil((n + 1)(1 - delta))``, and ``q(delta) = +inf`` when
       ``k > n`` (:meth:`quantile`).
    3. :meth:`choose_threshold` searches the ``deltas`` grid on the
       validation split and keeps the level whose threshold meets tau at
       minimum cost (Section 4.3 selection rules), so ``alpha* = q(delta*)``.
       When several levels give the same threshold, ``delta*`` is the
       smallest of them (the strongest guarantee for that routing).
    4. :meth:`escalate` sends a query to the large model when
       ``u(x) > alpha*``.

    Coverage: for exchangeable data (for example i.i.d. queries), a new
    query that the small model answers correctly is kept with probability at
    least ``k / (n + 1) >= 1 - delta``, and at most
    ``k / (n + 1) < 1 - delta + 1 / (n + 1)`` when the scores have no ties
    (Angelopoulos and Bates, 2021, arXiv:2107.07511, Theorem 1 and
    Appendix D). The probability is over the calibration set and the new
    query, conditional on the new query being answered correctly. It holds
    for every fixed level delta; the selected delta* depends on the
    calibration quantiles through step 3, so the guarantee is not claimed
    for delta* itself. :meth:`keep_probability_bound` returns
    ``k / (n + 1)``.

    ``delta`` is read as the shortest decimal that represents the given
    float, so ``delta = 0.3`` means exactly 3/10 in the ceiling above.

    Parameters
    ----------
    c_small, c_large : float
        Per-query costs (defaults: the paper's normalized 1.0 and 3.02).
    cost_model : {"routing", "sequential"}
        As in :func:`ucci.policy_cost`.
    deltas : array_like of float, optional
        Miscoverage levels to search, each in (0, 1). Default
        :data:`DEFAULT_DELTA_GRID` (0.005 to 0.995, step 0.005).

    Attributes
    ----------
    scores_ : numpy.ndarray or None
        Sorted u(x) of the small-correct calibration queries.
    delta_ : float or None
        Selected miscoverage level delta*.
    alpha_ : float or None
        Selected threshold alpha* on u(x) (``+inf`` escalates nothing).
    """

    def __init__(
        self,
        c_small: float = DEFAULT_COST_SMALL,
        c_large: float = DEFAULT_COST_LARGE,
        cost_model: str = "routing",
        *,
        deltas: ArrayLike = DEFAULT_DELTA_GRID,
    ) -> None:
        super().__init__(c_small, c_large, cost_model)
        d = np.unique(as_1d(deltas, "deltas"))
        bad = (d <= 0.0) | (d >= 1.0)
        if bad.any():
            raise ValueError(f"deltas must lie strictly between 0 and 1; {d[bad][0]!r} does not")
        self.deltas: NDArray[np.float64] = d
        self.scores_: NDArray[np.float64] | None = None
        self.delta_: float | None = None
        self.alpha_: float | None = None

    def calibrate(
        self, scores: ArrayLike, e: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> SplitConformalRouter:
        """Store the nonconformity scores of the small-correct calibration queries.

        Parameters
        ----------
        scores : array_like of float, shape (n,)
            Raw u(x) of each calibration query (Eq. 4).
        e : array_like of {0, 1}, shape (n,)
            Error label: 1 if the small model was wrong, 0 if right. Must be
            binary: the conformal event "small model is correct" is binary.
        sample_weight : None
            Not supported: weighted conformal prediction is a different
            procedure with a different guarantee.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``e`` is not 0/1, no calibration query is small-correct, or
            ``sample_weight`` is given.
        """
        if sample_weight is not None:
            raise ValueError(
                "SplitConformalRouter does not take sample weights: weighted split "
                "conformal prediction is a different procedure"
            )
        s = as_1d(scores, "scores")
        ee = _binary_labels(as_1d(e, "e"), "e")
        check_same_length(scores=s, e=ee)
        correct = np.sort(s[ee == 0.0], kind="mergesort")
        if correct.size == 0:
            raise ValueError(
                "no calibration query was answered correctly by the small model "
                "(all e == 1); split conformal needs at least one"
            )
        self.scores_ = correct
        return self

    def _k(self, delta: float) -> int:
        if self.scores_ is None:
            raise RuntimeError("SplitConformalRouter: call calibrate() first")
        d = check_finite_scalar(delta, "delta")
        if not 0.0 < d < 1.0:
            raise ValueError(f"delta must lie strictly between 0 and 1, got {d!r}")
        n = int(self.scores_.size)
        return math.ceil((n + 1) * (1 - _delta_fraction(d)))

    def quantile(self, delta: float) -> float:
        """Conformal threshold ``q(delta)`` on u(x).

        Parameters
        ----------
        delta : float
            Miscoverage level in (0, 1).

        Returns
        -------
        float
            ``s_(k)`` with ``k = ceil((n + 1)(1 - delta))``, or ``+inf`` when
            ``k > n``.
        """
        k = self._k(delta)
        assert self.scores_ is not None
        if k > self.scores_.size:
            return math.inf
        return float(self.scores_[k - 1])

    def keep_probability_bound(self, delta: float) -> float:
        """Guaranteed lower bound ``k / (n + 1)`` on P(keep | small correct).

        Equals the exact keep probability when the scores have no ties, and
        is 1.0 when ``q(delta) = +inf``. Always >= ``1 - delta``.
        """
        k = self._k(delta)
        assert self.scores_ is not None
        n = int(self.scores_.size)
        return 1.0 if k > n else k / (n + 1)

    def choose_threshold(
        self,
        scores: ArrayLike,
        small_score: ArrayLike,
        large_score: ArrayLike,
        tau: float | None = None,
        *,
        budget: float | None = None,
        metric: Metric | None = None,
    ) -> ThresholdChoice:
        """Choose delta* on the validation split; ``theta`` is ``alpha* = q(delta*)``.

        Parameters
        ----------
        scores : array_like of float, shape (n,)
            Raw u(x) of each validation query.
        small_score, large_score : array_like of float, shape (n,)
            Per-query score of each model's actual answer.
        tau : float, optional
            Accuracy target: minimum cost subject to accuracy >= tau.
        budget : float, optional
            Cost budget: maximum accuracy subject to mean cost <= budget.
        metric : callable, optional
            ``metric(mask) -> float`` replacing the mean per-query score.

        Returns
        -------
        ucci.ThresholdChoice

        Raises
        ------
        InfeasibleTargetError
            If no level in ``deltas`` meets tau (or the budget).
        """
        tau_v, budget_v = _objective(tau, budget)
        s, sm, lg = _problem(scores, small_score, large_score)
        qs = np.array([self.quantile(float(d)) for d in self.deltas], dtype=np.float64)
        choice = _select_on_candidates(
            s,
            sm,
            lg,
            qs,
            tau=tau_v,
            budget=budget_v,
            c_small=self.c_small,
            c_large=self.c_large,
            cost_model=self.cost_model,
            metric=metric,
        )
        # q is non-increasing in delta: the first match is the smallest delta.
        self.delta_ = float(self.deltas[int(np.flatnonzero(qs == choice.theta)[0])])
        self.alpha_ = choice.theta
        self.choice = choice
        return choice

    def escalate(self, scores: ArrayLike) -> NDArray[np.bool_]:
        """Boolean mask ``u(x) > alpha*``."""
        s = as_1d(scores, "scores", allow_empty=True)
        return np.asarray(s > self.theta, dtype=bool)


# ---------------------------------------------------------------------------
# Calibrators for the calibration ablation (Appendix B.4)
# ---------------------------------------------------------------------------


class Calibrator(Protocol):
    """A map from a score to P(small model wrong), fit on calibration data.

    :class:`ucci.IsotonicCalibrator` and the calibrators below implement
    this interface.
    """

    def fit(self, u: ArrayLike, e: ArrayLike, sample_weight: ArrayLike | None = None) -> object:
        """Fit on calibration pairs ``(u_i, e_i)``."""
        ...

    def predict(self, u: ArrayLike) -> NDArray[np.float64]:
        """Calibrated error probability for each score."""
        ...


def _sigmoid(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Logistic function, accurate in both tails."""
    return np.asarray(np.exp(-np.logaddexp(0.0, -x)), dtype=np.float64)


def _fit_inputs(
    u: ArrayLike, e: ArrayLike, sample_weight: ArrayLike | None
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate calibration inputs and drop zero-weight points."""
    uu = as_1d(u, "u")
    ee = check_labels(as_1d(e, "e"), "e")
    n = check_same_length(u=uu, e=ee)
    ww = check_weights(sample_weight, n)
    keep = ww > 0.0
    return uu[keep], ee[keep], ww[keep]


def _predict_shape(u: ArrayLike, values: NDArray[np.float64]) -> float | NDArray[np.float64]:
    """Return a float for scalar non-array input, as IsotonicCalibrator does."""
    if values.ndim == 0 and not isinstance(u, np.ndarray):
        return float(values)
    return values


class IdentityCalibrator:
    """Uncalibrated routing: ``p_hat = u`` (Appendix B.4 ablation).

    The paper compares isotonic regression against "uncalibrated routing"
    under the same calibration set, validation set and threshold-selection
    procedure (Appendix B.4). Used inside :class:`CalibratedThresholdRouter`,
    this calibrator routes on raw u(x) with exactly UCCI's Eq. 6 policy and
    theta grid. Predictions are clipped to [0, 1], which leaves u(x) (always
    in [0, 1]) unchanged.
    """

    def fit(
        self, u: ArrayLike, e: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> IdentityCalibrator:
        """Check the inputs; there is nothing to fit.

        Returns
        -------
        self
        """
        _fit_inputs(u, e, sample_weight)
        return self

    @overload
    def predict(self, u: float) -> float: ...

    @overload
    def predict(self, u: ArrayLike) -> NDArray[np.float64]: ...

    def predict(self, u: ArrayLike) -> float | NDArray[np.float64]:
        """``clip(u, 0, 1)``, any shape."""
        return _predict_shape(u, np.clip(as_float_array(u, "u"), 0.0, 1.0))

    __call__ = predict

    def __repr__(self) -> str:
        """Show the configuration."""
        return "IdentityCalibrator()"


class TemperatureScalingCalibrator:
    """Temperature scaling of u(x) (Appendix B.4 ablation).

    The paper compares isotonic regression against temperature scaling
    (Guo et al., 2017, arXiv:1706.04599), described as "a single-parameter
    monotone rescaling" (Appendix B.4). The paper does not give the exact
    form; ours treats u(x) as an uncalibrated error probability and rescales
    its log-odds by one temperature:

        p_hat = sigmoid(logit(clip(u, eps, 1 - eps)) / T),  T > 0.

    The map is increasing in u for every T and fixes ``p_hat(0.5) = 0.5``
    (there is no bias term). T is fit by minimizing the weighted binary
    negative log-likelihood of the error labels on the calibration set. The
    NLL is convex in ``s = 1/T``, so the fit is a safeguarded Newton
    bisection on its derivative over ``T in [t_min, t_max]``: robust, exact
    to floating-point precision, and free of scipy. When the optimum lies
    outside the bounds, T is set to the nearer bound and a
    :class:`RuntimeWarning` is issued (``at_bound_`` is True).

    Parameters
    ----------
    eps : float, default 1e-6
        Clip level that keeps ``logit(u)`` finite at u = 0 and u = 1.
    t_min, t_max : float, default 1e-3 and 1e3
        Search bounds for T.

    Attributes
    ----------
    temperature_ : float or None
        Fitted temperature T.
    nll_ : float or None
        Weighted mean NLL at the fitted T.
    at_bound_ : bool
        True when T sits at ``t_min`` or ``t_max``.
    """

    def __init__(self, *, eps: float = 1e-6, t_min: float = 1e-3, t_max: float = 1e3) -> None:
        self.eps = check_finite_scalar(eps, "eps")
        if not 0.0 < self.eps < 0.5:
            raise ValueError(f"eps must lie in (0, 0.5), got {self.eps!r}")
        self.t_min = check_finite_scalar(t_min, "t_min")
        self.t_max = check_finite_scalar(t_max, "t_max")
        if not 0.0 < self.t_min <= self.t_max:
            raise ValueError(
                f"need 0 < t_min <= t_max, got t_min={self.t_min!r}, t_max={self.t_max!r}"
            )
        self.temperature_: float | None = None
        self.nll_: float | None = None
        self.at_bound_ = False

    def _logit(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        c = np.clip(u, self.eps, 1.0 - self.eps)
        return np.asarray(np.log(c) - np.log1p(-c), dtype=np.float64)

    @staticmethod
    def _nll(
        s: float, z: NDArray[np.float64], e: NDArray[np.float64], w: NDArray[np.float64]
    ) -> float:
        x = s * z
        return float(np.sum(w * (np.logaddexp(0.0, x) - e * x)) / np.sum(w))

    def fit(
        self, u: ArrayLike, e: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> TemperatureScalingCalibrator:
        """Fit T by minimizing the binary NLL of ``e`` given ``p_hat(u)``.

        Parameters
        ----------
        u : array_like of float, shape (n,)
            Uncertainty u(x) in [0, 1] (values outside are clipped).
        e : array_like of float, shape (n,)
            Error labels in [0, 1] (1 = small model wrong).
        sample_weight : array_like of float, shape (n,), optional
            Non-negative weights.

        Returns
        -------
        self
        """
        uu, ee, ww = _fit_inputs(u, e, sample_weight)
        z = self._logit(uu)

        def grad(s: float) -> float:
            return float(np.sum(ww * (_sigmoid(s * z) - ee) * z))

        def hess(s: float) -> float:
            p = _sigmoid(s * z)
            return float(np.sum(ww * p * (1.0 - p) * z * z))

        lo, hi = 1.0 / self.t_max, 1.0 / self.t_min
        self.at_bound_ = False
        if not (np.abs(z) * ww).any():
            s_star = min(max(1.0, lo), hi)  # NLL is flat in s: keep T = 1.
        elif grad(lo) >= 0.0:
            s_star, self.at_bound_ = lo, grad(lo) > 0.0
        elif grad(hi) <= 0.0:
            s_star, self.at_bound_ = hi, grad(hi) < 0.0
        else:
            s_star = min(max(1.0, lo), hi)
            for _ in range(200):
                g = grad(s_star)
                if g == 0.0:
                    break
                if g < 0.0:
                    lo = s_star
                else:
                    hi = s_star
                h = hess(s_star)
                step = s_star - g / h if h > 0.0 else math.nan
                nxt = step if lo < step < hi else 0.5 * (lo + hi)
                if abs(nxt - s_star) <= 4.0 * _EPS * s_star or hi - lo <= 4.0 * _EPS * hi:
                    s_star = nxt
                    break
                s_star = nxt
        if self.at_bound_:
            warnings.warn(
                f"temperature scaling optimum lies outside [t_min, t_max] = "
                f"[{self.t_min:g}, {self.t_max:g}]; T was set to {1.0 / s_star:g}. "
                "The error labels may not increase with u.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.temperature_ = 1.0 / s_star
        self.nll_ = self._nll(s_star, z, ee, ww)
        return self

    @overload
    def predict(self, u: float) -> float: ...

    @overload
    def predict(self, u: ArrayLike) -> NDArray[np.float64]: ...

    def predict(self, u: ArrayLike) -> float | NDArray[np.float64]:
        """``sigmoid(logit(clip(u)) / T)``, any shape."""
        if self.temperature_ is None:
            raise RuntimeError("TemperatureScalingCalibrator is not fitted; call fit(u, e)")
        z = self._logit(as_float_array(u, "u"))
        return _predict_shape(u, _sigmoid(z / self.temperature_))

    __call__ = predict

    def __repr__(self) -> str:
        """Show the configuration."""
        return f"TemperatureScalingCalibrator(temperature_={self.temperature_!r})"


def _separable(u: NDArray[np.float64], t: NDArray[np.float64]) -> bool:
    """Whether a cut point orders the targets perfectly (no finite logistic MLE).

    For the one-feature logistic model the maximum-likelihood fit fails to
    exist exactly when some cut c has every target 0 for u < c and 1 for
    u > c, or the reverse (targets at u = c are unrestricted): the
    likelihood then keeps increasing as the slope grows along that cut.
    Soft labels in (0, 1) are covered as well.
    """
    order = np.argsort(u, kind="mergesort")
    us, ts = u[order], t[order]
    _, first = np.unique(us, return_index=True)
    zero = np.logical_and.reduceat(ts == 0.0, first)
    one = np.logical_and.reduceat(ts == 1.0, first)

    def all_before(flags: NDArray[np.bool_]) -> NDArray[np.bool_]:
        acc = np.logical_and.accumulate(flags)
        return np.concatenate(([True], acc[:-1]))

    def all_after(flags: NDArray[np.bool_]) -> NDArray[np.bool_]:
        return all_before(flags[::-1])[::-1]

    rising = all_before(zero) & all_after(one)
    falling = all_before(one) & all_after(zero)
    return bool(rising.any() or falling.any())


class PlattCalibrator:
    """Platt (logistic) calibration ``p_hat = sigmoid(a u + b)`` (extension).

    Extension, not in the paper: some cascade implementations calibrate
    their routing score with a logistic fit (Platt, 1999), so it is offered
    for comparison under the identical protocol of
    :class:`CalibratedThresholdRouter`.

    The fit maximizes the weighted Bernoulli likelihood of the error labels,
    optionally with an L2 penalty ``0.5 * l2 * a**2`` on the slope (the
    intercept is not penalized). With ``l2 = 1 / C`` this is the exact
    objective of scikit-learn's ``LogisticRegression(C=C)`` on the single
    feature u, and ``l2 = 0`` (default) is the plain maximum-likelihood fit.
    It is solved by Newton's method (IRLS) with a backtracking line search,
    numpy only, on a standardized copy of u for conditioning.

    ``target_smoothing=True`` applies Platt's (1999) prior correction:
    labels become ``(N+ + 1) / (N+ + 2)`` for errors and ``1 / (N- + 2)`` for
    correct answers, where N+ and N- are the (weighted) counts of each
    class. This keeps the fit finite when u separates the classes.

    Parameters
    ----------
    l2 : float, default 0.0
        Slope penalty (``1 / C`` in scikit-learn terms).
    target_smoothing : bool, default False
        Use Platt's smoothed targets (binary labels only).
    max_iter : int, default 100
        Newton iterations.

    Attributes
    ----------
    a_, b_ : float or None
        Fitted slope and intercept on the original u scale.
    n_iter_ : int
        Newton iterations used by the last fit.

    Raises
    ------
    ValueError
        From :meth:`fit`, when the fit does not exist or is not unique: all
        labels are 0 or all are 1 (the intercept diverges; use
        ``target_smoothing``); with ``l2 = 0``, u is constant (the slope is
        not identifiable) or a cut point on u orders the labels perfectly
        (the slope diverges; use ``l2 > 0`` or ``target_smoothing``). The
        check is exact, so Newton's method only runs when a unique finite
        optimum exists.
    """

    def __init__(
        self, *, l2: float = 0.0, target_smoothing: bool = False, max_iter: int = 100
    ) -> None:
        self.l2 = check_finite_scalar(l2, "l2")
        if self.l2 < 0.0:
            raise ValueError(f"l2 must be non-negative, got {self.l2!r}")
        self.target_smoothing = bool(target_smoothing)
        self.max_iter = int(max_iter)
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be at least 1, got {max_iter!r}")
        self.a_: float | None = None
        self.b_: float | None = None
        self.n_iter_ = 0

    def _targets(
        self, uu: NDArray[np.float64], ee: NDArray[np.float64], ww: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Training targets, after checking that the fit exists."""
        if self.l2 == 0.0 and float(np.ptp(uu)) == 0.0:
            raise ValueError("u is constant, so the slope is not identifiable; set l2 > 0")
        if self.target_smoothing:
            if not bool(((ee == 0.0) | (ee == 1.0)).all()):
                raise ValueError("target_smoothing needs binary labels e in {0, 1}")
            n_pos = float(ww[ee == 1.0].sum())
            n_neg = float(ww[ee == 0.0].sum())
            return np.where(ee == 1.0, (n_pos + 1.0) / (n_pos + 2.0), 1.0 / (n_neg + 2.0))
        if bool((ee == 0.0).all()) or bool((ee == 1.0).all()):
            raise ValueError(
                "all labels are equal (all 0 or all 1), so the intercept of the "
                "logistic fit is infinite; use target_smoothing=True"
            )
        if self.l2 == 0.0:
            hint = "set l2 > 0 or target_smoothing=True"
            if _separable(uu, ee):
                raise ValueError(
                    "u separates the errors from the correct answers (all labels are "
                    "0 on one side of a cut point and 1 on the other), so no finite "
                    f"maximum-likelihood fit exists; {hint}"
                )
        return ee

    def fit(
        self, u: ArrayLike, e: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> PlattCalibrator:
        """Fit ``a`` and ``b`` by (penalized) maximum likelihood.

        Parameters
        ----------
        u : array_like of float, shape (n,)
            Calibration scores.
        e : array_like of float, shape (n,)
            Error labels in [0, 1] (1 = small model wrong); soft labels are
            accepted unless ``target_smoothing`` is set.
        sample_weight : array_like of float, shape (n,), optional
            Non-negative weights.

        Returns
        -------
        self
        """
        uu, ee, ww = _fit_inputs(u, e, sample_weight)
        t = self._targets(uu, ee, ww)
        mu = float(uu.mean())
        sd = float(uu.std())
        sd = sd if sd > 0.0 else 1.0
        z = (uu - mu) / sd
        pen = self.l2 / (sd * sd)  # penalty on the standardized slope

        def objective(a: float, b: float) -> float:
            x = a * z + b
            return float(np.sum(ww * (np.logaddexp(0.0, x) - t * x)) + 0.5 * pen * a * a)

        tbar = float(np.clip(np.sum(ww * t) / np.sum(ww), 1e-12, 1.0 - 1e-12))
        a, b = 0.0, math.log(tbar) - math.log1p(-tbar)
        f = objective(a, b)
        self.n_iter_ = 0
        converged = False
        for it in range(1, self.max_iter + 1):
            p = _sigmoid(a * z + b)
            r = ww * (p - t)
            g = np.array([float(np.sum(r * z)) + pen * a, float(np.sum(r))])
            v = ww * p * (1.0 - p)
            h = np.array(
                [
                    [float(np.sum(v * z * z)) + pen, float(np.sum(v * z))],
                    [float(np.sum(v * z)), float(np.sum(v))],
                ]
            )
            try:
                d = -np.linalg.solve(h, g)
            except np.linalg.LinAlgError:
                d = -g
            decrement = -float(g @ d)
            self.n_iter_ = it
            if decrement <= 1e-12 * abs(f):
                # decrement / 2 predicts the remaining decrease of f, which is
                # now below what the line search can resolve in double
                # precision. The standardized step is O(1e-6) here, deep in
                # the region where Newton converges quadratically, so the
                # full step lands on the optimum up to rounding.
                a, b = a + float(d[0]), b + float(d[1])
                converged = True
                break
            step = 1.0
            while True:
                a_new, b_new = a + step * float(d[0]), b + step * float(d[1])
                f_new = objective(a_new, b_new)
                if f_new <= f - 1e-4 * step * decrement or step < 1e-12:
                    break
                step *= 0.5
            if step < 1e-12:
                converged = True  # no further decrease possible at double precision
                break
            a, b, f = a_new, b_new, f_new
            if not (math.isfinite(a) and math.isfinite(b)) or abs(a) > 1e8:
                break
        if not (math.isfinite(a) and math.isfinite(b)) or abs(a) > 1e8 or not converged:
            hint = "set l2 > 0" if self.target_smoothing else "set l2 > 0 or target_smoothing=True"
            raise ValueError(
                f"logistic fit did not converge in max_iter={self.max_iter} Newton "
                f"iterations (the data may be nearly separable); {hint}, or raise max_iter"
            )
        self.a_ = a / sd
        self.b_ = b - a * mu / sd
        return self

    @overload
    def predict(self, u: float) -> float: ...

    @overload
    def predict(self, u: ArrayLike) -> NDArray[np.float64]: ...

    def predict(self, u: ArrayLike) -> float | NDArray[np.float64]:
        """``sigmoid(a u + b)``, any shape."""
        if self.a_ is None or self.b_ is None:
            raise RuntimeError("PlattCalibrator is not fitted; call fit(u, e)")
        x = as_float_array(u, "u")
        return _predict_shape(u, _sigmoid(self.a_ * x + self.b_))

    __call__ = predict

    def __repr__(self) -> str:
        """Show the configuration."""
        return f"PlattCalibrator(a_={self.a_!r}, b_={self.b_!r}, l2={self.l2!r})"


class CalibratedThresholdRouter(_CostMixin):
    """UCCI's threshold policy on top of any calibrator (Appendix B.4 protocol).

    Runs the same three steps as :class:`ucci.UCCIRouter` with the
    calibrator swapped out: fit ``calibrator`` on the calibration split,
    choose theta on the validation split by the Section 4.3 selection
    (Eq. 7) over the same grid (:data:`ucci.DEFAULT_GRID`, resolution
    0.005), and escalate when ``p_hat > theta`` (Eq. 6). This is how the
    paper compares calibration methods "under the same calibration set,
    validation set, and threshold-selection procedure" (Appendix B.4).

    With :class:`ucci.IsotonicCalibrator` this is UCCI itself: the
    selection calls :func:`ucci.select_threshold` and
    :func:`ucci.select_threshold_for_budget` on the same grid, so its choice
    is identical to :meth:`ucci.UCCIRouter.choose_threshold` and
    :meth:`ucci.UCCIRouter.choose_threshold_for_budget`.

    Parameters
    ----------
    calibrator : Calibrator
        Any object with ``fit(u, e, sample_weight=None)`` and
        ``predict(u) -> array`` returning probabilities in [0, 1]:
        :class:`ucci.IsotonicCalibrator`, :class:`TemperatureScalingCalibrator`,
        :class:`PlattCalibrator` or :class:`IdentityCalibrator`.
    c_small, c_large : float
        Per-query costs (defaults: the paper's normalized 1.0 and 3.02).
    cost_model : {"routing", "sequential"}
        As in :func:`ucci.policy_cost`.
    grid : array_like of float, optional
        Theta grid in [0, 1]. Default :data:`ucci.DEFAULT_GRID`.
    """

    def __init__(
        self,
        calibrator: Calibrator,
        c_small: float = DEFAULT_COST_SMALL,
        c_large: float = DEFAULT_COST_LARGE,
        cost_model: str = "routing",
        *,
        grid: ArrayLike = DEFAULT_GRID,
    ) -> None:
        super().__init__(c_small, c_large, cost_model)
        self.calibrator = calibrator
        self.grid: NDArray[np.float64] = check_grid(grid)

    def calibrate(
        self, scores: ArrayLike, e: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> CalibratedThresholdRouter:
        """Fit the calibrator on ``(scores, e)`` from the calibration split.

        Returns
        -------
        self
        """
        if sample_weight is None:
            self.calibrator.fit(scores, e)
        else:
            self.calibrator.fit(scores, e, sample_weight)
        return self

    def error_probability(self, scores: ArrayLike) -> NDArray[np.float64]:
        """Calibrated forecast ``p_hat = g(score)`` for each query."""
        s = as_1d(scores, "scores", allow_empty=True)
        p = as_float_array(self.calibrator.predict(s), "calibrated probabilities")
        if p.shape != s.shape:
            raise ValueError(f"calibrator.predict returned shape {p.shape}, expected {s.shape}")
        bad = (p < 0.0) | (p > 1.0)
        if bad.any():
            raise ValueError(f"calibrator.predict returned a value outside [0, 1]: {p[bad][0]!r}")
        return p

    def choose_threshold(
        self,
        scores: ArrayLike,
        small_score: ArrayLike,
        large_score: ArrayLike,
        tau: float | None = None,
        *,
        budget: float | None = None,
        metric: Metric | None = None,
    ) -> ThresholdChoice:
        """Select theta on the validation split (Section 4.3, Eq. 7).

        Parameters
        ----------
        scores : array_like of float, shape (n,)
            Uncalibrated score of each validation query (u(x) for UCCI).
        small_score, large_score : array_like of float, shape (n,)
            Per-query score of each model's actual answer.
        tau : float, optional
            Accuracy target: minimum cost subject to accuracy >= tau.
        budget : float, optional
            Cost budget: maximum accuracy subject to mean cost <= budget.
        metric : callable, optional
            ``metric(mask) -> float`` replacing the mean per-query score.

        Returns
        -------
        ucci.ThresholdChoice

        Raises
        ------
        InfeasibleTargetError
            If no theta on the grid meets tau (or the budget).
        """
        tau_v, budget_v = _objective(tau, budget)
        s, sm, lg = _problem(scores, small_score, large_score)
        p = self.error_probability(s)
        if tau_v is not None:
            self.choice = select_threshold(
                p,
                sm,
                lg,
                tau_v,
                self.c_small,
                self.c_large,
                self.grid,
                metric,
                self.cost_model,
            )
        else:
            assert budget_v is not None
            self.choice = select_threshold_for_budget(
                p,
                sm,
                lg,
                budget_v,
                self.c_small,
                self.c_large,
                self.grid,
                metric,
                self.cost_model,
            )
        return self.choice

    def escalate(self, scores: ArrayLike) -> NDArray[np.bool_]:
        """Boolean mask ``p_hat > theta`` (Eq. 6)."""
        return np.asarray(escalate(self.error_probability(scores), self.theta), dtype=bool)

    def __repr__(self) -> str:
        """Show the configuration."""
        return (
            f"CalibratedThresholdRouter({self.calibrator!r}, c_small={self.c_small!r}, "
            f"c_large={self.c_large!r}, cost_model={self.cost_model!r})"
        )


class _UCCIRouterAdapter:
    """Run :class:`ucci.UCCIRouter` itself under the :class:`RouterLike` interface.

    Used for the "UCCI" row of :func:`compare_routers`, so that row is the
    core router, not a re-implementation.
    """

    def __init__(self, c_small: float, c_large: float, cost_model: str) -> None:
        self.router = UCCIRouter(c_small, c_large, cost_model)

    @property
    def theta(self) -> float:
        """The selected threshold of the wrapped router."""
        return self.router.theta

    def calibrate(
        self, scores: ArrayLike, e: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> _UCCIRouterAdapter:
        """Fit the isotonic map (:meth:`ucci.UCCIRouter.calibrate`)."""
        self.router.calibrate(scores, e, sample_weight)
        return self

    def choose_threshold(
        self,
        scores: ArrayLike,
        small_score: ArrayLike,
        large_score: ArrayLike,
        tau: float | None = None,
        *,
        budget: float | None = None,
        metric: Metric | None = None,
    ) -> ThresholdChoice:
        """Select theta with the router's own accuracy-target or budget method."""
        tau_v, budget_v = _objective(tau, budget)
        if tau_v is not None:
            return self.router.choose_threshold(
                scores, small_score, large_score, tau_v, metric=metric
            )
        assert budget_v is not None
        return self.router.choose_threshold_for_budget(
            scores, small_score, large_score, budget_v, metric=metric
        )

    def escalate(self, scores: ArrayLike) -> NDArray[np.bool_]:
        """Boolean mask ``p_hat > theta`` (Eq. 6)."""
        s = as_1d(scores, "scores", allow_empty=True)
        return np.asarray(self.router.escalate(s), dtype=bool)


# ---------------------------------------------------------------------------
# The comparison protocol (Section 6.1, Table 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SplitData:
    """One data split for :func:`compare_routers`.

    Parameters
    ----------
    signals : mapping of str to array_like
        Per-query routing signals by name. The built-in methods read
        ``"u"`` (token-margin uncertainty, Eq. 4), ``"entropy"`` (mean token
        entropy), ``"max_prob"`` (mean top-1 probability) and
        ``"confidence"`` (any confidence score for the FrugalGPT-style
        baseline); these match the JSONL record fields.
    small_score, large_score : array_like of float
        Per-query score of each model's actual answer (0/1 correctness or a
        per-query metric such as per-query F1).
    small_error : array_like of {0, 1}, optional
        The calibration event e(x) (Section 4.2: 1 when the small model's
        output is wrong, by exact match in the paper). Only read on the
        calibration split. Default: ``small_score < 1``, which for 0/1
        scores is ``1 - small_score`` and for per-query F1 means "not every
        field right".
    metric : callable, optional
        ``metric(mask) -> float`` for this split, replacing the mean
        per-query score (for corpus micro-F1 over the returned answers).
    """

    signals: Mapping[str, ArrayLike]
    small_score: ArrayLike
    large_score: ArrayLike
    small_error: ArrayLike | None = None
    metric: Metric | None = None

    def error(self) -> NDArray[np.float64]:
        """Return the binary calibration label e(x) for this split."""
        if self.small_error is not None:
            return check_labels(as_1d(self.small_error, "small_error"), "small_error")
        return (as_1d(self.small_score, "small_score") < 1.0).astype(np.float64)

    def signal(self, name: str) -> NDArray[np.float64]:
        """Return the named signal as a validated 1-D array."""
        if name not in self.signals:
            raise KeyError(f"signal {name!r} not found; available: {sorted(self.signals)}")
        return as_1d(self.signals[name], f"signals[{name!r}]")


@dataclass(frozen=True)
class MethodSpec:
    """A routing method for :func:`compare_routers`.

    Parameters
    ----------
    name : str
        Row label.
    signal : str or None
        Key of :attr:`SplitData.signals` the method routes on; None for
        methods that ignore signals (always-small, always-large).
    factory : callable
        ``factory(c_small, c_large, cost_model)`` returning a fresh router
        with the :class:`RouterLike` interface.
    transform : callable, optional
        Applied to the signal on every split before the router sees it,
        for example ``lambda x: 1 - x`` to turn a confidence into an
        uncertainty.
    note : str
        Free-text description carried into the result row.
    """

    name: str
    signal: str | None
    factory: Callable[[float, float, str], RouterLike]
    transform: Callable[[NDArray[np.float64]], NDArray[np.float64]] | None = None
    note: str = ""


@dataclass(frozen=True)
class ComparisonRow:
    """One result row of :func:`compare_routers`.

    Attributes
    ----------
    method : str
        Method name.
    signal : str or None
        Signal the method routed on.
    threshold : float
        Selected threshold in the method's own units (theta on p_hat for
        calibrated routers, the raw score for raw thresholds, alpha* on u
        for conformal); ``+inf``/``-inf`` for the single-model anchors, NaN
        for the oracle or when no threshold was feasible.
    cost : float
        Mean per-query cost on the test split.
    accuracy : float
        Accuracy on the test split (mean per-query score or ``metric``).
    escalation_rate : float
        Fraction of test queries sent to the large model.
    delta_vs_target : float or None
        ``accuracy - tau`` (the "Delta F1 vs target" column of Table 2), or
        None when no tau was given.
    val_cost, val_accuracy : float
        Cost and accuracy of the selected operating point on the validation
        split (NaN for the oracle, which is not selected on validation).
    feasible_on_val : bool
        Whether the operating point meets the constraint (accuracy >= tau,
        or cost <= budget) on the validation split. Selected methods that
        cannot meet it have NaN test numbers; the single-model anchors are
        always reported. The oracle is computed on the test split, so for it
        this flag means the constraint is met there.
    note : str
        Explanation, for example why a row is infeasible.
    params : mapping of str to float
        Fitted parameters worth reporting (delta*, temperature, Platt a and
        b).
    """

    method: str
    signal: str | None
    threshold: float
    cost: float
    accuracy: float
    escalation_rate: float
    delta_vs_target: float | None
    val_cost: float
    val_accuracy: float
    feasible_on_val: bool
    note: str = ""
    params: Mapping[str, float] = field(default_factory=dict)


def _one_minus(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.asarray(1.0 - x, dtype=np.float64)


def table2_methods(confidence_signal: str = "confidence") -> list[MethodSpec]:
    """Return the methods of the paper's Table 2, in the table's row order.

    UCCI, conformal prediction, FrugalGPT-style, entropy threshold,
    large-only and small-only (Section 6.1). The UCCI row runs
    :class:`ucci.UCCIRouter` itself.

    Parameters
    ----------
    confidence_signal : str, default "confidence"
        Signal key the FrugalGPT-style baseline reads.

    Returns
    -------
    list of MethodSpec
    """
    return [
        MethodSpec(
            "UCCI",
            "u",
            lambda cs, cl, cm: _UCCIRouterAdapter(cs, cl, cm),
            note="ucci.UCCIRouter: isotonic calibration of u, Eq. 6 threshold",
        ),
        MethodSpec(
            "Conformal prediction",
            "u",
            lambda cs, cl, cm: SplitConformalRouter(cs, cl, cm),
            note="split conformal on 'small correct', raw u as score",
        ),
        MethodSpec(
            "FrugalGPT-style",
            confidence_signal,
            lambda cs, cl, cm: FrugalGPTStyleRouter(cs, cl, cm),
            note="confidence threshold tuned on validation",
        ),
        MethodSpec(
            "Entropy threshold",
            "entropy",
            lambda cs, cl, cm: EntropyThresholdRouter(cs, cl, cm),
            note="uncalibrated mean token entropy",
        ),
        MethodSpec("Large-only", None, lambda cs, cl, cm: AlwaysLarge(cs, cl, cm)),
        MethodSpec("Small-only", None, lambda cs, cl, cm: AlwaysSmall(cs, cl, cm)),
    ]


def ablation_methods() -> list[MethodSpec]:
    """Ablations of Section 6.3 and Appendix B.4, plus the Platt extension.

    Calibration method (Appendix B.4), each under the identical Eq. 6 grid
    protocol: temperature scaling, uncalibrated u, and Platt scaling
    (extension, not in the paper). Uncertainty signal (Section 6.3): the
    UCCI pipeline (isotonic, Eq. 6) on mean token entropy and on
    ``1 - mean max probability``.

    Returns
    -------
    list of MethodSpec
    """
    return [
        MethodSpec(
            "Temperature scaling",
            "u",
            lambda cs, cl, cm: CalibratedThresholdRouter(
                TemperatureScalingCalibrator(), cs, cl, cm
            ),
            note="Appendix B.4 ablation",
        ),
        MethodSpec(
            "Uncalibrated u",
            "u",
            lambda cs, cl, cm: CalibratedThresholdRouter(IdentityCalibrator(), cs, cl, cm),
            note="Appendix B.4 ablation: p_hat = u",
        ),
        MethodSpec(
            "Platt scaling (extension)",
            "u",
            lambda cs, cl, cm: CalibratedThresholdRouter(
                PlattCalibrator(target_smoothing=True), cs, cl, cm
            ),
            note="extension, not in the paper",
        ),
        MethodSpec(
            "Isotonic on entropy",
            "entropy",
            lambda cs, cl, cm: CalibratedThresholdRouter(IsotonicCalibrator(), cs, cl, cm),
            note="Section 6.3 signal ablation",
        ),
        MethodSpec(
            "Isotonic on max prob",
            "max_prob",
            lambda cs, cl, cm: CalibratedThresholdRouter(IsotonicCalibrator(), cs, cl, cm),
            transform=_one_minus,
            note="Section 6.3 signal ablation: 1 - mean max probability",
        ),
    ]


def _router_params(router: object) -> dict[str, float]:
    """Fitted parameters worth reporting in a result row."""
    if isinstance(router, SplitConformalRouter) and router.delta_ is not None:
        return {"delta": router.delta_}
    if isinstance(router, CalibratedThresholdRouter):
        cal = router.calibrator
        if isinstance(cal, TemperatureScalingCalibrator) and cal.temperature_ is not None:
            return {"temperature": cal.temperature_}
        if isinstance(cal, PlattCalibrator) and cal.a_ is not None and cal.b_ is not None:
            return {"a": cal.a_, "b": cal.b_}
    return {}


def _nan_row(name: str, signal: str | None, note: str, has_tau: bool) -> ComparisonRow:
    nan = math.nan
    return ComparisonRow(
        method=name,
        signal=signal,
        threshold=nan,
        cost=nan,
        accuracy=nan,
        escalation_rate=nan,
        delta_vs_target=nan if has_tau else None,
        val_cost=nan,
        val_accuracy=nan,
        feasible_on_val=False,
        note=note,
    )


def compare_routers(
    cal: SplitData,
    val: SplitData,
    test: SplitData,
    *,
    tau: float | None = None,
    budget: float | None = None,
    c_small: float = DEFAULT_COST_SMALL,
    c_large: float = DEFAULT_COST_LARGE,
    cost_model: str = "routing",
    methods: Sequence[MethodSpec] | None = None,
    include_ablations: bool = False,
    include_oracle: bool = True,
) -> list[ComparisonRow]:
    """Run the paper's three-step evaluation protocol for several routers.

    For every method (Section 6.1, "Cascade evaluation methodology"):

    1. fit it on the calibration split (``router.calibrate``);
    2. select its operating point on the validation split
       (``router.choose_threshold``), where both models have been run;
    3. route every test query end to end with the selected rule, taking the
       actual output and cost of the model it chooses, and report test cost,
       accuracy and escalation rate.

    Two forms, matching Table 2's two blocks:

    * ``tau`` only: each method meets the accuracy target tau at minimum
      validation cost (top block, "F1 = 0.91 operating point");
    * ``budget``: each method maximizes validation accuracy at mean cost <=
      budget (bottom block, "matched cost budget"). If ``tau`` is also
      given, it is only the reference for ``delta_vs_target``.

    The single-model anchors (Large-only, Small-only) are always reported.
    With ``include_oracle`` a label-dependent :class:`Oracle` row, computed
    directly on the test labels, is appended as a cost lower bound (for
    analysis only; not a deployable method). Everything is deterministic.

    Parameters
    ----------
    cal, val, test : SplitData
        Disjoint calibration, validation and test splits (the paper uses
        30% / 20% / 50%).
    tau : float, optional
        Accuracy target.
    budget : float, optional
        Mean cost budget.
    c_small, c_large : float
        Per-query costs (defaults: the paper's normalized 1.0 and 3.02).
    cost_model : {"routing", "sequential"}
        As in :func:`ucci.policy_cost`.
    methods : sequence of MethodSpec, optional
        Methods to run, in row order. Default: :func:`table2_methods`
        (plus :func:`ablation_methods` if ``include_ablations``), keeping
        only methods whose signal is present in all three splits. The
        FrugalGPT-style baseline reads ``"confidence"`` if present, else
        ``"max_prob"``; the row's ``signal`` field records which. Explicit
        methods must have their signals present.
    include_ablations : bool, default False
        Add the ablation methods to the default list.
    include_oracle : bool, default True
        Append the oracle row.

    Returns
    -------
    list of ComparisonRow
        One row per method, in order, then the oracle.

    Raises
    ------
    ValueError
        If neither tau nor budget is given, the costs are invalid, or an
        explicitly requested method's signal is missing.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> def split(n):
    ...     u = rng.random(n)
    ...     small = (rng.random(n) > u).astype(float)
    ...     return SplitData({"u": u}, small, np.ones(n))
    >>> rows = compare_routers(split(300), split(200), split(500), tau=0.9)
    >>> [r.method for r in rows]
    ['UCCI', 'Conformal prediction', 'Large-only', 'Small-only', 'Oracle']
    """
    if tau is None and budget is None:
        raise ValueError("pass tau (accuracy target) or budget (matched cost budget)")
    tau_v = check_finite_scalar(tau, "tau") if tau is not None else None
    budget_v = check_finite_scalar(budget, "budget") if budget is not None else None
    cs, cl, cm = _check_router_costs(c_small, c_large, cost_model)
    splits = (cal, val, test)
    val_small = as_1d(val.small_score, "val.small_score")
    val_large = as_1d(val.large_score, "val.large_score")
    test_small = as_1d(test.small_score, "test.small_score")
    test_large = as_1d(test.large_score, "test.large_score")
    check_same_length(val_small_score=val_small, val_large_score=val_large)
    check_same_length(test_small_score=test_small, test_large_score=test_large)

    def present(sig: str | None) -> bool:
        return sig is None or all(sig in sp.signals for sp in splits)

    if methods is None:
        conf = "confidence" if present("confidence") else "max_prob"
        specs = table2_methods(conf)
        if include_ablations:
            specs = specs[:4] + ablation_methods() + specs[4:]
        specs = [sp for sp in specs if present(sp.signal)]
    else:
        specs = list(methods)
        for sp in specs:
            if not present(sp.signal):
                raise ValueError(
                    f"method {sp.name!r} needs signal {sp.signal!r}, which is missing "
                    "from at least one split"
                )

    select_tau = tau_v if budget_v is None else None
    rows: list[ComparisonRow] = []
    for spec in specs:
        router = spec.factory(cs, cl, cm)
        if spec.signal is None:
            # Signal-free anchors only use the number of queries.
            x_val, x_test = np.zeros(val_small.shape[0]), np.zeros(test_small.shape[0])
        else:
            x_cal, x_val, x_test = (sp.signal(spec.signal) for sp in splits)
            if spec.transform is not None:
                x_cal, x_val, x_test = (
                    as_1d(spec.transform(x), f"transformed {spec.signal!r}")
                    for x in (x_cal, x_val, x_test)
                )
            router.calibrate(x_cal, cal.error())
        try:
            val_choice = router.choose_threshold(
                x_val,
                val_small,
                val_large,
                select_tau,
                budget=budget_v,
                metric=val.metric,
            )
        except InfeasibleTargetError as exc:
            rows.append(
                _nan_row(
                    spec.name,
                    spec.signal,
                    f"infeasible on validation: {exc}",
                    tau_v is not None,
                )
            )
            continue
        if budget_v is not None:
            ok = val_choice.cost <= budget_v
        else:
            assert tau_v is not None
            ok = val_choice.accuracy >= tau_v
        mask = np.asarray(router.escalate(x_test), dtype=bool)
        res = _evaluate_mask(
            mask, test_small, test_large, val_choice.theta, cs, cl, cm, test.metric
        )
        rows.append(
            ComparisonRow(
                method=spec.name,
                signal=spec.signal,
                threshold=val_choice.theta,
                cost=res.cost,
                accuracy=res.accuracy,
                escalation_rate=res.escalation_rate,
                delta_vs_target=None if tau_v is None else res.accuracy - tau_v,
                val_cost=val_choice.cost,
                val_accuracy=val_choice.accuracy,
                feasible_on_val=bool(ok),
                note=spec.note,
                params=_router_params(router),
            )
        )

    if include_oracle:
        name = "Oracle"
        note = "label-dependent lower bound on cost, analysis only"
        try:
            res = Oracle(cs, cl, cm).evaluate(
                test_small, test_large, select_tau, budget=budget_v, metric=test.metric
            )
        except InfeasibleTargetError as exc:
            rows.append(
                _nan_row(name, None, f"{note}; infeasible on test: {exc}", tau_v is not None)
            )
        else:
            rows.append(
                ComparisonRow(
                    method=name,
                    signal=None,
                    threshold=math.nan,
                    cost=res.cost,
                    accuracy=res.accuracy,
                    escalation_rate=res.escalation_rate,
                    delta_vs_target=None if tau_v is None else res.accuracy - tau_v,
                    val_cost=math.nan,
                    val_accuracy=math.nan,
                    feasible_on_val=True,
                    note=note,
                )
            )
    return rows


def format_comparison(rows: Sequence[ComparisonRow], digits: int = 3) -> str:
    """Render comparison rows as a Markdown table (Table 2 layout).

    Parameters
    ----------
    rows : sequence of ComparisonRow
        Output of :func:`compare_routers`.
    digits : int, default 3
        Decimal places.

    Returns
    -------
    str
    """

    def fmt(x: float | None, signed: bool = False) -> str:
        if x is None:
            return ""
        if math.isnan(x):
            return "n/a"
        if math.isinf(x):
            return "+inf" if x > 0 else "-inf"
        return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"

    head = "| Method | Signal | Threshold | Cost | Accuracy | Escalated | Accuracy vs target |"
    lines = [head, "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {r.method} | {r.signal or ''} | {fmt(r.threshold)} | {fmt(r.cost)} "
            f"| {fmt(r.accuracy)} | {fmt(r.escalation_rate)} "
            f"| {fmt(r.delta_vs_target, signed=True)} |"
        )
    return "\n".join(lines)
