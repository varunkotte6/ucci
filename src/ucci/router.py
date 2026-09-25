"""UCCI router: calibrate, choose the threshold, then route (Sections 4 and 6.1).

:class:`UCCIRouter` packages the three steps of UCCI and the evaluation
protocol of Section 6.1:

1. :meth:`UCCIRouter.calibrate` fits the isotonic map g on the calibration
   split (Section 4.2).
2. :meth:`UCCIRouter.choose_threshold` selects theta* on the validation
   split, where both models have been run (Section 4.3, Eq. 7).
   :meth:`UCCIRouter.choose_threshold_for_budget` is the matched-budget
   variant (Table 2, bottom block).
3. :meth:`UCCIRouter.route` / :meth:`UCCIRouter.escalate` apply pi_theta*
   (Eq. 6) to new queries; :meth:`UCCIRouter.evaluate` does so on a labelled
   test split and reports the actual cost and accuracy.

The paper uses disjoint calibration, validation and test splits of 30%, 20%
and 50% (Section 6.1). A router is saved to and loaded from the JSON format
described in :mod:`ucci.io`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NamedTuple, overload

import numpy as np

from . import io as _io
from ._validation import (
    as_float_array,
    check_cost_model,
    check_costs,
    check_finite_scalar,
)
from .calibration import IsotonicCalibrator
from .policy import (
    DEFAULT_COST_LARGE,
    DEFAULT_COST_SMALL,
    DEFAULT_GRID_STEP,
    ThresholdChoice,
    escalate,
    evaluate,
    make_grid,
    select_threshold,
    select_threshold_for_budget,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike, NDArray

    from .io import PathLike
    from .policy import Metric

__all__ = ["RouteResult", "UCCIRouter"]


class RouteResult(NamedTuple):
    """Routing decisions for a batch of queries.

    Attributes
    ----------
    escalate : numpy.ndarray of bool
        True where the query goes to the large model (``p_hat > theta``).
    p_hat : numpy.ndarray of float64
        Calibrated error probability of each query.
    """

    escalate: NDArray[np.bool_]
    p_hat: NDArray[np.float64]


class UCCIRouter:
    """Small-to-large cascade router with a calibrated threshold.

    Parameters
    ----------
    c_small, c_large : float, default 1.0 and 3.02
        Per-query cost of the small and large model. The defaults are the
        paper's normalized costs (Section 6.1). They set the reported costs
        and the budget scale; with a fixed accuracy target they do not change
        the selected threshold.
    cost_model : {"routing", "sequential"}, default "routing"
        See :func:`ucci.policy.policy_cost`.
    grid_step : float, default 0.005
        Resolution of the threshold grid used by :meth:`choose_threshold`
        and :meth:`choose_threshold_for_budget` when no grid is passed.

    Attributes
    ----------
    calibrator : IsotonicCalibrator
        The map g from u(x) to P(small model wrong).
    choice : ThresholdChoice or None
        Validation cost and accuracy of the last threshold selection; None
        after loading or after setting ``theta`` by hand.
    tau : float or None
        Accuracy target of the last :meth:`choose_threshold` call.
    budget : float or None
        Cost budget of the last :meth:`choose_threshold_for_budget` call.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> u = rng.random(3000)
    >>> small_ok = (rng.random(3000) > u**2).astype(float)
    >>> large_ok = np.ones(3000)
    >>> router = UCCIRouter().calibrate(u[:1000], 1 - small_ok[:1000])
    >>> choice = router.choose_threshold(u[1000:2000], small_ok[1000:2000],
    ...                                  large_ok[1000:2000], tau=0.9)
    >>> decisions = router.route(u[2000:])
    >>> bool(decisions.escalate.mean() < 1.0)
    True
    """

    def __init__(
        self,
        c_small: float = DEFAULT_COST_SMALL,
        c_large: float = DEFAULT_COST_LARGE,
        cost_model: str = "routing",
        grid_step: float = DEFAULT_GRID_STEP,
    ) -> None:
        self.c_small, self.c_large = check_costs(c_small, c_large)
        self.cost_model = check_cost_model(cost_model)
        make_grid(grid_step)  # validates the step
        self.grid_step = float(grid_step)
        self.calibrator = IsotonicCalibrator()
        self.choice: ThresholdChoice | None = None
        self.tau: float | None = None
        self.budget: float | None = None
        self._theta: float | None = None

    # ------------------------------------------------------------------ state

    @property
    def is_calibrated(self) -> bool:
        """True once the calibrator has been fitted or loaded."""
        return self.calibrator.is_fitted

    @property
    def has_threshold(self) -> bool:
        """True once a threshold has been chosen, loaded or set."""
        return self._theta is not None

    @property
    def theta(self) -> float:
        """The escalation threshold theta* (Eq. 6).

        Setting it by hand clears ``choice``, ``tau`` and ``budget``, since
        they no longer describe the threshold.

        Raises
        ------
        RuntimeError
            On read, if no threshold has been chosen or set.
        """
        if self._theta is None:
            raise RuntimeError(
                "no threshold yet; call choose_threshold() or "
                "choose_threshold_for_budget(), or set theta"
            )
        return self._theta

    @theta.setter
    def theta(self, value: float) -> None:
        self._theta = check_finite_scalar(value, "theta")
        self.choice = None
        self.tau = None
        self.budget = None

    def _grid(self, grid: ArrayLike | None) -> ArrayLike:
        return make_grid(self.grid_step) if grid is None else grid

    # --------------------------------------------------------------- fitting

    def calibrate(
        self, u_cal: ArrayLike, e_cal: ArrayLike, sample_weight: ArrayLike | None = None
    ) -> UCCIRouter:
        """Fit g on the calibration split (Section 4.2; step 1 of Section 6.1).

        Parameters
        ----------
        u_cal : array_like of float, shape (n,)
            u(x) of each calibration query.
        e_cal : array_like of float, shape (n,)
            1 where the small model's output was wrong, 0 where it was right
            (soft labels in [0, 1] are accepted).
        sample_weight : array_like of float, shape (n,), optional
            Non-negative weights.

        Returns
        -------
        UCCIRouter
            ``self``.
        """
        self.calibrator.fit(u_cal, e_cal, sample_weight)
        return self

    def _p_hat_1d(self, u: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(self.calibrator.predict(as_float_array(u, "u")), dtype=np.float64)

    def choose_threshold(
        self,
        u_val: ArrayLike,
        small_score: ArrayLike | None,
        large_score: ArrayLike | None,
        tau: float,
        grid: ArrayLike | None = None,
        metric: Metric | None = None,
    ) -> ThresholdChoice:
        """Choose theta* on the validation split (Eq. 7; step 2 of Section 6.1).

        Parameters
        ----------
        u_val : array_like of float, shape (n,)
            u(x) of each validation query.
        small_score, large_score : array_like of float, shape (n,), or None
            Per-query scores of both models' actual outputs on the
            validation split; may be None when ``metric`` is given.
        tau : float
            Accuracy target.
        grid : array_like of float, optional
            Candidate thresholds; defaults to the router's grid
            (``grid_step``). The saved ``grid_step`` field always records the
            router's configured step.
        metric : callable, optional
            Corpus-level metric of the routed answers; see
            :func:`ucci.policy.select_threshold`.

        Returns
        -------
        ThresholdChoice
            Also stored in ``self.choice``; the threshold becomes ``theta``.

        Raises
        ------
        RuntimeError
            If the router is not calibrated.
        ucci.policy.InfeasibleTargetError
            If no grid threshold reaches ``tau``.
        """
        choice = select_threshold(
            self._p_hat_1d(u_val),
            small_score,
            large_score,
            tau,
            self.c_small,
            self.c_large,
            self._grid(grid),
            metric,
            self.cost_model,
        )
        self._theta = choice.theta
        self.choice = choice
        self.tau = float(tau)
        self.budget = None
        return choice

    def choose_threshold_for_budget(
        self,
        u_val: ArrayLike,
        small_score: ArrayLike | None,
        large_score: ArrayLike | None,
        budget: float,
        grid: ArrayLike | None = None,
        metric: Metric | None = None,
    ) -> ThresholdChoice:
        """Choose the most accurate theta within a cost budget (Table 2, bottom block).

        Parameters are as in :meth:`choose_threshold`, with ``budget`` (mean
        per-query cost in the units of ``c_small`` and ``c_large``) in place
        of ``tau``. The saved router has ``tau = null``.

        Raises
        ------
        RuntimeError
            If the router is not calibrated.
        ucci.policy.InfeasibleTargetError
            If every grid threshold costs more than ``budget``.
        """
        choice = select_threshold_for_budget(
            self._p_hat_1d(u_val),
            small_score,
            large_score,
            budget,
            self.c_small,
            self.c_large,
            self._grid(grid),
            metric,
            self.cost_model,
        )
        self._theta = choice.theta
        self.choice = choice
        self.tau = None
        self.budget = float(budget)
        return choice

    # --------------------------------------------------------------- routing

    @overload
    def error_probability(self, u: float) -> float: ...

    @overload
    def error_probability(self, u: ArrayLike) -> NDArray[np.float64]: ...

    def error_probability(self, u: ArrayLike) -> float | NDArray[np.float64]:
        """Calibrated forecast p_hat(x) = g(u(x)) (Section 4.3).

        A float for a scalar input, otherwise an array of the input's shape.
        """
        return self.calibrator.predict(u)

    @overload
    def escalate(self, u: float) -> bool: ...

    @overload
    def escalate(self, u: ArrayLike) -> NDArray[np.bool_]: ...

    def escalate(self, u: ArrayLike) -> bool | NDArray[np.bool_]:
        """True where the query should go to the large model (Eq. 6).

        A bool for a scalar input, otherwise a mask of the input's shape.

        Raises
        ------
        RuntimeError
            If the router is not calibrated or has no threshold.
        """
        theta = self.theta
        return escalate(self.calibrator.predict(u), theta)

    def route(self, u: ArrayLike) -> RouteResult:
        """Routing decisions and calibrated probabilities for a batch.

        Parameters
        ----------
        u : array_like of float
            u(x) of each query, any shape.

        Returns
        -------
        RouteResult
            ``(escalate, p_hat)``, arrays of the input's shape.
        """
        theta = self.theta
        p_hat = self._p_hat_1d(u)
        return RouteResult(escalate=np.asarray(p_hat > theta, dtype=bool), p_hat=p_hat)

    def evaluate(
        self,
        u_test: ArrayLike,
        small_score: ArrayLike | None,
        large_score: ArrayLike | None,
        metric: Metric | None = None,
    ) -> ThresholdChoice:
        """Route a labelled test split end to end (step 3 of Section 6.1).

        Returns the actual cost, accuracy and escalation rate of pi_theta* on
        the given queries. See :func:`ucci.policy.evaluate`.
        """
        return evaluate(
            self._p_hat_1d(u_test),
            small_score,
            large_score,
            self.theta,
            self.c_small,
            self.c_large,
            metric,
            self.cost_model,
        )

    # --------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, Any]:
        """The router as a ``ucci-router`` version 1 document (see :mod:`ucci.io`).

        Raises
        ------
        RuntimeError
            If the router is not calibrated or has no threshold.
        """
        if not self.is_calibrated:
            raise RuntimeError("cannot serialize an uncalibrated router; call calibrate() first")
        return {
            "format": _io.FORMAT_NAME,
            "version": _io.FORMAT_VERSION,
            "calibrator": self.calibrator.to_dict(),
            "theta": self.theta,
            "c_small": self.c_small,
            "c_large": self.c_large,
            "cost_model": self.cost_model,
            "tau": self.tau,
            "grid_step": self.grid_step,
            "created_by": f"ucci-python {_io._package_version()}",
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> UCCIRouter:
        """Rebuild a router from a ``ucci-router`` document.

        Unknown keys are ignored. A missing or null ``grid_step`` falls back
        to 0.005.

        Raises
        ------
        ValueError
            If the document fails :func:`ucci.io.validate_router_dict`.
        """
        doc = _io.validate_router_dict(data)
        step = doc["grid_step"] if doc["grid_step"] is not None else DEFAULT_GRID_STEP
        router = cls(doc["c_small"], doc["c_large"], doc["cost_model"], step)
        router.calibrator = IsotonicCalibrator.from_dict(doc["calibrator"])
        router._theta = doc["theta"]
        router.tau = doc["tau"]
        return router

    def save(self, path: PathLike) -> Path:
        """Write the router to ``path`` as JSON (atomically). Returns the path."""
        return _io.save_router(self, path)

    @classmethod
    def load(cls, path: PathLike) -> UCCIRouter:
        """Read a router saved with :meth:`save` (or by the Rust crate).

        Raises
        ------
        ValueError
            If the file is malformed; the message names the path and the
            problem.
        """
        return cls.from_dict(_io.read_router_dict(path))

    def __repr__(self) -> str:
        parts = [
            f"c_small={self.c_small!r}",
            f"c_large={self.c_large!r}",
            f"cost_model={self.cost_model!r}",
        ]
        if self.is_calibrated:
            assert self.calibrator.x_ is not None
            parts.append(f"n_knots={self.calibrator.x_.size}")
        else:
            parts.append("uncalibrated")
        if self._theta is not None:
            parts.append(f"theta={self._theta!r}")
        if self.tau is not None:
            parts.append(f"tau={self.tau!r}")
        if self.budget is not None:
            parts.append(f"budget={self.budget!r}")
        return f"UCCIRouter({', '.join(parts)})"
