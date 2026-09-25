"""UCCI: calibrated uncertainty for cost-optimal LLM cascade routing.

Reference implementation of Varun Kotte, "UCCI: Calibrated Uncertainty for
Cost-Optimal LLM Cascade Routing", arXiv:2605.18796 (2026).

The core needs numpy only:

* :mod:`ucci.signal`: token-margin uncertainty u(x) (Section 4.1, Eq. 4).
* :mod:`ucci.calibration`: isotonic map from u(x) to P(small model wrong)
  (Section 4.2).
* :mod:`ucci.policy`: threshold policy and constrained threshold selection
  (Section 4.3, Eq. 6 and 7; Theorem 1).
* :mod:`ucci.router`: :class:`UCCIRouter`, the three steps plus the
  evaluation protocol of Section 6.1.
* :mod:`ucci.metrics`: ECE, reliability tables, Brier score, bootstrap
  intervals, micro-F1.
* :mod:`ucci.io`: the JSON router format shared with the Rust crate.

Optional submodules load on first access and may need extra dependencies:
``ucci.baselines``, ``ucci.integrations``, ``ucci.online``,
``ucci.plotting`` (matplotlib) and ``ucci.cli``. Importing ``ucci`` itself
never imports them.
"""

from __future__ import annotations

import importlib
from typing import Any

from .calibration import IsotonicCalibrator, pav
from .io import _package_version, load_router, save_router
from .metrics import (
    ConfidenceInterval,
    ReliabilityRow,
    RoutedMicroF1,
    bootstrap_ci,
    brier_score,
    ece,
    micro_f1,
    reliability_table,
    routed_micro_f1,
)
from .policy import (
    DEFAULT_COST_LARGE,
    DEFAULT_COST_SMALL,
    DEFAULT_GRID,
    DEFAULT_GRID_STEP,
    InfeasibleTargetError,
    ParetoFrontier,
    ThresholdChoice,
    escalate,
    evaluate,
    make_grid,
    pareto_frontier,
    policy_accuracy,
    policy_cost,
    select_threshold,
    select_threshold_for_budget,
)
from .router import RouteResult, UCCIRouter
from .signal import (
    batch_uncertainty,
    from_openai_logprobs,
    from_vllm_logprobs,
    margins_from_top2,
    token_margin_uncertainty,
    top2_from_logprobs,
    uncertainty_from_logprobs,
    uncertainty_from_margins,
    uncertainty_from_probs,
)

__version__: str = _package_version()

__all__ = [
    "DEFAULT_COST_LARGE",
    "DEFAULT_COST_SMALL",
    "DEFAULT_GRID",
    "DEFAULT_GRID_STEP",
    "ConfidenceInterval",
    "InfeasibleTargetError",
    "IsotonicCalibrator",
    "ParetoFrontier",
    "ReliabilityRow",
    "RouteResult",
    "RoutedMicroF1",
    "ThresholdChoice",
    "UCCIRouter",
    "__version__",
    "batch_uncertainty",
    "bootstrap_ci",
    "brier_score",
    "ece",
    "escalate",
    "evaluate",
    "from_openai_logprobs",
    "from_vllm_logprobs",
    "load_router",
    "make_grid",
    "margins_from_top2",
    "micro_f1",
    "pareto_frontier",
    "pav",
    "policy_accuracy",
    "policy_cost",
    "reliability_table",
    "routed_micro_f1",
    "save_router",
    "select_threshold",
    "select_threshold_for_budget",
    "token_margin_uncertainty",
    "top2_from_logprobs",
    "uncertainty_from_logprobs",
    "uncertainty_from_margins",
    "uncertainty_from_probs",
]

#: Submodules imported on first attribute access (``ucci.plotting`` etc.).
_LAZY_SUBMODULES = ("baselines", "cli", "integrations", "online", "plotting")


def __getattr__(name: str) -> Any:
    """Import optional submodules lazily, so ``import ucci`` stays numpy-only."""
    if name in _LAZY_SUBMODULES:
        try:
            module = importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError as exc:
            if exc.name != f"{__name__}.{name}":
                raise  # the submodule exists but an optional dependency is missing
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r} (submodule not installed)"
            ) from exc
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))
