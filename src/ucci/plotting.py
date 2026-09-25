"""Figures for calibration and cost-accuracy trade-offs (paper Figures 1 and 2).

* :func:`reliability_diagram`: forecast probability against observed
  frequency per bin, the diagnostic of paper Figure 1 (Section 6.2), which
  compares the raw token-margin uncertainty u(x) with the isotonic-calibrated
  p_hat(x). The default bins are deciles of the forecast, and the legend
  shows each forecast's ECE (computed by :func:`ucci.ece`, so the figure and
  the reported number always agree).
* :func:`pareto_plot`: mean cost per query against accuracy for one or more
  threshold sweeps plus single operating points, the view of paper Figure 2
  (Section 6.2, cost-accuracy Pareto frontier with end-to-end evaluation).
* :func:`efficient_mask`: which (cost, accuracy) points are not dominated.

matplotlib is an optional dependency (``pip install "ucci[plot]"``) and is
imported only when a plotting function is called, so ``import ucci.plotting``
works without it. The functions build figures with the object-oriented API
(:class:`matplotlib.figure.Figure`), never touch pyplot's global state, and
return the figure; save it with ``fig.savefig(path)``. Pass ``ax=`` to draw
into an existing axes instead.

Colors follow a fixed categorical order (series keep their color when others
are added or removed), lines are 2 px, markers are at least 8 px, and every
series is named in a legend, so identity never depends on color alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import numpy as np

from .metrics import ece, reliability_table

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike, NDArray

__all__ = ["SERIES_COLORS", "efficient_mask", "pareto_plot", "reliability_diagram"]

#: Categorical series colors, assigned in this fixed order.
SERIES_COLORS: Tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#d9d8d4"
_MARKERS = ("o", "s", "D", "^", "v", "P", "X", "*")


def _require_matplotlib() -> Any:
    """Import matplotlib or raise an ImportError that says how to install it."""
    try:
        import matplotlib
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "ucci.plotting needs matplotlib, which is not installed. "
            'Install it with: pip install "ucci[plot]"  (or: pip install matplotlib)'
        ) from exc
    del matplotlib
    return Figure


def _new_axes(ax: Optional[Axes], figsize: Tuple[float, float]) -> Tuple[Figure, Axes]:
    figure_cls = _require_matplotlib()
    if ax is not None:
        fig = ax.get_figure()
        if fig is None:
            raise ValueError("ax is not attached to a figure")
        return cast("Figure", fig), ax
    fig = (
        figure_cls(figsize=figsize, layout="constrained")
        if _supports_layout(figure_cls)
        else figure_cls(figsize=figsize)
    )
    return fig, fig.add_subplot(1, 1, 1)


def _supports_layout(figure_cls: Any) -> bool:
    """Whether ``Figure(layout=...)`` is accepted (matplotlib 3.5 and later)."""
    import inspect

    try:
        return "layout" in inspect.signature(figure_cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic builds
        return False


def _style(ax: Axes) -> None:
    """Recessive grid and axes, ink-colored text."""
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK_2)
    ax.tick_params(colors=_INK_2, labelcolor=_INK_2)
    ax.xaxis.label.set_color(_INK)
    ax.yaxis.label.set_color(_INK)
    ax.title.set_color(_INK)


def _legend(ax: Axes, loc: Any) -> None:
    """Frameless legend with ink-colored text (works on matplotlib 3.3 and later)."""
    leg = ax.legend(loc=loc, frameon=False, fontsize="small")
    for text in leg.get_texts():
        text.set_color(_INK)


def _color(i: int) -> str:
    if i >= len(SERIES_COLORS):
        raise ValueError(
            f"at most {len(SERIES_COLORS)} series per figure (got more); "
            "split them across figures so each keeps a distinct color"
        )
    return SERIES_COLORS[i]


def reliability_diagram(
    forecasts: Union[ArrayLike, Mapping[str, ArrayLike]],
    outcomes: ArrayLike,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    show_ece: bool = True,
    size_by_count: bool = True,
    figsize: Tuple[float, float] = (4.8, 4.4),
) -> Figure:
    """Reliability diagram of one or more probability forecasts (paper Figure 1).

    For each forecast, the points are the bins of :func:`ucci.reliability_table`:
    mean forecast probability on the x axis, observed frequency of the event
    on the y axis. A perfectly calibrated forecast lies on the diagonal.

    Parameters
    ----------
    forecasts : array_like or mapping of str to array_like
        Forecast probabilities of the event ``outcomes == 1``, in [0, 1]. A
        mapping draws one series per entry, labelled by its key, for example
        ``{"raw u(x)": u, "isotonic p_hat": p_hat}``. For UCCI the event is
        "the small model is wrong" (e(x), Section 4.2).
    outcomes : array_like
        Outcomes in [0, 1] (0/1 for the paper's binary error event), same
        length as every forecast.
    n_bins : int, default 10
        Number of bins.
    strategy : {"quantile", "uniform"}, default "quantile"
        ``"quantile"``: equal-count bins at the forecast's quantiles (deciles
        with 10 bins, as the paper's reliability diagram); ``"uniform"``:
        equal-width bins on [0, 1]. Passed to :func:`ucci.reliability_table`.
    ax : matplotlib.axes.Axes, optional
        Draw into this axes; by default a new figure is created.
    title : str, optional
        Axes title.
    show_ece : bool, default True
        Append ``ECE = ...`` (same bins) to each legend label.
    size_by_count : bool, default True
        Scale marker area with the number of forecasts in the bin (minimum
        8 px), so sparsely populated bins read as such.
    figsize : (float, float)
        Size of a new figure in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The figure holding the axes.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        On invalid forecasts or outcomes (checked by
        :func:`ucci.reliability_table`), an empty mapping, or more than eight
        series.

    Examples
    --------
    >>> fig = reliability_diagram({"raw u": u_test, "UCCI p_hat": p_hat_test}, e_test)  # doctest: +SKIP
    >>> fig.savefig("reliability.png", dpi=200)  # doctest: +SKIP
    """
    series = dict(forecasts) if isinstance(forecasts, Mapping) else {"forecast": forecasts}
    if not series:
        raise ValueError("forecasts is empty; pass at least one forecast array")
    tables = {}
    for name, p in series.items():
        rows = reliability_table(p, outcomes, n_bins=n_bins, strategy=strategy)
        e_val = ece(p, outcomes, n_bins=n_bins, strategy=strategy) if show_ece else None
        tables[name] = (rows, e_val)
    if len(tables) > len(SERIES_COLORS):
        _color(len(tables))

    fig, axes = _new_axes(ax, figsize)
    axes.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.0,
        color=_INK_2,
        label="perfect calibration",
        zorder=1,
    )
    total = max(sum(r.count for r in rows) for rows, _ in tables.values())
    for i, (name, (rows, e_val)) in enumerate(tables.items()):
        x = np.array([r.mean_forecast for r in rows])
        y = np.array([r.observed_frequency for r in rows])
        counts = np.array([r.count for r in rows], dtype=float)
        label = name if e_val is None else f"{name} (ECE = {e_val:.3f})"
        color = _color(i)
        axes.plot(
            x,
            y,
            color=color,
            linewidth=2.0,
            zorder=2 + i,
            label=label,
            marker=_MARKERS[i],
            markersize=8,
            markeredgecolor="white",
            markeredgewidth=1.0,
        )
        if size_by_count and total > 0:
            area = 64.0 + 400.0 * counts / total
            axes.scatter(x, y, s=area, color=color, alpha=0.25, linewidths=0, zorder=2 + i)
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.set_aspect("equal", adjustable="box")
    bins = "deciles" if (strategy == "quantile" and n_bins == 10) else f"{n_bins} {strategy} bins"
    axes.set_xlabel(f"Mean forecast probability ({bins})")
    axes.set_ylabel("Observed frequency")
    if title:
        axes.set_title(title)
    _style(axes)
    _legend(axes, "upper left")
    return fig


def efficient_mask(cost: ArrayLike, accuracy: ArrayLike) -> NDArray[np.bool_]:
    """True for points not dominated by another (cheaper-or-equal and at-least-as-accurate).

    A point is dominated when some other point has ``cost <= c`` and
    ``accuracy >= a`` with at least one inequality strict. Of several
    identical points, all are kept.

    Parameters
    ----------
    cost, accuracy : array_like of float, shape (n,)

    Returns
    -------
    numpy.ndarray of bool, shape (n,)

    Raises
    ------
    ValueError
        On mismatched lengths or non-finite values.
    """
    c = np.asarray(cost, dtype=float).ravel()
    a = np.asarray(accuracy, dtype=float).ravel()
    if c.shape != a.shape:
        raise ValueError(f"cost has {c.size} points but accuracy has {a.size}")
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(a))):
        raise ValueError("cost and accuracy must be finite")
    keep = np.ones(c.size, dtype=bool)
    order = np.lexsort((-a, c))  # by cost, then higher accuracy first
    best = -np.inf
    prev_c: Optional[float] = None
    prev_a: Optional[float] = None
    prev_i = -1
    for i in order:
        if prev_c is not None and c[i] == prev_c and a[i] == prev_a:
            keep[i] = keep[prev_i]  # identical point: same verdict
        elif a[i] <= best:
            keep[i] = False
        else:
            best = a[i]
        prev_c, prev_a, prev_i = c[i], a[i], i
    return keep


def _curve_arrays(curve: Any) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Accept (cost, accuracy), a mapping with those keys, or a ParetoFrontier."""
    if hasattr(curve, "cost") and hasattr(curve, "accuracy"):
        cost, acc = curve.cost, curve.accuracy
    elif isinstance(curve, Mapping):
        if "cost" not in curve:
            raise ValueError("a curve mapping needs a 'cost' key")
        acc_key = next((k for k in ("accuracy", "micro_f1", "f1") if k in curve), None)
        if acc_key is None:
            raise ValueError("a curve mapping needs an 'accuracy' (or 'micro_f1') key")
        cost, acc = curve["cost"], curve[acc_key]
    elif isinstance(curve, Sequence) and len(curve) == 2:
        cost, acc = curve
    else:
        raise ValueError(
            "each curve must be (cost, accuracy), a mapping with 'cost' and 'accuracy', "
            f"or a ucci.ParetoFrontier; got {type(curve).__name__}"
        )
    c = np.asarray(cost, dtype=float).ravel()
    a = np.asarray(acc, dtype=float).ravel()
    if c.size != a.size or c.size == 0:
        raise ValueError(
            f"a curve needs equal, non-zero numbers of costs and accuracies ({c.size} vs {a.size})"
        )
    return c, a


def pareto_plot(
    curves: Optional[Mapping[str, Any]] = None,
    points: Optional[Mapping[str, Tuple[float, float]]] = None,
    *,
    frontier_only: bool = True,
    target: Optional[float] = None,
    budget: Optional[float] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    xlabel: str = "Mean cost per query (small model = 1)",
    ylabel: str = "Accuracy (micro-F1)",
    figsize: Tuple[float, float] = (6.0, 4.4),
) -> Figure:
    """Cost against accuracy for threshold sweeps and single operating points (paper Figure 2).

    Parameters
    ----------
    curves : mapping of str to curve, optional
        One line per entry. A curve is a ``(cost, accuracy)`` pair of arrays,
        a mapping with ``"cost"`` and ``"accuracy"`` (or ``"micro_f1"``)
        keys, or a :class:`ucci.ParetoFrontier` (from
        :func:`ucci.pareto_frontier`). Points are drawn in cost order.
    points : mapping of str to (cost, accuracy), optional
        Single operating points, such as the always-small and always-large
        anchors or each method's selected threshold.
    frontier_only : bool, default True
        Draw only the non-dominated points of each curve
        (:func:`efficient_mask`), as a step-free line through them.
    target : float, optional
        Draw a horizontal reference line at this accuracy target (tau).
    budget : float, optional
        Draw a vertical reference line at this cost budget.
    ax : matplotlib.axes.Axes, optional
        Draw into this axes; by default a new figure is created.
    title, xlabel, ylabel : str
        Labels.
    figsize : (float, float)
        Size of a new figure in inches.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If there is nothing to draw, a curve is malformed, or there are more
        than eight curves plus points.
    """
    curves = dict(curves or {})
    points = dict(points or {})
    if not curves and not points:
        raise ValueError("nothing to plot: pass curves, points or both")
    if len(curves) + len(points) > len(SERIES_COLORS):
        _color(len(curves) + len(points))
    parsed = {name: _curve_arrays(c) for name, c in curves.items()}

    fig, axes = _new_axes(ax, figsize)
    i = 0
    for name, (c, a) in parsed.items():
        if frontier_only:
            keep = efficient_mask(c, a)
            c, a = c[keep], a[keep]
        order = np.lexsort((a, c))
        axes.plot(c[order], a[order], color=_color(i), linewidth=2.0, label=name, zorder=2)
        i += 1
    for name, (pc, pa) in points.items():
        axes.plot(
            [float(pc)],
            [float(pa)],
            linestyle="none",
            marker=_MARKERS[i % len(_MARKERS)],
            markersize=9,
            color=_color(i),
            markeredgecolor="white",
            markeredgewidth=1.2,
            label=name,
            zorder=3,
        )
        i += 1
    if target is not None:
        axes.axhline(float(target), color=_INK_2, linestyle="--", linewidth=1.0, zorder=1)
        axes.annotate(
            f"target {float(target):.3f}",
            xy=(1.0, float(target)),
            xycoords=("axes fraction", "data"),
            xytext=(-4, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize="small",
            color=_INK_2,
        )
    if budget is not None:
        axes.axvline(float(budget), color=_INK_2, linestyle=":", linewidth=1.0, zorder=1)
        axes.annotate(
            f"budget {float(budget):.2f}",
            xy=(float(budget), 0.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, 4),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize="small",
            color=_INK_2,
        )
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    if title:
        axes.set_title(title)
    _style(axes)
    _legend(axes, "lower right")
    return fig
