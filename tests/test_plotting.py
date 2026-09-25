"""Tests for ucci.plotting (paper Figures 1 and 2). Skipped without matplotlib."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

import ucci  # noqa: E402
from ucci import plotting  # noqa: E402


@pytest.fixture()
def forecast_data() -> tuple:
    rng = np.random.default_rng(0)
    u = rng.beta(2.0, 5.0, 2000)
    e = (rng.random(2000) < 1.0 / (1.0 + np.exp(-10.0 * (u - 0.45)))).astype(float)
    cal = ucci.IsotonicCalibrator().fit(u, e)
    return u, np.asarray(cal.predict(u)), e


# ---------------------------------------------------------------------------
# Import behaviour
# ---------------------------------------------------------------------------


def test_import_does_not_load_matplotlib() -> None:
    code = "import sys, ucci.plotting; print('matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_import_ucci_does_not_load_plotting() -> None:
    code = "import sys, ucci; print('ucci.plotting' in sys.modules, 'matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False False"


def test_missing_matplotlib_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.figure", None)
    with pytest.raises(ImportError, match=r"pip install"):
        plotting.reliability_diagram([0.1, 0.9], [0, 1])
    with pytest.raises(ImportError, match=r"ucci-router\[plot\]"):
        plotting.pareto_plot({"a": ([1.0, 2.0], [0.5, 0.6])})


def test_lazy_attribute_on_package() -> None:
    assert ucci.plotting is plotting


# ---------------------------------------------------------------------------
# reliability_diagram
# ---------------------------------------------------------------------------


def test_reliability_single_forecast_matches_table(forecast_data: tuple) -> None:
    u, _, e = forecast_data
    fig = plotting.reliability_diagram(u, e)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    rows = ucci.reliability_table(u, e, n_bins=10, strategy="quantile")
    line = next(ln for ln in ax.lines if ln.get_label().startswith("forecast"))
    np.testing.assert_allclose(line.get_xdata(), [r.mean_forecast for r in rows])
    np.testing.assert_allclose(line.get_ydata(), [r.observed_frequency for r in rows])
    assert ax.get_xlim() == (0.0, 1.0) and ax.get_ylim() == (0.0, 1.0)


def test_reliability_mapping_labels_show_ece(forecast_data: tuple) -> None:
    u, p_hat, e = forecast_data
    fig = plotting.reliability_diagram({"raw u": u, "isotonic": p_hat}, e, strategy="uniform")
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels[0] == "perfect calibration"
    assert f"raw u (ECE = {ucci.ece(u, e, strategy='uniform'):.3f})" in labels
    assert f"isotonic (ECE = {ucci.ece(p_hat, e, strategy='uniform'):.3f})" in labels
    # One colored line per series plus the diagonal.
    assert len(fig.axes[0].lines) == 3


def test_reliability_series_colors_follow_fixed_order(forecast_data: tuple) -> None:
    u, p_hat, e = forecast_data
    fig = plotting.reliability_diagram({"a": u, "b": p_hat}, e, show_ece=False)
    colored = [ln for ln in fig.axes[0].lines if ln.get_label() in ("a", "b")]
    assert [matplotlib.colors.to_hex(ln.get_color()) for ln in colored] == list(
        plotting.SERIES_COLORS[:2]
    )


def test_reliability_without_ece_or_sizes(forecast_data: tuple) -> None:
    u, _, e = forecast_data
    fig = plotting.reliability_diagram({"u": u}, e, show_ece=False, size_by_count=False, title="t")
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["perfect calibration", "u"]
    assert not ax.collections  # no count-scaled scatter layer
    assert ax.get_title() == "t"


def test_reliability_draws_into_given_axes(forecast_data: tuple) -> None:
    u, _, e = forecast_data
    fig = Figure()
    ax = fig.add_subplot(1, 2, 2)
    out = plotting.reliability_diagram(u, e, ax=ax)
    assert out is fig
    assert ax.lines


def test_reliability_saves_png(forecast_data: tuple, tmp_path: Path) -> None:
    u, p_hat, e = forecast_data
    fig = plotting.reliability_diagram({"raw": u, "cal": p_hat}, e)
    path = tmp_path / "rel.png"
    fig.savefig(path, dpi=60)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_reliability_constant_forecast() -> None:
    fig = plotting.reliability_diagram([0.3] * 10, [0, 1] * 5)
    assert fig.axes[0].lines


@pytest.mark.parametrize(
    ("p", "y", "match"),
    [
        ([0.2, 1.5], [0, 1], "p"),
        ([0.2, 0.3], [0, 1, 1], "length|same"),
        ([np.nan, 0.3], [0, 1], "NaN|finite|nan"),
    ],
)
def test_reliability_rejects_bad_input(p: list, y: list, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        plotting.reliability_diagram(p, y)


def test_reliability_rejects_empty_mapping() -> None:
    with pytest.raises(ValueError, match="empty"):
        plotting.reliability_diagram({}, [0, 1])


def test_reliability_rejects_too_many_series() -> None:
    p = np.linspace(0.05, 0.95, 20)
    y = (p > 0.5).astype(float)
    with pytest.raises(ValueError, match="at most 8"):
        plotting.reliability_diagram({str(i): p for i in range(9)}, y)


# ---------------------------------------------------------------------------
# efficient_mask
# ---------------------------------------------------------------------------


def _brute_force_efficient(c: np.ndarray, a: np.ndarray) -> np.ndarray:
    keep = np.ones(c.size, dtype=bool)
    for i in range(c.size):
        for j in range(c.size):
            if i != j and c[j] <= c[i] and a[j] >= a[i] and (c[j] < c[i] or a[j] > a[i]):
                keep[i] = False
                break
    return keep


@pytest.mark.parametrize("seed", range(25))
def test_efficient_mask_matches_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    # Coarse values force ties in cost, accuracy and whole points.
    c = rng.integers(0, 6, n).astype(float)
    a = rng.integers(0, 6, n).astype(float)
    np.testing.assert_array_equal(plotting.efficient_mask(c, a), _brute_force_efficient(c, a))


def test_efficient_mask_keeps_identical_frontier_points() -> None:
    mask = plotting.efficient_mask([1.0, 1.0, 2.0, 2.0], [0.5, 0.5, 0.4, 0.9])
    assert mask.tolist() == [True, True, False, True]


def test_efficient_mask_agrees_with_core_frontier() -> None:
    rng = np.random.default_rng(3)
    p = rng.random(300)
    small = (rng.random(300) > p).astype(float)
    large = np.ones(300)
    front = ucci.pareto_frontier(p, small, large, 1.0, 3.0)
    np.testing.assert_array_equal(
        plotting.efficient_mask(front.cost, front.accuracy), front.efficient
    )


def test_efficient_mask_validates() -> None:
    with pytest.raises(ValueError, match="points"):
        plotting.efficient_mask([1.0, 2.0], [0.5])
    with pytest.raises(ValueError, match="finite"):
        plotting.efficient_mask([1.0, np.inf], [0.5, 0.6])


# ---------------------------------------------------------------------------
# pareto_plot
# ---------------------------------------------------------------------------


def test_pareto_plot_accepts_all_curve_forms() -> None:
    rng = np.random.default_rng(1)
    p = rng.random(200)
    small = (rng.random(200) > p).astype(float)
    front = ucci.pareto_frontier(p, small, np.ones(200), 1.0, 3.02)
    fig = plotting.pareto_plot(
        {
            "frontier object": front,
            "tuple": ([1.0, 2.0, 3.0], [0.8, 0.9, 0.95]),
            "mapping": {"cost": [1.0, 3.0], "micro_f1": [0.7, 0.9]},
        },
        {"Small-only": (1.0, 0.8), "Large-only": (3.02, 0.95)},
        target=0.9,
        budget=2.0,
        title="test",
    )
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["frontier object", "tuple", "mapping", "Small-only", "Large-only"]
    texts = [t.get_text() for t in ax.texts]
    assert "target 0.900" in texts and "budget 2.00" in texts
    assert ax.get_title() == "test"


def test_pareto_plot_frontier_only_draws_sorted_efficient_points() -> None:
    cost = [3.0, 1.0, 2.0, 2.5]
    acc = [0.9, 0.5, 0.8, 0.7]  # (2.5, 0.7) is dominated by (2.0, 0.8)
    fig = plotting.pareto_plot({"c": (cost, acc)})
    line = fig.axes[0].lines[0]
    assert line.get_xdata().tolist() == [1.0, 2.0, 3.0]
    assert line.get_ydata().tolist() == [0.5, 0.8, 0.9]
    fig_all = plotting.pareto_plot({"c": (cost, acc)}, frontier_only=False)
    assert fig_all.axes[0].lines[0].get_xdata().tolist() == [1.0, 2.0, 2.5, 3.0]


def test_pareto_plot_points_only() -> None:
    fig = plotting.pareto_plot(points={"UCCI": (2.08, 0.91)})
    assert fig.axes[0].lines[0].get_xdata().tolist() == [2.08]


def test_pareto_plot_draws_into_given_axes() -> None:
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    assert plotting.pareto_plot({"c": ([1.0, 2.0], [0.5, 0.6])}, ax=ax) is fig


@pytest.mark.parametrize(
    ("curves", "match"),
    [
        ({"bad": "x"}, "each curve"),
        ({"bad": {"costs": [1.0]}}, "'cost'"),
        ({"bad": {"cost": [1.0]}}, "accuracy"),
        ({"bad": ([1.0, 2.0], [0.5])}, "equal"),
        ({"bad": ([], [])}, "non-zero"),
    ],
)
def test_pareto_plot_rejects_malformed_curves(curves: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        plotting.pareto_plot(curves)


def test_pareto_plot_needs_something() -> None:
    with pytest.raises(ValueError, match="nothing to plot"):
        plotting.pareto_plot()


def test_pareto_plot_rejects_too_many_series() -> None:
    with pytest.raises(ValueError, match="at most 8"):
        plotting.pareto_plot(points={str(i): (1.0 + i, 0.5) for i in range(9)})


def test_pareto_plot_saves_png(tmp_path: Path) -> None:
    fig = plotting.pareto_plot({"c": ([1.0, 2.0], [0.5, 0.6])}, {"p": (1.5, 0.55)})
    path = tmp_path / "pareto.png"
    fig.savefig(path, dpi=60)
    assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# The synthetic example draws both figures
# ---------------------------------------------------------------------------


def test_synthetic_demo_writes_both_figures(tmp_path: Path) -> None:
    demo = Path(__file__).resolve().parents[1] / "examples" / "synthetic_demo.py"
    if not demo.exists():
        pytest.skip("examples/ not present")
    out = subprocess.run(
        [sys.executable, str(demo), "--n", "4000", "--plot", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "SYNTHETIC DATA" in out.stdout
    for name in ("synthetic_reliability.png", "synthetic_pareto.png"):
        assert (tmp_path / name).read_bytes()[:4] == b"\x89PNG"
