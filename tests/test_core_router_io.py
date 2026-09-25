"""Tests for ucci.router and ucci.io: the router and its JSON format."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

import ucci
from ucci import (
    InfeasibleTargetError,
    RouteResult,
    ThresholdChoice,
    UCCIRouter,
    evaluate,
    load_router,
    routed_micro_f1,
    save_router,
    select_threshold,
)
from ucci.io import (
    FORMAT_NAME,
    FORMAT_VERSION,
    dumps_router_dict,
    read_router_dict,
    validate_router_dict,
    write_router_dict,
)


def _draw(rng: np.random.Generator, n: int):
    u = rng.beta(2, 4, n)
    small_ok = (rng.random(n) >= u**2).astype(float)
    large_ok = (rng.random(n) < 0.95).astype(float)
    return u, small_ok, large_ok


@pytest.fixture
def fitted() -> UCCIRouter:
    rng = np.random.default_rng(4)
    u_c, s_c, _ = _draw(rng, 6000)
    u_v, s_v, l_v = _draw(rng, 4000)
    r = UCCIRouter(c_small=1.0, c_large=3.02).calibrate(u_c, 1 - s_c)
    r.choose_threshold(u_v, s_v, l_v, tau=0.5 * (s_v.mean() + l_v.mean()))
    return r


def _valid_doc() -> dict:
    return {
        "format": "ucci-router",
        "version": 1,
        "calibrator": {"x": [0.1, 0.5, 0.9], "y": [0.0, 0.25, 1.0]},
        "theta": 0.3,
        "c_small": 1.0,
        "c_large": 3.02,
        "cost_model": "routing",
        "tau": 0.91,
        "grid_step": 0.005,
        "created_by": "ucci-python 0.1.0",
    }


# ------------------------------------------------------------ end to end


def test_router_end_to_end_meets_target_on_fresh_data():
    rng = np.random.default_rng(4)
    u_c, s_c, _ = _draw(rng, 6000)
    u_v, s_v, l_v = _draw(rng, 4000)
    u_t, s_t, l_t = _draw(rng, 10000)
    r = UCCIRouter(c_small=1.0, c_large=3.0).calibrate(u_c, 1 - s_c)
    tau = 0.5 * (s_v.mean() + l_v.mean())
    choice = r.choose_threshold(u_v, s_v, l_v, tau)
    assert choice.accuracy >= tau and r.choice == choice and r.tau == tau
    test = r.evaluate(u_t, s_t, l_t)
    assert test == evaluate(r.error_probability(u_t), s_t, l_t, choice.theta, 1.0, 3.0)
    assert test.accuracy >= tau - 0.02
    assert test.cost < 3.0


def test_router_matches_functional_api(fitted):
    rng = np.random.default_rng(11)
    u_v, s_v, l_v = _draw(rng, 3000)
    via_router = fitted.choose_threshold(u_v, s_v, l_v, tau=0.85)
    via_functions = select_threshold(
        fitted.calibrator.predict(u_v), s_v, l_v, 0.85, fitted.c_small, fitted.c_large
    )
    assert via_router == via_functions


def test_defaults_are_the_papers_normalized_costs():
    r = UCCIRouter()
    assert (r.c_small, r.c_large, r.cost_model, r.grid_step) == (
        1.0,
        3.02,
        "routing",
        0.005,
    )


def test_theta_before_selection_raises():
    r = UCCIRouter().calibrate([0.1, 0.9], [0, 1])
    assert not r.has_threshold
    with pytest.raises(RuntimeError, match="no threshold yet"):
        _ = r.theta
    with pytest.raises(RuntimeError, match="no threshold yet"):
        r.escalate(0.5)
    with pytest.raises(RuntimeError, match="not fitted"):
        UCCIRouter().choose_threshold([0.1], [1], [1], tau=0.5)


def test_setting_theta_by_hand(fitted):
    fitted.theta = 0.25
    assert fitted.theta == 0.25 and fitted.choice is None and fitted.tau is None
    with pytest.raises(ValueError, match="theta"):
        fitted.theta = float("nan")


def test_scalar_and_array_routing(fitted):
    fitted.theta = 0.2
    u = np.array([0.05, 0.3, 0.6, 0.95])
    p = fitted.error_probability(u)
    assert isinstance(fitted.error_probability(0.3), float)
    assert isinstance(fitted.escalate(0.3), bool)
    np.testing.assert_array_equal(fitted.escalate(u), p > 0.2)
    res = fitted.route(u)
    assert isinstance(res, RouteResult)
    np.testing.assert_array_equal(res.escalate, p > 0.2)
    np.testing.assert_array_equal(res.p_hat, p)
    esc, p_hat = res  # unpacks
    assert esc.dtype == bool and p_hat.dtype == np.float64
    assert fitted.route(np.ones((2, 2)) * 0.5).escalate.shape == (2, 2)


def test_budget_selection(fitted):
    rng = np.random.default_rng(7)
    u_v, s_v, l_v = _draw(rng, 3000)
    c = fitted.choose_threshold_for_budget(u_v, s_v, l_v, budget=2.0)
    assert isinstance(c, ThresholdChoice) and c.cost <= 2.0
    assert fitted.budget == 2.0 and fitted.tau is None and fitted.theta == c.theta
    assert "budget=2.0" in repr(fitted)
    with pytest.raises(InfeasibleTargetError):
        fitted.choose_threshold_for_budget(u_v, s_v, l_v, budget=0.5)


def test_custom_grid_and_metric(fitted):
    rng = np.random.default_rng(8)
    n = 500
    u_v = rng.random(n)
    small = np.stack([rng.integers(0, 3, n), rng.integers(0, 2, n), rng.integers(0, 2, n)], 1)
    large = np.stack([small[:, 0] + small[:, 2], small[:, 1] * 0, small[:, 2] * 0], 1)
    f1 = routed_micro_f1(small, large)
    c = fitted.choose_threshold(u_v, None, None, tau=0.0, grid=[0.2, 0.4], metric=f1)
    assert c.theta == 0.4
    c = fitted.choose_threshold(u_v, None, None, tau=0.95, metric=f1)
    assert c.accuracy >= 0.95 and f1(fitted.escalate(u_v)) == c.accuracy
    assert fitted.evaluate(u_v, None, None, metric=f1).accuracy == c.accuracy


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"c_small": 0.0}, "c_small"),
        ({"c_large": float("nan")}, "c_large"),
        ({"cost_model": "cheap"}, "cost_model"),
        ({"grid_step": 0.3}, "step"),
        ({"grid_step": 0.0}, "step"),
    ],
)
def test_router_init_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        UCCIRouter(**kwargs)


def test_repr_after_selection(fitted):
    assert repr(fitted).startswith("UCCIRouter(c_small=1.0, c_large=3.02, cost_model='routing'")
    assert f"theta={fitted.theta!r}, tau={fitted.tau!r})" in repr(fitted)


def test_repr():
    assert repr(UCCIRouter()) == (
        "UCCIRouter(c_small=1.0, c_large=3.02, cost_model='routing', uncalibrated)"
    )
    r = UCCIRouter(cost_model="sequential").calibrate([0.1, 0.5, 0.9], [0, 1, 1])
    r.theta = 0.3
    assert repr(r) == (
        "UCCIRouter(c_small=1.0, c_large=3.02, cost_model='sequential', n_knots=3, theta=0.3)"
    )


# ---------------------------------------------------------- serialization


def test_to_dict_has_the_contract_format(fitted):
    d = fitted.to_dict()
    assert list(d) == [
        "format",
        "version",
        "calibrator",
        "theta",
        "c_small",
        "c_large",
        "cost_model",
        "tau",
        "grid_step",
        "created_by",
    ]
    assert d["format"] == FORMAT_NAME == "ucci-router"
    assert d["version"] == FORMAT_VERSION == 1
    assert set(d["calibrator"]) == {"x", "y"}
    assert all(type(v) is float for v in d["calibrator"]["x"] + d["calibrator"]["y"])
    assert type(d["theta"]) is float and type(d["tau"]) is float
    assert d["grid_step"] == 0.005
    assert d["created_by"] == f"ucci-python {ucci.__version__}"
    json.dumps(d, allow_nan=False)


def test_save_load_round_trip_is_bit_identical(fitted, tmp_path):
    path = fitted.save(tmp_path / "router.json")
    assert path == tmp_path / "router.json"
    back = UCCIRouter.load(path)
    u = np.random.default_rng(0).random(20000) * 1.2 - 0.1
    np.testing.assert_array_equal(back.error_probability(u), fitted.error_probability(u))
    np.testing.assert_array_equal(back.escalate(u), fitted.escalate(u))
    assert (back.theta, back.tau, back.c_small, back.c_large, back.cost_model) == (
        fitted.theta,
        fitted.tau,
        fitted.c_small,
        fitted.c_large,
        fitted.cost_model,
    )
    assert back.choice is None and back.is_calibrated
    assert back.to_dict() == fitted.to_dict()
    # Module-level helpers and str paths work the same way.
    save_router(fitted, str(tmp_path / "again.json"))
    assert load_router(str(tmp_path / "again.json")).to_dict() == fitted.to_dict()


def test_saved_file_is_compact_valid_json(fitted, tmp_path):
    path = fitted.save(tmp_path / "r.json")
    text = path.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert len(text.splitlines()) == 12  # braces plus one line per key
    assert json.loads(text) == validate_router_dict(fitted.to_dict())
    assert "NaN" not in text and "Infinity" not in text


def test_save_is_atomic_and_overwrites(fitted, tmp_path):
    path = tmp_path / "r.json"
    path.write_text("old contents", encoding="utf-8")
    fitted.save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["format"] == "ucci-router"
    assert sorted(os.listdir(tmp_path)) == ["r.json"]  # no temporary files left


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_saved_file_mode_follows_umask(fitted, tmp_path):
    # Same permissions as a plain open(path, "w"), not mkstemp's 0600.
    old = os.umask(0o022)
    try:
        fitted.save(tmp_path / "r.json")
        write_router_dict(_valid_doc(), tmp_path / "doc.json")
    finally:
        os.umask(old)
    assert (tmp_path / "r.json").stat().st_mode & 0o777 == 0o644
    assert (tmp_path / "doc.json").stat().st_mode & 0o777 == 0o644


def test_failed_write_leaves_no_file(tmp_path):
    bad = _valid_doc()
    bad["theta"] = "high"
    with pytest.raises(ValueError, match="theta"):
        write_router_dict(bad, tmp_path / "r.json")
    assert os.listdir(tmp_path) == []


def test_interrupted_write_cleans_up(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        write_router_dict(_valid_doc(), tmp_path / "r.json")
    assert os.listdir(tmp_path) == []


def test_save_requires_calibration_and_threshold(tmp_path):
    with pytest.raises(RuntimeError, match="uncalibrated"):
        UCCIRouter().save(tmp_path / "r.json")
    r = UCCIRouter().calibrate([0.1, 0.9], [0, 1])
    with pytest.raises(RuntimeError, match="no threshold"):
        r.save(tmp_path / "r.json")


def test_sequential_router_round_trip(tmp_path):
    r = UCCIRouter(c_small=2.0, c_large=5.0, cost_model="sequential", grid_step=0.01)
    r.calibrate([0.1, 0.4, 0.8], [0, 1, 1])
    r.choose_threshold([0.1, 0.4, 0.8], [1, 0, 0], [1, 1, 1], tau=1.0)
    back = UCCIRouter.load(r.save(tmp_path / "seq.json"))
    assert back.cost_model == "sequential" and back.grid_step == 0.01
    assert back.to_dict() == r.to_dict()


def test_reads_documents_from_other_writers(tmp_path):
    # Compact single-line JSON with ints, extra keys, and no optional fields.
    text = (
        '{"version":1,"format":"ucci-router","theta":0,"c_small":1,"c_large":3,'
        '"cost_model":"routing","calibrator":{"x":[0,1],"y":[0,1],"extra":true},'
        '"future_field":{"a":1}}'
    )
    path = tmp_path / "rust.json"
    path.write_text(text, encoding="utf-8")
    r = load_router(path)
    assert r.theta == 0.0 and r.tau is None and r.grid_step == 0.005
    assert r.error_probability(0.25) == 0.25
    doc = read_router_dict(path)
    assert "future_field" not in doc and doc["created_by"] is None
    assert doc["grid_step"] is None


def test_created_by_ignored_if_not_a_string():
    doc = _valid_doc()
    doc["created_by"] = 7
    assert validate_router_dict(doc)["created_by"] is None


def _mutate(path, value):
    doc = copy.deepcopy(_valid_doc())
    target = doc
    for key in path[:-1]:
        target = target[key]
    if value is _DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return doc


_DELETE = object()


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("format",), "ucci-model", "not a UCCI router file"),
        (("format",), _DELETE, "not a UCCI router file"),
        (("version",), 2, "unsupported router format version 2"),
        (("version",), "1", "must be an integer"),
        (("version",), True, "must be an integer"),
        (("version",), 1.0, "must be an integer"),
        (("calibrator",), _DELETE, "'calibrator' must be an object"),
        (("calibrator",), [0.1, 0.2], "'calibrator' must be an object"),
        (("calibrator", "x"), _DELETE, "missing 'calibrator.x'"),
        (("calibrator", "y"), _DELETE, "missing 'calibrator.y'"),
        (("calibrator", "x"), "0.1,0.5", "must be a list"),
        (
            ("calibrator", "x"),
            [0.1, "0.5", 0.9],
            r"calibrator.x\[1\]' must be a number",
        ),
        (("calibrator", "x"), [0.1, True, 0.9], r"calibrator.x\[1\]' must be a number"),
        (("calibrator", "x"), [0.1, float("nan"), 0.9], "must be finite"),
        (("calibrator", "x"), [0.1, 0.5, 0.5], "strictly increasing"),
        (("calibrator", "x"), [0.9, 0.5, 0.1], "strictly increasing"),
        (("calibrator", "y"), [0.0, 0.5, 0.25], "non-decreasing"),
        (("calibrator", "y"), [0.0, 0.5, 1.5], "not a probability"),
        (("calibrator", "y"), [-0.1, 0.5, 0.6], "not a probability"),
        (("calibrator", "y"), [0.0, 0.5], "has 3 values but 'calibrator.y' has 2"),
        (("calibrator",), {"x": [], "y": []}, "empty"),
        (("theta",), _DELETE, "missing 'theta'"),
        (("theta",), "0.3", "'theta' must be a number"),
        (("theta",), float("inf"), "'theta' must be finite"),
        (("theta",), None, "'theta' must be a number"),
        (("c_small",), 0.0, "'c_small' must be positive"),
        (("c_large",), -3.0, "'c_large' must be positive"),
        (("c_large",), _DELETE, "missing 'c_large'"),
        (("cost_model",), "parallel", "'cost_model' must be"),
        (("cost_model",), _DELETE, "'cost_model' must be"),
        (("tau",), "high", "'tau' must be a number"),
        (("grid_step",), 0.0, "'grid_step' must lie in"),
        (("grid_step",), 2.0, "'grid_step' must lie in"),
        (("grid_step",), 0.003, "'grid_step' must divide 1"),
    ],
)
def test_rejects_malformed_documents(path, value, match, tmp_path):
    doc = _mutate(path, value)
    with pytest.raises(ValueError, match=match):
        validate_router_dict(doc)
    with pytest.raises(ValueError, match=match):
        UCCIRouter.from_dict(doc)
    # The same error through a file, prefixed with the path.
    file = tmp_path / "bad.json"
    file.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.json"):
        load_router(file)


def test_rejects_non_object_and_invalid_json(tmp_path):
    with pytest.raises(ValueError, match="JSON object"):
        validate_router_dict([1, 2])
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "ucci-router", ', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_router(bad)
    nan_doc = json.dumps(_valid_doc()).replace('"theta": 0.3', '"theta": NaN')
    bad.write_text(nan_doc, encoding="utf-8")
    with pytest.raises(ValueError, match="NaN, which is not valid JSON"):
        UCCIRouter.load(bad)
    with pytest.raises(FileNotFoundError):
        load_router(tmp_path / "missing.json")


def test_optional_fields_default():
    doc = _valid_doc()
    del doc["tau"], doc["grid_step"], doc["created_by"]
    r = UCCIRouter.from_dict(doc)
    assert r.tau is None and r.grid_step == 0.005
    doc["grid_step"] = None
    doc["tau"] = None
    assert UCCIRouter.from_dict(doc).grid_step == 0.005


def test_dumps_router_dict_round_trips():
    doc = validate_router_dict(_valid_doc())
    assert json.loads(dumps_router_dict(doc)) == doc
    with pytest.raises(ValueError):
        dumps_router_dict({**doc, "theta": float("nan")})


def test_path_objects_accepted(fitted, tmp_path):
    p = Path(tmp_path) / "sub"
    p.mkdir()
    out = save_router(fitted, p / "r.json")
    assert isinstance(out, Path) and out.exists()
