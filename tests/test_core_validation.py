"""Tests for ucci._validation: shared input checks and their error messages."""

from __future__ import annotations

import numpy as np
import pytest

from ucci._validation import (
    PROB_ATOL,
    as_1d,
    as_float_array,
    check_cost_model,
    check_costs,
    check_finite_scalar,
    check_grid,
    check_labels,
    check_positive_int,
    check_probabilities,
    check_same_length,
    check_weights,
)


def test_as_float_array_accepts_numbers():
    assert as_float_array([True, False], "x").tolist() == [1.0, 0.0]
    assert as_float_array(np.array([1, 2], dtype=np.int32), "x").dtype == np.float64
    assert as_float_array(np.array([0.5], dtype=np.float32), "x").dtype == np.float64
    assert as_float_array(np.array([1, 2.5], dtype=object), "x").tolist() == [1.0, 2.5]
    assert as_float_array(3, "x").shape == ()
    assert as_float_array([[1, 2], [3, 4]], "x").shape == (2, 2)


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (["0.1", "0.2"], "numeric"),
        (np.array([1 + 2j]), "numeric"),
        ([[1, 2], [3]], "x must"),
        ([None, 1.0], "only numbers"),
        ([{}, 1.0], "only numbers"),
        ([1.0, float("nan")], r"NaN or inf at index 1 \(value nan\)"),
        ([[1.0, 2.0], [3.0, float("inf")]], r"index \(1, 1\)"),
    ],
)
def test_as_float_array_rejects(value, match):
    with pytest.raises(ValueError, match=match):
        as_float_array(value, "x")


def test_as_float_array_non_finite_allowed_on_request():
    out = as_float_array([np.nan, np.inf], "x", finite=False)
    assert np.isnan(out[0]) and np.isinf(out[1])


def test_as_1d():
    assert as_1d([[1.0], [2.0]], "x").tolist() == [1.0, 2.0]
    assert as_1d([], "x", allow_empty=True).shape == (0,)
    with pytest.raises(ValueError, match="x is empty"):
        as_1d([], "x")
    with pytest.raises(ValueError, match=r"x must be a 1-D sequence, got shape \(2, 2\)"):
        as_1d([[1, 2], [3, 4]], "x")
    with pytest.raises(ValueError, match="got a scalar"):
        as_1d(0.5, "x")


def test_check_same_length():
    a, b = np.zeros(3), np.zeros(3)
    assert check_same_length(a=a, b=b) == 3
    with pytest.raises(ValueError, match="length mismatch: a has 3, b has 2"):
        check_same_length(a=a, b=np.zeros(2))


def test_check_probabilities():
    ok = np.array([0.0, 0.5, 1.0, 1.0 + PROB_ATOL / 2])
    assert check_probabilities(ok, "p") is ok
    for bad in (1.0 + 10 * PROB_ATOL, -1e-12, np.nan):
        with pytest.raises(ValueError, match="p must lie"):
            check_probabilities(np.array([0.5, bad]), "p")
    with pytest.raises(ValueError):
        check_probabilities(np.array([1.0 + 1e-12]), "p", atol=0.0)


def test_check_labels():
    e = np.array([0.0, 0.3, 1.0])
    assert check_labels(e, "e") is e
    with pytest.raises(ValueError, match=r"index 0 \(value 1.5\)"):
        check_labels(np.array([1.5]), "e")


def test_check_weights():
    assert check_weights(None, 3).tolist() == [1.0, 1.0, 1.0]
    assert check_weights([0, 2, 1], 3).tolist() == [0.0, 2.0, 1.0]
    with pytest.raises(ValueError, match="length 2, expected 3"):
        check_weights([1, 1], 3)
    with pytest.raises(ValueError, match="non-negative"):
        check_weights([1, -1, 1], 3)
    with pytest.raises(ValueError, match="sums to zero"):
        check_weights([0, 0, 0], 3)
    with pytest.raises(ValueError, match="NaN or inf"):
        check_weights([1, np.nan, 1], 3)
    with pytest.raises(ValueError, match="w has length"):
        check_weights([], 1, "w")


def test_check_finite_scalar():
    assert check_finite_scalar(1, "t") == 1.0
    assert check_finite_scalar(np.float32(0.5), "t") == 0.5
    assert check_finite_scalar(np.int64(2), "t") == 2.0
    for bad in (True, np.bool_(False), "0.5", None, [0.5], complex(1, 0)):
        with pytest.raises(ValueError, match="must be a real number"):
            check_finite_scalar(bad, "t")
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="must be finite"):
            check_finite_scalar(bad, "t")


def test_check_costs_and_cost_model():
    assert check_costs(1, 3.02) == (1.0, 3.02)
    with pytest.raises(ValueError, match="c_small must be positive"):
        check_costs(0.0, 1.0)
    with pytest.raises(ValueError, match="c_large must be positive"):
        check_costs(1.0, -2.0)
    assert check_cost_model("routing") == "routing"
    assert check_cost_model("sequential") == "sequential"
    for bad in ("Routing", None, 1):
        with pytest.raises(ValueError, match="cost_model"):
            check_cost_model(bad)


def test_check_grid_sorts_and_deduplicates():
    assert check_grid([0.5, 0.1, 0.5, 1.0, 0.0]).tolist() == [0.0, 0.1, 0.5, 1.0]
    for bad in ([], [1.1], [-0.01], [0.2, np.nan], [[0.1, 0.2]]):
        with pytest.raises(ValueError, match="grid"):
            check_grid(bad)


def test_check_positive_int():
    assert check_positive_int(3, "k") == 3
    assert check_positive_int(np.int64(2), "k") == 2
    for bad in (0, -1):
        with pytest.raises(ValueError, match="at least 1"):
            check_positive_int(bad, "k")
    for bad in (True, 2.0, "3", None):
        with pytest.raises(ValueError, match="integer"):
            check_positive_int(bad, "k")
