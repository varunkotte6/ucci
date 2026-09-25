"""Tests for ucci.signal: token-margin uncertainty u(x) (paper Section 4.1, Eq. 4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ucci import (
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


def _random_pairs(rng: np.random.Generator, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Valid top-1/top-2 probabilities: p2 <= p1 and p1 + p2 <= 1."""
    p1 = rng.uniform(0.0, 1.0, t)
    p2 = rng.uniform(0.0, 1.0, t) * np.minimum(p1, 1.0 - p1)
    return p1, p2


# ----------------------------------------------------------------- Eq. 4


def test_eq4_worked_example():
    top2 = [(0.9, 0.05), (0.6, 0.3), (0.5, 0.5)]
    margins = [0.85, 0.3, 0.0]
    assert token_margin_uncertainty(top2) == pytest.approx(1 - sum(margins) / 3)
    assert margins_from_top2(top2) == pytest.approx(margins)


def test_single_token():
    assert token_margin_uncertainty([(0.7, 0.2)]) == pytest.approx(0.5)


def test_pair_order_does_not_matter():
    assert token_margin_uncertainty([(0.1, 0.8)]) == pytest.approx(1 - 0.7)
    assert uncertainty_from_probs([0.1], [0.8]) == pytest.approx(0.3)


def test_certain_and_tied_extremes():
    assert token_margin_uncertainty([(1.0, 0.0)] * 4) == 0.0
    assert token_margin_uncertainty([(0.4, 0.4)] * 4) == 1.0
    assert uncertainty_from_probs(np.ones(5), np.zeros(5)) == 0.0
    assert uncertainty_from_probs(np.full(5, 0.5), np.full(5, 0.5)) == 1.0


def test_u_is_mean_of_one_minus_margins():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p1, p2 = _random_pairs(rng, int(rng.integers(1, 40)))
        expected = float(np.mean(1.0 - (p1 - p2)))
        assert uncertainty_from_probs(p1, p2) == pytest.approx(expected, abs=1e-12)
        assert token_margin_uncertainty(zip(p1, p2)) == pytest.approx(expected, abs=1e-12)
        u = uncertainty_from_probs(p1, p2)
        assert 0.0 <= u <= 1.0


def test_roundoff_above_one_is_capped():
    # exp(logprob) of a certain token can round to 1 + a few ulps.
    assert token_margin_uncertainty([(1.0 + 1e-12, 0.0)]) == 0.0
    assert uncertainty_from_probs([1.0 + 1e-10], [0.0]) == 0.0


@pytest.mark.parametrize(
    "pair", [(1.0 + 1e-6, 0.0), (0.5, -0.1), (float("nan"), 0.1), (0.3, float("inf"))]
)
def test_invalid_probabilities_rejected(pair):
    with pytest.raises(ValueError, match="token 0"):
        token_margin_uncertainty([pair])


def test_pair_must_have_two_values():
    with pytest.raises(ValueError, match="pair"):
        margins_from_top2([(0.9, 0.05, 0.01)])


def test_empty_generation_raises():
    with pytest.raises(ValueError, match="empty generation"):
        token_margin_uncertainty([])
    with pytest.raises(ValueError, match="empty generation"):
        uncertainty_from_margins([])
    with pytest.raises(ValueError, match="empty generation"):
        uncertainty_from_probs([], [])
    with pytest.raises(ValueError, match="empty generation"):
        uncertainty_from_logprobs([], [])


def test_uncertainty_from_margins_validates():
    assert uncertainty_from_margins([0.2, 0.4]) == pytest.approx(0.7)
    assert uncertainty_from_margins(np.array([1.0])) == 0.0
    with pytest.raises(ValueError, match="margins"):
        uncertainty_from_margins([0.5, 1.5])
    with pytest.raises(ValueError, match="margins"):
        uncertainty_from_margins([-0.1])
    with pytest.raises(ValueError, match="NaN or inf"):
        uncertainty_from_margins([float("nan")])


def test_uncertainty_from_probs_validates():
    with pytest.raises(ValueError, match="length mismatch"):
        uncertainty_from_probs([0.9, 0.8], [0.1])
    with pytest.raises(ValueError, match="p1"):
        uncertainty_from_probs([1.5], [0.1])
    with pytest.raises(ValueError, match="p2"):
        uncertainty_from_probs([0.9], [float("nan")])
    with pytest.raises(ValueError, match="1-D"):
        uncertainty_from_probs([[0.9, 0.1], [0.8, 0.1]], [[0.1, 0.0], [0.1, 0.0]])


# ------------------------------------------------------------ log-probs


def test_logprob_arrays_match_probability_arrays():
    rng = np.random.default_rng(1)
    for _ in range(50):
        p1, p2 = _random_pairs(rng, int(rng.integers(1, 30)))
        p2 = np.maximum(p2, 1e-300)
        u_p = uncertainty_from_probs(p1, p2)
        u_lp = uncertainty_from_logprobs(np.log(p1), np.log(p2))
        assert u_lp == pytest.approx(u_p, abs=1e-12)


def test_logprob_minus_inf_is_probability_zero():
    assert uncertainty_from_logprobs([0.0], [-np.inf]) == 0.0
    assert uncertainty_from_logprobs([math.log(0.6)], [-np.inf]) == pytest.approx(0.4)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.1])
def test_invalid_logprob_arrays(bad):
    with pytest.raises(ValueError, match="lp1"):
        uncertainty_from_logprobs([bad], [-1.0])


def test_logprob_array_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        uncertainty_from_logprobs([-0.1, -0.2], [-1.0])


def test_top2_from_logprobs_sorts_and_needs_two():
    out = top2_from_logprobs([[math.log(0.2), math.log(0.7), math.log(0.05)]])
    assert out[0] == pytest.approx((0.7, 0.2))
    with pytest.raises(ValueError, match="top_logprobs >= 2"):
        top2_from_logprobs([[math.log(0.9)]])


def test_top2_from_logprobs_rejects_bad_values():
    with pytest.raises(ValueError, match="NaN"):
        top2_from_logprobs([[float("nan"), -1.0]])
    with pytest.raises(ValueError, match="above 0"):
        top2_from_logprobs([[0.5, -1.0]])
    # Round-off just above zero is fine.
    assert top2_from_logprobs([[1e-12, -50.0]])[0][0] == pytest.approx(1.0)


# ----------------------------------------------------------------- batch


def test_batch_ragged_matches_loop():
    rng = np.random.default_rng(2)
    lens = rng.integers(1, 25, 200)
    top1, top2 = zip(*(_random_pairs(rng, int(k)) for k in lens))
    got = batch_uncertainty(list(top1), list(top2))
    expected = [uncertainty_from_probs(a, b) for a, b in zip(top1, top2)]
    assert got.shape == (200,)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_batch_padded_with_lengths_ignores_padding():
    rng = np.random.default_rng(3)
    n, t_max = 64, 30
    p1 = np.full((n, t_max), np.nan)
    p2 = np.full((n, t_max), np.nan)
    lens = rng.integers(1, t_max + 1, n)
    for i, k in enumerate(lens):
        p1[i, :k], p2[i, :k] = _random_pairs(rng, int(k))
    got = batch_uncertainty(p1, p2, lens)
    expected = [uncertainty_from_probs(p1[i, :k], p2[i, :k]) for i, k in enumerate(lens)]
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_batch_padded_without_lengths_uses_every_column():
    p1 = np.array([[0.9, 0.6], [0.5, 1.0]])
    p2 = np.array([[0.05, 0.3], [0.5, 0.0]])
    np.testing.assert_allclose(
        batch_uncertainty(p1, p2),
        [uncertainty_from_probs(p1[0], p2[0]), uncertainty_from_probs(p1[1], p2[1])],
    )


def test_batch_list_with_lengths_is_read_as_padded():
    got = batch_uncertainty([[0.9, 0.0], [0.5, 0.0]], [[0.05, 0.0], [0.5, 0.0]], [1, 1])
    np.testing.assert_allclose(got, [0.15, 1.0])


def test_batch_logprobs():
    rng = np.random.default_rng(4)
    p1, p2 = _random_pairs(rng, 12)
    p2 = np.maximum(p2, 1e-12)
    got = batch_uncertainty([np.log(p1)], [np.log(p2)], logprobs=True)
    assert got[0] == pytest.approx(uncertainty_from_probs(p1, p2), abs=1e-12)


def test_batch_empty_sequence_raises_or_fills():
    with pytest.raises(ValueError, match="sequence 1"):
        batch_uncertainty([[0.9], []], [[0.1], []])
    got = batch_uncertainty([[0.9], [], [0.6, 0.5]], [[0.1], [], [0.3, 0.5]], empty_value=1.0)
    np.testing.assert_allclose(got, [0.2, 1.0, 1 - 0.3 / 2])
    padded = batch_uncertainty(
        np.array([[0.9, 0.2], [0.8, 0.1]]),
        np.array([[0.1, 0.1], [0.1, 0.1]]),
        [0, 2],
        empty_value=0.5,
    )
    np.testing.assert_allclose(padded, [0.5, 1 - (0.7 + 0.0) / 2])
    all_empty = batch_uncertainty([[], []], [[], []], empty_value=1.0)
    np.testing.assert_array_equal(all_empty, [1.0, 1.0])


def test_batch_no_sequences():
    assert batch_uncertainty([], []).shape == (0,)
    assert batch_uncertainty(np.zeros((0, 3)), np.zeros((0, 3))).shape == (0,)


@pytest.mark.parametrize("value", [-0.1, 1.5, float("nan"), True])
def test_batch_empty_value_validated(value):
    with pytest.raises(ValueError, match="empty_value"):
        batch_uncertainty([[0.9]], [[0.1]], empty_value=value)


def test_batch_shape_errors():
    with pytest.raises(ValueError, match="2 sequences but top2 has 1"):
        batch_uncertainty([[0.9], [0.8]], [[0.1]])
    with pytest.raises(ValueError, match="sequence 0: top1 has 2 tokens"):
        batch_uncertainty([[0.9, 0.8]], [[0.1]])
    with pytest.raises(ValueError, match="shape mismatch"):
        batch_uncertainty(np.ones((2, 3)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="2-D"):
        batch_uncertainty(np.ones(3), np.zeros(3))
    with pytest.raises(ValueError, match="scalar"):
        batch_uncertainty([0.9, 0.8], [0.1, 0.1])
    with pytest.raises(ValueError, match="list of sequences"):
        batch_uncertainty(5, 5)


@pytest.mark.parametrize(
    ("lengths", "match"),
    [
        ([1], "one per row"),
        ([1, 4], "outside"),
        ([1, -1], "outside"),
        ([1, 1.5], "whole"),
    ],
)
def test_batch_lengths_validated(lengths, match):
    with pytest.raises(ValueError, match=match):
        batch_uncertainty(np.full((2, 3), 0.6), np.full((2, 3), 0.3), lengths)


def test_batch_invalid_probabilities_name_the_position():
    with pytest.raises(ValueError, match=r"top1: sequence 0, token 0: 1\.2 is not"):
        batch_uncertainty([[1.2]], [[0.1]])
    with pytest.raises(ValueError, match=r"top2: sequence 0, token 0: 0\.1 is not a log"):
        batch_uncertainty([[-0.5]], [[0.1]], logprobs=True)
    with pytest.raises(ValueError, match="top1: sequence 2, token 1: nan"):
        batch_uncertainty(
            [[0.9], [], [0.8, float("nan")]], [[0.1], [], [0.1, 0.1]], empty_value=1.0
        )
    padded = np.array([[0.9, 0.8, np.nan], [0.7, 2.0, 0.5]])
    with pytest.raises(ValueError, match=r"top1: sequence 1, token 1: 2\.0"):
        batch_uncertainty(padded, np.zeros((2, 3)), [2, 3])


# -------------------------------------------------------------- adapters


class _LP:
    def __init__(self, lp: float) -> None:
        self.logprob = lp


class _Top:
    def __init__(self, token: str, lp: float) -> None:
        self.token = token
        self.logprob = lp


class _Tok:
    def __init__(self, tops: list) -> None:
        self.top_logprobs = tops


def test_openai_and_vllm_adapters_agree():
    probs = [(0.8, 0.1), (0.55, 0.4), (0.99, 0.004)]
    expected = token_margin_uncertainty(probs)
    content = [
        {
            "token": "a",
            "logprob": math.log(p1),
            "top_logprobs": [
                {"token": "a", "logprob": math.log(p1)},
                {"token": "b", "logprob": math.log(p2)},
            ],
        }
        for p1, p2 in probs
    ]
    sdk = [_Tok([_Top("a", math.log(p1)), _Top("b", math.log(p2))]) for p1, p2 in probs]
    # vLLM returns the sampled token plus the top-k, in any order.
    vllm = [
        {7: _LP(math.log(p2)), 3: _LP(math.log(p1)), 9: _LP(math.log(p2 / 4))} for p1, p2 in probs
    ]
    vllm_floats = [{1: math.log(p1), 2: math.log(p2)} for p1, p2 in probs]
    assert from_openai_logprobs(content) == pytest.approx(expected, abs=1e-12)
    assert from_openai_logprobs(sdk) == pytest.approx(expected, abs=1e-12)
    assert from_vllm_logprobs(vllm) == pytest.approx(expected, abs=1e-12)
    assert from_vllm_logprobs(vllm_floats) == pytest.approx(expected, abs=1e-12)


def test_openai_adapter_errors():
    with pytest.raises(ValueError, match="top_logprobs is missing"):
        from_openai_logprobs([{"token": "a", "logprob": -0.1, "top_logprobs": None}])
    with pytest.raises(ValueError, match="top_logprobs >= 2"):
        from_openai_logprobs([{"top_logprobs": [{"token": "a", "logprob": -0.1}]}])
    with pytest.raises(ValueError, match="empty generation"):
        from_openai_logprobs([])
    with pytest.raises(ValueError, match="top_logprobs is missing"):
        from_openai_logprobs([{"token": "a", "logprob": -0.1}])
    with pytest.raises(ValueError, match=r"token 1: .*no 'logprob' field"):
        from_openai_logprobs(
            [
                {"top_logprobs": [{"logprob": -0.1}, {"logprob": -2.5}]},
                {"top_logprobs": [{"logprob": -0.1}, {"token": "b"}]},
            ]
        )


def test_vllm_adapter_errors():
    with pytest.raises(ValueError, match="no logprobs"):
        from_vllm_logprobs([None])
    with pytest.raises(ValueError, match="need the top-2"):
        from_vllm_logprobs([{1: _LP(-0.2)}])
    with pytest.raises(ValueError, match=r"token 0: .*no 'logprob' field"):
        from_vllm_logprobs([{1: _LP(-0.2), 2: object()}])
    with pytest.raises(ValueError, match="empty generation"):
        from_vllm_logprobs([])
