"""Token-margin uncertainty u(x) (paper Section 4.1, Eq. 4).

For a greedy generation of T tokens with top-1 and top-2 next-token
probabilities ``p_{t,1} >= p_{t,2}`` at position t, the per-token margin is

    m_t = p_{t,1} - p_{t,2}  in [0, 1]

and the query-level uncertainty is

    u(x) = 1 - (1/T) * sum_{t=1..T} m_t.

Larger u(x) means the small model was less decisive. The paper uses greedy
decoding throughout so that p_{t,1} and p_{t,2} are well defined at every
position; the top-1 token is then the generated token.

Token convention
----------------
Count every generated content token and exclude padding and the terminating
EOS/stop token. This is an implementation choice (the paper does not state
it), recorded in ``docs/paper_mapping.md``. Every function in this module
treats its input as the content tokens only: if a serving stack reports a
log-probability for the stop token, drop that position before calling. For
padded batches, pass the number of content tokens per row as ``lengths`` to
:func:`batch_uncertainty`.

Inputs
------
The functions accept whatever the serving stack returns:

* per-token ``(p1, p2)`` probability pairs: :func:`token_margin_uncertainty`;
* arrays of top-1 and top-2 probabilities: :func:`uncertainty_from_probs`;
* arrays of top-1 and top-2 log-probabilities: :func:`uncertainty_from_logprobs`;
* the top-k candidate log-probabilities per position:
  :func:`top2_from_logprobs`;
* OpenAI-compatible chat responses: :func:`from_openai_logprobs`;
* vLLM offline outputs: :func:`from_vllm_logprobs`;
* many sequences at once, ragged or padded: :func:`batch_uncertainty`.

Probabilities may exceed 1 by at most ``1e-9`` (``exp`` round-off); such
values are capped at 1 before the margin is taken. The two values of a pair
may be given in either order: the larger one is treated as the top-1.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ._validation import (
    PROB_ATOL,
    as_1d,
    as_float_array,
    check_finite_scalar,
    check_probabilities,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "batch_uncertainty",
    "from_openai_logprobs",
    "from_vllm_logprobs",
    "margins_from_top2",
    "token_margin_uncertainty",
    "top2_from_logprobs",
    "uncertainty_from_logprobs",
    "uncertainty_from_margins",
    "uncertainty_from_probs",
]

_EMPTY_MSG = (
    "empty generation: u(x) needs at least one generated content token "
    "(Eq. 4 averages over T >= 1 tokens)"
)


def _margins(p1: NDArray[np.float64], p2: NDArray[np.float64]) -> NDArray[np.float64]:
    """Elementwise ``min(max(p1, p2), 1) - min(p1, p2)`` for validated inputs."""
    hi = np.minimum(np.maximum(p1, p2), 1.0)
    lo = np.minimum(p1, p2)
    return np.asarray(hi - lo, dtype=np.float64)


def _logprobs_to_probs(lp: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    """``exp`` of log-probabilities after checking they are valid.

    ``-inf`` (probability 0) is accepted. NaN, ``+inf`` and values above
    ``log(1 + PROB_ATOL)`` are rejected.
    """
    bad = np.isnan(lp) | (lp > math.log1p(PROB_ATOL))
    if bad.any():
        i = int(np.argmax(bad))
        raise ValueError(
            f"{name} must be log-probabilities (<= 0, not NaN); index {i} is {float(lp[i])!r}"
        )
    return np.asarray(np.exp(lp), dtype=np.float64)


def margins_from_top2(top2: Iterable[Sequence[float]]) -> list[float]:
    """Per-token margins ``m_t = p_{t,1} - p_{t,2}`` (Section 4.1).

    Parameters
    ----------
    top2 : iterable of (float, float)
        One pair of top-1 and top-2 next-token probabilities per content
        token. The order inside a pair does not matter.

    Returns
    -------
    list of float
        Margins in ``[0, 1]``, one per token.

    Raises
    ------
    ValueError
        If a pair does not hold two probabilities in ``[0, 1]``.

    Examples
    --------
    >>> margins_from_top2([(0.9, 0.05), (0.3, 0.6)])
    [0.85, 0.3]
    """
    margins: list[float] = []
    for t, pair in enumerate(top2):
        if len(pair) != 2:
            raise ValueError(f"token {t}: expected a (p1, p2) pair, got {pair!r}")
        a, b = float(pair[0]), float(pair[1])
        p1, p2 = (a, b) if a >= b else (b, a)
        if not (0.0 <= p2 <= p1 <= 1.0 + PROB_ATOL):
            raise ValueError(f"token {t}: probabilities must lie in [0, 1], got {pair!r}")
        margins.append(min(1.0, p1) - p2)
    return margins


def uncertainty_from_margins(margins: ArrayLike) -> float:
    """Query-level uncertainty ``u(x) = 1 - mean(m_t)`` (Eq. 4).

    Parameters
    ----------
    margins : array_like of float
        Per-token margins in ``[0, 1]``, content tokens only.

    Returns
    -------
    float
        u(x) in ``[0, 1]``.

    Raises
    ------
    ValueError
        If there are no margins or a margin lies outside ``[0, 1]``.
    """
    m = as_1d(margins, "margins", allow_empty=True)
    if m.size == 0:
        raise ValueError(_EMPTY_MSG)
    check_probabilities(m, "margins", atol=0.0)
    return float(1.0 - np.mean(m))


def token_margin_uncertainty(top2: Iterable[Sequence[float]]) -> float:
    """u(x) from per-token ``(p1, p2)`` probability pairs (Eq. 4).

    Parameters
    ----------
    top2 : iterable of (float, float)
        Top-1 and top-2 next-token probabilities for each content token.

    Returns
    -------
    float
        u(x) in ``[0, 1]``; 0 when every token was certain, 1 when every
        position was an exact tie.

    Examples
    --------
    >>> round(token_margin_uncertainty([(0.9, 0.05), (0.6, 0.3), (0.5, 0.5)]), 6)
    0.616667
    """
    return uncertainty_from_margins(margins_from_top2(top2))


def uncertainty_from_probs(p1: ArrayLike, p2: ArrayLike) -> float:
    """u(x) for one sequence from arrays of top-1 and top-2 probabilities.

    Vectorized form of :func:`token_margin_uncertainty` (Eq. 4).

    Parameters
    ----------
    p1, p2 : array_like of float, shape (T,)
        Top-1 and top-2 next-token probabilities per content token. Swapped
        entries are allowed: the larger value of each pair is the top-1.

    Returns
    -------
    float
        u(x) in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the arrays are empty, differ in length, or hold values outside
        ``[0, 1]`` (``1 + 1e-9`` is tolerated).
    """
    a = check_probabilities(as_1d(p1, "p1", allow_empty=True, finite=False), "p1")
    b = check_probabilities(as_1d(p2, "p2", allow_empty=True, finite=False), "p2")
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: p1 has {a.size}, p2 has {b.size}")
    if a.size == 0:
        raise ValueError(_EMPTY_MSG)
    return float(1.0 - np.mean(_margins(a, b)))


def uncertainty_from_logprobs(lp1: ArrayLike, lp2: ArrayLike) -> float:
    """u(x) for one sequence from arrays of top-1 and top-2 log-probabilities.

    Parameters
    ----------
    lp1, lp2 : array_like of float, shape (T,)
        Natural-log probabilities of the top-1 and top-2 candidates per
        content token. ``-inf`` (probability 0) is allowed.

    Returns
    -------
    float
        u(x) in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the arrays are empty, differ in length, contain NaN or ``+inf``,
        or contain a value above 0 beyond round-off.
    """
    a = as_1d(lp1, "lp1", allow_empty=True, finite=False)
    b = as_1d(lp2, "lp2", allow_empty=True, finite=False)
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: lp1 has {a.size}, lp2 has {b.size}")
    if a.size == 0:
        raise ValueError(_EMPTY_MSG)
    return float(
        1.0 - np.mean(_margins(_logprobs_to_probs(a, "lp1"), _logprobs_to_probs(b, "lp2")))
    )


def _flatten_ragged(seqs: Any, name: str) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Concatenate a list of 1-D sequences and return (values, lengths)."""
    try:
        items = list(seqs)
    except TypeError as exc:
        raise ValueError(f"{name} must be a list of sequences or a 2-D array") from exc
    parts = [as_1d(s, f"{name}[{i}]", allow_empty=True, finite=False) for i, s in enumerate(items)]
    lengths = np.array([p.size for p in parts], dtype=np.int64)
    values = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
    return values, lengths


def batch_uncertainty(
    top1: Any,
    top2: Any,
    lengths: ArrayLike | None = None,
    *,
    logprobs: bool = False,
    empty_value: float | None = None,
) -> NDArray[np.float64]:
    """u(x) for many sequences at once (Eq. 4, vectorized).

    Parameters
    ----------
    top1, top2 : list of 1-D arrays, or 2-D array of shape (n, T_max)
        Top-1 and top-2 next-token probabilities (or log-probabilities with
        ``logprobs=True``) for ``n`` sequences: either a ragged list with one
        1-D array per sequence, or a padded 2-D array. A numeric
        ``numpy.ndarray`` is always read as padded; a list is read as ragged
        unless ``lengths`` is given.
    lengths : array_like of int, shape (n,), optional
        Number of content tokens per row of a padded input. Positions at or
        after ``lengths[i]`` are padding (or the EOS token) and are ignored,
        so their values may be anything, NaN included. Without ``lengths``
        every column of a 2-D array counts.
    logprobs : bool, default False
        Inputs are natural-log probabilities.
    empty_value : float in [0, 1], optional
        Value returned for a sequence with no content tokens. By default an
        empty sequence raises, because Eq. 4 is undefined for T = 0. Pass
        ``1.0`` to mark empty generations as maximally uncertain, so they are
        escalated. This option is an extension, not in the paper.

    Returns
    -------
    numpy.ndarray of float64, shape (n,)
        u(x) per sequence.

    Raises
    ------
    ValueError
        On shape mismatches, invalid probabilities, bad ``lengths``, or an
        empty sequence when ``empty_value`` is None.

    Examples
    --------
    >>> batch_uncertainty([[0.9, 0.6], [0.5]], [[0.05, 0.3], [0.5]]).round(3).tolist()
    [0.425, 1.0]
    """
    fill: float | None = None
    if empty_value is not None:
        fill = check_finite_scalar(empty_value, "empty_value")
        if not 0.0 <= fill <= 1.0:
            raise ValueError(f"empty_value must lie in [0, 1], got {fill!r}")

    padded = (isinstance(top1, np.ndarray) and top1.dtype != object) or lengths is not None
    if padded:
        a2 = as_float_array(top1, "top1", finite=False)
        b2 = as_float_array(top2, "top2", finite=False)
        if a2.ndim != 2:
            raise ValueError(f"padded top1 must be a 2-D array, got shape {a2.shape}")
        if a2.shape != b2.shape:
            raise ValueError(f"shape mismatch: top1 is {a2.shape}, top2 is {b2.shape}")
        n, t_max = a2.shape
        lens = (
            np.full(n, t_max, dtype=np.int64)
            if lengths is None
            else _check_lengths(lengths, n, t_max)
        )
        keep = np.arange(t_max)[None, :] < lens[:, None]
        a, b = a2[keep], b2[keep]
    else:
        a, lens = _flatten_ragged(top1, "top1")
        b, len_b = _flatten_ragged(top2, "top2")
        if lens.size != len_b.size:
            raise ValueError(f"top1 has {lens.size} sequences but top2 has {len_b.size}")
        if not np.array_equal(lens, len_b):
            i = int(np.argmax(lens != len_b))
            raise ValueError(f"sequence {i}: top1 has {lens[i]} tokens but top2 has {len_b[i]}")

    empty = lens == 0
    if empty.any() and fill is None:
        i = int(np.argmax(empty))
        raise ValueError(f"sequence {i}: {_EMPTY_MSG}; pass empty_value to allow it")

    for label, vals in (("top1", a), ("top2", b)):
        if logprobs:
            bad = np.isnan(vals) | (vals > math.log1p(PROB_ATOL))
            kind = "a log-probability (<= 0, not NaN)"
        else:
            bad = ~((vals >= 0.0) & (vals <= 1.0 + PROB_ATOL))
            kind = "a probability in [0, 1]"
        if bad.any():
            i = int(np.argmax(bad))
            seq = int(np.searchsorted(np.cumsum(lens), i, side="right"))
            pos = i - int(np.sum(lens[:seq]))
            raise ValueError(
                f"{label}: sequence {seq}, token {pos}: {float(vals[i])!r} is not {kind}"
            )
    if logprobs:
        a, b = np.exp(a), np.exp(b)
    m = _margins(a, b)

    out = np.empty(lens.size, dtype=np.float64)
    full = ~empty
    if full.any():
        starts = (np.cumsum(lens) - lens)[full]
        out[full] = 1.0 - np.add.reduceat(m, starts) / lens[full]
    if fill is not None:
        out[empty] = fill
    return out


def _check_lengths(lengths: ArrayLike, n: int, t_max: int) -> NDArray[np.int64]:
    """Validate per-row content-token counts for a padded batch."""
    raw = as_1d(lengths, "lengths", allow_empty=True)
    if raw.size != n:
        raise ValueError(f"lengths has {raw.size} entries, expected one per row ({n})")
    if not np.all(raw == np.round(raw)):
        raise ValueError("lengths must be whole numbers")
    lens = raw.astype(np.int64)
    if (lens < 0).any() or (lens > t_max).any():
        i = int(np.argmax((lens < 0) | (lens > t_max)))
        raise ValueError(f"lengths[{i}] = {lens[i]} is outside [0, {t_max}]")
    return lens


def top2_from_logprobs(
    per_token: Iterable[Iterable[float]],
) -> list[tuple[float, float]]:
    """Top-2 probabilities per token from each position's candidate log-probs.

    Parameters
    ----------
    per_token : iterable of iterable of float
        For each content token, the natural-log probabilities of the top-k
        candidates (k >= 2, any order). Extra candidates, such as the sampled
        token that vLLM adds to the top-k, are fine.

    Returns
    -------
    list of (float, float)
        ``(p1, p2)`` per token with ``p1 >= p2``.

    Raises
    ------
    ValueError
        If a position has fewer than two candidates, or a log-probability is
        NaN, ``+inf`` or above 0 beyond round-off.
    """
    out: list[tuple[float, float]] = []
    for t, cands in enumerate(per_token):
        lps = sorted((float(lp) for lp in cands), reverse=True)
        if len(lps) < 2:
            raise ValueError(
                f"token {t}: need the top-2 log-probs, got {len(lps)}; "
                "request top_logprobs >= 2 (OpenAI API) or logprobs >= 2 (vLLM)"
            )
        if any(math.isnan(lp) for lp in lps):
            raise ValueError(f"token {t}: log-prob is NaN")
        if lps[0] > math.log1p(PROB_ATOL):
            raise ValueError(f"token {t}: log-prob {lps[0]!r} is above 0")
        out.append((math.exp(lps[0]), math.exp(lps[1])))
    return out


def _get(obj: Any, name: str) -> Any:
    """Field access that works for SDK objects and plain dicts (None if absent)."""
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _logprob(obj: Any, t: int) -> Any:
    """The ``logprob`` field of one candidate at token ``t``."""
    lp = _get(obj, "logprob")
    if lp is None:
        raise ValueError(f"token {t}: a top_logprobs candidate has no 'logprob' field")
    return lp


def from_openai_logprobs(content: Sequence[Any]) -> float:
    """u(x) from an OpenAI-compatible chat completion.

    Parameters
    ----------
    content : sequence
        ``choice.logprobs.content`` from a response requested with
        ``logprobs=True, top_logprobs=2`` (or more) and temperature 0: one
        entry per generated token, each with a ``top_logprobs`` list of
        ``{token, logprob}`` entries. SDK objects and plain dicts both work.
        Pass content tokens only (see the module docstring). For a whole
        response, streamed chunks, vLLM or llama.cpp servers (which report a
        stop-token entry) use :mod:`ucci.integrations.openai`, which finds the
        content list and can drop that entry.

    Returns
    -------
    float
        u(x) in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the response has no tokens, a token has no ``top_logprobs`` or
        fewer than two of them, or a candidate has no ``logprob``.
    """
    per_token = []
    for t, tok in enumerate(content):
        top = _get(tok, "top_logprobs")
        if top is None:
            raise ValueError(f"token {t}: top_logprobs is missing; request top_logprobs=2")
        per_token.append([_logprob(c, t) for c in top])
    return token_margin_uncertainty(top2_from_logprobs(per_token))


def from_vllm_logprobs(logprobs: Sequence[Any]) -> float:
    """u(x) from vLLM offline inference.

    Parameters
    ----------
    logprobs : sequence of mapping
        ``request_output.outputs[0].logprobs`` from a run with
        ``SamplingParams(temperature=0, logprobs=2)``: one mapping per
        generated token from token id to a ``Logprob`` object (or a float).
        Pass content tokens only: if the last position is the EOS token,
        drop it first (see the module docstring).
        :mod:`ucci.integrations.vllm` reads a whole ``RequestOutput`` and
        drops the stop token automatically.

    Returns
    -------
    float
        u(x) in ``[0, 1]``.

    Raises
    ------
    ValueError
        If there are no tokens, a position is missing (logprobs were not
        requested), a position has fewer than two candidates, or a candidate
        has no ``logprob``.
    """
    per_token = []
    for t, pos in enumerate(logprobs):
        if pos is None:
            raise ValueError(f"token {t}: no logprobs; run vLLM with SamplingParams(logprobs=2)")
        vals = [
            float(v) if isinstance(v, (int, float)) else float(_logprob(v, t)) for v in pos.values()
        ]
        per_token.append(vals)
    return token_margin_uncertainty(top2_from_logprobs(per_token))
