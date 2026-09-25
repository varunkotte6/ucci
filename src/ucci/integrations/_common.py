"""Shared machinery for the serving-stack adapters (private module).

Every adapter in :mod:`ucci.integrations` turns a provider payload into, for
each generated content token t, the log-probabilities of the top-k candidate
next tokens. :func:`summarize_candidates` then computes the paper's signal

    m_t = p_{t,1} - p_{t,2},    u(x) = 1 - (1/T) * sum_t m_t

(Section 4.1, Eq. 4) through :func:`ucci.signal.uncertainty_from_margins`, so
there is one implementation of Eq. 4 in the package. It also computes the two
alternative signals of the Section 6.3 ablation and the entropy baseline of
Section 6.1 (mean token entropy and mean max probability), and counts
positions that were not decoded greedily.

The names here may change between releases. The public surface is
:class:`TokenSignals` and :class:`PayloadError`, re-exported by
:mod:`ucci.integrations`.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..signal import uncertainty_from_margins

__all__ = [
    "GREEDY_ATOL",
    "LOGPROB_ATOL",
    "MASS_ATOL",
    "PayloadError",
    "TokenSignals",
    "as_logprob",
    "as_sequence",
    "empty_signals",
    "get_field",
    "require_field",
    "summarize_candidates",
]

#: Log-probabilities up to this far above 0 are treated as float round-off
#: and clamped to 0. Anything larger is rejected.
LOGPROB_ATOL = 1e-6

#: The candidate probabilities reported at one position belong to one
#: distribution over distinct tokens, so they sum to at most 1. A sum above
#: ``1 + MASS_ATOL`` means the values are not log-probabilities (for example
#: raw logits) and is rejected.
MASS_ATOL = 1e-3

#: A generated token whose log-probability is more than this below the top-1
#: candidate's counts as not greedily decoded.
GREEDY_ATOL = 1e-6

_MISSING: Any = object()


class PayloadError(ValueError):
    """A provider payload does not have the documented shape or values.

    Subclass of :class:`ValueError`, so ``except ValueError`` also catches it.
    The message names the offending path inside the payload (for example
    ``choices[0].logprobs.content[3].top_logprobs``) and says how to fix the
    request when the cause is a missing request option.
    """


@dataclass(frozen=True)
class TokenSignals:
    """Per-query uncertainty signals extracted from one greedy generation.

    Attributes
    ----------
    u : float
        Token-margin uncertainty u(x) = 1 - mean_t (p_{t,1} - p_{t,2}), the
        UCCI routing signal (Section 4.1, Eq. 4). In ``[0, 1]``; NaN only for
        an empty generation when the caller asked for NaN instead of an error.
    n_tokens : int
        T, the number of content tokens averaged over (EOS and padding
        excluded, see ``docs/paper_mapping.md``).
    mean_max_prob : float
        Mean over tokens of the top-1 probability p_{t,1}. This is the
        "max probability" signal of the Section 6.3 ablation (higher means
        more confident). It is not used by UCCI itself.
    mean_entropy : float
        Mean over tokens of the next-token entropy in nats, the signal of the
        "entropy threshold" baseline (Section 6.1) and of the Section 6.3
        ablation. See ``entropy_support`` for what it is computed over.
    entropy_support : str
        ``"full_vocabulary"`` when the adapter saw every logit (Hugging Face
        transformers); ``"top_k"`` when only the top-k candidates were
        reported, in which case the entropy is the truncated sum
        ``-sum_{j <= k} p_j log p_j``, a lower bound on the full entropy that
        tightens as k grows; ``"top_k_renormalized"`` for the entropy of the
        top-k probabilities rescaled to sum to 1. The paper does not say how
        its entropy baseline was computed, so this is an implementation
        choice, not a paper detail. :func:`ucci.baselines.mean_token_entropy`
        uses the renormalized definition; request
        ``renormalize_entropy=True`` from the adapter to match it.
    top_k : int or None
        Smallest number of candidates reported at any position (None for the
        full vocabulary).
    n_non_greedy : int or None
        Number of positions where the generated token was not the top-1
        candidate. The paper decodes greedily (Section 4.1 and Appendix B.2),
        so this should be 0; a positive count usually means sampling was on
        or a logits processor changed the argmax. None when the payload does
        not identify the generated token.
    margins : tuple of float
        The per-token margins m_t, in generation order.
    source : str
        Which adapter and field produced the numbers, for example
        ``"openai.chat_completion"`` or ``"transformers.logits"``.
    """

    u: float
    n_tokens: int
    mean_max_prob: float
    mean_entropy: float
    entropy_support: str
    top_k: int | None
    n_non_greedy: int | None
    margins: tuple[float, ...] = field(repr=False)
    source: str = ""

    def to_record(self) -> dict[str, float]:
        """Signal fields of the JSONL record format used by ``ucci`` tools.

        Returns
        -------
        dict
            ``{"u": ..., "entropy": ..., "max_prob": ...}``, ready to merge
            into a record written by :class:`ucci.integrations.cascade.JsonlLogger`.
        """
        return {"u": self.u, "entropy": self.mean_entropy, "max_prob": self.mean_max_prob}


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping or an object, returning ``default`` if absent.

    Works for plain dicts decoded from JSON, pydantic models (the OpenAI SDK),
    dataclasses (vLLM) and ``ModelOutput`` (transformers, which is a mapping
    that omits keys whose value is None).
    """
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def require_field(obj: Any, name: str, path: str, hint: str = "") -> Any:
    """Like :func:`get_field` but raise :class:`PayloadError` when absent or None."""
    value = get_field(obj, name, _MISSING)
    if value is _MISSING or value is None:
        what = "missing" if value is _MISSING else "null"
        suffix = f"; {hint}" if hint else ""
        raise PayloadError(f"{path}.{name} is {what}{suffix}")
    return value


def as_sequence(value: Any, path: str) -> Sequence[Any]:
    """Return ``value`` if it is a list-like sequence, else raise PayloadError.

    Strings, bytes and mappings are rejected even though they are sequences or
    iterables, because a payload field that should hold per-token entries
    never legitimately holds one of those.
    """
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Sequence):
        raise PayloadError(f"{path} must be a list, got {type(value).__name__}")
    return value


def as_logprob(value: Any, path: str) -> float:
    """Validate one natural-log probability from a payload.

    ``-inf`` (probability 0) is accepted. Values in ``(0, LOGPROB_ATOL]`` are
    round-off and are clamped to 0. Booleans, non-numbers, NaN, ``+inf`` and
    larger positive values raise :class:`PayloadError`.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise PayloadError(f"{path} must be a number, got {value!r}")
    x = float(value)
    if math.isnan(x) or x == math.inf:
        raise PayloadError(f"{path} must be a log-probability, got {x!r}")
    if x > LOGPROB_ATOL:
        raise PayloadError(
            f"{path} = {x!r} is above 0, so it is not a log-probability "
            "(raw logits or unnormalized scores were probably passed)"
        )
    return min(x, 0.0)


def summarize_candidates(
    positions: Sequence[Sequence[float]],
    sampled: Sequence[float | None] | None = None,
    *,
    source: str,
    hint: str,
    require_greedy: bool = False,
    renormalize_entropy: bool = False,
    labels: Sequence[str] | None = None,
) -> TokenSignals:
    """Compute :class:`TokenSignals` from per-position candidate log-probs.

    Parameters
    ----------
    positions : sequence of sequence of float
        For each content token, the validated log-probabilities of the
        reported candidates (top-k, possibly plus the generated token), in
        any order. Every position needs at least two candidates.
    sampled : sequence of float or None, optional
        Log-probability of the generated token at each position, used only to
        count non-greedy positions. Entries may be None where unknown.
    source : str
        Stored in :attr:`TokenSignals.source` and used in error messages.
    hint : str
        How to request top-2 log-probabilities from this server, appended to
        the error raised when a position has fewer than two candidates.
    require_greedy : bool, default False
        Raise if any position was not decoded greedily.
    renormalize_entropy : bool, default False
        Rescale each position's candidate probabilities to sum to 1 before
        taking the entropy (``entropy_support="top_k_renormalized"``).
    labels : sequence of str, optional
        Payload path of each position, for error messages.

    Returns
    -------
    TokenSignals

    Raises
    ------
    PayloadError
        On an empty generation, a position with fewer than two candidates,
        candidate probabilities summing above 1, or (with ``require_greedy``)
        a non-greedy position.
    """
    n = len(positions)
    if n == 0:
        raise PayloadError(
            f"{source}: no generated content tokens with log-probabilities; u(x) "
            "averages over T >= 1 tokens (Eq. 4)"
        )
    if sampled is not None and len(sampled) != n:
        raise PayloadError(f"{source}: {len(sampled)} generated-token log-probs for {n} positions")

    margins: list[float] = []
    max_probs: list[float] = []
    entropies: list[float] = []
    top_k = min(len(c) for c in positions)
    non_greedy = 0 if sampled is not None else None

    for t, cands in enumerate(positions):
        where = labels[t] if labels is not None else f"{source} token {t}"
        if len(cands) < 2:
            raise PayloadError(
                f"{where}: {len(cands)} candidate log-prob(s) reported, the top-2 "
                f"margin (Eq. 4) needs at least 2; {hint}"
            )
        lps = sorted(cands, reverse=True)
        probs = [math.exp(lp) for lp in lps]
        mass = math.fsum(probs)
        if mass > 1.0 + MASS_ATOL:
            raise PayloadError(
                f"{where}: candidate probabilities sum to {mass:.6f} > 1, so these "
                "are not log-probabilities of one next-token distribution"
            )
        if probs[0] == 0.0:
            raise PayloadError(f"{where}: every candidate has probability 0")
        p1 = min(probs[0], 1.0)
        margins.append(p1 - min(probs[1], p1))
        max_probs.append(p1)
        if renormalize_entropy:
            log_mass = math.log(mass)
            entropies.append(
                -math.fsum((p / mass) * (lp - log_mass) for p, lp in zip(probs, lps) if p > 0.0)
            )
        else:
            entropies.append(-math.fsum(p * lp for p, lp in zip(probs, lps) if p > 0.0))
        if sampled is not None and non_greedy is not None:
            s = sampled[t]
            if s is not None and s < lps[0] - GREEDY_ATOL:
                non_greedy += 1

    if require_greedy and non_greedy:
        raise PayloadError(
            f"{source}: {non_greedy} of {n} tokens were not the top-1 candidate. "
            "UCCI's signal is defined under greedy decoding (Section 4.1): request "
            "temperature 0 and no sampling"
        )

    return TokenSignals(
        u=uncertainty_from_margins(margins),
        n_tokens=n,
        mean_max_prob=math.fsum(max_probs) / n,
        mean_entropy=math.fsum(entropies) / n,
        entropy_support="top_k_renormalized" if renormalize_entropy else "top_k",
        top_k=top_k,
        n_non_greedy=non_greedy,
        margins=tuple(margins),
        source=source,
    )


def empty_signals(source: str, entropy_support: str, top_k: int | None) -> TokenSignals:
    """NaN-valued :class:`TokenSignals` for an empty generation (T = 0)."""
    nan = float("nan")
    return TokenSignals(
        u=nan,
        n_tokens=0,
        mean_max_prob=nan,
        mean_entropy=nan,
        entropy_support=entropy_support,
        top_k=top_k,
        n_non_greedy=0,
        margins=(),
        source=source,
    )
