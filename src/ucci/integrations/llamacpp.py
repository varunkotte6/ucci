"""u(x) from the llama.cpp server's native ``/completion`` endpoint.

UCCI's routing signal is the token-margin uncertainty of the small model's
greedy generation (Section 4.1, Eq. 4). The llama.cpp server returns the
top-N next-token probabilities of every generated token when the request sets
``n_probs``:

    POST /completion
    {"prompt": "...", "n_predict": 256, "n_probs": 2, "top_k": 1}

``top_k: 1`` makes decoding greedy. The OpenAI-compatible endpoints
(``/v1/chat/completions``, ``/v1/completions``) are read by
:mod:`ucci.integrations.openai` instead.

Formats
-------
Three shapes of ``completion_probabilities`` exist, and all are accepted:

1. Current servers, since the change in
   https://github.com/ggml-org/llama.cpp/pull/10783 (merged in December 2024),
   with the default ``post_sampling_probs: false``: one entry
   per generated token, ``{"id", "token", "bytes", "logprob",
   "top_logprobs": [{"id", "token", "bytes", "logprob"}, ...]}``. The
   log-probabilities come from a softmax over the raw logits, before any
   sampler, so they are the model's next-token distribution that Eq. 4 uses.
   A probability of exactly 0 is sent as the most negative float32 instead
   of ``-inf``.
2. Current servers with ``post_sampling_probs: true``: ``logprob`` becomes
   ``prob`` and ``top_logprobs`` becomes ``top_probs``, holding the
   probabilities after the sampler chain, with zero-probability candidates
   dropped. Under greedy decoding that leaves a single candidate with
   probability 1, and in general it is not the model's distribution, so this
   format is rejected unless ``allow_post_sampling=True``.
3. Servers before that change: ``{"content": "<token>", "probs": [{"tok_str",
   "prob"}, ...]}`` with exactly ``n_probs`` candidates. These are the
   sampler chain's candidates; the old README states that with
   ``temperature < 0`` decoding is greedy and the probabilities are a plain
   softmax of the logits, which is the setting to use there. A response in
   which every position reports probabilities of exactly 1 and 0 is
   degenerate (post-sampler output of greedy decoding) and is rejected.

Formats checked against the server README and source (September 2026):

* https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
* https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-task.cpp
* https://github.com/ggml-org/llama.cpp/blob/b4000/examples/server/README.md (old format)

The terminating stop token
--------------------------
The server appends the EOS token's entry to ``completion_probabilities``
before it notices the EOS (``process_token`` in
``tools/server/server-context.cpp``), and it trims the entries of a matched
stop word itself. UCCI excludes the terminating EOS token (token convention,
``docs/paper_mapping.md``), so with the default ``drop_stop_token=None`` the
last entry is dropped when the response says it stopped on EOS:
``"stop_type": "eos"`` on current servers, ``"stopped_eos": true`` on old
ones. Pass the whole response to get this; with a bare
``completion_probabilities`` list set ``drop_stop_token`` explicitly.

Streaming
---------
With ``"stream": true`` each server-sent event carries the entries of its
own tokens and the final event carries ``stop_type``; pass the list of
parsed events to :func:`signals_from_llamacpp_stream`.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ._common import (
    PayloadError,
    TokenSignals,
    as_logprob,
    as_sequence,
    get_field,
    require_field,
    summarize_candidates,
)

__all__ = [
    "signals_from_llamacpp",
    "signals_from_llamacpp_stream",
    "u_from_llamacpp",
    "u_from_llamacpp_stream",
]

_HINT = (
    "request n_probs >= 2 with greedy decoding (for example top_k=1) and post_sampling_probs=false"
)


def _prob_to_logprob(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise PayloadError(f"{where} must be a number, got {value!r}")
    p = float(value)
    if math.isnan(p) or not 0.0 <= p <= 1.0 + 1e-6:
        raise PayloadError(f"{where} = {p!r} is not a probability in [0, 1]")
    return -math.inf if p == 0.0 else min(math.log(p), 0.0)


def _parse_entry(
    entry: Any, where: str, allow_post_sampling: bool
) -> tuple[list[float], float | None, str]:
    """Candidates, generated-token log-prob and format name for one entry."""
    if not isinstance(entry, Mapping):
        raise PayloadError(f"{where} must be an object, got {entry!r}")
    if "top_logprobs" in entry:
        top = as_sequence(entry["top_logprobs"], f"{where}.top_logprobs")
        cands = [
            as_logprob(
                require_field(c, "logprob", f"{where}.top_logprobs[{j}]"),
                f"{where}.top_logprobs[{j}].logprob",
            )
            for j, c in enumerate(top)
        ]
        lp = entry.get("logprob")
        return cands, (None if lp is None else as_logprob(lp, f"{where}.logprob")), "logprobs"
    if "top_probs" in entry:
        if not allow_post_sampling:
            raise PayloadError(
                f"{where} holds post-sampling probabilities (post_sampling_probs=true). "
                "They are not the model's next-token distribution used by Eq. 4 and "
                "collapse to one candidate under greedy decoding; request with "
                "post_sampling_probs=false, or pass allow_post_sampling=True"
            )
        top = as_sequence(entry["top_probs"], f"{where}.top_probs")
        cands = [
            _prob_to_logprob(
                require_field(c, "prob", f"{where}.top_probs[{j}]"), f"{where}.top_probs[{j}].prob"
            )
            for j, c in enumerate(top)
        ]
        p = entry.get("prob")
        return cands, (None if p is None else _prob_to_logprob(p, f"{where}.prob")), "post"
    if "probs" in entry:
        top = as_sequence(entry["probs"], f"{where}.probs")
        cands = []
        sampled: float | None = None
        tok = entry.get("content")
        for j, c in enumerate(top):
            lp = _prob_to_logprob(
                require_field(c, "prob", f"{where}.probs[{j}]"), f"{where}.probs[{j}].prob"
            )
            cands.append(lp)
            if sampled is None and tok is not None and get_field(c, "tok_str") == tok:
                sampled = lp
        return cands, sampled, "legacy"
    raise PayloadError(f"{where} has none of top_logprobs, top_probs or probs; {_HINT}")


def _summarize(
    entries: Sequence[Any],
    paths: Sequence[str],
    drop: bool,
    *,
    source: str,
    require_greedy: bool,
    allow_post_sampling: bool,
    renormalize_entropy: bool,
) -> TokenSignals:
    positions: list[list[float]] = []
    sampled: list[float | None] = []
    formats: set[str] = set()
    for entry, where in zip(entries, paths):
        cands, s, fmt = _parse_entry(entry, where, allow_post_sampling)
        positions.append(cands)
        sampled.append(s)
        formats.add(fmt)
    labels = list(paths)
    if drop and positions:
        del positions[-1], sampled[-1], labels[-1]
    if (
        "legacy" in formats
        and positions
        and all(len(c) >= 2 and max(c) == 0.0 and sorted(c)[-2] == -math.inf for c in positions)
    ):
        raise PayloadError(
            f"{source}: every position reports probabilities of exactly 1 and 0. These are "
            "post-sampler probabilities of greedy decoding on a pre-December-2024 server; "
            "on those servers request temperature < 0 (greedy, with softmax "
            "probabilities), or upgrade the server"
        )
    return summarize_candidates(
        positions,
        sampled,
        source=source,
        hint=_HINT,
        require_greedy=require_greedy,
        renormalize_entropy=renormalize_entropy,
        labels=labels,
    )


def _stopped_on_eos(obj: Any) -> bool:
    return get_field(obj, "stop_type") == "eos" or get_field(obj, "stopped_eos") is True


def signals_from_llamacpp(
    response: Any,
    *,
    drop_stop_token: bool | None = None,
    require_greedy: bool = False,
    allow_post_sampling: bool = False,
    renormalize_entropy: bool = False,
) -> TokenSignals:
    """Token signals of one non-streamed ``/completion`` response (Eq. 4).

    Parameters
    ----------
    response : dict or list
        The parsed JSON response (with ``completion_probabilities``), or the
        ``completion_probabilities`` list itself.
    drop_stop_token : bool or None, default None
        None: drop the last entry when the response reports an EOS stop
        (``stop_type == "eos"`` or ``stopped_eos``). True: drop it
        unconditionally. False: keep every entry.
    require_greedy : bool, default False
        Raise if a generated token is not the top-1 candidate.
    allow_post_sampling : bool, default False
        Accept ``post_sampling_probs=true`` responses (see the module
        docstring for why they are rejected by default).
    renormalize_entropy : bool, default False
        Entropy of the renormalized top-N instead of the truncated sum.

    Returns
    -------
    TokenSignals

    Raises
    ------
    PayloadError
        If ``n_probs`` was not set, a position has fewer than two candidates,
        the probabilities are post-sampling (see above), or the payload is
        malformed.
    """
    if isinstance(response, Mapping):
        probs = response.get("completion_probabilities")
        if probs is None:
            raise PayloadError(f"response.completion_probabilities is missing; {_HINT}")
        entries = as_sequence(probs, "completion_probabilities")
        auto = _stopped_on_eos(response)
    else:
        entries = as_sequence(response, "completion_probabilities")
        auto = False
    drop = auto if drop_stop_token is None else drop_stop_token
    paths = [f"completion_probabilities[{t}]" for t in range(len(entries))]
    return _summarize(
        entries,
        paths,
        drop,
        source="llamacpp.completion",
        require_greedy=require_greedy,
        allow_post_sampling=allow_post_sampling,
        renormalize_entropy=renormalize_entropy,
    )


def u_from_llamacpp(response: Any, **kwargs: Any) -> float:
    """u(x) (Eq. 4) of one ``/completion`` response; see :func:`signals_from_llamacpp`."""
    return signals_from_llamacpp(response, **kwargs).u


def signals_from_llamacpp_stream(
    events: Iterable[Any],
    *,
    drop_stop_token: bool | None = None,
    require_greedy: bool = False,
    allow_post_sampling: bool = False,
    renormalize_entropy: bool = False,
) -> TokenSignals:
    """Token signals of a streamed ``/completion`` response (``"stream": true``).

    Parameters
    ----------
    events : iterable of dict
        The parsed ``data:`` payloads in arrival order. Entries of
        ``completion_probabilities`` are concatenated; ``stop_type`` (or
        ``stopped_eos``) is read from whichever event carries it.
    drop_stop_token, require_greedy, allow_post_sampling, renormalize_entropy
        As in :func:`signals_from_llamacpp`.

    Returns
    -------
    TokenSignals
    """
    entries: list[Any] = []
    paths: list[str] = []
    eos = False
    for i, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise PayloadError(f"events[{i}] must be an object, got {event!r}")
        eos = eos or _stopped_on_eos(event)
        probs = event.get("completion_probabilities")
        if probs:
            for t, e in enumerate(as_sequence(probs, f"events[{i}].completion_probabilities")):
                entries.append(e)
                paths.append(f"events[{i}].completion_probabilities[{t}]")
    drop = eos if drop_stop_token is None else drop_stop_token
    return _summarize(
        entries,
        paths,
        drop,
        source="llamacpp.completion_stream",
        require_greedy=require_greedy,
        allow_post_sampling=allow_post_sampling,
        renormalize_entropy=renormalize_entropy,
    )


def u_from_llamacpp_stream(events: Iterable[Any], **kwargs: Any) -> float:
    """u(x) (Eq. 4) of a streamed ``/completion`` response."""
    return signals_from_llamacpp_stream(events, **kwargs).u
