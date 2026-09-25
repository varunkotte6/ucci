"""u(x) from vLLM offline inference (``LLM.generate``).

UCCI's routing signal is the token-margin uncertainty of the small model's
greedy generation (Section 4.1, Eq. 4). The paper served both models with
vLLM (Section 6.1, Appendix B.2). Run the small model with

    SamplingParams(temperature=0, logprobs=2, max_tokens=256)

(:func:`greedy_sampling_params` builds exactly this) and pass each
``RequestOutput`` to :func:`u_from_vllm`.

Format
------
``request_output.outputs[i]`` is a ``CompletionOutput`` with ``token_ids``,
``logprobs``, ``finish_reason`` and ``stop_reason``. ``logprobs`` holds one
entry per generated token: a dict from token id to a ``Logprob`` object with
fields ``logprob``, ``rank`` and ``decoded_token`` (recent releases may use
``FlatLogprobs``, a sequence that yields the same dicts). With
``logprobs=k`` vLLM returns the top-k candidates plus the sampled token, so a
position has k or k + 1 entries. Under greedy decoding the sampled token is
the top-1, so ``logprobs=2`` gives exactly the top-1 and top-2 that Eq. 4
needs, while ``logprobs=1`` gives only one. Checked against the vLLM source
(September 2026):

* ``SamplingParams.logprobs``:
  https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
* ``Logprob``, ``FlatLogprobs``:
  https://github.com/vllm-project/vllm/blob/main/vllm/logprobs.py
* ``CompletionOutput``:
  https://github.com/vllm-project/vllm/blob/main/vllm/outputs.py

The values are the model's log-probabilities before any logits processor,
since vLLM's default ``logprobs_mode`` is ``"raw_logprobs"``
(https://github.com/vllm-project/vllm/blob/main/docs/usage/v1_guide.md). The
``"raw_logits"`` and ``"processed_logits"`` modes return logits, which are
not log-probabilities; the adapter rejects values above 0 and candidate sets
whose probabilities sum above 1, which catches them.

The terminating stop token
--------------------------
vLLM keeps the token that ended generation in ``token_ids`` and
``logprobs`` (``vllm/v1/core/sched/utils.py`` and
``vllm/v1/engine/detokenizer.py``). UCCI excludes the terminating EOS/stop
token (token convention, ``docs/paper_mapping.md``). With the default
``drop_stop_token=None`` the adapter drops the last position exactly when
vLLM reports that generation ended on a token: ``finish_reason == "stop"``
with ``stop_reason`` None (the EOS token) or an int (a ``stop_token_ids``
entry). It keeps every position when generation hit ``max_tokens`` or a stop
string (``stop_reason`` is the string; the string's tokens were generated
content).

Examples
--------
>>> from vllm import LLM                                     # doctest: +SKIP
>>> from ucci.integrations.vllm import greedy_sampling_params, u_from_vllm_batch
>>> outs = LLM("small-model").generate(prompts, greedy_sampling_params())  # doctest: +SKIP
>>> us = u_from_vllm_batch(outs)                             # doctest: +SKIP
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ._common import (
    PayloadError,
    TokenSignals,
    as_logprob,
    as_sequence,
    get_field,
    summarize_candidates,
)

__all__ = [
    "greedy_sampling_params",
    "signals_from_vllm",
    "signals_from_vllm_batch",
    "u_from_vllm",
    "u_from_vllm_batch",
]

_HINT = (
    "run vLLM with SamplingParams(temperature=0, logprobs=2) and keep logprobs_mode "
    "at its default 'raw_logprobs'"
)
_MISSING: Any = object()


def greedy_sampling_params(max_tokens: int = 256, logprobs: int = 2, **kwargs: Any) -> Any:
    """``vllm.SamplingParams`` for the UCCI small model.

    Parameters
    ----------
    max_tokens : int, default 256
        The paper's generation limit (Appendix B.2).
    logprobs : int, default 2
        Candidates per token; at least 2 for the top-2 margin (Eq. 4).
    **kwargs
        Any other ``SamplingParams`` argument (``stop``, ``seed``, ...).

    Returns
    -------
    vllm.SamplingParams
        With ``temperature=0`` (greedy, Section 4.1).

    Raises
    ------
    ImportError
        If vLLM is not installed.
    ValueError
        If ``logprobs < 2`` or ``temperature`` is passed.
    """
    if (
        isinstance(logprobs, bool)
        or not isinstance(logprobs, int)
        or (logprobs != -1 and logprobs < 2)
    ):
        raise ValueError(f"logprobs must be an int >= 2 (or -1 for all), got {logprobs!r}")
    if "temperature" in kwargs:
        raise ValueError("temperature is fixed at 0: UCCI's signal uses greedy decoding")
    try:
        from vllm import SamplingParams
    except ImportError as exc:  # pragma: no cover - exercised only without vLLM
        raise ImportError(
            "greedy_sampling_params needs vLLM; install it with `pip install vllm`"
        ) from exc
    return SamplingParams(temperature=0.0, logprobs=logprobs, max_tokens=max_tokens, **kwargs)


def _lp_value(value: Any, where: str) -> float:
    """Log-prob from a ``Logprob`` object, a ``{"logprob": ...}`` dict or a bare float."""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return as_logprob(value, where)
    lp = get_field(value, "logprob", _MISSING)
    if lp is _MISSING:
        raise PayloadError(f"{where} must be a Logprob or a number, got {value!r}")
    return as_logprob(lp, f"{where}.logprob")


def _resolve(output: Any, completion_index: int) -> tuple[Any, Any, Any, Any, str]:
    """Return (logprobs, token_ids, finish_reason, stop_reason, path) for one completion."""
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes, bytearray)):
        if len(output) and get_field(output[0], "outputs") is not None:
            raise PayloadError(
                "got a list of RequestOutput objects (the result of LLM.generate); use "
                "signals_from_vllm_batch or u_from_vllm_batch for a list"
            )
        return output, None, None, None, "logprobs"
    outputs = get_field(output, "outputs")
    if outputs is not None:
        seq = as_sequence(outputs, "outputs")
        if not 0 <= completion_index < len(seq):
            raise PayloadError(
                f"completion_index {completion_index} is out of range for {len(seq)} outputs"
            )
        comp, path = seq[completion_index], f"outputs[{completion_index}]"
    else:
        comp, path = output, "output"
    logprobs = get_field(comp, "logprobs", _MISSING)
    if logprobs is _MISSING:
        raise PayloadError(
            f"expected a vLLM RequestOutput, CompletionOutput or logprobs list; got "
            f"{type(output).__name__}"
        )
    if logprobs is None:
        raise PayloadError(f"{path}.logprobs is None; {_HINT}")
    return (
        logprobs,
        get_field(comp, "token_ids"),
        get_field(comp, "finish_reason"),
        get_field(comp, "stop_reason"),
        f"{path}.logprobs",
    )


def signals_from_vllm(
    output: Any,
    *,
    completion_index: int = 0,
    drop_stop_token: bool | None = None,
    require_greedy: bool = False,
    renormalize_entropy: bool = False,
) -> TokenSignals:
    """Token signals of one vLLM generation (Section 4.1, Eq. 4).

    Parameters
    ----------
    output : RequestOutput, CompletionOutput, or logprobs sequence
        One element of the list ``LLM.generate`` returns, one of its
        ``outputs``, or ``outputs[i].logprobs`` itself. Dicts with the same
        fields (for example from a JSON dump, with string token ids) work too.
    completion_index : int, default 0
        Which of ``outputs`` to read when ``SamplingParams.n > 1``.
    drop_stop_token : bool or None, default None
        None: drop the last position exactly when vLLM reports that
        generation ended on the EOS token or a ``stop_token_ids`` entry (see
        the module docstring). True: drop it whenever ``finish_reason`` is
        ``"stop"``, or unconditionally for a bare logprobs sequence. False:
        keep every position.
    require_greedy : bool, default False
        Raise if a generated token is not the top-1 candidate (needs
        ``token_ids``).
    renormalize_entropy : bool, default False
        Entropy of the renormalized top-k instead of the truncated sum.

    Returns
    -------
    TokenSignals

    Raises
    ------
    PayloadError
        If logprobs were not requested, a position has fewer than two
        candidates, the values are logits rather than log-probabilities, or
        ``token_ids`` and ``logprobs`` disagree in length.
    """
    logprobs, token_ids, finish, stop_reason, path = _resolve(output, completion_index)
    positions_raw = list(logprobs)
    ids: list[Any] | None = None
    if token_ids is not None:
        ids = list(token_ids)
        if len(ids) != len(positions_raw):
            raise PayloadError(
                f"{path} has {len(positions_raw)} positions but token_ids has {len(ids)}"
            )

    if drop_stop_token is None:
        ends_on_token = stop_reason is None or (
            isinstance(stop_reason, int) and not isinstance(stop_reason, bool)
        )
        drop = finish == "stop" and ends_on_token
    elif drop_stop_token:
        drop = finish in ("stop", None)
    else:
        drop = False

    positions: list[list[float]] = []
    sampled: list[float | None] = []
    labels: list[str] = []
    for t, pos in enumerate(positions_raw):
        where = f"{path}[{t}]"
        if pos is None:
            raise PayloadError(f"{where} is None; {_HINT}")
        if not isinstance(pos, Mapping):
            raise PayloadError(f"{where} must map token ids to Logprob objects, got {pos!r}")
        positions.append([_lp_value(v, f"{where}[{k!r}]") for k, v in pos.items()])
        s: float | None = None
        if ids is not None:
            tok = ids[t]
            hit = pos.get(tok, _MISSING)
            if hit is _MISSING:
                hit = pos.get(str(tok), _MISSING)
            if hit is not _MISSING:
                s = _lp_value(hit, f"{where}[{tok!r}]")
        sampled.append(s)
        labels.append(where)
    if drop and positions:
        del positions[-1], sampled[-1], labels[-1]
    return summarize_candidates(
        positions,
        sampled if ids is not None else None,
        source="vllm",
        hint=_HINT,
        require_greedy=require_greedy,
        renormalize_entropy=renormalize_entropy,
        labels=labels,
    )


def u_from_vllm(output: Any, **kwargs: Any) -> float:
    """u(x) (Eq. 4) of one vLLM generation; see :func:`signals_from_vllm`."""
    return signals_from_vllm(output, **kwargs).u


def signals_from_vllm_batch(outputs: Iterable[Any], **kwargs: Any) -> list[TokenSignals]:
    """:func:`signals_from_vllm` for every ``RequestOutput`` of ``LLM.generate``."""
    return [signals_from_vllm(o, **kwargs) for o in outputs]


def u_from_vllm_batch(outputs: Iterable[Any], **kwargs: Any) -> list[float]:
    """u(x) (Eq. 4) for every ``RequestOutput`` of ``LLM.generate``, in order."""
    return [signals_from_vllm(o, **kwargs).u for o in outputs]
