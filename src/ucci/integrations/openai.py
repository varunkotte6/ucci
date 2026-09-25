"""u(x) from OpenAI-format responses (Chat Completions, Completions, Responses).

UCCI's routing signal is the token-margin uncertainty of the small model's
greedy generation (Section 4.1, Eq. 4):

    m_t = p_{t,1} - p_{t,2},    u(x) = 1 - (1/T) * sum_t m_t.

OpenAI-format APIs return the top-k next-token log-probabilities for every
generated token when asked, so u(x) costs nothing beyond the small model's
own call. Request greedy decoding and at least two candidates per token:

* Chat Completions: ``temperature=0, logprobs=True, top_logprobs=2`` (0 to
  20). Each ``choice.logprobs.content[i]`` holds ``token``, ``logprob``,
  ``bytes`` and ``top_logprobs``, a list of ``{token, logprob, bytes}``. The
  API reports ``-9999.0`` for a token outside the top 20.
* Completions (legacy): ``temperature=0, logprobs=2`` (an integer, at most
  5). ``choice.logprobs`` holds parallel lists ``tokens``,
  ``token_logprobs`` and ``top_logprobs``, the last a list of
  ``{token: logprob}`` dicts that also contain the sampled token, so a dict
  may have ``logprobs + 1`` entries.
* Responses: ``temperature=0, top_logprobs=2,
  include=["message.output_text.logprobs"]``. Every ``output_text`` part of
  every ``message`` item in ``response.output`` then carries ``logprobs``, a
  list of ``{token, logprob, bytes, top_logprobs}`` entries, which are
  concatenated in order. The Responses API does expose per-token top
  log-probabilities, so it is supported.

Response formats were checked against the official API reference and the
official Python SDK types (September 2026):

* https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
* https://developers.openai.com/api/reference/resources/completions/methods/create
* https://developers.openai.com/api/reference/resources/responses/methods/create
* https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion_token_logprob.py
* https://github.com/openai/openai-python/blob/main/src/openai/types/completion_choice.py
* https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_output_text.py

Every function accepts SDK objects (pydantic models) or the equivalent
plain dicts, for example parsed from a JSON log. The ``openai`` package is
never imported.

OpenAI-compatible servers
-------------------------
The same functions read any server that returns this format. Servers whose
documentation or source confirms logprobs support:

* vLLM's OpenAI-compatible server (Chat Completions and Completions), whose
  documentation declares compatibility with both APIs and whose request
  schema carries ``logprobs`` and ``top_logprobs``:
  https://github.com/vllm-project/vllm/blob/main/docs/serving/online_serving/openai_compatible_server.md
  and https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/chat_completion/protocol.py
* the llama.cpp server's ``/v1/chat/completions`` endpoint, whose logprobs
  support was added in https://github.com/ggml-org/llama.cpp/pull/10783
  (for its native ``/completion`` endpoint see :mod:`ucci.integrations.llamacpp`);
* the LiteLLM proxy, which accepts ``logprobs`` and ``top_logprobs``
  (https://docs.litellm.ai/docs/completion/input); whether values come back
  depends on the provider behind it.

The terminating stop token
--------------------------
UCCI counts every generated content token and excludes the terminating
EOS/stop token (token convention, ``docs/paper_mapping.md``). The OpenAI API
does not report a log-probability for the stop token. vLLM and llama.cpp do:
their OpenAI endpoints walk the generated token ids, which end with the EOS
token when generation stopped on it (vLLM:
``vllm/entrypoints/openai/chat_completion/serving.py``; llama.cpp:
``tools/server/server-context.cpp``). For those servers pass
``drop_stop_token=True``: the last entry is then dropped whenever
``finish_reason`` is ``"stop"`` (``status == "completed"`` for the Responses
API), and kept when generation hit the token limit.

Greedy decoding
---------------
The paper decodes greedily (Section 4.1, Appendix B.2). Each result reports
:attr:`~ucci.integrations.TokenSignals.n_non_greedy`, the number of tokens
whose log-probability is below the top-1 candidate's; pass
``require_greedy=True`` to raise instead. This matters for vLLM in
particular: its ``top_logprobs`` list starts with the sampled token, so
under sampling the "top-2" it returns can be (sampled, top-1).

Examples
--------
>>> from openai import OpenAI                                    # doctest: +SKIP
>>> from ucci.integrations.openai import u_from_chat_completion  # doctest: +SKIP
>>> resp = OpenAI().chat.completions.create(                     # doctest: +SKIP
...     model="gpt-4o-mini", messages=[{"role": "user", "content": "Hi"}],
...     temperature=0, logprobs=True, top_logprobs=2)
>>> u = u_from_chat_completion(resp)                             # doctest: +SKIP
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable

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
    "make_chat_fn",
    "signals_from_chat_completion",
    "signals_from_chat_completion_chunks",
    "signals_from_completion",
    "signals_from_openai",
    "signals_from_responses",
    "u_from_chat_completion",
    "u_from_chat_completion_chunks",
    "u_from_completion",
    "u_from_openai",
    "u_from_responses",
]

_CHAT_HINT = "request logprobs=True and top_logprobs=2 (or more) with temperature=0"
_COMPLETION_HINT = "request logprobs=2 (an integer, at most 5) with temperature=0"
_RESPONSES_HINT = (
    "request top_logprobs=2 (or more) and include=['message.output_text.logprobs'] "
    "with temperature=0"
)
_PARTS = ("content", "refusal")


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------


def _pick_choice(choices: Any, choice_index: int, path: str) -> tuple[Any, str]:
    """Return the choice whose ``index`` is ``choice_index`` (falling back to position)."""
    seq = as_sequence(choices, path)
    for pos, choice in enumerate(seq):
        if get_field(choice, "index", pos) == choice_index:
            return choice, f"{path}[{pos}]"
    raise PayloadError(f"{path} has no choice with index {choice_index} ({len(seq)} choices)")


def _looks_like_token_list(obj: Any) -> bool:
    return isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray, Mapping))


def _token_entry(entry: Any, where: str, hint: str) -> tuple[list[float], float | None]:
    """Candidate log-probs and the generated token's log-prob from one token entry."""
    top = get_field(entry, "top_logprobs")
    if top is None:
        raise PayloadError(f"{where}.top_logprobs is missing; {hint}")
    cands = [
        as_logprob(
            require_field(c, "logprob", f"{where}.top_logprobs[{j}]"),
            f"{where}.top_logprobs[{j}].logprob",
        )
        for j, c in enumerate(as_sequence(top, f"{where}.top_logprobs"))
    ]
    lp = get_field(entry, "logprob")
    return cands, (None if lp is None else as_logprob(lp, f"{where}.logprob"))


def _token_entries(
    entries: Sequence[Any], paths: Sequence[str], hint: str
) -> tuple[list[list[float]], list[float | None], list[str]]:
    """Candidates, generated-token log-probs and labels from ``{logprob, top_logprobs}`` entries."""
    positions: list[list[float]] = []
    sampled: list[float | None] = []
    for entry, where in zip(entries, paths):
        cands, lp = _token_entry(entry, where, hint)
        positions.append(cands)
        sampled.append(lp)
    return positions, sampled, list(paths)


def _drop_last(
    positions: list[list[float]],
    sampled: list[float | None],
    labels: list[str],
    drop: bool,
) -> None:
    if drop and positions:
        del positions[-1], sampled[-1], labels[-1]


def signals_from_chat_completion(
    response: Any,
    *,
    choice_index: int = 0,
    part: str = "content",
    drop_stop_token: bool = False,
    require_greedy: bool = False,
    renormalize_entropy: bool = False,
) -> TokenSignals:
    """Token signals of one Chat Completions choice (Section 4.1, Eq. 4).

    Parameters
    ----------
    response : ChatCompletion, choice, or list of token entries
        A ``ChatCompletion`` (SDK object or dict), one of its ``choices``, or
        the ``choice.logprobs.content`` list itself. Request it with
        ``temperature=0, logprobs=True, top_logprobs=2`` (or more).
    choice_index : int, default 0
        Which choice to read when the response has several (``n > 1``).
    part : {"content", "refusal"}, default "content"
        Which token list of ``choice.logprobs`` to read. A refusal carries
        its tokens under ``refusal`` and leaves ``content`` null.
    drop_stop_token : bool, default False
        Set True for servers that report the terminating EOS token (vLLM,
        llama.cpp): the last entry is excluded when ``finish_reason`` is
        ``"stop"`` or absent, and kept for ``"length"`` and other reasons.
        With a bare token list there is no ``finish_reason``, so the last
        entry is excluded unconditionally.
    require_greedy : bool, default False
        Raise :class:`~ucci.integrations.PayloadError` if a generated token is
        not the top-1 candidate.
    renormalize_entropy : bool, default False
        Entropy of the renormalized top-k instead of the truncated sum (see
        :attr:`~ucci.integrations.TokenSignals.entropy_support`).

    Returns
    -------
    TokenSignals
        ``.u`` is u(x); ``.mean_entropy`` (over the returned top-k) and
        ``.mean_max_prob`` feed the Section 6.1 and 6.3 baselines.

    Raises
    ------
    PayloadError
        If logprobs were not requested, fewer than two candidates were
        returned for a token, a value is not a log-probability, or there are
        no tokens.
    """
    if part not in _PARTS:
        raise ValueError(f"part must be one of {_PARTS}, got {part!r}")
    finish: Any = None
    if _looks_like_token_list(response):
        entries = as_sequence(response, "content")
        path = "content"
        finish = "stop"
    else:
        choices = get_field(response, "choices")
        if choices is not None:
            choice, cpath = _pick_choice(choices, choice_index, "choices")
        elif (
            get_field(response, "logprobs", None) is not None
            or get_field(response, "message", None) is not None
        ):
            choice, cpath = response, "choice"
        else:
            raise PayloadError(
                "expected a ChatCompletion, one of its choices, or "
                f"choice.logprobs.content; got {type(response).__name__}"
            )
        finish = get_field(choice, "finish_reason")
        logprobs = require_field(choice, "logprobs", cpath, _CHAT_HINT)
        raw = get_field(logprobs, part)
        if raw is None:
            other = _PARTS[1 - _PARTS.index(part)]
            alt = get_field(logprobs, other)
            extra = f" (logprobs.{other} has tokens: pass part={other!r})" if alt else ""
            raise PayloadError(f"{cpath}.logprobs.{part} is null{extra}; {_CHAT_HINT}")
        path = f"{cpath}.logprobs.{part}"
        entries = as_sequence(raw, path)
    paths = [f"{path}[{t}]" for t in range(len(entries))]
    positions, sampled, labels = _token_entries(entries, paths, _CHAT_HINT)
    _drop_last(positions, sampled, labels, drop_stop_token and finish in ("stop", None))
    return summarize_candidates(
        positions,
        sampled,
        source="openai.chat_completion",
        hint=_CHAT_HINT,
        require_greedy=require_greedy,
        renormalize_entropy=renormalize_entropy,
        labels=labels,
    )


def u_from_chat_completion(response: Any, **kwargs: Any) -> float:
    """u(x) (Eq. 4) of one Chat Completions choice.

    Shorthand for ``signals_from_chat_completion(response, **kwargs).u``; see
    :func:`signals_from_chat_completion` for the arguments.
    """
    return signals_from_chat_completion(response, **kwargs).u


def signals_from_chat_completion_chunks(
    chunks: Iterable[Any],
    *,
    choice_index: int = 0,
    part: str = "content",
    drop_stop_token: bool = False,
    require_greedy: bool = False,
    renormalize_entropy: bool = False,
) -> TokenSignals:
    """Token signals from a streamed Chat Completion (``stream=True``).

    Each ``chat.completion.chunk`` carries the log-probabilities of the
    tokens in its delta under ``choices[i].logprobs.content``; this function
    concatenates them in arrival order and reads ``finish_reason`` from the
    chunk that sets it. Arguments are as in
    :func:`signals_from_chat_completion`.

    Parameters
    ----------
    chunks : iterable of ChatCompletionChunk
        The chunks as received (SDK objects or dicts). Chunks without
        choices, such as the final usage chunk, are skipped.

    Returns
    -------
    TokenSignals
    """
    if part not in _PARTS:
        raise ValueError(f"part must be one of {_PARTS}, got {part!r}")
    entries: list[Any] = []
    labels: list[str] = []
    finish: Any = None
    for i, chunk in enumerate(chunks):
        choices = get_field(chunk, "choices")
        if choices is None:
            continue
        for pos, choice in enumerate(as_sequence(choices, f"chunks[{i}].choices")):
            if get_field(choice, "index", pos) != choice_index:
                continue
            fr = get_field(choice, "finish_reason")
            if fr is not None:
                finish = fr
            logprobs = get_field(choice, "logprobs")
            raw = None if logprobs is None else get_field(logprobs, part)
            if raw:
                path = f"chunks[{i}].choices[{pos}].logprobs.{part}"
                for t, entry in enumerate(as_sequence(raw, path)):
                    entries.append(entry)
                    labels.append(f"{path}[{t}]")
    positions, sampled, labels = _token_entries(entries, labels, _CHAT_HINT)
    _drop_last(positions, sampled, labels, drop_stop_token and finish in ("stop", None))
    return summarize_candidates(
        positions,
        sampled,
        source="openai.chat_completion_chunks",
        hint=_CHAT_HINT,
        require_greedy=require_greedy,
        renormalize_entropy=renormalize_entropy,
        labels=labels,
    )


def u_from_chat_completion_chunks(chunks: Iterable[Any], **kwargs: Any) -> float:
    """u(x) (Eq. 4) of a streamed Chat Completion.

    Shorthand for ``signals_from_chat_completion_chunks(chunks, **kwargs).u``.
    """
    return signals_from_chat_completion_chunks(chunks, **kwargs).u


# ---------------------------------------------------------------------------
# Completions (legacy)
# ---------------------------------------------------------------------------


def signals_from_completion(
    response: Any,
    *,
    choice_index: int = 0,
    drop_stop_token: bool = False,
    require_greedy: bool = False,
    renormalize_entropy: bool = False,
) -> TokenSignals:
    """Token signals of one legacy Completions choice (``/v1/completions``).

    Parameters
    ----------
    response : Completion, choice, or logprobs object
        A ``Completion`` (``object == "text_completion"``), one of its
        choices, or ``choice.logprobs``. Request it with ``temperature=0,
        logprobs=2`` (an integer, at most 5) and without ``echo``.
    choice_index, drop_stop_token, require_greedy, renormalize_entropy
        As in :func:`signals_from_chat_completion`.

    Returns
    -------
    TokenSignals

    Raises
    ------
    PayloadError
        If ``logprobs`` is missing, a ``top_logprobs`` entry is null (this is
        what the prompt positions look like with ``echo=True``), the parallel
        lists disagree in length, or a position has fewer than two
        candidates.

    Notes
    -----
    ``top_logprobs`` maps token strings to log-probabilities. Two distinct
    tokens that decode to the same string (for example two partial UTF-8
    byte tokens) collapse into one dict entry, which can leave a position
    with a single candidate at ``logprobs=2``; request ``logprobs=3`` or more
    if that happens.
    """
    finish: Any = None
    if get_field(response, "choices") is not None:
        choice, cpath = _pick_choice(get_field(response, "choices"), choice_index, "choices")
        finish = get_field(choice, "finish_reason")
        logprobs = require_field(choice, "logprobs", cpath, _COMPLETION_HINT)
        lpath = f"{cpath}.logprobs"
    elif get_field(response, "top_logprobs") is not None:
        logprobs, lpath, finish = response, "logprobs", "stop"
    elif get_field(response, "text") is not None or get_field(response, "logprobs") is not None:
        cpath = "choice"
        finish = get_field(response, "finish_reason")
        logprobs = require_field(response, "logprobs", cpath, _COMPLETION_HINT)
        lpath = f"{cpath}.logprobs"
    else:
        raise PayloadError(
            "expected a Completion, one of its choices, or choice.logprobs; got "
            f"{type(response).__name__}"
        )
    top = as_sequence(
        require_field(logprobs, "top_logprobs", lpath, _COMPLETION_HINT), f"{lpath}.top_logprobs"
    )
    token_lps = get_field(logprobs, "token_logprobs")
    if token_lps is not None and len(as_sequence(token_lps, f"{lpath}.token_logprobs")) != len(top):
        raise PayloadError(
            f"{lpath}: token_logprobs has {len(token_lps)} entries but top_logprobs has {len(top)}"
        )
    positions: list[list[float]] = []
    sampled: list[float | None] = []
    labels: list[str] = []
    for t, cand in enumerate(top):
        where = f"{lpath}.top_logprobs[{t}]"
        if cand is None:
            raise PayloadError(
                f"{where} is null; this happens for prompt tokens with echo=True. "
                "Request without echo so every position is a generated token"
            )
        if not isinstance(cand, Mapping):
            raise PayloadError(f"{where} must be a {{token: logprob}} mapping, got {cand!r}")
        positions.append([as_logprob(v, f"{where}[{k!r}]") for k, v in cand.items()])
        s = None if token_lps is None else token_lps[t]
        sampled.append(None if s is None else as_logprob(s, f"{lpath}.token_logprobs[{t}]"))
        labels.append(where)
    _drop_last(positions, sampled, labels, drop_stop_token and finish in ("stop", None))
    return summarize_candidates(
        positions,
        sampled,
        source="openai.completion",
        hint=_COMPLETION_HINT,
        require_greedy=require_greedy,
        renormalize_entropy=renormalize_entropy,
        labels=labels,
    )


def u_from_completion(response: Any, **kwargs: Any) -> float:
    """u(x) (Eq. 4) of a legacy Completions choice; see :func:`signals_from_completion`."""
    return signals_from_completion(response, **kwargs).u


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


def signals_from_responses(
    response: Any,
    *,
    drop_stop_token: bool = False,
    require_greedy: bool = False,
    renormalize_entropy: bool = False,
) -> TokenSignals:
    """Token signals of a Responses API result (``/v1/responses``).

    Parameters
    ----------
    response : Response
        A ``Response`` (``object == "response"``, SDK object or dict)
        requested with ``temperature=0, top_logprobs=2`` (or more) and
        ``include=["message.output_text.logprobs"]``. The ``logprobs`` of all
        ``output_text`` parts of all ``message`` output items are
        concatenated in order; reasoning, tool-call and refusal items carry
        no text log-probabilities and are skipped.
    drop_stop_token : bool, default False
        Exclude the last entry when ``status == "completed"``. Only for
        servers that report the stop token.
    require_greedy, renormalize_entropy
        As in :func:`signals_from_chat_completion`.

    Returns
    -------
    TokenSignals

    Raises
    ------
    PayloadError
        If there is no ``output_text`` part, its ``logprobs`` are missing
        (the ``include`` option was not set), or a token has fewer than two
        candidates.
    """
    output = as_sequence(require_field(response, "output", "response"), "response.output")
    positions: list[list[float]] = []
    sampled: list[float | None] = []
    labels: list[str] = []
    n_text = 0
    for i, item in enumerate(output):
        if get_field(item, "type") != "message":
            continue
        content = get_field(item, "content") or []
        for j, piece in enumerate(as_sequence(content, f"response.output[{i}].content")):
            if get_field(piece, "type") != "output_text":
                continue
            n_text += 1
            path = f"response.output[{i}].content[{j}].logprobs"
            raw = get_field(piece, "logprobs")
            if raw is None:
                raise PayloadError(f"{path} is missing; {_RESPONSES_HINT}")
            seq = as_sequence(raw, path)
            p, s, lab = _token_entries(
                seq, [f"{path}[{t}]" for t in range(len(seq))], _RESPONSES_HINT
            )
            positions.extend(p)
            sampled.extend(s)
            labels.extend(lab)
    if n_text == 0:
        raise PayloadError("response.output has no message with an output_text part")
    completed = get_field(response, "status") in ("completed", None)
    _drop_last(positions, sampled, labels, drop_stop_token and completed)
    return summarize_candidates(
        positions,
        sampled,
        source="openai.responses",
        hint=_RESPONSES_HINT,
        require_greedy=require_greedy,
        renormalize_entropy=renormalize_entropy,
        labels=labels,
    )


def u_from_responses(response: Any, **kwargs: Any) -> float:
    """u(x) (Eq. 4) of a Responses API result; see :func:`signals_from_responses`."""
    return signals_from_responses(response, **kwargs).u


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


def signals_from_openai(response: Any, **kwargs: Any) -> TokenSignals:
    """Token signals from any supported OpenAI-format payload.

    Dispatches on the payload's ``object`` field (``"chat.completion"``,
    ``"text_completion"``, ``"response"``) or, when that is absent, on its
    shape: a list of chunks, a list of chat token entries, a response with
    ``choices`` holding a ``message`` (chat) or ``text`` (completion), or a
    response with ``output`` (Responses API).

    Parameters
    ----------
    response : object
        Any payload accepted by :func:`signals_from_chat_completion`,
        :func:`signals_from_chat_completion_chunks`,
        :func:`signals_from_completion` or :func:`signals_from_responses`.
    **kwargs
        Passed to the selected function.

    Returns
    -------
    TokenSignals

    Raises
    ------
    PayloadError
        If the payload type cannot be recognised.
    """
    kind = get_field(response, "object") if not _looks_like_token_list(response) else None
    if kind == "chat.completion":
        return signals_from_chat_completion(response, **kwargs)
    if kind == "text_completion":
        return signals_from_completion(response, **kwargs)
    if kind == "response":
        return signals_from_responses(response, **kwargs)
    if kind == "chat.completion.chunk":
        raise PayloadError(
            "got a single chat.completion.chunk; pass the whole list of chunks "
            "(signals_from_chat_completion_chunks)"
        )
    if _looks_like_token_list(response):
        first = response[0] if len(response) else None
        if first is not None and get_field(first, "choices") is not None:
            return signals_from_chat_completion_chunks(response, **kwargs)
        return signals_from_chat_completion(response, **kwargs)
    choices = get_field(response, "choices")
    if choices is not None and _looks_like_token_list(choices) and len(choices):
        first = choices[0]
        if get_field(first, "message") is not None:
            return signals_from_chat_completion(response, **kwargs)
        if get_field(first, "text") is not None:
            return signals_from_completion(response, **kwargs)
        if get_field(first, "delta") is not None:
            raise PayloadError("got a chat.completion.chunk; pass the whole list of chunks")
    if get_field(response, "output") is not None:
        return signals_from_responses(response, **kwargs)
    if get_field(response, "message") is not None:
        return signals_from_chat_completion(response, **kwargs)
    raise PayloadError(
        f"cannot recognise an OpenAI-format payload of type {type(response).__name__}"
    )


def u_from_openai(response: Any, **kwargs: Any) -> float:
    """u(x) (Eq. 4) from any supported OpenAI-format payload; see :func:`signals_from_openai`."""
    return signals_from_openai(response, **kwargs).u


# ---------------------------------------------------------------------------
# Client helper for ucci.integrations.cascade.Cascade
# ---------------------------------------------------------------------------


def make_chat_fn(
    client: Any,
    model: str,
    *,
    top_logprobs: int = 2,
    temperature: float = 0.0,
    system_prompt: str | None = None,
    return_response: bool = True,
    **create_kwargs: Any,
) -> Callable[[Any], Any]:
    """Wrap ``client.chat.completions.create`` as a cascade model function.

    Parameters
    ----------
    client : openai.OpenAI or openai.AsyncOpenAI
        Any client exposing ``chat.completions.create``, including one
        pointed at an OpenAI-compatible server through ``base_url``.
    model : str
        Model name sent with every request.
    top_logprobs : int, default 2
        Candidates per token (0 to 20). Two is all u(x) needs; ask for more
        (up to 20) if you also want a tighter top-k entropy baseline. Set 0
        to request no log-probabilities (for the large model).
    temperature : float, default 0.0
        Greedy decoding, as in the paper (Appendix B.2).
    system_prompt : str, optional
        Prepended as a system message.
    return_response : bool, default True
        Return ``(text, response)``, the ``small_fn`` contract of
        :class:`~ucci.integrations.cascade.Cascade`; False returns only the
        text, the ``large_fn`` contract.
    **create_kwargs
        Extra arguments for ``create`` (``max_tokens``, ``stop``, ...).

    Returns
    -------
    callable
        ``fn(query)`` where ``query`` is a user message string or a list of
        chat messages. With an ``AsyncOpenAI`` client ``fn`` returns an
        awaitable, which :meth:`Cascade.acall` awaits.

    Raises
    ------
    ValueError
        If ``top_logprobs`` is outside ``[0, 20]``.
    """
    if isinstance(top_logprobs, bool) or not isinstance(top_logprobs, int):
        raise TypeError(f"top_logprobs must be an int, got {top_logprobs!r}")
    if not 0 <= top_logprobs <= 20:
        raise ValueError(f"top_logprobs must lie in [0, 20], got {top_logprobs}")
    create = client.chat.completions.create

    def build(query: Any) -> dict[str, Any]:
        if isinstance(query, str):
            messages: list[Any] = [{"role": "user", "content": query}]
        else:
            messages = list(query)
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}, *messages]
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if top_logprobs > 0:
            kwargs.update(logprobs=True, top_logprobs=top_logprobs)
        kwargs.update(create_kwargs)
        return kwargs

    def finish(resp: Any) -> Any:
        choices = get_field(resp, "choices") or []
        message = get_field(choices[0], "message") if choices else None
        text = get_field(message, "content") if message is not None else None
        return (text, resp) if return_response else text

    async def finish_async(pending: Any) -> Any:
        return finish(await pending)

    def fn(query: Any) -> Any:
        resp = create(**build(query))
        if inspect.isawaitable(resp):
            return finish_async(resp)
        return finish(resp)

    return fn
