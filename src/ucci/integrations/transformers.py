"""u(x) from Hugging Face ``transformers`` generation.

UCCI's routing signal is the token-margin uncertainty of the small model's
greedy generation (Section 4.1, Eq. 4):

    m_t = p_{t,1} - p_{t,2},    u(x) = 1 - (1/T) * sum_t m_t,

where p_{t,1} and p_{t,2} are the two largest next-token probabilities at
step t. With the full per-step logits available this adapter computes them
exactly, with a softmax over the whole vocabulary, and returns one u(x) per
batch item. It also returns the mean full-vocabulary entropy and mean max
probability used by the Section 6.1 entropy baseline and the Section 6.3
signal ablation.

Generate with greedy decoding and per-step logits:

    out = model.generate(**inputs, do_sample=False, num_beams=1,
                         return_dict_in_generate=True, output_logits=True,
                         max_new_tokens=256)
    us = u_from_generate(out, eos_token_id=model.generation_config.eos_token_id)

or let :func:`generate_with_signals` do it. ``output_logits`` (transformers
4.38 and later) returns the raw logits; ``output_scores`` returns the scores
after the logits processors (repetition penalty, ``min_new_tokens``, bad
words, forced tokens), which are not the model's next-token distribution.
The adapter therefore prefers ``logits`` and falls back to ``scores`` only
when logits were not requested. Checked against the ``GenerateDecoderOnlyOutput``
and ``GenerateEncoderDecoderOutput`` definitions (September 2026):

* https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
* https://huggingface.co/docs/transformers/internal/generation_utils

Token convention
----------------
Step t of ``output.logits`` produced generated token ``sequences[:, -S + t]``
(S steps). UCCI counts every generated content token and excludes padding and
the terminating EOS token (``docs/paper_mapping.md``). For each batch item the
adapter keeps the steps before the first EOS token (any id in
``eos_token_id``); ``generate`` pads a row that finished on EOS with
``pad_token_id``, and those steps come after the EOS, so they are excluded
too. A row can also finish without an EOS token, when a stopping criterion
such as ``stop_strings`` ends it; its padding then follows the last content
token directly. Pass ``pad_token_id`` in that case and a trailing run of the
pad id is excluded as well. Leave it out otherwise: a model may generate the
pad id as content (small or untrained models do), and those tokens would be
dropped. Left padding of the prompt does not matter, because only the
generated part of ``sequences`` is read, and rows that stop at different
steps are handled per row. Beam search is rejected: UCCI uses greedy
decoding.

Arrays
------
Nothing here imports ``torch``. Each step tensor is copied to host memory and
promoted to float64 (``bfloat16`` and ``float16`` through float32, which is
exact), and the softmax, top-2, entropy and argmax are computed with NumPy one
step at a time, so peak extra memory is a few (batch, vocab) float64 arrays.
NumPy arrays and other array-likes work as step inputs, which also makes the
adapter usable with JAX or TensorFlow outputs of the same layout.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..signal import uncertainty_from_margins
from ._common import PayloadError, TokenSignals, empty_signals, get_field

__all__ = [
    "GenerationWithSignals",
    "content_lengths",
    "generate_with_signals",
    "signals_from_generate",
    "u_from_generate",
]

_SOURCES = ("auto", "logits", "scores")
_GREEDY_RTOL = 1e-9


def _to_host_float64(x: Any, where: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Copy one (batch, vocab) step tensor to a float64 NumPy array."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    dtype = str(getattr(x, "dtype", ""))
    if dtype in ("torch.bfloat16", "torch.float16") and hasattr(x, "float"):
        x = x.float()
    if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        x = x.numpy()
    try:
        arr = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PayloadError(f"{where} is not a numeric array: {exc}") from exc
    if arr.ndim != 2:
        raise PayloadError(f"{where} must have shape (batch, vocab), got {arr.shape}")
    return arr


def _to_host_int(x: Any, where: str) -> np.ndarray[Any, np.dtype[np.int64]]:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        x = x.numpy()
    arr = np.asarray(x)
    if arr.dtype.kind not in "iu":
        raise PayloadError(f"{where} must hold integer token ids, got dtype {arr.dtype}")
    if arr.ndim != 2:
        raise PayloadError(f"{where} must have shape (batch, length), got {arr.shape}")
    return arr.astype(np.int64, copy=False)


def _id_set(ids: Any, name: str) -> list[int]:
    """Normalise an int, an iterable of ints or None to a list of ints."""
    if ids is None:
        return []
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, bool):
        raise TypeError(f"{name} must be an int or a list of ints, got {ids!r}")
    if isinstance(ids, int):
        return [ids]
    try:
        out = list(ids)
    except TypeError as exc:
        raise TypeError(f"{name} must be an int or a list of ints, got {ids!r}") from exc
    for v in out:
        if isinstance(v, bool) or not isinstance(v, int):
            raise TypeError(f"{name} must contain ints, got {v!r}")
    return out


def _select_steps(output: Any, source: str) -> tuple[Sequence[Any], str]:
    if source not in _SOURCES:
        raise ValueError(f"source must be one of {_SOURCES}, got {source!r}")
    logits = get_field(output, "logits")
    scores = get_field(output, "scores")
    if source in ("auto", "logits") and logits is not None and len(logits):
        return logits, "logits"
    if source == "logits":
        raise PayloadError(
            "output.logits is missing; call generate(..., return_dict_in_generate=True, "
            "output_logits=True) (transformers >= 4.38)"
        )
    if scores is not None and len(scores):
        return scores, "scores"
    raise PayloadError(
        "output has neither logits nor scores; call generate(..., "
        "return_dict_in_generate=True, output_logits=True) (or output_scores=True "
        "before transformers 4.38)"
    )


def content_lengths(
    generated: Any, eos_token_id: Any, pad_token_id: int | None = None
) -> np.ndarray[Any, np.dtype[np.int64]]:
    """Number of content tokens per row of the generated block.

    Parameters
    ----------
    generated : array_like of int, shape (batch, steps)
        The generated token ids, ``sequences[:, -steps:]``.
    eos_token_id : int, list of int, or None
        Every id that ends generation. The first occurrence and everything
        after it are excluded.
    pad_token_id : int, optional
        When given and not an EOS id, a trailing run of this id is excluded
        as well (rows ended by a stopping criterion other than EOS).

    Returns
    -------
    numpy.ndarray of int64, shape (batch,)
    """
    gen = _to_host_int(generated, "generated")
    n_rows, n_steps = gen.shape
    lengths = np.full(n_rows, n_steps, dtype=np.int64)
    eos = _id_set(eos_token_id, "eos_token_id")
    if eos:
        is_eos = np.isin(gen, np.asarray(eos, dtype=np.int64))
        lengths = np.where(is_eos.any(axis=1), is_eos.argmax(axis=1), lengths)
    if pad_token_id is not None and pad_token_id not in eos:
        not_pad = gen != pad_token_id
        last_real = n_steps - 1 - not_pad[:, ::-1].argmax(axis=1)
        trail_start = np.where(not_pad.any(axis=1), last_real + 1, 0)
        lengths = np.minimum(lengths, trail_start)
    return lengths


def signals_from_generate(
    output: Any,
    *,
    eos_token_id: Any,
    pad_token_id: int | None = None,
    source: str = "auto",
    require_greedy: bool = False,
    on_empty: str = "raise",
) -> list[TokenSignals]:
    """Token signals for every item of a ``model.generate`` batch (Eq. 4).

    Parameters
    ----------
    output : GenerateDecoderOnlyOutput or GenerateEncoderDecoderOutput
        The result of ``generate(..., do_sample=False, num_beams=1,
        return_dict_in_generate=True, output_logits=True)``. A mapping with
        ``sequences`` and ``logits`` or ``scores`` also works.
    eos_token_id : int, list of int, or None
        The ids that end generation, normally
        ``model.generation_config.eos_token_id``. Required so that the EOS
        token and the padding after it are never averaged in by accident;
        pass None only if generation cannot stop on a token.
    pad_token_id : int, optional
        Pass the id ``generate`` wrote after a row finished (normally
        ``model.generation_config.pad_token_id``) only when rows can finish
        without an EOS token, for example with ``stop_strings`` or a custom
        per-row stopping criterion. A trailing run of it is then excluded.
        See the module docstring for why this is not the default.
    source : {"auto", "logits", "scores"}, default "auto"
        Which per-step tensors to read. "auto" uses raw ``logits`` when
        present, else ``scores``.
    require_greedy : bool, default False
        Raise if a generated token is not the argmax of the chosen tensors.
        With raw logits a mismatch means a logits processor changed the
        greedy choice (for example ``repetition_penalty``); with scores it
        means sampling was on.
    on_empty : {"raise", "nan"}, default "raise"
        A row whose first generated token is EOS has no content tokens and
        u(x) is undefined (Eq. 4 needs T >= 1). "raise" raises; "nan" returns
        NaN signals for that row (extension, not in the paper).

    Returns
    -------
    list of TokenSignals
        One per batch row, in order. ``entropy_support`` is
        ``"full_vocabulary"``.

    Raises
    ------
    PayloadError
        On beam-search outputs, missing logits and scores, inconsistent
        shapes, NaN or ``+inf`` logits at a content step, a token id outside
        the vocabulary, an empty row with ``on_empty="raise"``, or (with
        ``require_greedy``) a non-greedy step.
    ValueError
        On an invalid ``source`` or ``on_empty``.
    """
    if on_empty not in ("raise", "nan"):
        raise ValueError(f"on_empty must be 'raise' or 'nan', got {on_empty!r}")
    if get_field(output, "beam_indices") is not None:
        raise PayloadError(
            "beam-search output: UCCI's signal is defined under greedy decoding "
            "(Section 4.1); generate with num_beams=1, do_sample=False"
        )
    steps, kind = _select_steps(output, source)
    sequences = get_field(output, "sequences")
    if sequences is None:
        raise PayloadError("output.sequences is missing")
    seqs = _to_host_int(sequences, "output.sequences")
    n_rows, total = seqs.shape
    n_steps = len(steps)
    if n_steps > total:
        raise PayloadError(
            f"output has {n_steps} {kind} steps but sequences has only {total} columns"
        )
    gen = seqs[:, total - n_steps :]
    lengths = content_lengths(gen, eos_token_id, pad_token_id)
    label = f"transformers.{kind}"

    p1 = np.zeros((n_rows, n_steps))
    p2 = np.zeros((n_rows, n_steps))
    ent = np.zeros((n_rows, n_steps))
    non_greedy = np.zeros(n_rows, dtype=np.int64)
    vocab: int | None = None
    for t in range(n_steps):
        active = t < lengths
        if not active.any():
            continue
        x = _to_host_float64(steps[t], f"output.{kind}[{t}]")
        if x.shape[0] != n_rows:
            raise PayloadError(
                f"output.{kind}[{t}] has {x.shape[0]} rows but sequences has {n_rows}; "
                "UCCI needs num_return_sequences=1 and num_beams=1"
            )
        if vocab is None:
            vocab = x.shape[1]
            if vocab < 2:
                raise PayloadError(f"vocabulary of size {vocab}: no top-2 margin exists")
        elif x.shape[1] != vocab:
            raise PayloadError(f"output.{kind}[{t}] has vocabulary {x.shape[1]}, expected {vocab}")
        rows = np.flatnonzero(active)
        xa = x[rows]
        bad = np.isnan(xa).any(axis=1) | np.isposinf(xa).any(axis=1)
        if bad.any():
            r = int(rows[int(np.argmax(bad))])
            raise PayloadError(f"output.{kind}[{t}] row {r} contains NaN or +inf")
        m = xa.max(axis=1)
        if np.isneginf(m).any():
            r = int(rows[int(np.argmax(np.isneginf(m)))])
            raise PayloadError(f"output.{kind}[{t}] row {r} is -inf everywhere")
        z = xa - m[:, None]
        e = np.exp(z)
        s = e.sum(axis=1)
        logp = z - np.log(s)[:, None]
        prob = e / s[:, None]
        top2 = np.partition(prob, vocab - 2, axis=1)[:, vocab - 2 :]
        p1[rows, t] = top2.max(axis=1)
        p2[rows, t] = top2.min(axis=1)
        plogp = np.multiply(prob, logp, out=np.zeros_like(prob), where=prob > 0.0)
        ent[rows, t] = -plogp.sum(axis=1)
        tok = gen[rows, t]
        if (tok < 0).any() or (tok >= vocab).any():
            r = int(rows[int(np.argmax((tok < 0) | (tok >= vocab)))])
            raise PayloadError(
                f"generated token id {int(gen[r, t])} at step {t}, row {r} is outside "
                f"the {kind} vocabulary of size {vocab}"
            )
        chosen = xa[np.arange(rows.size), tok]
        below = chosen < m - _GREEDY_RTOL * np.maximum(1.0, np.abs(m))
        non_greedy[rows] += below.astype(np.int64)

    results: list[TokenSignals] = []
    for r in range(n_rows):
        n = int(lengths[r])
        if n == 0:
            if on_empty == "raise":
                raise PayloadError(
                    f"row {r}: the first generated token is EOS, so there are no content "
                    "tokens and u(x) is undefined (Eq. 4 needs T >= 1); pass "
                    "on_empty='nan' to get NaN for such rows"
                )
            results.append(empty_signals(label, "full_vocabulary", None))
            continue
        a, b = p1[r, :n], p2[r, :n]
        margins = a - b
        results.append(
            TokenSignals(
                u=uncertainty_from_margins(margins),
                n_tokens=n,
                mean_max_prob=float(np.mean(a)),
                mean_entropy=float(np.mean(ent[r, :n])),
                entropy_support="full_vocabulary",
                top_k=None,
                n_non_greedy=int(non_greedy[r]),
                margins=tuple(float(v) for v in margins),
                source=label,
            )
        )
    if require_greedy:
        bad_rows = [r for r, sig in enumerate(results) if sig.n_non_greedy]
        if bad_rows:
            r = bad_rows[0]
            raise PayloadError(
                f"row {r}: {results[r].n_non_greedy} generated tokens are not the argmax "
                f"of output.{kind}; UCCI's signal is defined under greedy decoding "
                "(Section 4.1)"
            )
    return results


def u_from_generate(output: Any, **kwargs: Any) -> list[float]:
    """u(x) (Eq. 4) for every batch row; see :func:`signals_from_generate`."""
    return [s.u for s in signals_from_generate(output, **kwargs)]


@dataclass(frozen=True)
class GenerationWithSignals:
    """Result of :func:`generate_with_signals`.

    Attributes
    ----------
    texts : list of str
        Decoded content tokens per prompt (EOS and padding removed).
    signals : list of TokenSignals
        Per-prompt signals; ``signals[i].u`` is u(x) (Eq. 4).
    output : object
        The raw ``generate`` output, for anything else you need.
    """

    texts: list[str]
    signals: list[TokenSignals]
    output: Any

    @property
    def u(self) -> list[float]:
        """u(x) per prompt."""
        return [s.u for s in self.signals]


def _transformers_has_output_logits() -> bool:
    try:
        import transformers
    except ImportError as exc:
        raise ImportError(
            "generate_with_signals needs transformers; install it with "
            "`pip install transformers torch`"
        ) from exc
    parts = str(transformers.__version__).split(".")
    try:
        major, minor = int(parts[0]), int("".join(ch for ch in parts[1] if ch.isdigit()) or 0)
    except (IndexError, ValueError):
        return True
    return (major, minor) >= (4, 38)


_FIXED = {"do_sample": False, "num_beams": 1, "num_return_sequences": 1}


def generate_with_signals(
    model: Any,
    tokenizer: Any,
    prompts: str | Iterable[str],
    *,
    max_new_tokens: int = 256,
    source: str = "auto",
    on_empty: str = "raise",
    **generate_kwargs: Any,
) -> GenerationWithSignals:
    """Greedy generation with u(x) for every prompt, in one call.

    Tokenizes ``prompts`` with left padding (decoder-only models), runs
    ``model.generate`` with greedy decoding and per-step logits, and returns
    the decoded texts together with :class:`TokenSignals` per prompt.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        A causal LM or an encoder-decoder model.
    tokenizer : transformers.PreTrainedTokenizerBase
        Its tokenizer. Its ``padding_side`` and ``pad_token`` are restored
        after the call.
    prompts : str or iterable of str
        Already formatted prompts (apply a chat template first if needed).
    max_new_tokens : int, default 256
        The paper's generation limit (Appendix B.2).
    source, on_empty
        As in :func:`signals_from_generate`.
    **generate_kwargs
        Extra ``generate`` arguments. ``do_sample``, ``num_beams`` and
        ``num_return_sequences`` are fixed for greedy decoding and raise if
        given a different value.

    Returns
    -------
    GenerationWithSignals

    Raises
    ------
    ImportError
        If torch or transformers is missing.
    ValueError
        If a greedy-decoding argument is overridden.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "generate_with_signals needs PyTorch; install it with `pip install torch`"
        ) from exc
    has_logits = _transformers_has_output_logits()
    for key, fixed in _FIXED.items():
        if key in generate_kwargs and generate_kwargs[key] != fixed:
            raise ValueError(
                f"{key}={generate_kwargs[key]!r} is not allowed: UCCI's signal uses greedy "
                f"decoding (Section 4.1), which needs {key}={fixed!r}"
            )
    batch = [prompts] if isinstance(prompts, str) else list(prompts)
    if not batch:
        raise ValueError("prompts is empty")

    gen_cfg = getattr(model, "generation_config", None)
    eos = generate_kwargs.get("eos_token_id", getattr(gen_cfg, "eos_token_id", None))
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = _id_set(eos, "eos_token_id")
    pad = generate_kwargs.get("pad_token_id", getattr(gen_cfg, "pad_token_id", None))
    if pad is None:
        pad = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else (eos_ids[0] if eos_ids else None)
        )
    # Only rows ended by a stopping criterion other than EOS end in bare padding.
    stops_without_eos = bool(
        generate_kwargs.get("stop_strings") or generate_kwargs.get("stopping_criteria")
    )

    is_enc_dec = bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False))
    old_side, old_pad = tokenizer.padding_side, tokenizer.pad_token
    try:
        if not is_enc_dec:
            tokenizer.padding_side = "left"
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        enc = tokenizer(batch, return_tensors="pt", padding=True)
    finally:
        tokenizer.padding_side = old_side
        if old_pad is None:
            tokenizer.pad_token = old_pad
    device = getattr(model, "device", None)
    if device is not None:
        enc = enc.to(device)

    kwargs: dict[str, Any] = dict(generate_kwargs)
    kwargs.update(_FIXED)
    kwargs.update(return_dict_in_generate=True, max_new_tokens=max_new_tokens)
    if has_logits:
        kwargs["output_logits"] = True
    else:
        kwargs["output_scores"] = True
    if pad is not None:
        kwargs.setdefault("pad_token_id", pad)
    with torch.inference_mode():
        out = model.generate(**enc, **kwargs)

    signals = signals_from_generate(
        out,
        eos_token_id=eos_ids or None,
        pad_token_id=pad if stops_without_eos else None,
        source=source,
        on_empty=on_empty,
    )
    seqs = _to_host_int(get_field(out, "sequences"), "output.sequences")
    n_steps = len(_select_steps(out, source)[0])
    gen = seqs[:, seqs.shape[1] - n_steps :]
    texts = [
        str(tokenizer.decode(gen[i, : s.n_tokens].tolist(), skip_special_tokens=True))
        for i, s in enumerate(signals)
    ]
    return GenerationWithSignals(texts=texts, signals=signals, output=out)
