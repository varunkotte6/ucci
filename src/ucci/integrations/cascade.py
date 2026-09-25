"""Run a UCCI small-to-large cascade on live queries, and log traffic for calibration.

:class:`Cascade` wires a fitted router to two model functions. For each query
it calls the small model, computes u(x) from the raw response (Section 4.1,
Eq. 4), maps it to the calibrated error probability p_hat(x) = g(u(x))
(Section 4.2, Eq. 5) and applies the threshold policy of Section 4.3, Eq. 6:
keep the small model's answer if p_hat(x) <= theta, otherwise call the large
model and return its answer. This is step 3 of the evaluation protocol of
Section 6.1, run online.

:class:`JsonlLogger` writes the JSONL record format that the ``ucci`` tools
read, so a deployment can collect the calibration and validation data UCCI is
fitted on. Section 4.3 selects theta on a validation set where both models
have been run; ``Cascade(..., shadow_large=True)`` runs the large model on
every query for exactly that purpose, while still returning the routed
answer.

Example
-------
>>> from ucci.integrations.cascade import Cascade, JsonlLogger
>>> from ucci.integrations.openai import make_chat_fn, signals_from_chat_completion
>>> cascade = Cascade(router,                                    # doctest: +SKIP
...                   small_fn=make_chat_fn(client, "small-model"),
...                   large_fn=make_chat_fn(client, "large-model", top_logprobs=0,
...                                         return_response=False),
...                   signal_fn=signals_from_chat_completion)
>>> result = cascade("Canon 5D with a 50mm f/1.8 lens")          # doctest: +SKIP
>>> result.answer, result.escalated, result.p_hat                # doctest: +SKIP
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import math
import numbers
import os
import threading
import time
from collections.abc import Awaitable, Iterable, Mapping
from dataclasses import dataclass
from typing import IO, Any, Callable, Generic, Protocol, TypeVar

import numpy as np

from ._common import TokenSignals

__all__ = [
    "RECORD_FIELDS",
    "SPLITS",
    "Cascade",
    "CascadeResult",
    "JsonlLogger",
    "RouterLike",
    "attach_labels",
    "make_record",
]

Q = TypeVar("Q")
A = TypeVar("A")

#: Allowed values of the optional ``split`` field (calibration, validation,
#: test: the three disjoint splits of Section 6.1).
SPLITS = ("cal", "val", "test")

#: Standard fields of a JSONL traffic record, in the order they are written.
RECORD_FIELDS = (
    "id",
    "split",
    "u",
    "small_correct",
    "large_correct",
    "small_score",
    "large_score",
    "latency_small_ms",
    "latency_large_ms",
    "entropy",
    "max_prob",
)


class RouterLike(Protocol):
    """What :class:`Cascade` needs from a router.

    :class:`ucci.UCCIRouter` satisfies it once ``calibrate`` and
    ``choose_threshold`` have run (or after loading a saved router).
    """

    def error_probability(self, u: Any) -> Any:
        """Calibrated p_hat = g(u) for an array of u values (Section 4.2)."""

    def escalate(self, u: Any) -> Any:
        """Boolean array, True where p_hat > theta (Section 4.3, Eq. 6)."""


@dataclass(frozen=True)
class CascadeResult(Generic[A]):
    """Outcome of routing one query.

    Attributes
    ----------
    answer : A
        The answer to serve: the small model's if kept, the large model's if
        escalated.
    escalated : bool
        True if p_hat > theta, so the large model answered (Eq. 6).
    p_hat : float
        Calibrated probability that the small model's answer is wrong,
        g(u(x)) (Eq. 5).
    u : float
        The small model's token-margin uncertainty u(x) (Eq. 4).
    small_answer : A
        The small model's answer, kept even when escalated.
    large_answer : A or None
        The large model's answer when it ran (escalated, or shadow mode).
    signals : TokenSignals or None
        Full signal set when ``signal_fn`` returned :class:`TokenSignals`.
    latency_small_ms, latency_large_ms : float or None
        Wall-clock time of each model call in milliseconds, useful for
        measuring the cost ratio c_l / c_s the way Section 6.1 does.
    """

    answer: A
    escalated: bool
    p_hat: float
    u: float
    small_answer: A
    large_answer: A | None = None
    signals: TokenSignals | None = None
    latency_small_ms: float | None = None
    latency_large_ms: float | None = None

    def to_record(
        self,
        id: str,
        *,
        small_correct: Any = None,
        large_correct: Any = None,
        split: str | None = None,
        include_answers: bool = False,
    ) -> dict[str, Any]:
        """JSONL record for this result (see :func:`make_record`).

        Parameters
        ----------
        id : str
            Query identifier.
        small_correct, large_correct : 0, 1 or float in [0, 1], optional
            Correctness of each model's answer, when already known.
        split : {"cal", "val", "test"}, optional
            Which split the query belongs to.
        include_answers : bool, default False
            Also store ``small_answer`` and ``large_answer`` (non-JSON values
            are stored as ``str``), so correctness can be labelled later.

        Returns
        -------
        dict
            Standard fields plus ``escalated`` and ``p_hat``.
        """
        extra: dict[str, Any] = {"escalated": self.escalated, "p_hat": self.p_hat}
        if include_answers:
            extra["small_answer"] = _jsonable(self.small_answer)
            if self.large_answer is not None:
                extra["large_answer"] = _jsonable(self.large_answer)
        sig = self.signals
        return make_record(
            id,
            self.u,
            small_correct=small_correct,
            large_correct=large_correct,
            split=split,
            latency_small_ms=self.latency_small_ms,
            latency_large_ms=self.latency_large_ms,
            entropy=None if sig is None or math.isnan(sig.mean_entropy) else sig.mean_entropy,
            max_prob=None if sig is None or math.isnan(sig.mean_max_prob) else sig.mean_max_prob,
            extra=extra,
        )


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return str(value)
    return value


def _coerce_signal(value: Any) -> tuple[float, TokenSignals | None]:
    """Turn what ``signal_fn`` returned into (u, signals)."""
    if isinstance(value, TokenSignals):
        u, sig = value.u, value
    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
        u, sig = float(value), None
    elif hasattr(value, "u"):
        u, sig = float(value.u), None
    else:
        raise TypeError(
            f"signal_fn must return u(x) as a float or a TokenSignals, got {type(value).__name__}"
        )
    if not math.isfinite(u):
        raise ValueError(f"signal_fn returned a non-finite u(x): {u!r}")
    return u, sig


def _unpack_small(value: Any) -> tuple[Any, Any]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(
            f"small_fn must return an (answer, raw_response) pair, got {type(value).__name__}"
        )
    return value[0], value[1]


def _reject_awaitable(value: Any, name: str) -> None:
    if inspect.isawaitable(value):
        if inspect.iscoroutine(value):
            value.close()
        raise TypeError(f"{name} returned an awaitable; use `await cascade.acall(query)`")


async def _resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class Cascade(Generic[Q, A]):
    """Small-to-large LLM cascade routed by a calibrated UCCI router.

    Parameters
    ----------
    router : RouterLike
        A fitted :class:`ucci.UCCIRouter` (calibrated, threshold chosen), or
        any object with the same ``error_probability`` and ``escalate``
        methods.
    small_fn : callable
        ``small_fn(query) -> (answer, raw_response)``. ``raw_response`` must
        carry the small model's top-2 log-probabilities; it is passed to
        ``signal_fn``. May be ``async`` (use :meth:`acall`).
    large_fn : callable
        ``large_fn(query) -> answer``. May be ``async``.
    signal_fn : callable
        ``signal_fn(raw_response) -> u``: a float, or :class:`TokenSignals`
        to also keep the entropy and max-prob baselines. Use one of the
        adapters, for example
        :func:`ucci.integrations.openai.signals_from_chat_completion` or
        :func:`ucci.integrations.vllm.signals_from_vllm`. May be ``async``.
    shadow_large : bool, default False
        Also run the large model on queries the router keeps, and store its
        answer in :attr:`CascadeResult.large_answer`. The served answer is
        unchanged. Use it to collect data where both models ran, which
        threshold selection needs (Section 4.3); it costs a large-model call
        per query.

    Notes
    -----
    The decision is the router's: ``escalated = router.escalate([u])[0]``,
    which for :class:`ucci.UCCIRouter` is p_hat > theta (Eq. 6). The model
    calls run one after another, so the recorded latencies are those of
    each model alone.
    """

    def __init__(
        self,
        router: RouterLike,
        small_fn: Callable[[Q], Any],
        large_fn: Callable[[Q], Any],
        signal_fn: Callable[[Any], Any],
        *,
        shadow_large: bool = False,
    ) -> None:
        for name, fn in (("small_fn", small_fn), ("large_fn", large_fn), ("signal_fn", signal_fn)):
            if not callable(fn):
                raise TypeError(f"{name} must be callable, got {type(fn).__name__}")
        for method in ("error_probability", "escalate"):
            if not callable(getattr(router, method, None)):
                raise TypeError(f"router must have an {method}() method")
        self.router = router
        self.small_fn = small_fn
        self.large_fn = large_fn
        self.signal_fn = signal_fn
        self.shadow_large = bool(shadow_large)

    def decide(self, u: float) -> tuple[float, bool]:
        """p_hat and the escalation decision for one u(x) (Eq. 5 and Eq. 6).

        Parameters
        ----------
        u : float
            Token-margin uncertainty of the small model's output.

        Returns
        -------
        (float, bool)
            ``(p_hat, escalated)``.
        """
        p_hat = float(np.asarray(self.router.error_probability([u]), dtype=np.float64).ravel()[0])
        escalated = bool(np.asarray(self.router.escalate([u])).ravel()[0])
        return p_hat, escalated

    def _result(
        self,
        small_answer: Any,
        u: float,
        sig: TokenSignals | None,
        large_answer: Any,
        ran_large: bool,
        escalated: bool,
        p_hat: float,
        t_small: float,
        t_large: float | None,
    ) -> CascadeResult[A]:
        return CascadeResult(
            answer=large_answer if escalated else small_answer,
            escalated=escalated,
            p_hat=p_hat,
            u=u,
            small_answer=small_answer,
            large_answer=large_answer if ran_large else None,
            signals=sig,
            latency_small_ms=t_small,
            latency_large_ms=t_large,
        )

    def __call__(self, query: Q) -> CascadeResult[A]:
        """Route one query synchronously (all three functions must be sync)."""
        t0 = time.perf_counter()
        small = self.small_fn(query)
        _reject_awaitable(small, "small_fn")
        t_small = (time.perf_counter() - t0) * 1e3
        small_answer, raw = _unpack_small(small)
        signal = self.signal_fn(raw)
        _reject_awaitable(signal, "signal_fn")
        u, sig = _coerce_signal(signal)
        p_hat, escalated = self.decide(u)
        large_answer: Any = None
        t_large: float | None = None
        ran_large = escalated or self.shadow_large
        if ran_large:
            t1 = time.perf_counter()
            large_answer = self.large_fn(query)
            _reject_awaitable(large_answer, "large_fn")
            t_large = (time.perf_counter() - t1) * 1e3
        return self._result(
            small_answer, u, sig, large_answer, ran_large, escalated, p_hat, t_small, t_large
        )

    async def acall(self, query: Q) -> CascadeResult[A]:
        """Route one query; sync or async model and signal functions both work."""
        t0 = time.perf_counter()
        small = await _resolve(self.small_fn(query))
        t_small = (time.perf_counter() - t0) * 1e3
        small_answer, raw = _unpack_small(small)
        u, sig = _coerce_signal(await _resolve(self.signal_fn(raw)))
        p_hat, escalated = self.decide(u)
        large_answer: Any = None
        t_large: float | None = None
        ran_large = escalated or self.shadow_large
        if ran_large:
            t1 = time.perf_counter()
            large_answer = await _resolve(self.large_fn(query))
            t_large = (time.perf_counter() - t1) * 1e3
        return self._result(
            small_answer, u, sig, large_answer, ran_large, escalated, p_hat, t_small, t_large
        )

    def map(self, queries: Iterable[Q]) -> list[CascadeResult[A]]:
        """Route queries one after another; results are in input order."""
        return [self(q) for q in queries]

    async def amap(
        self, queries: Iterable[Q], *, max_concurrency: int | None = None
    ) -> list[CascadeResult[A]]:
        """Route queries concurrently with :meth:`acall`; results keep input order.

        Parameters
        ----------
        queries : iterable
            The queries.
        max_concurrency : int, optional
            At most this many queries in flight at once. Unlimited if None.

        Returns
        -------
        list of CascadeResult
        """
        if max_concurrency is not None and (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
            or max_concurrency < 1
        ):
            raise ValueError(f"max_concurrency must be a positive int, got {max_concurrency!r}")
        sem = asyncio.Semaphore(max_concurrency) if max_concurrency else None

        async def one(q: Q) -> CascadeResult[A]:
            if sem is None:
                return await self.acall(q)
            async with sem:
                return await self.acall(q)

        tasks: list[Awaitable[CascadeResult[A]]] = [one(q) for q in queries]
        return list(await asyncio.gather(*tasks))


# ---------------------------------------------------------------------------
# JSONL traffic records
# ---------------------------------------------------------------------------


def _check_number(
    value: Any, name: str, *, low: float | None = None, high: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number, got {value!r}")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"{name} must be finite, got {x!r}")
    if (low is not None and x < low) or (high is not None and x > high):
        raise ValueError(f"{name} must lie in [{low}, {high}], got {x!r}")
    return x


def _check_correct(value: Any, name: str) -> int | float:
    """0/1 correctness (bools allowed) or a float score in [0, 1] (Acc in Section 3)."""
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    if isinstance(value, numbers.Integral):
        if int(value) not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1 (or a float score in [0, 1]), got {value!r}")
        return int(value)
    return _check_number(value, name, low=0.0, high=1.0)


def make_record(
    id: str,
    u: float,
    *,
    small_correct: Any = None,
    large_correct: Any = None,
    split: str | None = None,
    small_score: float | None = None,
    large_score: float | None = None,
    latency_small_ms: float | None = None,
    latency_large_ms: float | None = None,
    entropy: float | None = None,
    max_prob: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate one JSONL traffic record.

    The format, shared with the ``ucci`` command line and the benchmarks, is
    one JSON object per query with at least ``id``, ``u``, ``small_correct``
    and ``large_correct``, plus optional ``split``, ``small_score``,
    ``large_score``, ``latency_small_ms``, ``latency_large_ms``, ``entropy``
    and ``max_prob``. Records logged at serving time usually lack the two
    correctness fields until the answers are labelled; add them with
    :func:`attach_labels` before fitting.

    Parameters
    ----------
    id : str
        Non-empty query identifier.
    u : float in [0, 1]
        u(x), Eq. 4. Log other signals under ``entropy``, ``max_prob`` or an
        extra field, not here.
    small_correct, large_correct : 0, 1, bool or float in [0, 1], optional
        Whether each model's answer is right (e(x) = 1 - small_correct in
        Section 4.2), or a per-query score in [0, 1].
    split : {"cal", "val", "test"}, optional
    small_score, large_score : float in [0, 1], optional
        Per-query accuracy scores (for example micro-F1) when correctness is
        binary exact match.
    latency_small_ms, latency_large_ms : float >= 0, optional
    entropy : float >= 0, optional
        Mean token entropy (entropy baseline).
    max_prob : float in [0, 1], optional
        Mean max probability (signal ablation).
    extra : mapping, optional
        Additional JSON-serialisable fields; they must not reuse a standard
        field name.

    Returns
    -------
    dict
        Fields in :data:`RECORD_FIELDS` order, None values omitted, extras
        last.

    Raises
    ------
    ValueError
        On an invalid value or a clashing extra field.
    """
    if not isinstance(id, str) or not id:
        raise ValueError(f"id must be a non-empty string, got {id!r}")
    rec: dict[str, Any] = {"id": id}
    if split is not None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        rec["split"] = split
    rec["u"] = _check_number(u, "u", low=0.0, high=1.0)
    if small_correct is not None:
        rec["small_correct"] = _check_correct(small_correct, "small_correct")
    if large_correct is not None:
        rec["large_correct"] = _check_correct(large_correct, "large_correct")
    for name, value, low, high in (
        ("small_score", small_score, 0.0, 1.0),
        ("large_score", large_score, 0.0, 1.0),
        ("latency_small_ms", latency_small_ms, 0.0, None),
        ("latency_large_ms", latency_large_ms, 0.0, None),
        ("entropy", entropy, 0.0, None),
        ("max_prob", max_prob, 0.0, 1.0),
    ):
        if value is not None:
            rec[name] = _check_number(value, name, low=low, high=high)
    if extra:
        for key, value in extra.items():
            if key in RECORD_FIELDS:
                raise ValueError(f"extra field {key!r} clashes with a standard record field")
            rec[key] = value
    return rec


def attach_labels(
    records: Iterable[Mapping[str, Any]],
    labels: Mapping[str, tuple[Any, Any]],
    *,
    missing: str = "raise",
) -> list[dict[str, Any]]:
    """Fill in ``small_correct`` and ``large_correct`` from labels keyed by id.

    Parameters
    ----------
    records : iterable of mapping
        Records as logged (for example read back from a JSONL file).
    labels : mapping from id to (small_correct, large_correct)
        Correctness of each model's answer for that query. A value of None
        leaves the field as it was.
    missing : {"raise", "skip", "keep"}, default "raise"
        What to do with a record whose id has no label: raise, drop it, or
        keep it unlabelled.

    Returns
    -------
    list of dict
        New, validated records; the inputs are not modified.
    """
    if missing not in ("raise", "skip", "keep"):
        raise ValueError(f"missing must be 'raise', 'skip' or 'keep', got {missing!r}")
    out: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        rid = rec.get("id")
        if not isinstance(rid, str) or rid not in labels:
            if missing == "raise":
                raise KeyError(f"records[{i}] (id {rid!r}) has no label")
            if missing == "skip":
                continue
            out.append(dict(rec))
            continue
        small, large = labels[rid]
        new = dict(rec)
        if small is not None:
            new["small_correct"] = _check_correct(small, f"labels[{rid!r}][0]")
        if large is not None:
            new["large_correct"] = _check_correct(large, f"labels[{rid!r}][1]")
        out.append(new)
    return out


class JsonlLogger(contextlib.AbstractContextManager["JsonlLogger"]):
    """Append traffic records to a JSONL file, one validated object per line.

    Thread-safe; each record is flushed as it is written, so a crash loses at
    most the line being written. Usable as a context manager.

    Parameters
    ----------
    path : str or os.PathLike
        Output file. Parent directories must exist.
    mode : {"a", "w"}, default "a"
        Append to or overwrite an existing file.

    Examples
    --------
    >>> import os, tempfile
    >>> path = os.path.join(tempfile.mkdtemp(), "traffic.jsonl")
    >>> with JsonlLogger(path) as log:
    ...     _ = log.write(make_record("q1", 0.12, small_correct=1, large_correct=1))
    >>> open(path).read().strip()
    '{"id": "q1", "u": 0.12, "small_correct": 1, "large_correct": 1}'
    """

    def __init__(self, path: str | os.PathLike[str], *, mode: str = "a") -> None:
        if mode not in ("a", "w"):
            raise ValueError(f"mode must be 'a' or 'w', got {mode!r}")
        self.path = os.fspath(path)
        self._fh: IO[str] | None = open(  # noqa: SIM115 (closed in close())
            self.path, mode, encoding="utf-8", newline="\n"
        )
        self._lock = threading.Lock()

    def write(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Validate ``record`` with :func:`make_record` and append it.

        Returns
        -------
        dict
            The record as written.
        """
        rec = dict(record)
        known = {k: rec.pop(k) for k in RECORD_FIELDS if k in rec}
        if "id" not in known or "u" not in known:
            raise ValueError("a record needs at least 'id' and 'u'")
        clean = make_record(known.pop("id"), known.pop("u"), extra=rec, **known)
        line = json.dumps(clean, allow_nan=False)
        with self._lock:
            if self._fh is None:
                raise ValueError(f"JsonlLogger for {self.path} is closed")
            self._fh.write(line + "\n")
            self._fh.flush()
        return clean

    def log(
        self,
        result: CascadeResult[Any],
        *,
        id: str,
        small_correct: Any = None,
        large_correct: Any = None,
        split: str | None = None,
        include_answers: bool = False,
    ) -> dict[str, Any]:
        """Write the record of one :class:`CascadeResult` (see :meth:`CascadeResult.to_record`)."""
        return self.write(
            result.to_record(
                id,
                small_correct=small_correct,
                large_correct=large_correct,
                split=split,
                include_answers=include_answers,
            )
        )

    def close(self) -> None:
        """Close the file. Further writes raise."""
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __exit__(self, *exc: object) -> None:
        self.close()
