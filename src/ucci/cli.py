"""Command-line interface for UCCI: ``ucci fit | route | evaluate | report | version``.

The CLI runs the paper's three-step protocol (Section 6.1) on logged traffic
with :class:`ucci.UCCIRouter`:

1. ``ucci fit`` fits the isotonic map g on the calibration split (Section 4.2)
   and chooses theta* on the validation split by constrained cost
   minimisation (Section 4.3, Eq. 7), or by maximum accuracy within a cost
   budget (the matched-budget block of Table 2). It writes a router file.
2. ``ucci route`` applies the threshold policy of Eq. 6 to new queries.
3. ``ucci evaluate`` routes every query of a split end to end with the actual
   outputs of both models and reports cost, accuracy, escalation rate and
   savings against always-large (Table 2, Table 3), with percentile
   bootstrap CIs over queries as reported in Section 6.2 (1000 resamples by
   default; arXiv v1 does not print the count).

``ucci report`` gives ECE and the reliability table of raw u(x) and of the
calibrated p_hat (Figure 1). ``ucci version`` prints the version.

Input records
-------------
JSON Lines (one object per line), a JSON array of objects, or CSV with a
header row (tab-separated for ``.tsv``). The format comes from the extension
(``.jsonl``, ``.ndjson``, ``.json``, ``.csv``, ``.tsv``) or is sniffed from
the first character; ``--format`` overrides it and ``--data -`` reads
standard input. Fields:

``id`` (string, optional)
    Query identifier. Defaults to the 0-based record index.
``u`` (number in [0, 1])
    Token-margin uncertainty u(x) of the small model (Section 4.1, Eq. 4).
``small_correct``, ``large_correct`` (number in [0, 1])
    Correctness of each model's output: 0/1 (``true``/``false`` accepted), or
    a per-query score. The calibration event is ``e = 1 - small_correct``
    (Section 4.2); with 0/1 values this is the paper's binary error event.
``small_score``, ``large_score`` (number in [0, 1], optional)
    Per-query quality used for accuracy instead of the ``*_correct`` fields,
    for example per-query F1. ``--metric auto`` (the default) uses them when
    every record has both.
``split`` (optional)
    ``cal``, ``val`` or ``test`` (also ``calibration``, ``validation``,
    ``dev``; case-insensitive).
``latency_small_ms``, ``latency_large_ms`` (number > 0, optional)
    Used by ``fit --cost-from-latency``.

Every value of a required field is validated on every record, so a bad
record anywhere in the file is reported with its line and id. Accuracy in
the CLI is the mean per-query score of the returned answers; corpus-level
metrics such as the paper's micro-F1 need the Python API
(``ucci.routed_micro_f1`` passed as ``metric=``).

Splits
------
``fit`` takes its calibration and validation splits from a split field
(``--split-field NAME``) or from a seeded random split (``--cal-frac``,
``--val-frac``, ``--seed``; defaults 0.3, 0.2 and 0, the paper's 30/20/50).
With none of these options it uses the ``split`` field when every record has
one, the default random split when no record has one, and stops with an
error when only some records have one. The random split does not depend
on record order, platform or numpy version: records are ordered by the
SHA-256 hex digest of ``"<seed>:<id>"`` (ties by position), the first
``floor(cal_frac * n + 0.5)`` form the calibration split, the next
``floor(val_frac * n + 0.5)`` the validation split, and the rest the test
split. The router file records how the split was made, so
``evaluate --split test`` and ``report --split test`` re-derive the same
test split from the same records (checked by count and by a digest of the
ids).

Router file
-----------
The shared ``ucci-router`` version 1 format (:mod:`ucci.io`): calibrator
knots ``x`` and ``y``, ``theta``, ``c_small``, ``c_large``, ``cost_model``,
``tau``, ``grid_step`` and ``created_by``. ``fit`` adds a ``"fit"`` object
with the provenance of the fit (data digests, split, objective, metric,
costs, validation and calibration results); readers ignore keys they do not
know.

JSON output (``--json``)
------------------------
Every command prints one JSON object with ``"command"``,
``"schema_version": 1`` and ``"ucci_version"``. NaN and undefined values are
``null``. Within a schema version keys are only ever added, never renamed
or removed.

``fit``
    ``router_path``; ``data`` {source, format, n_records, sha256,
    ids_sha256}; ``split`` {method ("field" or "random"), field, cal_frac,
    val_frac, seed, sizes {cal, val, test}}; ``objective`` {type
    ("accuracy_target" or "cost_budget"), tau, budget}; ``metric``
    ("correct" or "score"); ``costs`` {c_small, c_large, cost_model, source
    ("options" or "latency")}; ``grid_step``; ``theta``; ``validation`` {n,
    cost, accuracy, escalation_rate, savings_vs_large, always_small {cost,
    accuracy}, always_large {cost, accuracy}}; ``calibration`` {n, n_knots,
    bins, strategy, ece_raw_cal, ece_calibrated_cal, ece_raw_val,
    ece_calibrated_val}. ``ece_calibrated_cal`` is in-sample and is 0 up to
    round-off: an isotonic fit matches the bin frequencies of its own data.
``route``
    ``theta``; ``n``; ``n_escalated``; ``queries``: list of {id, u, p_hat,
    escalate, decision ("small" or "large")}.
``evaluate``
    ``data``; ``split`` (null for every record); ``n``; ``metric``;
    ``router`` {theta, tau, c_small, c_large, cost_model}; ``ucci`` {cost,
    accuracy, escalation_rate, savings_vs_large, accuracy_minus_tau};
    ``always_small`` {cost, accuracy}; ``always_large`` {cost, accuracy};
    ``assumption_ii`` {large_accuracy_escalated, large_accuracy_all, gap}
    (the check of Theorem 1, assumption (ii), in Section 6.3);
    ``bootstrap`` {n_boot, seed, level, ci {cost, accuracy,
    escalation_rate, savings_vs_large}: [low, high]}, or null with
    ``--bootstrap 0``.
``report``
    ``data``; ``split``; ``n``; ``event``; ``bins``; ``strategy``; ``raw``
    and ``calibrated``, each {ece, ece_ci ([low, high] or null),
    reliability: list of {bin_lower, bin_upper, count, mean_forecast,
    observed_frequency}}; ``bootstrap`` {n_boot, seed, level} or null.
``version``
    ``python``; ``numpy``; ``router_format_version``.

Exit codes
----------
0 success; 2 usage error (bad or conflicting options); 3 input error
(unreadable file, invalid record, missing column, empty split, invalid
router file, unwritable output); 4 infeasible objective (no threshold on the
grid reaches tau on the validation split, or every threshold costs more
than the budget).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn, TextIO

import numpy as np

from . import __version__
from .io import FORMAT_VERSION, _create_temp_file, validate_router_dict
from .metrics import bootstrap_ci, ece, reliability_table
from .policy import (
    DEFAULT_COST_LARGE,
    DEFAULT_COST_SMALL,
    DEFAULT_GRID_STEP,
    InfeasibleTargetError,
    ThresholdChoice,
    evaluate,
    make_grid,
    pareto_frontier,
    policy_accuracy,
    policy_cost,
)
from .router import UCCIRouter

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
    IntArray = NDArray[np.int64]

__all__ = [
    "EXIT_INFEASIBLE",
    "EXIT_INPUT",
    "EXIT_OK",
    "EXIT_USAGE",
    "JSON_SCHEMA_VERSION",
    "build_parser",
    "main",
]

#: Exit code on success.
EXIT_OK = 0
#: Exit code for bad or conflicting options.
EXIT_USAGE = 2
#: Exit code for unreadable or invalid input data, router files or outputs.
EXIT_INPUT = 3
#: Exit code when no threshold meets the accuracy target or the budget.
EXIT_INFEASIBLE = 4

#: Version of the ``--json`` output schemas documented in the module docstring.
JSON_SCHEMA_VERSION = 1

#: Defaults: the paper's 30/20/50 splits (Section 6.1) and 1000 bootstrap resamples.
PAPER_CAL_FRAC = 0.3
PAPER_VAL_FRAC = 0.2
PAPER_N_BOOT = 1000

_SPLIT_ALIASES = {
    "cal": "cal",
    "calib": "cal",
    "calibration": "cal",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
}
_SPLIT_NAMES = ("cal", "val", "test")
_SPLIT_LONG = {"cal": "calibration", "val": "validation", "test": "test"}
_FORMATS = ("auto", "jsonl", "json", "csv")
_METRICS = ("auto", "correct", "score")
_COST_MODELS = ("routing", "sequential")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CLIError(Exception):
    """An error reported as ``ucci <command>: error: <message>``."""

    exit_code = EXIT_INPUT


class UsageError(CLIError):
    """Bad or conflicting options (exit code 2)."""

    exit_code = EXIT_USAGE


class InputError(CLIError):
    """Unreadable or invalid data, router file or output path (exit code 3)."""

    exit_code = EXIT_INPUT


class InfeasibleError(CLIError):
    """No threshold meets the requested objective (exit code 4)."""

    exit_code = EXIT_INFEASIBLE


# ---------------------------------------------------------------------------
# Reading records
# ---------------------------------------------------------------------------


def _present(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and value.strip() == "")


@dataclass
class _Table:
    """Records read from one input, with their positions for error messages."""

    source: str
    fmt: str
    rows: list[dict[str, Any]]
    locs: list[str]
    sha256: str
    _numeric: dict[str, FloatArray] = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return len(self.rows)

    def where(self, i: int) -> str:
        """Position of record ``i`` for messages: source, line and id."""
        rid = self.rows[i].get("id")
        idpart = f", id {rid!r}" if _present(rid) else ""
        return f"{self.source}: {self.locs[i]}{idpart}"

    def has_everywhere(self, name: str) -> bool:
        return all(_present(r.get(name)) for r in self.rows)

    def has_anywhere(self, name: str) -> bool:
        return any(_present(r.get(name)) for r in self.rows)

    def columns(self) -> list[str]:
        seen: dict[str, None] = {}
        for r in self.rows:
            for k in r:
                seen.setdefault(str(k), None)
        return list(seen)

    def require(self, names: Sequence[str], command: str) -> None:
        """Fail with one clear message if a required field is absent anywhere."""
        missing = [n for n in names if not self.has_anywhere(n)]
        if missing:
            found = ", ".join(self.columns()) or "none"
            raise InputError(
                f"{self.source}: missing column(s) {', '.join(missing)}; "
                f"'ucci {command}' needs {', '.join(names)} (found: {found})"
            )
        for n in names:
            for i, r in enumerate(self.rows):
                if not _present(r.get(n)):
                    raise InputError(f"{self.where(i)}: missing value for '{n}'")

    def numeric(
        self, name: str, *, lo: float = 0.0, hi: float | None = 1.0, lo_open: bool = False
    ) -> FloatArray:
        """Column ``name`` as floats, every record validated against the range.

        The range is ``[lo, hi]`` (``(lo, hi]`` with ``lo_open``; no upper
        bound when ``hi`` is None). Results are cached per column.
        """
        cached = self._numeric.get(name)
        if cached is not None:
            return cached
        vals = np.empty(self.n, dtype=np.float64)
        for i, r in enumerate(self.rows):
            v = self._to_float(r.get(name), i, name)
            below = v <= lo if lo_open else v < lo
            if below or (hi is not None and v > hi):
                bounds = (
                    f"{'(' if lo_open else '['}{lo}, {hi}]"
                    if hi is not None
                    else f"{'(' if lo_open else '['}{lo}, inf)"
                )
                raise InputError(f"{self.where(i)}: '{name}' = {v!r} is outside {bounds}")
            vals[i] = v
        self._numeric[name] = vals
        return vals

    def _to_float(self, value: Any, i: int, name: str) -> float:
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            out = float(value)
        elif isinstance(value, str):
            text = value.strip()
            if text.lower() in ("true", "false"):
                return 1.0 if text.lower() == "true" else 0.0
            try:
                out = float(text)
            except ValueError:
                raise InputError(f"{self.where(i)}: '{name}' is not a number: {value!r}") from None
        elif value is None:  # pragma: no cover - require() reports missing values first
            raise InputError(f"{self.where(i)}: missing value for '{name}'")
        else:
            raise InputError(
                f"{self.where(i)}: '{name}' must be a number, got {type(value).__name__}"
            )
        if not math.isfinite(out):
            raise InputError(f"{self.where(i)}: '{name}' is {out!r}; values must be finite")
        return out

    def ids(self) -> list[str]:
        """Record ids as strings; a record without an id gets its 0-based index."""
        return [str(r["id"]) if _present(r.get("id")) else str(i) for i, r in enumerate(self.rows)]

    def ids_digest(self) -> str:
        """SHA-256 of the sorted ids, one per line: identifies the set of records."""
        return hashlib.sha256("\n".join(sorted(self.ids())).encode("utf-8")).hexdigest()

    def describe(self) -> dict[str, Any]:
        name = self.source if self.source == "<stdin>" else os.path.basename(self.source)
        return {
            "source": name,
            "format": self.fmt,
            "n_records": self.n,
            "sha256": self.sha256,
            "ids_sha256": self.ids_digest(),
        }


def _detect_format(path: str, text: str, requested: str) -> str:
    if requested != "auto":
        return requested
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jsonl", ".ndjson"):
        return "jsonl"
    if ext == ".json":
        return "json" if text.lstrip().startswith("[") else "jsonl"
    if ext in (".csv", ".tsv"):
        return "csv"
    head = text.lstrip()[:1]
    if head == "[":
        return "json"
    if head == "{":
        return "jsonl"
    return "csv"


def _read_bytes(path: str) -> tuple[bytes, str]:
    if path == "-":
        stream = getattr(sys.stdin, "buffer", None)
        data = stream.read() if stream is not None else sys.stdin.read().encode("utf-8")
        return data, "<stdin>"
    try:
        with open(path, "rb") as fh:
            return fh.read(), path
    except FileNotFoundError:
        raise InputError(f"cannot read {path}: no such file") from None
    except IsADirectoryError:
        raise InputError(f"cannot read {path}: it is a directory") from None
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc.strerror or exc}") from None


def _parse_jsonl(text: str, source: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    locs: list[str] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(
                f"{source}: line {lineno}: invalid JSON ({exc.msg} at column {exc.colno})"
            ) from None
        if not isinstance(obj, dict):
            raise InputError(
                f"{source}: line {lineno}: expected a JSON object, got {type(obj).__name__}"
            )
        rows.append(obj)
        locs.append(f"line {lineno}")
    return rows, locs


def _parse_json_array(text: str, source: str) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InputError(
            f"{source}: invalid JSON ({exc.msg} at line {exc.lineno}, column {exc.colno})"
        ) from None
    if not isinstance(doc, list):
        raise InputError(f"{source}: expected a JSON array of objects, or JSON Lines")
    rows: list[dict[str, Any]] = []
    for i, obj in enumerate(doc):
        if not isinstance(obj, dict):
            raise InputError(
                f"{source}: element {i}: expected a JSON object, got {type(obj).__name__}"
            )
        rows.append(obj)
    return rows, [f"element {i}" for i in range(len(rows))]


def _parse_csv(text: str, source: str, delimiter: str) -> tuple[list[dict[str, Any]], list[str]]:
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames:
        raise InputError(f"{source}: no records (empty file or missing header row)")
    reader.fieldnames = [h.strip() for h in reader.fieldnames]
    rows: list[dict[str, Any]] = []
    locs: list[str] = []
    for rec in reader:
        if None in rec:
            raise InputError(f"{source}: line {reader.line_num}: more values than header columns")
        rows.append({k: v for k, v in rec.items() if _present(v)})
        locs.append(f"line {reader.line_num}")
    return rows, locs


def _read_table(path: str, fmt: str = "auto") -> _Table:
    """Read JSONL, a JSON array or CSV records from ``path`` (``-`` is stdin)."""
    raw, source = _read_bytes(path)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputError(f"{source}: not UTF-8 text ({exc.reason} at byte {exc.start})") from None
    kind = _detect_format(path, text, fmt)
    if kind == "jsonl":
        rows, locs = _parse_jsonl(text, source)
    elif kind == "json":
        rows, locs = _parse_json_array(text, source)
    else:
        delimiter = "\t" if path.lower().endswith(".tsv") else ","
        rows, locs = _parse_csv(text, source, delimiter)
    if not rows:
        raise InputError(f"{source}: no records")
    return _Table(source, kind, rows, locs, hashlib.sha256(raw).hexdigest())


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


@dataclass
class _Split:
    """Assignment of records to the calibration, validation and test splits."""

    method: str  # "field" or "random"
    idx: dict[str, IntArray]
    field_name: str | None = None
    cal_frac: float | None = None
    val_frac: float | None = None
    seed: int | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "field": self.field_name,
            "cal_frac": self.cal_frac,
            "val_frac": self.val_frac,
            "seed": self.seed,
            "sizes": {k: int(self.idx[k].size) for k in _SPLIT_NAMES},
        }


def _split_by_field(table: _Table, name: str) -> _Split:
    labels: dict[str, list[int]] = {k: [] for k in _SPLIT_NAMES}
    for i, r in enumerate(table.rows):
        v = r.get(name)
        if not _present(v):
            raise InputError(f"{table.where(i)}: missing value for split field '{name}'")
        key = _SPLIT_ALIASES.get(str(v).strip().lower())
        if key is None:
            raise InputError(
                f"{table.where(i)}: split field '{name}' = {v!r}; expected cal, val or "
                "test (or calibration, validation, dev)"
            )
        labels[key].append(i)
    return _Split(
        "field", {k: np.asarray(v, dtype=np.int64) for k, v in labels.items()}, field_name=name
    )


def _round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def _split_random(table: _Table, cal_frac: float, val_frac: float, seed: int) -> _Split:
    """The documented seeded split: order by SHA-256 of ``"<seed>:<id>"``, then cut."""
    keys = [hashlib.sha256(f"{seed}:{rid}".encode()).hexdigest() for rid in table.ids()]
    order = sorted(range(table.n), key=lambda i: (keys[i], i))
    n_cal = min(_round_half_up(cal_frac * table.n), table.n)
    n_val = min(_round_half_up(val_frac * table.n), table.n - n_cal)
    parts = {
        "cal": order[:n_cal],
        "val": order[n_cal : n_cal + n_val],
        "test": order[n_cal + n_val :],
    }
    return _Split(
        "random",
        {k: np.asarray(sorted(v), dtype=np.int64) for k, v in parts.items()},
        cal_frac=cal_frac,
        val_frac=val_frac,
        seed=seed,
    )


def _check_fracs(cal_frac: float, val_frac: float) -> None:
    for name, v in (("--cal-frac", cal_frac), ("--val-frac", val_frac)):
        if not (math.isfinite(v) and 0.0 < v < 1.0):
            raise UsageError(f"{name} must lie in (0, 1), got {v}")
    if cal_frac + val_frac > 1.0 + 1e-12:
        raise UsageError(f"--cal-frac + --val-frac must be at most 1, got {cal_frac + val_frac:g}")


def _fit_split(table: _Table, args: argparse.Namespace) -> _Split:
    random_opts = [
        opt
        for opt, v in (
            ("--cal-frac", args.cal_frac),
            ("--val-frac", args.val_frac),
            ("--seed", args.seed),
        )
        if v is not None
    ]
    if args.split_field is not None and random_opts:
        raise UsageError(
            f"--split-field cannot be combined with {', '.join(random_opts)}; "
            "use either a split field or a random split"
        )
    if args.split_field is not None:
        if not table.has_anywhere(args.split_field):
            raise InputError(f"{table.source}: no split field '{args.split_field}' in the records")
        return _split_by_field(table, args.split_field)
    if not random_opts and table.has_everywhere("split"):
        return _split_by_field(table, "split")
    if not random_opts and table.has_anywhere("split"):
        missing = next(i for i, r in enumerate(table.rows) if not _present(r.get("split")))
        raise InputError(
            f"{table.where(missing)}: no 'split' value, while other records have one; "
            "label every record, or pass --cal-frac, --val-frac or --seed for a random split"
        )
    cal = PAPER_CAL_FRAC if args.cal_frac is None else float(args.cal_frac)
    val = PAPER_VAL_FRAC if args.val_frac is None else float(args.val_frac)
    _check_fracs(cal, val)
    return _split_random(table, cal, val, 0 if args.seed is None else int(args.seed))


def _rederive_random(table: _Table, loaded: _LoadedRouter, prov: dict[str, Any]) -> _Split:
    data = loaded.fit.get("data")
    data = data if isinstance(data, dict) else {}
    n_fit = data.get("n_records")
    if n_fit is not None and n_fit != table.n:
        raise InputError(
            f"{table.source}: has {table.n} records but the router was fit on {n_fit}; "
            "a random split can only be re-derived on the same records (pass "
            "--split-field NAME to use a split field of this file instead)"
        )
    digest = data.get("ids_sha256")
    if digest is not None and digest != table.ids_digest():
        raise InputError(
            f"{table.source}: the record ids differ from the data the router was fit on; "
            "a random split can only be re-derived on the same records (pass "
            "--split-field NAME to use a split field of this file instead)"
        )
    try:
        cal, val, seed = float(prov["cal_frac"]), float(prov["val_frac"]), int(prov["seed"])
    except (KeyError, TypeError, ValueError):
        raise InputError(f"{loaded.path}: 'fit.split' is incomplete") from None
    return _split_random(table, cal, val, seed)


def _select_rows(
    table: _Table, split: str | None, split_field: str | None, loaded: _LoadedRouter
) -> tuple[IntArray, str | None]:
    """Rows for ``evaluate`` and ``report``: every record, or one split.

    The split comes from ``--split-field``, else from the router's fit
    provenance (a random split is re-derived, a field split reuses its
    field), else from a ``split`` field present on every record.
    """
    if split is None:
        if split_field is not None:
            raise UsageError("--split-field needs --split")
        return np.arange(table.n, dtype=np.int64), None
    key = _SPLIT_ALIASES.get(split.strip().lower())
    if key is None:
        raise UsageError(f"--split must be cal, val or test, got {split!r}")
    prov = loaded.fit.get("split")
    prov = prov if isinstance(prov, dict) else {}
    prov_field = prov.get("field")
    if split_field is not None:
        if not table.has_anywhere(split_field):
            raise InputError(f"{table.source}: no split field '{split_field}' in the records")
        plan = _split_by_field(table, split_field)
    elif prov.get("method") == "random":
        plan = _rederive_random(table, loaded, prov)
    elif (
        prov.get("method") == "field"
        and isinstance(prov_field, str)
        and table.has_everywhere(prov_field)
    ):
        plan = _split_by_field(table, prov_field)
    elif table.has_everywhere("split"):
        plan = _split_by_field(table, "split")
    else:
        raise InputError(
            f"{table.source}: cannot select split '{key}': the records have no split field "
            "and the router file does not record a random split; pass --split-field NAME, "
            "or evaluate a separate file without --split"
        )
    rows = plan.idx[key]
    if rows.size == 0:
        raise InputError(f"{table.source}: the {_SPLIT_LONG[key]} split is empty")
    return rows, key


# ---------------------------------------------------------------------------
# Router files
# ---------------------------------------------------------------------------


@dataclass
class _LoadedRouter:
    """A router read from a file, plus the file's ``fit`` provenance if any."""

    path: str
    router: UCCIRouter
    fit: dict[str, Any]

    def p_hat(self, u: FloatArray) -> FloatArray:
        return self.router.route(u).p_hat


def _reject_constant(name: str) -> float:
    raise ValueError(f"{name} is not valid JSON")


def _load_router(path: str) -> _LoadedRouter:
    """Read a router file; the contract is checked by :func:`ucci.io.validate_router_dict`."""
    raw, source = _read_bytes(path)
    try:
        doc = json.loads(raw.decode("utf-8-sig"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise InputError(f"{source}: not a valid router JSON file ({exc})") from None
    try:
        router = UCCIRouter.from_dict(doc)
    except ValueError as exc:
        raise InputError(f"{source}: {exc}") from None
    fit = doc.get("fit")
    return _LoadedRouter(source, router, fit if isinstance(fit, dict) else {})


def _jsonable(obj: Any) -> Any:
    """Plain JSON values: numpy scalars converted, NaN and inf written as null."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def _write_json_atomic(path: str, doc: dict[str, Any]) -> None:
    """Write ``doc`` through a temporary file and an atomic rename."""
    text = json.dumps(_jsonable(doc), indent=2, allow_nan=False) + "\n"
    directory = os.path.dirname(os.path.abspath(path))
    try:
        fd, tmp = _create_temp_file(directory, os.path.basename(path))
    except OSError as exc:
        raise InputError(f"cannot write {path}: {exc.strerror or exc}") from None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except OSError as exc:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise InputError(f"cannot write {path}: {exc.strerror or exc}") from None


# ---------------------------------------------------------------------------
# Shared computations
# ---------------------------------------------------------------------------


def _metric_kind(table: _Table, requested: str) -> str:
    """Resolve ``--metric``: which fields hold the per-query accuracy."""
    if requested == "correct":
        return "correct"
    if requested == "score":
        table.require(["small_score", "large_score"], "--metric score")
        return "score"
    names = ("small_score", "large_score")
    if not any(table.has_anywhere(n) for n in names):
        return "correct"
    incomplete = [n for n in names if not table.has_everywhere(n)]
    if incomplete:
        raise InputError(
            f"{table.source}: {', '.join(incomplete)} is missing on some or all records while "
            "other score fields are present; give small_score and large_score on every "
            "record, or pass --metric correct"
        )
    return "score"


def _scores(table: _Table, metric: str) -> tuple[FloatArray, FloatArray]:
    if metric == "score":
        return table.numeric("small_score"), table.numeric("large_score")
    return table.numeric("small_correct"), table.numeric("large_correct")


def _check_costs(c_small: float, c_large: float, cost_model: str) -> None:
    for name, v in (("--c-small", c_small), ("--c-large", c_large)):
        if not (math.isfinite(v) and v > 0.0):
            raise UsageError(f"{name} must be a positive number, got {v}")
    if cost_model == "routing" and not c_large > c_small:
        raise UsageError(
            "the routing cost model needs c_large > c_small (Theorem 1, assumption (i)); "
            f"got c_small = {c_small:g}, c_large = {c_large:g}"
        )


def _savings(cost: float, c_large: float) -> float:
    """Relative cost reduction against always-large, ``1 - cost / c_large`` (Table 3)."""
    return 1.0 - cost / c_large


def _ece(p: FloatArray, e: FloatArray, bins: int, strategy: str) -> float:
    return float(ece(p, e, n_bins=bins, strategy=strategy))


def _check_boot(n_boot: int, level: float) -> None:
    if n_boot < 0:
        raise UsageError(f"--bootstrap must be >= 0, got {n_boot}")
    if not (math.isfinite(level) and 0.0 < level < 1.0):
        raise UsageError(f"--level must lie in (0, 1), got {level}")


def _ci(
    stat: Callable[[IntArray], float], n: int, n_boot: int, level: float, seed: int
) -> list[float]:
    """Percentile bootstrap CI over queries (:func:`ucci.bootstrap_ci`)."""
    low, high = bootstrap_ci(stat, n, n_boot=n_boot, alpha=1.0 - level, seed=seed)
    return [float(low), float(high)]


def _header(command: str) -> dict[str, Any]:
    return {"command": command, "schema_version": JSON_SCHEMA_VERSION, "ucci_version": __version__}


def _emit_json(doc: dict[str, Any], out: TextIO) -> None:
    out.write(json.dumps(_jsonable(doc), indent=2, allow_nan=False) + "\n")


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def _fmt_ci(ci: Sequence[float] | None, pct: bool = False) -> str:
    if ci is None:
        return ""
    if pct:
        return f"[{100 * ci[0]:.1f}%, {100 * ci[1]:.1f}%]"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


# ---------------------------------------------------------------------------
# ucci fit
# ---------------------------------------------------------------------------


def _cmd_fit(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if args.tau is not None and not (math.isfinite(args.tau) and 0.0 <= args.tau <= 1.0):
        raise UsageError(f"--tau must lie in [0, 1], got {args.tau}")
    if args.budget is not None and not (math.isfinite(args.budget) and args.budget > 0.0):
        raise UsageError(f"--budget must be a positive number, got {args.budget}")
    try:
        make_grid(args.grid_step)
    except ValueError as exc:
        raise UsageError(f"--grid-step: {exc}") from None
    if args.cost_from_latency and (args.c_small is not None or args.c_large is not None):
        raise UsageError("--cost-from-latency cannot be combined with --c-small or --c-large")
    if not args.cost_from_latency:
        c_small = DEFAULT_COST_SMALL if args.c_small is None else float(args.c_small)
        c_large = DEFAULT_COST_LARGE if args.c_large is None else float(args.c_large)
        _check_costs(c_small, c_large, args.cost_model)

    table = _read_table(args.data, args.format)
    table.require(["u", "small_correct", "large_correct"], "fit")
    u = table.numeric("u")
    e = 1.0 - table.numeric("small_correct")
    table.numeric("large_correct")
    metric = _metric_kind(table, args.metric)
    small, large = _scores(table, metric)

    split = _fit_split(table, args)
    for key in ("cal", "val"):
        if split.idx[key].size == 0:
            why = (
                f"no record has {split.field_name} = {key}"
                if split.method == "field"
                else f"{table.n} records are too few for the requested fractions"
            )
            raise InputError(f"{table.source}: the {_SPLIT_LONG[key]} split is empty ({why})")
    cal, val = split.idx["cal"], split.idx["val"]

    if args.cost_from_latency:
        table.require(["latency_small_ms", "latency_large_ms"], "fit --cost-from-latency")
        fit_rows = np.concatenate([cal, val])
        lat_s = table.numeric("latency_small_ms", hi=None, lo_open=True)[fit_rows]
        lat_l = table.numeric("latency_large_ms", hi=None, lo_open=True)[fit_rows]
        c_small, c_large = 1.0, float(lat_l.mean() / lat_s.mean())
        cost_source = "latency"
        if args.cost_model == "routing" and not c_large > c_small:
            raise InputError(
                f"{table.source}: the measured latency ratio c_large / c_small = "
                f"{c_large:.4g} is not above 1, which the routing cost model needs "
                "(Theorem 1, assumption (i))"
            )
    else:
        cost_source = "options"

    router = UCCIRouter(c_small, c_large, args.cost_model, args.grid_step)
    router.calibrate(u[cal], e[cal])
    choice = _choose(router, u[val], small[val], large[val], args)

    bins, strategy = 10, "uniform"
    p_cal = router.route(u[cal]).p_hat
    p_val = router.route(u[val]).p_hat
    calibration: dict[str, Any] = {
        "n": int(cal.size),
        "n_knots": len(router.calibrator.to_dict()["x"]),
        "bins": bins,
        "strategy": strategy,
        "ece_raw_cal": _ece(u[cal], e[cal], bins, strategy),
        "ece_calibrated_cal": _ece(p_cal, e[cal], bins, strategy),
        "ece_raw_val": _ece(u[val], e[val], bins, strategy),
        "ece_calibrated_val": _ece(p_val, e[val], bins, strategy),
    }
    validation: dict[str, Any] = {
        "n": int(val.size),
        "cost": choice.cost,
        "accuracy": choice.accuracy,
        "escalation_rate": choice.escalation_rate,
        "savings_vs_large": _savings(choice.cost, c_large),
        "always_small": {"cost": c_small, "accuracy": float(small[val].mean())},
        "always_large": {"cost": c_large, "accuracy": float(large[val].mean())},
    }
    objective = (
        {"type": "accuracy_target", "tau": float(args.tau), "budget": None}
        if args.tau is not None
        else {"type": "cost_budget", "tau": None, "budget": float(args.budget)}
    )
    costs = {
        "c_small": c_small,
        "c_large": c_large,
        "cost_model": args.cost_model,
        "source": cost_source,
    }
    fit_meta: dict[str, Any] = {
        "data": table.describe(),
        "split": split.describe(),
        "objective": objective,
        "metric": metric,
        "costs": costs,
        "validation": validation,
        "calibration": calibration,
    }
    doc = router.to_dict()
    doc["fit"] = fit_meta
    validate_router_dict(doc)  # the shared contract, checked before anything is written
    _write_json_atomic(args.out, doc)

    if args.json:
        result = _header("fit")
        result.update(
            {
                "router_path": args.out,
                "data": fit_meta["data"],
                "split": fit_meta["split"],
                "objective": objective,
                "metric": metric,
                "costs": costs,
                "grid_step": router.grid_step,
                "theta": choice.theta,
                "validation": validation,
                "calibration": calibration,
            }
        )
        _emit_json(result, out)
        return EXIT_OK

    sizes = fit_meta["split"]["sizes"]
    how = (
        f"field '{split.field_name}'"
        if split.method == "field"
        else f"random (seed {split.seed}, fractions {split.cal_frac:g}/{split.val_frac:g})"
    )
    target = (
        f"accuracy >= {args.tau:g} on validation"
        if args.tau is not None
        else f"highest accuracy with cost <= {args.budget:g} on validation"
    )
    lines = [
        f"ucci fit: wrote {args.out}",
        f"  data         {table.source} ({table.n} records, {table.fmt})",
        f"  split        {how}: cal {sizes['cal']}, val {sizes['val']}, test {sizes['test']}",
        f"  objective    {target} (metric: {metric})",
        (
            f"  costs        c_small {c_small:g}, c_large {c_large:.4g}, "
            f"{args.cost_model} cost model (from {cost_source})"
        ),
        (
            f"  calibration  ECE raw u {_fmt(calibration['ece_raw_val'])} -> calibrated "
            f"{_fmt(calibration['ece_calibrated_val'])} on the validation split "
            f"(held out, {bins} {strategy} bins)"
        ),
        (
            f"               ECE raw u {_fmt(calibration['ece_raw_cal'])} on the calibration "
            f"split; isotonic map with {calibration['n_knots']} knots"
        ),
        f"  threshold    theta* = {choice.theta:g} (grid step {router.grid_step:g})",
        (
            f"  validation   cost {_fmt(choice.cost)}, accuracy {_fmt(choice.accuracy)}, "
            f"escalation rate {_fmt(choice.escalation_rate)}, savings vs always-large "
            f"{100 * validation['savings_vs_large']:.1f}%"
        ),
        (
            f"  reference    always-small accuracy {_fmt(validation['always_small']['accuracy'])} "
            f"(cost {c_small:g}), always-large accuracy "
            f"{_fmt(validation['always_large']['accuracy'])} (cost {c_large:.4g})"
        ),
    ]
    out.write("\n".join(lines) + "\n")
    return EXIT_OK


def _choose(
    router: UCCIRouter,
    u_val: FloatArray,
    small: FloatArray,
    large: FloatArray,
    args: argparse.Namespace,
) -> ThresholdChoice:
    """Step 2 of Section 6.1: Eq. 7, or its budget form (Table 2, bottom block)."""
    try:
        if args.tau is not None:
            return router.choose_threshold(u_val, small, large, tau=args.tau)
        return router.choose_threshold_for_budget(u_val, small, large, budget=args.budget)
    except InfeasibleTargetError:
        pass
    p_val = np.asarray(router.error_probability(u_val), dtype=np.float64)
    front = pareto_frontier(
        p_val,
        small,
        large,
        c_small=router.c_small,
        c_large=router.c_large,
        grid=make_grid(router.grid_step),
        cost_model=router.cost_model,
    )
    if args.tau is not None:
        j = int(np.argmax(front.accuracy))
        raise InfeasibleError(
            f"no threshold reaches tau = {args.tau:g} on the validation split "
            f"(n = {u_val.size}); the best achievable validation accuracy is "
            f"{float(front.accuracy[j]):.4f} (theta = {float(front.theta[j]):g}, escalation "
            f"rate {float(front.escalation_rate[j]):.4f}); always-large accuracy is "
            f"{float(large.mean()):.4f}"
        )
    raise InfeasibleError(
        f"no threshold meets the cost budget {args.budget:g} on the validation split; "
        f"the cheapest threshold costs {float(front.cost.min()):.4f} per query"
    )


# ---------------------------------------------------------------------------
# ucci route
# ---------------------------------------------------------------------------


def _cmd_route(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    loaded = _load_router(args.router)
    if args.u is not None:
        u = np.asarray(args.u, dtype=np.float64)
        bad = ~np.isfinite(u) | (u < 0.0) | (u > 1.0)
        if bad.any():
            raise UsageError(f"--u values must be finite and lie in [0, 1]; got {u[bad][0]!r}")
        ids = [str(i) for i in range(u.size)]
    else:
        table = _read_table(args.data, args.format)
        table.require(["u"], "route")
        u = table.numeric("u")
        ids = table.ids()
    res = loaded.router.route(u)
    theta = loaded.router.theta
    if args.json:
        result = _header("route")
        result.update(
            {
                "theta": theta,
                "n": int(u.size),
                "n_escalated": int(res.escalate.sum()),
                "queries": [
                    {
                        "id": i,
                        "u": ui,
                        "p_hat": pi,
                        "escalate": ei,
                        "decision": "large" if ei else "small",
                    }
                    for i, ui, pi, ei in zip(
                        ids, u.tolist(), res.p_hat.tolist(), res.escalate.tolist()
                    )
                ],
            }
        )
        _emit_json(result, out)
        return EXIT_OK
    out.write("id\tu\tp_hat\tdecision\n")
    out.writelines(
        f"{i}\t{ui:.6g}\t{pi:.6f}\t{'large' if ei else 'small'}\n"
        for i, ui, pi, ei in zip(ids, u.tolist(), res.p_hat.tolist(), res.escalate.tolist())
    )
    return EXIT_OK


# ---------------------------------------------------------------------------
# ucci evaluate
# ---------------------------------------------------------------------------


def _cmd_evaluate(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    _check_boot(args.bootstrap, args.level)
    loaded = _load_router(args.router)
    router = loaded.router
    c_small = router.c_small if args.c_small is None else float(args.c_small)
    c_large = router.c_large if args.c_large is None else float(args.c_large)
    cost_model = router.cost_model if args.cost_model is None else args.cost_model
    _check_costs(c_small, c_large, cost_model)

    table = _read_table(args.data, args.format)
    table.require(["u", "small_correct", "large_correct"], "evaluate")
    u_all = table.numeric("u")
    table.numeric("small_correct")
    table.numeric("large_correct")
    metric = _metric_kind(table, args.metric)
    small_all, large_all = _scores(table, metric)
    rows, split = _select_rows(table, args.split, args.split_field, loaded)
    if split is None:
        _note_in_sample(table, loaded, err)
    u, small, large = u_all[rows], small_all[rows], large_all[rows]
    n = int(rows.size)

    # Step 3 of Section 6.1: route each query and take the actual output.
    decisions = router.route(u)
    esc = decisions.escalate
    res = evaluate(
        decisions.p_hat,
        small,
        large,
        router.theta,
        c_small=c_small,
        c_large=c_large,
        cost_model=cost_model,
    )
    acc_large = float(large.mean())
    acc_large_esc = float(large[esc].mean()) if esc.any() else math.nan

    ci: dict[str, list[float]] | None = None
    if args.bootstrap > 0:

        def cost_stat(i: IntArray) -> float:
            return policy_cost(esc[i], c_small, c_large, cost_model)

        def acc_stat(i: IntArray) -> float:
            return policy_accuracy(esc[i], small[i], large[i])

        def rate_stat(i: IntArray) -> float:
            return float(np.count_nonzero(esc[i])) / i.size

        ci = {
            "cost": _ci(cost_stat, n, args.bootstrap, args.level, args.seed),
            "accuracy": _ci(acc_stat, n, args.bootstrap, args.level, args.seed),
            "escalation_rate": _ci(rate_stat, n, args.bootstrap, args.level, args.seed),
        }
        # Savings are a decreasing function of cost, so its percentile CI maps over.
        ci["savings_vs_large"] = [
            _savings(ci["cost"][1], c_large),
            _savings(ci["cost"][0], c_large),
        ]

    tau = router.tau
    result = _header("evaluate")
    result.update(
        {
            "data": table.describe(),
            "split": split,
            "n": n,
            "metric": metric,
            "router": {
                "theta": router.theta,
                "tau": tau,
                "c_small": c_small,
                "c_large": c_large,
                "cost_model": cost_model,
            },
            "ucci": {
                "cost": res.cost,
                "accuracy": res.accuracy,
                "escalation_rate": res.escalation_rate,
                "savings_vs_large": _savings(res.cost, c_large),
                "accuracy_minus_tau": None if tau is None else res.accuracy - tau,
            },
            "always_small": {"cost": c_small, "accuracy": float(small.mean())},
            "always_large": {"cost": c_large, "accuracy": acc_large},
            "assumption_ii": {
                "large_accuracy_escalated": acc_large_esc,
                "large_accuracy_all": acc_large,
                "gap": acc_large - acc_large_esc,
            },
            "bootstrap": None
            if ci is None
            else {"n_boot": args.bootstrap, "seed": args.seed, "level": args.level, "ci": ci},
        }
    )
    if args.json:
        _emit_json(result, out)
        return EXIT_OK

    def row(label: str, value: str, key: str, pct: bool = False) -> str:
        interval = _fmt_ci(ci[key], pct=pct) if ci is not None else ""
        return f"  {label:<18}{value:<12}{interval}".rstrip()

    where = f"split '{split}'" if split else "all records"
    level = f"{100 * args.level:g}% CI" if ci is not None else ""
    lines = [
        f"ucci evaluate: {n} queries ({where}) from {table.source}",
        (
            f"  router            theta {router.theta:g}, {cost_model} cost model, "
            f"c_small {c_small:g}, c_large {c_large:.4g}, metric {metric}"
        ),
        f"  {'':<18}{'estimate':<12}{level}".rstrip(),
        row("cost", _fmt(res.cost), "cost"),
        row("accuracy", _fmt(res.accuracy), "accuracy"),
        row("escalation rate", _fmt(res.escalation_rate), "escalation_rate"),
        row(
            "savings vs large",
            f"{100 * _savings(res.cost, c_large):.1f}%",
            "savings_vs_large",
            pct=True,
        ),
    ]
    if tau is not None:
        lines.append(f"  {'accuracy - tau':<18}{res.accuracy - tau:+.4f}")
    lines += [
        f"  always-small      cost {c_small:.4g}, accuracy {_fmt(float(small.mean()))}",
        f"  always-large      cost {c_large:.4g}, accuracy {_fmt(acc_large)}",
        (
            f"  assumption (ii)   large-model accuracy {_fmt(acc_large_esc)} on escalated "
            f"queries vs {_fmt(acc_large)} on all (Theorem 1)"
        ),
    ]
    out.write("\n".join(lines) + "\n")
    return EXIT_OK


def _note_in_sample(table: _Table, loaded: _LoadedRouter, err: TextIO) -> None:
    data = loaded.fit.get("data")
    if isinstance(data, dict) and data.get("ids_sha256") == table.ids_digest():
        err.write(
            "ucci: note: these are the records the router was fit on, including its "
            "calibration and validation splits; pass --split test for the held-out "
            "evaluation of Section 6.1\n"
        )


# ---------------------------------------------------------------------------
# ucci report
# ---------------------------------------------------------------------------


def _cmd_report(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    _check_boot(args.bootstrap, args.level)
    if args.bins < 1:
        raise UsageError(f"--bins must be >= 1, got {args.bins}")
    loaded = _load_router(args.router)
    table = _read_table(args.data, args.format)
    table.require(["u", "small_correct"], "report")
    u_all = table.numeric("u")
    e_all = 1.0 - table.numeric("small_correct")
    rows, split = _select_rows(table, args.split, args.split_field, loaded)
    u, e = u_all[rows], e_all[rows]
    p = loaded.p_hat(u)
    n = int(rows.size)

    def block(forecast: FloatArray) -> dict[str, Any]:
        ece_ci = None
        if args.bootstrap > 0:

            def stat(i: IntArray) -> float:
                return _ece(forecast[i], e[i], args.bins, args.strategy)

            ece_ci = _ci(stat, n, args.bootstrap, args.level, args.seed)
        return {
            "ece": _ece(forecast, e, args.bins, args.strategy),
            "ece_ci": ece_ci,
            "reliability": [
                r._asdict()
                for r in reliability_table(forecast, e, n_bins=args.bins, strategy=args.strategy)
            ],
        }

    raw, calibrated = block(u), block(p)
    result = _header("report")
    result.update(
        {
            "data": table.describe(),
            "split": split,
            "n": n,
            "event": "small model wrong (e = 1 - small_correct)",
            "bins": args.bins,
            "strategy": args.strategy,
            "raw": raw,
            "calibrated": calibrated,
            "bootstrap": None
            if args.bootstrap == 0
            else {"n_boot": args.bootstrap, "seed": args.seed, "level": args.level},
        }
    )
    if args.json:
        _emit_json(result, out)
        return EXIT_OK

    where = f"split '{split}'" if split else "all records"
    lines = [
        (
            f"ucci report: {n} queries ({where}) from {table.source}; event: small model "
            f"wrong; {args.bins} {args.strategy} bins"
        )
    ]
    for name, blk in (("raw u", raw), ("calibrated p_hat", calibrated)):
        ci_txt = (
            f"  {100 * args.level:g}% CI {_fmt_ci(blk['ece_ci'])}"
            if blk["ece_ci"] is not None
            else ""
        )
        lines.append(f"  ECE {name:<17}{_fmt(blk['ece'])}{ci_txt}")
    for name, blk in (("raw u", raw), ("calibrated p_hat", calibrated)):
        lines += [
            "",
            f"  reliability: {name}",
            (
                f"  {'bin':>17}  {'count':>7}  {'mean forecast':>13}  "
                f"{'observed error':>14}  {'gap':>8}"
            ),
        ]
        for r in blk["reliability"]:
            gap = r["observed_frequency"] - r["mean_forecast"]
            span = f"[{r['bin_lower']:.3f}, {r['bin_upper']:.3f}]"
            lines.append(
                f"  {span:>17}  {r['count']:>7}  {r['mean_forecast']:>13.4f}  "
                f"{r['observed_frequency']:>14.4f}  {gap:>+8.4f}"
            )
    out.write("\n".join(lines) + "\n")
    return EXIT_OK


# ---------------------------------------------------------------------------
# ucci version
# ---------------------------------------------------------------------------


def _cmd_version(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if args.json:
        result = _header("version")
        result.update(
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "router_format_version": FORMAT_VERSION,
            }
        )
        _emit_json(result, out)
    else:
        out.write(f"ucci {__version__}\n")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class _ParseError(Exception):
    """Raised instead of exiting so that :func:`main` can return exit code 2."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        raise _ParseError(f"{self.prog}: error: {message}")


def _add_data(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data",
        required=True,
        metavar="FILE",
        help="JSONL, JSON array or CSV records ('-' reads standard input)",
    )
    p.add_argument(
        "--format",
        choices=_FORMATS,
        default="auto",
        help="input format (default: from the extension, else sniffed)",
    )


def _add_split_select(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--split",
        default=None,
        metavar="NAME",
        help="use only this split: cal, val or test (default: every record). "
        "Taken from --split-field, else re-derived from the router's fit, "
        "else from a 'split' field",
    )
    p.add_argument(
        "--split-field", default=None, metavar="FIELD", help="record field holding the split labels"
    )


def _add_boot(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--bootstrap",
        type=int,
        default=PAPER_N_BOOT,
        metavar="B",
        help="bootstrap resamples over queries, 0 to skip (default: 1000; "
        "the paper's CIs are bootstrap CIs over queries, Section 6.2)",
    )
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed (default: 0)")
    p.add_argument(
        "--level", type=float, default=0.95, help="level of the percentile CIs (default: 0.95)"
    )


def build_parser() -> argparse.ArgumentParser:
    """The ``ucci`` argument parser (also used by documentation tools)."""
    parser = _Parser(
        prog="ucci",
        description=(
            "UCCI: calibrated uncertainty for cost-optimal LLM cascade routing "
            "(Kotte, arXiv:2605.18796). Fit a router on logged traffic, route "
            "queries, evaluate end to end and report calibration."
        ),
        epilog="Exit codes: 0 ok, 2 usage error, 3 input error, 4 infeasible objective.",
    )
    parser.add_argument("--version", action="version", version=f"ucci {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", parser_class=_Parser)

    fit = sub.add_parser(
        "fit",
        help="fit the calibration map and choose theta (Sections 4.2, 4.3)",
        description=(
            "Fit g on the calibration split and choose theta* on the validation "
            "split: the cheapest threshold with accuracy >= tau (Eq. 7), or the "
            "most accurate threshold with cost <= budget (Table 2, bottom block)."
        ),
    )
    _add_data(fit)
    grp = fit.add_argument_group(
        "split",
        "default: the 'split' field when every record has one; a random "
        "30/20/50 split with seed 0 when no record has one",
    )
    grp.add_argument(
        "--split-field",
        default=None,
        metavar="FIELD",
        help="record field with cal / val / test labels",
    )
    grp.add_argument(
        "--cal-frac",
        type=float,
        default=None,
        help="calibration fraction of a random split (default: 0.3)",
    )
    grp.add_argument(
        "--val-frac",
        type=float,
        default=None,
        help="validation fraction of a random split (default: 0.2)",
    )
    grp.add_argument("--seed", type=int, default=None, help="seed of the random split (default: 0)")
    obj = fit.add_mutually_exclusive_group(required=True)
    obj.add_argument("--tau", type=float, help="accuracy target on the validation split")
    obj.add_argument(
        "--budget", type=float, help="mean cost per query allowed; maximise accuracy within it"
    )
    fit.add_argument(
        "--c-small",
        type=float,
        default=None,
        help="cost of a query answered by the small model (default: 1.0)",
    )
    fit.add_argument(
        "--c-large",
        type=float,
        default=None,
        help="cost of an escalated query (default: 3.02, the paper's measured latency ratio)",
    )
    fit.add_argument(
        "--cost-from-latency",
        action="store_true",
        help="c_small = 1 and c_large = mean latency_large_ms / mean "
        "latency_small_ms over the calibration and validation records",
    )
    fit.add_argument(
        "--cost-model",
        choices=_COST_MODELS,
        default="routing",
        help="routing: an escalated query costs c_large; sequential: "
        "c_small + c_large (default: routing)",
    )
    fit.add_argument(
        "--grid-step",
        type=float,
        default=DEFAULT_GRID_STEP,
        help="theta grid resolution on [0, 1] (default: 0.005)",
    )
    fit.add_argument(
        "--metric",
        choices=_METRICS,
        default="auto",
        help="per-query accuracy from the *_score or *_correct fields; auto "
        "uses scores when every record has them (default: auto)",
    )
    fit.add_argument(
        "--out", required=True, metavar="ROUTER_JSON", help="where to write the router file"
    )
    fit.add_argument("--json", action="store_true", help="print the summary as JSON")
    fit.set_defaults(func=_cmd_fit)

    route = sub.add_parser(
        "route",
        help="apply the threshold policy (Eq. 6) to queries",
        description="Print p_hat = g(u) and the decision (large if p_hat > theta) per query.",
    )
    route.add_argument("--router", required=True, metavar="ROUTER_JSON")
    src = route.add_mutually_exclusive_group(required=True)
    src.add_argument("--u", type=float, nargs="+", metavar="U", help="uncertainty values")
    src.add_argument(
        "--data", metavar="FILE", help="records with a 'u' field ('-' reads standard input)"
    )
    route.add_argument(
        "--format", choices=_FORMATS, default="auto", help="input format for --data (default: auto)"
    )
    route.add_argument("--json", action="store_true", help="print JSON")
    route.set_defaults(func=_cmd_route)

    ev = sub.add_parser(
        "evaluate",
        help="end-to-end routing evaluation with bootstrap CIs (Section 6.1)",
        description=(
            "Route every selected query with the router's theta using the actual "
            "outputs of both models; report cost, accuracy, escalation rate and "
            "savings against always-large with percentile bootstrap CIs."
        ),
    )
    ev.add_argument("--router", required=True, metavar="ROUTER_JSON")
    _add_data(ev)
    _add_split_select(ev)
    _add_boot(ev)
    ev.add_argument(
        "--metric",
        choices=_METRICS,
        default="auto",
        help="per-query accuracy source (default: auto)",
    )
    ev.add_argument(
        "--c-small",
        type=float,
        default=None,
        help="report costs with this c_small instead of the router's",
    )
    ev.add_argument(
        "--c-large",
        type=float,
        default=None,
        help="report costs with this c_large instead of the router's (Table 3)",
    )
    ev.add_argument(
        "--cost-model",
        choices=_COST_MODELS,
        default=None,
        help="report costs under this cost model instead of the router's",
    )
    ev.add_argument("--json", action="store_true", help="print JSON")
    ev.set_defaults(func=_cmd_evaluate)

    rep = sub.add_parser(
        "report",
        help="ECE and reliability tables of raw u and calibrated p_hat (Figure 1)",
        description=(
            "Expected calibration error and reliability table of raw u(x) and of "
            "the calibrated p_hat for the event 'small model wrong'."
        ),
    )
    rep.add_argument("--router", required=True, metavar="ROUTER_JSON")
    _add_data(rep)
    _add_split_select(rep)
    rep.add_argument("--bins", type=int, default=10, help="number of bins (default: 10)")
    rep.add_argument(
        "--strategy",
        choices=("uniform", "quantile"),
        default="uniform",
        help="equal-width or equal-count bins (default: uniform)",
    )
    _add_boot(rep)
    rep.add_argument("--json", action="store_true", help="print JSON")
    rep.set_defaults(func=_cmd_report)

    ver = sub.add_parser("version", help="print the version")
    ver.add_argument("--json", action="store_true", help="print JSON")
    ver.set_defaults(func=_cmd_version)
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``ucci`` command line and return its exit code.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments without the program name; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        0 on success, 2 for usage errors, 3 for input errors and 4 when the
        objective is infeasible (see the module docstring).
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except _ParseError as exc:
        sys.stderr.write(f"{exc}\n")
        return EXIT_USAGE
    except SystemExit as exc:  # --help and --version
        return exc.code if isinstance(exc.code, int) else EXIT_OK
    if getattr(args, "command", None) is None:
        parser.print_help(sys.stderr)
        return EXIT_USAGE
    func: Callable[[argparse.Namespace, TextIO, TextIO], int] = args.func
    try:
        return func(args, sys.stdout, sys.stderr)
    except CLIError as exc:
        sys.stderr.write(f"ucci {args.command}: error: {exc}\n")
        return exc.exit_code
    except BrokenPipeError:  # pragma: no cover - output piped into a closed reader
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
