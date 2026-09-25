"""Reading and writing routers as JSON (the ``ucci-router`` format, version 1).

A fitted router is fully described by its calibration knots, its threshold
and its cost settings. The file format is shared by the Python package and
the Rust crate::

    {
      "format": "ucci-router",
      "version": 1,
      "calibrator": {"x": [float, ...], "y": [float, ...]},
      "theta": float,
      "c_small": float,
      "c_large": float,
      "cost_model": "routing" | "sequential",
      "tau": float | null,
      "grid_step": 0.005,
      "created_by": "ucci-python <version>"
    }

Rules checked on read: ``format`` and ``version`` are known; ``x`` is
strictly increasing; ``y`` is non-decreasing within [0, 1]; ``x`` and ``y``
have the same length, at least 1; all numbers are finite; costs are
positive; ``grid_step``, when present, lies in (0, 1] and divides 1. ``tau``,
``grid_step`` and ``created_by`` may be missing or null.
Unknown keys are ignored, so later versions can add fields without breaking
older readers.

Prediction from the file is linear interpolation of ``(x, y)``, clipped at
both ends (``numpy.interp``), and the policy escalates when the prediction
is strictly greater than ``theta`` (Eq. 6).

Floats are written with Python's shortest round-trip representation, so a
saved router predicts bit-identically after loading.
"""

from __future__ import annotations

import json
import math
import numbers
import os
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from ._validation import COST_MODELS

if TYPE_CHECKING:
    from .router import UCCIRouter

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "dumps_router_dict",
    "load_router",
    "read_router_dict",
    "save_router",
    "validate_router_dict",
    "write_router_dict",
]

#: Value of the ``format`` field.
FORMAT_NAME = "ucci-router"
#: The format version this package writes and reads.
FORMAT_VERSION = 1

_FALLBACK_VERSION = "0.1.0"

PathLike = Union[str, "os.PathLike[str]"]


def _package_version() -> str:
    """Installed version of this package, or ``"0.1.0"`` when not installed."""
    from importlib import metadata

    try:
        return metadata.version("ucci")
    except metadata.PackageNotFoundError:
        pass
    finder = getattr(metadata, "packages_distributions", None)  # Python >= 3.10
    if finder is not None:
        for dist in finder().get("ucci", []):
            try:
                return metadata.version(dist)
            except metadata.PackageNotFoundError:  # pragma: no cover
                continue
    return _FALLBACK_VERSION


def _number(data: Mapping[str, Any], key: str, where: str = "") -> float:
    """A finite real number (bool rejected) from ``data[key]``."""
    label = f"{where}{key}"
    if key not in data:
        raise ValueError(f"router file is missing '{label}'")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"'{label}' must be a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"'{label}' must be finite, got {out!r}")
    return out


def _optional_number(data: Mapping[str, Any], key: str) -> float | None:
    if data.get(key) is None:
        return None
    return _number(data, key)


def _number_list(value: Any, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"'{label}' must be a list of numbers, got {type(value).__name__}")
    out = []
    for i, v in enumerate(value):
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            raise ValueError(f"'{label}[{i}]' must be a number, got {v!r}")
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"'{label}[{i}]' must be finite, got {f!r}")
        out.append(f)
    return out


def validate_router_dict(data: Any) -> dict[str, Any]:
    """Check a decoded router document and return its normalized fields.

    Parameters
    ----------
    data : mapping
        The decoded JSON object.

    Returns
    -------
    dict
        The known fields only, with numbers as floats and optional fields
        filled in: ``tau`` (None), ``grid_step`` (None when absent) and
        ``created_by`` (None).

    Raises
    ------
    ValueError
        With a message naming the first problem found.
    """
    if not isinstance(data, Mapping):
        raise ValueError(f"router file must hold a JSON object, got {type(data).__name__}")
    fmt = data.get("format")
    if fmt != FORMAT_NAME:
        raise ValueError(f"not a UCCI router file: 'format' is {fmt!r}, expected {FORMAT_NAME!r}")
    version = data.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(f"'version' must be an integer, got {version!r}")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported router format version {version}; this ucci reads version "
            f"{FORMAT_VERSION} (a newer file needs a newer ucci)"
        )

    cal = data.get("calibrator")
    if not isinstance(cal, Mapping):
        raise ValueError("'calibrator' must be an object with lists 'x' and 'y'")
    for key in ("x", "y"):
        if key not in cal:
            raise ValueError(f"router file is missing 'calibrator.{key}'")
    x = _number_list(cal["x"], "calibrator.x")
    y = _number_list(cal["y"], "calibrator.y")
    if len(x) != len(y):
        raise ValueError(f"'calibrator.x' has {len(x)} values but 'calibrator.y' has {len(y)}")
    if not x:
        raise ValueError("'calibrator.x' is empty; at least one knot is required")
    for i in range(1, len(x)):
        if not x[i] > x[i - 1]:
            raise ValueError(
                "'calibrator.x' must be strictly increasing; "
                f"x[{i - 1}] = {x[i - 1]!r} >= x[{i}] = {x[i]!r}"
            )
        if y[i] < y[i - 1]:
            raise ValueError(
                f"'calibrator.y' must be non-decreasing; y[{i - 1}] = {y[i - 1]!r} "
                f"> y[{i}] = {y[i]!r}"
            )
    for i, v in enumerate(y):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"'calibrator.y[{i}]' = {v!r} is not a probability in [0, 1]")

    theta = _number(data, "theta")
    c_small = _number(data, "c_small")
    c_large = _number(data, "c_large")
    for label, c in (("c_small", c_small), ("c_large", c_large)):
        if c <= 0.0:
            raise ValueError(f"'{label}' must be positive, got {c!r}")
    cost_model = data.get("cost_model")
    if cost_model not in COST_MODELS:
        raise ValueError(f"'cost_model' must be 'routing' or 'sequential', got {cost_model!r}")
    tau = _optional_number(data, "tau")
    grid_step = _optional_number(data, "grid_step")
    if grid_step is not None:
        if not 0.0 < grid_step <= 1.0:
            raise ValueError(f"'grid_step' must lie in (0, 1], got {grid_step!r}")
        if abs(round(1.0 / grid_step) * grid_step - 1.0) > 1e-9:
            raise ValueError(f"'grid_step' must divide 1 exactly, got {grid_step!r}")
    created_by = data.get("created_by")
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "calibrator": {"x": x, "y": y},
        "theta": theta,
        "c_small": c_small,
        "c_large": c_large,
        "cost_model": str(cost_model),
        "tau": tau,
        "grid_step": grid_step,
        "created_by": created_by if isinstance(created_by, str) else None,
    }


def _reject_constant(name: str) -> float:
    raise ValueError(f"router file contains {name}, which is not valid JSON")


def read_router_dict(path: PathLike) -> dict[str, Any]:
    """Read and validate a router file.

    Parameters
    ----------
    path : str or os.PathLike
        File written by :func:`save_router` (or the Rust crate).

    Returns
    -------
    dict
        The normalized fields, as returned by :func:`validate_router_dict`.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not valid JSON or fails validation; the message
        includes the path.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except ValueError as exc:
        raise ValueError(f"{p}: invalid JSON: {exc}") from exc
    try:
        return validate_router_dict(data)
    except ValueError as exc:
        raise ValueError(f"{p}: {exc}") from exc


def dumps_router_dict(data: Mapping[str, Any]) -> str:
    """Serialize a validated router document: one top-level key per line.

    Lists stay on one line, so a calibrator with hundreds of knots remains a
    short, diff-friendly file. The output is standard JSON (no NaN or
    Infinity) and ends with a newline.
    """
    lines = [
        f"  {json.dumps(key)}: {json.dumps(value, allow_nan=False, separators=(', ', ': '))}"
        for key, value in data.items()
    ]
    return "{\n" + ",\n".join(lines) + "\n}\n"


def _create_temp_file(directory: PathLike, name: str) -> tuple[int, str]:
    """Create ``<directory>/.<name>.<random hex>.tmp`` for an atomic write.

    :func:`tempfile.mkstemp` creates its file with mode 0600, which the
    rename would carry over to the saved file. This goes through
    :func:`os.open` with mode 0666 instead, so the process umask applies as it
    does for a plain ``open(path, "w")`` (and for the Rust crate's writer).

    Returns
    -------
    tuple of (int, str)
        An open file descriptor and the temporary path.
    """
    tmp = os.path.join(os.fspath(directory), f".{name}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    return os.open(tmp, flags, 0o666), tmp


def write_router_dict(data: Mapping[str, Any], path: PathLike) -> Path:
    """Validate a router document and write it to ``path`` atomically.

    The document is written to a temporary file in the same directory and
    then renamed over ``path``, so readers never see a half-written file.

    Returns
    -------
    pathlib.Path
        The path written.

    Raises
    ------
    ValueError
        If ``data`` fails :func:`validate_router_dict`.
    """
    doc = validate_router_dict(data)
    text = dumps_router_dict(doc)
    p = Path(path)
    directory = p.parent if str(p.parent) else Path(".")
    fd, tmp = _create_temp_file(directory, p.name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return p


def save_router(router: UCCIRouter, path: PathLike) -> Path:
    """Write a calibrated router with a chosen threshold to ``path``.

    Raises
    ------
    RuntimeError
        If the router is not calibrated or has no threshold yet.
    """
    return write_router_dict(router.to_dict(), path)


def load_router(path: PathLike) -> UCCIRouter:
    """Read a router written by :func:`save_router` (or the Rust crate).

    Raises
    ------
    ValueError
        If the file is malformed; the message names the path and the problem.
    """
    from .router import UCCIRouter

    return UCCIRouter.from_dict(read_router_dict(path))
