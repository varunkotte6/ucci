"""The Python core reproduces every golden vector in tests/golden.

The golden files are written by ``tools/make_golden.py`` from the Python core
and replayed here against the current core, and in ``rust/tests/golden.rs``
against the Rust crate. A failure here means the core's behaviour changed:
if the change is intended, regenerate with ``python tools/make_golden.py``
(and rerun ``cargo test`` in ``rust/``).

``tests/golden/rust_written_routers.json`` holds routers written by the Rust
crate (``UCCI_BLESS=1 cargo test --test golden``); the Python package must
read each one and route exactly like the Python router it came from.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "tests" / "golden"


def _load_generator() -> Any:
    spec = importlib.util.spec_from_file_location("make_golden", ROOT / "tools" / "make_golden.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mg = _load_generator()


def _read(name: str) -> dict[str, Any]:
    with open(GOLDEN_DIR / name, encoding="utf-8") as fh:
        doc: dict[str, Any] = json.load(fh)
    return doc


def _cases() -> list[Any]:
    params = []
    for name in mg.FILES:
        doc = _read(name)
        for case in doc["cases"]:
            params.append(pytest.param(name, doc, case, id=f"{name[:-5]}::{case['id']}"))
    return params


def _num(v: Any) -> Any:
    if isinstance(v, str) and v in ("nan", "inf", "-inf"):
        return float(v)
    return v


def assert_close(got: Any, want: Any, tol: float, path: str = "") -> None:
    """Recursive comparison: floats within ``tol``, everything else exact."""
    got, want = _num(got), _num(want)
    if isinstance(want, dict):
        assert isinstance(got, dict), f"{path}: expected an object, got {got!r}"
        assert sorted(got) == sorted(want), f"{path}: keys {sorted(got)} != {sorted(want)}"
        for k in want:
            assert_close(got[k], want[k], tol, f"{path}.{k}")
    elif isinstance(want, list):
        assert isinstance(got, list), f"{path}: expected a list, got {got!r}"
        assert len(got) == len(want), f"{path}: length {len(got)} != {len(want)}"
        for i, (g, w) in enumerate(zip(got, want)):
            assert_close(g, w, tol, f"{path}[{i}]")
    elif (
        want is None
        or isinstance(want, (bool, str))
        or (isinstance(want, int) and not isinstance(got, float))
    ):
        # Booleans, strings, null and integer counts must match exactly.
        assert got == want, f"{path}: {got!r} != {want!r}"
    else:
        g, w = float(got), float(want)
        if math.isnan(w):
            assert math.isnan(g), f"{path}: {g!r} != nan"
        elif math.isinf(w):
            assert g == w, f"{path}: {g!r} != {w!r}"
        else:
            assert abs(g - w) <= tol, f"{path}: {g!r} != {w!r} (tolerance {tol})"


@pytest.mark.parametrize("name", list(mg.FILES))
def test_file_header_and_case_ids(name: str) -> None:
    doc = _read(name)
    assert doc["schema"] == mg.SCHEMA
    assert doc["module"] == name[:-5]
    assert doc["tolerance"] == mg.TOLERANCE
    stored = [c["id"] for c in doc["cases"]]
    assert len(stored) == len(set(stored)), "duplicate case ids"
    built = mg.build_cases(name)
    assert stored == [c["id"] for c in built], (
        f"{name} is out of date; run python tools/make_golden.py"
    )
    for case in doc["cases"]:
        assert ("error" in case) == ("error" in case["id"]), case["id"]
        assert ("error" in case) != ("expected" in case), case["id"]


@pytest.mark.parametrize("name,doc,case", _cases())
def test_core_reproduces_golden(name: str, doc: dict[str, Any], case: dict[str, Any]) -> None:
    got = mg.compute(case["fn"], mg.resolve(case["input"], doc["data"]))
    tol = doc["tolerance"]
    if "error" in case:
        assert "error" in got, f"expected {case['error']['type']}, got {got.get('expected')!r}"
        want = dict(case["error"])
        have = dict(got["error"])
        want.pop("message")
        have.pop("message")
        assert_close(have, want, tol, "error")
    else:
        assert "expected" in got, f"core raised {got['error']!r}"
        assert_close(got["expected"], case["expected"], tol, "expected")


def test_golden_files_cover_every_function() -> None:
    fns = {c["fn"] for name in mg.FILES for c in _read(name)["cases"]}
    assert fns == {
        "margins_from_top2",
        "token_margin_uncertainty",
        "uncertainty_from_margins",
        "uncertainty_from_probs",
        "uncertainty_from_logprobs",
        "top2_from_logprobs",
        "uncertainty_from_top_logprobs",
        "from_openai_logprobs",
        "pav",
        "calibrator_fit",
        "calibrator_from_knots",
        "calibrator_predict",
        "default_grid",
        "make_grid",
        "escalate",
        "escalate_scalar",
        "policy_cost",
        "policy_accuracy",
        "select_threshold",
        "select_threshold_for_budget",
        "pareto_frontier",
        "evaluate",
        "ece",
        "reliability_table",
        "brier_score",
        "micro_f1",
        "routed_micro_f1",
        "router_load",
        "router_fit",
    }


def test_golden_generation_is_deterministic() -> None:
    for name in ("signal.json", "policy.json"):
        assert mg.dumps(mg.build_file(name)) == mg.dumps(mg.build_file(name))


RUST_WRITTEN = GOLDEN_DIR / "rust_written_routers.json"


@pytest.mark.skipif(not RUST_WRITTEN.exists(), reason="no routers written by the Rust crate yet")
def test_python_reads_routers_written_by_rust() -> None:
    written = json.loads(RUST_WRITTEN.read_text(encoding="utf-8"))
    router_cases = {c["id"]: c for c in _read("router.json")["cases"]}
    assert written["cases"], "rust_written_routers.json has no cases"
    for entry in written["cases"]:
        source = router_cases[entry["id"]]
        assert "expected" in source, entry["id"]
        got = mg.compute(
            "router_load",
            {"json_text": entry["json_text"], "queries": source["input"]["queries"]},
        )
        assert "expected" in got, f"{entry['id']}: Python rejected the Rust router: {got}"
        have, want = dict(got["expected"]), dict(source["expected"])
        assert have.pop("created_by").startswith("ucci-rust ")
        want.pop("created_by")
        if want["tau"] is None:
            assert have["tau"] is None
        # Floats must survive the Rust writer bit for bit.
        assert_close(have, want, 0.0, entry["id"])
