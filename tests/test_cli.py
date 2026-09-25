"""End-to-end tests for the ``ucci`` command line (ucci.cli).

Every command is exercised in process through ``ucci.cli.main(argv)`` and, for
the main flows, in a subprocess through ``python -m ucci.cli`` and the
``ucci`` console script when it is installed. Every documented error path is
checked for its exit code and message. Numbers printed by the CLI are checked
against the Python API on the same data, so the CLI cannot drift from the
library.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import ucci
from ucci import IsotonicCalibrator, ece, reliability_table
from ucci.cli import (
    EXIT_INFEASIBLE,
    EXIT_INPUT,
    EXIT_OK,
    EXIT_USAGE,
    JSON_SCHEMA_VERSION,
    build_parser,
    main,
)
from ucci.policy import (
    evaluate,
    policy_accuracy,
    policy_cost,
    select_threshold,
    select_threshold_for_budget,
)

SRC_DIR = str(Path(ucci.__file__).resolve().parents[1])

# ---------------------------------------------------------------------------
# Synthetic logged traffic
# ---------------------------------------------------------------------------


def make_records(
    n: int = 2000,
    seed: int = 0,
    *,
    split: bool = True,
    scores: bool = False,
    latency: bool = False,
    split_values: tuple[str, str, str] = ("cal", "val", "test"),
) -> list[dict[str, Any]]:
    """Records in the shared JSONL format.

    P(small wrong | u) = min(1, 2.5 u^2), so raw u is miscalibrated as an
    error probability and the isotonic map has something to correct.
    """
    rng = np.random.default_rng(seed)
    u = rng.beta(2.0, 5.0, n)
    small = (rng.random(n) >= np.minimum(1.0, 2.5 * u**2)).astype(int)
    large = (rng.random(n) < 0.95).astype(int)
    labels = np.array(
        [split_values[0]] * int(0.3 * n)
        + [split_values[1]] * int(0.2 * n)
        + [split_values[2]] * (n - int(0.3 * n) - int(0.2 * n))
    )
    rng.shuffle(labels)
    recs = []
    for i in range(n):
        r: dict[str, Any] = {
            "id": f"q{i:05d}",
            "u": float(u[i]),
            "small_correct": int(small[i]),
            "large_correct": int(large[i]),
        }
        if split:
            r["split"] = str(labels[i])
        if scores:
            r["small_score"] = float(np.clip(small[i] * 0.9 + 0.1 * rng.random(), 0, 1))
            r["large_score"] = float(np.clip(large[i] * 0.9 + 0.1 * rng.random(), 0, 1))
        if latency:
            r["latency_small_ms"] = float(40.0 + 10.0 * rng.random())
            r["latency_large_ms"] = float(130.0 + 20.0 * rng.random())
        recs.append(r)
    return recs


def write_jsonl(path: Path, recs: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return path


def write_csv(path: Path, recs: list[dict[str, Any]]) -> Path:
    fields: list[str] = []
    for r in recs:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in recs:
            w.writerow({k: repr(v) if isinstance(v, float) else v for k, v in r.items()})
    return path


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = main(argv)
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def run_json(argv: list[str], capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    code, out, err = run([*argv, "--json"], capsys)
    assert code == EXIT_OK, err
    doc: dict[str, Any] = json.loads(out)
    return doc


def columns(recs: list[dict[str, Any]], rows: list[int] | None = None) -> dict[str, np.ndarray]:
    sel = recs if rows is None else [recs[i] for i in rows]
    return {
        k: np.array([r[k] for r in sel], dtype=float)
        for k in ("u", "small_correct", "large_correct")
    }


def rows_of(recs: list[dict[str, Any]], split: str) -> list[int]:
    return [i for i, r in enumerate(recs) if r["split"] == split]


def reference_random_split(
    ids: list[str], cal_frac: float, val_frac: float, seed: int
) -> dict[str, list[int]]:
    """The documented random split, written out independently of ucci.cli."""
    keys = [hashlib.sha256(f"{seed}:{i}".encode()).hexdigest() for i in ids]
    order = sorted(range(len(ids)), key=lambda i: (keys[i], i))
    n_cal = math.floor(cal_frac * len(ids) + 0.5)
    n_val = math.floor(val_frac * len(ids) + 0.5)
    return {
        "cal": sorted(order[:n_cal]),
        "val": sorted(order[n_cal : n_cal + n_val]),
        "test": sorted(order[n_cal + n_val :]),
    }


@pytest.fixture
def recs() -> list[dict[str, Any]]:
    return make_records()


@pytest.fixture
def data(tmp_path: Path, recs: list[dict[str, Any]]) -> Path:
    return write_jsonl(tmp_path / "traffic.jsonl", recs)


@pytest.fixture
def router(tmp_path: Path, data: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    out = tmp_path / "router.json"
    code, _, err = run(["fit", "--data", str(data), "--tau", "0.9", "--out", str(out)], capsys)
    assert code == EXIT_OK, err
    return out


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


class TestFit:
    def test_router_file_follows_the_shared_contract(self, router: Path) -> None:
        doc = json.loads(router.read_text())
        assert doc["format"] == "ucci-router" and doc["version"] == 1
        x, y = np.array(doc["calibrator"]["x"]), np.array(doc["calibrator"]["y"])
        assert x.size == y.size >= 1
        assert np.all(np.diff(x) > 0) and np.all(np.diff(y) >= 0)
        assert y.min() >= 0 and y.max() <= 1
        assert doc["c_small"] == 1.0 and doc["c_large"] == 3.02
        assert doc["cost_model"] == "routing" and doc["tau"] == 0.9
        assert doc["grid_step"] == 0.005
        assert doc["created_by"] == f"ucci-python {ucci.__version__}"
        assert 0.0 <= doc["theta"] <= 1.0
        assert abs(doc["theta"] / 0.005 - round(doc["theta"] / 0.005)) < 1e-9
        # Provenance extension: readers ignore it, evaluate uses it.
        assert doc["fit"]["split"]["method"] == "field"
        assert doc["fit"]["data"]["n_records"] == 2000
        # The core reader accepts the file.
        loaded = ucci.io.read_router_dict(router)
        assert loaded["theta"] == doc["theta"]

    def test_matches_python_api(self, router: Path, recs: list[dict[str, Any]]) -> None:
        cal_c = columns(recs, rows_of(recs, "cal"))
        val_c = columns(recs, rows_of(recs, "val"))
        cal = IsotonicCalibrator().fit(cal_c["u"], 1 - cal_c["small_correct"])
        choice = select_threshold(
            cal.predict(val_c["u"]),
            val_c["small_correct"],
            val_c["large_correct"],
            tau=0.9,
            c_small=1.0,
            c_large=3.02,
        )
        doc = json.loads(router.read_text())
        assert doc["calibrator"] == cal.to_dict()
        assert doc["theta"] == choice.theta
        v = doc["fit"]["validation"]
        assert v["cost"] == pytest.approx(choice.cost)
        assert v["accuracy"] == pytest.approx(choice.accuracy)
        assert v["escalation_rate"] == pytest.approx(choice.escalation_rate)

    def test_text_summary(
        self, data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = run(
            ["fit", "--data", str(data), "--tau", "0.9", "--out", str(tmp_path / "r.json")], capsys
        )
        assert code == EXIT_OK and err == ""
        for needle in (
            "wrote",
            "theta* =",
            "cal 600, val 400, test 1000",
            "ECE raw u",
            "on the validation split (held out",
            "savings vs always-large",
            "always-large accuracy",
        ):
            assert needle in out, needle

    def test_json_summary_schema(
        self, data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "r.json"
        doc = run_json(["fit", "--data", str(data), "--tau", "0.9", "--out", str(out)], capsys)
        assert set(doc) == {
            "command",
            "schema_version",
            "ucci_version",
            "router_path",
            "data",
            "split",
            "objective",
            "metric",
            "costs",
            "grid_step",
            "theta",
            "validation",
            "calibration",
        }
        assert doc["command"] == "fit" and doc["schema_version"] == JSON_SCHEMA_VERSION
        assert doc["ucci_version"] == ucci.__version__
        assert set(doc["data"]) == {"source", "format", "n_records", "sha256", "ids_sha256"}
        assert set(doc["split"]) == {"method", "field", "cal_frac", "val_frac", "seed", "sizes"}
        assert doc["split"]["sizes"] == {"cal": 600, "val": 400, "test": 1000}
        assert doc["objective"] == {"type": "accuracy_target", "tau": 0.9, "budget": None}
        assert set(doc["validation"]) == {
            "n",
            "cost",
            "accuracy",
            "escalation_rate",
            "savings_vs_large",
            "always_small",
            "always_large",
        }
        assert set(doc["calibration"]) == {
            "n",
            "n_knots",
            "bins",
            "strategy",
            "ece_raw_cal",
            "ece_calibrated_cal",
            "ece_raw_val",
            "ece_calibrated_val",
        }
        assert doc["validation"]["accuracy"] >= 0.9
        assert doc["validation"]["savings_vs_large"] == pytest.approx(
            1 - doc["validation"]["cost"] / 3.02
        )
        # Isotonic regression matches bin frequencies on its own fit data.
        assert doc["calibration"]["ece_calibrated_cal"] == pytest.approx(0.0, abs=1e-12)
        assert doc["calibration"]["ece_calibrated_val"] < doc["calibration"]["ece_raw_val"]
        assert json.loads(out.read_text())["theta"] == doc["theta"]

    def test_budget_form(
        self,
        data: Path,
        recs: list[dict[str, Any]],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        out = tmp_path / "r.json"
        doc = run_json(["fit", "--data", str(data), "--budget", "2.0", "--out", str(out)], capsys)
        cal_c = columns(recs, rows_of(recs, "cal"))
        val_c = columns(recs, rows_of(recs, "val"))
        cal = IsotonicCalibrator().fit(cal_c["u"], 1 - cal_c["small_correct"])
        ref = select_threshold_for_budget(
            cal.predict(val_c["u"]), val_c["small_correct"], val_c["large_correct"], budget=2.0
        )
        assert doc["theta"] == ref.theta
        assert doc["validation"]["cost"] <= 2.0
        assert doc["objective"] == {"type": "cost_budget", "tau": None, "budget": 2.0}
        assert json.loads(out.read_text())["tau"] is None

    def test_default_random_split_is_30_20_50_and_documented(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        recs = make_records(1001, seed=3, split=False)
        data = write_jsonl(tmp_path / "d.jsonl", recs)
        doc = run_json(
            ["fit", "--data", str(data), "--tau", "0.85", "--out", str(tmp_path / "r.json")], capsys
        )
        assert doc["split"]["method"] == "random"
        assert (doc["split"]["cal_frac"], doc["split"]["val_frac"], doc["split"]["seed"]) == (
            0.3,
            0.2,
            0,
        )
        assert doc["split"]["sizes"] == {"cal": 300, "val": 200, "test": 501}
        # Recompute theta from the documented split rule.
        ref = reference_random_split([r["id"] for r in recs], 0.3, 0.2, 0)
        cal_c, val_c = columns(recs, ref["cal"]), columns(recs, ref["val"])
        cal = IsotonicCalibrator().fit(cal_c["u"], 1 - cal_c["small_correct"])
        choice = select_threshold(
            cal.predict(val_c["u"]), val_c["small_correct"], val_c["large_correct"], tau=0.85
        )
        assert doc["theta"] == choice.theta

    def test_random_split_seed_and_fractions(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data = write_jsonl(tmp_path / "d.jsonl", make_records(1000, split=False))
        a = run_json(
            [
                "fit",
                "--data",
                str(data),
                "--tau",
                "0.85",
                "--seed",
                "0",
                "--out",
                str(tmp_path / "a.json"),
            ],
            capsys,
        )
        b = run_json(
            [
                "fit",
                "--data",
                str(data),
                "--tau",
                "0.85",
                "--seed",
                "0",
                "--out",
                str(tmp_path / "b.json"),
            ],
            capsys,
        )
        c = run_json(
            [
                "fit",
                "--data",
                str(data),
                "--tau",
                "0.85",
                "--seed",
                "7",
                "--cal-frac",
                "0.5",
                "--val-frac",
                "0.25",
                "--out",
                str(tmp_path / "c.json"),
            ],
            capsys,
        )
        assert a["calibration"] == b["calibration"] and a["theta"] == b["theta"]
        assert c["split"]["sizes"] == {"cal": 500, "val": 250, "test": 250}
        assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()

    def test_random_split_ignores_record_order(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        recs = make_records(800, split=False)
        a = write_jsonl(tmp_path / "a.jsonl", recs)
        b = write_jsonl(tmp_path / "b.jsonl", list(reversed(recs)))
        ra = run_json(
            ["fit", "--data", str(a), "--tau", "0.85", "--out", str(tmp_path / "ra.json")], capsys
        )
        rb = run_json(
            ["fit", "--data", str(b), "--tau", "0.85", "--out", str(tmp_path / "rb.json")], capsys
        )
        assert ra["theta"] == rb["theta"]
        for key, value in ra["calibration"].items():
            assert rb["calibration"][key] == pytest.approx(value, abs=1e-12), key
        assert ra["data"]["ids_sha256"] == rb["data"]["ids_sha256"]

    def test_split_field_is_used_automatically_and_can_be_overridden(
        self, data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        auto = run_json(
            ["fit", "--data", str(data), "--tau", "0.9", "--out", str(tmp_path / "a.json")], capsys
        )
        assert auto["split"]["method"] == "field" and auto["split"]["field"] == "split"
        rnd = run_json(
            [
                "fit",
                "--data",
                str(data),
                "--tau",
                "0.9",
                "--cal-frac",
                "0.3",
                "--out",
                str(tmp_path / "b.json"),
            ],
            capsys,
        )
        assert rnd["split"]["method"] == "random"

    def test_custom_split_field_and_aliases(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        recs = make_records(1000, split_values=("calibration", "Validation", "TEST"))
        for r in recs:
            r["fold"] = r.pop("split")
        data = write_jsonl(tmp_path / "d.jsonl", recs)
        doc = run_json(
            [
                "fit",
                "--data",
                str(data),
                "--tau",
                "0.85",
                "--split-field",
                "fold",
                "--out",
                str(tmp_path / "r.json"),
            ],
            capsys,
        )
        assert doc["split"]["field"] == "fold"
        assert doc["split"]["sizes"] == {"cal": 300, "val": 200, "test": 500}

    def test_csv_and_json_array_match_jsonl(
        self,
        recs: list[dict[str, Any]],
        data: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        csv_path = write_csv(tmp_path / "d.csv", recs)
        arr_path = tmp_path / "d.json"
        arr_path.write_text(json.dumps(recs))
        docs = {}
        for name, path in (("jsonl", data), ("csv", csv_path), ("json", arr_path)):
            out = tmp_path / f"{name}.json"
            run_json(["fit", "--data", str(path), "--tau", "0.9", "--out", str(out)], capsys)
            docs[name] = json.loads(out.read_text())
            assert docs[name]["fit"]["data"]["format"] == name
        for name in ("csv", "json"):
            for key in ("calibrator", "theta", "c_small", "c_large", "tau"):
                assert docs[name][key] == docs["jsonl"][key], (name, key)

    def test_format_override_and_sniffing(
        self, recs: list[dict[str, Any]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        odd = write_jsonl(tmp_path / "traffic.log", recs)
        assert (
            run_json(
                ["fit", "--data", str(odd), "--tau", "0.9", "--out", str(tmp_path / "a.json")],
                capsys,
            )["data"]["format"]
            == "jsonl"
        )
        odd_csv = write_csv(tmp_path / "traffic.txt", recs)
        assert (
            run_json(
                ["fit", "--data", str(odd_csv), "--tau", "0.9", "--out", str(tmp_path / "b.json")],
                capsys,
            )["data"]["format"]
            == "csv"
        )
        forced = write_jsonl(tmp_path / "x.csv", recs)
        assert (
            run_json(
                [
                    "fit",
                    "--data",
                    str(forced),
                    "--format",
                    "jsonl",
                    "--tau",
                    "0.9",
                    "--out",
                    str(tmp_path / "c.json"),
                ],
                capsys,
            )["data"]["format"]
            == "jsonl"
        )

    def test_reads_stdin(
        self,
        recs: list[dict[str, Any]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        payload = "".join(json.dumps(r) + "\n" for r in recs).encode()
        monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(payload)))
        doc = run_json(
            ["fit", "--data", "-", "--tau", "0.9", "--out", str(tmp_path / "r.json")], capsys
        )
        assert doc["data"]["source"] == "<stdin>" and doc["data"]["n_records"] == len(recs)

    def test_cost_from_latency(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        recs = make_records(1000, latency=True)
        data = write_jsonl(tmp_path / "d.jsonl", recs)
        doc = run_json(
            [
                "fit",
                "--data",
                str(data),
                "--tau",
                "0.85",
                "--cost-from-latency",
                "--out",
                str(tmp_path / "r.json"),
            ],
            capsys,
        )
        fit_rows = [r for r in recs if r["split"] in ("cal", "val")]
        ratio = np.mean([r["latency_large_ms"] for r in fit_rows]) / np.mean(
            [r["latency_small_ms"] for r in fit_rows]
        )
        assert doc["costs"] == {
            "c_small": 1.0,
            "c_large": pytest.approx(ratio),
            "cost_model": "routing",
            "source": "latency",
        }

    def test_sequential_cost_model_and_custom_costs(
        self, data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        doc = run_json(
            [
                "fit",
                "--data",
                str(data),
                "--tau",
                "0.9",
                "--cost-model",
                "sequential",
                "--c-small",
                "2",
                "--c-large",
                "5",
                "--out",
                str(tmp_path / "r.json"),
            ],
            capsys,
        )
        v = doc["validation"]
        assert v["cost"] == pytest.approx(2 + 5 * v["escalation_rate"])
        routed = run_json(
            ["fit", "--data", str(data), "--tau", "0.9", "--out", str(tmp_path / "q.json")], capsys
        )
        # Theta does not depend on the cost scale (constant marginal cost).
        assert doc["theta"] == routed["theta"]

    def test_grid_step(
        self, data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        doc = run_json(
            [
                "fit",
                "--data",
                str(data),
                "--tau",
                "0.9",
                "--grid-step",
                "0.05",
                "--out",
                str(tmp_path / "r.json"),
            ],
            capsys,
        )
        assert abs(doc["theta"] / 0.05 - round(doc["theta"] / 0.05)) < 1e-9
        assert json.loads((tmp_path / "r.json").read_text())["grid_step"] == 0.05

    def test_score_metric(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        recs = make_records(1000, scores=True)
        data = write_jsonl(tmp_path / "d.jsonl", recs)
        doc = run_json(
            ["fit", "--data", str(data), "--tau", "0.85", "--out", str(tmp_path / "r.json")], capsys
        )
        assert doc["metric"] == "score"
        forced = run_json(
            [
                "fit",
                "--data",
                str(data),
                "--tau",
                "0.85",
                "--metric",
                "correct",
                "--out",
                str(tmp_path / "c.json"),
            ],
            capsys,
        )
        assert forced["metric"] == "correct"
        val = [r for r in recs if r["split"] == "val"]
        assert doc["validation"]["always_small"]["accuracy"] == pytest.approx(
            np.mean([r["small_score"] for r in val])
        )


# ---------------------------------------------------------------------------
# route
# ---------------------------------------------------------------------------


class TestRoute:
    def test_values(self, router: Path, capsys: pytest.CaptureFixture[str]) -> None:
        doc = json.loads(router.read_text())
        cal = IsotonicCalibrator.from_dict(doc["calibrator"])
        us = [0.0, 0.05, 0.2, 0.35, 0.5, 0.9, 1.0]
        code, out, _ = run(["route", "--router", str(router), "--u", *map(str, us)], capsys)
        assert code == EXIT_OK
        lines = out.strip().splitlines()
        assert lines[0] == "id\tu\tp_hat\tdecision"
        for line, u in zip(lines[1:], us):
            _, _, p_txt, decision = line.split("\t")
            p = float(cal.predict(u))
            assert float(p_txt) == pytest.approx(p, abs=1e-6)
            assert decision == ("large" if p > doc["theta"] else "small")

    def test_json(self, router: Path, capsys: pytest.CaptureFixture[str]) -> None:
        doc = run_json(["route", "--router", str(router), "--u", "0.1", "0.8"], capsys)
        assert set(doc) == {
            "command",
            "schema_version",
            "ucci_version",
            "theta",
            "n",
            "n_escalated",
            "queries",
        }
        assert doc["n"] == 2
        for q in doc["queries"]:
            assert set(q) == {"id", "u", "p_hat", "escalate", "decision"}
            assert q["escalate"] == (q["p_hat"] > doc["theta"])
            assert q["decision"] == ("large" if q["escalate"] else "small")
        assert doc["n_escalated"] == sum(q["escalate"] for q in doc["queries"])

    def test_data_file_keeps_ids(
        self,
        router: Path,
        recs: list[dict[str, Any]],
        data: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        doc = run_json(["route", "--router", str(router), "--data", str(data)], capsys)
        assert [q["id"] for q in doc["queries"]] == [r["id"] for r in recs]
        cal = IsotonicCalibrator.from_dict(json.loads(router.read_text())["calibrator"])
        p = cal.predict(np.array([r["u"] for r in recs]))
        assert np.allclose([q["p_hat"] for q in doc["queries"]], p, rtol=0, atol=0)

    def test_data_needs_only_u(
        self, router: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data = tmp_path / "u.csv"
        data.write_text("u\n0.1\n0.7\n")
        doc = run_json(["route", "--router", str(router), "--data", str(data)], capsys)
        assert [q["id"] for q in doc["queries"]] == ["0", "1"]

    def test_bad_u(self, router: Path, capsys: pytest.CaptureFixture[str]) -> None:
        for bad in ("1.5", "nan", "-0.1"):
            code, _, err = run(["route", "--router", str(router), "--u", bad], capsys)
            assert code == EXIT_USAGE and "[0, 1]" in err


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_matches_python_api_on_test_split(
        self,
        router: Path,
        data: Path,
        recs: list[dict[str, Any]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        doc = run_json(
            [
                "evaluate",
                "--router",
                str(router),
                "--data",
                str(data),
                "--split",
                "test",
                "--bootstrap",
                "300",
            ],
            capsys,
        )
        rdoc = json.loads(router.read_text())
        cal = IsotonicCalibrator.from_dict(rdoc["calibrator"])
        t = columns(recs, rows_of(recs, "test"))
        ref = evaluate(
            cal.predict(t["u"]),
            t["small_correct"],
            t["large_correct"],
            rdoc["theta"],
            c_small=1.0,
            c_large=3.02,
        )
        assert doc["n"] == 1000 and doc["split"] == "test"
        assert doc["ucci"]["cost"] == pytest.approx(ref.cost)
        assert doc["ucci"]["accuracy"] == pytest.approx(ref.accuracy)
        assert doc["ucci"]["escalation_rate"] == pytest.approx(ref.escalation_rate)
        assert doc["ucci"]["savings_vs_large"] == pytest.approx(1 - ref.cost / 3.02)
        assert doc["ucci"]["accuracy_minus_tau"] == pytest.approx(ref.accuracy - 0.9)
        assert doc["always_small"] == {
            "cost": 1.0,
            "accuracy": pytest.approx(t["small_correct"].mean()),
        }
        assert doc["always_large"] == {
            "cost": 3.02,
            "accuracy": pytest.approx(t["large_correct"].mean()),
        }
        esc = cal.predict(t["u"]) > rdoc["theta"]
        a2 = doc["assumption_ii"]
        assert a2["large_accuracy_escalated"] == pytest.approx(t["large_correct"][esc].mean())
        assert a2["gap"] == pytest.approx(a2["large_accuracy_all"] - a2["large_accuracy_escalated"])
        ci = doc["bootstrap"]["ci"]
        for key in ("cost", "accuracy", "escalation_rate", "savings_vs_large"):
            lo, hi = ci[key]
            assert lo <= doc["ucci"][key] <= hi, key
        assert ci["savings_vs_large"][0] == pytest.approx(1 - ci["cost"][1] / 3.02)

    def test_bootstrap_matches_core_percentile_bootstrap(
        self,
        router: Path,
        data: Path,
        recs: list[dict[str, Any]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        doc = run_json(
            [
                "evaluate",
                "--router",
                str(router),
                "--data",
                str(data),
                "--split",
                "test",
                "--bootstrap",
                "200",
                "--seed",
                "5",
                "--level",
                "0.9",
            ],
            capsys,
        )
        rdoc = json.loads(router.read_text())
        cal = IsotonicCalibrator.from_dict(rdoc["calibrator"])
        t = columns(recs, rows_of(recs, "test"))
        esc = cal.predict(t["u"]) > rdoc["theta"]
        lo, hi = ucci.bootstrap_ci(
            lambda i: policy_accuracy(esc[i], t["small_correct"][i], t["large_correct"][i]),
            esc.size,
            n_boot=200,
            alpha=0.1,
            seed=5,
        )
        assert doc["bootstrap"] == {
            "n_boot": 200,
            "seed": 5,
            "level": 0.9,
            "ci": doc["bootstrap"]["ci"],
        }
        assert doc["bootstrap"]["ci"]["accuracy"] == [pytest.approx(lo), pytest.approx(hi)]
        lo_c, hi_c = ucci.bootstrap_ci(
            lambda i: policy_cost(esc[i], 1.0, 3.02), esc.size, n_boot=200, alpha=0.1, seed=5
        )
        assert doc["bootstrap"]["ci"]["cost"] == [pytest.approx(lo_c), pytest.approx(hi_c)]

    def test_json_schema_and_no_bootstrap(
        self, router: Path, data: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        doc = run_json(
            [
                "evaluate",
                "--router",
                str(router),
                "--data",
                str(data),
                "--split",
                "test",
                "--bootstrap",
                "0",
            ],
            capsys,
        )
        assert set(doc) == {
            "command",
            "schema_version",
            "ucci_version",
            "data",
            "split",
            "n",
            "metric",
            "router",
            "ucci",
            "always_small",
            "always_large",
            "assumption_ii",
            "bootstrap",
        }
        assert doc["command"] == "evaluate" and doc["bootstrap"] is None
        assert set(doc["router"]) == {"theta", "tau", "c_small", "c_large", "cost_model"}
        assert set(doc["ucci"]) == {
            "cost",
            "accuracy",
            "escalation_rate",
            "savings_vs_large",
            "accuracy_minus_tau",
        }
        assert set(doc["assumption_ii"]) == {
            "large_accuracy_escalated",
            "large_accuracy_all",
            "gap",
        }

    def test_text_output(
        self, router: Path, data: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = run(
            [
                "evaluate",
                "--router",
                str(router),
                "--data",
                str(data),
                "--split",
                "test",
                "--bootstrap",
                "100",
            ],
            capsys,
        )
        assert code == EXIT_OK and err == ""
        for needle in (
            "1000 queries (split 'test')",
            "95% CI",
            "cost",
            "savings vs large",
            "accuracy - tau",
            "assumption (ii)",
        ):
            assert needle in out, needle

    def test_cost_ratio_sensitivity_keeps_routing(
        self, router: Path, data: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Table 3: other cost ratios change the reported cost, not the routing."""
        base = run_json(
            [
                "evaluate",
                "--router",
                str(router),
                "--data",
                str(data),
                "--split",
                "test",
                "--bootstrap",
                "0",
            ],
            capsys,
        )
        for c_large in (5.0, 10.0):
            doc = run_json(
                [
                    "evaluate",
                    "--router",
                    str(router),
                    "--data",
                    str(data),
                    "--split",
                    "test",
                    "--bootstrap",
                    "0",
                    "--c-large",
                    str(c_large),
                ],
                capsys,
            )
            rate = doc["ucci"]["escalation_rate"]
            assert rate == base["ucci"]["escalation_rate"]
            assert doc["ucci"]["accuracy"] == base["ucci"]["accuracy"]
            assert doc["ucci"]["cost"] == pytest.approx(1 + (c_large - 1) * rate)
            assert doc["ucci"]["savings_vs_large"] == pytest.approx(
                1 - doc["ucci"]["cost"] / c_large
            )

    def test_random_split_is_rederived(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        recs = make_records(1500, seed=9, split=False)
        data = write_jsonl(tmp_path / "d.jsonl", recs)
        rpath = tmp_path / "r.json"
        run_json(
            ["fit", "--data", str(data), "--tau", "0.85", "--seed", "4", "--out", str(rpath)],
            capsys,
        )
        doc = run_json(
            [
                "evaluate",
                "--router",
                str(rpath),
                "--data",
                str(data),
                "--split",
                "test",
                "--bootstrap",
                "0",
            ],
            capsys,
        )
        ref_rows = reference_random_split([r["id"] for r in recs], 0.3, 0.2, 4)["test"]
        t = columns(recs, ref_rows)
        rdoc = json.loads(rpath.read_text())
        cal = IsotonicCalibrator.from_dict(rdoc["calibrator"])
        ref = evaluate(cal.predict(t["u"]), t["small_correct"], t["large_correct"], rdoc["theta"])
        assert doc["n"] == len(ref_rows) == 750
        assert doc["ucci"]["cost"] == pytest.approx(ref.cost)
        assert doc["ucci"]["accuracy"] == pytest.approx(ref.accuracy)
        # Shuffled copy of the same records: same test split.
        shuffled = write_jsonl(tmp_path / "s.jsonl", list(reversed(recs)))
        doc2 = run_json(
            [
                "evaluate",
                "--router",
                str(rpath),
                "--data",
                str(shuffled),
                "--split",
                "test",
                "--bootstrap",
                "0",
            ],
            capsys,
        )
        assert doc2["ucci"] == doc["ucci"]

    def test_random_split_refuses_other_records(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data = write_jsonl(tmp_path / "d.jsonl", make_records(1000, split=False))
        rpath = tmp_path / "r.json"
        run_json(["fit", "--data", str(data), "--tau", "0.85", "--out", str(rpath)], capsys)
        other = make_records(1000, seed=1, split=False)
        for r in other:
            r["id"] = "x" + r["id"]
        other_path = write_jsonl(tmp_path / "o.jsonl", other)
        code, _, err = run(
            ["evaluate", "--router", str(rpath), "--data", str(other_path), "--split", "test"],
            capsys,
        )
        assert code == EXIT_INPUT and "ids differ" in err
        fewer = write_jsonl(tmp_path / "f.jsonl", make_records(999, split=False))
        code, _, err = run(
            ["evaluate", "--router", str(rpath), "--data", str(fewer), "--split", "test"], capsys
        )
        assert code == EXIT_INPUT and "999 records" in err

    def test_all_records_and_in_sample_note(
        self, router: Path, data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = run(
            ["evaluate", "--router", str(router), "--data", str(data), "--bootstrap", "0"], capsys
        )
        assert code == EXIT_OK and "2000 queries (all records)" in out
        assert "--split test" in err
        fresh = write_jsonl(tmp_path / "fresh.jsonl", make_records(500, seed=42, split=False))
        code, out, err = run(
            ["evaluate", "--router", str(router), "--data", str(fresh), "--bootstrap", "0"], capsys
        )
        assert code == EXIT_OK and err == ""

    def test_split_field_option(
        self,
        router: Path,
        tmp_path: Path,
        recs: list[dict[str, Any]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        for r in recs:
            r["part"] = r.pop("split")
        data = write_jsonl(tmp_path / "p.jsonl", recs)
        doc = run_json(
            [
                "evaluate",
                "--router",
                str(router),
                "--data",
                str(data),
                "--split",
                "val",
                "--split-field",
                "part",
                "--bootstrap",
                "0",
            ],
            capsys,
        )
        assert doc["n"] == 400 and doc["split"] == "val"


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


class TestReport:
    @pytest.mark.parametrize("strategy, bins", [("uniform", 10), ("quantile", 10), ("uniform", 5)])
    def test_matches_core_metrics(
        self,
        router: Path,
        data: Path,
        recs: list[dict[str, Any]],
        strategy: str,
        bins: int,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        doc = run_json(
            [
                "report",
                "--router",
                str(router),
                "--data",
                str(data),
                "--split",
                "test",
                "--strategy",
                strategy,
                "--bins",
                str(bins),
                "--bootstrap",
                "0",
            ],
            capsys,
        )
        cal = IsotonicCalibrator.from_dict(json.loads(router.read_text())["calibrator"])
        t = columns(recs, rows_of(recs, "test"))
        e = 1 - t["small_correct"]
        p = cal.predict(t["u"])
        assert doc["raw"]["ece"] == pytest.approx(ece(t["u"], e, n_bins=bins, strategy=strategy))
        assert doc["calibrated"]["ece"] == pytest.approx(ece(p, e, n_bins=bins, strategy=strategy))
        for key, forecast in (("raw", t["u"]), ("calibrated", p)):
            ref = reliability_table(forecast, e, n_bins=bins, strategy=strategy)
            got = doc[key]["reliability"]
            assert len(got) == len(ref)
            for row, r in zip(got, ref):
                assert row == {
                    "bin_lower": pytest.approx(r.bin_lower),
                    "bin_upper": pytest.approx(r.bin_upper),
                    "count": r.count,
                    "mean_forecast": pytest.approx(r.mean_forecast),
                    "observed_frequency": pytest.approx(r.observed_frequency),
                }
            assert sum(row["count"] for row in got) == 1000
        assert doc["calibrated"]["ece"] < doc["raw"]["ece"]
        assert doc["bins"] == bins and doc["strategy"] == strategy
        assert doc["bootstrap"] is None and doc["raw"]["ece_ci"] is None

    def test_json_schema_with_bootstrap(
        self, router: Path, data: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        doc = run_json(
            [
                "report",
                "--router",
                str(router),
                "--data",
                str(data),
                "--split",
                "test",
                "--bootstrap",
                "100",
                "--seed",
                "1",
            ],
            capsys,
        )
        assert set(doc) == {
            "command",
            "schema_version",
            "ucci_version",
            "data",
            "split",
            "n",
            "event",
            "bins",
            "strategy",
            "raw",
            "calibrated",
            "bootstrap",
        }
        assert doc["bootstrap"] == {"n_boot": 100, "seed": 1, "level": 0.95}
        for key in ("raw", "calibrated"):
            assert set(doc[key]) == {"ece", "ece_ci", "reliability"}
            lo, hi = doc[key]["ece_ci"]
            assert 0 <= lo <= hi

    def test_text_output(
        self, router: Path, data: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, out, err = run(
            ["report", "--router", str(router), "--data", str(data), "--bootstrap", "50"], capsys
        )
        assert code == EXIT_OK and err == ""
        for needle in (
            "2000 queries (all records)",
            "ECE raw u",
            "ECE calibrated p_hat",
            "95% CI",
            "reliability: raw u",
            "reliability: calibrated p_hat",
            "observed error",
        ):
            assert needle in out, needle


# ---------------------------------------------------------------------------
# version and parser
# ---------------------------------------------------------------------------


class TestVersion:
    def test_text_and_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert run(["version"], capsys)[1] == f"ucci {ucci.__version__}\n"
        code, out, _ = run(["--version"], capsys)
        assert code == EXIT_OK and out.strip() == f"ucci {ucci.__version__}"

    def test_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        doc = run_json(["version"], capsys)
        assert doc["command"] == "version" and doc["ucci_version"] == ucci.__version__
        assert doc["router_format_version"] == 1 and doc["numpy"] == np.__version__

    def test_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, out, _ = run(["--help"], capsys)
        assert code == EXIT_OK and "Exit codes" in out
        for cmd in ("fit", "route", "evaluate", "report", "version"):
            code, out, _ = run([cmd, "--help"], capsys)
            assert code == EXIT_OK and f"ucci {cmd}" in out

    def test_parser_is_buildable(self) -> None:
        assert build_parser().prog == "ucci"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def _fit(data: Path | str, tmp_path: Path, *extra: str) -> list[str]:
    return ["fit", "--data", str(data), "--out", str(tmp_path / "r.json"), *extra]


class TestUsageErrors:
    @pytest.mark.parametrize(
        "extra, needle",
        [
            ([], "one of the arguments --tau --budget is required"),
            (["--tau", "0.9", "--budget", "2"], "not allowed with"),
            (["--tau", "1.5"], "--tau must lie in [0, 1]"),
            (["--tau", "nan"], "--tau must lie in [0, 1]"),
            (["--budget", "-1"], "--budget must be a positive number"),
            (
                ["--tau", "0.9", "--split-field", "split", "--cal-frac", "0.3"],
                "--split-field cannot be combined",
            ),
            (["--tau", "0.9", "--cal-frac", "0.8", "--val-frac", "0.3"], "at most 1"),
            (["--tau", "0.9", "--cal-frac", "0"], "--cal-frac must lie in (0, 1)"),
            (["--tau", "0.9", "--c-small", "3", "--c-large", "2"], "c_large > c_small"),
            (["--tau", "0.9", "--c-large", "-3"], "--c-large must be a positive number"),
            (["--tau", "0.9", "--cost-from-latency", "--c-large", "3"], "cannot be combined"),
            (["--tau", "0.9", "--grid-step", "0.3"], "--grid-step"),
            (["--tau", "0.9", "--grid-step", "0"], "--grid-step"),
            (["--tau", "0.9", "--metric", "f1"], "invalid choice"),
        ],
    )
    def test_fit(
        self,
        data: Path,
        tmp_path: Path,
        extra: list[str],
        needle: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        code, _, err = run(_fit(data, tmp_path, *extra), capsys)
        assert code == EXIT_USAGE, err
        assert needle in err
        assert not (tmp_path / "r.json").exists()

    def test_no_command_and_unknown_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert run([], capsys)[0] == EXIT_USAGE
        code, _, err = run(["train"], capsys)
        assert code == EXIT_USAGE and "invalid choice" in err

    @pytest.mark.parametrize(
        "argv, needle",
        [
            (["--split", "holdout"], "--split must be cal, val or test"),
            (["--split-field", "split"], "--split-field needs --split"),
            (["--bootstrap", "-1"], "--bootstrap must be >= 0"),
            (["--level", "1.5"], "--level must lie in (0, 1)"),
            (["--c-large", "0.5"], "c_large > c_small"),
        ],
    )
    def test_evaluate(
        self,
        router: Path,
        data: Path,
        argv: list[str],
        needle: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        code, _, err = run(
            ["evaluate", "--router", str(router), "--data", str(data), *argv], capsys
        )
        assert code == EXIT_USAGE and needle in err

    def test_report(self, router: Path, data: Path, capsys: pytest.CaptureFixture[str]) -> None:
        code, _, err = run(
            ["report", "--router", str(router), "--data", str(data), "--bins", "0"], capsys
        )
        assert code == EXIT_USAGE and "--bins must be >= 1" in err
        code, _, err = run(
            ["report", "--router", str(router), "--data", str(data), "--strategy", "kmeans"], capsys
        )
        assert code == EXIT_USAGE

    def test_route_needs_exactly_one_source(
        self, router: Path, data: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert run(["route", "--router", str(router)], capsys)[0] == EXIT_USAGE
        assert (
            run(["route", "--router", str(router), "--u", "0.1", "--data", str(data)], capsys)[0]
            == EXIT_USAGE
        )


class TestInputErrors:
    def _expect(self, argv: list[str], needle: str, capsys: pytest.CaptureFixture[str]) -> str:
        code, _, err = run(argv, capsys)
        assert code == EXIT_INPUT, err
        assert needle in err, err
        assert err.startswith(f"ucci {argv[0]}: error: ")
        return err

    def test_missing_and_unreadable_files(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._expect(
            _fit(tmp_path / "nope.jsonl", tmp_path, "--tau", "0.9"), "no such file", capsys
        )
        self._expect(_fit(tmp_path, tmp_path, "--tau", "0.9"), "directory", capsys)
        empty = tmp_path / "empty.jsonl"
        empty.write_text("\n\n")
        self._expect(_fit(empty, tmp_path, "--tau", "0.9"), "no records", capsys)
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("")
        self._expect(_fit(empty_csv, tmp_path, "--tau", "0.9"), "no records", capsys)
        binary = tmp_path / "bin.jsonl"
        binary.write_bytes(b"\xff\xfe\x00garbage")
        self._expect(_fit(binary, tmp_path, "--tau", "0.9"), "not UTF-8", capsys)

    def test_invalid_json(
        self, tmp_path: Path, recs: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = write_jsonl(tmp_path / "d.jsonl", recs[:5])
        with path.open("a") as fh:
            fh.write('{"id": "broken", "u": 0.3,\n')
        self._expect(_fit(path, tmp_path, "--tau", "0.9"), "line 6: invalid JSON", capsys)
        path.write_text("[1, 2]\n")
        self._expect(
            _fit(path, tmp_path, "--tau", "0.9", "--format", "jsonl"),
            "line 1: expected a JSON object",
            capsys,
        )
        arr = tmp_path / "a.json"
        arr.write_text('[{"u": 0.1}, 3]')
        self._expect(_fit(arr, tmp_path, "--tau", "0.9"), "element 1", capsys)

    def test_missing_column(
        self, tmp_path: Path, recs: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        for r in recs:
            del r["large_correct"]
        path = write_jsonl(tmp_path / "d.jsonl", recs)
        err = self._expect(
            _fit(path, tmp_path, "--tau", "0.9"), "missing column(s) large_correct", capsys
        )
        assert "found: id, u, small_correct, split" in err

    def test_missing_value_names_the_record(
        self, tmp_path: Path, recs: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        del recs[16]["u"]
        path = write_jsonl(tmp_path / "d.jsonl", recs)
        self._expect(
            _fit(path, tmp_path, "--tau", "0.9"),
            "line 17, id 'q00016': missing value for 'u'",
            capsys,
        )

    @pytest.mark.parametrize(
        "field, value, needle",
        [
            ("u", "abc", "'u' is not a number: 'abc'"),
            ("u", [0.1], "'u' must be a number, got list"),
            ("u", 1.2, "'u' = 1.2 is outside [0.0, 1.0]"),
            ("small_correct", -1, "'small_correct' = -1.0 is outside"),
            ("large_correct", 2, "'large_correct' = 2.0 is outside"),
            ("split", "holdout", "split field 'split' = 'holdout'"),
        ],
    )
    def test_bad_values(
        self,
        tmp_path: Path,
        recs: list[dict[str, Any]],
        field: str,
        value: Any,
        needle: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        recs[3][field] = value
        path = write_jsonl(tmp_path / "d.jsonl", recs)
        err = self._expect(_fit(path, tmp_path, "--tau", "0.9"), needle, capsys)
        assert "line 4" in err

    def test_nan_and_inf(
        self, tmp_path: Path, recs: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "d.jsonl"
        lines = [json.dumps(r) for r in recs]
        lines[2] = lines[2].replace(f'"u": {recs[2]["u"]!r}', '"u": NaN')
        path.write_text("\n".join(lines) + "\n")
        self._expect(_fit(path, tmp_path, "--tau", "0.9"), "values must be finite", capsys)
        csv_path = write_csv(tmp_path / "d.csv", recs)
        text = csv_path.read_text().splitlines()
        text[5] = text[5].replace(repr(recs[4]["u"]), "inf")
        csv_path.write_text("\n".join(text) + "\n")
        self._expect(_fit(csv_path, tmp_path, "--tau", "0.9"), "values must be finite", capsys)

    def test_csv_ragged_row(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        path = tmp_path / "d.csv"
        path.write_text("id,u,small_correct,large_correct\nq1,0.1,1,1\nq2,0.2,1,1,extra\n")
        self._expect(_fit(path, tmp_path, "--tau", "0.9"), "line 3: more values", capsys)

    def test_empty_splits(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        recs = make_records(200)
        for r in recs:
            if r["split"] == "val":
                r["split"] = "test"
        path = write_jsonl(tmp_path / "d.jsonl", recs)
        self._expect(
            _fit(path, tmp_path, "--tau", "0.9"),
            "the validation split is empty (no record has split = val)",
            capsys,
        )
        tiny = write_jsonl(tmp_path / "t.jsonl", make_records(1, split=False))
        self._expect(_fit(tiny, tmp_path, "--tau", "0.9"), "split is empty", capsys)

    def test_partial_scores(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        recs = make_records(300, scores=True)
        del recs[10]["small_score"]
        path = write_jsonl(tmp_path / "d.jsonl", recs)
        self._expect(
            _fit(path, tmp_path, "--tau", "0.8"),
            "small_score is missing on some or all records",
            capsys,
        )
        only_small = make_records(300, scores=True)
        for r in only_small:
            del r["large_score"]
        path2 = write_jsonl(tmp_path / "s.jsonl", only_small)
        self._expect(
            _fit(path2, tmp_path, "--tau", "0.8"),
            "large_score is missing on some or all records",
            capsys,
        )
        self._expect(
            _fit(path, tmp_path, "--tau", "0.8", "--metric", "score"),
            "missing value for 'small_score'",
            capsys,
        )
        code, _, _ = run(_fit(path, tmp_path, "--tau", "0.8", "--metric", "correct"), capsys)
        assert code == EXIT_OK

    def test_missing_split_field_and_latency(
        self, data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._expect(
            _fit(data, tmp_path, "--tau", "0.9", "--split-field", "fold"),
            "no split field 'fold'",
            capsys,
        )
        self._expect(
            _fit(data, tmp_path, "--tau", "0.9", "--cost-from-latency"),
            "missing column(s) latency_small_ms, latency_large_ms",
            capsys,
        )

    def test_latency_ratio_must_exceed_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        recs = make_records(300, latency=True)
        for r in recs:
            r["latency_small_ms"], r["latency_large_ms"] = r["latency_large_ms"], 1.0
        path = write_jsonl(tmp_path / "d.jsonl", recs)
        self._expect(
            _fit(path, tmp_path, "--tau", "0.8", "--cost-from-latency"),
            "latency ratio c_large / c_small",
            capsys,
        )
        recs[0]["latency_large_ms"] = 0
        path = write_jsonl(tmp_path / "z.jsonl", recs)
        self._expect(
            _fit(path, tmp_path, "--tau", "0.8", "--cost-from-latency"),
            "'latency_large_ms' = 0.0 is outside (0.0, inf)",
            capsys,
        )

    def test_unwritable_output(
        self, data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        argv = [
            "fit",
            "--data",
            str(data),
            "--tau",
            "0.9",
            "--out",
            str(tmp_path / "missing_dir" / "r.json"),
        ]
        self._expect(argv, "cannot write", capsys)

    def test_router_files(
        self, tmp_path: Path, router: Path, data: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        good = json.loads(router.read_text())

        def route_with(doc: Any, raw: str | None = None) -> str:
            path = tmp_path / "bad_router.json"
            path.write_text(raw if raw is not None else json.dumps(doc))
            return self._expect(["route", "--router", str(path), "--u", "0.3"], "", capsys)

        assert "no such file" in self._expect(
            ["route", "--router", str(tmp_path / "none.json"), "--u", "0.3"], "", capsys
        )
        assert "not a valid router JSON file" in route_with(None, raw="{not json")
        assert "not a valid router JSON file" in route_with(
            None, raw=router.read_text().replace(str(good["theta"]), "NaN", 1)
        )
        assert "not a UCCI router file" in route_with({**good, "format": "other"})
        assert "unsupported router format version 2" in route_with({**good, "version": 2})
        assert "strictly increasing" in route_with(
            {**good, "calibrator": {"x": [0.2, 0.1], "y": [0.0, 1.0]}}
        )
        assert "non-decreasing" in route_with(
            {**good, "calibrator": {"x": [0.1, 0.2], "y": [0.9, 0.1]}}
        )
        missing_theta = dict(good)
        del missing_theta["theta"]
        assert "theta" in route_with(missing_theta)
        assert "cost_model" in route_with({**good, "cost_model": "fixed"})
        # Unknown keys are ignored.
        extra = tmp_path / "extra.json"
        extra.write_text(json.dumps({**good, "comment": "hello", "fit": None}))
        assert run(["route", "--router", str(extra), "--u", "0.3"], capsys)[0] == EXIT_OK

    def test_split_selection_without_information(
        self, tmp_path: Path, router: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        doc = json.loads(router.read_text())
        del doc["fit"]
        bare = tmp_path / "bare.json"
        bare.write_text(json.dumps(doc))
        data = write_jsonl(tmp_path / "n.jsonl", make_records(100, split=False))
        self._expect(
            ["evaluate", "--router", str(bare), "--data", str(data), "--split", "test"],
            "cannot select split 'test'",
            capsys,
        )
        recs = make_records(100)
        for r in recs:
            r["split"] = "cal"
        only_cal = write_jsonl(tmp_path / "c.jsonl", recs)
        self._expect(
            ["report", "--router", str(bare), "--data", str(only_cal), "--split", "test"],
            "the test split is empty",
            capsys,
        )

    def test_evaluate_and_report_need_columns(
        self, router: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "u.csv"
        path.write_text("u,small_correct\n0.1,1\n")
        self._expect(
            ["evaluate", "--router", str(router), "--data", str(path)],
            "missing column(s) large_correct",
            capsys,
        )
        path.write_text("u\n0.1\n")
        self._expect(
            ["report", "--router", str(router), "--data", str(path)],
            "missing column(s) small_correct",
            capsys,
        )
        self._expect(
            ["route", "--router", str(router), "--data", str(tmp_path / "x.csv")],
            "no such file",
            capsys,
        )


class TestMoreInputs:
    def test_booleans_and_true_false_strings(
        self, tmp_path: Path, recs: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        as_bool = [
            {
                **r,
                "small_correct": bool(r["small_correct"]),
                "large_correct": bool(r["large_correct"]),
            }
            for r in recs
        ]
        a = run_json(
            [
                "fit",
                "--data",
                str(write_jsonl(tmp_path / "b.jsonl", as_bool)),
                "--tau",
                "0.9",
                "--out",
                str(tmp_path / "a.json"),
            ],
            capsys,
        )
        as_text = [
            {
                **r,
                "small_correct": "true" if r["small_correct"] else "False",
                "large_correct": "TRUE" if r["large_correct"] else "false",
            }
            for r in recs
        ]
        b = run_json(
            [
                "fit",
                "--data",
                str(write_csv(tmp_path / "t.csv", as_text)),
                "--tau",
                "0.9",
                "--out",
                str(tmp_path / "b.json"),
            ],
            capsys,
        )
        c = run_json(
            [
                "fit",
                "--data",
                str(write_jsonl(tmp_path / "n.jsonl", recs)),
                "--tau",
                "0.9",
                "--out",
                str(tmp_path / "c.json"),
            ],
            capsys,
        )
        assert a["theta"] == b["theta"] == c["theta"]
        assert a["validation"] == c["validation"] == b["validation"]

    def test_tsv_and_sniffed_json_array(
        self, tmp_path: Path, recs: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        tsv = tmp_path / "d.tsv"
        header = list(recs[0])
        tsv.write_text(
            "\t".join(header)
            + "\n"
            + "".join("\t".join(str(r[k]) for k in header) + "\n" for r in recs)
        )
        arr = tmp_path / "records.txt"
        arr.write_text(json.dumps(recs))
        docs = [
            run_json(
                ["fit", "--data", str(p), "--tau", "0.9", "--out", str(tmp_path / f"r{i}.json")],
                capsys,
            )
            for i, p in enumerate((tsv, arr))
        ]
        assert docs[0]["data"]["format"] == "csv" and docs[1]["data"]["format"] == "json"
        assert docs[0]["theta"] == docs[1]["theta"]

    def test_bad_json_array_documents(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "a.json"
        path.write_text('[{"u": 0.1},')
        code, _, err = run(_fit(path, tmp_path, "--tau", "0.9"), capsys)
        assert code == EXIT_INPUT and "invalid JSON" in err
        path.write_text('{"u": 0.1}')
        code, _, err = run(_fit(path, tmp_path, "--tau", "0.9", "--format", "json"), capsys)
        assert code == EXIT_INPUT and "expected a JSON array" in err

    @pytest.mark.skipif(
        sys.platform.startswith("win") or os.geteuid() == 0,
        reason="needs POSIX permissions and a non-root user",
    )
    def test_permission_denied(
        self, tmp_path: Path, data: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        locked = tmp_path / "locked.jsonl"
        locked.write_bytes(data.read_bytes())
        locked.chmod(0)
        try:
            code, _, err = run(_fit(locked, tmp_path, "--tau", "0.9"), capsys)
        finally:
            locked.chmod(0o600)
        assert code == EXIT_INPUT and "cannot read" in err

    def test_output_path_is_a_directory(
        self, data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = tmp_path / "taken"
        target.mkdir()
        code, _, err = run(
            ["fit", "--data", str(data), "--tau", "0.9", "--out", str(target)], capsys
        )
        assert code == EXIT_INPUT and "cannot write" in err
        assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []

    @pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
    def test_router_file_mode_follows_umask(self, data: Path, tmp_path: Path) -> None:
        old = os.umask(0o022)
        try:
            assert main(_fit(data, tmp_path, "--tau", "0.9")) == EXIT_OK
        finally:
            os.umask(old)
        assert (tmp_path / "r.json").stat().st_mode & 0o777 == 0o644

    def test_split_field_with_missing_labels(
        self, tmp_path: Path, recs: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
    ) -> None:
        del recs[7]["split"]
        path = write_jsonl(tmp_path / "d.jsonl", recs)
        code, _, err = run(_fit(path, tmp_path, "--tau", "0.9", "--split-field", "split"), capsys)
        assert code == EXIT_INPUT and "line 8, id 'q00007': missing value for split field" in err
        code, _, err = run(_fit(path, tmp_path, "--tau", "0.9"), capsys)
        assert code == EXIT_INPUT
        assert "line 8, id 'q00007': no 'split' value, while other records have one" in err
        # An explicit random split ignores the incomplete field.
        doc = run_json(_fit(path, tmp_path, "--tau", "0.9", "--seed", "0"), capsys)
        assert doc["split"]["method"] == "random"

    def test_incomplete_split_provenance(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data = write_jsonl(tmp_path / "d.jsonl", make_records(500, split=False))
        rpath = tmp_path / "r.json"
        run_json(["fit", "--data", str(data), "--tau", "0.8", "--out", str(rpath)], capsys)
        doc = json.loads(rpath.read_text())
        del doc["fit"]["split"]["seed"]
        rpath.write_text(json.dumps(doc))
        code, _, err = run(
            ["evaluate", "--router", str(rpath), "--data", str(data), "--split", "test"], capsys
        )
        assert code == EXIT_INPUT and "'fit.split' is incomplete" in err

    def test_evaluate_split_field_must_exist(
        self, router: Path, data: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, _, err = run(
            [
                "evaluate",
                "--router",
                str(router),
                "--data",
                str(data),
                "--split",
                "test",
                "--split-field",
                "fold",
            ],
            capsys,
        )
        assert code == EXIT_INPUT and "no split field 'fold'" in err

    def test_no_escalation_reports_undefined_as_null(
        self, data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rpath = tmp_path / "r.json"
        fit = run_json(["fit", "--data", str(data), "--tau", "0", "--out", str(rpath)], capsys)
        assert fit["theta"] == 1.0 and fit["validation"]["escalation_rate"] == 0.0
        doc = run_json(
            [
                "evaluate",
                "--router",
                str(rpath),
                "--data",
                str(data),
                "--split",
                "test",
                "--bootstrap",
                "0",
            ],
            capsys,
        )
        assert doc["assumption_ii"]["large_accuracy_escalated"] is None
        assert doc["assumption_ii"]["gap"] is None
        assert doc["ucci"]["cost"] == 1.0
        code, out, _ = run(
            [
                "evaluate",
                "--router",
                str(rpath),
                "--data",
                str(data),
                "--split",
                "test",
                "--bootstrap",
                "0",
            ],
            capsys,
        )
        assert code == EXIT_OK and "n/a on escalated queries" in out
        assert "95% CI" not in out


class TestInfeasible:
    def test_tau_reports_best_accuracy(
        self,
        data: Path,
        tmp_path: Path,
        recs: list[dict[str, Any]],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        code, out, err = run(_fit(data, tmp_path, "--tau", "0.999"), capsys)
        assert code == EXIT_INFEASIBLE and out == ""
        assert "no threshold reaches tau = 0.999 on the validation split (n = 400)" in err
        assert "best achievable validation accuracy is" in err
        best = float(err.split("best achievable validation accuracy is ")[1].split()[0])
        cal_c = columns(recs, rows_of(recs, "cal"))
        val_c = columns(recs, rows_of(recs, "val"))
        cal = IsotonicCalibrator().fit(cal_c["u"], 1 - cal_c["small_correct"])
        p = cal.predict(val_c["u"])
        ref = max(
            policy_accuracy(p > t, val_c["small_correct"], val_c["large_correct"])
            for t in ucci.DEFAULT_GRID
        )
        assert best == pytest.approx(ref, abs=5e-5)
        assert not (tmp_path / "r.json").exists()

    def test_budget_below_cheapest(
        self, data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, _, err = run(_fit(data, tmp_path, "--budget", "0.5"), capsys)
        assert code == EXIT_INFEASIBLE
        assert "the cheapest threshold costs 1.0000 per query" in err


# ---------------------------------------------------------------------------
# Subprocess: python -m ucci.cli and the console script
# ---------------------------------------------------------------------------


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _sub(args: list[str], cwd: Path, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        env=_env(),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


class TestSubprocess:
    def test_module_entry_point_end_to_end(self, data: Path, tmp_path: Path) -> None:
        py = [sys.executable, "-m", "ucci.cli"]
        res = _sub([*py, "version"], tmp_path)
        assert res.returncode == 0 and res.stdout.strip() == f"ucci {ucci.__version__}"
        res = _sub([*py, "fit", "--data", str(data), "--tau", "0.9", "--out", "r.json"], tmp_path)
        assert res.returncode == 0, res.stderr
        assert (tmp_path / "r.json").exists()
        res = _sub(
            [
                *py,
                "evaluate",
                "--router",
                "r.json",
                "--data",
                str(data),
                "--split",
                "test",
                "--bootstrap",
                "50",
                "--json",
            ],
            tmp_path,
        )
        assert res.returncode == 0, res.stderr
        assert json.loads(res.stdout)["n"] == 1000
        res = _sub(
            [*py, "route", "--router", "r.json", "--data", "-", "--json"],
            tmp_path,
            stdin='{"id": "a", "u": 0.05}\n{"id": "b", "u": 0.95}\n',
        )
        assert res.returncode == 0, res.stderr
        assert [q["decision"] for q in json.loads(res.stdout)["queries"]] == ["small", "large"]

    def test_module_exit_codes(self, data: Path, tmp_path: Path) -> None:
        py = [sys.executable, "-m", "ucci.cli"]
        assert (
            _sub(
                [*py, "fit", "--data", "missing.jsonl", "--tau", "0.9", "--out", "r.json"], tmp_path
            ).returncode
            == EXIT_INPUT
        )
        assert (
            _sub([*py, "fit", "--data", str(data), "--out", "r.json"], tmp_path).returncode
            == EXIT_USAGE
        )
        res = _sub(
            [*py, "fit", "--data", str(data), "--tau", "0.9999", "--out", "r.json"], tmp_path
        )
        assert res.returncode == EXIT_INFEASIBLE
        assert "best achievable validation accuracy" in res.stderr

    def test_package_entry_point(self, tmp_path: Path) -> None:
        res = _sub([sys.executable, "-m", "ucci", "version"], tmp_path)
        assert res.returncode == 0, res.stderr
        assert res.stdout.strip() == f"ucci {ucci.__version__}"
        res = _sub([sys.executable, "-m", "ucci", "--help"], tmp_path)
        assert res.returncode == 0 and res.stdout.startswith("usage: ucci ")

    def test_package_main_module_imports_cli_main(self) -> None:
        import ucci.__main__ as entry

        assert entry.main is main

    def test_console_script(self, data: Path, tmp_path: Path) -> None:
        script = shutil.which("ucci", path=os.path.dirname(sys.executable))
        if script is None:
            pytest.skip("the ucci console script is not installed in this environment")
        res = _sub([script, "version", "--json"], tmp_path)
        assert res.returncode == 0, res.stderr
        assert json.loads(res.stdout)["command"] == "version"
        res = _sub(
            [script, "fit", "--data", str(data), "--tau", "0.9", "--out", "r.json", "--json"],
            tmp_path,
        )
        assert res.returncode == 0, res.stderr
