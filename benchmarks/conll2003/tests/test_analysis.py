"""Tests for the analysis pipeline on synthetic logs (no model, no network).

The synthetic logs are shaped exactly like ``generate.py`` output, so these
tests exercise the join, re-scoring, split, protocol, bootstrap and report
code end to end. They say nothing about CoNLL-2003 results.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analysis
import analyze
import conll

import ucci

NAMES = ["Ann Lee", "Bo Chen", "Paris", "Oslo", "EU", "UN", "German", "Dutch"]
TYPES = ["PER", "PER", "LOC", "LOC", "ORG", "ORG", "MISC", "MISC"]


def _gold(rng: np.random.Generator) -> Dict[str, List[str]]:
    g: Dict[str, List[str]] = {t: [] for t in conll.ENTITY_TYPES}
    for k in rng.choice(len(NAMES), size=int(rng.integers(0, 4)), replace=False):
        g[TYPES[k]].append(NAMES[k])
    return g


def _corrupt(gold: Dict[str, List[str]], rng: np.random.Generator) -> Dict[str, List[str]]:
    pred = {t: list(v) for t, v in gold.items()}
    if any(pred.values()) and rng.random() < 0.6:
        t = next(t for t in conll.ENTITY_TYPES if pred[t])
        pred[t] = pred[t][1:]
    else:
        pred["MISC"] = pred["MISC"] + ["Monday"]
    return pred


def _record(
    i: int,
    model: str,
    gold: Dict[str, List[str]],
    pred: Dict[str, List[str]],
    u: float,
    rng: np.random.Generator,
    raw: str = "",
) -> Dict[str, Any]:
    raw = raw or json.dumps(pred)
    ok, _, parsed = conll.parse_entities(raw)
    sc = conll.score_entities(parsed, gold, ok)
    return {
        "id": f"test-{i}",
        "source_split": "test",
        "model": model,
        "sentence": " ".join(["w"] * (3 + i % 7)),
        "raw_output": raw,
        "parse_ok": ok,
        "gold": gold,
        "exact_match": sc.exact_match,
        "tp": sc.tp,
        "fp": sc.fp,
        "fn": sc.fn,
        "per_type": sc.per_type,
        "u": u,
        "entropy": 2.0 * u + 0.1 * rng.random(),
        "max_prob": 1.0 - 0.8 * u,
        "n_tokens": int(20 + 10 * rng.random()),
        "latency_ms": 10.0 if model == "small" else 30.0,
    }


def make_logs(n: int = 600, seed: int = 0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Small model wrong with probability rising in u; large model right 90% of the time."""
    rng = np.random.default_rng(seed)
    small, large = [], []
    for i in range(n):
        gold = _gold(rng)
        u = float(rng.beta(2.0, 5.0))
        p_err = 1.0 / (1.0 + math.exp(-12.0 * (u - 0.35)))
        s_pred = _corrupt(gold, rng) if rng.random() < p_err else gold
        l_pred = _corrupt(gold, rng) if rng.random() < 0.1 else gold
        small.append(_record(i, "small", gold, s_pred, u, rng))
        large.append(_record(i, "large", gold, l_pred, float(rng.random()), rng))
    return small, large


@pytest.fixture(scope="module")
def logs() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    return make_logs()


@pytest.fixture(scope="module")
def result(logs: Tuple[list, list]) -> Dict[str, Any]:
    small, large = logs
    return analysis.analyze(small, large, 1.0, 3.0, analysis.AnalysisConfig(n_boot=200))


# ---------------------------------------------------------------------------
# Reading and joining
# ---------------------------------------------------------------------------


def test_read_jsonl_skips_torn_last_line(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    p.write_text('{"id": "a"}\n{"id": "b"}\n{"id": "c", "u"')
    assert [r["id"] for r in analysis.read_jsonl(p)] == ["a", "b"]


def test_read_jsonl_rejects_corruption_and_duplicates(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "a"}\nnot json\n{"id": "b"}\n')
    with pytest.raises(ValueError, match="invalid JSON"):
        analysis.read_jsonl(p)
    p.write_text('{"id": "a"}\n{"id": "a"}\n')
    with pytest.raises(ValueError, match="twice"):
        analysis.read_jsonl(p)


def test_join_inner_and_rescore(logs: Tuple[list, list]) -> None:
    small, large = logs
    j = analysis.join_logs(small[:50], large[10:60])
    assert j.n == 40
    assert j.ids == sorted(j.ids)
    assert j.rescore_mismatches == {"small": 0, "large": 0}
    for i, rid in enumerate(j.ids):
        rec = next(r for r in small if r["id"] == rid)
        assert j.small_counts[i].tolist() == [rec["tp"], rec["fp"], rec["fn"]]
        assert j.small_em[i] == rec["exact_match"]


def test_join_counts_stale_scores_and_handles_empty_generation(logs: Tuple[list, list]) -> None:
    small = [dict(r) for r in logs[0][:20]]
    large = logs[1][:20]
    small[0]["tp"] = small[0]["tp"] + 5  # a logged score that no longer matches re-scoring
    small[1]["u"] = None
    small[1]["entropy"] = None
    small[1]["max_prob"] = None
    j = analysis.join_logs(small, large)
    assert j.rescore_mismatches["small"] == 1
    k = j.ids.index(small[1]["id"])
    assert j.u_missing[k] and j.u[k] == 1.0 and j.max_prob[k] == 0.0
    assert j.entropy[k] == np.nanmax(j.entropy)


def test_join_errors(logs: Tuple[list, list]) -> None:
    small, large = logs
    with pytest.raises(ValueError, match="share no"):
        analysis.join_logs(small[:5], large[5:10])
    other = [dict(r) for r in large[:5]]
    other[0]["gold"] = {"PER": ["Someone Else"]}
    with pytest.raises(ValueError, match="different gold"):
        analysis.join_logs(small[:5], other)


# ---------------------------------------------------------------------------
# Splits and costs
# ---------------------------------------------------------------------------


def test_assign_splits_sizes_and_determinism() -> None:
    ids = [f"test-{i}" for i in range(1001)]
    s = analysis.assign_splits(ids)
    assert {k: int((s == k).sum()) for k in analysis.SPLITS} == {
        "cal": 300,
        "val": 200,
        "test": 501,
    }
    shuffled = list(reversed(ids))
    s2 = analysis.assign_splits(shuffled)
    assert dict(zip(shuffled, s2)) == dict(zip(ids, s))
    assert (analysis.assign_splits(ids, seed=1) != s).any()


def test_assign_splits_follows_the_documented_sha256_rule() -> None:
    ids = [f"validation-{i}" for i in range(37)]
    order = sorted(
        range(len(ids)), key=lambda i: (hashlib.sha256(f"7:{ids[i]}".encode()).hexdigest(), i)
    )
    expected = (
        ["cal"] * 11 + ["val"] * 7 + ["test"] * 19
    )  # floor(0.3*37+0.5)=11, floor(0.2*37+0.5)=7
    s = analysis.assign_splits(ids, seed=7)
    assert [s[i] for i in order] == expected


@pytest.mark.parametrize(("cal", "val"), [(0.0, 0.2), (0.3, 1.0), (0.7, 0.4), (float("nan"), 0.2)])
def test_assign_splits_rejects_bad_fractions(cal: float, val: float) -> None:
    with pytest.raises(ValueError):
        analysis.assign_splits(["a", "b"], cal, val)


def test_cost_from_latency() -> None:
    assert analysis.cost_from_latency(
        {"small": {"mean_ms": 50.0}, "large": {"mean_ms": 151.0}}
    ) == (1.0, 3.02)
    with pytest.raises(ValueError, match="not above 1"):
        analysis.cost_from_latency({"small": {"mean_ms": 50.0}, "large": {"mean_ms": 40.0}})
    with pytest.raises(ValueError, match="mean_ms"):
        analysis.cost_from_latency({"small": {}})


# ---------------------------------------------------------------------------
# Learned confidence (extension)
# ---------------------------------------------------------------------------


def test_logistic_confidence_learns_the_direction() -> None:
    rng = np.random.default_rng(0)
    u = rng.random(2000)
    correct = (rng.random(2000) > u).astype(float)
    X = analysis.LogisticConfidence.features(u, 2 * u, 1 - u, np.full(2000, 20.0))
    model = analysis.LogisticConfidence().fit(X, correct)
    p = model.predict(X)
    assert np.corrcoef(p, -u)[0, 1] > 0.9
    assert abs(p.mean() - correct.mean()) < 0.01
    with pytest.raises(RuntimeError):
        analysis.LogisticConfidence().predict(X)


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


def test_result_layout(result: Dict[str, Any]) -> None:
    assert result["n_sentences"] == 600
    assert result["split"]["sizes"] == {"cal": 180, "val": 120, "test": 300}
    t = result["target"]
    assert t["tau"] == pytest.approx(
        t["val_small_f1"] + 0.75 * (t["val_large_f1"] - t["val_small_f1"])
    )
    assert result["budget"]["budget"] == pytest.approx(2.0)
    names = [r["method"] for r in result["at_target"]]
    for m in (
        "UCCI",
        "Conformal prediction",
        "FrugalGPT-style",
        "Entropy threshold",
        "Large-only",
        "Small-only",
        "Temperature scaling",
        "Uncalibrated u",
        "Isotonic on entropy",
        "Isotonic on max prob",
        "FrugalGPT-style (learned score)",
    ):
        assert m in names


def test_every_feasible_method_meets_target_on_validation(result: Dict[str, Any]) -> None:
    tau = result["target"]["tau"]
    for r in result["at_target"]:
        if r["feasible_on_val"] and r["method"] not in ("Small-only", "Large-only"):
            assert r["val"]["micro_f1"] >= tau - 1e-12, r["method"]


def test_every_feasible_method_is_within_budget_on_validation(result: Dict[str, Any]) -> None:
    b = result["budget"]["budget"]
    for r in result["at_budget"]:
        if r["feasible_on_val"] and r["method"] not in ("Small-only", "Large-only"):
            assert r["val"]["cost"] <= b + 1e-12, r["method"]


def test_test_numbers_are_consistent(result: Dict[str, Any]) -> None:
    j = result["_joined"]
    te = np.flatnonzero(result["_split"] == "test")
    for r in result["at_target"]:
        if "cost" not in r:
            continue
        rate = r["escalation_rate"]
        assert r["cost"] == pytest.approx(1.0 * (1 - rate) + 3.0 * rate)
        assert r["saving_vs_large"] == pytest.approx(1 - r["cost"] / 3.0)
        for key in ("cost", "micro_f1", "escalation_rate"):
            lo, hi = r["ci95"][key]
            assert lo <= r[key] + 1e-12 and r[key] <= hi + 1e-12, (r["method"], key)
    anchors = {r["method"]: r for r in result["at_target"]}
    tot_s = j.small_counts[te].sum(axis=0)
    tot_l = j.large_counts[te].sum(axis=0)
    assert anchors["Small-only"]["micro_f1"] == pytest.approx(ucci.micro_f1(*tot_s))
    assert anchors["Large-only"]["micro_f1"] == pytest.approx(ucci.micro_f1(*tot_l))
    assert (
        anchors["Small-only"]["escalation_rate"] == 0.0
        and anchors["Large-only"]["escalation_rate"] == 1.0
    )
    assert result["single_model_test"]["small"]["micro_f1"] == pytest.approx(
        anchors["Small-only"]["micro_f1"]
    )


def test_agrees_with_compare_routers(result: Dict[str, Any]) -> None:
    agree = result["checks"]["compare_routers_agreement"]
    assert {a["method"] for a in agree} >= {"UCCI", "Large-only", "Small-only"}
    for a in agree:
        assert a["cost_abs_diff"] < 1e-12 and a["f1_abs_diff"] < 1e-12, a


def test_ucci_escalates_the_uncertain_queries(result: Dict[str, Any]) -> None:
    ucci_row = next(r for r in result["at_target"] if r["method"] == "UCCI")
    assert ucci_row["feasible_on_val"]
    assert 0.0 < ucci_row["escalation_rate"] < 1.0
    a = result["assumption_ii"]
    assert a["small_f1_escalated"] < a["small_f1_kept"]
    assert a["gap"] == pytest.approx(a["large_f1_all_test"] - a["large_f1_escalated"])


def test_paired_differences(result: Dict[str, Any]) -> None:
    for r in result["at_target"]:
        p = r.get("paired_vs_UCCI")
        if p is None:
            continue
        assert p["cost_diff_ci"][0] <= p["cost_diff"] + 1e-12 <= p["cost_diff_ci"][1] + 2e-12
    assert "paired_vs_UCCI" not in next(r for r in result["at_target"] if r["method"] == "UCCI")


def test_calibration_block(result: Dict[str, Any]) -> None:
    cal = result["calibration"]
    # In-sample isotonic calibration is near-perfect with equal-width bins; raw u is not calibrated here.
    assert cal["cal"]["isotonic"]["ece_uniform"] < cal["cal"]["raw_u"]["ece_uniform"]
    assert cal["test"]["isotonic"]["ece_uniform"] < cal["test"]["raw_u"]["ece_uniform"]
    for s in ("cal", "test"):
        for k in ("raw_u", "isotonic", "temperature_scaling"):
            b = cal[s][k]
            assert b["ece_uniform_ci95"][0] <= b["ece_uniform_ci95"][1]
            assert sum(r["count"] for r in b["reliability_uniform"]) == b["n"]
    assert cal["temperature"] > 0


def test_cost_ratio_sensitivity_keeps_theta(result: Dict[str, Any]) -> None:
    rows = result["cost_ratio_sensitivity"]
    assert [r["cost_ratio"] for r in rows] == [3.0, 3.02, 5.0, 10.0]
    assert [r["measured"] for r in rows] == [True, False, False, False]
    assert all(r["same_theta"] for r in rows)
    rate = next(r for r in result["at_target"] if r["method"] == "UCCI")["escalation_rate"]
    for r in rows:
        assert r["ucci_cost"] == pytest.approx(1 + (r["cost_ratio"] - 1) * rate)


def test_per_entity_and_summary(result: Dict[str, Any]) -> None:
    pe = result["per_entity_test"]
    assert set(pe) == {"small", "large", "ucci_routed"}
    assert set(pe["small"]) == set(conll.ENTITY_TYPES) | {"overall (micro)"}
    assert pe["small"]["overall (micro)"] == pytest.approx(
        result["single_model_test"]["small"]["micro_f1"]
    )
    summ = result["per_split_summary"]
    assert sum(summ[s]["n"] for s in analysis.SPLITS) == summ["all"]["n"] == 600
    assert summ["all"]["small"]["mean_latency_ms_amortized"] == pytest.approx(10.0)


def test_oracle_meets_the_target_and_the_budget(result: Dict[str, Any]) -> None:
    o = result["oracle_at_target"]
    assert o["micro_f1"] >= result["target"]["tau"] - 1e-12
    assert 0.0 < o["escalation_rate"] < 1.0
    b = result["oracle_at_budget"]
    assert b["cost"] <= result["budget"]["budget"] + 1e-12
    assert "Oracle (analysis only)" in result["_figures"]["points"]


def test_oracle_linearization_orders_by_micro_f1_gain() -> None:
    small, large = make_logs(200, seed=5)
    j = analysis.join_logs(small, large)
    te = np.arange(j.n)
    o = analysis.oracle_point(j, te, 1.0, 3.0, tau=1.01)
    assert "infeasible_reason" in o and "cost" not in o


def test_to_json_is_strict_json(result: Dict[str, Any]) -> None:
    out = analysis.to_json(result)
    assert not any(k.startswith("_") for k in out)
    json.dumps(out, allow_nan=False)


def test_joined_records_feed_the_cli_format(result: Dict[str, Any]) -> None:
    recs = list(analysis.iter_joined_records(result))
    assert len(recs) == 600
    for r in recs[:20]:
        assert {"id", "split", "u", "small_correct", "large_correct"} <= set(r)
        assert r["split"] in analysis.SPLITS and 0.0 <= r["u"] <= 1.0


def test_infeasible_target_is_reported_not_raised(logs: Tuple[list, list]) -> None:
    small, large = logs
    res = analysis.analyze(
        small,
        large,
        1.0,
        3.0,
        analysis.AnalysisConfig(target_f1=1.01, n_boot=20, include_ablations=False),
    )
    ucci_row = next(r for r in res["at_target"] if r["method"] == "UCCI")
    assert not ucci_row["feasible_on_val"] and "tau" in ucci_row["infeasible_reason"]
    assert res["assumption_ii"] == {} and res["cost_ratio_sensitivity"] == []


def test_analyze_validates_costs_and_splits(logs: Tuple[list, list]) -> None:
    small, large = logs
    with pytest.raises(ValueError, match="assumption"):
        analysis.analyze(small, large, 1.0, 1.0)
    with pytest.raises(ValueError, match="split is empty"):
        analysis.analyze(small[:2], large[:2], 1.0, 3.0, analysis.AnalysisConfig(n_boot=10))


# ---------------------------------------------------------------------------
# The command-line script
# ---------------------------------------------------------------------------


def _write(path: Path, rows: List[Dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_analyze_script_end_to_end(logs: Tuple[list, list], tmp_path: Path) -> None:
    small, large = logs
    s = _write(tmp_path / "small.jsonl", small)
    lg = _write(tmp_path / "large.jsonl", large)
    lat = tmp_path / "latency.json"
    lat.write_text(
        json.dumps({"small": {"mean_ms": 40.0}, "large": {"mean_ms": 100.0}, "n_queries": 100})
    )
    out = tmp_path / "out"
    pytest.importorskip("matplotlib")
    assert (
        analyze.main(
            [
                "--small",
                str(s),
                "--large",
                str(lg),
                "--latency",
                str(lat),
                "--out-dir",
                str(out),
                "--n-boot",
                "50",
            ]
        )
        == 0
    )
    res = json.loads((out / "results.json").read_text())
    assert res["costs"]["c_large"] == pytest.approx(2.5)
    assert res["provenance"]["inputs"]["small"]["sha256"] == analyze.sha256_file(s)
    md = (out / "results.md").read_text()
    assert "Routing at the F1 target" in md and "| UCCI |" in md
    assert "\u2014" not in md and "\u2013" not in md
    for f in ("reliability_test.png", "reliability_cal.png", "pareto_test.png"):
        assert (out / f).stat().st_size > 0
    lines = (out / "joined.jsonl").read_text().splitlines()
    assert len(lines) == 600


def test_analyze_script_cost_ratio_and_no_figures(logs: Tuple[list, list], tmp_path: Path) -> None:
    small, large = logs
    s = _write(tmp_path / "small.jsonl", small)
    lg = _write(tmp_path / "large.jsonl", large)
    out = tmp_path / "out"
    assert (
        analyze.main(
            [
                "--small",
                str(s),
                "--large",
                str(lg),
                "--cost-ratio",
                "3.02",
                "--out-dir",
                str(out),
                "--n-boot",
                "20",
                "--no-figures",
                "--no-ablations",
                "--target-f1",
                "0.8",
            ]
        )
        == 0
    )
    res = json.loads((out / "results.json").read_text())
    assert res["target"]["tau"] == 0.8 and res["figures"] == []
    assert not any(r["group"] == "ablation" for r in res["at_target"])


def test_analyze_script_argument_errors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        analyze.main(["--small", "a", "--large", "b", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):
        analyze.main(
            ["--small", "a", "--large", "b", "--out-dir", str(tmp_path), "--cost-ratio", "0.5"]
        )


def test_cli_fit_reads_joined_records(result: Dict[str, Any], tmp_path: Path) -> None:
    """The exported per-sentence file is valid input for ``ucci fit`` with its split field."""
    cli = pytest.importorskip("ucci.cli")
    data = tmp_path / "joined.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in analysis.iter_joined_records(result)))
    router = tmp_path / "router.json"
    code = cli.main(
        [
            "fit",
            "--data",
            str(data),
            "--tau",
            "0.3",
            "--c-small",
            "1",
            "--c-large",
            "3",
            "--out",
            str(router),
        ]
    )
    assert code == 0 and router.exists()


class _Broken:
    """A comparator whose calibration always fails."""

    def calibrate(self, scores: Any, e: Any, sample_weight: Any = None) -> "_Broken":
        raise ValueError("calibration exploded")


def test_a_failing_method_is_reported_not_fatal(
    logs: Tuple[list, list], capsys: pytest.CaptureFixture
) -> None:
    small, large = logs
    j = analysis.join_logs(small, large)
    split = analysis.assign_splits(j.ids)
    specs = [("table2", s) for s in analysis.B.table2_methods("max_prob")[:1]]
    specs.append(("extension", analysis.B.MethodSpec("Broken", "u", lambda cs, cl, cm: _Broken())))
    res = analysis.run_methods(j, split, 1.0, 3.0, tau=0.5, specs=specs)
    assert res[0].feasible and not res[1].feasible and res[1].failed
    assert "calibration exploded" in res[1].error
    assert "failed and is reported" in capsys.readouterr().err
    rows = analysis.bootstrap_rows(res, j, split, 1.0, 3.0, 0.5, n_boot=20)
    assert rows[1]["status"] == "failed" and "cost" not in rows[1]
    table = "\n".join(analyze.method_table(rows, True, ("table2", "extension")))
    assert "| Broken | not run: ValueError |" in table


def test_run_methods_needs_exactly_one_objective(logs: Tuple[list, list]) -> None:
    small, large = logs
    j = analysis.join_logs(small[:100], large[:100])
    split = analysis.assign_splits(j.ids)
    with pytest.raises(ValueError, match="exactly one"):
        analysis.run_methods(j, split, 1.0, 3.0)
    with pytest.raises(ValueError, match="exactly one"):
        analysis.run_methods(j, split, 1.0, 3.0, tau=0.5, budget=2.0)
