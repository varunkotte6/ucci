"""Analysis of a two-model CoNLL-2003 run: the paper's protocol end to end.

Given the generation logs of a small and a large model on the same sentences
(written by ``generate.py``) and a measured cost ratio (``latency.py``), this
module runs the evaluation protocol of paper Section 6.1:

1. split the pooled sentences into disjoint calibration, validation and test
   sets (30 / 20 / 50 by default) with a fixed seed;
2. fit the isotonic calibration map g on the calibration split, with the
   exact-match error event e(x) of Section 4.2;
3. select every method's operating point on the validation split, where both
   models have run, and route every test sentence end to end with the actual
   output of the chosen model, accumulating actual micro-F1 and cost.

It reports the methods of Table 2 (UCCI, split-conformal, FrugalGPT-style,
entropy threshold, large-only, small-only) at an F1 target and at a matched
cost budget, the ablations of Section 6.3 and Appendix B.4, calibration
quality (ECE, Figure 1), the Theorem 1 assumption (ii) check of Section 6.3,
per-entity F1 (the analogue of Table 4), cost-ratio sensitivity (Table 3),
per-split summary statistics, and 95% percentile bootstrap intervals over
test sentences with 1000 resamples (Section 6.2). Every number comes from the
logs.

The routing methods are the package's own: :class:`ucci.UCCIRouter` and the
classes of :mod:`ucci.baselines`. This module adds only the data handling,
the bootstrap bookkeeping and one extra comparator (a FrugalGPT-style router
on a learned confidence score, labelled as an extension).

The module needs numpy and ``ucci`` only (no torch), so an analysis can be
re-run anywhere from the logs.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import conll  # noqa: E402

import ucci  # noqa: E402
from ucci import baselines as B  # noqa: E402

__all__ = [
    "SPLITS",
    "AnalysisConfig",
    "Joined",
    "LogisticConfidence",
    "MethodResult",
    "analyze",
    "assign_splits",
    "bootstrap_rows",
    "cost_from_latency",
    "join_logs",
    "method_specs",
    "read_jsonl",
    "rescore",
    "run_methods",
]

#: Split names, in protocol order.
SPLITS: Tuple[str, str, str] = ("cal", "val", "test")


# ---------------------------------------------------------------------------
# Loading and joining the logs
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file; a torn last line (interrupted run) is skipped.

    Raises
    ------
    ValueError
        If a line other than the last is not valid JSON, or ids repeat.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    out: List[Dict[str, Any]] = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if i == len(lines) - 1:
                break
            raise ValueError(f"{path}:{i + 1}: invalid JSON ({exc})") from exc
    seen: Dict[str, int] = {}
    for i, r in enumerate(out):
        rid = str(r["id"])
        if rid in seen:
            raise ValueError(
                f"{path}: id {rid!r} appears twice (lines {seen[rid] + 1} and {i + 1})"
            )
        seen[rid] = i
    return out


def rescore(rec: Mapping[str, Any]) -> Tuple[conll.EntityScore, bool]:
    """Re-parse and re-score a logged raw output with the current scoring code.

    Returns
    -------
    score : conll.EntityScore
    parse_ok : bool
    """
    parse_ok, _, pred = conll.parse_entities(str(rec["raw_output"]))
    return conll.score_entities(pred, rec["gold"], parse_ok), parse_ok


@dataclass
class Joined:
    """Per-sentence arrays for the sentences both models answered.

    Counts arrays have shape (n, 3) holding (tp, fp, fn); per-type arrays
    map each entity type to such an array.
    """

    ids: List[str]
    source_split: List[str]
    sentence: List[str]
    u: np.ndarray
    u_missing: np.ndarray
    entropy: np.ndarray
    max_prob: np.ndarray
    small_counts: np.ndarray
    large_counts: np.ndarray
    small_em: np.ndarray
    large_em: np.ndarray
    small_parse_ok: np.ndarray
    large_parse_ok: np.ndarray
    small_per_type: Dict[str, np.ndarray]
    large_per_type: Dict[str, np.ndarray]
    small_ntok: np.ndarray
    large_ntok: np.ndarray
    small_latency_ms: np.ndarray
    large_latency_ms: np.ndarray
    n_words: np.ndarray
    n_gold: np.ndarray
    gold_per_type: Dict[str, np.ndarray]
    rescore_mismatches: Dict[str, int] = field(default_factory=dict)
    small_capped: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    large_capped: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    argmax_mismatch_tokens: Dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        """Number of joined sentences."""
        return len(self.ids)


def _gold_unique(gold: Mapping[str, Sequence[str]]) -> Dict[str, int]:
    return {
        t: len({conll.normalize_entity(s) for s in gold.get(t, ())} - {""})
        for t in conll.ENTITY_TYPES
    }


def join_logs(small: Sequence[Mapping[str, Any]], large: Sequence[Mapping[str, Any]]) -> Joined:
    """Inner-join two generation logs on ``id`` and build the analysis arrays.

    Outputs are re-scored from ``raw_output`` with the current scoring code
    (so a scoring fix never needs new generations); disagreements with the
    logged scores are counted in ``rescore_mismatches``. A sentence whose
    small-model generation was empty has no u(x) (Eq. 4 needs T >= 1); it is
    given u = 1, the maximum uncertainty, so every method escalates it, and
    is flagged in ``u_missing``.

    Raises
    ------
    ValueError
        If the logs share no ids, or the same id has different gold entities
        in the two logs (the logs come from different data).
    """
    by_id = {str(r["id"]): r for r in large}
    common = sorted(str(r["id"]) for r in small if str(r["id"]) in by_id)
    if not common:
        raise ValueError("the small and large logs share no sentence ids")
    s_by_id = {str(r["id"]): r for r in small}
    rows_s = [s_by_id[i] for i in common]
    rows_l = [by_id[i] for i in common]
    mism = {"small": 0, "large": 0}
    cols: Dict[str, List[Any]] = {
        k: []
        for k in (
            "u",
            "u_missing",
            "entropy",
            "max_prob",
            "sc",
            "lc",
            "sem",
            "lem",
            "sok",
            "lok",
            "sntok",
            "lntok",
            "slat",
            "llat",
            "nw",
            "ng",
            "scap",
            "lcap",
        )
    }
    mismatch_tokens = {"small": 0, "large": 0}
    s_pt: Dict[str, List[List[int]]] = {t: [] for t in conll.ENTITY_TYPES}
    l_pt: Dict[str, List[List[int]]] = {t: [] for t in conll.ENTITY_TYPES}
    g_pt: Dict[str, List[int]] = {t: [] for t in conll.ENTITY_TYPES}
    for rs, rl in zip(rows_s, rows_l):
        if rs["gold"] != rl["gold"]:
            raise ValueError(f"id {rs['id']!r} has different gold entities in the two logs")
        ss, sok = rescore(rs)
        ls, lok = rescore(rl)
        for name, sc, rec in (("small", ss, rs), ("large", ls, rl)):
            if "tp" in rec and (sc.tp, sc.fp, sc.fn, sc.exact_match) != (
                rec["tp"],
                rec["fp"],
                rec["fn"],
                rec["exact_match"],
            ):
                mism[name] += 1
        u_raw = rs.get("u")
        missing = u_raw is None or (isinstance(u_raw, float) and math.isnan(u_raw))
        cols["u"].append(1.0 if u_raw is None or missing else float(u_raw))
        cols["u_missing"].append(missing)
        cols["entropy"].append(float(rs["entropy"]) if rs.get("entropy") is not None else math.nan)
        cols["max_prob"].append(
            float(rs["max_prob"]) if rs.get("max_prob") is not None else math.nan
        )
        cols["sc"].append([ss.tp, ss.fp, ss.fn])
        cols["lc"].append([ls.tp, ls.fp, ls.fn])
        cols["sem"].append(ss.exact_match)
        cols["lem"].append(ls.exact_match)
        cols["sok"].append(sok)
        cols["lok"].append(lok)
        cols["sntok"].append(int(rs.get("n_tokens", 0)))
        cols["scap"].append(rs.get("stop_reason") == "length")
        cols["lcap"].append(rl.get("stop_reason") == "length")
        mismatch_tokens["small"] += int(rs.get("argmax_mismatch", 0) or 0)
        mismatch_tokens["large"] += int(rl.get("argmax_mismatch", 0) or 0)
        cols["lntok"].append(int(rl.get("n_tokens", 0)))
        cols["slat"].append(float(rs.get("latency_ms", math.nan)))
        cols["llat"].append(float(rl.get("latency_ms", math.nan)))
        cols["nw"].append(len(str(rs.get("sentence", "")).split()))
        gu = _gold_unique(rs["gold"])
        cols["ng"].append(sum(gu.values()))
        for t in conll.ENTITY_TYPES:
            s_pt[t].append(ss.per_type[t])
            l_pt[t].append(ls.per_type[t])
            g_pt[t].append(gu[t])
    ent = np.asarray(cols["entropy"], dtype=float)
    mp = np.asarray(cols["max_prob"], dtype=float)
    miss = np.asarray(cols["u_missing"], dtype=bool)
    # Empty generations have no entropy or max-prob either: treat as maximally uncertain.
    if miss.any():
        finite = ent[np.isfinite(ent)]
        ent[miss] = float(finite.max()) if finite.size else 0.0
        mp[miss] = 0.0
    return Joined(
        ids=common,
        source_split=[str(r.get("source_split", "")) for r in rows_s],
        sentence=[str(r.get("sentence", "")) for r in rows_s],
        u=np.asarray(cols["u"], dtype=float),
        u_missing=miss,
        entropy=ent,
        max_prob=mp,
        small_counts=np.asarray(cols["sc"], dtype=float),
        large_counts=np.asarray(cols["lc"], dtype=float),
        small_em=np.asarray(cols["sem"], dtype=float),
        large_em=np.asarray(cols["lem"], dtype=float),
        small_parse_ok=np.asarray(cols["sok"], dtype=bool),
        large_parse_ok=np.asarray(cols["lok"], dtype=bool),
        small_per_type={t: np.asarray(v, dtype=float) for t, v in s_pt.items()},
        large_per_type={t: np.asarray(v, dtype=float) for t, v in l_pt.items()},
        small_ntok=np.asarray(cols["sntok"], dtype=float),
        large_ntok=np.asarray(cols["lntok"], dtype=float),
        small_latency_ms=np.asarray(cols["slat"], dtype=float),
        large_latency_ms=np.asarray(cols["llat"], dtype=float),
        n_words=np.asarray(cols["nw"], dtype=float),
        n_gold=np.asarray(cols["ng"], dtype=float),
        gold_per_type={t: np.asarray(v, dtype=float) for t, v in g_pt.items()},
        rescore_mismatches=mism,
        small_capped=np.asarray(cols["scap"], dtype=bool),
        large_capped=np.asarray(cols["lcap"], dtype=bool),
        argmax_mismatch_tokens=mismatch_tokens,
    )


# ---------------------------------------------------------------------------
# Splits and costs
# ---------------------------------------------------------------------------


def _round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def assign_splits(
    ids: Sequence[str], cal_frac: float = 0.3, val_frac: float = 0.2, seed: int = 0
) -> np.ndarray:
    """Disjoint calibration / validation / test labels, one per id.

    The rule is the one ``ucci fit`` documents for its random split, so the
    CLI re-derives the same split from the same ids: order the ids by the
    SHA-256 hex digest of ``"<seed>:<id>"`` (ties by position), then the
    first ``floor(cal_frac * n + 0.5)`` are calibration, the next
    ``floor(val_frac * n + 0.5)`` validation and the rest test. It depends
    only on the ids and the seed, not on record order, platform or numpy.

    Raises
    ------
    ValueError
        If the fractions are not in (0, 1) or sum to more than 1.
    """
    for name, v in (("cal_frac", cal_frac), ("val_frac", val_frac)):
        if not (math.isfinite(v) and 0.0 < v < 1.0):
            raise ValueError(f"{name} must lie in (0, 1), got {v}")
    if cal_frac + val_frac > 1.0 + 1e-12:
        raise ValueError(f"cal_frac + val_frac must be at most 1, got {cal_frac + val_frac}")
    n = len(ids)
    keys = [hashlib.sha256(f"{seed}:{rid}".encode()).hexdigest() for rid in ids]
    order = sorted(range(n), key=lambda i: (keys[i], i))
    n_cal = min(_round_half_up(cal_frac * n), n)
    n_val = min(_round_half_up(val_frac * n), n - n_cal)
    labels = np.empty(n, dtype=object)
    for pos, i in enumerate(order):
        labels[i] = "cal" if pos < n_cal else ("val" if pos < n_cal + n_val else "test")
    return labels.astype(str)


def cost_from_latency(latency: Mapping[str, Any]) -> Tuple[float, float]:
    """(c_small, c_large) = (1, mean_large_ms / mean_small_ms) from ``latency.py`` output.

    Raises
    ------
    ValueError
        If the file does not have the expected fields or the ratio is not above 1.
    """
    try:
        ms_s = float(latency["small"]["mean_ms"])
        ms_l = float(latency["large"]["mean_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "latency file needs small.mean_ms and large.mean_ms (written by latency.py)"
        ) from exc
    if not (ms_s > 0 and ms_l > 0):
        raise ValueError(f"latencies must be positive, got small={ms_s}, large={ms_l}")
    ratio = ms_l / ms_s
    if not ratio > 1.0:
        raise ValueError(
            f"measured c_l/c_s = {ratio:.4f} is not above 1; Theorem 1 assumption (i) needs the "
            "large model to cost more. Check the latency run or pass --cost-ratio."
        )
    return 1.0, ratio


# ---------------------------------------------------------------------------
# Extension: FrugalGPT-style router on a learned confidence
# ---------------------------------------------------------------------------


class LogisticConfidence:
    """Learned confidence that the small model is right (extension, not in the paper).

    FrugalGPT (Chen, Zaharia and Zou, 2023, arXiv:2305.05176) scores answer
    reliability with a learned model and thresholds that score. This class is
    a small stand-in: an L2-regularized logistic regression, fit by Newton's
    method on the calibration split, from the small model's own signals
    (u, mean entropy, mean max probability, log number of tokens) to the
    exact-match event. Its output is used as the ``confidence`` signal of
    :class:`ucci.baselines.FrugalGPTStyleRouter`, whose threshold is then
    tuned on validation like every other method.

    Parameters
    ----------
    l2 : float
        Ridge penalty on the standardized weights (not on the intercept).
    """

    def __init__(self, l2: float = 1.0) -> None:
        self.l2 = float(l2)
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.w_: Optional[np.ndarray] = None

    @staticmethod
    def features(
        u: np.ndarray, entropy: np.ndarray, max_prob: np.ndarray, n_tokens: np.ndarray
    ) -> np.ndarray:
        """Feature matrix, one row per sentence."""
        return np.column_stack([u, entropy, max_prob, np.log1p(n_tokens)])

    def fit(self, X: np.ndarray, correct: np.ndarray, max_iter: int = 100) -> "LogisticConfidence":
        """Fit on features ``X`` and 0/1 labels ``correct`` (1 = small model right)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(correct, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        Z = np.column_stack([np.ones(len(X)), (X - self.mean_) / self.scale_])
        w = np.zeros(Z.shape[1])
        pen = np.full(Z.shape[1], self.l2)
        pen[0] = 0.0
        for _ in range(max_iter):
            p = 1.0 / (1.0 + np.exp(-np.clip(Z @ w, -35, 35)))
            grad = Z.T @ (p - y) + pen * w
            H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(pen) + 1e-9 * np.eye(Z.shape[1])
            step = np.linalg.solve(H, grad)
            w -= step
            if np.max(np.abs(step)) < 1e-10:
                break
        self.w_ = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """P(small model right) for each row of ``X``."""
        if self.w_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("call fit() first")
        Z = np.column_stack(
            [np.ones(len(X)), (np.asarray(X, dtype=float) - self.mean_) / self.scale_]
        )
        return 1.0 / (1.0 + np.exp(-np.clip(Z @ self.w_, -35, 35)))


# ---------------------------------------------------------------------------
# Running the methods
# ---------------------------------------------------------------------------


@dataclass
class MethodResult:
    """One method at one operating point, with its test routing mask.

    ``val_*`` describe the operating point on validation; ``test_mask`` is
    True for test sentences sent to the large model. ``feasible`` is False
    when no operating point met the target (or budget) on validation, in
    which case ``test_mask`` is None. The single-model anchors always have a
    mask; for them ``meets_constraint`` records whether they happen to meet
    the target (or budget) on validation.
    """

    name: str
    group: str
    signal: Optional[str]
    note: str
    feasible: bool
    meets_constraint: bool = False
    threshold: float = math.nan
    val_cost: float = math.nan
    val_f1: float = math.nan
    val_escalation: float = math.nan
    test_mask: Optional[np.ndarray] = None
    params: Dict[str, float] = field(default_factory=dict)
    error: str = ""
    failed: bool = False


def method_specs(
    include_ablations: bool = True, include_learned: bool = True
) -> List[Tuple[str, "B.MethodSpec"]]:
    """(group, spec) pairs: Table 2 methods, then extensions and ablations.

    Groups are ``"table2"`` (the paper's comparators, from
    :func:`ucci.baselines.table2_methods` with the mean max probability as
    the FrugalGPT-style confidence), ``"extension"`` (not in the paper) and
    ``"ablation"`` (Section 6.3 and Appendix B.4, from
    :func:`ucci.baselines.ablation_methods`).
    """
    out: List[Tuple[str, B.MethodSpec]] = [
        ("table2", s) for s in B.table2_methods(confidence_signal="max_prob")
    ]
    if include_learned:
        out.append(
            (
                "extension",
                B.MethodSpec(
                    "FrugalGPT-style (learned score)",
                    "learned_confidence",
                    lambda cs, cl, cm: B.FrugalGPTStyleRouter(cs, cl, cm),
                    note="extension, not in the paper: logistic score fit on the calibration split",
                ),
            )
        )
    if include_ablations:
        for s in B.ablation_methods():
            group = "extension" if "extension" in s.note or "extension" in s.name else "ablation"
            out.append((group, s))
    return out


def fitted_params(router: Any) -> Dict[str, float]:
    """Fitted parameters worth reporting: delta* and alpha* (conformal), the temperature."""
    out: Dict[str, float] = {}
    for name in ("delta_", "alpha_"):
        v = getattr(router, name, None)
        if isinstance(v, (int, float)):
            out[name.rstrip("_")] = float(v)
    cal = getattr(router, "calibrator", None)
    t = getattr(cal, "temperature_", None)
    if isinstance(t, (int, float)):
        out["temperature"] = float(t)
    return out


def _signals(j: Joined, idx: np.ndarray, learned: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    sig = {"u": j.u[idx], "entropy": j.entropy[idx], "max_prob": j.max_prob[idx]}
    if learned is not None:
        sig["learned_confidence"] = learned[idx]
    return sig


def run_methods(
    j: Joined,
    split: np.ndarray,
    c_small: float,
    c_large: float,
    tau: Optional[float] = None,
    budget: Optional[float] = None,
    specs: Optional[Sequence[Tuple[str, "B.MethodSpec"]]] = None,
    learned: Optional[np.ndarray] = None,
) -> List[MethodResult]:
    """The Section 6.1 protocol for every method, keeping each test mask.

    For each method: ``calibrate`` on the calibration split with
    e = 1 - exact match, ``choose_threshold`` on the validation split (for
    ``tau``: cheapest operating point with validation micro-F1 >= tau; for
    ``budget``: highest validation micro-F1 with validation cost <= budget),
    then ``escalate`` on the test split.
    """
    if (tau is None) == (budget is None):
        raise ValueError("pass exactly one of tau and budget")
    specs = list(specs) if specs is not None else method_specs()
    idx = {s: np.flatnonzero(split == s) for s in SPLITS}
    val_metric = ucci.routed_micro_f1(j.small_counts[idx["val"]], j.large_counts[idx["val"]])
    e_cal = 1.0 - j.small_em[idx["cal"]]
    sig = {s: _signals(j, idx[s], learned) for s in SPLITS}
    out: List[MethodResult] = []
    for group, spec in specs:
        res = MethodResult(spec.name, group, spec.signal, spec.note, feasible=False)
        if spec.signal is not None and spec.signal not in sig["cal"]:
            res.error = f"signal {spec.signal!r} not available"
            out.append(res)
            continue
        try:
            _run_one(spec, res, sig, idx, j, c_small, c_large, tau, budget, e_cal, val_metric)
        except ucci.InfeasibleTargetError as exc:
            res.error = str(exc)
        except Exception as exc:  # one failing comparator must not sink the whole analysis
            res.error = f"{type(exc).__name__}: {exc}"
            res.failed = True
            res.feasible = False
            res.test_mask = None
            print(
                f"[analysis] WARNING: method {spec.name!r} failed and is reported as such: {res.error}",
                file=sys.stderr,
            )
        out.append(res)
    return out


def _run_one(
    spec: "B.MethodSpec",
    res: MethodResult,
    sig: Mapping[str, Mapping[str, np.ndarray]],
    idx: Mapping[str, np.ndarray],
    j: Joined,
    c_small: float,
    c_large: float,
    tau: Optional[float],
    budget: Optional[float],
    e_cal: np.ndarray,
    val_metric: Callable[[np.ndarray], float],
) -> None:
    """Calibrate, select and route one method, filling ``res`` in place."""
    router = spec.factory(c_small, c_large, "routing")
    if spec.signal is None:
        x_val, x_test = np.zeros(idx["val"].size), np.zeros(idx["test"].size)
    else:
        xs = [sig[s][spec.signal] for s in SPLITS]
        if spec.transform is not None:
            xs = [np.asarray(spec.transform(x), dtype=float) for x in xs]
        router.calibrate(xs[0], e_cal)
        x_val, x_test = xs[1], xs[2]
    ch = router.choose_threshold(
        x_val,
        j.small_em[idx["val"]],
        j.large_em[idx["val"]],
        tau,
        budget=budget,
        metric=val_metric,
    )
    res.feasible = True
    if tau is not None:
        res.meets_constraint = bool(ch.accuracy >= tau)
    else:
        res.meets_constraint = budget is not None and bool(ch.cost <= budget)
    res.threshold = float(ch.theta)
    res.val_cost, res.val_f1, res.val_escalation = (
        float(ch.cost),
        float(ch.accuracy),
        float(ch.escalation_rate),
    )
    res.test_mask = np.asarray(router.escalate(x_test), dtype=bool)
    res.params = fitted_params(router)


# ---------------------------------------------------------------------------
# Test-split statistics with bootstrap intervals
# ---------------------------------------------------------------------------


def _routed_counts(mask: np.ndarray, small: np.ndarray, large: np.ndarray) -> np.ndarray:
    return np.where(mask[:, None], large, small)


def _f1_rows(counts: np.ndarray) -> float:
    tot = counts.sum(axis=0)
    return ucci.micro_f1(tot[0], tot[1], tot[2])


def _ci(stat: Callable[[np.ndarray], float], n: int, n_boot: int, seed: int) -> List[float]:
    lo, hi = ucci.bootstrap_ci(stat, n, n_boot=n_boot, alpha=0.05, seed=seed)
    return [float(lo), float(hi)]


def bootstrap_rows(
    results: Sequence[MethodResult],
    j: Joined,
    split: np.ndarray,
    c_small: float,
    c_large: float,
    tau: Optional[float],
    n_boot: int = 1000,
    seed: int = 0,
    reference: str = "UCCI",
) -> List[Dict[str, Any]]:
    """Test-split cost, micro-F1, escalation rate, delta vs target and saving, with 95% CIs.

    Intervals are percentile bootstrap intervals over test sentences
    (:func:`ucci.bootstrap_ci`, ``n_boot`` resamples). Every statistic uses
    the same seed, so all methods see the same resamples and the paired
    differences against ``reference`` (cost and F1) are proper paired
    bootstrap intervals. The thresholds stay fixed at their validation
    choice; the intervals cover test-set sampling only.
    """
    te = np.flatnonzero(split == "test")
    sc, lc = j.small_counts[te], j.large_counts[te]
    n = te.size
    ref = next((r for r in results if r.name == reference and r.feasible), None)
    ref_mask = ref.test_mask if ref is not None else None

    def cost_of(esc_frac: float) -> float:
        return c_small * (1.0 - esc_frac) + c_large * esc_frac

    def mean_stat(
        x: np.ndarray, f: Callable[[float], float] = float
    ) -> Callable[[np.ndarray], float]:
        def stat(ix: np.ndarray) -> float:
            return f(float(x[ix].mean()))

        return stat

    def f1_stat(c: np.ndarray) -> Callable[[np.ndarray], float]:
        def stat(ix: np.ndarray) -> float:
            return _f1_rows(c[ix])

        return stat

    def diff(
        a: Callable[[np.ndarray], float], b: Callable[[np.ndarray], float]
    ) -> Callable[[np.ndarray], float]:
        def stat(ix: np.ndarray) -> float:
            return a(ix) - b(ix)

        return stat

    def saving(frac: float) -> float:
        return 1.0 - cost_of(frac) / c_large

    rows: List[Dict[str, Any]] = []
    for r in results:
        row: Dict[str, Any] = {
            "method": r.name,
            "group": r.group,
            "signal": r.signal,
            "note": r.note,
            "feasible_on_val": r.meets_constraint,
            "threshold": r.threshold,
            "params": r.params,
            "val": {"cost": r.val_cost, "micro_f1": r.val_f1, "escalation_rate": r.val_escalation},
        }
        if not r.feasible or r.test_mask is None:
            row["status"] = "failed" if r.failed else "infeasible"
            row["infeasible_reason" if not r.failed else "error"] = r.error
            rows.append(row)
            continue
        row["status"] = "ok"
        m = r.test_mask
        esc = m.astype(float)
        counts = _routed_counts(m, sc, lc)
        f1 = _f1_rows(counts)
        cost = ucci.policy_cost(m, c_small, c_large)
        row.update(
            {
                "cost": cost,
                "micro_f1": f1,
                "escalation_rate": float(esc.mean()),
                "delta_vs_target": None if tau is None else f1 - tau,
                "saving_vs_large": 1.0 - cost / c_large,
                "n_test": int(n),
            }
        )
        ci = {
            "cost": _ci(mean_stat(esc, cost_of), n, n_boot, seed),
            "micro_f1": _ci(f1_stat(counts), n, n_boot, seed),
            "escalation_rate": _ci(mean_stat(esc), n, n_boot, seed),
            "saving_vs_large": _ci(mean_stat(esc, saving), n, n_boot, seed),
        }
        if tau is not None:
            ci["delta_vs_target"] = [ci["micro_f1"][0] - tau, ci["micro_f1"][1] - tau]
        if ref_mask is not None and r.name != reference:
            rc = _routed_counts(ref_mask, sc, lc)
            r_esc = ref_mask.astype(float)
            row["paired_vs_" + reference] = {
                "cost_diff": cost - ucci.policy_cost(ref_mask, c_small, c_large),
                "f1_diff": f1 - _f1_rows(rc),
                "cost_diff_ci": _ci(
                    diff(mean_stat(esc, cost_of), mean_stat(r_esc, cost_of)), n, n_boot, seed
                ),
                "f1_diff_ci": _ci(diff(f1_stat(counts), f1_stat(rc)), n, n_boot, seed),
            }
        row["ci95"] = ci
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _summary_stats(j: Joined, ix: np.ndarray) -> Dict[str, Any]:
    """Per-split summary statistics (the paper promises these for auditing)."""

    def model(
        counts: np.ndarray,
        em: np.ndarray,
        ok: np.ndarray,
        ntok: np.ndarray,
        lat: np.ndarray,
        capped: np.ndarray,
    ) -> Dict[str, Any]:
        tot = counts[ix].sum(axis=0)
        return {
            "micro_f1": ucci.micro_f1(tot[0], tot[1], tot[2]),
            "exact_match_rate": float(em[ix].mean()),
            "parse_failure_rate": float(1.0 - ok[ix].mean()),
            "tp": int(tot[0]),
            "fp": int(tot[1]),
            "fn": int(tot[2]),
            "mean_new_tokens": float(ntok[ix].mean()),
            "hit_max_new_tokens_rate": float(capped[ix].mean()) if capped.size else None,
            "mean_latency_ms_amortized": float(np.nanmean(lat[ix]))
            if np.isfinite(lat[ix]).any()
            else None,
        }

    ng = j.n_gold[ix]
    return {
        "n": int(ix.size),
        "source_splits": {
            s: int(sum(1 for i in ix if j.source_split[i] == s))
            for s in sorted(set(j.source_split))
        },
        "words_mean": float(j.n_words[ix].mean()),
        "words_median": float(np.median(j.n_words[ix])),
        "frac_with_entity": float((ng > 0).mean()),
        "entities_mean": float(ng.mean()),
        "entities_median": float(np.median(ng)),
        "entities_max": int(ng.max()) if ng.size else 0,
        "gold_entities_by_type": {t: int(j.gold_per_type[t][ix].sum()) for t in conll.ENTITY_TYPES},
        "u_mean": float(j.u[ix].mean()),
        "u_median": float(np.median(j.u[ix])),
        "empty_small_generations": int(j.u_missing[ix].sum()),
        "small": model(
            j.small_counts,
            j.small_em,
            j.small_parse_ok,
            j.small_ntok,
            j.small_latency_ms,
            j.small_capped,
        ),
        "large": model(
            j.large_counts,
            j.large_em,
            j.large_parse_ok,
            j.large_ntok,
            j.large_latency_ms,
            j.large_capped,
        ),
    }


def _per_type_f1(
    per_type: Mapping[str, np.ndarray],
    ix: np.ndarray,
    mask: Optional[np.ndarray] = None,
    other: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    total = np.zeros(3)
    for t in conll.ENTITY_TYPES:
        c = per_type[t][ix]
        if mask is not None and other is not None:
            c = np.where(mask[:, None], other[t][ix], c)
        s = c.sum(axis=0)
        total += s
        out[t] = ucci.micro_f1(s[0], s[1], s[2])
    out["overall (micro)"] = ucci.micro_f1(total[0], total[1], total[2])
    return out


def _ece_stat(p: np.ndarray, e: np.ndarray, strategy: str) -> Callable[[np.ndarray], float]:
    """ECE (10 bins) of the resampled rows, as a bootstrap statistic."""

    def stat(ix: np.ndarray) -> float:
        return ucci.ece(p[ix], e[ix], n_bins=10, strategy=strategy)

    return stat


def _calibration_block(p: np.ndarray, e: np.ndarray, n_boot: int, seed: int) -> Dict[str, Any]:
    """ECE (10 equal-width bins and deciles), Brier score and reliability rows, with ECE CIs."""
    n = p.size
    out: Dict[str, Any] = {"n": int(n), "brier": ucci.brier_score(p, e)}
    for strat in ("uniform", "quantile"):
        out[f"ece_{strat}"] = ucci.ece(p, e, n_bins=10, strategy=strat)
        out[f"ece_{strat}_ci95"] = _ci(_ece_stat(p, e, strat), n, n_boot, seed)
        out[f"reliability_{strat}"] = [
            r._asdict() for r in ucci.reliability_table(p, e, n_bins=10, strategy=strat)
        ]
    return out


def _curve_by_score(
    score: np.ndarray, small: np.ndarray, large: np.ndarray, c_small: float, c_large: float
) -> Dict[str, List[float]]:
    """Exact cost / micro-F1 curve of "escalate when score > t" over every distinct t (test split)."""
    order = np.argsort(-score, kind="mergesort")
    s = score[order]
    diff = (large - small)[order]
    cum = np.vstack([np.zeros(3), np.cumsum(diff, axis=0)])
    base = small.sum(axis=0)
    # Valid cut points: k = 0, the ends of runs of equal scores, and n.
    ks = [0] + [k for k in range(1, s.size) if s[k] != s[k - 1]] + [s.size]
    ks = sorted(set(ks))
    costs, f1s = [], []
    n = s.size
    for k in ks:
        tot = base + cum[k]
        f1s.append(ucci.micro_f1(tot[0], tot[1], tot[2]))
        r = k / n
        costs.append(c_small * (1 - r) + c_large * r)
    return {"cost": costs, "micro_f1": f1s, "escalation_rate": [k / n for k in ks]}


def oracle_point(
    j: Joined,
    te: np.ndarray,
    c_small: float,
    c_large: float,
    tau: Optional[float] = None,
    budget: Optional[float] = None,
    f_lin: float = 0.5,
) -> Dict[str, Any]:
    """Label-dependent oracle on the test split (extension, not in the paper; analysis only).

    :class:`ucci.baselines.Oracle` escalates test sentences in decreasing
    order of a per-sentence gain, seeing both models' labels. Micro-F1 is not
    a sum over sentences, so the gain is its linearization at F1 = ``f_lin``:
    a sentence's contribution is ``(2 - 2 f) tp - f (fp + fn)``, whose sum is
    positive exactly when the corpus micro-F1 exceeds f. Target and budget
    are checked on the true micro-F1 of the routed answers. The result is a
    greedy, approximate lower bound on the cost any router could reach on
    these labels; it cannot be deployed.
    """

    def lin(c: np.ndarray) -> np.ndarray:
        return (2.0 - 2.0 * f_lin) * c[:, 0] - f_lin * (c[:, 1] + c[:, 2])

    sc, lc = j.small_counts[te], j.large_counts[te]
    note = (
        "greedy on label-dependent per-sentence gains, micro-F1 linearized at the target; "
        "an approximate lower bound on cost"
    )
    try:
        ch = B.Oracle(c_small, c_large).evaluate(
            lin(sc), lin(lc), tau, budget=budget, metric=ucci.routed_micro_f1(sc, lc)
        )
    except ucci.InfeasibleTargetError as exc:
        return {"note": note, "infeasible_reason": str(exc)}
    return {
        "cost": ch.cost,
        "micro_f1": ch.accuracy,
        "escalation_rate": ch.escalation_rate,
        "note": note,
    }


# ---------------------------------------------------------------------------
# The full analysis
# ---------------------------------------------------------------------------


@dataclass
class AnalysisConfig:
    """Settings of one analysis run (all recorded in ``results.json``)."""

    cal_frac: float = 0.3
    val_frac: float = 0.2
    seed: int = 0
    target_frac: float = 0.75
    target_f1: Optional[float] = None
    budget_frac: float = 0.5
    budget: Optional[float] = None
    n_boot: int = 1000
    boot_seed: int = 0
    #: Hypothetical ratios for the Table 3 analogue, besides the measured one:
    #: the paper's measured H100 ratio 3.02 and its hypothetical 5 and 10.
    cost_ratios: Tuple[float, ...] = (3.02, 5.0, 10.0)
    include_ablations: bool = True
    include_learned: bool = True


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def analyze(
    small_log: Sequence[Mapping[str, Any]],
    large_log: Sequence[Mapping[str, Any]],
    c_small: float,
    c_large: float,
    config: Optional[AnalysisConfig] = None,
) -> Dict[str, Any]:
    """Run the full analysis and return a JSON-ready result dict.

    Parameters
    ----------
    small_log, large_log : sequence of dict
        Generation records of the small and the large model.
    c_small, c_large : float
        Normalized per-query costs (1 and the measured latency ratio).
    config : AnalysisConfig, optional

    Returns
    -------
    dict
        See ``README.md`` in this directory for the layout. Arrays needed
        for the figures are under ``"_figures"`` and are not part of the
        JSON written by ``analyze.py``.

    Raises
    ------
    ValueError
        If a split is empty or the costs violate Theorem 1 assumption (i).
    """
    cfg = config or AnalysisConfig()
    if not c_large > c_small > 0:
        raise ValueError(
            f"need c_large > c_small > 0 (Theorem 1, assumption (i)); got {c_small}, {c_large}"
        )
    j = join_logs(small_log, large_log)
    split = assign_splits(j.ids, cfg.cal_frac, cfg.val_frac, cfg.seed)
    idx = {s: np.flatnonzero(split == s) for s in SPLITS}
    for s in SPLITS:
        if idx[s].size == 0:
            raise ValueError(
                f"the {s} split is empty ({j.n} sentences); use more sentences or other fractions"
            )

    def f1_of(counts: np.ndarray, ix: np.ndarray) -> float:
        tot = counts[ix].sum(axis=0)
        return ucci.micro_f1(tot[0], tot[1], tot[2])

    f1_small_val = f1_of(j.small_counts, idx["val"])
    f1_large_val = f1_of(j.large_counts, idx["val"])
    if cfg.target_f1 is not None:
        tau = float(cfg.target_f1)
        tau_rule = "fixed by --target-f1"
    else:
        tau = f1_small_val + cfg.target_frac * (f1_large_val - f1_small_val)
        tau_rule = f"validation small-only F1 + {cfg.target_frac} x (large-only F1 - small-only F1)"
    if cfg.budget is not None:
        budget = float(cfg.budget)
        budget_rule = "fixed by --budget"
    else:
        budget = c_small + cfg.budget_frac * (c_large - c_small)
        budget_rule = f"c_small + {cfg.budget_frac} x (c_large - c_small)"

    learned = None
    learned_info: Dict[str, Any] = {}
    if cfg.include_learned:
        feats = LogisticConfidence.features(j.u, j.entropy, j.max_prob, j.small_ntok)
        lc = LogisticConfidence().fit(feats[idx["cal"]], j.small_em[idx["cal"]])
        learned = lc.predict(feats)
        learned_info = {
            "features": ["u", "entropy", "max_prob", "log1p(n_tokens)"],
            "weights_standardized": lc.w_.tolist() if lc.w_ is not None else None,
        }

    specs = method_specs(cfg.include_ablations, cfg.include_learned)
    res_t = run_methods(j, split, c_small, c_large, tau=tau, specs=specs, learned=learned)
    res_b = run_methods(j, split, c_small, c_large, budget=budget, specs=specs, learned=learned)
    rows_t = bootstrap_rows(res_t, j, split, c_small, c_large, tau, cfg.n_boot, cfg.boot_seed)
    rows_b = bootstrap_rows(res_b, j, split, c_small, c_large, tau, cfg.n_boot, cfg.boot_seed)

    # Cross-check against the package's own protocol runner.
    te = idx["test"]

    def split_data(ix: np.ndarray) -> "B.SplitData":
        return B.SplitData(
            signals={"u": j.u[ix], "entropy": j.entropy[ix], "max_prob": j.max_prob[ix]},
            small_score=j.small_em[ix],
            large_score=j.large_em[ix],
            small_error=1.0 - j.small_em[ix],
            metric=ucci.routed_micro_f1(j.small_counts[ix], j.large_counts[ix]),
        )

    cmp_rows = B.compare_routers(
        split_data(idx["cal"]),
        split_data(idx["val"]),
        split_data(te),
        tau=tau,
        c_small=c_small,
        c_large=c_large,
        methods=B.table2_methods(confidence_signal="max_prob"),
        include_oracle=False,
    )
    agree = []
    for cr in cmp_rows:
        mine = next((r for r in rows_t if r["method"] == cr.method), None)
        if mine is None or "cost" not in mine or not math.isfinite(cr.cost):
            continue
        agree.append(
            {
                "method": cr.method,
                "cost_abs_diff": abs(cr.cost - mine["cost"]),
                "f1_abs_diff": abs(cr.accuracy - mine["micro_f1"]),
            }
        )
    oracle_t = oracle_point(j, te, c_small, c_large, tau=tau, f_lin=tau)
    oracle_b = oracle_point(j, te, c_small, c_large, budget=budget, f_lin=tau)

    # Calibration quality (Figure 1, Section 6.2; Appendix B.4).
    router = ucci.UCCIRouter(c_small, c_large)
    router.calibrate(j.u[idx["cal"]], 1.0 - j.small_em[idx["cal"]])
    temp: Optional[Any] = None
    calib: Dict[str, Any] = {"n_knots": len(router.calibrator.to_dict()["x"])}
    try:
        temp = B.TemperatureScalingCalibrator().fit(j.u[idx["cal"]], 1.0 - j.small_em[idx["cal"]])
    except Exception as exc:  # report, do not abort the analysis
        calib["temperature_error"] = f"{type(exc).__name__}: {exc}"
        print(
            f"[analysis] WARNING: temperature scaling failed: {calib['temperature_error']}",
            file=sys.stderr,
        )
    for s in ("cal", "test"):
        ix = idx[s]
        e = 1.0 - j.small_em[ix]
        calib[s] = {
            "raw_u": _calibration_block(j.u[ix], e, cfg.n_boot, cfg.boot_seed),
            "isotonic": _calibration_block(
                np.asarray(router.error_probability(j.u[ix])), e, cfg.n_boot, cfg.boot_seed
            ),
        }
        if temp is not None:
            calib[s]["temperature_scaling"] = _calibration_block(
                np.asarray(temp.predict(j.u[ix])), e, cfg.n_boot, cfg.boot_seed
            )
    calib["temperature"] = (
        float(getattr(temp, "temperature_", math.nan)) if temp is not None else math.nan
    )

    # Assumption (ii) of Theorem 1 (Section 6.3) at the UCCI target operating point.
    ucci_t = next((r for r in res_t if r.name == "UCCI"), None)
    assumption: Dict[str, Any] = {}
    if ucci_t is not None and ucci_t.feasible and ucci_t.test_mask is not None:
        m = ucci_t.test_mask
        esc_ix, keep_ix = te[m], te[~m]
        assumption = {
            "large_f1_escalated": f1_of(j.large_counts, esc_ix) if esc_ix.size else None,
            "large_f1_all_test": f1_of(j.large_counts, te),
            "small_f1_escalated": f1_of(j.small_counts, esc_ix) if esc_ix.size else None,
            "small_f1_kept": f1_of(j.small_counts, keep_ix) if keep_ix.size else None,
            "n_escalated": int(esc_ix.size),
        }
        if assumption["large_f1_escalated"] is not None:
            assumption["gap"] = assumption["large_f1_all_test"] - assumption["large_f1_escalated"]

    # Per-entity F1 on test (Table 4 analogue).
    per_entity = {
        "small": _per_type_f1(j.small_per_type, te),
        "large": _per_type_f1(j.large_per_type, te),
    }
    if ucci_t is not None and ucci_t.feasible and ucci_t.test_mask is not None:
        per_entity["ucci_routed"] = _per_type_f1(
            j.small_per_type, te, ucci_t.test_mask, j.large_per_type
        )

    # Cost-ratio sensitivity (Table 3): the same UCCI routing re-costed.
    sens: List[Dict[str, Any]] = []
    if ucci_t is not None and ucci_t.feasible and ucci_t.test_mask is not None:
        ratios = [c_large] + [r for r in cfg.cost_ratios if r > c_small and abs(r - c_large) > 1e-9]
        for ratio in ratios:
            cost = ucci.policy_cost(ucci_t.test_mask, 1.0, ratio)
            r2 = ucci.UCCIRouter(1.0, ratio)
            r2.calibrate(j.u[idx["cal"]], 1.0 - j.small_em[idx["cal"]])
            ch = r2.choose_threshold(
                j.u[idx["val"]],
                None,
                None,
                tau,
                metric=ucci.routed_micro_f1(j.small_counts[idx["val"]], j.large_counts[idx["val"]]),
            )
            sens.append(
                {
                    "cost_ratio": ratio,
                    "measured": ratio == c_large,
                    "ucci_cost": cost,
                    "saving_vs_large": 1.0 - cost / ratio,
                    "theta_reselected": ch.theta,
                    "same_theta": bool(ch.theta == ucci_t.threshold),
                }
            )

    # Pareto curves on test (Figure 2).
    sc_t, lc_t = j.small_counts[te], j.large_counts[te]
    test_metric = ucci.routed_micro_f1(sc_t, lc_t)
    front = ucci.pareto_frontier(
        np.asarray(router.error_probability(j.u[te])),
        None,
        None,
        c_small,
        c_large,
        metric=test_metric,
    )
    curves = {
        "UCCI (isotonic p_hat, 0.005 grid)": {
            "cost": front.cost.tolist(),
            "micro_f1": front.accuracy.tolist(),
        },
        "Raw u": _curve_by_score(j.u[te], sc_t, lc_t, c_small, c_large),
        "Entropy": _curve_by_score(j.entropy[te], sc_t, lc_t, c_small, c_large),
        "1 - max prob": _curve_by_score(-j.max_prob[te], sc_t, lc_t, c_small, c_large),
    }

    summary = {s: _summary_stats(j, idx[s]) for s in SPLITS}
    summary["all"] = _summary_stats(j, np.arange(j.n))

    result: Dict[str, Any] = {
        "format": "ucci-conll2003-analysis",
        "version": 1,
        "n_sentences": j.n,
        "split": {
            "rule": "sha256('<seed>:<id>') order, then 30/20/50 cut (same rule as `ucci fit`)",
            "cal_frac": cfg.cal_frac,
            "val_frac": cfg.val_frac,
            "seed": cfg.seed,
            "sizes": {s: int(idx[s].size) for s in SPLITS},
        },
        "costs": {"c_small": c_small, "c_large": c_large, "cost_model": "routing"},
        "target": {
            "tau": tau,
            "rule": tau_rule,
            "val_small_f1": f1_small_val,
            "val_large_f1": f1_large_val,
        },
        "budget": {"budget": budget, "rule": budget_rule},
        "bootstrap": {
            "n_boot": cfg.n_boot,
            "seed": cfg.boot_seed,
            "level": 0.95,
            "method": "percentile, over test sentences, paired across methods",
        },
        "single_model_test": {
            "small": {
                "micro_f1": f1_of(j.small_counts, te),
                "exact_match_rate": float(j.small_em[te].mean()),
                "cost": c_small,
            },
            "large": {
                "micro_f1": f1_of(j.large_counts, te),
                "exact_match_rate": float(j.large_em[te].mean()),
                "cost": c_large,
            },
        },
        "at_target": rows_t,
        "at_budget": rows_b,
        "oracle_at_target": oracle_t,
        "oracle_at_budget": oracle_b,
        "calibration": calib,
        "assumption_ii": assumption,
        "per_entity_test": per_entity,
        "cost_ratio_sensitivity": sens,
        "learned_confidence": learned_info,
        "per_split_summary": summary,
        "checks": {
            "rescore_mismatches": j.rescore_mismatches,
            "argmax_mismatch_tokens": j.argmax_mismatch_tokens,
            "empty_small_generations": int(j.u_missing.sum()),
            "compare_routers_agreement": agree,
        },
        "_figures": {
            "reliability": {
                s: {
                    "u": j.u[idx[s]],
                    "p_hat": np.asarray(router.error_probability(j.u[idx[s]])),
                    "e": 1.0 - j.small_em[idx[s]],
                }
                for s in ("cal", "test")
            },
            "curves": curves,
            "points": {
                r["method"]: (r["cost"], r["micro_f1"])
                for r in rows_t
                if "cost" in r and r["group"] == "table2"
            },
            "tau": tau,
        },
        "_split": split,
        "_joined": j,
    }
    if oracle_t is not None and "cost" in oracle_t:
        result["_figures"]["points"]["Oracle (analysis only)"] = (
            oracle_t["cost"],
            oracle_t["micro_f1"],
        )
    return result


def to_json(result: Mapping[str, Any]) -> Dict[str, Any]:
    """The result without the private ``_``-prefixed entries, NaN and inf mapped to null."""
    return _jsonable({k: v for k, v in result.items() if not k.startswith("_")})


def iter_joined_records(result: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    """Per-sentence records in the package's JSONL traffic format (usable by ``ucci fit``)."""
    j: Joined = result["_joined"]
    split = result["_split"]
    for i in range(j.n):
        rec = _jsonable(
            {
                "id": j.ids[i],
                "split": str(split[i]),
                "u": float(j.u[i]),
                "small_correct": int(j.small_em[i]),
                "large_correct": int(j.large_em[i]),
                "entropy": float(j.entropy[i]),
                "max_prob": float(j.max_prob[i]),
                "small_tp": int(j.small_counts[i, 0]),
                "small_fp": int(j.small_counts[i, 1]),
                "small_fn": int(j.small_counts[i, 2]),
                "large_tp": int(j.large_counts[i, 0]),
                "large_fp": int(j.large_counts[i, 1]),
                "large_fn": int(j.large_counts[i, 2]),
                "latency_small_ms": float(j.small_latency_ms[i]),
                "latency_large_ms": float(j.large_latency_ms[i]),
                "source_split": j.source_split[i],
            }
        )
        yield {k: v for k, v in rec.items() if v is not None}
