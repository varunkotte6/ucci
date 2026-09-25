# Baselines and the comparison protocol

`ucci.baselines` implements every routing method the paper compares UCCI against (Section 6.1,
Table 2) and the ablations of Section 6.3 and Appendix B.4, and runs them under the paper's
evaluation protocol. Numpy only.

## The protocol

`compare_routers` runs, for every method (Section 6.1, "Cascade evaluation methodology"):

1. fit on the calibration split;
2. select the operating point on the validation split, where both models have been run, with
   the Section 4.3 rules (cheapest threshold meeting \(\tau\); or, in budget form, the most accurate
   within the budget);
3. route every test query end to end with the selected rule, taking the actual output and cost
   of the model it chooses.

Every threshold rule, including those on raw scores, is selected by the same core functions as
UCCI (`ucci.select_threshold`, `ucci.select_threshold_for_budget`), so all methods share UCCI's
tie-breaking and cost formulas exactly.

## Methods

| Row | Class | What it does |
|---|---|---|
| UCCI | `ucci.UCCIRouter` | isotonic \(g\) on \(u\), threshold on \(\hat p\) (Eq. 6) |
| Conformal prediction | `SplitConformalRouter` | split conformal on "small model is correct" with raw \(u\) as the nonconformity score; \(\alpha^* = q(\delta^*)\) with \(\delta^*\) chosen on validation; escalate when \(u > \alpha^*\) |
| FrugalGPT-style | `FrugalGPTStyleRouter` | a confidence threshold tuned on validation to meet the target (escalate when confidence is below it) |
| Entropy threshold | `EntropyThresholdRouter` | a threshold on uncalibrated mean token entropy |
| Large-only, Small-only | `AlwaysLarge`, `AlwaysSmall` | single-model anchors (Table 1) |
| Temperature scaling (ablation) | `CalibratedThresholdRouter(TemperatureScalingCalibrator())` | \(\hat p = \sigma(\mathrm{logit}(u)/T)\), Appendix B.4 |
| Uncalibrated u (ablation) | `CalibratedThresholdRouter(IdentityCalibrator())` | \(\hat p = u\), Appendix B.4 |
| Isotonic on entropy, on max prob (ablations) | `CalibratedThresholdRouter(IsotonicCalibrator())` | the UCCI pipeline on the other two signals, Section 6.3 |
| Platt scaling | `PlattCalibrator` | extension, not in the paper |
| Oracle | `Oracle` | label-dependent lower bound on cost; extension, for analysis only |

The FrugalGPT-style row is the paper's threshold baseline, not a reimplementation of FrugalGPT
(Chen, Zaharia and Zou, 2023), which also learns an answer-scoring function and the model
cascade. The paper does not say which confidence it used; `compare_routers` uses a
`"confidence"` signal when the splits have one and the mean top-1 probability (`"max_prob"`)
otherwise, and records which.

## Signals

Each split is a `SplitData` with named per-query signals and both models' per-query scores.
The built-in methods read `"u"` (Eq. 4), `"entropy"` (mean token entropy), `"max_prob"` (mean
top-1 probability) and, if present, `"confidence"`. The serving adapters return all three
(`TokenSignals.u`, `.mean_entropy`, `.mean_max_prob`); from raw top-\(k\) log-probabilities,
`mean_token_entropy` and `mean_max_prob` compute the last two.

```python
import numpy as np
from ucci.baselines import SplitData, compare_routers, format_comparison

# Simulated traffic with three signals (synthetic: shows the API, not a result).
rng = np.random.default_rng(1)
n = 12_000
u = rng.beta(2, 5, n)
entropy = 2.5 * u + rng.normal(0, 0.15, n).clip(-0.2, None)
max_prob = (1 - 0.6 * u + rng.normal(0, 0.05, n)).clip(0, 1)
small_ok = (rng.random(n) >= 1 / (1 + np.exp(-10 * (u - 0.45)))).astype(float)
large_ok = (rng.random(n) < 0.95).astype(float)
parts = np.split(rng.permutation(n), [int(0.3 * n), int(0.5 * n)])   # 30 / 20 / 50

def split(i):
    signals = {"u": u[i], "entropy": entropy[i], "max_prob": max_prob[i]}
    return SplitData(signals, small_ok[i], large_ok[i])

cal, val, test = (split(i) for i in parts)
```

## Table 2 on your data

The top block of Table 2 compares methods at an accuracy target, the bottom block at a matched
cost budget:

```python
rows = compare_routers(cal, val, test, tau=0.90, include_ablations=True)
print(format_comparison(rows))

rows = compare_routers(cal, val, test, budget=2.0, tau=0.90)   # tau only for "vs target"
print(format_comparison(rows))
```

Each `ComparisonRow` holds the selected threshold (in the method's own units), test cost,
accuracy and escalation rate, `delta_vs_target` (the "\(\Delta\)F1 vs target" column), the
validation operating point, whether it met the constraint on validation, and fitted
parameters (\(\delta^*\), \(T\), Platt's \(a\) and \(b\)). A method that cannot meet the target on
validation gets a row of NaN with the reason in `note`.

## Micro-F1 and other corpus metrics

Pass `metric=` per split for a corpus-level accuracy such as micro-F1 over entities; every
method then selects and is evaluated on it:

```python
from ucci import routed_micro_f1

def counts(ok):   # per-query (tp, fp, fn); simulated: one gold entity per query
    return np.stack([ok, 1 - ok, 1 - ok], axis=1)

def split_f1(i):
    signals = {"u": u[i], "entropy": entropy[i], "max_prob": max_prob[i]}
    f1 = routed_micro_f1(counts(small_ok[i]), counts(large_ok[i]))
    return SplitData(signals, small_ok[i], large_ok[i], metric=f1)

cal, val, test = (split_f1(i) for i in parts)
print(format_comparison(compare_routers(cal, val, test, tau=0.90)))
```

## Custom methods

`MethodSpec(name, signal, factory, transform=None, note="")` adds any router with the
`calibrate` / `choose_threshold` / `escalate` interface; `table2_methods()` and
`ablation_methods()` return the built-in lists to extend. Anything on a new signal is a new key
in `SplitData.signals`.

The CoNLL-2003 replication runs exactly this protocol on real model outputs
([Benchmarks](../benchmarks.md)).

## References

- A. N. Angelopoulos and S. Bates. A gentle introduction to conformal prediction and
  distribution-free uncertainty quantification. arXiv:[2107.07511](https://arxiv.org/abs/2107.07511), 2021.
- L. Chen, M. Zaharia and J. Zou. FrugalGPT: How to use large language models while reducing
  cost and improving performance. arXiv:[2305.05176](https://arxiv.org/abs/2305.05176), 2023.
- C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger. On calibration of modern neural networks.
  *ICML*, 2017. arXiv:[1706.04599](https://arxiv.org/abs/1706.04599).
- J. C. Platt. Probabilistic outputs for support vector machines and comparisons to regularized
  likelihood methods. *Advances in Large Margin Classifiers*, 1999.

API reference: [`ucci.baselines`](../api/baselines.md).
