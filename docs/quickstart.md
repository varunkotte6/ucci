# Quickstart

This page runs the whole method on simulated traffic. Every block runs as written, in order,
with numpy only. The data is synthetic: it shows the API, not a result.

## The data a router needs

For each logged query you need the small model's uncertainty \(u(x)\) (Eq. 4, computed from its
top-2 token log-probabilities; the [guides](guides/index.md) show how for each serving stack)
and whether each model's answer was right. The large model has to be run on the validation
queries too, because threshold selection compares actual outputs (Section 4.3).

```python
import numpy as np

rng = np.random.default_rng(0)
n = 10_000
u = rng.beta(2, 5, n)                                  # u(x) of the small model
p_wrong = 1 / (1 + np.exp(-10 * (u - 0.45)))           # true error curve, unknown in practice
small_ok = (rng.random(n) >= p_wrong).astype(float)    # 1 when the small model was right
large_ok = (rng.random(n) < 0.95).astype(float)        # 1 when the large model was right

# Disjoint calibration, validation and test splits: 30% / 20% / 50% (Section 6.1).
cal, val, test = np.split(rng.permutation(n), [3_000, 5_000])
```

## Step 1: calibrate

Fit the isotonic map \(g\) on the calibration split, with the error label \(e(x) = 1\) when the
small model was wrong (Section 4.2).

```python
from ucci import UCCIRouter, ece

router = UCCIRouter()            # c_small = 1.0, c_large = 3.02: the paper's normalized costs
router.calibrate(u[cal], 1 - small_ok[cal])

p_hat_test = router.error_probability(u[test])
print("ECE of raw u on test:        ", round(ece(u[test], 1 - small_ok[test]), 3))
print("ECE of calibrated p_hat on test:", round(ece(p_hat_test, 1 - small_ok[test]), 3))
```

## Step 2: choose the threshold

On the validation split, `choose_threshold` returns the cheapest threshold on the grid
\(\theta \in \{0, 0.005, \dots, 1\}\) whose accuracy reaches the target (Eq. 7).

```python
choice = router.choose_threshold(u[val], small_ok[val], large_ok[val], tau=0.90)
print(choice)   # ThresholdChoice(theta, cost, accuracy, escalation_rate) on validation
```

If no threshold reaches the target, `choose_threshold` raises `ucci.InfeasibleTargetError`
(a `ValueError`) with the best accuracy the grid reaches.

## Step 3: route and evaluate

`route` and `escalate` apply the policy \(\pi_\theta\) (Eq. 6) to new queries. On a labelled
test split, `evaluate` routes every query end to end and reports the actual cost and accuracy
(step 3 of Section 6.1).

```python
decisions = router.route(u[test])          # RouteResult(escalate, p_hat), arrays
print("escalated:", decisions.escalate.mean())
print("one query:", router.escalate(0.42))  # a bool for a scalar input

result = router.evaluate(u[test], small_ok[test], large_ok[test])
print(f"test cost {result.cost:.3f} vs {router.c_large} for large-only, accuracy {result.accuracy:.3f}")
```

A percentile bootstrap over test queries gives an interval for the saving (Section 6.2):

```python
from ucci import bootstrap_ci, policy_cost

esc = decisions.escalate
saving = lambda idx: 1 - policy_cost(esc[idx], router.c_small, router.c_large) / router.c_large
low, high = bootstrap_ci(saving, len(test), n_boot=1000, seed=0)
print(f"saving vs large-only: 95% CI [{low:.1%}, {high:.1%}]")
```

## A cost budget instead of an accuracy target

The bottom block of the paper's Table 2 compares methods at a matched cost budget. The budget
form returns the most accurate threshold whose mean cost stays within the budget:

```python
budget_router = UCCIRouter().calibrate(u[cal], 1 - small_ok[cal])
b = budget_router.choose_threshold_for_budget(u[val], small_ok[val], large_ok[val], budget=2.0)
print(b)
```

## Corpus-level metrics such as micro-F1

The paper's accuracy is micro-F1 over entities (Section 3), which is not a mean of per-query
scores. Pass a `metric` that scores the routed answers for an escalation mask;
`routed_micro_f1` builds one from per-query `(tp, fp, fn)` counts of each model:

```python
from ucci import routed_micro_f1

# Per-query entity counts (tp, fp, fn) of each model's answer on the validation split.
# Simulated: one gold entity per query, found when the model was right.
small_counts = np.stack([small_ok[val], 1 - small_ok[val], 1 - small_ok[val]], axis=1)
large_counts = np.stack([large_ok[val], 1 - large_ok[val], 1 - large_ok[val]], axis=1)

f1 = routed_micro_f1(small_counts, large_counts)
f1_router = UCCIRouter().calibrate(u[cal], 1 - small_ok[cal])
print(f1_router.choose_threshold(u[val], None, None, tau=0.90, metric=f1))
```

## The cost-accuracy frontier

`pareto_frontier` evaluates every grid threshold at once (the view of the paper's Figure 2);
`ucci.plotting.pareto_plot` draws it.

```python
from ucci import pareto_frontier

front = pareto_frontier(p_hat_test, small_ok[test], large_ok[test])
print(front.cost[front.efficient][:5], front.accuracy[front.efficient][:5])
```

## Save, load and reuse

A router is a small JSON file (the `ucci-router` format of `ucci.io`), read by
`UCCIRouter.load`, the `ucci` command line and the Rust crate.

```python
import os
import tempfile

path = os.path.join(tempfile.mkdtemp(), "router.json")
router.save(path)
again = UCCIRouter.load(path)
assert (again.route(u[test]).escalate == decisions.escalate).all()
print(again)
```

Next: compute \(u(x)\) from your serving stack ([guides](guides/index.md)), log traffic
([live traffic](guides/live_traffic.md)), and fit from the command line ([CLI](guides/cli.md)).
