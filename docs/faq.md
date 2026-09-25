# FAQ

## When do Theorem 1's assumptions fail?

Theorem 1 (Section 5) makes three assumptions. Each can fail in practice, and each failure has
a check and a remedy.

**The forecast is miscalibrated (assumption iii).** The theorem needs
\(\hat p(x) = P(e(x) = 1 \mid u(x))\). With too little calibration data, labels from a different
distribution, or a calibration split that overlaps the evaluation data, \(\hat p\) is off and the
threshold no longer means what it says.

- *Check:* ECE and the reliability table on held-out labelled data (`ucci report`,
  `ucci.ece`, `ucci.reliability_table`), not on the calibration split itself: an isotonic fit
  matches the bin frequencies of its own data, so in-sample ECE is near zero by construction.
- *Remedy:* more calibration data (Proposition 2 bounds the expected ECE by \(O(n^{-1/3})\)),
  disjoint splits, and labels from the traffic you route.

**The distribution shifts.** A new prompt, a new small-model version, a new user population
or a new mix of query types changes the relation between \(u(x)\) and the error rate. The paper
fits \(g\) once, on a held-out batch, and names this as a limitation (Section 7, "Static
calibration").

- *Check:* `ucci.online.CalibrationMonitor` runs a calibration test on a sliding window of
  fresh labels and raises a drift flag.
- *Remedy:* refit \(g\) on recent labels (`ucci.online.RecalibratingRouter` does it on a
  schedule, keeping \(\theta\) fixed as an error probability) and re-select \(\theta\) on fresh
  validation data when the accuracy target must be re-certified. The monitor and the
  recalibrating router are an extension, not in the paper.

**Margins are heavy-tailed (Section 6.3, Appendix B.5).** The optimality argument needs
\(\hat p(x)\) to be informative about each query's error rate. If a small fraction of queries sits
in the body of the \(u(x)\) distribution while having a high true error probability, calibration
on a finite sample averages them away, and a threshold escalates fewer of them than an oracle
would. The paper did not measure margin-tail statistics on its workload and does not claim a
bound on the size of the effect.

- *Check:* the gap between UCCI and the label-dependent `Oracle` row of
  `ucci.baselines.compare_routers`, broken down by query type where you have one (the paper
  looks at entity types, Table 4).
- *Remedy:* a signal that separates those queries (the paper's signal ablation compares token
  margin, entropy and max probability; `ucci.baselines.ablation_methods`), or routing rules per
  query type.

**The large model's accuracy depends on which queries reach it (assumption ii).** The theorem
assumes the large model has the same accuracy \(\alpha_\ell\) on escalated queries as on all
queries. If the queries the small model finds hard are also hard for the large model, escalated
queries gain less than Eq. 8 predicts. The paper measures 0.928 micro-F1 on escalated queries
against 0.932 overall on its workload (Section 6.3).

- *Check:* `ucci evaluate` reports `assumption_ii`: the large model's accuracy on the escalated
  queries against all queries of the split, with the gap.
- *Consequence:* the threshold policy is then no longer guaranteed to be globally optimal; it
  remains a calibrated heuristic (Section 7), and threshold selection itself stays valid,
  because Eq. 7 uses the large model's actual outputs on the validation queries.

**Cost does not grow with the escalation (assumption i).** The theorem needs \(c_\ell > c_s\).
Under the routing cost model the code refuses to select a threshold otherwise.

**The accuracy metric is not the calibration event.** The proof identifies the small model's
expected accuracy given \(u(x)\) with \(1 - \hat p(x)\), which is exact when accuracy is the 0/1
event that \(g\) is calibrated on (exact match in the paper). With micro-F1 as the accuracy (as
in the paper's evaluation), a partly right answer scores between 0 and 1 and the identity is an
approximation. Selection on validation (Eq. 7) uses the actual metric either way; pass
`metric=ucci.routed_micro_f1(...)` to select on corpus micro-F1.

## Which cost model should I use?

- `cost_model="routing"` (the paper's): a kept query costs \(c_s\), an escalated one \(c_\ell\). Use it
  when the cost that matters is incurred by the model that answers, for example when \(c_s\) and
  \(c_\ell\) are the measured end-to-end latencies of each model, or when escalation replaces the
  small call.
- `cost_model="sequential"`: an escalated query costs \(c_s + c_\ell\). Use it when the small model
  always runs first to produce \(u(x)\) and its cost is paid for every query, for example with
  per-token API prices.

The chosen \(\theta\) is the same under both (every escalation adds the same marginal cost); only
the reported cost changes. What matters is the ratio \(c_\ell / c_s\) in the units you care about:
latency (the paper uses the measured H100 latency ratio 3.02), dollars, energy or throughput.
Section 7 of the paper notes that each requires re-deriving \(c_s\) and \(c_\ell\) but changes neither
the calibration step nor the optimality argument. `ucci fit --cost-from-latency` sets the ratio
from logged latencies.

## Why does ECE on the calibration split come out near zero?

Isotonic regression matches the observed error frequency of its own data within each level set,
so in-sample ECE is zero up to round-off. Report ECE on held-out data as well: `ucci fit` prints
it on the validation split, and `ucci report --split test` gives it with a bootstrap interval.

## Why is my accuracy target infeasible?

`select_threshold` raises `InfeasibleTargetError` when no threshold on the grid reaches
\(\tau\) on the validation split; the message gives the best accuracy the grid reaches. Usually
\(\tau\) is above the large model's own validation accuracy, or the large model is not better than
the small one on this split. The CLI exits with code 4 and prints the always-large accuracy.

## Do I need scikit-learn?

No. The isotonic fit is implemented in numpy and reproduces scikit-learn's
`IsotonicRegression` (with `out_of_bounds="clip"`); scikit-learn is used only by the tests
that check this.

## Can I use a signal other than the token margin?

Yes. `UCCIRouter` calibrates any score where larger means "more likely wrong". The paper
reports that the token margin gave the best cost-accuracy curve among token margin,
predictive entropy and max probability on its workload (Section 6.3);
`ucci.baselines.ablation_methods` runs the same comparison on yours.

## Does the router work with sampling instead of greedy decoding?

The signal is defined under greedy decoding (Section 4.1), where the generated token is the
top-1 candidate at every step. Under sampling the reported top-2 can be (sampled token,
top-1), which changes \(u(x)\). The adapters count non-greedy steps
(`TokenSignals.n_non_greedy`) and raise with `require_greedy=True`.
