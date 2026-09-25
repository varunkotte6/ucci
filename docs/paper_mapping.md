# Paper to code

This page maps every element of the paper (arXiv:2605.18796 v1, whose numbering it uses) to
the code that implements it, lists every implementation choice the paper leaves open, and
records two places where the code follows the mathematics rather than a sentence of the
paper. Where the code goes beyond the paper, it says so: extensions are labelled "extension,
not in the paper" in their docstrings and are never on by default.

## Sections 3 and 4: formulation and method

| Paper | What it says | Code |
|---|---|---|
| Section 3, Eq. 1 | Routing policy \(\pi : \mathcal{X} \to \{s, \ell\}\) returns \(f_s(x)\) or \(f_\ell(x)\) | Escalation masks throughout: `True` means \(\pi(x) = \ell\) (`ucci.escalate`) |
| Section 3 | Per-query cost \(C_\pi(x) = c_s 1\{\pi(x)=s\} + c_\ell 1\{\pi(x)=\ell\}\) | `ucci.policy_cost(..., cost_model="routing")`; mean cost \(c_s(1-r) + c_\ell r\) at escalation rate \(r\) |
| Section 3, Eqs. 2 and 3 | Minimize expected cost subject to expected accuracy \(\ge \tau\) | `ucci.select_threshold` over the threshold family; `ucci.baselines.Oracle` gives the label-dependent optimum for analysis (extension) |
| Section 3 | \(\mathrm{Acc}\) is micro-averaged F1 | `ucci.micro_f1`; `ucci.routed_micro_f1` scores routed answers for threshold selection (`metric=`) |
| Section 4.1, Eq. 4 | \(u(x) = 1 - \frac{1}{T}\sum_t (p_{t,1} - p_{t,2})\) under greedy decoding | `ucci.token_margin_uncertainty`, `ucci.margins_from_top2`, `ucci.uncertainty_from_margins`, `ucci.uncertainty_from_probs`, `ucci.uncertainty_from_logprobs`, `ucci.batch_uncertainty`, `ucci.top2_from_logprobs`, `ucci.from_openai_logprobs`, `ucci.from_vllm_logprobs`; serving adapters in `ucci.integrations` |
| Section 4.1 | Greedy decoding so \(p_{t,1}, p_{t,2}\) are defined | `ucci.integrations.vllm.greedy_sampling_params` fixes temperature 0; `generate_with_signals` fixes `do_sample=False, num_beams=1`; every adapter reports `TokenSignals.n_non_greedy` and accepts `require_greedy=True` |
| Section 4.2 | \(e(x) = 1\) when the small output is wrong (exact match of the JSON across all fields) | Supplied by the user: `UCCIRouter.calibrate(u, e)`; CLI `e = 1 - small_correct`; `SplitData.error()` defaults to `small_score < 1`; the CoNLL-2003 benchmark uses JSON exact match |
| Section 4.2, Eq. 5 | Monotone \(g(u) \approx P(e = 1 \mid u)\) fit by isotonic regression | `ucci.IsotonicCalibrator`, `ucci.pav`, `UCCIRouter.calibrate` |
| Section 4.3 | \(\hat p(x) = g(u(x))\) | `IsotonicCalibrator.predict`, `UCCIRouter.error_probability` |
| Section 4.3, Eq. 6 | Keep if \(\hat p \le \theta\), escalate if \(\hat p > \theta\) | `ucci.escalate`, `UCCIRouter.escalate`, `UCCIRouter.route` |
| Section 4.3, Eq. 7 | \(\theta^* = \arg\min \widehat{\mathrm{Cost}}\) s.t. \(\widehat{\mathrm{Acc}} \ge \tau\) on validation, actual costs and outputs | `ucci.select_threshold`, `UCCIRouter.choose_threshold` |

## Section 5 and Appendix A: theory

| Paper | What it says | Code |
|---|---|---|
| Theorem 1, assumption (i) | \(c_\ell > c_s\) | Threshold selection under the routing cost model raises `ValueError` otherwise |
| Theorem 1, assumption (ii) | Large-model accuracy invariant to routing | Checked on data by `ucci evaluate` (`assumption_ii`: large-model accuracy on escalated vs all queries) and by the CoNLL-2003 benchmark |
| Theorem 1, assumption (iii) | Calibrated \(\hat p\) | Measured by `ucci.ece`, `ucci.reliability_table`, `ucci report`; monitored online by `ucci.online` (extension) |
| Theorem 1, conclusion | A threshold on \(\hat p\) is cost-optimal among policies on \(u(x)\), up to level-set tie-breaking | Brute force over all \(2^n\) routing subsets in `tests/test_core_policy.py` (`test_theorem1_threshold_is_cost_optimal_among_all_policies`, `test_theorem1_tie_breaking_on_level_sets`, `test_budget_form_threshold_is_optimal_among_all_policies`) |
| Appendix A.1, Eq. 8 | Marginal gain \(\alpha_\ell - 1 + \hat p(x)\) at cost \(c_\ell - c_s\) | Stated in the `ucci.policy` module docstring; the reason the selected \(\theta\) does not depend on the cost values |
| Appendix A.1, Eq. 9 | Greedy allocation as a threshold | `select_threshold` returns the largest feasible \(\theta\) |
| Proposition 2, Appendix A.2, Eqs. 10 and 11 | \(\mathbb{E}[\mathrm{ECE}(\hat g)] = O(n^{-1/3})\) | A statement about the estimator, not an algorithm; `ucci.ece` measures ECE on held-out data |

## Section 6: experiments

| Paper | What it says | Code |
|---|---|---|
| Section 6.1, splits | Disjoint calibration / validation / test, 30% / 20% / 50% | CLI defaults `--cal-frac 0.3 --val-frac 0.2`; benchmark splits; the three `SplitData` of `compare_routers` |
| Section 6.1, measured costs | \(c_s = 1.0\), \(c_\ell = 3.02\) from mean latency 47.2 ms and 142.3 ms over 100 queries | `ucci.DEFAULT_COST_SMALL`, `ucci.DEFAULT_COST_LARGE`; `ucci fit --cost-from-latency`; `CascadeResult.latency_small_ms` and `latency_large_ms`; `benchmarks/conll2003/latency.py` |
| Section 6.1, protocol | Fit \(g\) on calibration, select \(\theta^*\) on validation, route every test query end to end | `UCCIRouter.calibrate`, `choose_threshold`, `evaluate`; `ucci.evaluate`; `ucci.baselines.compare_routers`; `ucci fit` then `ucci evaluate` |
| Section 6.1, baselines | Always-small, always-large | `ucci.baselines.AlwaysSmall`, `AlwaysLarge`; `always_small` and `always_large` in `ucci evaluate` |
| Section 6.1, baselines | Entropy threshold: route by uncalibrated mean token entropy | `ucci.baselines.EntropyThresholdRouter`, signal `ucci.baselines.mean_token_entropy` |
| Section 6.1, baselines | Split conformal on "small model is correct" with raw \(u(x)\) as nonconformity score, \(\alpha^*\) chosen on validation, escalate \(u(x) > \alpha^*\) | `ucci.baselines.SplitConformalRouter` |
| Section 6.1, baselines | FrugalGPT-style: a confidence threshold tuned on validation to meet the target | `ucci.baselines.FrugalGPTStyleRouter` |
| Table 1 | Single-model micro-F1 and cost | `AlwaysSmall`, `AlwaysLarge`; `ucci evaluate` reference rows |
| Table 2, top block | Methods at the F1 target | `compare_routers(..., tau=...)`, `format_comparison`; the "\(\Delta\)F1 vs target" column is `ComparisonRow.delta_vs_target` |
| Table 2, bottom block | Methods at a matched cost budget (2.00) | `compare_routers(..., budget=...)`, `ucci.select_threshold_for_budget`, `UCCIRouter.choose_threshold_for_budget`, `ucci fit --budget` |
| Table 3 | The same routing re-costed at \(c_\ell / c_s\) = 3.02, 5, 10 | `ucci.policy_cost` with other costs; `ucci evaluate --c-large`; the benchmark's cost-ratio table |
| Table 4 | Per-entity micro-F1 | Benchmark only (`benchmarks/conll2003`, per entity type) |
| Figure 1, Section 6.2 | Reliability diagram and ECE, raw vs calibrated, with a bootstrap CI | `ucci.reliability_table`, `ucci.ece`, `ucci.plotting.reliability_diagram`, `ucci report` |
| Figure 2 | Cost-accuracy Pareto frontier, end to end | `ucci.pareto_frontier`, `ucci.plotting.pareto_plot` |
| Section 6.2 | 95% CIs by bootstrap over queries | `ucci.bootstrap_ci`; `ucci evaluate --bootstrap`, `ucci report --bootstrap` |
| Section 6.3, calibration method | Isotonic vs temperature scaling vs uncalibrated | `ucci.baselines.TemperatureScalingCalibrator`, `IdentityCalibrator`, `CalibratedThresholdRouter`, `ablation_methods` |
| Section 6.3, signal | Token margin vs predictive entropy vs max probability | `ucci.baselines.mean_token_entropy`, `mean_max_prob`; "Isotonic on entropy" and "Isotonic on max prob" in `ablation_methods` |
| Section 6.3, assumption (ii) | Large-model F1 on escalated queries vs all | `assumption_ii` in `ucci evaluate`; the benchmark |
| Section 6.3, falsification regime | Heavy-tailed margins | [FAQ](faq.md#when-do-theorem-1s-assumptions-fail); `Oracle` rows in `compare_routers` show the gap to the label-dependent optimum |

## Section 7 and Appendix B

| Paper | What it says | Code |
|---|---|---|
| Section 7, "Scope" | Replicate on a public NER benchmark such as CoNLL-2003 with another model family | `benchmarks/conll2003` (Qwen2.5 1.5B and 7B Instruct), see [Benchmarks](benchmarks.md) |
| Section 7, "Static calibration" | Streaming deployments with shift need online recalibration | `ucci.online.CalibrationMonitor`, `ucci.online.RecalibratingRouter` (extension, not in the paper) |
| Section 7, "Cost model" | Dollar cost, throughput or energy need re-derived \(c_s, c_\ell\) | `c_small`, `c_large` in every API; `cost_model="sequential"` for cascades that always run the small model first |
| Appendix B.1 | Prompt template (photo search entity fields) | The benchmark uses the same structure with the CoNLL-2003 fields `PER`, `ORG`, `LOC`, `MISC` |
| Appendix B.2 | Temperature 0, at most 256 generated tokens, vLLM 0.4.2, H100 80GB | `max_tokens=256` in `greedy_sampling_params`, `max_new_tokens=256` in `generate_with_signals`; the adapters were checked against current vLLM source rather than 0.4.2 |
| Appendix B.2 | Isotonic regression "as implemented in standard open-source libraries (default settings)" | `IsotonicCalibrator` reproduces scikit-learn's `IsotonicRegression` except for clipping (below) |
| Appendix B.3 | Latency averaged over 100 queries | `benchmarks/conll2003/latency.py` (100 queries per model at batch size 1, 3 untimed warm-up queries) |
| Appendix B.4 | Isotonic vs temperature scaling vs uncalibrated routing | The ablations above |
| Appendix B.5 | Heavy-tailed margins | [FAQ](faq.md#when-do-theorem-1s-assumptions-fail) |

## Implementation choices

The paper does not specify the following. Each is a deliberate choice, documented where it is
implemented.

**Signal.**

- *Token convention.* \(u(x)\) averages over every generated content token and excludes padding
  and the terminating EOS or stop token. Serving stacks differ in whether they report that
  token:

    | Stack | Reports the stop token? | What the code does |
    |---|---|---|
    | OpenAI API | No | Nothing to drop |
    | vLLM (offline and OpenAI-compatible server) | Yes, it keeps the token that ended generation | `ucci.integrations.vllm` drops it when `finish_reason == "stop"` and `stop_reason` is None or an int; for the server pass `drop_stop_token=True` |
    | llama.cpp server | Yes, the EOS entry is appended before the EOS is detected | `ucci.integrations.llamacpp` drops it when `stop_type == "eos"` (`stopped_eos` on old servers); for `/v1` endpoints pass `drop_stop_token=True` |
    | Hugging Face `generate` | Logits exist for every step | Cut at the first EOS; a trailing run of the pad id is trimmed only when `pad_token_id` is passed (`generate_with_signals` does so when `stop_strings` or `stopping_criteria` can end rows without EOS) |

    The low-level functions in `ucci.signal` treat their input as content tokens only.
- *Empty generations.* Eq. 4 is undefined for \(T = 0\), so every function raises by default.
  `batch_uncertainty(..., empty_value=1.0)` marks empty generations as maximally uncertain
  (extension, not in the paper); the transformers adapter offers `on_empty="nan"`.
- *Probability round-off.* Probabilities up to \(1 + 10^{-9}\) (from `exp` of a log-probability) are
  capped at 1; log-probabilities above \(\log(1 + 10^{-9})\) are rejected.
- *Entropy from top-\(k\) log-probabilities.* The paper does not say how its entropy baseline was
  computed. `ucci.baselines.mean_token_entropy` computes the entropy of the renormalized top-\(k\)
  distribution at each position. The serving adapters report `TokenSignals.mean_entropy` as the
  truncated sum \(-\sum_{j \le k} p_j \log p_j\), a lower bound on the full entropy, and the
  renormalized version with `renormalize_entropy=True`; `entropy_support` records which.
  With full logits (transformers) both are the exact entropy.
- *Max probability.* `mean_max_prob` uses the reported top-1 log-probability without
  renormalization (exact even with top-\(k\) output); `renormalize=True` renormalizes.

**Calibration.**

- *Out-of-range inputs.* Predictions are clipped to the end values outside the calibration
  range. scikit-learn's default (`out_of_bounds="nan"`) returns NaN there; clipping keeps
  \(\hat p\) defined for every query.
- *scikit-learn parity.* Zero-weight points are dropped, \(u\) values within \(10^{-15}\) of the first
  value of their group are pooled (scikit-learn's `_make_unique` rule), and interior knots equal
  to both neighbours are trimmed. Knots equal scikit-learn's `X_thresholds_` and predictions
  agree to \(10^{-12}\) (tested on scikit-learn 1.9.1 and 0.24.2).
- *Soft labels.* \(e_i \in [0, 1]\) is accepted, and sample weights are supported.

**Threshold selection.**

- *Grid.* \(\theta \in \{0, 0.005, \dots, 1\}\) (201 values, `ucci.DEFAULT_GRID`). arXiv v1 does not
  print the resolution; 0.005 is the value stated in the appendix of the paper's workshop
  version (Forecast@ICML 2026).
- *Tie-breaking.* Accuracy target: lowest cost, then highest accuracy, then largest \(\theta\).
  Budget: highest accuracy, then lowest cost, then largest \(\theta\). Comparisons are exact, with
  no tolerance; with 0/1 scores every accuracy is an exact count divided by \(n\).
- *Cost formulas.* Routing \(c_s(1-r) + c_\ell r\); sequential \(c_s + c_\ell r\). The same formula is
  used everywhere, so equal masks give bit-identical costs.
- *Sequential cost model.* Offered as an option (`cost_model="sequential"`), for cascades where
  the small model always runs to produce \(u(x)\). The paper's cost model is `"routing"`.
- *Infeasible targets* raise `ucci.InfeasibleTargetError` (a `ValueError`) naming the best
  accuracy (or the cheapest cost) on the grid; the CLI exits with code 4.

**Metrics.**

- *ECE binning.* The paper does not state it. `ucci.ece` defaults to 10 equal-width bins,
  right-closed with the first bin also containing 0 (as scikit-learn's `calibration_curve`);
  `strategy="quantile"` gives equal-count bins. `ucci.plotting.reliability_diagram` uses
  deciles of the forecast by default.
- *Bootstrap.* Percentile intervals with numpy's linear quantiles, `default_rng(seed=0)`, 1000
  resamples by default. arXiv v1 reports bootstrap CIs over queries without the count; 1000 is
  the value stated in the workshop version's appendix.
- *Micro-F1 with nothing to score* (\(2\,TP + FP + FN = 0\)) returns `zero_division`, 0.0 by default,
  as scikit-learn's `f1_score` does.

**Baselines** (`ucci.baselines`).

- *Raw-score thresholds* (entropy, FrugalGPT-style). The candidate thresholds are every distinct
  validation score plus \(\pm\infty\), so the search is exact. They are passed to
  `ucci.select_threshold` through an order-preserving relabelling onto \([0, 1]\), so masks, costs,
  accuracies and tie-breaking are identical to UCCI's.
- *Split conformal.* Nonconformity scores are the raw \(u\) of the small-correct calibration
  queries; the threshold for miscoverage \(\delta\) is the \(k\)-th smallest with
  \(k = \lceil (n+1)(1-\delta) \rceil\) (infinite when \(k > n\)); \(\delta\) is searched on 0.005 to 0.995 in
  steps of 0.005 and \(\delta^*\) is selected on validation by the Eq. 7 rules; ties go to the smallest
  \(\delta\). \(\delta\) is read as the shortest decimal that round-trips, so \(\delta = 0.3\) means exactly
  3/10 in the ceiling. The coverage guarantee (Angelopoulos and Bates, 2021) holds for each
  fixed \(\delta\), not for the data-selected \(\delta^*\). Sample weights are rejected.
- *FrugalGPT-style confidence.* The paper does not say which confidence its baseline used.
  `compare_routers` uses the `"confidence"` signal when present and `"max_prob"` otherwise, and
  records which in each row. This is the paper's threshold baseline, not FrugalGPT's learned
  scorer and model list (Chen, Zaharia and Zou, 2023).
- *Temperature scaling.* \(\hat p = \sigma(\mathrm{logit}(\mathrm{clip}(u, 10^{-6}, 1 - 10^{-6})) / T)\), \(T\) fit by exact
  one-dimensional minimization of the binary NLL on \([10^{-3}, 10^3]\); a `RuntimeWarning` is raised
  when the optimum lies at a bound.
- *Calibration labels.* `SplitData.error()` defaults to `small_score < 1` (for per-query F1: "not
  every field right"); `small_error` overrides it.
- *Extensions, not in the paper:* `Oracle` (label-dependent lower bound on cost, exact for the
  mean per-query metric) and `PlattCalibrator` (logistic calibration; `l2 = 1/C` matches
  scikit-learn's `LogisticRegression(C=C)`).

**Command line** (`ucci.cli`).

- *Random split.* Records are ordered by the SHA-256 hex digest of `"<seed>:<id>"` and cut at
  \(\lfloor \text{frac} \cdot n + 0.5 \rfloor\): 30% / 20% / 50% with seed 0 by default. The split does not
  depend on record order, platform or numpy version; a `split` field on every record overrides
  it.
- *Accuracy* is the mean per-query score of the returned answers; corpus micro-F1 needs the
  Python API with `metric=ucci.routed_micro_f1(...)`.
- *Provenance.* `ucci fit` writes an extra `fit` object into the router file (data digests,
  split, objective, validation results). Readers ignore unknown keys.

**Online monitoring** (`ucci.online`, extension, not in the paper).

- Drift tests on a sliding window: a binomial calibration-in-the-large z test and
  Spiegelhalter's z (Spiegelhalter, 1986), two-sided, normal approximation.
- `RecalibratingRouter` keeps \(\theta\) fixed on the probability scale and refits the isotonic map
  on the most recent labels every `refit_every` labels.

**Router file and Rust crate.**

- The `ucci-router` JSON format (version 1) is shared by Python and Rust; floats are written in
  shortest round-trip form, so a loaded router predicts bit-identically. The Rust crate matches
  the Python core bit for bit on every golden vector in `tests/golden`.

## References

- A. N. Angelopoulos and S. Bates. A gentle introduction to conformal prediction and
  distribution-free uncertainty quantification. arXiv:[2107.07511](https://arxiv.org/abs/2107.07511), 2021.
- L. Chen, M. Zaharia and J. Zou. FrugalGPT: How to use large language models while reducing
  cost and improving performance. arXiv:[2305.05176](https://arxiv.org/abs/2305.05176), 2023.
- C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger. On calibration of modern neural networks.
  *ICML*, 2017. arXiv:[1706.04599](https://arxiv.org/abs/1706.04599).
- J. C. Platt. Probabilistic outputs for support vector machines and comparisons to regularized
  likelihood methods. *Advances in Large Margin Classifiers*, 1999.
- D. J. Spiegelhalter. Probabilistic prediction in patient management and clinical trials.
  *Statistics in Medicine* 5(5):421-433, 1986. doi:[10.1002/sim.4780050506](https://doi.org/10.1002/sim.4780050506).
