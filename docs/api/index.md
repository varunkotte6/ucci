# API reference

Everything below is generated from the docstrings, which cite the paper's sections and
equations (arXiv v1 numbering). The names in the first table are importable from `ucci`
directly; the other modules are imported as submodules (`import ucci.baselines`) and load
their optional dependencies only when used.

| Module | Contents | Paper |
|---|---|---|
| [`ucci.router`](router.md) | `UCCIRouter`, `RouteResult` | Sections 4 and 6.1 |
| [`ucci.signal`](signal.md) | `token_margin_uncertainty`, `margins_from_top2`, `uncertainty_from_margins`, `uncertainty_from_probs`, `uncertainty_from_logprobs`, `batch_uncertainty`, `top2_from_logprobs`, `from_openai_logprobs`, `from_vllm_logprobs` | Section 4.1, Eq. 4 |
| [`ucci.calibration`](calibration.md) | `IsotonicCalibrator`, `pav` | Section 4.2, Eq. 5 |
| [`ucci.policy`](policy.md) | `escalate`, `select_threshold`, `select_threshold_for_budget`, `evaluate`, `pareto_frontier`, `policy_cost`, `policy_accuracy`, `make_grid`, `ThresholdChoice`, `ParetoFrontier`, `InfeasibleTargetError`, `DEFAULT_GRID`, `DEFAULT_GRID_STEP`, `DEFAULT_COST_SMALL`, `DEFAULT_COST_LARGE` | Section 4.3, Eqs. 6 and 7, Section 5, Tables 2 and 3 |
| [`ucci.metrics`](metrics.md) | `ece`, `reliability_table`, `ReliabilityRow`, `brier_score`, `bootstrap_ci`, `ConfidenceInterval`, `micro_f1`, `routed_micro_f1`, `RoutedMicroF1` | Section 6.2, Figure 1 |
| [`ucci.io`](io.md) | `save_router`, `load_router` and the `ucci-router` JSON format | |

| Submodule | Contents |
|---|---|
| [`ucci.baselines`](baselines.md) | the Table 2 comparators, the ablations and `compare_routers` (Sections 6.1 and 6.3) |
| [`ucci.integrations`](integrations.md) | serving adapters (OpenAI-compatible, vLLM, transformers, llama.cpp) and the cascade runner |
| [`ucci.online`](online.md) | calibration monitoring and recalibration (extension, not in the paper) |
| [`ucci.plotting`](plotting.md) | reliability diagrams and Pareto plots (matplotlib) |
| [`ucci.cli`](cli.md) | the `ucci` command line |
