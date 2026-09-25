# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The `ucci-router` JSON format has
its own version number (currently 1); readers ignore unknown keys, so new fields do not break
old readers.

## [Unreleased]

## [0.1.0]

First public release: the reference implementation of Varun Kotte, "UCCI: Calibrated
Uncertainty for Cost-Optimal LLM Cascade Routing", arXiv:2605.18796 (2026).

### Added

- **Core** (numpy only, Python 3.9 or newer, fully typed):
    - `ucci.signal`: token-margin uncertainty u(x) (Section 4.1, Eq. 4) from probability pairs,
      probabilities, log-probabilities, top-k candidates, OpenAI-compatible and vLLM
      outputs, and ragged or padded batches.
    - `ucci.calibration`: `IsotonicCalibrator` and `pav` (Section 4.2), reproducing
      scikit-learn's `IsotonicRegression` knot for knot, with clipping outside the calibration
      range.
    - `ucci.policy`: the threshold policy (Eq. 6), threshold selection at an accuracy target
      (Eq. 7) and at a cost budget (Table 2, bottom block), the Pareto frontier, and the routing
      and sequential cost models. Default costs are the paper's normalized 1.0 and 3.02.
    - `ucci.router`: `UCCIRouter`, the three steps and the evaluation protocol of Section 6.1.
    - `ucci.metrics`: ECE, reliability tables, Brier score, percentile bootstrap intervals,
      micro-F1 and routed micro-F1 for threshold selection on corpus metrics.
    - `ucci.io`: the `ucci-router` JSON format (version 1), shared with the Rust crate.
- **Baselines** (`ucci.baselines`): always-small, always-large, entropy threshold, split
  conformal routing, FrugalGPT-style threshold, the temperature-scaling, uncalibrated and
  signal ablations, and `compare_routers`, which runs the paper's protocol for all of them.
  Oracle and Platt scaling are included as labelled extensions.
- **Serving adapters** (`ucci.integrations`): OpenAI Chat Completions, Completions and
  Responses (and OpenAI-compatible servers), vLLM offline outputs, Hugging Face `generate`
  (exact full-vocabulary softmax), and the llama.cpp server; a sync and async `Cascade`
  runner, a `JsonlLogger` for traffic records, and `attach_labels`.
- **Command line** (`ucci fit`, `route`, `evaluate`, `report`, `version`; also `python -m ucci`)
  with documented JSON output and exit codes.
- **Online monitoring and recalibration** (`ucci.online`, extension, not in the paper):
  `CalibrationMonitor` with binomial and Spiegelhalter drift tests, and `RecalibratingRouter`.
- **Plotting** (`ucci.plotting`): reliability diagrams and Pareto plots.
- **Rust crate** (`rust/`, MSRV 1.70): signal, calibration, policy, metrics and router format,
  bit-identical to the Python core on the shared golden vectors (`tests/golden`,
  `tools/make_golden.py`).
- **CoNLL-2003 replication** (`benchmarks/conll2003`) with Qwen2.5 1.5B and 7B Instruct:
  measured latency costs, all Table 2 methods, the ablations and bootstrap intervals.
- **Documentation** (MkDocs): the method with Theorem 1 and Proposition 2, a paper-to-code
  mapping with every implementation choice, guides per serving stack, and the API reference.
  README and documentation snippets are executed by `docs/_ext/check_snippets.py`.

[Unreleased]: https://github.com/varunkotte6/ucci/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/varunkotte6/ucci/releases/tag/v0.1.0
