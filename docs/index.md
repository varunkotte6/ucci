# UCCI

**UCCI** (Uncertainty-Calibrated Cascaded Inference) routes each query of a two-model LLM
cascade: it keeps the small model's answer, or pays for the large model when the small one is
likely wrong. This site documents the reference implementation of

> Varun Kotte. *UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing.*
> arXiv:[2605.18796](https://arxiv.org/abs/2605.18796), 2026.

UCCI turns the small model's own token probabilities into a calibrated probability that its
answer is wrong, \(\hat p(x)\), and escalates when \(\hat p(x) > \theta\). The threshold
\(\theta^*\) is the cheapest one that meets an accuracy target on a validation split, measured with
actual outputs and costs. Under three stated assumptions this threshold policy is cost-optimal
(Theorem 1).

## What is in the package

| Component | Where | Needs |
|---|---|---|
| Token-margin signal \(u(x)\), isotonic calibration, threshold policy and selection, router, metrics, router file format | `ucci` ([API](api/index.md)) | numpy |
| Adapters that compute \(u(x)\) from OpenAI-compatible, vLLM, transformers and llama.cpp outputs; a cascade runner and traffic logger | `ucci.integrations` ([guides](guides/index.md)) | numpy (the model libraries only to run models) |
| The paper's baselines and ablations, and the comparison protocol of Section 6.1 | `ucci.baselines` ([guide](guides/baselines.md)) | numpy |
| `ucci fit / route / evaluate / report` | `ucci.cli` ([guide](guides/cli.md)) | numpy |
| Calibration monitoring and sliding-window recalibration (extension) | `ucci.online` ([guide](guides/monitoring.md)) | numpy |
| Reliability diagrams and Pareto plots | `ucci.plotting` | matplotlib |
| Rust crate with bit-identical results on the shared golden vectors | `rust/` ([guide](guides/rust.md)) | Rust 1.70 |
| CoNLL-2003 replication with open models | `benchmarks/conll2003` ([page](benchmarks.md)) | torch, transformers, datasets |

## Where to start

- [Installation](installation.md) and the [quickstart](quickstart.md): fit, select and route in a
  dozen lines.
- [The method](method.md): the three steps, Theorem 1 and Proposition 2, stated precisely, and
  why calibration matters when the policy is a threshold anyway.
- [Paper to code](paper_mapping.md): every element of Sections 4 to 6 and Appendices A and B
  mapped to the code, with every implementation choice the paper leaves open.
- [Collecting data from live traffic](guides/live_traffic.md): how to get the calibration and
  validation data a router is fitted on.
- [FAQ](faq.md): when Theorem 1's assumptions fail, and what to do about it.

## Paper results and this repository

The paper's experiments use a private production workload (a production enterprise photo
management system, 75,000 labelled queries). That data is not public and is not part of this
repository, so the paper's numbers are not reproduced here. Where this site quotes them, it
says so ("the paper reports"). Every number the repository presents as a result is produced by
a script in it: see [Benchmarks](benchmarks.md).

## Citing

If you use UCCI or this code, please cite the paper; BibTeX and a `CITATION.cff` are on the
[citing page](citing.md).
