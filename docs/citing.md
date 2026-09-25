# Citing

If you use UCCI or this code, please cite the paper:

> Varun Kotte. *UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing.*
> arXiv:[2605.18796](https://arxiv.org/abs/2605.18796), 2026.

```bibtex
@article{kotte2026ucci,
  title         = {{UCCI}: Calibrated Uncertainty for Cost-Optimal {LLM} Cascade Routing},
  author        = {Kotte, Varun},
  journal       = {arXiv preprint arXiv:2605.18796},
  year          = {2026},
  eprint        = {2605.18796},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2605.18796},
  url           = {https://arxiv.org/abs/2605.18796}
}
```

The repository's
[`CITATION.cff`](https://github.com/varunkotte6/ucci/blob/main/CITATION.cff) describes the
software and names the paper as its preferred citation; GitHub's "Cite this repository" button
reads it.

When you report results obtained with the CoNLL-2003 replication, please also cite the data and
the models it runs on; the entries are in the
[benchmark README](https://github.com/varunkotte6/ucci/blob/main/benchmarks/conll2003/README.md#citation).

## Works the paper builds on

The method and its baselines rest on these works, as cited in the paper:

- Isotonic regression: Barlow, Bartholomew, Bremner and Brunk, *Statistical Inference Under
  Order Restrictions*, Wiley, 1972; Zadrozny and Elkan, *KDD* 2002.
- Calibration: Guo, Pleiss, Sun and Weinberger, *ICML* 2017; Naeini, Cooper and Hauskrecht,
  *AAAI* 2015; Platt, 1999.
- Conformal prediction: Angelopoulos and Bates, arXiv:2107.07511, 2021.
- LLM cascades: Chen, Zaharia and Zou, *FrugalGPT*, arXiv:2305.05176, 2023.
