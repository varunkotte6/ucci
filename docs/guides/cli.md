# Command line

Installing the package installs the `ucci` command (also `python -m ucci`). It runs the
paper's three-step protocol (Section 6.1) on logged traffic with `ucci.UCCIRouter`:

| Command | Does | Paper |
|---|---|---|
| `ucci fit` | fits \(g\) on the calibration split and selects \(\theta^*\) on the validation split; writes a router file | Sections 4.2, 4.3, Eq. 7; Table 2 bottom block with `--budget` |
| `ucci route` | prints \(\hat p\) and the decision for new queries | Eq. 6 |
| `ucci evaluate` | routes a split end to end and reports cost, accuracy, escalation rate and savings with bootstrap intervals | Section 6.1 step 3, Tables 2 and 3, Section 6.3 |
| `ucci report` | ECE and reliability tables of raw \(u\) and calibrated \(\hat p\) | Figure 1 |
| `ucci version` | prints the version | |

## Input records

JSON Lines (one object per line), a JSON array, or CSV/TSV with a header row; `--data -` reads
standard input. One record per query:

| Field | Required | Meaning |
|---|---|---|
| `id` | no (defaults to the record index) | query identifier |
| `u` | yes | \(u(x)\) of the small model, in \([0, 1]\) (Eq. 4) |
| `small_correct`, `large_correct` | yes | 0/1 (or `true`/`false`), or a per-query score in \([0, 1]\); \(e(x) = 1 -\) `small_correct` |
| `small_score`, `large_score` | no | per-query quality used for accuracy instead (`--metric score`, or `auto` when every record has both) |
| `split` | no | `cal`, `val` or `test` (also `calibration`, `validation`, `dev`) |
| `latency_small_ms`, `latency_large_ms` | no | for `fit --cost-from-latency` |

Other fields are ignored, so the records written by `ucci.integrations.cascade.JsonlLogger`
work as they are. Every required value is validated on every record, and an error names the
line and the id.

## Fit

<!-- snippet: run -->
```bash
ucci fit --data traffic.jsonl --tau 0.90 --out router.json
```

- **Objective.** `--tau T` selects the cheapest threshold with validation accuracy \(\ge T\)
  (Eq. 7); `--budget B` selects the most accurate threshold with mean cost \(\le B\).
- **Splits.** With a `split` field on every record, those splits are used (`--split-field`
  names another field). Otherwise records are split 30 / 20 / 50 at random with seed 0
  (`--cal-frac`, `--val-frac`, `--seed`). The random split orders records by the SHA-256 digest
  of `"<seed>:<id>"`, so it does not depend on record order, platform or numpy version, and
  `evaluate --split test` re-derives it.
- **Costs.** `--c-small` and `--c-large` (defaults 1.0 and 3.02), `--cost-model routing` or
  `sequential`, or `--cost-from-latency`: \(c_s = 1\) and \(c_\ell\) = mean large latency / mean
  small latency over the calibration and validation records.
- `--grid-step` (default 0.005) and `--metric auto|correct|score`.

The summary shows \(\theta^*\), validation cost, accuracy, escalation rate and savings, and ECE
before and after calibration on the held-out validation split. The router file is the shared
`ucci-router` format with an extra `fit` object recording the provenance of the fit (data
digests, split, objective, validation results); readers ignore it.

## Route

<!-- snippet: run -->
```bash
ucci route --router router.json --u 0.08 0.35 0.61
ucci route --router router.json --data traffic.jsonl --json > decisions.json
```

## Evaluate

<!-- snippet: run -->
```bash
ucci evaluate --router router.json --data traffic.jsonl --split test
ucci evaluate --router router.json --data traffic.jsonl --split test --c-large 5.0 --bootstrap 0
```

`evaluate` routes every query of the split with the router's \(\theta\), takes each query's actual
outcome from the model the router chose, and reports cost, accuracy, escalation rate and
savings against always-large, with percentile bootstrap intervals over queries (`--bootstrap`,
default 1000 resamples; `--seed`, `--level`). It also reports both single-model anchors and
the check of Theorem 1, assumption (ii): the large model's accuracy on the escalated queries
against all queries. `--c-small`, `--c-large` and `--cost-model` re-cost the same routing,
which is how the paper's Table 3 is built.

## Report

<!-- snippet: run -->
```bash
ucci report --router router.json --data traffic.jsonl --split test --strategy quantile
```

ECE and the reliability table (bin edges, count, mean forecast, observed error, gap) for raw
\(u\) and calibrated \(\hat p\), with bootstrap intervals for ECE. `--bins` sets the number of bins
and `--strategy uniform|quantile` the binning.

## JSON output and exit codes

Every command takes `--json` and then prints one JSON object with `"command"`,
`"schema_version": 1` and `"ucci_version"`; within a schema version keys are only added,
never renamed or removed.

<!-- snippet: run -->
```bash
ucci evaluate --router router.json --data traffic.jsonl --split test --json \
  | python -c "import json, sys; d = json.load(sys.stdin); print(d['ucci']['cost'], d['bootstrap']['ci']['cost'])"
```

| Exit code | Meaning |
|---|---|
| 0 | success |
| 2 | usage error (bad or conflicting options) |
| 3 | input error (unreadable file, invalid record, missing column, empty split, invalid router file) |
| 4 | infeasible objective (no grid threshold reaches \(\tau\), or every threshold exceeds the budget) |

An infeasible target prints the best validation accuracy the grid reaches and the always-large
accuracy.

## Accuracy in the CLI

The CLI measures accuracy as the mean per-query score of the returned answers. Corpus-level
metrics such as the paper's micro-F1 over entities need the Python API with
`metric=ucci.routed_micro_f1(small_counts, large_counts)`.

The complete reference, including every `--json` schema, is the
[`ucci.cli` module documentation](../api/cli.md).
