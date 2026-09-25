# Benchmarks

The paper's experiments use a private production workload, so its numbers cannot be rerun.
Its Section 7 names the next step: replicate the method on a public NER benchmark such as
CoNLL-2003 and on a different small/large model family. `benchmarks/conll2003` is that
replication, runnable by anyone with the public data and open models. It does not reproduce
the paper's numbers (the data, the models and the hardware all differ); every number it
reports comes from its own logs and scripts.

## What it runs

| | |
|---|---|
| Data | CoNLL-2003 English NER (Tjong Kim Sang and De Meulder, 2003), `eriktks/conll2003` on the Hugging Face hub, parquet conversion pinned at commit `ce85b39f9dd99f552d0739d456814e95fb6a39b0`; the validation and test splits pooled (6,703 sentences) |
| Models | `Qwen/Qwen2.5-1.5B-Instruct` (small) and `Qwen/Qwen2.5-7B-Instruct` (large), both pinned to a commit, Apache-2.0 |
| Task | JSON entity extraction with fields `PER`, `ORG`, `LOC`, `MISC`, the same prompt for both models |
| Decoding | greedy, at most 256 new tokens (Appendix B.2) |
| Signal | \(u(x)\) from the full-vocabulary top-2 probabilities of every content token (EOS and padding excluded), cross-checked against `ucci.integrations.transformers` |
| Error event and metric | \(e(x)\) = 1 minus JSON exact match (Section 4.2); entity-level micro-F1 |
| Costs | measured latency, 100 queries per model at batch size 1 (Section 6.1, Appendix B.3) |
| Protocol | 30 / 20 / 50 splits with the `ucci fit` split rule; \(g\) on calibration, \(\theta\) on validation, every test sentence routed end to end (Section 6.1) |
| Comparisons | all Table 2 methods through `ucci.baselines.compare_routers`, at an F1 target and at a matched cost budget; the Section 6.3 and Appendix B.4 ablations |
| Diagnostics | reliability diagrams and ECE (Figure 1), Pareto frontier (Figure 2), cost-ratio sensitivity (Table 3), per-entity F1 (Table 4), the assumption (ii) check, 1000-resample paired bootstrap intervals |

The benchmark's
[README](https://github.com/varunkotte6/ucci/blob/main/benchmarks/conll2003/README.md) maps
each step to the paper in detail, prints the prompt, and documents scoring.

## Running it

From the repository root, with the benchmark requirements installed
(`pip install -e ".[bench]"`, or the exact pins of the reference run in
`benchmarks/conll2003/requirements.txt`):

```bash
# Full run: latency, both models over all 6,703 sentences, analysis (resumable).
PYTHON=python bash benchmarks/conll2003/run_full.sh
```

A 40-sentence smoke run of the same pipeline (a few minutes, both models downloaded):

```bash
python benchmarks/conll2003/generate.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --dtype bfloat16 --limit 40 \
    --out benchmarks/conll2003/runs/smoke/small.jsonl
python benchmarks/conll2003/generate.py --model Qwen/Qwen2.5-7B-Instruct \
    --revision a09a35458c702b33eeacc393d103063234e8bc28 --dtype bfloat16 --limit 40 \
    --out benchmarks/conll2003/runs/smoke/large.jsonl
python benchmarks/conll2003/latency.py \
    --small Qwen/Qwen2.5-1.5B-Instruct --small-revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
    --large Qwen/Qwen2.5-7B-Instruct --large-revision a09a35458c702b33eeacc393d103063234e8bc28 \
    --dtype bfloat16 --n-queries 10 --warmup 2 --out benchmarks/conll2003/runs/smoke/latency.json
python benchmarks/conll2003/analyze.py \
    --small benchmarks/conll2003/runs/smoke/small.jsonl \
    --large benchmarks/conll2003/runs/smoke/large.jsonl \
    --latency benchmarks/conll2003/runs/smoke/latency.json \
    --out-dir benchmarks/conll2003/runs/smoke
```

Re-running only the analysis on logs already in the repository needs numpy and matplotlib and
no model; `make bench-smoke` does this for the committed smoke logs, into a temporary
directory. The smoke run's 40 sentences only show that the pipeline works; they support no
conclusion.

`analyze.py` options set the split seed and fractions, the F1 target (`--target-f1`, or
`--target-frac` of the validation small-to-large gap, default 0.75), the matched budget
(`--budget`, or `--budget-frac` between \(c_s\) and \(c_\ell\), default 0.5), the cost ratio
(`--cost-ratio` overrides the measured one) and the bootstrap (`--n-boot`, `--boot-seed`).

## Outputs

Each run directory `benchmarks/conll2003/runs/<name>/` holds the per-sentence logs of both
models with their metadata (model and data revisions, library versions, timings, the adapter
cross-check), `latency.json`, `results.json` and `results.md` (with the SHA-256 of every input),
the figures, `joined.jsonl` in the traffic format that `ucci fit` reads, and the hardware and
software environment. Results are reported from these files only.

## Tests

```bash
python -m pytest benchmarks/conll2003/tests
```

IOB decoding, prompt construction, JSON parsing and scoring, the split rule, log joining,
the whole analysis on synthetic logs (including agreement with `compare_routers` and a
`ucci fit` run on the exported records) and the `analyze.py` command line. No model and no
network needed; the transformers tests skip when torch is missing.

## References

- E. F. Tjong Kim Sang and F. De Meulder. Introduction to the CoNLL-2003 shared task:
  language-independent named entity recognition. *CoNLL*, 2003.
  arXiv:[cs/0306050](https://arxiv.org/abs/cs/0306050).
- Qwen Team. Qwen2.5 technical report. arXiv:[2412.15115](https://arxiv.org/abs/2412.15115), 2024.
