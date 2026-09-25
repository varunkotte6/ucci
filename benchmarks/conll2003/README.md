# UCCI on CoNLL-2003: a public replication

The paper's experiments use a private production workload. Its Section 7
names the next step: *"replicate the method on a public NER benchmark such as
CoNLL-2003 and on a different small/large model family."* This directory is
that replication, runnable by anyone with the public data and open models.

It follows the paper's protocol (Section 6.1) end to end: JSON entity
extraction with identical prompts for both models, greedy decoding, the
exact-match error event e(x) for calibration (Section 4.2), micro-F1 as the
metric, a measured-latency cost ratio, disjoint 30 / 20 / 50 calibration /
validation / test splits with a fixed seed, g fit on calibration, theta
selected on validation (Eq. 7), every test sentence routed end to end with
the actual output of the chosen model, and 1000-resample bootstrap intervals.

**What this is not.** It does not reproduce the paper's numbers: the data,
the models and the hardware all differ, and the paper's data is not public.
Every number in `runs/*/results.md` comes from the logs in the same
directory and the scripts here. This README contains no results.

## Setup

| | |
|---|---|
| Data | CoNLL-2003 English NER (Tjong Kim Sang and De Meulder, 2003, [arXiv:cs/0306050](https://arxiv.org/abs/cs/0306050)) from the Hugging Face hub: `eriktks/conll2003`, parquet conversion (`refs/convert/parquet`) pinned at commit `ce85b39f9dd99f552d0739d456814e95fb6a39b0`. The original loading script does not run under `datasets` 4.x; the pinned parquet commit does and fixes the bytes. |
| Pool | The CoNLL-2003 validation (3,250 sentences) and test (3,453) splits, pooled: 6,703 sentences. The train split is never evaluated; it was used only to settle the prompt wording. |
| Splits | 30 / 20 / 50 calibration / validation / test, drawn from the pool with seed 0 by the rule `ucci fit` documents (order ids by SHA-256 of `"<seed>:<id>"`, then cut), so the CLI re-derives the same split. |
| Models | Small: [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Large: [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) at `a09a35458c702b33eeacc393d103063234e8bc28`. Both Apache-2.0; Qwen2.5 Technical Report, [arXiv:2412.15115](https://arxiv.org/abs/2412.15115). A different family from the paper's models, as Section 7 asks. |
| Decoding | Greedy, at most 256 new tokens (Appendix B.2), bfloat16 weights (the checkpoints' dtype), batch size 16 with left padding. |
| Reference hardware | Apple M5 Max, 48 GB unified memory, macOS 26.6.2, PyTorch 2.8.0 on the MPS backend, transformers 4.57.6, datasets 4.5.0, Python 3.9.6 (`requirements.txt`; each run also saves `environment.txt` and a full `pip freeze`). |

## How each step maps to the paper

| Paper | Here |
|---|---|
| Prompt (Appendix B.1): short instruction plus the JSON fields | `conll.PROMPT_TEMPLATE`, fields `PER`, `ORG`, `LOC`, `MISC`, each a list of strings; one user turn through the model's chat template. Printed below. |
| Greedy decoding (Section 4.1, Appendix B.2) | The model's generation config is replaced by a plain greedy one, so sampling defaults and the repetition penalty that instruction models ship are off. Every run checks that each emitted token is the arg-max of the logits the signal is computed from (`argmax_mismatch`, expected 0). |
| u(x), Eq. 4 | `ucci.token_margin_uncertainty` on the top-1 and top-2 next-token probabilities (full-vocabulary softmax, float32) of every content token; the end-of-sequence token and padding are excluded (the package's token convention). The first batch of every run is recomputed with `ucci.integrations.transformers` in float64 and the largest difference is stored in the run metadata. |
| e(x), Section 4.2 | 1 minus exact match: the output parses as a JSON object and its entity set equals the gold set for every type. |
| Micro-F1 (Sections 3, 6) | Entity-level micro-F1 over (type, string) pairs; see scoring below. |
| Costs (Section 6.1, Appendix B.3) | `latency.py`: mean end-to-end latency (prompt rendering, tokenization, greedy generation, detokenization) over 100 queries per model at batch size 1, no cache reuse between queries, device synchronized before the clock stops, 3 untimed warm-up queries; c_s = 1, c_l = mean large / mean small. |
| Three-step protocol (Section 6.1) | `analysis.run_methods`: calibrate on calibration, select on validation, route test end to end. |
| Table 2, top block | Every method at an F1 target tau (default: validation small-only F1 plus three quarters of the validation small-to-large gap; `--target-f1` fixes it). |
| Table 2, bottom block | Every method at a matched cost budget (default: halfway between c_s and c_l, as the paper's 2.00 is for 1.00 and 3.02; `--budget` fixes it). |
| Baselines (Section 6.1) | `ucci.baselines`: always-small, always-large, entropy threshold (mean full-vocabulary token entropy), split conformal on "small model correct" with raw u(x) as the score, FrugalGPT-style threshold with the mean top-1 probability as the confidence (the paper does not say which confidence it used). |
| Ablations (Section 6.3, Appendix B.4) | Temperature scaling and uncalibrated u in place of isotonic; isotonic on mean entropy and on 1 minus mean max probability in place of the token margin. |
| Extensions (not in the paper) | Platt scaling; a FrugalGPT-style threshold on a learned score (logistic regression on u, entropy, max probability and log length, fit on the calibration split); a label-dependent greedy oracle (per-sentence gains from micro-F1 linearized at the target) as an approximate lower bound on cost (analysis only). |
| Figure 1, ECE | Reliability diagrams (deciles) of raw u(x) and isotonic p_hat, on the calibration split (in sample, the paper's convention) and on the test split (out of sample); ECE with 10 equal-width bins and with deciles, Brier score, bootstrap intervals. |
| Figure 2 | Cost against micro-F1 on the test split for the UCCI threshold grid and the raw-u and entropy threshold sweeps, with the selected operating points. |
| Table 3 | The UCCI test routing re-costed at the measured ratio, at the paper's 3.02 and at its hypothetical 5 and 10; theta is re-selected at each ratio to confirm it does not move. |
| Table 4 | Per-entity-type micro-F1 of both models and of the UCCI-routed answers on the test split. |
| Theorem 1, assumption (ii) (Section 6.3) | Large-model micro-F1 on the escalated test sentences against all test sentences. |
| Reproducibility statement: per-split summary statistics | `per_split_summary` in `results.json`: size, sentence length, entity counts by type, both models' F1, exact match and parse-failure rates, mean u(x), tokens and latency, per split. |
| Bootstrap (Section 6.2) | Percentile intervals, 1000 resamples over test sentences, one seed for all statistics so the intervals and the paired differences against UCCI use the same resamples. Thresholds stay at their validation choice, so the intervals cover test-set sampling. |

## The prompt

Both models receive this user message (no system message is added, so the
chat template inserts the model's default one):

```text
Extract the named entities from this sentence.
Return only a JSON object with fields: PER, ORG, LOC, MISC.
Each field is a list of names copied exactly from the sentence; use an empty list when there are none.
PER: people. ORG: organizations, companies, sports teams, political parties. LOC: countries, cities, regions and other places. MISC: other proper names, such as nationalities, languages, events and titles.
Only include proper names. Do not include numbers, dates, scores or common nouns.

Sentence: {sentence}
Output:
```

`{sentence}` is the CoNLL tokens joined by single spaces. The type glosses
and the last instruction line were settled on sentences from the CoNLL-2003
train split, which is not part of the evaluation pool.

## Scoring

The models return entity strings, not token offsets, so each sentence is
scored on the set of (type, normalized string) pairs: Unicode NFKC, trimmed,
internal whitespace collapsed, case kept. A mention repeated in a sentence
counts once. The JSON is read from the first object in the output (inside a
Markdown code fence if there is one); keys are matched case-insensitively
and unknown keys are ignored. An output with no parseable JSON object scores
as an empty prediction and is never an exact match. A sentence whose small
model output is empty (u(x) undefined) is routed as if u = 1. The analysis
re-scores every raw output with the current code and reports any
disagreement with the logged scores.

## Running it

From the repository root, in an environment with `requirements.txt`
installed and `pip install -e .`:

```bash
# Smoke test: 40 sentences, a few minutes.
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
    --small benchmarks/conll2003/runs/smoke/small.jsonl --large benchmarks/conll2003/runs/smoke/large.jsonl \
    --latency benchmarks/conll2003/runs/smoke/latency.json --out-dir benchmarks/conll2003/runs/smoke

# Full run: latency, both models over all 6,703 sentences, analysis.
PYTHON=python bash benchmarks/conll2003/run_full.sh
```

`runs/smoke` only proves the pipeline runs: 40 sentences give a 12 / 8 / 20
split, far too small for any conclusion.

`analyze.py` also checks itself: it re-scores every raw output, reruns the
Table 2 methods through `ucci.baselines.compare_routers` and reports the
largest disagreement with its own numbers, and lists any comparator that
raised an error as "not run" (with the error) instead of stopping.

Every step resumes: `generate.py` keeps the records already in `--out` and
refuses to append under different settings (model, revision, dtype, batch
size, prompt, data revision); `run_full.sh` keeps an existing
`latency.json`. With the models and data already downloaded, set
`HF_HUB_OFFLINE=1` so a network hiccup cannot stop a long run.

**Runtime.** On the reference machine, samples of 40 to 96 sentences ran at
about 4 to 6 sentences per second for the 1.5B model and 2 to 3 for the 7B
model (batch size 16), and single queries took roughly 0.7 s and 1.5 s, so the
full run takes about one to one and a half hours. Each run records its own
timings in the `.meta.json` files.

**CUDA and vLLM.** `--backend vllm` (in `generate.py` and `latency.py`) runs
the models with vLLM, as the paper did (Section 6.1). vLLM returns the top-k
log-probabilities only (k = 20 here), so its entropy signal is the truncated
entropy over those k candidates; u(x) and the max probability are exact.
With the transformers backend on CUDA the default attention kernel is used.

**Apple MPS notes.** PyTorch's fused attention on MPS returns NaN for the
padding rows of a left-padded batch, and the NaN leaks into the real rows, so
`generate.py` uses the eager attention implementation on MPS (and refuses to
log NaN signals). A batch of one has no padding, so `latency.py` times with
the default kernel. In bfloat16 the batch a sentence falls into can change
low-order bits of its probabilities; runs are deterministic for a fixed
configuration because batches are always formed the same way (longest
prompts first, ties by pool order).

## Outputs

`runs/<name>/`:

| File | Content |
|---|---|
| `small.jsonl`, `large.jsonl` | One record per sentence: `id`, `source_split`, `model`, `sentence`, `raw_output`, `parse_ok`, `schema_ok`, `pred`, `gold`, `exact_match`, `tp`, `fp`, `fn`, `per_type`, `u`, `entropy`, `max_prob`, `n_tokens`, `stop_reason`, `argmax_mismatch`, `prompt_tokens`, `latency_ms` (batch time divided by batch size), `batch_ms`, `batch_size`. |
| `*.jsonl.meta.json` | Model and data revisions, dtype, device, attention kernel, batch size, prompt text and hash, a rendered prompt, library versions, per-session timings and the adapter cross-check. |
| `*.tokens.jsonl` | Per-token top-1, top-2 probabilities and entropy (not kept in git). |
| `latency.json` | Every timed query and the per-model summary; `cost_ratio` is c_l / c_s. |
| `results.json`, `results.md` | All results with configuration and input SHA-256 hashes. |
| `reliability_test.png`, `reliability_cal.png`, `pareto_test.png` | Figures 1 and 2 analogues, drawn with `ucci.plotting`. |
| `joined.jsonl` | One record per sentence in the package's traffic format with its `split`, ready for `ucci fit` / `ucci evaluate`. |
| `environment.txt`, `environment-pip-freeze.txt` | Hardware and software of the run. |

## Tests

```bash
python -m pytest benchmarks/conll2003/tests
```

The tests cover IOB decoding, prompt construction, JSON parsing and scoring,
the split rule, log joining and re-scoring, the whole analysis on synthetic
logs (including agreement with `ucci.baselines.compare_routers` and a
`ucci fit` run on the exported records), and the `analyze.py` command line.
They need neither a model nor the network.

## Citation

If you use this replication, cite the UCCI paper and the data and models it
runs on:

```bibtex
@article{kotte2026ucci,
  title   = {{UCCI}: Calibrated Uncertainty for Cost-Optimal {LLM} Cascade Routing},
  author  = {Kotte, Varun},
  journal = {arXiv preprint arXiv:2605.18796},
  year    = {2026}
}
@inproceedings{tjongkimsang2003conll,
  title     = {Introduction to the {CoNLL}-2003 Shared Task: Language-Independent Named Entity Recognition},
  author    = {Tjong Kim Sang, Erik F. and De Meulder, Fien},
  booktitle = {Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003},
  pages     = {142--147},
  year      = {2003}
}
@article{qwen2024qwen25,
  title   = {Qwen2.5 Technical Report},
  author  = {{Qwen Team}},
  journal = {arXiv preprint arXiv:2412.15115},
  year    = {2024}
}
```
