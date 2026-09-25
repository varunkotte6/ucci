# Examples

| Script | Needs | What it shows |
|---|---|---|
| [`synthetic_demo.py`](synthetic_demo.py) | numpy (matplotlib for `--plot`) | The whole method on simulated data: calibration, threshold selection at an accuracy target and at a cost budget, end-to-end test routing, a bootstrap interval, saving and reloading a router. The data is synthetic; its numbers are not results. |
| [`transformers_cascade.py`](transformers_cascade.py) | torch, transformers; runs on CPU, MPS or CUDA | A live cascade with two local Hugging Face models: u(x) from the small model's greedy generation, routing, the large model on escalation. |
| [`vllm_cascade.py`](vllm_cascade.py) | vLLM on a CUDA GPU | A batched cascade with vLLM offline inference: one small-model batch, one large-model batch for the escalated queries only. |
| [`openai_cascade.py`](openai_cascade.py) | `openai`, an API key or an OpenAI-compatible server | A cascade over the Chat Completions API with `logprobs`, logging every query in the traffic format `ucci fit` reads. |

The three live examples need a router fitted on labelled traffic from your
own workload (`ucci fit`, or `UCCIRouter.calibrate` plus `choose_threshold`).
Without `--router` they fall back to a clearly labelled placeholder that only
demonstrates the mechanics. The CoNLL-2003 benchmark writes a fitted-ready
file, `benchmarks/conll2003/runs/<name>/joined.jsonl`, for example:

```bash
ucci fit --data benchmarks/conll2003/runs/full/joined.jsonl --tau 0.6 --out router.json
python examples/transformers_cascade.py --small Qwen/Qwen2.5-1.5B-Instruct \
    --large Qwen/Qwen2.5-7B-Instruct --router router.json
```

`ucci fit` selects theta on mean per-query exact match; the paper's
micro-F1 target needs the Python API (`metric=ucci.routed_micro_f1(...)`),
which is what `benchmarks/conll2003/analyze.py` uses.
