# Guides

UCCI needs one number per query from the small model: the token-margin uncertainty
\(u(x) = 1 - \frac{1}{T}\sum_t (p_{t,1} - p_{t,2})\) of its greedy generation (Section 4.1, Eq. 4).
Every serving stack can return the top-2 next-token log-probabilities that this needs, at no
extra model cost. The adapters in `ucci.integrations` read them from each stack's native
response and return a `TokenSignals`:

| Field | Meaning |
|---|---|
| `u` | \(u(x)\), the routing signal (Eq. 4) |
| `n_tokens` | \(T\), content tokens averaged over (stop token and padding excluded) |
| `mean_max_prob` | mean top-1 probability, the "max probability" signal of the Section 6.3 ablation |
| `mean_entropy` | mean next-token entropy (nats), the signal of the entropy-threshold baseline |
| `entropy_support` | `"full_vocabulary"`, `"top_k"` (truncated sum) or `"top_k_renormalized"` |
| `n_non_greedy` | positions where the generated token was not the top-1 candidate (should be 0) |
| `margins` | the per-token margins \(m_t\) |
| `source` | which adapter and field produced the numbers |

`TokenSignals.to_record()` gives the `u`, `entropy` and `max_prob` fields of the traffic record
format that `ucci fit` and `ucci.baselines` read.

Three rules hold for every stack:

1. **Greedy decoding** (temperature 0, no beam search). The signal is defined for greedy
   generation, where the generated token is the top-1 candidate.
2. **At least two candidates per token** (`top_logprobs=2`, `logprobs=2`, `n_probs=2`).
3. **Content tokens only.** The terminating EOS or stop token is excluded. Some stacks report
   it and some do not; the adapters handle each (see the table in
   [Paper to code](../paper_mapping.md#implementation-choices)).

| Stack | Guide |
|---|---|
| OpenAI API and OpenAI-compatible servers (vLLM, llama.cpp, LiteLLM) | [OpenAI-compatible APIs](openai.md) |
| vLLM offline inference | [vLLM](vllm.md) |
| Hugging Face transformers | [transformers](transformers.md) |
| llama.cpp server, native endpoint | [llama.cpp](llamacpp.md) |
| Anything else | `ucci.uncertainty_from_logprobs(top1_logprobs, top2_logprobs)` or `ucci.token_margin_uncertainty(pairs)` |

Then:

- [Collecting data from live traffic](live_traffic.md): run the cascade, log records, label
  them, fit.
- [Command line](cli.md): fit, route, evaluate and report from logged records.
- [Baselines and the comparison protocol](baselines.md): the paper's Table 2 on your data.
- [Monitoring and recalibration](monitoring.md): detect calibration drift and refit.
- [Rust](rust.md): route with the same router files from Rust.
