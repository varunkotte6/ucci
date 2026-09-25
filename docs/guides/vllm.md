# vLLM

The paper served both models with vLLM (Section 6.1). `ucci.integrations.vllm` reads the
`RequestOutput` objects of vLLM's offline `LLM.generate`. For vLLM's OpenAI-compatible server
(`vllm serve`), use the [OpenAI adapters](openai.md) with `drop_stop_token=True`.

## Offline inference

Generate the small model's answers greedily with the top-2 log-probabilities.
`greedy_sampling_params()` builds `SamplingParams(temperature=0, logprobs=2, max_tokens=256)`
(the paper's generation limit, Appendix B.2) and refuses a `temperature` argument:

<!-- snippet: skip (needs vLLM on a GPU) -->
```python
from vllm import LLM
from ucci.integrations.vllm import greedy_sampling_params, signals_from_vllm_batch

prompts = ["Extract the named entities from this sentence: ...", "..."]
small = LLM("Qwen/Qwen2.5-1.5B-Instruct")
outputs = small.generate(prompts, greedy_sampling_params())
signals = signals_from_vllm_batch(outputs)
u = [s.u for s in signals]
```

With `logprobs=k`, vLLM returns the top-\(k\) candidates plus the sampled token; under greedy
decoding the sampled token is the top-1, so `logprobs=2` gives exactly the top-1 and top-2
that Eq. 4 needs.

## The stop token

vLLM keeps the token that ended generation in `token_ids` and `logprobs`. UCCI excludes the
terminating EOS token, and with the default `drop_stop_token=None` the adapter drops the last
position exactly when vLLM reports that generation ended on a token: `finish_reason == "stop"`
with `stop_reason` None (the EOS token) or an int (a `stop_token_ids` entry). It keeps every
position when generation hit `max_tokens` or a stop string. `drop_stop_token=True` or `False`
overrides the rule.

The adapter also reads JSON dumps of the same objects (string token ids included). Here the
third position is the EOS token, which is dropped, and the two content tokens give
\(u = 1 - (0.85 + 0.3)/2 = 0.425\):

```python
import math
from ucci.integrations.vllm import signals_from_vllm

dumped = {"outputs": [{
    "index": 0,
    "text": "{\"PER\"",
    "token_ids": [515, 1740, 151645],
    "finish_reason": "stop",
    "stop_reason": None,
    "logprobs": [
        {"515": {"logprob": math.log(0.9), "rank": 1}, "90": {"logprob": math.log(0.05), "rank": 2}},
        {"1740": {"logprob": math.log(0.6), "rank": 1}, "2726": {"logprob": math.log(0.3), "rank": 2}},
        {"151645": {"logprob": math.log(0.99), "rank": 1}, "198": {"logprob": math.log(0.005), "rank": 2}},
    ],
}]}
sig = signals_from_vllm(dumped)
assert sig.n_tokens == 2 and abs(sig.u - 0.425) < 1e-12
```

## Log-probability mode

The values are the model's log-probabilities before any logits processor, since vLLM's default
`logprobs_mode` is `"raw_logprobs"`. The `"raw_logits"` and `"processed_logits"` modes return
logits, which the adapter rejects (values above 0, or candidate sets whose probabilities sum
above 1).

## A batched cascade

Route the whole batch with one small-model pass and send only the escalated queries to the
large model:

<!-- snippet: skip (needs vLLM on a GPU) -->
```python
from ucci import UCCIRouter

router = UCCIRouter.load("router.json")
decisions = router.route(u)
escalated = [p for p, esc in zip(prompts, decisions.escalate) if esc]
large_outputs = LLM("Qwen/Qwen2.5-7B-Instruct").generate(escalated, greedy_sampling_params())
```

[`examples/vllm_cascade.py`](https://github.com/varunkotte6/ucci/blob/main/examples/vllm_cascade.py)
is the complete script (both engines on one GPU).

API reference: [`ucci.integrations.vllm`](../api/integrations.md#ucci.integrations.vllm).
