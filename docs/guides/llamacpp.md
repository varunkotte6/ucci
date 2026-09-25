# llama.cpp

`ucci.integrations.llamacpp` reads the llama.cpp server's native `/completion` endpoint,
streamed or not. For its OpenAI-compatible endpoints (`/v1/chat/completions`,
`/v1/completions`) use the [OpenAI adapters](openai.md) with `drop_stop_token=True`.

## Request

Ask for the top-2 probabilities of every generated token and decode greedily
(`top_k: 1`). Keep `post_sampling_probs` at its default `false`, so the values are the model's
next-token distribution rather than the sampler's output:

<!-- snippet: skip (needs a running llama.cpp server) -->
```python
import json
import urllib.request

request = {"prompt": "Extract the named entities: ...", "n_predict": 256, "n_probs": 2, "top_k": 1}
req = urllib.request.Request("http://localhost:8080/completion",
                             data=json.dumps(request).encode(),
                             headers={"Content-Type": "application/json"})
response = json.load(urllib.request.urlopen(req))
```

## Parse

Pass the whole response. On current servers each entry of `completion_probabilities` holds the
generated token's `logprob` and a `top_logprobs` list. The server appends the EOS token's entry
before it notices the EOS, so with the default `drop_stop_token=None` the adapter drops the
last entry when the response says `"stop_type": "eos"` (`"stopped_eos": true` on older
servers). Here the third entry is the EOS and the two content tokens give
\(u = 1 - (0.85 + 0.3)/2 = 0.425\):

```python
import math
from ucci.integrations.llamacpp import signals_from_llamacpp

def entry(token, p1, alt, p2):
    return {"token": token, "logprob": math.log(p1), "top_logprobs": [
        {"token": token, "logprob": math.log(p1)}, {"token": alt, "logprob": math.log(p2)}]}

response = {
    "content": " Canon R5",
    "stop_type": "eos",
    "completion_probabilities": [
        entry(" Canon", 0.9, " Nikon", 0.05),
        entry(" R5", 0.6, " R6", 0.3),
        entry("</s>", 0.99, "\n", 0.005),
    ],
}
sig = signals_from_llamacpp(response)
assert sig.n_tokens == 2 and abs(sig.u - 0.425) < 1e-12
```

With a bare `completion_probabilities` list there is no `stop_type`, so set `drop_stop_token`
explicitly.

## Formats

Three shapes of `completion_probabilities` exist, and all are read:

1. **Current servers** (since llama.cpp pull request
   [#10783](https://github.com/ggml-org/llama.cpp/pull/10783), December 2024) with the default
   `post_sampling_probs: false`: `{"id", "token", "bytes", "logprob", "top_logprobs": [...]}`,
   from a softmax over the raw logits. A probability of exactly 0 arrives as the most negative
   float32 value.
2. **`post_sampling_probs: true`**: `prob` and `top_probs` after the sampler chain. Under greedy
   decoding that leaves one candidate with probability 1, which is not the model's
   distribution, so this format is rejected unless `allow_post_sampling=True`.
3. **Servers before that change**: `{"content", "probs": [{"tok_str", "prob"}, ...]}`. A
   response in which every position reports probabilities of exactly 1 and 0 is degenerate
   (post-sampler output of greedy decoding) and is rejected.

## Streaming

With `"stream": true`, each server-sent event carries the entries of its own tokens and the
final event carries `stop_type`. Parse the events and pass the list to
`signals_from_llamacpp_stream` or `u_from_llamacpp_stream`.

API reference: [`ucci.integrations.llamacpp`](../api/integrations.md#ucci.integrations.llamacpp).
