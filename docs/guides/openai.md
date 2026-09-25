# OpenAI-compatible APIs

`ucci.integrations.openai` computes \(u(x)\) from OpenAI-format responses: Chat Completions
(whole, per choice, or streamed), legacy Completions and the Responses API. It reads SDK objects
and plain dicts (for example parsed from a JSON log) alike, and never imports the `openai`
package; install the SDK (`pip install "ucci[openai]"`) only to make calls.

## Chat Completions

Request greedy decoding and at least two candidates per token:

```python
from openai import OpenAI
from ucci.integrations.openai import signals_from_chat_completion, u_from_chat_completion

client = OpenAI()
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Extract the named entities: ..."}],
    temperature=0,
    logprobs=True,
    top_logprobs=2,     # 2 is all u(x) needs; up to 20 for a tighter top-k entropy
)
u = u_from_chat_completion(resp)
sig = signals_from_chat_completion(resp)
print(u, sig.n_tokens, sig.mean_max_prob, sig.n_non_greedy)
```

A logged response works the same way. Here two tokens with top-2 probabilities (0.9, 0.05) and
(0.6, 0.3) give margins 0.85 and 0.3, so \(u = 1 - (0.85 + 0.3)/2 = 0.425\):

```python
import math

logged = {
    "choices": [{
        "index": 0,
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "{\"PER\""},
        "logprobs": {"content": [
            {"token": "{", "logprob": math.log(0.9), "top_logprobs": [
                {"token": "{", "logprob": math.log(0.9)},
                {"token": "```", "logprob": math.log(0.05)}]},
            {"token": "\"PER\"", "logprob": math.log(0.6), "top_logprobs": [
                {"token": "\"PER\"", "logprob": math.log(0.6)},
                {"token": "\"ORG\"", "logprob": math.log(0.3)}]},
        ]},
    }]
}
assert abs(u_from_chat_completion(logged) - 0.425) < 1e-12
```

Useful arguments of `signals_from_chat_completion`:

- `choice_index`: which choice to read when `n > 1`;
- `part="refusal"`: a refusal carries its tokens under `logprobs.refusal`;
- `drop_stop_token=True`: for servers that report the terminating EOS token (below);
- `require_greedy=True`: raise if a generated token is not the top-1 candidate;
- `renormalize_entropy=True`: entropy of the renormalized top-\(k\) instead of the truncated sum.

Malformed payloads raise `ucci.integrations.PayloadError` (a `ValueError`) whose message names
the path inside the payload and the request option to fix, for example a missing
`top_logprobs`.

## Streaming

With `stream=True`, each chunk carries the log-probabilities of its own tokens. Collect the
chunks and pass them all:

```python
from ucci.integrations.openai import u_from_chat_completion_chunks

chunks = [
    {"choices": [{"index": 0, "finish_reason": None, "logprobs": {"content": [
        {"token": "{", "logprob": math.log(0.9), "top_logprobs": [
            {"token": "{", "logprob": math.log(0.9)}, {"token": "```", "logprob": math.log(0.05)}]}]}}]},
    {"choices": [{"index": 0, "finish_reason": "stop", "logprobs": {"content": [
        {"token": "}", "logprob": math.log(0.6), "top_logprobs": [
            {"token": "}", "logprob": math.log(0.6)}, {"token": ",", "logprob": math.log(0.3)}]}]}}]},
]
print(round(u_from_chat_completion_chunks(chunks), 6))   # 0.425
```

## Completions and Responses

The legacy Completions API takes `logprobs=2` (an integer, at most 5) and returns
`{token: logprob}` dicts per position, which also contain the sampled token:

```python
from ucci.integrations.openai import u_from_completion

completion = {"choices": [{"finish_reason": "stop", "text": "{\"PER\"", "logprobs": {
    "tokens": ["{", "\"PER\""],
    "token_logprobs": [math.log(0.9), math.log(0.6)],
    "top_logprobs": [{"{": math.log(0.9), "```": math.log(0.05)},
                     {"\"PER\"": math.log(0.6), "\"ORG\"": math.log(0.3)}],
}}]}
print(round(u_from_completion(completion), 6))   # 0.425
```

The Responses API returns per-token log-probabilities when the request sets
`top_logprobs=2` and `include=["message.output_text.logprobs"]`; read the result with
`u_from_responses`. `signals_from_openai` and `u_from_openai` detect which of the three formats
they were given.

## OpenAI-compatible servers

The same functions read any server that speaks this format. Servers whose documentation or
source confirms log-probability support: vLLM's OpenAI-compatible server, the llama.cpp
server's `/v1/chat/completions`, and the LiteLLM proxy (values depend on the provider behind
it).

vLLM and llama.cpp report a log-probability for the terminating EOS token; the OpenAI API does
not. UCCI excludes that token, so pass `drop_stop_token=True` for those servers. The last entry
is then dropped when `finish_reason` is `"stop"` and kept when generation hit the token limit.

```python
local = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")   # e.g. `vllm serve`
resp = local.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Extract the named entities: ..."}],
    temperature=0, logprobs=True, top_logprobs=2, max_tokens=256,
)
u = u_from_chat_completion(resp, drop_stop_token=True)
```

## A cascade over the API

`make_chat_fn` wraps `client.chat.completions.create` as a model function for
`ucci.integrations.cascade.Cascade`: greedy, with `top_logprobs=2` for the small model and no
log-probabilities for the large one. With an `AsyncOpenAI` client the functions return
awaitables, and `Cascade.acall` or `Cascade.amap(queries, max_concurrency=...)` await them.

```python
from ucci import UCCIRouter
from ucci.integrations.cascade import Cascade
from ucci.integrations.openai import make_chat_fn

# A fitted router (UCCIRouter.load("router.json") in practice); this one is a placeholder.
router = UCCIRouter.from_dict({
    "format": "ucci-router", "version": 1,
    "calibrator": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
    "theta": 0.1, "c_small": 1.0, "c_large": 3.02, "cost_model": "routing",
})
cascade = Cascade(
    router,
    small_fn=make_chat_fn(client, "gpt-4o-mini", top_logprobs=2, max_tokens=256),
    large_fn=make_chat_fn(client, "gpt-4o", top_logprobs=0, return_response=False, max_tokens=256),
    signal_fn=signals_from_chat_completion,
)
result = cascade("Extract the named entities: ...")
print(result.escalated, result.p_hat, result.u, result.answer)
```

[`examples/openai_cascade.py`](https://github.com/varunkotte6/ucci/blob/main/examples/openai_cascade.py)
is a complete script that also logs every query for later labelling; see
[collecting data from live traffic](live_traffic.md).

API reference: [`ucci.integrations.openai`](../api/integrations.md#ucci.integrations.openai).
