# Hugging Face transformers

With the full per-step logits available, `ucci.integrations.transformers` computes the top-2
probabilities exactly, with a softmax over the whole vocabulary, and returns one
`TokenSignals` per batch item. It also returns the full-vocabulary mean entropy and mean max
probability used by the paper's entropy baseline and signal ablation. Install with
`pip install "ucci-router[transformers]"` (transformers 4.38 or newer returns the raw logits).

## One call

`generate_with_signals` tokenizes with left padding, generates greedily with per-step logits
and returns the decoded texts with their signals. Apply the chat template first for chat
models. Instruction models often ship a generation config with sampling settings or a
repetition penalty; `generate_with_signals` fixes `do_sample=False` and `num_beams=1`, and a
repetition penalty of 1.0 keeps the generated token equal to the arg-max of the logits the
signal is computed from (`TokenSignals.n_non_greedy` counts any position where it is not).

<!-- snippet: requires torch transformers -->
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from ucci.integrations.transformers import generate_with_signals

name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

sentences = ["EU rejects German call to boycott British lamb .", "Peter Blackburn"]
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": f"List the named entities in this sentence as JSON: {s}"}],
        tokenize=False, add_generation_prompt=True)
    for s in sentences
]
gen = generate_with_signals(model, tokenizer, prompts, max_new_tokens=64, repetition_penalty=1.0)
for text, sig in zip(gen.texts, gen.signals):
    print(f"u = {sig.u:.4f} over {sig.n_tokens} tokens, non-greedy steps {sig.n_non_greedy}: {text!r}")
```

`gen.u` is the list of \(u(x)\) values and `gen.output` the raw `generate` output.
`max_new_tokens` defaults to 256, the paper's limit (Appendix B.2).

## From your own `generate` call

Ask for greedy decoding and the raw logits, then pass the output and the EOS id(s):

<!-- snippet: requires torch transformers -->
```python
from ucci.integrations.transformers import signals_from_generate

tokenizer.padding_side = "left"
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
out = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=64,
                     repetition_penalty=1.0, return_dict_in_generate=True, output_logits=True)
signals = signals_from_generate(out, eos_token_id=model.generation_config.eos_token_id)
print([round(s.u, 4) for s in signals])
```

`output_logits=True` returns the raw logits. `output_scores=True` returns the scores after the
logits processors (repetition penalty, forced tokens and so on), which are not the model's
next-token distribution; the adapter prefers `logits` and falls back to `scores` only when
logits were not requested (`source="auto"`; `source="logits"` or `"scores"` forces one).

## Which tokens count

For each row the adapter keeps the steps before the first EOS token (any id in
`eos_token_id`). `generate` pads a row that finished early with the pad id after its EOS, so
padding is excluded too. A row that ends without EOS (a `stop_strings` or `stopping_criteria`
stop) is followed directly by padding: pass `pad_token_id` and a trailing run of the pad id is
excluded as well. `generate_with_signals` does this automatically when `stop_strings` or
`stopping_criteria` is given. Otherwise leave `pad_token_id` out, because a model may generate
the pad id as content. Beam search is rejected. An empty generation raises by default;
`on_empty="nan"` returns NaN instead, and such a query should go to the large model.

Nothing in the adapter imports torch: each step is copied to host memory and promoted to
float64 (bfloat16 and float16 through float32, which is exact), so it also reads NumPy arrays or
other array-likes of the same layout.

[`examples/transformers_cascade.py`](https://github.com/varunkotte6/ucci/blob/main/examples/transformers_cascade.py)
runs a live two-model cascade on CPU, MPS or CUDA.

API reference: [`ucci.integrations.transformers`](../api/integrations.md#ucci.integrations.transformers).
