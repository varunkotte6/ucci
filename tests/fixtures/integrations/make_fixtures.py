"""Regenerate the provider payload fixtures used by tests/test_integrations.py.

Run ``python tests/fixtures/integrations/make_fixtures.py``. The payloads are
synthetic: no API or server was called. Each follows the response format
documented by its provider (see the docstrings of the matching modules in
``ucci.integrations`` for the sources), and every per-token probability was
chosen by hand so the expected u(x) (Section 4.1, Eq. 4) can be written down
in the tests. Log-probabilities are ``math.log`` of those probabilities.
"""

from __future__ import annotations

import json
import math
import os

OUT = os.path.dirname(os.path.abspath(__file__))
L = math.log


def b(tok):
    return list(tok.encode("utf-8"))


def dump(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


# ---- OpenAI Chat Completions --------------------------------------------------
# (token, [(candidate, prob), ...]) with the generated token first (greedy).
chat_tokens = [
    ('{"', [('{"', 0.98), ("{", 0.01), ("```", 0.005)]),
    ("camera", [("camera", 0.90), ("lens", 0.06), ("aperture", 0.02)]),
    ('":"', [('":"', 0.75), ('":', 0.20), ('":["', 0.03)]),
    ("Canon", [("Canon", 0.55), ("Nikon", 0.40), ("Sony", 0.02)]),
    ('"}', [('"}', 0.97), ('",', 0.02), ('"', 0.005)]),
]


def chat_entry(tok, cands):
    return {
        "token": tok,
        "logprob": L(cands[0][1]),
        "bytes": b(tok),
        "top_logprobs": [{"token": c, "logprob": L(p), "bytes": b(c)} for c, p in cands],
    }


text = "".join(t for t, _ in chat_tokens)
dump(
    "openai_chat_completion.json",
    {
        "id": "chatcmpl-B9MHDbslfkBeAs8l4bebGdFOJ6PeG",
        "object": "chat.completion",
        "created": 1741570283,
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                    "refusal": None,
                    "annotations": [],
                },
                "logprobs": {
                    "content": [chat_entry(t, c) for t, c in chat_tokens],
                    "refusal": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 41,
            "completion_tokens": 5,
            "total_tokens": 46,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
        "service_tier": "default",
        "system_fingerprint": "fp_06737a9306",
    },
)

# vLLM OpenAI server: same content tokens, top_logprobs=2, plus the EOS entry,
# vLLM's extra "stop_reason" field (null = stopped on EOS).
vllm_tokens = [(t, c[:2]) for t, c in chat_tokens] + [
    ("<|im_end|>", [("<|im_end|>", 0.99), ("\n", 0.004)])
]
dump(
    "openai_chat_completion_vllm.json",
    {
        "id": "chatcmpl-5a0c6b1e2f7d4e0f9c8b7a6d5e4f3a2b",
        "object": "chat.completion",
        "created": 1758700000,
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                    "refusal": None,
                    "annotations": None,
                    "audio": None,
                    "function_call": None,
                    "tool_calls": [],
                    "reasoning_content": None,
                },
                "logprobs": {"content": [chat_entry(t, c) for t, c in vllm_tokens]},
                "finish_reason": "stop",
                "stop_reason": None,
            }
        ],
        "service_tier": None,
        "system_fingerprint": None,
        "usage": {
            "prompt_tokens": 38,
            "total_tokens": 44,
            "completion_tokens": 6,
            "prompt_tokens_details": None,
        },
        "prompt_logprobs": None,
        "kv_transfer_params": None,
    },
)

# Streamed chat completion: one chunk per token, then a finish chunk and a usage chunk.
chunks = []
base = {
    "id": "chatcmpl-B9MHDbslfkBeAs8l4bebGdFOJ6PeG",
    "object": "chat.completion.chunk",
    "created": 1741570283,
    "model": "gpt-4o-mini-2024-07-18",
    "system_fingerprint": "fp_06737a9306",
}
chunks.append(
    {
        **base,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "", "refusal": None},
                "logprobs": {"content": [], "refusal": None},
                "finish_reason": None,
            }
        ],
        "usage": None,
    }
)
for t, c in chat_tokens:
    chunks.append(
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": t},
                    "logprobs": {"content": [chat_entry(t, c)], "refusal": None},
                    "finish_reason": None,
                }
            ],
            "usage": None,
        }
    )
chunks.append(
    {
        **base,
        "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}],
        "usage": None,
    }
)
chunks.append(
    {
        **base,
        "choices": [],
        "usage": {"prompt_tokens": 41, "completion_tokens": 5, "total_tokens": 46},
    }
)
dump("openai_chat_completion_chunks.json", chunks)

# ---- OpenAI Completions (legacy) ---------------------------------------------
comp = [
    (" Canon", [(" Canon", 0.6), (" Nikon", 0.3)]),
    (" EOS", [(" EOS", 0.8), (" 5", 0.1)]),
    (" R5", [(" R5", 0.7), (" R6", 0.25)]),
]
offs, pos = [], 30
for t, _ in comp:
    offs.append(pos)
    pos += len(t)
dump(
    "openai_completion.json",
    {
        "id": "cmpl-8f3a7b2c9d1e4f5a6b7c8d9e0f1a2b3c",
        "object": "text_completion",
        "created": 1741570300,
        "model": "gpt-3.5-turbo-instruct",
        "choices": [
            {
                "text": "".join(t for t, _ in comp),
                "index": 0,
                "logprobs": {
                    "tokens": [t for t, _ in comp],
                    "token_logprobs": [L(c[0][1]) for _, c in comp],
                    "top_logprobs": [{tok: L(p) for tok, p in c} for _, c in comp],
                    "text_offset": offs,
                },
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
    },
)

# ---- OpenAI Responses ----------------------------------------------------------
resp = [
    ('{"', [('{"', 0.9), ("{", 0.05)]),
    ("iso", [("iso", 0.6), ("ISO", 0.35)]),
    ('":"', [('":"', 0.8), ('":', 0.15)]),
]


def resp_entry(tok, cands):
    return {
        "token": tok,
        "bytes": b(tok),
        "logprob": L(cands[0][1]),
        "top_logprobs": [{"token": c, "bytes": b(c), "logprob": L(p)} for c, p in cands],
    }


dump(
    "openai_responses.json",
    {
        "id": "resp_67ccd2bed1ec8190b14f964abc0542670bb6a6b452d3795b",
        "object": "response",
        "created_at": 1741476542,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": "gpt-4.1-mini-2025-04-14",
        "output": [
            {
                "type": "message",
                "id": "msg_67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "".join(t for t, _ in resp),
                        "annotations": [],
                        "logprobs": [resp_entry(t, c) for t, c in resp],
                    }
                ],
            }
        ],
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": True,
        "temperature": 0.0,
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_logprobs": 2,
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 36,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 39,
        },
        "user": None,
        "metadata": {},
    },
)

# ---- vLLM offline RequestOutput (as a JSON dump; token ids become string keys) ----
vt = [
    (515, '{"', [(515, '{"', 0.95), (90, "{", 0.03)]),
    (43023, "camera", [(43023, "camera", 0.7), (18449, "lens", 0.2)]),
    (3252, '":"', [(3252, '":"', 0.5), (788, '":', 0.45)]),
    (6713, "Canon", [(6713, "Canon", 0.88), (45, "N", 0.1)]),
    (151645, "<|im_end|>", [(151645, "<|im_end|>", 0.999), (198, "\n", 0.0005)]),
]
lps = []
for _tid, _tok, cands in vt:
    lps.append(
        {
            str(i): {"logprob": L(p), "rank": r + 1, "decoded_token": d}
            for r, (i, d, p) in enumerate(cands)
        }
    )
dump(
    "vllm_request_output.json",
    {
        "request_id": "0",
        "prompt": (
            "Extract entities from this photo search query.\n"
            "Return JSON with fields: camera, lens,\n"
            "aperture, shutter_speed, iso, focal_length.\n\n"
            "Query: canon with 50mm\nOutput:\n"
        ),
        "prompt_token_ids": [28959, 14744, 504, 419, 6548, 2711, 3239, 13],
        "prompt_logprobs": None,
        "outputs": [
            {
                "index": 0,
                "text": '{"camera":"Canon',
                "token_ids": [tid for tid, _, _ in vt],
                "cumulative_logprob": sum(L(c[0][2]) for _, _, c in vt),
                "logprobs": lps,
                "finish_reason": "stop",
                "stop_reason": None,
            }
        ],
        "finished": True,
    },
)

# ---- llama.cpp /completion ------------------------------------------------------
lc = [
    (8290, " Canon", [(8290, " Canon", 0.8), (37231, " Nikon", 0.15)]),
    (220, " ", [(220, " ", 0.6), (10, "5", 0.3)]),
    (20, "5", [(20, "5", 0.9), (21, "6", 0.05)]),
    (2, "", [(2, "", 0.97), (13, "\n", 0.01)]),
]


def lc_entry(tid, tok, cands):
    return {
        "id": tid,
        "token": tok,
        "bytes": b(tok),
        "logprob": L(cands[0][2]),
        "top_logprobs": [
            {"id": i, "token": d, "bytes": b(d), "logprob": L(p)} for i, d, p in cands
        ],
    }


settings = {
    "n_predict": 256,
    "seed": 4294967295,
    "temperature": 0.800000011920929,
    "dynatemp_range": 0.0,
    "dynatemp_exponent": 1.0,
    "top_k": 1,
    "top_p": 0.949999988079071,
    "min_p": 0.05000000074505806,
    "n_probs": 2,
    "min_keep": 0,
    "post_sampling_probs": False,
    "stop": [],
    "samplers": ["penalties", "dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"],
}
final = {
    "index": 0,
    "content": " Canon 5",
    "tokens": [],
    "id_slot": 0,
    "stop": True,
    "model": "gpt-3.5-turbo",
    "tokens_predicted": 4,
    "tokens_evaluated": 21,
    "generation_settings": settings,
    "prompt": "<s> Query: canon 5d\nCamera:",
    "has_new_line": False,
    "truncated": False,
    "stop_type": "eos",
    "stopping_word": "",
    "tokens_cached": 24,
    "timings": {
        "prompt_n": 21,
        "prompt_ms": 12.3,
        "prompt_per_token_ms": 0.59,
        "prompt_per_second": 1707.3,
        "predicted_n": 4,
        "predicted_ms": 31.9,
        "predicted_per_token_ms": 7.98,
        "predicted_per_second": 125.4,
    },
    "completion_probabilities": [lc_entry(*x) for x in lc],
}
dump("llamacpp_completion.json", final)

post = dict(final)
post["generation_settings"] = {**settings, "post_sampling_probs": True}
post["completion_probabilities"] = [
    {
        "id": tid,
        "token": tok,
        "bytes": b(tok),
        "prob": 1.0,
        "top_probs": [{"id": tid, "token": tok, "bytes": b(tok), "prob": 1.0}],
    }
    for tid, tok, _ in lc
]
dump("llamacpp_completion_post_sampling.json", post)

events = []
for k, x in enumerate(lc):
    events.append(
        {
            "index": 0,
            "content": x[1],
            "tokens": [x[0]],
            "stop": False,
            "id_slot": 0,
            "tokens_predicted": k + 1,
            "tokens_evaluated": 21,
            "completion_probabilities": [lc_entry(*x)],
        }
    )
ev_final = {k: v for k, v in final.items() if k != "completion_probabilities"}
ev_final["content"] = ""
events.append(ev_final)
dump("llamacpp_stream.json", events)

legacy = [
    (" Canon", [(" Canon", 0.7), (" Nikon", 0.2)]),
    (" EOS", [(" EOS", 0.85), (" R", 0.1)]),
    (" R5", [(" R5", 0.6), (" R6", 0.3)]),
    ("", [("", 0.99), ("\n", 0.005)]),
]
dump(
    "llamacpp_completion_legacy.json",
    {
        "content": " Canon EOS R5",
        "id_slot": 0,
        "stop": True,
        "model": "models/7B/ggml-model-q4_0.gguf",
        "tokens_predicted": 4,
        "tokens_evaluated": 18,
        "generation_settings": {
            "n_ctx": 4096,
            "n_predict": 256,
            "model": "models/7B/ggml-model-q4_0.gguf",
            "seed": 4294967295,
            "temperature": -1.0,
            "top_k": 40,
            "top_p": 0.95,
            "n_probs": 2,
            "min_keep": 0,
        },
        "prompt": "Query: canon eos r5\nCamera:",
        "has_new_line": False,
        "truncated": False,
        "stopped_eos": True,
        "stopped_word": False,
        "stopped_limit": False,
        "stopping_word": "",
        "tokens_cached": 21,
        "timings": {"prompt_n": 18, "prompt_ms": 10.1, "predicted_n": 4, "predicted_ms": 30.2},
        "index": 0,
        "completion_probabilities": [
            {"content": tok, "probs": [{"tok_str": c, "prob": p} for c, p in cands]}
            for tok, cands in legacy
        ],
    },
)
print(f"wrote fixtures to {OUT}")
