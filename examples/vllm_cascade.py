"""A batched UCCI cascade with vLLM offline inference (CUDA).

The small model answers the whole batch with greedy decoding and top-2
log-probabilities; UCCI computes u(x) per query (paper Section 4.1, Eq. 4),
the router escalates the queries with p_hat > theta (Eq. 6), and only those
go to the large model, in one second batch. The paper served its models with
vLLM (Section 6.1).

Both models share one GPU here, so each engine gets part of its memory; put
them on separate GPUs or servers in production. Pass a router fitted on
your own labelled traffic with ``--router`` (see ``ucci fit``); without it a
PLACEHOLDER router (identity calibration, theta = 0.1) shows the mechanics.

    pip install vllm "ucci"
    python examples/vllm_cascade.py --router router.json
"""

from __future__ import annotations

import argparse

from vllm import LLM, SamplingParams

from ucci import UCCIRouter
from ucci.integrations.vllm import greedy_sampling_params, signals_from_vllm

# The prompt of benchmarks/conll2003 (conll.PROMPT_TEMPLATE), so a router fitted on
# that benchmark's joined.jsonl applies: a router transfers only to the prompt,
# models and decoding it was fitted on.
PROMPT = (
    "Extract the named entities from this sentence.\n"
    "Return only a JSON object with fields: PER, ORG, LOC, MISC.\n"
    "Each field is a list of names copied exactly from the sentence; "
    "use an empty list when there are none.\n"
    "PER: people. ORG: organizations, companies, sports teams, political parties. "
    "LOC: countries, cities, regions and other places. "
    "MISC: other proper names, such as nationalities, languages, events and titles.\n"
    "Only include proper names. Do not include numbers, dates, scores or common nouns.\n"
    "\n"
    "Sentence: {query}\n"
    "Output:"
)
# The first sentences of the CoNLL-2003 train split, as tokenized there.
QUERIES = [
    "EU rejects German call to boycott British lamb .",
    "Peter Blackburn",
    "The European Commission said on Thursday it disagreed with German advice to consumers to shun "
    "British lamb until scientists determine whether mad cow disease can be transmitted to sheep .",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--small", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--large", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--router", default=None, help="router JSON from `ucci fit` or UCCIRouter.save"
    )
    parser.add_argument(
        "--gpu-memory", type=float, default=0.4, help="fraction of GPU memory per engine"
    )
    args = parser.parse_args()

    if args.router:
        router = UCCIRouter.load(args.router)
    else:
        print(
            "Using a PLACEHOLDER router (identity calibration, theta = 0.1); pass --router for a fitted one."
        )
        router = UCCIRouter.from_dict(
            {
                "format": "ucci-router",
                "version": 1,
                "calibrator": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
                "theta": 0.1,
                "c_small": 1.0,
                "c_large": 3.02,
                "cost_model": "routing",
            }
        )

    messages = [[{"role": "user", "content": PROMPT.format(query=q)}] for q in QUERIES]

    small = LLM(model=args.small, gpu_memory_utilization=args.gpu_memory, seed=0)
    small_out = small.chat(
        messages, greedy_sampling_params(max_tokens=256, logprobs=2), use_tqdm=False
    )
    # signals_from_vllm drops the terminating EOS position, which vLLM reports (docs/paper_mapping.md).
    u = [signals_from_vllm(o).u for o in small_out]
    decision = router.route(u)

    answers = [o.outputs[0].text for o in small_out]
    escalated = [i for i, e in enumerate(decision.escalate) if e]
    if escalated:
        large = LLM(model=args.large, gpu_memory_utilization=args.gpu_memory, seed=0)
        large_out = large.chat(
            [messages[i] for i in escalated],
            SamplingParams(temperature=0.0, max_tokens=256),
            use_tqdm=False,
        )
        for i, o in zip(escalated, large_out):
            answers[i] = o.outputs[0].text

    for q, ui, p, e, a in zip(QUERIES, u, decision.p_hat, decision.escalate, answers):
        print(f"\n{q}\n  u = {ui:.4f}, p_hat = {p:.4f} -> {'large' if e else 'small'} model")
        answer = " ".join(a.split())
        print("  answer:", answer if len(answer) <= 200 else answer[:200] + " [...]")
    print(f"\nescalated {len(escalated)} of {len(QUERIES)} queries")


if __name__ == "__main__":
    main()
