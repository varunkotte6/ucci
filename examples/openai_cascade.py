"""A UCCI cascade over an OpenAI-compatible Chat Completions API.

The small model is called with ``temperature=0, logprobs=True,
top_logprobs=2``; UCCI computes u(x) from the returned top-2 log-probabilities
(paper Section 4.1, Eq. 4), the router maps it to p_hat and escalates to the
large model when p_hat > theta (Eq. 6). Every result is also written as a
JSONL record in the package's traffic format, the data ``ucci fit`` needs
once the records carry correctness labels.

Needs ``pip install openai`` and an API key in ``OPENAI_API_KEY``. The small
model must return log-probabilities for chat completions; pass model names
that do with ``--small`` and ``--large``. For a self-hosted OpenAI-compatible
server (``vllm serve``, llama.cpp), pass ``--base-url`` and
``--server-reports-eos``: those servers include the terminating EOS token in
the log-probabilities, and UCCI's token convention excludes it
(``docs/paper_mapping.md``).

Pass a router fitted on your own labelled traffic with ``--router``; without
it a PLACEHOLDER router (identity calibration, theta = 0.1) shows the
mechanics.

    export OPENAI_API_KEY=...
    python examples/openai_cascade.py --small gpt-4o-mini --large gpt-4o
"""

from __future__ import annotations

import argparse
import functools
import os

from openai import OpenAI

from ucci import UCCIRouter
from ucci.integrations.cascade import Cascade, JsonlLogger
from ucci.integrations.openai import make_chat_fn, signals_from_chat_completion

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
    parser.add_argument("--small", default="gpt-4o-mini", help="small model; must support logprobs")
    parser.add_argument("--large", default="gpt-4o", help="large model")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible server URL")
    parser.add_argument(
        "--server-reports-eos",
        action="store_true",
        help="drop the terminating EOS token that vLLM and llama.cpp servers report",
    )
    parser.add_argument(
        "--router", default=None, help="router JSON from `ucci fit` or UCCIRouter.save"
    )
    parser.add_argument(
        "--log", default="cascade_log.jsonl", help="JSONL file for the routed queries"
    )
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY") and args.base_url is None:
        raise SystemExit("set OPENAI_API_KEY (or pass --base-url for a local server)")

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

    client = OpenAI(base_url=args.base_url) if args.base_url else OpenAI()
    cascade = Cascade(
        router,
        # (text, response) with top-2 log-probabilities, greedy decoding.
        small_fn=make_chat_fn(client, args.small, top_logprobs=2, max_tokens=256),
        # Text only; the large model needs no log-probabilities.
        large_fn=make_chat_fn(
            client, args.large, top_logprobs=0, return_response=False, max_tokens=256
        ),
        signal_fn=functools.partial(
            signals_from_chat_completion, drop_stop_token=args.server_reports_eos
        ),
    )
    with JsonlLogger(args.log) as log:
        for i, query in enumerate(QUERIES):
            r = cascade(PROMPT.format(query=query))
            log.write(r.to_record(f"q{i}", include_answers=True))  # answers kept for labelling
            print(
                f"\n{query}\n  u = {r.u:.4f}, p_hat = {r.p_hat:.4f} -> {'large' if r.escalated else 'small'} model"
            )
            answer = " ".join(str(r.answer).split())
            print("  answer:", answer if len(answer) <= 200 else answer[:200] + " [...]")
    print(
        f"\nlogged {len(QUERIES)} records to {args.log}; add small_correct/large_correct labels and run `ucci fit`"
    )


if __name__ == "__main__":
    main()
