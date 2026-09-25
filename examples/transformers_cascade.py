"""A UCCI cascade with Hugging Face transformers, run locally.

The small model answers every query with greedy decoding; UCCI turns its
top-2 token probabilities into u(x) (paper Section 4.1, Eq. 4), the router
maps u(x) to p_hat and escalates to the large model when p_hat > theta
(Eq. 6). Defaults: Qwen2.5-0.5B-Instruct as the small model and
Qwen2.5-1.5B-Instruct as the large one (about 4 GB downloaded on first run;
runs on CPU, Apple MPS or CUDA).

A router must be fitted on labelled traffic from your own workload (both
models run on a calibration and a validation split). Pass one with
``--router``, for example the output of::

    ucci fit --data benchmarks/conll2003/runs/full/joined.jsonl --tau 0.6 --out router.json

Without ``--router`` the script uses a PLACEHOLDER router (identity
calibration, theta = 0.1), which only demonstrates the mechanics.

    python examples/transformers_cascade.py
    python examples/transformers_cascade.py --router router.json
"""

from __future__ import annotations

import argparse
from typing import Any, Tuple

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ucci import UCCIRouter
from ucci.integrations import TokenSignals
from ucci.integrations.cascade import Cascade
from ucci.integrations.transformers import generate_with_signals

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


def load(name: str, device: str) -> Tuple[Any, Any]:
    """Tokenizer and model in bfloat16 on an accelerator, float32 on CPU."""
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    key = (
        "dtype"
        if tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56)
        else "torch_dtype"
    )
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, **{key: dtype}).to(device).eval()
    return tok, model


def model_fn(tok: Any, model: Any, with_signals: bool) -> Any:
    """``query -> answer`` (large model) or ``query -> (answer, TokenSignals)`` (small model)."""

    def fn(query: str) -> Any:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(query=query)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # Greedy decoding on unmodified logits: switch off the repetition penalty
        # that some instruction models ship in their generation config.
        out = generate_with_signals(
            model, tok, [prompt], max_new_tokens=256, repetition_penalty=1.0
        )
        return (out.texts[0], out.signals[0]) if with_signals else out.texts[0]

    return fn


def placeholder_router() -> UCCIRouter:
    """Identity calibration (p_hat = u) and theta = 0.1. Mechanics only; fit a real router."""
    return UCCIRouter.from_dict(
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--small", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--large", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--router", default=None, help="router JSON from `ucci fit` or UCCIRouter.save"
    )
    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    router = UCCIRouter.load(args.router) if args.router else placeholder_router()
    if not args.router:
        print(
            "Using a PLACEHOLDER router (identity calibration, theta = 0.1); pass --router for a fitted one."
        )

    small_tok, small_model = load(args.small, device)
    large_tok, large_model = load(args.large, device)
    cascade = Cascade(
        router,
        small_fn=model_fn(small_tok, small_model, with_signals=True),
        large_fn=model_fn(large_tok, large_model, with_signals=False),
        signal_fn=lambda signals: signals,  # small_fn already returns TokenSignals
    )
    for query in QUERIES:
        r = cascade(query)
        s: TokenSignals = r.signals  # type: ignore[assignment]
        who = "large" if r.escalated else "small"
        print(
            f"\n{query}\n  u = {r.u:.4f} over {s.n_tokens} tokens, p_hat = {r.p_hat:.4f} -> {who} model "
            f"(non-greedy steps: {s.n_non_greedy})"
        )
        answer = " ".join(str(r.answer).split())
        print("  answer:", answer if len(answer) <= 200 else answer[:200] + " [...]")


if __name__ == "__main__":
    main()
