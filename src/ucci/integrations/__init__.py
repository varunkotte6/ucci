"""Serving-stack adapters for UCCI: u(x) from real model responses, and a cascade runner.

UCCI routes on the small model's token-margin uncertainty u(x) (Section 4.1,
Eq. 4), computed from the top-2 next-token probabilities of its greedy
generation. Each submodule reads those probabilities from one serving stack
and returns :class:`TokenSignals` (u(x) plus the entropy and max-probability
signals used by the paper's baselines and ablations):

* :mod:`ucci.integrations.openai`: OpenAI Chat Completions, Completions and
  Responses payloads, and OpenAI-compatible servers (vLLM, llama.cpp,
  LiteLLM);
* :mod:`ucci.integrations.vllm`: vLLM offline ``RequestOutput``;
* :mod:`ucci.integrations.transformers`: Hugging Face ``generate`` outputs
  (exact softmax over the full vocabulary, batched);
* :mod:`ucci.integrations.llamacpp`: the llama.cpp server's native
  ``/completion`` endpoint;
* :mod:`ucci.integrations.cascade`: :class:`~ucci.integrations.cascade.Cascade`,
  which runs the small model, routes with a fitted router and calls the
  large model when needed (sync and async), and a JSONL logger for
  collecting calibration data from live traffic.

Importing this package or any submodule imports no optional dependency:
payloads are read by duck typing, and ``torch``, ``transformers`` and
``vllm`` are imported only inside the helpers that run a model.
"""

from __future__ import annotations

from ._common import PayloadError, TokenSignals

__all__ = ["PayloadError", "TokenSignals"]
