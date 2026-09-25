"""Tests for ucci.integrations: provider adapters (u(x), Eq. 4) and the cascade runner.

No test touches the network. Payload fixtures in tests/fixtures/integrations/
follow the documented response formats of each provider (see the module
docstrings for the sources); their probabilities were chosen so that the
expected u(x) can be written down by hand. Tests that need torch and
transformers run a tiny random model and skip cleanly when either is missing.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import math
import pathlib
import subprocess
import sys
import threading
import types
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from ucci import token_margin_uncertainty
from ucci.integrations import PayloadError, TokenSignals
from ucci.integrations import cascade as cc
from ucci.integrations import llamacpp as lc
from ucci.integrations import openai as oa
from ucci.integrations import transformers as hf
from ucci.integrations import vllm as vl

FIX = pathlib.Path(__file__).parent / "fixtures" / "integrations"
SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
TOL = 1e-12


def load(name: str) -> Any:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def ns(obj: Any) -> Any:
    """Recursively turn dicts into attribute objects, like SDK models."""
    if isinstance(obj, dict):
        return types.SimpleNamespace(**{k: ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [ns(v) for v in obj]
    return obj


def u_of(margins: Sequence[float]) -> float:
    return 1.0 - sum(margins) / len(margins)


def trunc_entropy(probs: Sequence[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0)


# Hand-written expectations for the fixtures (probabilities as generated).
CHAT_PROBS = [
    (0.98, 0.01, 0.005),
    (0.90, 0.06, 0.02),
    (0.75, 0.20, 0.03),
    (0.55, 0.40, 0.02),
    (0.97, 0.02, 0.005),
]
CHAT_U = 1.0 - (0.97 + 0.84 + 0.55 + 0.15 + 0.95) / 5  # 0.308
COMPLETION_U = 1.0 - (0.3 + 0.7 + 0.45) / 3
RESPONSES_U = 1.0 - (0.85 + 0.25 + 0.65) / 3
VLLM_U = 1.0 - (0.92 + 0.5 + 0.05 + 0.78) / 4  # 0.4375, EOS dropped
LLAMACPP_U = 1.0 - (0.65 + 0.3 + 0.85) / 3  # 0.4, EOS dropped
LLAMACPP_LEGACY_U = 1.0 - (0.5 + 0.75 + 0.3) / 3


# ---------------------------------------------------------------------------
# Package hygiene
# ---------------------------------------------------------------------------


def test_importing_integrations_loads_no_optional_dependency() -> None:
    code = (
        "import sys\n"
        "import ucci.integrations, ucci.integrations.openai, ucci.integrations.vllm\n"
        "import ucci.integrations.transformers, ucci.integrations.llamacpp\n"
        "import ucci.integrations.cascade\n"
        "bad = [m for m in ('torch', 'transformers', 'vllm', 'openai', 'matplotlib', "
        "'sklearn') if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    env = {"PYTHONPATH": str(SRC), "PYTHONNOUSERSITE": "1", "PATH": "/usr/bin:/bin"}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert out.stdout.strip() == ""


def test_payload_error_is_value_error() -> None:
    assert issubclass(PayloadError, ValueError)


def test_fixtures_have_no_long_dashes() -> None:
    for path in FIX.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert chr(0x2014) not in text, path.name
        assert chr(0x2013) not in text, path.name


# ---------------------------------------------------------------------------
# OpenAI Chat Completions
# ---------------------------------------------------------------------------


def test_chat_completion_matches_hand_computed_u() -> None:
    sig = oa.signals_from_chat_completion(load("openai_chat_completion.json"))
    assert sig.u == pytest.approx(CHAT_U, abs=TOL)
    assert sig.u == pytest.approx(0.308, abs=TOL)
    assert sig.n_tokens == 5
    assert sig.margins == pytest.approx((0.97, 0.84, 0.55, 0.15, 0.95), abs=TOL)
    assert sig.mean_max_prob == pytest.approx(np.mean([p[0] for p in CHAT_PROBS]), abs=TOL)
    assert sig.mean_entropy == pytest.approx(
        np.mean([trunc_entropy(p) for p in CHAT_PROBS]), abs=TOL
    )
    assert sig.entropy_support == "top_k"
    assert sig.top_k == 3
    assert sig.n_non_greedy == 0
    assert sig.source == "openai.chat_completion"
    assert oa.u_from_chat_completion(load("openai_chat_completion.json")) == sig.u


def test_chat_completion_agrees_with_core_signal() -> None:
    top2 = [(p[0], p[1]) for p in CHAT_PROBS]
    assert oa.u_from_chat_completion(load("openai_chat_completion.json")) == pytest.approx(
        token_margin_uncertainty(top2), abs=TOL
    )


def test_chat_completion_accepts_objects_choice_and_content_list() -> None:
    payload = load("openai_chat_completion.json")
    choice = payload["choices"][0]
    for obj in (
        ns(payload),
        choice,
        ns(choice),
        choice["logprobs"]["content"],
        ns(choice["logprobs"]["content"]),
    ):
        assert oa.u_from_chat_completion(obj) == pytest.approx(CHAT_U, abs=TOL)


def test_chat_completion_with_openai_sdk_models() -> None:
    types_mod = pytest.importorskip("openai.types.chat")
    payload = load("openai_chat_completion_vllm.json")
    resp = types_mod.ChatCompletion.model_validate(payload)
    assert oa.u_from_chat_completion(resp, drop_stop_token=True) == pytest.approx(CHAT_U, abs=TOL)
    resp = types_mod.ChatCompletion.model_validate(load("openai_chat_completion.json"))
    assert oa.u_from_openai(resp) == pytest.approx(CHAT_U, abs=TOL)


def test_chat_renormalized_entropy() -> None:
    sig = oa.signals_from_chat_completion(
        load("openai_chat_completion.json"), renormalize_entropy=True
    )
    expected = np.mean([trunc_entropy([q / sum(p) for q in p]) for p in CHAT_PROBS])
    assert sig.mean_entropy == pytest.approx(expected, abs=TOL)
    assert sig.entropy_support == "top_k_renormalized"
    assert sig.u == pytest.approx(CHAT_U, abs=TOL)


def test_vllm_openai_server_stop_token_handling() -> None:
    payload = load("openai_chat_completion_vllm.json")
    kept = oa.signals_from_chat_completion(payload)
    assert kept.n_tokens == 6
    assert kept.u == pytest.approx(u_of([0.97, 0.84, 0.55, 0.15, 0.95, 0.99 - 0.004]), abs=TOL)
    dropped = oa.signals_from_chat_completion(payload, drop_stop_token=True)
    assert dropped.n_tokens == 5
    assert dropped.u == pytest.approx(CHAT_U, abs=TOL)
    assert dropped.top_k == 2
    # Hitting the token limit means no stop token was generated: keep everything.
    payload["choices"][0]["finish_reason"] = "length"
    assert oa.signals_from_chat_completion(payload, drop_stop_token=True).n_tokens == 6


def test_drop_stop_token_on_bare_list_is_unconditional() -> None:
    content = load("openai_chat_completion_vllm.json")["choices"][0]["logprobs"]["content"]
    assert oa.signals_from_chat_completion(content, drop_stop_token=True).n_tokens == 5


def test_chat_chunks_match_full_response() -> None:
    chunks = load("openai_chat_completion_chunks.json")
    sig = oa.signals_from_chat_completion_chunks(chunks)
    assert sig.u == pytest.approx(CHAT_U, abs=TOL)
    assert sig.n_tokens == 5
    assert sig.source == "openai.chat_completion_chunks"
    assert oa.u_from_chat_completion_chunks(ns(chunks)) == pytest.approx(CHAT_U, abs=TOL)
    assert oa.signals_from_chat_completion_chunks(chunks, drop_stop_token=True).n_tokens == 4
    assert oa.u_from_openai(chunks) == pytest.approx(CHAT_U, abs=TOL)


def test_chat_chunks_error_path_names_the_chunk() -> None:
    chunks = load("openai_chat_completion_chunks.json")
    chunks[3]["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = [
        {"token": "x", "logprob": -0.1}
    ]
    with pytest.raises(PayloadError, match=r"chunks\[3\]\.choices\[0\]\.logprobs\.content\[0\]"):
        oa.signals_from_chat_completion_chunks(chunks)
    with pytest.raises(PayloadError, match="no generated content tokens"):
        oa.signals_from_chat_completion_chunks([])


def test_chat_choice_index_selects_the_right_choice() -> None:
    payload = load("openai_chat_completion.json")
    second = copy.deepcopy(payload["choices"][0])
    second["index"] = 1
    for entry in second["logprobs"]["content"]:
        entry["top_logprobs"] = [
            {"token": "a", "logprob": math.log(0.5)},
            {"token": "b", "logprob": math.log(0.5)},
        ]
    payload["choices"].append(second)
    assert oa.u_from_chat_completion(payload, choice_index=1) == pytest.approx(1.0, abs=TOL)
    assert oa.u_from_chat_completion(payload, choice_index=0) == pytest.approx(CHAT_U, abs=TOL)
    with pytest.raises(PayloadError, match="no choice with index 2"):
        oa.u_from_chat_completion(payload, choice_index=2)


def test_chat_refusal_part() -> None:
    payload = load("openai_chat_completion.json")
    lp = payload["choices"][0]["logprobs"]
    lp["refusal"], lp["content"] = lp["content"], None
    with pytest.raises(PayloadError, match="part='refusal'"):
        oa.u_from_chat_completion(payload)
    assert oa.u_from_chat_completion(payload, part="refusal") == pytest.approx(CHAT_U, abs=TOL)
    with pytest.raises(ValueError, match="part must be one of"):
        oa.u_from_chat_completion(payload, part="reasoning")


def test_chat_missing_logprobs_explains_the_request() -> None:
    payload = load("openai_chat_completion.json")
    payload["choices"][0]["logprobs"] = None
    with pytest.raises(PayloadError, match="top_logprobs=2"):
        oa.u_from_chat_completion(payload)
    del payload["choices"][0]["logprobs"]
    with pytest.raises(PayloadError, match="logprobs is missing"):
        oa.u_from_chat_completion(payload)


def test_chat_needs_two_candidates_per_token() -> None:
    payload = load("openai_chat_completion.json")
    payload["choices"][0]["logprobs"]["content"][2]["top_logprobs"] = payload["choices"][0][
        "logprobs"
    ]["content"][2]["top_logprobs"][:1]
    with pytest.raises(PayloadError, match=r"content\[2\]: 1 candidate.*at least 2"):
        oa.u_from_chat_completion(payload)
    payload["choices"][0]["logprobs"]["content"][2]["top_logprobs"] = []
    with pytest.raises(PayloadError, match="0 candidate"):
        oa.u_from_chat_completion(payload)


@pytest.mark.parametrize(
    ("value", "pattern"),
    [
        (0.5, "above 0"),
        (float("nan"), "log-probability"),
        (float("inf"), "log-probability"),
        ("-0.1", "must be a number"),
        (True, "must be a number"),
        (None, "logprob is null"),
    ],
)
def test_chat_rejects_invalid_logprob_values(value: Any, pattern: str) -> None:
    payload = load("openai_chat_completion.json")
    payload["choices"][0]["logprobs"]["content"][1]["top_logprobs"][0]["logprob"] = value
    with pytest.raises(PayloadError, match=pattern):
        oa.u_from_chat_completion(payload)


def test_chat_rejects_logits_disguised_as_logprobs() -> None:
    payload = load("openai_chat_completion.json")
    tops = payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    for t in tops:
        t["logprob"] = -0.01  # probabilities 0.99 each: not one distribution
    with pytest.raises(PayloadError, match="sum to"):
        oa.u_from_chat_completion(payload)


def test_tiny_positive_logprob_is_round_off() -> None:
    payload = load("openai_chat_completion.json")
    entry = payload["choices"][0]["logprobs"]["content"][0]
    entry["top_logprobs"][0]["logprob"] = 5e-8
    entry["top_logprobs"][1]["logprob"] = -9999.0  # OpenAI's floor for "very unlikely"
    entry["top_logprobs"] = entry["top_logprobs"][:2]
    sig = oa.signals_from_chat_completion(payload)
    assert sig.margins[0] == pytest.approx(1.0, abs=TOL)


def test_minus_infinity_candidate_is_probability_zero() -> None:
    payload = load("openai_chat_completion.json")
    payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"][1]["logprob"] = -math.inf
    payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"][2]["logprob"] = -math.inf
    sig = oa.signals_from_chat_completion(payload)
    assert sig.margins[0] == pytest.approx(0.98, abs=TOL)
    assert math.isfinite(sig.mean_entropy)


def test_non_greedy_tokens_are_counted_and_can_be_rejected() -> None:
    payload = load("openai_chat_completion.json")
    entry = payload["choices"][0]["logprobs"]["content"][3]
    entry["logprob"] = entry["top_logprobs"][1]["logprob"]  # sampled "Nikon", not top-1
    sig = oa.signals_from_chat_completion(payload)
    assert sig.n_non_greedy == 1
    assert sig.u == pytest.approx(CHAT_U, abs=TOL)  # u uses the distribution, not the sample
    with pytest.raises(PayloadError, match="greedy decoding"):
        oa.signals_from_chat_completion(payload, require_greedy=True)


def test_chat_unrecognised_payload() -> None:
    with pytest.raises(PayloadError, match="expected a ChatCompletion"):
        oa.u_from_chat_completion({"foo": 1})
    with pytest.raises(PayloadError, match="must be a list"):
        oa.u_from_chat_completion({"choices": "nope"})


# ---------------------------------------------------------------------------
# OpenAI Completions (legacy)
# ---------------------------------------------------------------------------


def test_completion_matches_hand_computed_u() -> None:
    payload = load("openai_completion.json")
    sig = oa.signals_from_completion(payload)
    assert sig.u == pytest.approx(COMPLETION_U, abs=TOL)
    assert sig.n_tokens == 3
    assert sig.n_non_greedy == 0
    # finish_reason is "length": no stop token to drop even when asked.
    assert oa.signals_from_completion(payload, drop_stop_token=True).n_tokens == 3
    choice = payload["choices"][0]
    sdk_like = ns(payload)  # the SDK keeps each top_logprobs entry a plain dict
    sdk_like.choices[0].logprobs.top_logprobs = choice["logprobs"]["top_logprobs"]
    for obj in (sdk_like, choice, choice["logprobs"]):
        assert oa.u_from_completion(obj) == pytest.approx(COMPLETION_U, abs=TOL)
    assert oa.u_from_openai(payload) == pytest.approx(COMPLETION_U, abs=TOL)


def test_completion_with_openai_sdk_model() -> None:
    types_mod = pytest.importorskip("openai.types")
    resp = types_mod.Completion.model_validate(load("openai_completion.json"))
    assert oa.u_from_completion(resp) == pytest.approx(COMPLETION_U, abs=TOL)


def test_completion_sampled_token_outside_top_k() -> None:
    payload = load("openai_completion.json")
    lp = payload["choices"][0]["logprobs"]
    lp["top_logprobs"][1][" Mark"] = math.log(0.05)  # logprobs+1 entries: sampled token added
    lp["tokens"][1], lp["token_logprobs"][1] = " Mark", math.log(0.05)
    sig = oa.signals_from_completion(payload)
    assert sig.n_non_greedy == 1
    assert sig.u == pytest.approx(COMPLETION_U, abs=TOL)


def test_completion_echo_positions_are_rejected() -> None:
    payload = load("openai_completion.json")
    payload["choices"][0]["logprobs"]["top_logprobs"][0] = None
    with pytest.raises(PayloadError, match="echo=True"):
        oa.u_from_completion(payload)


def test_completion_malformed() -> None:
    payload = load("openai_completion.json")
    payload["choices"][0]["logprobs"]["token_logprobs"].pop()
    with pytest.raises(PayloadError, match="token_logprobs has 2 entries"):
        oa.u_from_completion(payload)
    payload = load("openai_completion.json")
    payload["choices"][0]["logprobs"]["top_logprobs"][0] = [-0.1, -2.0]
    with pytest.raises(PayloadError, match="mapping"):
        oa.u_from_completion(payload)
    payload["choices"][0]["logprobs"] = None
    with pytest.raises(PayloadError, match="logprobs=2"):
        oa.u_from_completion(payload)
    with pytest.raises(PayloadError, match="expected a Completion"):
        oa.u_from_completion({"foo": 1})


# ---------------------------------------------------------------------------
# OpenAI Responses
# ---------------------------------------------------------------------------


def test_responses_matches_hand_computed_u() -> None:
    payload = load("openai_responses.json")
    sig = oa.signals_from_responses(payload)
    assert sig.u == pytest.approx(RESPONSES_U, abs=TOL)
    assert sig.n_tokens == 3
    assert sig.source == "openai.responses"
    assert oa.u_from_responses(ns(payload)) == pytest.approx(RESPONSES_U, abs=TOL)
    assert oa.u_from_openai(payload) == pytest.approx(RESPONSES_U, abs=TOL)


def test_responses_with_openai_sdk_model() -> None:
    types_mod = pytest.importorskip("openai.types.responses")
    resp = types_mod.Response.model_validate(load("openai_responses.json"))
    assert oa.u_from_responses(resp) == pytest.approx(RESPONSES_U, abs=TOL)


def test_responses_skips_non_text_items_and_concatenates_parts() -> None:
    payload = load("openai_responses.json")
    msg = payload["output"][0]
    payload["output"].insert(0, {"type": "reasoning", "id": "rs_1", "summary": []})
    extra = copy.deepcopy(msg["content"][0])
    msg["content"].append({"type": "refusal", "refusal": "no"})
    msg["content"].append(extra)
    sig = oa.signals_from_responses(payload)
    assert sig.n_tokens == 6
    assert sig.u == pytest.approx(RESPONSES_U, abs=TOL)
    assert oa.signals_from_responses(payload, drop_stop_token=True).n_tokens == 5
    payload["status"] = "incomplete"
    assert oa.signals_from_responses(payload, drop_stop_token=True).n_tokens == 6


def test_responses_missing_logprobs_explains_include() -> None:
    payload = load("openai_responses.json")
    del payload["output"][0]["content"][0]["logprobs"]
    with pytest.raises(PayloadError, match=r"message\.output_text\.logprobs"):
        oa.u_from_responses(payload)
    payload["output"] = [{"type": "function_call", "name": "f", "arguments": "{}"}]
    with pytest.raises(PayloadError, match="no message with an output_text part"):
        oa.u_from_responses(payload)
    with pytest.raises(PayloadError, match="output is missing"):
        oa.u_from_responses({"object": "response"})


def test_openai_auto_dispatch_errors() -> None:
    chunk = load("openai_chat_completion_chunks.json")[1]
    with pytest.raises(PayloadError, match="whole list of chunks"):
        oa.u_from_openai(chunk)
    chunk.pop("object")
    with pytest.raises(PayloadError, match="whole list of chunks"):
        oa.u_from_openai(chunk)
    with pytest.raises(PayloadError, match="cannot recognise"):
        oa.u_from_openai({"hello": "world"})
    content = load("openai_chat_completion.json")["choices"][0]["logprobs"]["content"]
    assert oa.u_from_openai(content) == pytest.approx(CHAT_U, abs=TOL)
    assert oa.u_from_openai(load("openai_chat_completion.json")["choices"][0]) == pytest.approx(
        CHAT_U, abs=TOL
    )


# ---------------------------------------------------------------------------
# make_chat_fn
# ---------------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self, payload: Any, is_async: bool = False) -> None:
        self.payload, self.is_async, self.calls = payload, is_async, []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.is_async:

            async def later() -> Any:
                return self.payload

            return later()
        return self.payload


def _fake_client(payload: Any, is_async: bool = False) -> Any:
    comp = _FakeCompletions(payload, is_async)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=comp)), comp


def test_make_chat_fn_sync_builds_greedy_logprob_request() -> None:
    payload = load("openai_chat_completion.json")
    client, comp = _fake_client(payload)
    fn = oa.make_chat_fn(client, "small", system_prompt="Extract entities.", max_tokens=256)
    text, resp = fn("canon 5d")
    assert text == payload["choices"][0]["message"]["content"]
    assert resp is payload
    call = comp.calls[0]
    assert call["model"] == "small"
    assert call["temperature"] == 0.0
    assert call["logprobs"] is True
    assert call["top_logprobs"] == 2
    assert call["max_tokens"] == 256
    assert call["messages"] == [
        {"role": "system", "content": "Extract entities."},
        {"role": "user", "content": "canon 5d"},
    ]
    large = oa.make_chat_fn(client, "large", top_logprobs=0, return_response=False)
    msgs = [{"role": "user", "content": "hi"}]
    assert large(msgs) == text
    assert "logprobs" not in comp.calls[1]
    assert comp.calls[1]["messages"] == msgs


def test_make_chat_fn_async_client() -> None:
    payload = load("openai_chat_completion.json")
    client, _ = _fake_client(payload, is_async=True)
    fn = oa.make_chat_fn(client, "small")
    text, resp = asyncio.run(fn("q"))
    assert resp is payload
    assert text == payload["choices"][0]["message"]["content"]


def test_make_chat_fn_validates_top_logprobs() -> None:
    client, _ = _fake_client({})
    with pytest.raises(ValueError, match=r"\[0, 20\]"):
        oa.make_chat_fn(client, "m", top_logprobs=21)
    with pytest.raises(TypeError, match="int"):
        oa.make_chat_fn(client, "m", top_logprobs=2.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# vLLM offline
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Logprob:
    """Mirror of vllm.logprobs.Logprob."""

    logprob: float
    rank: int | None = None
    decoded_token: str | None = None


@dataclasses.dataclass
class CompletionOutput:
    """Mirror of vllm.outputs.CompletionOutput (fields used by the adapter)."""

    index: int
    text: str
    token_ids: list[int]
    cumulative_logprob: float | None
    logprobs: Any
    finish_reason: str | None = None
    stop_reason: int | str | None = None


@dataclasses.dataclass
class RequestOutput:
    """Mirror of vllm.outputs.RequestOutput (fields used by the adapter)."""

    request_id: str
    prompt: str
    outputs: list[CompletionOutput]
    finished: bool = True


class FlatLike(Sequence):  # type: ignore[type-arg]
    """Minimal stand-in for vllm.logprobs.FlatLogprobs: a sequence of dicts."""

    def __init__(self, positions: list[dict[int, Logprob]]) -> None:
        self._p = positions

    def __len__(self) -> int:
        return len(self._p)

    def __getitem__(self, i: Any) -> Any:
        return self._p[i]


def vllm_objects(payload: dict[str, Any]) -> RequestOutput:
    outs = []
    for o in payload["outputs"]:
        lps = [{int(k): Logprob(**v) for k, v in pos.items()} for pos in o["logprobs"]]
        outs.append(
            CompletionOutput(
                o["index"],
                o["text"],
                list(o["token_ids"]),
                o["cumulative_logprob"],
                lps,
                o["finish_reason"],
                o["stop_reason"],
            )
        )
    return RequestOutput(payload["request_id"], payload["prompt"], outs)


def test_vllm_request_output_drops_eos_automatically() -> None:
    payload = load("vllm_request_output.json")
    for obj in (payload, vllm_objects(payload), vllm_objects(payload).outputs[0]):
        sig = vl.signals_from_vllm(obj)
        assert sig.u == pytest.approx(VLLM_U, abs=TOL)
        assert sig.u == pytest.approx(0.4375, abs=TOL)
        assert sig.n_tokens == 4
        assert sig.n_non_greedy == 0
        assert sig.top_k == 2
        assert sig.mean_max_prob == pytest.approx((0.95 + 0.7 + 0.5 + 0.88) / 4, abs=TOL)
    kept = vl.signals_from_vllm(payload, drop_stop_token=False)
    assert kept.n_tokens == 5
    assert kept.u == pytest.approx(u_of([0.92, 0.5, 0.05, 0.78, 0.999 - 0.0005]), abs=TOL)


def test_vllm_stop_reason_rules() -> None:
    obj = vllm_objects(load("vllm_request_output.json"))
    comp = obj.outputs[0]
    comp.stop_reason = 151645  # a stop_token_ids entry: generation ended on a token
    assert vl.signals_from_vllm(obj).n_tokens == 4
    comp.stop_reason = "</json>"  # a stop string: the last token is content
    assert vl.signals_from_vllm(obj).n_tokens == 5
    assert vl.signals_from_vllm(obj, drop_stop_token=True).n_tokens == 4
    comp.stop_reason, comp.finish_reason = None, "length"
    assert vl.signals_from_vllm(obj).n_tokens == 5
    assert vl.signals_from_vllm(obj, drop_stop_token=True).n_tokens == 5


def test_vllm_bare_logprobs_and_flat_container() -> None:
    comp = vllm_objects(load("vllm_request_output.json")).outputs[0]
    assert vl.signals_from_vllm(comp.logprobs).n_tokens == 5
    assert vl.signals_from_vllm(comp.logprobs, drop_stop_token=True).u == pytest.approx(
        VLLM_U, abs=TOL
    )
    assert vl.signals_from_vllm(comp.logprobs).n_non_greedy is None
    comp.logprobs = FlatLike(comp.logprobs)
    assert vl.u_from_vllm(comp) == pytest.approx(VLLM_U, abs=TOL)


def test_vllm_plain_float_values_and_batch() -> None:
    comp = vllm_objects(load("vllm_request_output.json")).outputs[0]
    comp.logprobs = [{k: v.logprob for k, v in pos.items()} for pos in comp.logprobs]
    assert vl.u_from_vllm(comp) == pytest.approx(VLLM_U, abs=TOL)
    payload = load("vllm_request_output.json")
    assert vl.u_from_vllm_batch([payload, payload]) == pytest.approx([VLLM_U, VLLM_U], abs=TOL)
    assert [s.n_tokens for s in vl.signals_from_vllm_batch([payload])] == [4]


def test_vllm_sampled_token_outside_top_k() -> None:
    obj = vllm_objects(load("vllm_request_output.json"))
    comp = obj.outputs[0]
    comp.token_ids[2] = 999
    comp.logprobs[2][999] = Logprob(math.log(0.01), rank=7, decoded_token="x")
    sig = vl.signals_from_vllm(obj)
    assert sig.n_non_greedy == 1
    assert sig.u == pytest.approx(VLLM_U, abs=TOL)
    with pytest.raises(PayloadError, match="greedy"):
        vl.signals_from_vllm(obj, require_greedy=True)


def test_vllm_errors() -> None:
    obj = vllm_objects(load("vllm_request_output.json"))
    obj.outputs[0].logprobs = None
    with pytest.raises(PayloadError, match=r"logprobs=2"):
        vl.u_from_vllm(obj)
    obj = vllm_objects(load("vllm_request_output.json"))
    obj.outputs[0].logprobs[1] = {43023: Logprob(math.log(0.7))}  # logprobs=1
    with pytest.raises(PayloadError, match="at least 2"):
        vl.u_from_vllm(obj)
    obj = vllm_objects(load("vllm_request_output.json"))
    obj.outputs[0].logprobs[0] = {515: Logprob(12.5), 90: Logprob(9.1)}  # raw_logits mode
    with pytest.raises(PayloadError, match="above 0"):
        vl.u_from_vllm(obj)
    obj = vllm_objects(load("vllm_request_output.json"))
    obj.outputs[0].token_ids.pop()
    with pytest.raises(PayloadError, match="token_ids has 4"):
        vl.u_from_vllm(obj)
    obj = vllm_objects(load("vllm_request_output.json"))
    obj.outputs[0].logprobs[3] = None
    with pytest.raises(PayloadError, match=r"logprobs\[3\] is None"):
        vl.u_from_vllm(obj)
    with pytest.raises(PayloadError, match="out of range"):
        vl.u_from_vllm(load("vllm_request_output.json"), completion_index=1)
    with pytest.raises(PayloadError, match="expected a vLLM"):
        vl.u_from_vllm(object())
    with pytest.raises(PayloadError, match="Logprob or a number"):
        vl.u_from_vllm([{1: "x", 2: "y"}])


def test_greedy_sampling_params_validation() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        vl.greedy_sampling_params(logprobs=1)
    with pytest.raises(ValueError, match="temperature"):
        vl.greedy_sampling_params(temperature=0.7)
    try:
        import vllm  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pip install vllm"):
            vl.greedy_sampling_params()
    else:  # pragma: no cover - only where vLLM is installed
        sp = vl.greedy_sampling_params(max_tokens=64)
        assert sp.temperature == 0.0
        assert sp.logprobs == 2
        assert sp.max_tokens == 64


# ---------------------------------------------------------------------------
# llama.cpp /completion
# ---------------------------------------------------------------------------


def test_llamacpp_current_format() -> None:
    payload = load("llamacpp_completion.json")
    sig = lc.signals_from_llamacpp(payload)
    assert sig.u == pytest.approx(LLAMACPP_U, abs=TOL)
    assert sig.u == pytest.approx(0.4, abs=TOL)
    assert sig.n_tokens == 3
    assert sig.n_non_greedy == 0
    assert lc.signals_from_llamacpp(payload, drop_stop_token=False).n_tokens == 4
    bare = payload["completion_probabilities"]
    assert lc.signals_from_llamacpp(bare).n_tokens == 4
    assert lc.u_from_llamacpp(bare, drop_stop_token=True) == pytest.approx(LLAMACPP_U, abs=TOL)
    payload["stop_type"] = "limit"
    assert lc.signals_from_llamacpp(payload).n_tokens == 4


def test_llamacpp_zero_probability_sentinel() -> None:
    payload = load("llamacpp_completion.json")
    payload["completion_probabilities"][0]["top_logprobs"][1]["logprob"] = -3.4028234663852886e38
    sig = lc.signals_from_llamacpp(payload)
    assert sig.margins[0] == pytest.approx(0.8, abs=TOL)


def test_llamacpp_stream_matches_non_streamed() -> None:
    events = load("llamacpp_stream.json")
    sig = lc.signals_from_llamacpp_stream(events)
    assert sig.u == pytest.approx(LLAMACPP_U, abs=TOL)
    assert sig.n_tokens == 3
    assert lc.u_from_llamacpp_stream(events, drop_stop_token=False) == pytest.approx(
        u_of([0.65, 0.3, 0.85, 0.96]), abs=TOL
    )
    with pytest.raises(PayloadError, match="must be an object"):
        lc.signals_from_llamacpp_stream([1])


def test_llamacpp_post_sampling_is_rejected_by_default() -> None:
    payload = load("llamacpp_completion_post_sampling.json")
    with pytest.raises(PayloadError, match="post_sampling_probs"):
        lc.u_from_llamacpp(payload)
    # Under greedy decoding post-sampling output collapses to one candidate.
    with pytest.raises(PayloadError, match="at least 2"):
        lc.u_from_llamacpp(payload, allow_post_sampling=True)
    for entry, (p1, p2) in zip(
        payload["completion_probabilities"], [(0.8, 0.2), (0.6, 0.4), (0.9, 0.1), (1.0, 0.0)]
    ):
        entry["top_probs"] = [
            {"id": 1, "token": "a", "bytes": [97], "prob": p1},
            {"id": 2, "token": "b", "bytes": [98], "prob": p2},
        ]
    sig = lc.signals_from_llamacpp(payload, allow_post_sampling=True)
    assert sig.u == pytest.approx(u_of([0.6, 0.2, 0.8]), abs=TOL)


def test_llamacpp_legacy_format() -> None:
    payload = load("llamacpp_completion_legacy.json")
    sig = lc.signals_from_llamacpp(payload)
    assert sig.u == pytest.approx(LLAMACPP_LEGACY_U, abs=TOL)
    assert sig.n_tokens == 3
    assert sig.n_non_greedy == 0
    payload["stopped_eos"] = False
    assert lc.signals_from_llamacpp(payload).n_tokens == 4


def test_llamacpp_legacy_degenerate_probabilities() -> None:
    payload = load("llamacpp_completion_legacy.json")
    for entry in payload["completion_probabilities"]:
        entry["probs"][0]["prob"], entry["probs"][1]["prob"] = 1.0, 0.0
    with pytest.raises(PayloadError, match="temperature < 0"):
        lc.u_from_llamacpp(payload)


def test_llamacpp_errors() -> None:
    with pytest.raises(PayloadError, match="n_probs"):
        lc.u_from_llamacpp({"content": "x", "stop": True})
    payload = load("llamacpp_completion.json")
    payload["completion_probabilities"][1] = {"id": 3, "token": "x"}
    with pytest.raises(PayloadError, match=r"completion_probabilities\[1\] has none of"):
        lc.u_from_llamacpp(payload)
    payload = load("llamacpp_completion_legacy.json")
    payload["completion_probabilities"][0]["probs"][0]["prob"] = 1.5
    with pytest.raises(PayloadError, match="not a probability"):
        lc.u_from_llamacpp(payload)
    with pytest.raises(PayloadError, match="must be an object"):
        lc.u_from_llamacpp([3])


# ---------------------------------------------------------------------------
# transformers adapter on NumPy arrays (no torch needed)
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max())
    return z / z.sum()


def _manual_signals(logits: list[np.ndarray], row: int, n: int) -> tuple[float, float, float]:
    margins, maxp, ents = [], [], []
    for t in range(n):
        p = _softmax(logits[t][row].astype(np.float64))
        top = np.sort(p)[::-1]
        margins.append(top[0] - top[1])
        maxp.append(top[0])
        ents.append(-np.sum(p[p > 0] * np.log(p[p > 0])))
    return 1.0 - float(np.mean(margins)), float(np.mean(ents)), float(np.mean(maxp))


def _synthetic_generate(seed: int = 0) -> tuple[types.SimpleNamespace, list[np.ndarray]]:
    """A left-padded batch of 3 rows, 5 steps, vocab 7, EOS 6, pad 5 (not EOS).

    Row 0 stops on EOS at step 2, row 1 never stops, row 2 is stopped by a
    stopping criterion after 3 tokens and padded.
    """
    rng = np.random.default_rng(seed)
    n_rows, n_steps, vocab, eos, pad = 3, 5, 7, 6, 5
    logits = [rng.normal(size=(n_rows, vocab)) for _ in range(n_steps)]
    for x in logits:
        x[:, pad] = -np.inf  # the pad token is never predicted
        x[:, eos] -= 5.0
    gen = np.array([[int(np.argmax(logits[t][r])) for t in range(n_steps)] for r in range(n_rows)])
    logits[2][0, eos] = 50.0
    gen[0, 2] = eos
    gen[0, 3:] = pad
    gen[2, 3:] = pad
    for t in (3, 4):  # after a row finished its logits are irrelevant
        logits[t][0, :] = np.nan
        logits[t][2, :] = np.nan
    prompt = np.array([[5, 5, 1], [2, 3, 1], [5, 4, 1]])
    seqs = np.concatenate([prompt, gen], axis=1)
    return types.SimpleNamespace(sequences=seqs, logits=tuple(logits), scores=None), logits


def test_transformers_numpy_batch_with_different_stop_steps() -> None:
    out, logits = _synthetic_generate()
    sigs = hf.signals_from_generate(out, eos_token_id=6, pad_token_id=5)
    assert [s.n_tokens for s in sigs] == [2, 5, 3]
    for row, sig in enumerate(sigs):
        u, ent, maxp = _manual_signals(logits, row, sig.n_tokens)
        assert sig.u == pytest.approx(u, abs=TOL)
        assert sig.mean_entropy == pytest.approx(ent, abs=TOL)
        assert sig.mean_max_prob == pytest.approx(maxp, abs=TOL)
        assert sig.entropy_support == "full_vocabulary"
        assert sig.top_k is None
        assert sig.n_non_greedy == 0
        assert sig.source == "transformers.logits"
    assert hf.u_from_generate(out, eos_token_id=[6], pad_token_id=5) == pytest.approx(
        [s.u for s in sigs], abs=TOL
    )
    as_dict = {"sequences": out.sequences, "logits": out.logits}
    assert hf.u_from_generate(as_dict, eos_token_id=6, pad_token_id=5) == pytest.approx(
        [s.u for s in sigs], abs=TOL
    )


def test_transformers_padding_without_pad_id_is_counted() -> None:
    out, _ = _synthetic_generate()
    out.logits[3][2, :] = 0.0
    out.logits[4][2, :] = 0.0
    sigs = hf.signals_from_generate(out, eos_token_id=6)  # pad id unknown
    assert [s.n_tokens for s in sigs] == [2, 5, 5]


def test_content_lengths_rules() -> None:
    gen = np.array([[3, 1, 9, 9], [9, 0, 0, 0], [3, 4, 0, 0], [0, 0, 0, 0], [3, 2, 1, 4]])
    assert hf.content_lengths(gen, 9).tolist() == [2, 0, 4, 4, 4]
    assert hf.content_lengths(gen, [9, 4]).tolist() == [2, 0, 1, 4, 3]
    assert hf.content_lengths(gen, 9, pad_token_id=0).tolist() == [2, 0, 2, 0, 4]
    assert hf.content_lengths(gen, None).tolist() == [4, 4, 4, 4, 4]
    assert hf.content_lengths(gen, 9, pad_token_id=9).tolist() == [2, 0, 4, 4, 4]
    assert hf.content_lengths(gen, np.array([9])).tolist() == [2, 0, 4, 4, 4]
    with pytest.raises(TypeError):
        hf.content_lengths(gen, "9")
    with pytest.raises(TypeError):
        hf.content_lengths(gen, True)
    with pytest.raises(PayloadError, match="integer token ids"):
        hf.content_lengths(gen.astype(float), 9)


def test_transformers_prefers_raw_logits_over_processed_scores() -> None:
    out, logits = _synthetic_generate()
    scores = tuple(np.where(np.isfinite(x), x * 3.0, x) for x in logits)
    out.scores = scores
    auto = hf.signals_from_generate(out, eos_token_id=6, pad_token_id=5)
    from_scores = hf.signals_from_generate(out, eos_token_id=6, pad_token_id=5, source="scores")
    assert auto[1].u == pytest.approx(_manual_signals(logits, 1, 5)[0], abs=TOL)
    assert from_scores[1].u == pytest.approx(_manual_signals(list(scores), 1, 5)[0], abs=TOL)
    assert from_scores[1].source == "transformers.scores"
    assert auto[1].u != pytest.approx(from_scores[1].u)
    out.logits = None
    assert hf.signals_from_generate(out, eos_token_id=6, pad_token_id=5)[1].source == (
        "transformers.scores"
    )
    with pytest.raises(PayloadError, match="output_logits=True"):
        hf.signals_from_generate(out, eos_token_id=6, source="logits")
    out.scores = ()
    with pytest.raises(PayloadError, match="neither logits nor scores"):
        hf.signals_from_generate(out, eos_token_id=6)
    with pytest.raises(ValueError, match="source must be"):
        hf.signals_from_generate(out, eos_token_id=6, source="probs")


def test_transformers_non_greedy_detection() -> None:
    out, _ = _synthetic_generate()
    seqs = out.sequences.copy()
    second = int(np.argsort(out.logits[1][1])[-2])
    seqs[1, 3 + 1] = second
    out.sequences = seqs
    sigs = hf.signals_from_generate(out, eos_token_id=6, pad_token_id=5)
    assert [s.n_non_greedy for s in sigs] == [0, 1, 0]
    with pytest.raises(PayloadError, match="row 1: 1 generated tokens are not the argmax"):
        hf.signals_from_generate(out, eos_token_id=6, pad_token_id=5, require_greedy=True)


def test_transformers_empty_rows() -> None:
    out, _ = _synthetic_generate()
    seqs = out.sequences.copy()
    seqs[1, 3:] = [6, 5, 5, 5, 5]
    out.sequences = seqs
    with pytest.raises(PayloadError, match="row 1: the first generated token is EOS"):
        hf.signals_from_generate(out, eos_token_id=6, pad_token_id=5)
    sigs = hf.signals_from_generate(out, eos_token_id=6, pad_token_id=5, on_empty="nan")
    assert math.isnan(sigs[1].u)
    assert sigs[1].n_tokens == 0
    assert sigs[1].margins == ()
    assert not math.isnan(sigs[0].u)
    with pytest.raises(ValueError, match="on_empty"):
        hf.signals_from_generate(out, eos_token_id=6, on_empty="skip")


def test_transformers_malformed_outputs() -> None:
    out, _ = _synthetic_generate()
    with pytest.raises(PayloadError, match="beam-search"):
        hf.signals_from_generate(
            types.SimpleNamespace(**vars(out), beam_indices=np.zeros(1)), eos_token_id=6
        )
    bad = types.SimpleNamespace(**vars(out))
    bad.logits = tuple(x.copy() for x in out.logits)
    bad.logits[0][1, 2] = np.nan
    with pytest.raises(PayloadError, match=r"logits\[0\] row 1 contains NaN"):
        hf.signals_from_generate(bad, eos_token_id=6, pad_token_id=5)
    bad.logits = tuple(x.copy() for x in out.logits)
    bad.logits[0][1, :] = -np.inf
    with pytest.raises(PayloadError, match="-inf everywhere"):
        hf.signals_from_generate(bad, eos_token_id=6, pad_token_id=5)
    bad.logits = tuple(x[:2] for x in out.logits)
    with pytest.raises(PayloadError, match="num_return_sequences=1"):
        hf.signals_from_generate(bad, eos_token_id=6, pad_token_id=5)
    bad.logits = out.logits[:1] + tuple(x[:, :4] for x in out.logits[1:])
    with pytest.raises(PayloadError, match="vocabulary 4, expected 7"):
        hf.signals_from_generate(bad, eos_token_id=6, pad_token_id=5)
    bad.logits = out.logits + out.logits + out.logits
    with pytest.raises(PayloadError, match="only 8 columns"):
        hf.signals_from_generate(bad, eos_token_id=6)
    bad = types.SimpleNamespace(**vars(out))
    bad.sequences = out.sequences.copy()
    bad.sequences[1, 3] = 40
    with pytest.raises(PayloadError, match="outside the logits vocabulary"):
        hf.signals_from_generate(bad, eos_token_id=6, pad_token_id=5)
    with pytest.raises(PayloadError, match="sequences is missing"):
        hf.signals_from_generate({"logits": out.logits}, eos_token_id=6)
    bad = types.SimpleNamespace(sequences=out.sequences, logits=(np.zeros(7),))
    with pytest.raises(PayloadError, match=r"shape \(batch, vocab\)"):
        hf.signals_from_generate(bad, eos_token_id=6)


# ---------------------------------------------------------------------------
# transformers adapter with a real tiny model (needs torch and transformers)
# ---------------------------------------------------------------------------

TINY_GPT2 = "hf-internal-testing/tiny-random-gpt2"
TINY_T5 = "hf-internal-testing/tiny-random-t5"


def _hf() -> tuple[Any, Any]:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    try:
        # transformers imports its submodules lazily, so a broken dependency of
        # the installed release (for example an incompatible pyOpenSSL) only
        # surfaces on first attribute access, as a RuntimeError.
        transformers.AutoTokenizer  # noqa: B018
        transformers.AutoModelForCausalLM  # noqa: B018
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"transformers is installed but cannot be imported here: {exc}")
    return torch, transformers


def _load(transformers: Any, name: str, seq2seq: bool = False) -> tuple[Any, Any]:
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name)
        cls = transformers.AutoModelForSeq2SeqLM if seq2seq else transformers.AutoModelForCausalLM
        model = cls.from_pretrained(name).eval()
    except OSError as exc:  # pragma: no cover - offline without a cached model
        pytest.skip(f"cannot load {name}: {exc}")
    return tok, model


def _manual_torch(torch: Any, steps: Any, row: int, n: int) -> tuple[float, int]:
    """u(x) from an independent per-step torch softmax, and the non-greedy count."""
    margins = []
    for t in range(n):
        p = torch.softmax(steps[t][row].to(torch.float64), dim=-1)
        top = torch.topk(p, 2).values
        margins.append(float(top[0] - top[1]))
    return 1.0 - sum(margins) / n, 0


def test_transformers_real_model_batch_with_different_stop_steps() -> None:
    torch, transformers = _hf()
    tok, model = _load(transformers, TINY_GPT2)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    prompts = ["hello world", "the quick brown fox jumps over", "a"]
    enc = tok(prompts, return_tensors="pt", padding=True)
    prompt_len = enc["input_ids"].shape[1]
    greedy = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": 8,
        "return_dict_in_generate": True,
        "output_logits": True,
        "output_scores": True,
    }
    with torch.no_grad():
        first = model.generate(**enc, eos_token_id=None, pad_token_id=tok.pad_token_id, **greedy)
    gen0 = first.sequences[:, prompt_len:].tolist()
    # EOS ids chosen from the no-EOS run so rows stop at different steps.
    eos_ids = sorted({gen0[0][1], gen0[2][-2]})
    stop_after = 3  # a stopping criterion ends row 1 after 3 tokens
    used = {t for row in gen0 for t in row}
    pad = next(i for i in range(5, 1000) if i not in used and i not in eos_ids)

    def first_eos(row: list[int]) -> int:
        return next((i for i, t in enumerate(row) if t in eos_ids), len(row))

    expected = [first_eos(gen0[0]), min(first_eos(gen0[1]), stop_after), first_eos(gen0[2])]
    assert len(set(expected)) == 3, f"test precondition: distinct stop steps, got {expected}"

    class StopRow(transformers.StoppingCriteria):  # type: ignore[misc]
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            if input_ids.shape[1] - prompt_len >= stop_after:
                done[1] = True
            return done

    with torch.no_grad():
        out = model.generate(
            **enc,
            eos_token_id=eos_ids,
            pad_token_id=pad,
            stopping_criteria=transformers.StoppingCriteriaList([StopRow()]),
            **greedy,
        )
    sigs = hf.signals_from_generate(out, eos_token_id=eos_ids, pad_token_id=pad)
    assert [s.n_tokens for s in sigs] == expected
    for row, sig in enumerate(sigs):
        u, _ = _manual_torch(torch, out.logits, row, sig.n_tokens)
        assert sig.u == pytest.approx(u, abs=1e-12)
        assert sig.n_non_greedy == 0
        p = torch.softmax(out.logits[0][row].to(torch.float64), dim=-1)
        assert sig.margins[0] == pytest.approx(
            float(torch.topk(p, 2).values.diff().abs()), abs=1e-12
        )


def test_transformers_real_model_logits_versus_processed_scores() -> None:
    torch, transformers = _hf()
    tok, model = _load(transformers, TINY_GPT2)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    enc = tok(["hello world", "a b c"], return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=False,
            num_beams=1,
            max_new_tokens=6,
            repetition_penalty=1.8,
            return_dict_in_generate=True,
            output_logits=True,
            output_scores=True,
            pad_token_id=tok.pad_token_id,
            eos_token_id=None,
        )
    assert any(not torch.equal(a, b) for a, b in zip(out.logits, out.scores))
    auto = hf.signals_from_generate(out, eos_token_id=None)
    scored = hf.signals_from_generate(out, eos_token_id=None, source="scores")
    gen = out.sequences[:, -len(out.logits) :]
    for row in range(2):
        assert auto[row].u == pytest.approx(_manual_torch(torch, out.logits, row, 6)[0], abs=1e-12)
        assert scored[row].u == pytest.approx(
            _manual_torch(torch, out.scores, row, 6)[0], abs=1e-12
        )
        n_ng = sum(int(gen[row, t]) != int(torch.argmax(out.logits[t][row])) for t in range(6))
        assert auto[row].n_non_greedy == n_ng
        assert scored[row].n_non_greedy == 0
    bf16 = types.SimpleNamespace(
        sequences=out.sequences, logits=tuple(x.to(torch.bfloat16) for x in out.logits)
    )
    low = hf.signals_from_generate(bf16, eos_token_id=None)
    ref = tuple(x.to(torch.bfloat16).to(torch.float64) for x in out.logits)
    assert low[0].u == pytest.approx(_manual_torch(torch, ref, 0, 6)[0], abs=1e-12)


def test_generate_with_signals_matches_manual_generate() -> None:
    _, transformers = _hf()
    tok, model = _load(transformers, TINY_GPT2)
    side, pad_token = tok.padding_side, tok.pad_token
    eos = int(model.generation_config.eos_token_id)
    res = hf.generate_with_signals(
        model, tok, ["hello world", "the quick brown fox"], max_new_tokens=5, eos_token_id=eos
    )
    assert tok.padding_side == side
    assert tok.pad_token == pad_token
    assert len(res.texts) == 2
    assert len(res.signals) == 2
    assert len(res.u) == 2
    manual = hf.signals_from_generate(res.output, eos_token_id=eos)
    assert res.u == pytest.approx([s.u for s in manual], abs=1e-12)
    gen = res.output.sequences[:, -len(res.output.logits) :]
    for i, sig in enumerate(res.signals):
        assert res.texts[i] == tok.decode(gen[i, : sig.n_tokens].tolist(), skip_special_tokens=True)
    single = hf.generate_with_signals(model, tok, "hello world", max_new_tokens=5, eos_token_id=eos)
    assert single.u[0] == pytest.approx(res.u[0], abs=1e-9)
    with pytest.raises(ValueError, match="do_sample"):
        hf.generate_with_signals(model, tok, "x", do_sample=True)
    with pytest.raises(ValueError, match="num_beams"):
        hf.generate_with_signals(model, tok, "x", num_beams=4)
    with pytest.raises(ValueError, match="prompts is empty"):
        hf.generate_with_signals(model, tok, [])


def test_transformers_encoder_decoder_model() -> None:
    torch, transformers = _hf()
    tok, model = _load(transformers, TINY_T5, seq2seq=True)
    enc = tok(
        ["translate: hello", "a longer input sentence here"], return_tensors="pt", padding=True
    )
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=False,
            num_beams=1,
            max_new_tokens=6,
            return_dict_in_generate=True,
            output_logits=True,
        )
    eos, pad = model.generation_config.eos_token_id, model.generation_config.pad_token_id
    assert out.sequences.shape[1] == len(out.logits) + 1  # decoder start token first
    # This random model generates the pad id (0) as content, which is why pad_token_id
    # is only for rows ended by a stopping criterion; with it, those tokens vanish.
    sigs = hf.signals_from_generate(out, eos_token_id=eos, on_empty="nan")
    with_pad = hf.signals_from_generate(out, eos_token_id=eos, pad_token_id=pad, on_empty="nan")
    gen = out.sequences[:, 1:].tolist()
    for row, (plain, trimmed) in enumerate(zip(sigs, with_pad)):
        if gen[row] and gen[row][-1] == pad:
            assert trimmed.n_tokens < plain.n_tokens
        else:
            assert trimmed.n_tokens == plain.n_tokens
    for row, sig in enumerate(sigs):
        eos_ids = [eos] if isinstance(eos, int) else list(eos)
        n = next((i for i, t in enumerate(gen[row]) if t in eos_ids), len(gen[row]))
        assert sig.n_tokens == n
        if n:
            assert sig.u == pytest.approx(_manual_torch(torch, out.logits, row, n)[0], abs=1e-12)
    res = hf.generate_with_signals(
        model, tok, ["translate: hello"], max_new_tokens=6, on_empty="nan"
    )
    assert len(res.signals) == 1


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------


class ThresholdRouter:
    """Router stub: p_hat = u (already calibrated), escalate when p_hat > theta (Eq. 6)."""

    def __init__(self, theta: float) -> None:
        self.theta = theta
        self.seen: list[float] = []

    def error_probability(self, u: Any) -> np.ndarray:
        arr = np.asarray(u, dtype=float)
        self.seen.extend(arr.tolist())
        return np.clip(arr, 0.0, 1.0)

    def escalate(self, u: Any) -> np.ndarray:
        return np.asarray(self.error_probability(u)) > self.theta


def _models(u_by_query: dict[str, float]) -> tuple[Any, Any, list[str]]:
    large_calls: list[str] = []

    def small(q: str) -> tuple[str, dict[str, float]]:
        return f"small:{q}", {"u": u_by_query[q]}

    def large(q: str) -> str:
        large_calls.append(q)
        return f"large:{q}"

    return small, large, large_calls


def test_cascade_routes_by_threshold() -> None:
    small, large, calls = _models({"easy": 0.1, "hard": 0.9, "edge": 0.5})
    router = ThresholdRouter(theta=0.5)
    cascade: cc.Cascade[str, str] = cc.Cascade(router, small, large, lambda raw: raw["u"])
    easy, hard, edge = cascade("easy"), cascade("hard"), cascade("edge")
    assert (easy.answer, easy.escalated, easy.u, easy.p_hat) == ("small:easy", False, 0.1, 0.1)
    assert easy.large_answer is None
    assert easy.latency_large_ms is None
    assert easy.latency_small_ms is not None
    assert easy.latency_small_ms >= 0.0
    assert (hard.answer, hard.escalated, hard.small_answer) == ("large:hard", True, "small:hard")
    assert hard.large_answer == "large:hard"
    assert hard.latency_large_ms is not None
    assert edge.escalated is False  # p_hat == theta keeps the small model (Eq. 6)
    assert calls == ["hard"]
    assert cascade.decide(0.7) == (0.7, True)
    assert [r.answer for r in cascade.map(["easy", "hard"])] == ["small:easy", "large:hard"]


def test_cascade_shadow_large_runs_both_but_serves_the_routed_answer() -> None:
    small, large, calls = _models({"easy": 0.1, "hard": 0.9})
    cascade = cc.Cascade(
        ThresholdRouter(0.5), small, large, lambda raw: raw["u"], shadow_large=True
    )
    easy = cascade("easy")
    assert easy.answer == "small:easy"
    assert easy.large_answer == "large:easy"
    assert not easy.escalated
    assert calls == ["easy"]


def test_cascade_with_openai_adapter_and_real_router() -> None:
    from ucci import UCCIRouter

    rng = np.random.default_rng(0)
    u_cal = rng.random(400)
    e_cal = (rng.random(400) < u_cal).astype(float)
    u_val = rng.random(300)
    small_ok = (rng.random(300) >= u_val).astype(float)
    router = UCCIRouter(c_small=1.0, c_large=3.02)
    router.calibrate(u_cal, e_cal)
    router.choose_threshold(u_val, small_ok, np.ones(300), tau=0.8)
    payload = load("openai_chat_completion.json")
    cascade = cc.Cascade(
        router, lambda q: ("small", payload), lambda q: "large", oa.signals_from_chat_completion
    )
    res = cascade("q")
    assert res.u == pytest.approx(CHAT_U, abs=TOL)
    assert isinstance(res.signals, TokenSignals)
    assert res.signals.n_tokens == 5
    p_hat = float(router.error_probability([res.u])[0])
    assert res.p_hat == pytest.approx(p_hat)
    assert res.escalated == bool(router.escalate([res.u])[0])
    assert res.answer == ("large" if res.escalated else "small")


def test_cascade_async_and_amap_concurrency() -> None:
    in_flight, peak = [0], [0]

    async def small(q: str) -> tuple[str, float]:
        in_flight[0] += 1
        peak[0] = max(peak[0], in_flight[0])
        await asyncio.sleep(0.01)
        in_flight[0] -= 1
        return f"s{q}", float(q) / 10

    async def large(q: str) -> str:
        await asyncio.sleep(0)
        return f"l{q}"

    async def signal(raw: float) -> float:
        return raw

    cascade = cc.Cascade(ThresholdRouter(0.45), small, large, signal)
    one = asyncio.run(cascade.acall("9"))
    assert one.answer == "l9"
    assert one.escalated
    queries = [str(i) for i in range(10)]
    results = asyncio.run(cascade.amap(queries, max_concurrency=3))
    assert [r.answer for r in results] == [f"s{i}" if i <= 4 else f"l{i}" for i in range(10)]
    assert peak[0] <= 3
    unlimited = asyncio.run(cascade.amap(queries))
    assert [r.escalated for r in unlimited] == [i > 4 for i in range(10)]
    with pytest.raises(ValueError, match="max_concurrency"):
        asyncio.run(cascade.amap(queries, max_concurrency=0))


def test_cascade_acall_accepts_sync_functions() -> None:
    small, large, _ = _models({"hard": 0.9})
    cascade = cc.Cascade(ThresholdRouter(0.5), small, large, lambda raw: raw["u"])
    assert asyncio.run(cascade.acall("hard")).answer == "large:hard"


@pytest.mark.filterwarnings("error")
def test_sync_call_rejects_async_functions_without_leaking_coroutines() -> None:
    async def small(q: str) -> tuple[str, float]:
        return "s", 0.1

    cascade = cc.Cascade(ThresholdRouter(0.5), small, lambda q: "l", lambda raw: raw)
    with pytest.raises(TypeError, match="acall"):
        cascade("q")

    async def signal(raw: float) -> float:
        return raw

    cascade = cc.Cascade(ThresholdRouter(0.5), lambda q: ("s", 0.9), lambda q: "l", signal)
    with pytest.raises(TypeError, match="signal_fn returned an awaitable"):
        cascade("q")

    async def large(q: str) -> str:
        return "l"

    cascade = cc.Cascade(ThresholdRouter(0.5), lambda q: ("s", 0.9), large, lambda raw: raw)
    with pytest.raises(TypeError, match="large_fn returned an awaitable"):
        cascade("q")


def test_cascade_validates_functions_and_signals() -> None:
    router = ThresholdRouter(0.5)
    with pytest.raises(TypeError, match="small_fn must be callable"):
        cc.Cascade(router, None, lambda q: q, lambda r: 0.1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="error_probability"):
        cc.Cascade(object(), lambda q: ("a", 0.1), lambda q: q, lambda r: r)  # type: ignore[arg-type]
    half = types.SimpleNamespace(error_probability=lambda u: u)
    with pytest.raises(TypeError, match="escalate"):
        cc.Cascade(half, lambda q: ("a", 0.1), lambda q: q, lambda r: r)  # type: ignore[arg-type]
    cascade = cc.Cascade(router, lambda q: "not a pair", lambda q: q, lambda r: 0.1)
    with pytest.raises(TypeError, match="answer, raw_response"):
        cascade("q")
    for bad, err in ((float("nan"), ValueError), ("0.3", TypeError), (True, TypeError)):
        cascade = cc.Cascade(router, lambda q: ("a", None), lambda q: q, lambda r, bad=bad: bad)
        with pytest.raises(err):
            cascade("q")
    cascade = cc.Cascade(
        router, lambda q: ("a", None), lambda q: q, lambda r: types.SimpleNamespace(u=0.2)
    )
    res = cascade("q")
    assert res.u == 0.2
    assert res.signals is None


# ---------------------------------------------------------------------------
# JSONL records and logger
# ---------------------------------------------------------------------------


def test_make_record_format_and_order() -> None:
    rec = cc.make_record(
        "q1",
        0.3,
        small_correct=True,
        large_correct=0,
        split="val",
        small_score=0.5,
        latency_small_ms=47.2,
        latency_large_ms=142.3,
        entropy=1.2,
        max_prob=0.8,
        extra={"p_hat": 0.1},
    )
    assert list(rec) == [
        "id",
        "split",
        "u",
        "small_correct",
        "large_correct",
        "small_score",
        "latency_small_ms",
        "latency_large_ms",
        "entropy",
        "max_prob",
        "p_hat",
    ]
    assert rec["small_correct"] == 1
    assert isinstance(rec["small_correct"], int)
    assert cc.make_record("q2", 0.0) == {"id": "q2", "u": 0.0}
    assert cc.make_record("q3", 0.5, small_correct=np.bool_(False))["small_correct"] == 0
    assert cc.make_record("q4", 0.5, small_correct=0.75)["small_correct"] == 0.75


@pytest.mark.parametrize(
    ("kwargs", "err", "pattern"),
    [
        ({"id": ""}, ValueError, "non-empty"),
        ({"id": 3}, ValueError, "non-empty"),
        ({"u": float("nan")}, ValueError, "finite"),
        ({"u": "0.3"}, TypeError, "number"),
        ({"split": "train"}, ValueError, "split"),
        ({"small_correct": 2}, ValueError, "0 or 1"),
        ({"large_correct": 1.5}, ValueError, r"\[0.0, 1.0\]"),
        ({"latency_small_ms": -1.0}, ValueError, "latency_small_ms"),
        ({"max_prob": 1.2}, ValueError, "max_prob"),
        ({"entropy": -0.1}, ValueError, "entropy"),
        ({"extra": {"u": 1}}, ValueError, "clashes"),
    ],
)
def test_make_record_validation(kwargs: dict[str, Any], err: type, pattern: str) -> None:
    base: dict[str, Any] = {"id": "q", "u": 0.5}
    base.update(kwargs)
    with pytest.raises(err, match=pattern):
        cc.make_record(base.pop("id"), base.pop("u"), **base)


def test_jsonl_logger_round_trip(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "traffic.jsonl"
    small, large, _ = _models({"a": 0.2, "b": 0.8})
    payload = load("openai_chat_completion.json")
    cascade = cc.Cascade(ThresholdRouter(0.5), small, large, lambda raw: raw["u"])
    rich = cc.Cascade(
        ThresholdRouter(0.5),
        lambda q: ({"q": q, "obj": object()}, payload),
        large,
        oa.signals_from_chat_completion,
    )
    with cc.JsonlLogger(path) as log:
        log.log(cascade("a"), id="a", small_correct=1, large_correct=1, split="cal")
        log.log(cascade("b"), id="b")
        written = log.log(rich("c"), id="c", include_answers=True)
        log.write({"id": "d", "u": 0.4, "small_correct": 0, "large_correct": 1, "note": "x"})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["a", "b", "c", "d"]
    assert rows[0]["split"] == "cal"
    assert rows[0]["small_correct"] == 1
    assert rows[0]["escalated"] is False
    assert rows[0]["p_hat"] == 0.2
    assert "small_correct" not in rows[1]
    assert rows[1]["escalated"] is True
    assert rows[2]["u"] == pytest.approx(CHAT_U, abs=TOL)
    assert rows[2]["entropy"] == written["entropy"] > 0
    assert 0 < rows[2]["max_prob"] <= 1
    assert isinstance(rows[2]["small_answer"], str)  # non-JSON answer stored as str
    assert rows[3]["note"] == "x"
    with pytest.raises(ValueError, match="closed"):
        log.write({"id": "e", "u": 0.1})
    with cc.JsonlLogger(path, mode="w") as log:
        log.write({"id": "z", "u": 0.9})
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_jsonl_logger_rejects_bad_records(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="mode"):
        cc.JsonlLogger(tmp_path / "x.jsonl", mode="r")
    with cc.JsonlLogger(tmp_path / "x.jsonl") as log:
        with pytest.raises(ValueError, match="at least 'id' and 'u'"):
            log.write({"id": "a"})
        with pytest.raises(ValueError, match="Out of range float"):
            log.write({"id": "a", "u": 0.1, "extra_nan": float("nan")})
        with pytest.raises(ValueError, match="0 or 1"):
            log.write({"id": "a", "u": 0.1, "small_correct": 3})
    assert (tmp_path / "x.jsonl").read_text(encoding="utf-8") == ""


def test_jsonl_logger_is_thread_safe(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "t.jsonl"
    with cc.JsonlLogger(path) as log:

        def worker(k: int) -> None:
            for i in range(50):
                log.write({"id": f"{k}-{i}", "u": i / 50, "note": "x" * 200})

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 400
    assert len({r["id"] for r in rows}) == 400


def test_attach_labels() -> None:
    recs = [{"id": "a", "u": 0.1}, {"id": "b", "u": 0.7, "small_correct": 1}, {"id": "c", "u": 0.2}]
    labels = {"a": (1, 1), "b": (None, 0)}
    with pytest.raises(KeyError, match="'c'"):
        cc.attach_labels(recs, labels)
    out = cc.attach_labels(recs, labels, missing="skip")
    assert out == [
        {"id": "a", "u": 0.1, "small_correct": 1, "large_correct": 1},
        {"id": "b", "u": 0.7, "small_correct": 1, "large_correct": 0},
    ]
    assert "small_correct" not in recs[0]  # inputs untouched
    kept = cc.attach_labels(recs, labels, missing="keep")
    assert kept[2] == {"id": "c", "u": 0.2}
    with pytest.raises(ValueError, match="0 or 1"):
        cc.attach_labels(recs[:1], {"a": (5, 1)})
    with pytest.raises(ValueError, match="missing must be"):
        cc.attach_labels(recs, labels, missing="drop")


def test_token_signals_to_record() -> None:
    sig = oa.signals_from_chat_completion(load("openai_chat_completion.json"))
    rec = sig.to_record()
    assert rec == {"u": sig.u, "entropy": sig.mean_entropy, "max_prob": sig.mean_max_prob}
    assert cc.make_record("q", **rec)["entropy"] == sig.mean_entropy
    assert "margins" not in repr(sig)


# ---------------------------------------------------------------------------
# Property tests (hypothesis, optional)
# ---------------------------------------------------------------------------

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover - hypothesis is a dev dependency
    HAVE_HYPOTHESIS = False
else:
    HAVE_HYPOTHESIS = True

if HAVE_HYPOTHESIS:
    _dist = st.lists(st.floats(0.001, 1.0), min_size=2, max_size=8).map(
        lambda w: [x / sum(w) for x in w]
    )

    @settings(max_examples=150, deadline=None)
    @given(st.lists(_dist, min_size=1, max_size=12), st.randoms())
    def test_property_chat_adapter_equals_core_eq4(dists: list[list[float]], rnd: Any) -> None:
        content = []
        for probs in dists:
            cands = [{"token": f"t{j}", "logprob": math.log(p)} for j, p in enumerate(probs)]
            rnd.shuffle(cands)
            content.append({"token": "x", "logprob": math.log(max(probs)), "top_logprobs": cands})
        sig = oa.signals_from_chat_completion(content)
        top2 = [tuple(sorted(p, reverse=True)[:2]) for p in dists]
        assert sig.u == pytest.approx(token_margin_uncertainty(top2), abs=1e-12)
        assert 0.0 <= sig.u <= 1.0
        assert sig.n_non_greedy == 0
        full_entropy = np.mean([trunc_entropy(p) for p in dists])
        assert sig.mean_entropy == pytest.approx(full_entropy, abs=1e-9)

    @settings(max_examples=60, deadline=None)
    @given(st.integers(1, 4), st.integers(1, 6), st.integers(2, 9), st.integers(0, 2**31 - 1))
    def test_property_transformers_numpy_equals_direct(
        rows: int, steps: int, vocab: int, seed: int
    ) -> None:
        rng = np.random.default_rng(seed)
        logits = [rng.normal(scale=3.0, size=(rows, vocab)) for _ in range(steps)]
        gen = np.stack([np.argmax(x, axis=1) for x in logits], axis=1)
        out = {"sequences": gen, "logits": tuple(logits)}
        sigs = hf.signals_from_generate(out, eos_token_id=None)
        for r, sig in enumerate(sigs):
            assert sig.u == pytest.approx(_manual_signals(logits, r, steps)[0], abs=1e-12)
            assert sig.n_non_greedy == 0


def test_no_warnings_from_adapters_on_fixtures() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        oa.u_from_openai(load("openai_chat_completion.json"))
        vl.u_from_vllm(load("vllm_request_output.json"))
        lc.u_from_llamacpp(load("llamacpp_completion.json"))
        out, _ = _synthetic_generate()
        hf.u_from_generate(out, eos_token_id=6, pad_token_id=5)


# ---------------------------------------------------------------------------
# Remaining edge cases
# ---------------------------------------------------------------------------


def test_chat_entry_without_top_logprobs_and_all_zero_candidates() -> None:
    payload = load("openai_chat_completion.json")
    del payload["choices"][0]["logprobs"]["content"][4]["top_logprobs"]
    with pytest.raises(PayloadError, match=r"content\[4\]\.top_logprobs is missing"):
        oa.u_from_chat_completion(payload)
    payload = load("openai_chat_completion.json")
    for cand in payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
        cand["logprob"] = -math.inf
    with pytest.raises(PayloadError, match="every candidate has probability 0"):
        oa.u_from_chat_completion(payload)


def test_chat_chunks_skip_other_choices_and_empty_chunks() -> None:
    chunks = load("openai_chat_completion_chunks.json")
    other = copy.deepcopy(chunks[2])
    other["choices"][0]["index"] = 1
    other["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = []
    chunks.insert(3, other)
    chunks.insert(0, {"object": "chat.completion.chunk", "choices": None})
    assert oa.u_from_chat_completion_chunks(chunks) == pytest.approx(CHAT_U, abs=TOL)
    with pytest.raises(PayloadError, match="0 candidate"):
        oa.u_from_chat_completion_chunks(chunks, choice_index=1)
    with pytest.raises(ValueError, match="part must be one of"):
        oa.u_from_chat_completion_chunks(chunks, part="text")


def test_openai_auto_dispatch_without_object_field() -> None:
    chat = load("openai_chat_completion.json")
    del chat["object"]
    assert oa.u_from_openai(chat) == pytest.approx(CHAT_U, abs=TOL)
    comp = load("openai_completion.json")
    del comp["object"]
    assert oa.u_from_openai(comp) == pytest.approx(COMPLETION_U, abs=TOL)
    resp = load("openai_responses.json")
    del resp["object"]
    assert oa.u_from_openai(resp) == pytest.approx(RESPONSES_U, abs=TOL)


def test_vllm_rejects_a_list_of_request_outputs_and_bad_positions() -> None:
    payload = load("vllm_request_output.json")
    with pytest.raises(PayloadError, match="u_from_vllm_batch"):
        vl.u_from_vllm([vllm_objects(payload), vllm_objects(payload)])
    with pytest.raises(PayloadError, match="must map token ids"):
        vl.u_from_vllm([[-0.1, -2.0]])


def test_llamacpp_rejects_non_numeric_probability() -> None:
    payload = load("llamacpp_completion_legacy.json")
    payload["completion_probabilities"][0]["probs"][1]["prob"] = "0.2"
    with pytest.raises(PayloadError, match="must be a number"):
        lc.u_from_llamacpp(payload)


def test_cascade_record_with_answers_of_both_models() -> None:
    small, large, _ = _models({"hard": 0.9})
    res = cc.Cascade(ThresholdRouter(0.5), small, large, lambda raw: raw["u"])("hard")
    rec = res.to_record("hard", small_correct=0, large_correct=1, include_answers=True)
    assert rec["small_answer"] == "small:hard"
    assert rec["large_answer"] == "large:hard"
    assert rec["escalated"] is True
    assert json.loads(json.dumps(rec)) == rec


def test_transformers_input_validation_helpers() -> None:
    out, _ = _synthetic_generate()
    with pytest.raises(PayloadError, match="no top-2 margin"):
        hf.signals_from_generate(
            {"sequences": out.sequences, "logits": tuple(x[:, :1] for x in out.logits)},
            eos_token_id=None,
        )
    with pytest.raises(PayloadError, match="not a numeric array"):
        hf.signals_from_generate(
            {"sequences": out.sequences, "logits": (np.array([["a", "b"]] * 3),) * 5},
            eos_token_id=None,
        )
    with pytest.raises(PayloadError, match=r"shape \(batch, length\)"):
        hf.signals_from_generate(
            {"sequences": out.sequences[0], "logits": out.logits}, eos_token_id=None
        )
    with pytest.raises(TypeError, match="int or a list of ints"):
        hf.signals_from_generate(out, eos_token_id=3.5)
    with pytest.raises(TypeError, match="must contain ints"):
        hf.signals_from_generate(out, eos_token_id=[6, "7"])


@pytest.mark.parametrize(
    ("version", "expected"), [("4.37.2", False), ("4.38.0.dev0", True), ("5.1", True)]
)
def test_output_logits_version_gate(
    monkeypatch: pytest.MonkeyPatch, version: str, expected: bool
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(__version__=version))
    assert hf._transformers_has_output_logits() is expected


def test_generate_with_signals_explains_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    with pytest.raises(ImportError, match="PyTorch"):
        hf.generate_with_signals(object(), object(), "x")


def test_generate_with_signals_stopping_criteria_and_old_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch, transformers = _hf()
    tok, model = _load(transformers, TINY_GPT2)
    prompts = ["hello world", "the quick brown fox"]
    base = hf.generate_with_signals(model, tok, prompts, max_new_tokens=6)
    assert base.output.logits is not None

    class StopSecondRow(transformers.StoppingCriteria):  # type: ignore[misc]
        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            done[1] = input_ids.shape[1] - prompt_len >= 2
            return done

    prompt_len = base.output.sequences.shape[1] - len(base.output.logits)
    stopped = hf.generate_with_signals(
        model,
        tok,
        prompts,
        max_new_tokens=6,
        pad_token_id=5,
        stopping_criteria=transformers.StoppingCriteriaList([StopSecondRow()]),
    )
    assert stopped.signals[1].n_tokens == 2
    assert stopped.signals[0].n_tokens == base.signals[0].n_tokens
    assert stopped.u[0] == pytest.approx(base.u[0], abs=1e-12)
    # Before transformers 4.38 there is no output_logits: scores are used instead.
    monkeypatch.setattr(hf, "_transformers_has_output_logits", lambda: False)
    old = hf.generate_with_signals(model, tok, prompts, max_new_tokens=6)
    assert old.output.logits is None
    assert old.signals[0].source == "transformers.scores"
    assert old.u == pytest.approx(base.u, abs=1e-12)


def test_logged_traffic_round_trips_through_the_cli(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Shadow-mode logging, offline labelling, `ucci fit`, then routing with the fitted router."""
    cli = pytest.importorskip("ucci.cli")
    io = pytest.importorskip("ucci.io")
    rng = np.random.default_rng(7)
    n = 800
    us = rng.random(n)
    small_ok = (rng.random(n) >= us).astype(int)  # P(small wrong | u) = u
    cascade = cc.Cascade(
        ThresholdRouter(0.5),
        lambda q: (f"s{q}", float(us[q])),
        lambda q: f"l{q}",
        lambda raw: raw,
        shadow_large=True,
    )
    raw_path = tmp_path / "traffic.jsonl"
    with cc.JsonlLogger(raw_path) as log:
        for i, res in enumerate(cascade.map(range(n))):
            log.log(res, id=f"q{i}", include_answers=True)
    logged = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    assert all("large_answer" in r for r in logged)  # shadow mode ran the large model
    labelled = cc.attach_labels(logged, {f"q{i}": (int(small_ok[i]), 1) for i in range(n)})
    data = tmp_path / "labelled.jsonl"
    with cc.JsonlLogger(data, mode="w") as log:
        for rec in labelled:
            log.write(rec)
    out = tmp_path / "router.json"
    assert cli.main(["fit", "--data", str(data), "--tau", "0.9", "--out", str(out)]) == 0
    capsys.readouterr()
    fitted = io.load_router(out)
    served = cc.Cascade(fitted, lambda q: ("s", q), lambda q: "l", lambda raw: raw)
    for u in (0.05, 0.5, 0.95):
        res = served(u)
        assert res.p_hat == pytest.approx(float(fitted.error_probability([u])[0]))
        assert res.escalated == bool(fitted.escalate([u])[0])
    assert served(0.99).escalated
