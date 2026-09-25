"""Tests for the model backends.

The transformers tests build a tiny randomly initialized Qwen2 model and a
word-level tokenizer in a temporary directory (no download) and run on the
CPU; they are skipped when torch or transformers is missing. The vLLM tests
drive :class:`backends.VLLMBackend` with a fake engine shaped like vLLM's
``RequestOutput``, so they need neither vLLM nor a GPU.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backends

import ucci

# ---------------------------------------------------------------------------
# vLLM backend with a fake engine
# ---------------------------------------------------------------------------


class _Tok:
    eos_token_id = 2

    def __call__(self, text: str, add_special_tokens: bool = False) -> Dict[str, List[int]]:
        return {"input_ids": [10 + (ord(c) % 50) for c in text]}

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"t{i}" for i in ids if not (skip_special_tokens and i == 2))


def _lp(pairs: Dict[int, float]) -> Dict[int, SimpleNamespace]:
    return {k: SimpleNamespace(logprob=v) for k, v in pairs.items()}


def _vllm_output(
    token_ids: List[int], logprobs: List[Dict[int, float]], finish: str, stop_reason: Any = None
) -> Any:
    comp = SimpleNamespace(
        token_ids=token_ids,
        logprobs=[_lp(d) for d in logprobs],
        finish_reason=finish,
        stop_reason=stop_reason,
        text="",
    )
    return SimpleNamespace(outputs=[comp])


class _Engine:
    def __init__(self, outs: List[Any]) -> None:
        self.outs = outs
        self.prompts: List[Any] = []

    def generate(self, prompts: List[Any], sampling: Any, use_tqdm: bool = False) -> List[Any]:
        self.prompts = prompts
        return self.outs


def _vllm_backend(outs: List[Any], logprobs: int = 3) -> backends.VLLMBackend:
    b = object.__new__(backends.VLLMBackend)
    b.llm = _Engine(outs)
    b.tokenizer = _Tok()
    b.sampling = None
    b.stop_ids = [2]
    b.logprobs = logprobs
    b.max_new_tokens = 256
    b.last_cross_check = None
    return b


def test_vllm_drops_the_stop_token_and_matches_the_adapter() -> None:
    lp = [
        {5: math.log(0.7), 6: math.log(0.2), 7: math.log(0.05)},
        {8: math.log(0.9), 5: math.log(0.06), 6: math.log(0.01)},
        {2: math.log(0.95), 5: math.log(0.03), 6: math.log(0.01)},  # the EOS position
    ]
    outs = [
        _vllm_output([5, 8, 2], lp, "stop"),
        _vllm_output([5, 8], lp[:2], "length"),
    ]
    b = _vllm_backend(outs)
    gens = b.generate(["ab", "abc"], cross_check=True)
    assert [g.n_tokens for g in gens] == [2, 2]
    assert [g.stop_reason for g in gens] == ["stop", "length"]
    g = gens[0]
    assert g.p1 == pytest.approx([0.7, 0.9]) and g.p2 == pytest.approx([0.2, 0.06])
    assert ucci.token_margin_uncertainty(list(zip(g.p1, g.p2))) == pytest.approx(
        1 - (0.5 + 0.84) / 2
    )
    ent0 = -(0.7 * math.log(0.7) + 0.2 * math.log(0.2) + 0.05 * math.log(0.05))
    assert g.entropy[0] == pytest.approx(ent0)
    assert g.argmax_mismatch == 0 and g.prompt_tokens == 2
    assert b.last_cross_check is not None and b.last_cross_check["max_abs_diff_u"] < 1e-12
    assert b.llm.prompts[1] == {"prompt_token_ids": _Tok()("abc")["input_ids"]}


def test_vllm_counts_non_greedy_tokens_and_needs_two_candidates() -> None:
    lp = [{5: math.log(0.6), 6: math.log(0.3)}]
    b = _vllm_backend([_vllm_output([6], lp, "length")])
    assert b.generate(["x"])[0].argmax_mismatch == 1
    b = _vllm_backend([_vllm_output([5], [{5: math.log(0.6)}], "length")])
    with pytest.raises(RuntimeError, match="need 2"):
        b.generate(["x"])
    comp = SimpleNamespace(
        token_ids=[5], logprobs=None, finish_reason="length", stop_reason=None, text=""
    )
    b = _vllm_backend([SimpleNamespace(outputs=[comp])])
    with pytest.raises(RuntimeError, match="logprobs"):
        b.generate(["x"])


def test_vllm_empty_generation() -> None:
    lp = [{2: math.log(0.9), 5: math.log(0.05)}]
    g = _vllm_backend([_vllm_output([2], lp, "stop")]).generate(["x"], cross_check=True)[0]
    assert g.n_tokens == 0 and g.p1 == [] and g.stop_reason == "stop"


# ---------------------------------------------------------------------------
# transformers backend with a tiny random model
# ---------------------------------------------------------------------------

WORDS = ["<pad>", "<eos>", "<unk>"] + [f"w{i}" for i in range(60)]
CHAT_TEMPLATE = (
    "{% for m in messages %}<{{ m['role'] }}> {{ m['content'] }} {% endfor %}<assistant>"
)


@pytest.fixture(scope="module")
def torch() -> Any:
    """torch, or skip: only the transformers tests need it."""
    pytest.importorskip("tokenizers")
    mod = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    try:  # transformers imports lazily; a broken dependency only shows up here
        from transformers import AutoModelForCausalLM, GenerationConfig  # noqa: F401
    except Exception as exc:
        pytest.skip(f"transformers is installed but not importable here: {type(exc).__name__}")
    return mod


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory, torch: Any) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    d = tmp_path_factory.mktemp("tiny")
    vocab = {w: i for i, w in enumerate(WORDS + ["<user>", "<assistant>"])}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk, eos_token="<eos>", pad_token="<pad>", unk_token="<unk>"
    )
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(d)
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        eos_token_id=1,
        pad_token_id=0,
        bos_token_id=None,
        tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(cfg)
    model.generation_config.eos_token_id = 1
    model.generation_config.repetition_penalty = 1.3  # a default the backend must switch off
    model.generation_config.do_sample = True
    model.save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def tiny(tiny_model_dir: Path) -> backends.TransformersBackend:
    return backends.TransformersBackend(
        str(tiny_model_dir), device="cpu", dtype="float32", max_new_tokens=6
    )


PROMPTS = ["w1 w2 w3 w4 w5 w6", "w7", "w8 w9 w10"]


def test_greedy_config_replaces_model_defaults(tiny: backends.TransformersBackend) -> None:
    gc = tiny.model.generation_config
    assert gc.do_sample is False and gc.num_beams == 1
    assert gc.repetition_penalty in (None, 1.0)
    assert tiny.stop_ids == [1]


def test_stats_match_the_package_adapter_on_a_padded_batch(
    tiny: backends.TransformersBackend,
) -> None:
    gens = tiny.generate(PROMPTS, cross_check=True)
    chk = tiny.last_cross_check
    assert chk is not None and chk["rows"] == 3
    assert chk["max_abs_diff_u"] < 1e-6
    assert chk["max_abs_diff_mean_entropy"] < 1e-5 and chk["max_abs_diff_mean_max_prob"] < 1e-6
    for g in gens:
        assert g.argmax_mismatch == 0
        assert len(g.p1) == len(g.p2) == len(g.entropy) == g.n_tokens <= 6
        assert all(0.0 <= b <= a <= 1.0 for a, b in zip(g.p1, g.p2))
        assert all(0.0 <= h <= math.log(len(WORDS) + 2) + 1e-6 for h in g.entropy)


def test_batched_equals_single_in_float32(tiny: backends.TransformersBackend) -> None:
    batched = tiny.generate(PROMPTS)
    for p, g in zip(PROMPTS, batched):
        single = tiny.generate([p])[0]
        assert single.token_ids == g.token_ids
        assert single.p1 == pytest.approx(g.p1, abs=1e-5)


def test_stop_token_is_excluded(tiny: backends.TransformersBackend) -> None:
    first = tiny.generate(PROMPTS[:1])[0]
    assert first.n_tokens >= 3
    stop = first.token_ids[2]
    old_stop, old_eos = tiny.stop_ids, tiny.model.generation_config.eos_token_id
    try:
        tiny.stop_ids = [stop]
        tiny.model.generation_config.eos_token_id = [stop]
        g = tiny.generate(PROMPTS[:1], cross_check=True)[0]
    finally:
        tiny.stop_ids = old_stop
        tiny.model.generation_config.eos_token_id = old_eos
    assert g.stop_reason == "stop"
    assert g.token_ids == first.token_ids[: first.token_ids.index(stop)]
    assert g.p1 == pytest.approx(first.p1[: g.n_tokens], abs=1e-6)
    assert tiny.last_cross_check is not None and tiny.last_cross_check["max_abs_diff_u"] < 1e-6


def test_latency_mode_skips_stats(tiny: backends.TransformersBackend) -> None:
    g = tiny.generate(PROMPTS[:1], collect_stats=False)[0]
    assert g.p1 == [] and g.entropy == [] and g.n_tokens > 0


def test_render_and_describe(tiny: backends.TransformersBackend) -> None:
    assert tiny.render([{"role": "user", "content": "w1"}]) == "<user> w1 <assistant>"
    assert tiny.prompt_length("w1 w2") == 2
    d = tiny.describe()
    assert d["dtype"] == "float32" and d["device"] == "cpu" and d["stop_token_ids"] == [1]
    assert d["vocab_size"] == len(WORDS) + 2
    tiny.synchronize()


def test_dtype_names_are_validated(torch: Any) -> None:
    with pytest.raises(ValueError, match="dtype"):
        backends._resolve_dtype("float8", "cpu")
    assert backends._resolve_dtype("auto", "cpu") == torch.float32
