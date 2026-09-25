"""Model backends for the CoNLL-2003 replication: Hugging Face transformers and vLLM.

Both backends run greedy decoding (paper Section 4.1 and Appendix B.2:
temperature 0) and return, for every generated content token, the top-1 and
top-2 next-token probabilities (for u(x), paper Eq. 4), the predictive
entropy and the maximum probability (for the entropy and max-probability
signals used by the baselines and ablations of paper Section 6.1 and 6.3).

Token convention (shared with the rest of the package): every generated
content token counts, the terminating end-of-sequence token and any padding
after it do not. A generation that hits ``max_new_tokens`` has no stop token
and all its tokens count.

``torch``, ``transformers`` and ``vllm`` are imported only when a backend is
built, so the module is importable without them.
"""

from __future__ import annotations

import math
import platform
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["Generation", "TransformersBackend", "VLLMBackend", "make_backend"]


@dataclass
class Generation:
    """One greedy generation with per-token statistics over its content tokens.

    Attributes
    ----------
    text : str
        Decoded content tokens, special tokens skipped.
    token_ids : list of int
        Content token ids (stop token and padding excluded).
    p1, p2 : list of float
        Top-1 and top-2 next-token probabilities at each content position.
        Under greedy decoding ``p1`` is the probability of the emitted token.
    entropy : list of float
        Predictive entropy (nats) of the next-token distribution at each
        content position: over the full output vocabulary for the
        transformers backend, over the returned top-k for vLLM.
    stop_reason : str
        ``"stop"`` when a stop token ended the generation, ``"length"`` when
        ``max_new_tokens`` did.
    argmax_mismatch : int
        Content positions where the emitted token is not the arg-max of the
        logits used for the statistics. Always 0 for true greedy decoding on
        unmodified logits; anything else means a logits processor was active.
    prompt_tokens : int
        Prompt length in tokens.
    """

    text: str
    token_ids: List[int]
    p1: List[float]
    p2: List[float]
    entropy: List[float]
    stop_reason: str
    argmax_mismatch: int = 0
    prompt_tokens: int = 0

    @property
    def n_tokens(self) -> int:
        """Number of content tokens T."""
        return len(self.token_ids)


def _content_length(ids: Sequence[int], stop_ids: Sequence[int]) -> int:
    """Index of the first stop token, or ``len(ids)`` when there is none."""
    stop = set(stop_ids)
    for i, t in enumerate(ids):
        if t in stop:
            return i
    return len(ids)


def _resolve_device(device: str) -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype: str, device: str) -> Any:
    """Map a dtype name to a torch dtype; ``"auto"`` picks bf16, then fp16 on GPUs, fp32 on CPU."""
    import torch

    names = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if dtype != "auto":
        if dtype not in names:
            raise ValueError(f"dtype must be 'auto' or one of {sorted(names)}, got {dtype!r}")
        return names[dtype]
    if device == "cpu":
        return torch.float32
    try:
        torch.ones(2, device=device, dtype=torch.bfloat16).sum().item()
        return torch.bfloat16
    except (RuntimeError, TypeError):
        return torch.float16


class TransformersBackend:
    """Greedy generation with Hugging Face transformers, batched with left padding.

    Per-token statistics are computed from the raw next-token logits by a
    logits processor that leaves the logits unchanged. The model's own
    generation config is replaced by a plain greedy one, because instruction
    models often ship sampling defaults and a repetition penalty that would
    otherwise change the decoded tokens. Every run checks that each emitted
    token is the arg-max of the logits the statistics were computed from,
    and refuses to return NaN statistics.

    Parameters
    ----------
    model_id : str
        Hugging Face model id or local path.
    revision : str, optional
        Model commit.
    device : str
        ``"auto"`` (cuda, then mps, then cpu), or any torch device string.
    dtype : str
        ``"auto"``, ``"bfloat16"``, ``"float16"`` or ``"float32"``.
    max_new_tokens : int
        Generation cap, 256 as in paper Appendix B.2.
    attn_implementation : str
        ``"auto"`` uses ``"eager"`` on Apple MPS and the transformers default
        elsewhere; ``"default"`` always uses the transformers default
        (normally ``"sdpa"``); any other value is passed to
        ``from_pretrained``. PyTorch's fused attention on MPS returns NaN for
        the padding rows of a left-padded batch, which then leak into the
        real rows; the eager implementation masks with a finite value and
        does not. A batch of one has no padding, so latency timing at batch
        size 1 can use the default kernel.
    """

    name = "transformers"

    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        attn_implementation: str = "auto",
    ) -> None:
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "the transformers backend needs torch and transformers: "
                "pip install torch 'transformers>=4.43'"
            ) from exc
        self._torch = torch
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = int(max_new_tokens)
        self.device = _resolve_device(device)
        self.torch_dtype = _resolve_dtype(dtype, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        major_minor = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        dtype_kw = "dtype" if major_minor >= (4, 56) else "torch_dtype"
        load_kw: Dict[str, Any] = {dtype_kw: self.torch_dtype}
        if attn_implementation == "auto":
            attn_implementation = "eager" if self.device == "mps" else ""
        if attn_implementation == "default":
            attn_implementation = ""
        if attn_implementation:
            load_kw["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, **load_kw)
        self.attn_implementation = str(
            getattr(model.config, "_attn_implementation", attn_implementation or "default")
        )
        self.model = model.to(self.device).eval()

        stop = model.generation_config.eos_token_id
        stop_ids = [stop] if isinstance(stop, int) else list(stop or [])
        if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id not in stop_ids:
            stop_ids.append(self.tokenizer.eos_token_id)
        if not stop_ids:
            raise ValueError(f"{model_id} defines no end-of-sequence token")
        self.stop_ids: List[int] = [int(s) for s in stop_ids]
        self.model.generation_config = GenerationConfig(
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.stop_ids,
            pad_token_id=int(self.tokenizer.pad_token_id),
            bos_token_id=model.generation_config.bos_token_id,
        )
        self._versions = {"torch": torch.__version__, "transformers": transformers.__version__}
        self.last_cross_check: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ helpers
    def render(self, messages: List[Dict[str, str]]) -> str:
        """Apply the model's chat template and add the assistant turn header."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def prompt_length(self, prompt: str) -> int:
        """Prompt length in tokens (the chat template already holds any special tokens)."""
        return len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    def synchronize(self) -> None:
        """Block until queued device work finishes, so wall-clock timings are complete."""
        torch = self._torch
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def describe(self) -> Dict[str, Any]:
        """Settings and versions for the run metadata."""
        return {
            "backend": self.name,
            "model": self.model_id,
            "revision": self.revision,
            "device": self.device,
            "dtype": str(self.torch_dtype).replace("torch.", ""),
            "attn_implementation": self.attn_implementation,
            "max_new_tokens": self.max_new_tokens,
            "stop_token_ids": self.stop_ids,
            "entropy_support": "full output vocabulary",
            "vocab_size": int(self.model.get_output_embeddings().weight.shape[0]),
            "versions": dict(self._versions),
            "platform": platform.platform(),
        }

    # --------------------------------------------------------------- generation
    def generate(
        self, prompts: Sequence[str], collect_stats: bool = True, cross_check: bool = False
    ) -> List[Generation]:
        """Greedy-decode a batch of rendered prompts.

        Parameters
        ----------
        prompts : sequence of str
            Outputs of :meth:`render`.
        collect_stats : bool
            False skips the per-token statistics (used for latency timing, so
            the measured cost is plain greedy generation). The lists ``p1``,
            ``p2`` and ``entropy`` are then empty.
        cross_check : bool
            Also keep the raw logits and recompute u(x), mean entropy and mean
            maximum probability with the package adapter
            :func:`ucci.integrations.transformers.signals_from_generate`
            (float64 on the host). The largest absolute differences are stored
            in :attr:`last_cross_check`. Costs memory; use on a few batches.
        """
        torch = self._torch
        from transformers import LogitsProcessor, LogitsProcessorList

        class _StepStats(LogitsProcessor):
            """Records top-2 probabilities, entropy and arg-max; returns logits unchanged."""

            def __init__(self) -> None:
                self.p1: List[Any] = []
                self.p2: List[Any] = []
                self.ent: List[Any] = []
                self.arg: List[Any] = []

            def __call__(self, input_ids: Any, scores: Any) -> Any:
                with torch.no_grad():
                    logp = torch.log_softmax(scores.float(), dim=-1)
                    p = logp.exp()
                    top = torch.topk(logp, 2, dim=-1)
                    plogp = torch.where(p > 0, p * logp, torch.zeros_like(p))
                    self.p1.append(top.values[:, 0].exp())
                    self.p2.append(top.values[:, 1].exp())
                    self.ent.append(-plogp.sum(dim=-1))
                    self.arg.append(top.indices[:, 0])
                return scores

        enc = self.tokenizer(
            list(prompts), return_tensors="pt", padding=True, add_special_tokens=False
        )
        # Only ids and mask: some tokenizers add token_type_ids, which causal LMs reject.
        enc = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        stats = _StepStats() if collect_stats else None
        kwargs: Dict[str, Any] = {"return_dict_in_generate": True}
        if stats is not None:
            kwargs["logits_processor"] = LogitsProcessorList([stats])
        if cross_check and stats is not None:
            kwargs["output_logits"] = True
        with torch.no_grad():
            out = self.model.generate(**enc, **kwargs)
        gen = out.sequences[:, enc["input_ids"].shape[1] :].cpu().tolist()

        if stats is not None and stats.p1:
            # MPS has no float64: move to the CPU before widening.
            p1 = torch.stack(stats.p1, dim=1).cpu().double().tolist()
            p2 = torch.stack(stats.p2, dim=1).cpu().double().tolist()
            ent = torch.stack(stats.ent, dim=1).cpu().double().tolist()
            arg = torch.stack(stats.arg, dim=1).cpu().tolist()
            for b, ids in enumerate(gen):
                t = _content_length(ids, self.stop_ids)
                if any(math.isnan(x) for x in p1[b][:t]) or any(math.isnan(x) for x in ent[b][:t]):
                    raise FloatingPointError(
                        f"NaN next-token probabilities in batch row {b} on {self.device} "
                        f"({self.attn_implementation} attention, {self.torch_dtype}); "
                        "try attn_implementation='eager' or dtype='float32'"
                    )
        else:
            p1 = p2 = ent = arg = [[] for _ in gen]

        results: List[Generation] = []
        for b, ids in enumerate(gen):
            t = _content_length(ids, self.stop_ids)
            content = [int(x) for x in ids[:t]]
            mism = sum(1 for i in range(t) if collect_stats and arg[b][i] != content[i])
            results.append(
                Generation(
                    text=self.tokenizer.decode(content, skip_special_tokens=True),
                    token_ids=content,
                    p1=[float(x) for x in p1[b][:t]] if collect_stats else [],
                    p2=[float(x) for x in p2[b][:t]] if collect_stats else [],
                    entropy=[float(x) for x in ent[b][:t]] if collect_stats else [],
                    stop_reason="stop" if t < len(ids) else "length",
                    argmax_mismatch=mism,
                    prompt_tokens=int(prompt_lens[b]),
                )
            )
        if cross_check and stats is not None:
            self.last_cross_check = self._cross_check(out, results)
        return results

    def _cross_check(self, out: Any, results: Sequence[Generation]) -> Dict[str, Any]:
        """Largest differences between these statistics and the package adapter's."""
        from ucci.integrations.transformers import signals_from_generate

        sigs = signals_from_generate(
            out, eos_token_id=self.stop_ids, require_greedy=True, on_empty="nan"
        )
        du = de = dm = 0.0
        n_tok = 0
        for g, sg in zip(results, sigs):
            if g.n_tokens != sg.n_tokens:
                raise RuntimeError(
                    f"content length {g.n_tokens} differs from the adapter's {sg.n_tokens}"
                )
            if g.n_tokens == 0:
                continue
            u = 1.0 - sum(a - b for a, b in zip(g.p1, g.p2)) / g.n_tokens
            du = max(du, abs(u - sg.u))
            de = max(de, abs(sum(g.entropy) / g.n_tokens - sg.mean_entropy))
            dm = max(dm, abs(sum(g.p1) / g.n_tokens - sg.mean_max_prob))
            n_tok += g.n_tokens
        return {
            "rows": len(results),
            "tokens": n_tok,
            "max_abs_diff_u": du,
            "max_abs_diff_mean_entropy": de,
            "max_abs_diff_mean_max_prob": dm,
        }

    def close(self) -> None:
        """Release the model and free accelerator memory."""
        torch = self._torch
        del self.model
        import gc

        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()


class VLLMBackend:
    """Greedy generation with vLLM offline inference (CUDA).

    vLLM returns log-probabilities for the top ``logprobs`` candidates only,
    so the entropy recorded here is the truncated entropy over those
    candidates, ``-sum_{k<=K} p_k log p_k`` with K = ``logprobs`` (default
    20, vLLM's default cap), not the full-vocabulary entropy of the
    transformers backend. u(x) and the maximum probability are exact.

    Parameters
    ----------
    model_id, revision, max_new_tokens
        As for :class:`TransformersBackend`.
    dtype : str
        Passed to ``vllm.LLM`` (``"auto"`` uses the checkpoint dtype).
    logprobs : int
        Candidates per position, at least 2.
    llm_kwargs : dict, optional
        Extra keyword arguments for ``vllm.LLM`` (for example
        ``gpu_memory_utilization`` or ``tensor_parallel_size``).
    """

    name = "vllm"

    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: str = "auto",
        max_new_tokens: int = 256,
        logprobs: int = 20,
        llm_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            import vllm
            from vllm import LLM, SamplingParams
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "the vllm backend needs vLLM on a CUDA machine: pip install vllm"
            ) from exc
        if logprobs < 2:
            raise ValueError(f"logprobs must be at least 2 for the top-2 margin, got {logprobs}")
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = int(max_new_tokens)
        self.logprobs = int(logprobs)
        kwargs = dict(llm_kwargs or {})
        kwargs.setdefault("max_logprobs", max(20, self.logprobs))
        self.llm = LLM(model=model_id, revision=revision, dtype=dtype, seed=0, **kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = SamplingParams(
            temperature=0.0, max_tokens=self.max_new_tokens, logprobs=self.logprobs
        )
        stop_ids: List[int] = []
        try:
            from transformers import GenerationConfig

            stop = GenerationConfig.from_pretrained(model_id, revision=revision).eos_token_id
            stop_ids = [stop] if isinstance(stop, int) else list(stop or [])
        except (OSError, ValueError):
            pass  # no generation_config.json: fall back to the tokenizer's EOS below
        if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id not in stop_ids:
            stop_ids.append(self.tokenizer.eos_token_id)
        self.stop_ids = [int(s) for s in stop_ids]
        self.dtype = dtype
        self._version = vllm.__version__
        self.last_cross_check: Optional[Dict[str, Any]] = None

    def render(self, messages: List[Dict[str, str]]) -> str:
        """Apply the model's chat template and add the assistant turn header."""
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def prompt_length(self, prompt: str) -> int:
        """Prompt length in tokens."""
        return len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    def synchronize(self) -> None:
        """vLLM calls return after the work is done; nothing to wait for."""

    def describe(self) -> Dict[str, Any]:
        """Settings and versions for the run metadata."""
        return {
            "backend": self.name,
            "model": self.model_id,
            "revision": self.revision,
            "device": "cuda",
            "dtype": self.dtype,
            "max_new_tokens": self.max_new_tokens,
            "stop_token_ids": self.stop_ids,
            "entropy_support": f"top-{self.logprobs} candidates (truncated entropy)",
            "versions": {"vllm": self._version},
            "platform": platform.platform(),
        }

    def generate(
        self, prompts: Sequence[str], collect_stats: bool = True, cross_check: bool = False
    ) -> List[Generation]:
        """Greedy-decode a batch of rendered prompts; see :meth:`TransformersBackend.generate`.

        With ``cross_check`` the signals are recomputed from the same outputs
        with :func:`ucci.integrations.vllm.signals_from_vllm` and the largest
        differences are stored in :attr:`last_cross_check`.
        """
        token_prompts = [
            {"prompt_token_ids": self.tokenizer(p, add_special_tokens=False)["input_ids"]}
            for p in prompts
        ]
        outs = self.llm.generate(token_prompts, self.sampling, use_tqdm=False)
        results: List[Generation] = []
        for prompt, o in zip(token_prompts, outs):
            c = o.outputs[0]
            ids = [int(x) for x in c.token_ids]
            t = _content_length(ids, self.stop_ids)
            p1: List[float] = []
            p2: List[float] = []
            ent: List[float] = []
            mism = 0
            if collect_stats:
                if c.logprobs is None or len(c.logprobs) < t:
                    raise RuntimeError(
                        "vLLM returned no per-token logprobs; is logprobs set in SamplingParams?"
                    )
                for i in range(t):
                    cand = c.logprobs[i]
                    lps = sorted(
                        ((float(getattr(v, "logprob", v)), int(k)) for k, v in cand.items()),
                        reverse=True,
                    )
                    if len(lps) < 2:
                        raise RuntimeError(
                            f"position {i}: vLLM returned {len(lps)} logprobs, need 2"
                        )
                    p1.append(math.exp(lps[0][0]))
                    p2.append(math.exp(lps[1][0]))
                    ent.append(-sum(math.exp(lp) * lp for lp, _ in lps[: self.logprobs]))
                    mism += int(lps[0][1] != ids[i])
            results.append(
                Generation(
                    text=self.tokenizer.decode(ids[:t], skip_special_tokens=True),
                    token_ids=ids[:t],
                    p1=p1,
                    p2=p2,
                    entropy=ent,
                    stop_reason="stop" if (t < len(ids) or c.finish_reason == "stop") else "length",
                    argmax_mismatch=mism,
                    prompt_tokens=len(prompt["prompt_token_ids"]),
                )
            )
        if cross_check and collect_stats:
            self.last_cross_check = self._cross_check(outs, results)
        return results

    def _cross_check(self, outs: Sequence[Any], results: Sequence[Generation]) -> Dict[str, Any]:
        """Largest differences between these statistics and the package adapter's."""
        from ucci.integrations.vllm import signals_from_vllm

        du = de = dm = 0.0
        n_tok = 0
        for o, g in zip(outs, results):
            if g.n_tokens == 0:
                continue
            sg = signals_from_vllm(o)
            if sg.n_tokens != g.n_tokens:
                raise RuntimeError(
                    f"content length {g.n_tokens} differs from the adapter's {sg.n_tokens}"
                )
            u = 1.0 - sum(a - b for a, b in zip(g.p1, g.p2)) / g.n_tokens
            du = max(du, abs(u - sg.u))
            de = max(de, abs(sum(g.entropy) / g.n_tokens - sg.mean_entropy))
            dm = max(dm, abs(sum(g.p1) / g.n_tokens - sg.mean_max_prob))
            n_tok += g.n_tokens
        return {
            "rows": len(results),
            "tokens": n_tok,
            "max_abs_diff_u": du,
            "max_abs_diff_mean_entropy": de,
            "max_abs_diff_mean_max_prob": dm,
        }

    def close(self) -> None:
        """Drop the engine."""
        del self.llm


def make_backend(
    backend: str,
    model_id: str,
    revision: Optional[str] = None,
    device: str = "auto",
    dtype: str = "auto",
    max_new_tokens: int = 256,
    attn_implementation: str = "auto",
) -> Any:
    """Build a backend by name (``"transformers"`` or ``"vllm"``)."""
    if backend == "transformers":
        return TransformersBackend(
            model_id,
            revision,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            attn_implementation=attn_implementation,
        )
    if backend == "vllm":
        if device not in ("auto", "cuda"):
            raise ValueError("the vllm backend runs on CUDA only; use --device auto")
        return VLLMBackend(model_id, revision, dtype=dtype, max_new_tokens=max_new_tokens)
    raise ValueError(f"backend must be 'transformers' or 'vllm', got {backend!r}")


def now_utc() -> str:
    """ISO-8601 UTC timestamp for run metadata."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
