"""Tests for generate.py and latency.py with a fake backend (no model, no network)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backends
import conll
import generate
import latency

import ucci


def _pool(n: int) -> List[conll.Sentence]:
    out = []
    for i in range(n):
        tokens = tuple(["Ann", "met", "Bo", "in", "Oslo"][: 2 + i % 4]) + (f"w{i}",)  # unique text
        tags = ["B-PER", "O", "B-PER", "O", "B-LOC"][: len(tokens) - 1] + ["O"]
        out.append(
            conll.Sentence(f"test-{i}", "test", tokens, conll.entities_from_tags(tokens, tags))
        )
    return out


class FakeBackend:
    """Deterministic stand-in: answers with the gold entities of every other sentence."""

    name = "transformers"

    def __init__(self, pool: Sequence[conll.Sentence]) -> None:
        self.by_text = {s.text: s for s in pool}
        self.calls: List[int] = []
        self.last_cross_check: Dict[str, Any] = {}

    def render(self, messages: List[Dict[str, str]]) -> str:
        return "<user>" + messages[0]["content"]

    def prompt_length(self, prompt: str) -> int:
        return len(prompt.split())

    def synchronize(self) -> None:
        pass

    def describe(self) -> Dict[str, Any]:
        return {"backend": "fake", "device": "cpu", "dtype": "float32", "versions": {}}

    def generate(
        self, prompts: Sequence[str], collect_stats: bool = True, cross_check: bool = False
    ) -> List[backends.Generation]:
        self.calls.append(len(prompts))
        out = []
        for p in prompts:
            text = p.split("Sentence: ")[1].split("\nOutput:")[0]
            s = self.by_text[text]
            idx = int(s.id.split("-")[1])
            ents = s.gold if idx % 2 == 0 else {t: [] for t in conll.ENTITY_TYPES}
            n = 5 + idx % 3
            p1 = [0.9 - 0.01 * k for k in range(n)]
            p2 = [0.05] * n
            out.append(
                backends.Generation(
                    json.dumps(ents),
                    list(range(n)),
                    p1 if collect_stats else [],
                    p2 if collect_stats else [],
                    [0.3] * n if collect_stats else [],
                    "stop",
                    0,
                    self.prompt_length(p),
                )
            )
        if cross_check:
            self.last_cross_check = {
                "rows": len(prompts),
                "tokens": 10,
                "max_abs_diff_u": 0.0,
                "max_abs_diff_mean_entropy": 0.0,
                "max_abs_diff_mean_max_prob": 0.0,
            }
        return out

    def close(self) -> None:
        pass


@pytest.fixture()
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    pool = _pool(23)
    backend = FakeBackend(pool)
    monkeypatch.setattr(conll, "load_pool", lambda *a, **k: list(pool))
    monkeypatch.setattr(generate, "make_backend", lambda *a, **k: backend)
    monkeypatch.setattr(latency, "make_backend", lambda *a, **k: backend)
    return backend


def _run(out: Path, *extra: str) -> int:
    return generate.main(["--model", "fake/model", "--out", str(out), "--batch-size", "4", *extra])


def test_generate_writes_one_scored_record_per_sentence(fake: FakeBackend, tmp_path: Path) -> None:
    out = tmp_path / "small.jsonl"
    assert _run(out, "--save-token-stats") == 0
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(recs) == 23 and len({r["id"] for r in recs}) == 23
    for r in recs:
        idx = int(r["id"].split("-")[1])
        assert r["exact_match"] == (1 if idx % 2 == 0 or not any(r["gold"].values()) else 0)
        n = 5 + idx % 3
        margins = [(0.9 - 0.01 * k) - 0.05 for k in range(n)]
        assert r["u"] == pytest.approx(1 - sum(margins) / n)
        assert r["u"] == pytest.approx(ucci.uncertainty_from_margins(margins))
        assert r["n_tokens"] == n and r["entropy"] == pytest.approx(0.3)
        assert r["max_prob"] == pytest.approx(sum(0.9 - 0.01 * k for k in range(n)) / n)
    tokens = [
        json.loads(line) for line in out.with_suffix(".tokens.jsonl").read_text().splitlines()
    ]
    assert len(tokens) == 23 and len(tokens[0]["p1"]) == len(tokens[0]["p2"])
    meta = json.loads(Path(str(out) + ".meta.json").read_text())
    assert meta["n_logged"] == 23 and meta["prompt_sha256"] == conll.prompt_sha256()
    assert meta["sessions"][0]["adapter_cross_check"]["batches"] == 1
    assert meta["dataset_revision"] == conll.DATASET_REVISION


def test_generate_batches_longest_prompts_first(fake: FakeBackend, tmp_path: Path) -> None:
    _run(tmp_path / "a.jsonl")
    assert fake.calls == [4, 4, 4, 4, 4, 3]


def test_generate_resumes_without_redoing_work(fake: FakeBackend, tmp_path: Path) -> None:
    out = tmp_path / "small.jsonl"
    _run(out)
    first = out.read_text()
    fake.calls.clear()
    _run(out)
    assert fake.calls == [] and out.read_text() == first


def test_generate_resume_after_a_crash_matches_a_clean_run(
    fake: FakeBackend, tmp_path: Path
) -> None:
    clean = tmp_path / "clean.jsonl"
    _run(clean)
    crashed = tmp_path / "crashed.jsonl"
    _run(crashed)
    lines = crashed.read_text().splitlines(keepends=True)
    crashed.write_text("".join(lines[:9]) + lines[9][:15])  # torn line mid-record
    _run(crashed)
    key = lambda text: sorted(json.loads(x)["id"] for x in text.splitlines())  # noqa: E731
    assert key(crashed.read_text()) == key(clean.read_text())
    by_id = {json.loads(x)["id"]: json.loads(x) for x in crashed.read_text().splitlines()}
    for x in clean.read_text().splitlines():
        a = json.loads(x)
        b = by_id[a["id"]]
        for k in ("u", "exact_match", "tp", "fp", "fn", "raw_output", "batch_size"):
            assert a[k] == b[k]


def test_generate_refuses_to_resume_under_other_settings(fake: FakeBackend, tmp_path: Path) -> None:
    out = tmp_path / "small.jsonl"
    _run(out)
    with pytest.raises(SystemExit, match="batch_size"):
        generate.main(["--model", "fake/model", "--out", str(out), "--batch-size", "8"])
    assert (
        generate.main(
            ["--model", "fake/model", "--out", str(out), "--batch-size", "8", "--force-resume"]
        )
        == 0
    )


def test_generate_limit_is_a_deterministic_subset(fake: FakeBackend, tmp_path: Path) -> None:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _run(a, "--limit", "7", "--sample-seed", "3")
    _run(b, "--limit", "7", "--sample-seed", "3")
    ids = lambda p: sorted(json.loads(x)["id"] for x in p.read_text().splitlines())  # noqa: E731
    assert ids(a) == ids(b) and len(ids(a)) == 7


def test_generate_argument_validation() -> None:
    with pytest.raises(SystemExit):
        generate.parse_args(["--model", "m", "--out", "o", "--batch-size", "0"])
    with pytest.raises(SystemExit):
        generate.parse_args(["--model", "m", "--out", "o", "--max-new-tokens", "0"])


def test_u_from_generation_empty_and_nonempty() -> None:
    empty = backends.Generation("", [], [], [], [], "stop")
    assert generate.u_from_generation(empty) is None
    g = backends.Generation("x", [1, 2], [0.9, 0.6], [0.1, 0.3], [0.1, 0.2], "stop")
    assert generate.u_from_generation(g) == pytest.approx(1 - (0.8 + 0.3) / 2)
    assert generate.mean_or_none([]) is None


def test_content_length_stops_at_first_stop_token() -> None:
    assert backends._content_length([5, 6, 2, 7, 2], [2, 9]) == 2
    assert backends._content_length([2, 2], [2]) == 0
    assert backends._content_length([5, 6], [2]) == 2


def test_make_backend_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="backend"):
        backends.make_backend("onnx", "m")
    with pytest.raises(ValueError, match="CUDA"):
        backends.make_backend("vllm", "m", device="mps")


def test_latency_times_each_query_and_writes_the_ratio(fake: FakeBackend, tmp_path: Path) -> None:
    out = tmp_path / "latency.json"
    assert (
        latency.main(
            ["--small", "s", "--large", "l", "--n-queries", "5", "--warmup", "2", "--out", str(out)]
        )
        == 0
    )
    res = json.loads(out.read_text())
    assert len(res["small"]["per_query"]) == 5 and len(res["large"]["per_query"]) == 5
    assert res["cost_ratio"] == pytest.approx(res["large"]["mean_ms"] / res["small"]["mean_ms"])
    assert [q["id"] for q in res["small"]["per_query"]] == [
        q["id"] for q in res["large"]["per_query"]
    ]
    assert fake.calls.count(1) == 2 * (5 + 2)  # batch size 1, warm-up included


def test_pick_queries_disjoint_and_validated() -> None:
    pool = _pool(10)
    q = latency.pick_queries(pool, 6, 3, seed=1)
    ids = [s.id for s in q["warmup"] + q["timed"]]
    assert len(ids) == len(set(ids)) == 9
    with pytest.raises(ValueError, match="pool has"):
        latency.pick_queries(pool, 9, 3, seed=1)
    s = latency.summarize([1.0, 2.0, 3.0])
    assert s["mean_ms"] == 2.0 and s["median_ms"] == 2.0 and math.isclose(s["std_ms"], 1.0)
