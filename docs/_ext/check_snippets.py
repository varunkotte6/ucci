"""Run the code snippets of README.md and the documentation, so the docs stay true.

Usage, from the repository root::

    python docs/_ext/check_snippets.py                 # README.md and docs/**/*.md
    python docs/_ext/check_snippets.py docs/quickstart.md

Rules:

* Every fenced ``python`` block runs. The blocks of one Markdown file run in order
  in a single namespace, in a fresh temporary directory, so a page reads as one
  script.
* A block directly preceded by ``<!-- snippet: skip -->`` is not run (for code
  that needs a GPU, a running server or network access). A block preceded by
  ``<!-- snippet: requires mod1 mod2 -->`` runs only when those modules import.
* A fenced ``bash`` block runs only when preceded by ``<!-- snippet: run -->``,
  with ``bash -euo pipefail`` in the same temporary directory and the current
  interpreter's ``bin`` directory first on ``PATH`` (so ``ucci`` and ``python``
  are this environment's).
* Before a file runs, the temporary directory gets ``traffic.jsonl``: 10,000
  simulated traffic records in the format ``ucci fit`` reads.
* In files that import ``openai``, the module is replaced by a stub whose
  ``chat.completions.create`` checks the request (greedy, ``logprobs=True``,
  ``top_logprobs >= 2``) and returns a recorded Chat Completions payload from
  ``tests/fixtures/integrations``, so the OpenAI snippets exercise the real
  parsing code without network access.

Exit status 0 when every snippet passes, 1 otherwise.
"""

from __future__ import annotations

import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "integrations"
_FENCE = re.compile(r"^(?P<indent>[ \t]*)```(?P<lang>[A-Za-z0-9_+-]*)[^\n]*$")
_MARKER = re.compile(r"^\s*<!--\s*snippet:\s*(?P<body>.*?)\s*-->\s*$")


@dataclass
class Block:
    path: Path
    line: int
    lang: str
    code: str
    marker: str


def extract(path: Path) -> list[Block]:
    """Fenced code blocks of a Markdown file, with the marker comment above each."""
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        m = _FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        indent, lang = m.group("indent"), m.group("lang").lower()
        start = i
        body: list[str] = []
        i += 1
        while i < len(lines) and lines[i].strip() != "```":
            line = lines[i]
            body.append(line[len(indent) :] if line.startswith(indent) else line)
            i += 1
        marker = ""
        j = start - 1
        while j >= 0 and not lines[j].strip():
            j -= 1
        if j >= 0:
            mm = _MARKER.match(lines[j])
            if mm:
                marker = mm.group("body")
        blocks.append(Block(path, start + 1, lang, "\n".join(body) + "\n", marker))
        i += 1
    return blocks


def _write_traffic(directory: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    n = 10_000
    u = rng.beta(2.0, 5.0, n)
    small = (rng.random(n) >= 1.0 / (1.0 + np.exp(-10.0 * (u - 0.45)))).astype(int)
    large = (rng.random(n) < 0.95).astype(int)
    lat_s = rng.normal(47.0, 3.0, n).clip(30.0)
    lat_l = rng.normal(142.0, 8.0, n).clip(90.0)
    with open(directory / "traffic.jsonl", "w", encoding="utf-8") as fh:
        for i in range(n):
            rec = {
                "id": f"q{i}",
                "u": float(u[i]),
                "small_correct": int(small[i]),
                "large_correct": int(large[i]),
                "latency_small_ms": round(float(lat_s[i]), 2),
                "latency_large_ms": round(float(lat_l[i]), 2),
            }
            fh.write(json.dumps(rec) + "\n")


def _fake_openai() -> types.ModuleType:
    payload = json.loads((FIXTURES / "openai_chat_completion.json").read_text("utf-8"))

    def create(**kwargs: Any) -> Any:
        if kwargs.get("logprobs"):
            assert kwargs.get("temperature") == 0, "request greedy decoding (temperature=0)"
            assert int(kwargs.get("top_logprobs", 0)) >= 2, "request top_logprobs >= 2"
        return json.loads(json.dumps(payload))

    class OpenAI:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    class AsyncOpenAI:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            async def acreate(**kwargs: Any) -> Any:
                return create(**kwargs)

            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=acreate))

    mod = types.ModuleType("openai")
    mod.__spec__ = importlib.machinery.ModuleSpec("openai", None)
    mod.OpenAI = OpenAI  # type: ignore[attr-defined]
    mod.AsyncOpenAI = AsyncOpenAI  # type: ignore[attr-defined]
    return mod


def _missing(modules: list[str]) -> list[str]:
    return [m for m in modules if importlib.util.find_spec(m) is None]


def run_file(path: Path) -> tuple[int, int, int]:
    """Run the snippets of one file; returns (passed, skipped, failed)."""
    blocks = [b for b in extract(path) if b.lang in ("python", "py", "bash", "sh")]
    passed = skipped = failed = 0
    if not blocks:
        return passed, skipped, failed
    namespace: dict[str, Any] = {"__name__": "__snippet__"}
    old_cwd = os.getcwd()
    saved_openai = sys.modules.get("openai")
    with tempfile.TemporaryDirectory(prefix="ucci-snippets-") as tmp:
        os.chdir(tmp)
        uses_openai = any(
            b.lang in ("python", "py") and re.search(r"^\s*(from|import) openai\b", b.code, re.M)
            for b in blocks
        )
        if uses_openai:
            sys.modules["openai"] = _fake_openai()
        try:
            _write_traffic(Path(tmp))
            for b in blocks:
                where = f"{b.path.relative_to(ROOT)}:{b.line}"
                words = b.marker.split()
                if words[:1] == ["skip"]:
                    skipped += 1
                    print(f"SKIP  {where} ({b.marker})")
                    continue
                if words[:1] == ["requires"]:
                    missing = _missing(words[1:])
                    if missing:
                        skipped += 1
                        print(f"SKIP  {where} (missing: {', '.join(missing)})")
                        continue
                if b.lang in ("bash", "sh"):
                    if words[:1] != ["run"]:
                        continue
                    env = dict(os.environ)
                    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
                    res = subprocess.run(
                        ["bash", "-euo", "pipefail", "-c", b.code],
                        capture_output=True,
                        text=True,
                        env=env,
                        check=False,
                    )
                    if res.returncode != 0:
                        failed += 1
                        print(f"FAIL  {where} (exit {res.returncode})")
                        print(res.stdout[-2000:] + res.stderr[-2000:])
                    else:
                        passed += 1
                        print(f"ok    {where}")
                    continue
                out = io.StringIO()
                try:
                    with contextlib.redirect_stdout(out):
                        exec(compile(b.code, where, "exec"), namespace)
                except Exception:
                    failed += 1
                    print(f"FAIL  {where}")
                    print(out.getvalue()[-2000:])
                    traceback.print_exc()
                else:
                    passed += 1
                    print(f"ok    {where}")
        finally:
            os.chdir(old_cwd)
            if uses_openai:
                if saved_openai is None:
                    sys.modules.pop("openai", None)
                else:
                    sys.modules["openai"] = saved_openai
    return passed, skipped, failed


def main(argv: list[str]) -> int:
    targets = [Path(a).resolve() for a in argv] or [ROOT / "README.md", ROOT / "docs"]
    files: list[Path] = []
    for t in targets:
        files.extend(sorted(t.rglob("*.md")) if t.is_dir() else [t])
    totals = [0, 0, 0]
    for f in files:
        for k, v in enumerate(run_file(f)):
            totals[k] += v
    print(f"\nsnippets: {totals[0]} passed, {totals[1]} skipped, {totals[2]} failed")
    return 1 if totals[2] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
