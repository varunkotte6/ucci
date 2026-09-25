# Installation

UCCI needs Python 3.9 or newer and numpy 1.20 or newer. Importing `ucci` imports numpy and
nothing else; the optional dependencies below are imported only by the functions that need
them.

## From PyPI

```bash
pip install ucci-router
```

With extras, for example plotting and the OpenAI SDK:

```bash
pip install "ucci-router[plot,openai]"
```

The development version installs from GitHub:

```bash
pip install "ucci-router @ git+https://github.com/varunkotte6/ucci"
```

Check the installation:

```bash
ucci version
python -c "import ucci; print(ucci.__version__)"
```

## Extras

| Extra | Installs | Needed for |
|---|---|---|
| (none) | numpy | the core, the CLI, the baselines, the online extension, and parsing every serving payload |
| `plot` | matplotlib | `ucci.plotting` |
| `openai` | openai | calling an API with `ucci.integrations.openai.make_chat_fn`; parsing responses needs no SDK |
| `transformers` | torch, transformers (4.38 or newer returns raw logits) | `ucci.integrations.transformers.generate_with_signals` |
| `vllm` | vllm | `ucci.integrations.vllm.greedy_sampling_params` (Linux with a supported GPU) |
| `sklearn` | scikit-learn | the isotonic parity tests |
| `bench` | torch, transformers, datasets, matplotlib | the CoNLL-2003 replication |
| `test`, `lint`, `docs` | pytest, hypothesis; ruff, mypy; mkdocs-material, mkdocstrings | development |
| `dev` | `test`, `lint`, `docs`, build, twine, pre-commit | development |
| `all` | `sklearn`, `plot`, `openai`, `transformers` | everything except `vllm` |

The benchmark's reference run pins exact versions in
[`benchmarks/conll2003/requirements.txt`](https://github.com/varunkotte6/ucci/blob/main/benchmarks/conll2003/requirements.txt).

## From source

```bash
git clone https://github.com/varunkotte6/ucci
cd ucci
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

[Contributing](https://github.com/varunkotte6/ucci/blob/main/CONTRIBUTING.md) describes the
full development workflow: tests on Python 3.9, golden vectors, the Rust crate and the docs.

## Rust

The Rust crate lives in `rust/` and is not on crates.io yet. Depend on it through git:

```toml
[dependencies]
ucci = { git = "https://github.com/varunkotte6/ucci" }
```

It supports Rust 1.70 and newer; see the [Rust guide](guides/rust.md).
