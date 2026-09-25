# Development tasks for ucci. CI (.github/workflows/ci.yml) runs the same commands.
#
# Variables can be overridden on the command line, for example
#   make test PYTHON=.venv/bin/python
#   make test39 PY39=/path/to/python3.9
#   make rust CARGO=/path/to/cargo

PYTHON ?= python
PY39 ?= python3.9
CARGO ?= cargo
RUST_MSRV ?= 1.70.0

.PHONY: help install test test-core test-bench doctest test39 lint typecheck snippets docs \
        docs-serve golden golden-check rust rust-msrv bench-smoke build check clean

help:
	@echo "install      editable install with the dev extras"
	@echo "test         unit tests, benchmark tests and doctests"
	@echo "test39       the unit tests on Python 3.9 (PY39=...)"
	@echo "lint         ruff check and ruff format --check"
	@echo "typecheck    mypy --strict on src/ucci"
	@echo "snippets     run the code snippets of README.md and docs/"
	@echo "docs         mkdocs build --strict, then the snippets"
	@echo "golden       regenerate the golden vectors and bless the Rust-written routers"
	@echo "golden-check fail if the golden vectors are out of date"
	@echo "rust         cargo fmt, clippy, test, doc and package for the Rust crate"
	@echo "rust-msrv    cargo test on Rust $(RUST_MSRV)"
	@echo "bench-smoke  re-run the CoNLL-2003 analysis on the committed smoke logs"
	@echo "build        sdist and wheel, then twine check"
	@echo "check        lint, typecheck, test, golden-check, snippets and docs"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test: test-core test-bench doctest

test-core:
	$(PYTHON) -m pytest

test-bench:
	$(PYTHON) -m pytest benchmarks/conll2003/tests

doctest:
	$(PYTHON) -m pytest --doctest-modules src/ucci

# OMP_NUM_THREADS=1 avoids an OpenMP runtime clash between torch and scikit-learn
# builds on some macOS Python 3.9 installations; it does not change any result.
test39:
	PYTHONPATH=src PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 $(PY39) -m pytest -q -p no:cacheprovider

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy

snippets:
	$(PYTHON) docs/_ext/check_snippets.py

docs:
	$(PYTHON) -m mkdocs build --strict
	$(PYTHON) docs/_ext/check_snippets.py

docs-serve:
	$(PYTHON) -m mkdocs serve

golden:
	$(PYTHON) tools/make_golden.py
	cd rust && UCCI_BLESS=1 $(CARGO) test --test golden

golden-check:
	$(PYTHON) tools/make_golden.py --check

rust:
	cd rust && $(CARGO) fmt --check
	cd rust && $(CARGO) clippy --all-targets --all-features -- -D warnings
	cd rust && $(CARGO) clippy --all-targets --no-default-features -- -D warnings
	cd rust && $(CARGO) test --all-features
	cd rust && $(CARGO) test --no-default-features
	cd rust && RUSTDOCFLAGS="-D warnings" $(CARGO) doc --no-deps --all-features
	cd rust && $(CARGO) package --allow-dirty --no-verify

# The lockfile is resolved by the default (newer) cargo with the MSRV-aware
# fallback resolver, then serde is pinned: serde_derive 1.0.229 and later need
# Rust 1.71 without declaring it. The tests then run on Rust $(RUST_MSRV), which
# must be installed (rustup toolchain install $(RUST_MSRV)).
rust-msrv:
	cd rust && CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback $(CARGO) generate-lockfile
	cd rust && $(CARGO) update -p serde --precise 1.0.228
	cd rust && $(CARGO) +$(RUST_MSRV) test --all-features
	cd rust && $(CARGO) +$(RUST_MSRV) test --no-default-features

bench-smoke:
	out=$$(mktemp -d) && \
	$(PYTHON) benchmarks/conll2003/analyze.py \
	    --small benchmarks/conll2003/runs/smoke/small.jsonl \
	    --large benchmarks/conll2003/runs/smoke/large.jsonl \
	    --latency benchmarks/conll2003/runs/smoke/latency.json \
	    --out-dir "$$out" && \
	echo "smoke analysis written to $$out"

build:
	rm -rf dist
	$(PYTHON) -m build
	$(PYTHON) -m twine check --strict dist/*

check: lint typecheck test golden-check docs

clean:
	rm -rf build dist site .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -path ./.venv -prune -o -name __pycache__ -type d -prune -exec rm -rf {} +
