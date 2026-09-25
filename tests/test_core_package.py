"""Tests for the package surface: exports, lazy submodules, version, typing marker."""

from __future__ import annotations

import importlib
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

import ucci
import ucci.io as ucci_io

#: Public names the build spec requires to stay importable from ``ucci``.
SPEC_NAMES = [
    "UCCIRouter",
    "IsotonicCalibrator",
    "pav",
    "token_margin_uncertainty",
    "margins_from_top2",
    "uncertainty_from_margins",
    "top2_from_logprobs",
    "from_openai_logprobs",
    "from_vllm_logprobs",
    "select_threshold",
    "evaluate",
    "escalate",
    "policy_cost",
    "policy_accuracy",
    "ThresholdChoice",
    "DEFAULT_GRID",
    "ece",
    "reliability_table",
    "bootstrap_ci",
]


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_spec_names_importable(name):
    assert hasattr(ucci, name)
    assert name in ucci.__all__


def test_all_entries_exist():
    for name in ucci.__all__:
        assert hasattr(ucci, name), name
    assert len(set(ucci.__all__)) == len(ucci.__all__)


def test_version_is_a_string():
    assert isinstance(ucci.__version__, str) and ucci.__version__


def test_version_fallback(monkeypatch):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    monkeypatch.setattr(metadata, "packages_distributions", lambda: {}, raising=False)
    assert ucci_io._package_version() == "0.1.0"


def test_version_from_renamed_distribution(monkeypatch):
    def version(name):
        if name == "ucci-router-dist":
            return "9.9.9"
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", version)
    monkeypatch.setattr(
        metadata,
        "packages_distributions",
        lambda: {"ucci": ["ucci-router-dist"]},
        raising=False,
    )
    assert ucci_io._package_version() == "9.9.9"


def test_import_does_not_pull_optional_dependencies():
    code = (
        "import sys, ucci\n"
        "heavy = ['matplotlib', 'torch', 'transformers', 'vllm', 'openai', 'sklearn', "
        "'scipy', 'pandas']\n"
        "print(','.join(m for m in heavy if m in sys.modules))\n"
        "lazy = ['ucci.baselines', 'ucci.integrations', 'ucci.online', "
        "'ucci.plotting', 'ucci.cli']\n"
        "print(','.join(m for m in lazy if m in sys.modules))\n"
    )
    src = str(Path(ucci.__file__).resolve().parents[1])
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONPATH": src, "PATH": ""},
    ).stdout.splitlines()
    assert out == ["", ""]


def test_lazy_submodule_access(monkeypatch):
    existing = [
        name
        for name in ucci._LAZY_SUBMODULES
        if (Path(ucci.__file__).parent / f"{name}.py").exists()
        or (Path(ucci.__file__).parent / name / "__init__.py").exists()
    ]
    for name in existing:
        module = getattr(ucci, name)
        assert module.__name__ == f"ucci.{name}"
    assert set(ucci._LAZY_SUBMODULES) <= set(dir(ucci))


def test_missing_submodule_is_attribute_error(monkeypatch):
    monkeypatch.setattr(ucci, "_LAZY_SUBMODULES", (*ucci._LAZY_SUBMODULES, "not_there"))
    with pytest.raises(AttributeError, match="submodule not installed"):
        _ = ucci.not_there
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = ucci.nope


def test_missing_optional_dependency_propagates(monkeypatch):
    def fake_import(name, package=None):
        raise ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.delitem(ucci.__dict__, "plotting", raising=False)
    with pytest.raises(ModuleNotFoundError, match="matplotlib"):
        _ = ucci.plotting


def test_py_typed_marker_present():
    assert (Path(ucci.__file__).parent / "py.typed").exists()
