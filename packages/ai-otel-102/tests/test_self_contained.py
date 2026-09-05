"""Self-containment is the point of this package, so it gets a test.

If someone later adds a workspace sibling to the dependency list, this fails
and the README stops lying.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PACKAGE_ROOT / "src" / "ai_otel_102"
SIBLING_MODULES = {"ai_python_101", "ai_otel_101"}


def declared_dependencies() -> list[str]:
    with (PACKAGE_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["dependencies"]


def test_no_intra_repo_dependencies():
    packages = {dep.split()[0].split(">")[0].split("=")[0] for dep in declared_dependencies()}

    assert packages == {"openai", "opentelemetry-api", "opentelemetry-sdk"}


def test_no_uv_workspace_sources():
    with (PACKAGE_ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)

    assert "sources" not in config.get("tool", {}).get("uv", {})


def test_source_never_imports_a_sibling_example():
    offenders = [
        path.name
        for path in SOURCE_DIR.glob("*.py")
        if any(module in path.read_text() for module in SIBLING_MODULES)
    ]

    assert offenders == []


def test_the_example_lives_in_one_module():
    # __init__ re-exports and __main__ runs it; the substance is one file you
    # can copy out.
    modules = {path.name for path in SOURCE_DIR.glob("*.py")}

    assert modules == {"__init__.py", "__main__.py", "observe.py"}


def test_runs_on_the_supported_python():
    assert sys.version_info >= (3, 10)
