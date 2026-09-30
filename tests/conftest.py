"""Test levels and shared fixtures.

tests/unit/         pure logic, no files beyond tmp_path
tests/integration/  the whole pipeline and CLI on a tiny synthetic corpus (tests/fixtures/synthetic)
                    plus checks of the committed frozen benchmark; no corpus, no model, no network
tests/corpus/       invariants over the real SATC corpus; skipped when it is not present locally
tests/model/        the real 2.3 GB BGE-M3 model; opt in with --run-model or RUN_MODEL_TESTS=1
"""

import os
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "synthetic"


def pytest_addoption(parser: pytest.Parser) -> None:
    """--run-model enables the real-model tests."""
    parser.addoption("--run-model", action="store_true", help="run tests that load the real BGE-M3 model")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip model tests unless asked for, and corpus tests when the corpus is not present."""
    run_model = config.getoption("--run-model") or os.environ.get("RUN_MODEL_TESTS") == "1"
    has_corpus = (REPO_ROOT / "data" / "processed" / "lines.jsonl").is_file() and (
        REPO_ROOT / "data" / "raw" / "SATC_all_lines.csv"
    ).is_file()
    for item in items:  # markers, not keywords: keywords also contain parametrize ids such as "model"
        if item.get_closest_marker("model") and not run_model:
            item.add_marker(pytest.mark.skip(reason="real-model test: pass --run-model or set RUN_MODEL_TESTS=1"))
        if item.get_closest_marker("corpus") and not has_corpus:
            item.add_marker(
                pytest.mark.skip(reason="needs the local SATC corpus (sexandrag download && sexandrag parse)")
            )


@pytest.fixture
def repo_root() -> Path:
    """The repository checkout the tests run from."""
    return REPO_ROOT


@pytest.fixture
def synthetic_root(tmp_path: Path) -> Path:
    """A private copy of the synthetic project (corpus, metadata, config, eval items) to run the CLI in."""
    root = tmp_path / "project"
    shutil.copytree(SYNTHETIC, root)
    return root
