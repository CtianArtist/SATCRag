"""The repository-hygiene check (scripts/check_repo_hygiene.py)."""

import importlib.util
import shutil

import pytest

from tests.conftest import REPO_ROOT


def load_script():
    spec = importlib.util.spec_from_file_location("check_repo_hygiene", REPO_ROOT / "scripts" / "check_repo_hygiene.py")
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hygiene = load_script()


@pytest.mark.parametrize(
    "path",
    [
        ".venv/lib/python3.14/site.py",
        "data/raw/SATC_all_lines.csv",
        "data/processed/lines.jsonl",
        "data/processed/eval_review_benchmark_proposed.md",
        "data/stray.csv",
        "index/dense/tok512o64-bge-m3/bge-m3-4c49708845c1/vectors.npy",
        "results/runs/20260930T120000Z-dev-bf75352/metrics.json",
        "src/sexandrag/__pycache__/cli.cpython-314.pyc",
        ".pytest_cache/v/cache/nodeids",
        "models/bge-m3/model.safetensors",
        ".env",
        "config/kaggle.json",
        "src/sexandrag.egg-info/PKG-INFO",
        "notes.txt~",
    ],
)
def test_prohibited_paths_are_caught(path):
    assert hygiene.violations([path])


@pytest.mark.parametrize(
    "path",
    [
        "data/meta/episodes.csv",
        "data/processed/chunks/tok512o64-bge-m3.manifest.json",
        "eval/frozen/MANIFEST.json",
        "src/sexandrag/cli.py",
        "tests/fixtures/synthetic/data/raw/SATC_all_lines.csv",
        "results/README.md",
        "results/reports/.gitkeep",
    ],
)
def test_legitimate_project_files_are_allowed(path):
    assert hygiene.violations([path]) == []


def test_a_report_is_published_only_by_a_deliberate_git_add_f():
    report = "results/reports/dev-baseline.md"
    assert hygiene.violations([report]) == []  # tracked: someone ran `git add -f` on purpose
    assert hygiene.violations([report], hygiene.UNPUBLISHED)  # addable by `git add -A`: the ignore rule is gone
    assert hygiene.violations(["results/reports/.gitkeep"], hygiene.UNPUBLISHED) == []


@pytest.mark.skipif(shutil.which("git") is None or not (REPO_ROOT / ".git").exists(), reason="needs a git checkout")
def test_this_checkout_is_clean():
    assert hygiene.check(REPO_ROOT) == []
