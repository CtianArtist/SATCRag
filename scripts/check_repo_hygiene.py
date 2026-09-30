#!/usr/bin/env python3
"""Fail when git would publish something that must stay local.

Checks every tracked file and every untracked file that `git add -A` would pick up against the
prohibited categories (virtual environments, the source corpus and anything derived from it,
embedding caches and model weights, raw run outputs, Python and tool caches, editor/OS files,
secrets and credentials), and confirms that representative local paths are really ignored.

Reports and figures (results/reports/, results/figures/) are git-ignored, so none is published by
accident; `git add -f <file>` is the deliberate act of publishing one, and a tracked report is
allowed. One that git no longer ignores (and `git add -A` would sweep up) is a problem.

Standard library only, so it runs before anything is installed. Exit code 1 lists every problem.
"""

import re
import subprocess
import sys
from pathlib import Path

PROHIBITED = [
    (r"^\.?venv[^/]*/", "virtual environment"),
    (r"^data/raw/", "source corpus (never redistributed)"),
    (r"^data/processed/(?!chunks/[^/]+\.manifest\.json$)", "derived corpus text (rebuilt locally)"),
    (r"^data/(?!meta/|processed/)", "stray file under data/"),
    (r"^index/", "embedding cache"),
    (r"^results/runs/", "raw run output (copy what you publish into results/reports/)"),
    (r"(^|/)__pycache__/|\.py[cod]$", "Python bytecode cache"),
    (r"(^|/)\.(pytest|mypy|ruff)_cache/|(^|/)\.coverage", "test or tool cache"),
    (r"\.(safetensors|bin|onnx|onnx_data|pt|pth|ckpt|npy|npz)$", "model weights or array data"),
    (r"(^|/)\.env(\.[^/]*)?$|(^|/)kaggle\.json$|\.(pem|key)$", "secret or credential"),
    (r"(^|/)[^/]+\.egg-info/|^(build|dist)/", "build output"),
    (r"(^|/)(\.DS_Store|[^/]*\.swp|[^/]*~)$", "editor or OS file"),
    (r"(^|/)\.cache/|\.incomplete$", "download cache"),
]
UNPUBLISHED = [
    (
        r"^results/(reports|figures)/(?!\.gitkeep$)",
        "experiment output that git no longer ignores (publish one deliberately with `git add -f`)",
    ),
]
MUST_BE_IGNORED = [
    ".venv/bin/python",
    "data/raw/SATC_all_lines.csv",
    "data/processed/lines.jsonl",
    "data/processed/eval_review_benchmark.md",
    "data/processed/chunks/tok512o64-bge-m3.jsonl",
    "index/dense/tok512o64-bge-m3/bge-m3-000000000000/vectors.npy",
    "results/runs/20260101T000000Z-dev-abcdef0/metrics.json",
    "results/reports/dev-baseline.md",
    "results/figures/recall-by-chunk-size.png",
    "src/sexandrag/__pycache__/cli.cpython-314.pyc",
    ".env",
]
MAX_TRACKED_BYTES = 2 * 1024 * 1024  # anything larger deserves a deliberate decision


def git(root: Path, *args: str) -> list[str]:
    """Run git in `root` and return its output lines (empty when git is unavailable)."""
    done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    return done.stdout.splitlines() if done.returncode == 0 else []


def violations(paths: list[str], rules: list[tuple[str, str]] = PROHIBITED) -> list[str]:
    """Paths that fall into one of the categories in `rules`, with the category."""
    found = []
    for path in paths:
        for pattern, category in rules:
            if re.search(pattern, path):
                found.append(f"{path}: {category}")
                break
    return found


def unignored(root: Path, paths: list[str]) -> list[str]:
    """Paths from MUST_BE_IGNORED that git would not ignore."""
    ignored = set(git(root, "check-ignore", "--no-index", *paths))
    return [f"{path}: should be git-ignored" for path in paths if path not in ignored]


def oversized(root: Path, paths: list[str]) -> list[str]:
    """Tracked files over the size limit."""
    return [
        f"{path}: {(root / path).stat().st_size / 1e6:.1f} MB tracked file"
        for path in paths
        if (root / path).is_file() and (root / path).stat().st_size > MAX_TRACKED_BYTES
    ]


def check(root: Path) -> list[str]:
    """Every hygiene problem in the checkout at `root`."""
    tracked = git(root, "ls-files")
    addable = git(root, "ls-files", "--others", "--exclude-standard")
    if not tracked and not addable:
        return ["not a git checkout (or git is unavailable)"]
    return (
        violations(tracked + addable)
        + violations(addable, UNPUBLISHED)
        + unignored(root, MUST_BE_IGNORED)
        + oversized(root, tracked)
    )


def main() -> int:
    """Print problems and return 1, or confirm the checkout is clean."""
    root = Path(__file__).resolve().parent.parent
    problems = check(root)
    for problem in problems:
        print(f"HYGIENE: {problem}")
    if problems:
        return 1
    print("repository hygiene: OK (nothing prohibited is tracked or addable; local artifacts are ignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
