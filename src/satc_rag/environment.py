"""What a run ran on: library versions, platform, and the git state of the checkout."""

import importlib.metadata
import platform
import subprocess
from pathlib import Path
from typing import Any

import satc_rag

VERSIONED_PACKAGES = (
    "torch",
    "sentence-transformers",
    "transformers",
    "tokenizers",
    "huggingface_hub",
    "numpy",
    "rank-bm25",
)


def library_versions() -> dict[str, str | None]:
    """Versions of the libraries that can change embeddings or rankings."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in VERSIONED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def environment() -> dict[str, Any]:
    """Library versions plus platform facts for a run record."""
    return {
        "satc_rag": satc_rag.__version__,
        "versions": library_versions(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
    }


def git_state(root: Path) -> dict[str, Any]:
    """Current commit and whether the working tree has uncommitted changes (None outside a git checkout)."""

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)  # noqa: S603, S607
        except OSError:
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "commit": commit if commit and len(commit) == 40 else None,
        "dirty": None if status is None else bool(status),
    }
