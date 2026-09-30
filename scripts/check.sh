#!/usr/bin/env bash
# The production-readiness gate: formatting, lint, types, tests, frozen-benchmark hashes, repository hygiene.
# Needs no corpus, no model and no network; it never runs a retrieval benchmark (and never the held-out test set).
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n== %s\n' "$1"; }

step "ruff format --check"
ruff format --check .
step "ruff check"
ruff check .
step "mypy (strict)"
mypy
step "pytest (unit + integration; corpus and real-model tests skip without local data)"
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider
step "frozen benchmark: hashes and split"
sexandrag --quiet verify --only benchmark
step "repository hygiene"
python scripts/check_repo_hygiene.py

printf '\nAll checks passed.\n'
