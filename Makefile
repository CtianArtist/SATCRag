.DEFAULT_GOAL := help
PYTHON ?= python
TORCH_INDEX := https://download.pytorch.org/whl/cpu

.PHONY: help install install-download check format lint typecheck test test-model coverage verify hygiene

help:  ## list the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-17s %s\n", $$1, $$2}'

install:  ## CPU-only torch, then the package with dev tools at the tested versions
	$(PYTHON) -m pip install torch==2.14.0 --index-url $(TORCH_INDEX)
	$(PYTHON) -m pip install -e ".[dev]" -c constraints.txt

install-download:  ## add kagglehub for `sexandrag download`
	$(PYTHON) -m pip install -e ".[download]" -c constraints.txt

check:  ## the full production gate (format, lint, types, tests, frozen hashes, hygiene)
	./scripts/check.sh

format:  ## apply ruff formatting
	ruff format .

lint:  ## ruff lint
	ruff check .

typecheck:  ## mypy in strict mode
	mypy

test:  ## unit and integration tests (no corpus or model needed)
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -p no:cacheprovider

test-model:  ## also run the tests that load the real 2.3 GB model
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -p no:cacheprovider --run-model

coverage:  ## tests with a line and branch coverage report
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -p no:cacheprovider --cov --cov-report=term-missing

verify:  ## check every local artifact (corpus, chunks, indexes, model, frozen benchmark)
	sexandrag verify

hygiene:  ## fail if git would publish anything that must stay local
	$(PYTHON) scripts/check_repo_hygiene.py
