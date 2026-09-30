"""Experiment outputs: one new directory per run, never overwritten.

<results dir>/runs/<UTC timestamp>-<label>-<commit>[-dirty]/
    run.json          id, timestamps, wall time, eval file and hash, held-out flag, git state
    config.json       every setting, chunk-set and index identities, retriever parameters, input hashes
    environment.json  Python, platform and library versions
    metrics.json      aggregates per (chunk size, retriever) cell, comparisons and timing
    queries.jsonl     one row per (cell, question): metrics, target ranks and the stored top-k
                      provenance (chunk ids and row spans only, never chunk text)

Runs are git-ignored; publish one deliberately by copying what you need into results/reports/.
"""

import logging
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sexandrag.jsonl import write_json, write_jsonl

log = logging.getLogger(__name__)


def utc_now() -> datetime:
    """The current time in UTC."""
    return datetime.now(UTC)


def run_id(started: datetime, label: str, git: Mapping[str, Any]) -> str:
    """A readable, sortable run id: timestamp, label, short commit and dirty marker."""
    commit = (git.get("commit") or "nogit")[:7]
    dirty = "-dirty" if git.get("dirty") else ""
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "run"
    return f"{started.strftime('%Y%m%dT%H%M%SZ')}-{safe_label}-{commit}{dirty}"


def create_run_dir(runs_dir: Path, base_id: str) -> Path:
    """Create a fresh run directory; an existing one is never reused or overwritten."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 1000):
        path = runs_dir / (base_id if attempt == 1 else f"{base_id}-{attempt}")
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path
    raise FileExistsError(f"could not create a unique run directory for {base_id} in {runs_dir}")


def write_run(
    run_dir: Path,
    *,
    run: Mapping[str, Any],
    config: Mapping[str, Any],
    environment: Mapping[str, Any],
    metrics: Mapping[str, Any],
    query_rows: Iterable[Mapping[str, Any]],
) -> None:
    """Write every record of one run into its directory."""
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "metrics.json", metrics)
    write_jsonl(run_dir / "queries.jsonl", (dict(row) for row in query_rows))
    write_json(run_dir / "run.json", run)
    log.info("saved run results to %s", run_dir)
