"""The retrieval benchmark: every chunk setting x every retriever over one eval file.

Order of operations, each failing loudly before any expensive work:
  1. resolve the eval file (development set by default; the test set needs --allow-heldout),
  2. load and validate every item (schema errors stop the run here, never during scoring),
  3. refuse held-out data without the explicit opt-in (heldout.py),
  4. verify the frozen benchmark's hashes when running a frozen split,
  5. load the embedding model once and embed every question once, in batches,
  6. score each (chunk size, retriever) cell, then write a new run directory (results.py).
Retrievers only ever receive the question text; targets, quotes and answers stay here.
"""

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from satc_rag.artifacts import ChunkConfig, chunk_config_for_size, file_sha256
from satc_rag.chunk import ChunkSet, load_chunk_set
from satc_rag.config import Settings
from satc_rag.embedders import Embedder
from satc_rag.environment import environment, git_state
from satc_rag.errors import ConfigurationError, HeldOutSetError
from satc_rag.evaluation.heldout import guard_held_out
from satc_rag.evaluation.metrics import average, first_relevant_rank, metric_names, score_ranking, target_first_ranks
from satc_rag.evaluation.results import create_run_dir, run_id, utc_now, write_run
from satc_rag.evaluation.schema import EvalItem, load_items
from satc_rag.evaluation.split import require_intact
from satc_rag.evaluation.validate import by_episode
from satc_rag.index import index_location
from satc_rag.jsonl import read_jsonl
from satc_rag.provenance import quote_status
from satc_rag.retrieve import METHODS, Retriever, SearchResult, build_retrievers
from satc_rag.services import Services

log = logging.getLogger(__name__)
SPLITS = ("dev", "test")
RETRIEVED_KEYS = ("rank", "chunk_id", "season", "episode", "source_row_start", "source_row_end", "score", "components")


@dataclass(frozen=True)
class EvalRequest:
    """What to evaluate: a frozen split (dev by default) or an explicit eval file, and on what."""

    split: str | None = "dev"
    eval_file: Path | None = None
    sizes: tuple[int, ...] | None = None  # None: every configured chunk size
    methods: tuple[str, ...] = METHODS
    allow_heldout: bool = False


@dataclass(frozen=True)
class RunOutcome:
    """Where a run was saved and its aggregate metrics per cell."""

    run_dir: Path
    cells: dict[str, dict[str, Any]]
    held_out: bool


def resolve_eval_file(request: EvalRequest, settings: Settings) -> tuple[Path, str]:
    """(eval file, run label). Asking for the test split without the opt-in fails before anything loads."""
    paths = settings.paths
    if request.eval_file is not None:
        return request.eval_file.expanduser().resolve(), request.eval_file.stem
    if request.split == "dev":
        return paths.dev_file, "dev"
    if request.split == "test":
        if not request.allow_heldout:
            log.error("refused `--split test` without --allow-heldout")
            raise HeldOutSetError(
                "the held-out test set needs an explicit opt-in",
                recovery="run `satc-rag eval --split test --allow-heldout`, and only once the retrieval approach "
                "is final; tune and analyse on the development set (`satc-rag eval`)",
            )
        return paths.test_file, "test"
    raise ConfigurationError(f"unknown split {request.split!r}", expected=", ".join(SPLITS))


def score_query(
    item: EvalItem, results: Sequence[SearchResult], k_values: Sequence[int], all_targets_k: Sequence[int]
) -> dict[str, Any]:
    """Metrics on the top max(k) plus the whole stored ranking (provenance and relevance) for one query."""
    ranked = [result.provenance() for result in results]
    retrieved = [
        {key: chunk[key] for key in RETRIEVED_KEYS} | {"relevant": item.is_relevant(chunk)} for chunk in ranked
    ]
    return {
        "id": item.id,
        "question_type": item.question_type,
        "question": item.question,
        **score_ranking(ranked[: max(k_values)], item, k_values, all_targets_k),
        "first_relevant_rank_in_depth": first_relevant_rank(ranked, item),
        "target_first_ranks": target_first_ranks(ranked, item),
        "retrieved": retrieved,
    }


def run_cell(
    retriever: Retriever, items: Sequence[EvalItem], k_values: Sequence[int], depth: int, all_targets_k: Sequence[int]
) -> tuple[list[dict[str, Any]], float]:
    """Run every question through one retriever, keeping `depth` results; return rows and search seconds."""
    rows, started = [], time.perf_counter()
    for item in items:
        results = retriever.search(item.question, max(depth, *k_values))
        rows.append(score_query(item, results, k_values, all_targets_k))
    return rows, time.perf_counter() - started


def summarize(
    rows: Sequence[Mapping[str, Any]], k_values: Sequence[int], all_targets_k: Sequence[int]
) -> dict[str, Any]:
    """Averages over all queries and per question type."""
    names = metric_names(k_values, all_targets_k)
    types = sorted({row["question_type"] for row in rows if row["question_type"]})
    return {
        "all": average(rows, names),
        "by_type": {t: average([r for r in rows if r["question_type"] == t], names) for t in types},
    }


def hits(cells: Mapping[str, Any], size: int, method: str, k: int) -> dict[str, bool] | None:
    """{query id: hit@k} for one cell, or None when the cell was not run."""
    cell = cells.get(f"{size}/{method}")
    return {row["id"]: row[f"hit@{k}"] == 1.0 for row in cell["queries"]} if cell else None


def outcome_buckets(
    bm25: Mapping[str, bool], dense: Mapping[str, bool], hybrid: Mapping[str, bool]
) -> dict[str, list[str]]:
    """Sort query ids by which retrievers reached a relevant chunk."""
    ids = list(bm25)
    return {
        "both_hit": [i for i in ids if bm25[i] and dense[i]],
        "bm25_only": [i for i in ids if bm25[i] and not dense[i]],
        "dense_only": [i for i in ids if dense[i] and not bm25[i]],
        "both_missed": [i for i in ids if not bm25[i] and not dense[i]],
        "hybrid_recovery": [i for i in ids if hybrid[i] and not bm25[i] and not dense[i]],
        "hybrid_regression": [i for i in ids if not hybrid[i] and (bm25[i] or dense[i])],
    }


def rank_changes(cells: Mapping[str, Any], sizes: Sequence[int], method: str) -> dict[str, dict[int, int | None]]:
    """{query id: {chunk size: first relevant rank within the stored depth}} for one retriever."""
    per_size = {size: cells[f"{size}/{method}"]["queries"] for size in sizes if f"{size}/{method}" in cells}
    if len(per_size) < 2:
        return {}
    ids = [row["id"] for row in next(iter(per_size.values()))]
    ranks = {size: {row["id"]: row["first_relevant_rank_in_depth"] for row in rows} for size, rows in per_size.items()}
    return {i: {size: ranks[size][i] for size in per_size} for i in ids}


def multi_target_partial(cells: Mapping[str, Any], k: int) -> dict[str, list[str]]:
    """Per cell: queries that reached some but not all of their required evidence in the top k."""
    return {
        name: [row["id"] for row in cell["queries"] if row[f"hit@{k}"] == 1.0 and row[f"recall@{k}"] < 1.0]
        for name, cell in cells.items()
    }


def compare_outcomes(cells: Mapping[str, Any], sizes: Sequence[int], k: int) -> dict[str, Any]:
    """Where retrievers or chunk sizes disagree at Hit@k, and how ranks move (for failure analysis)."""
    by_size = {}
    for size in sizes:
        bm25, dense, hybrid = (hits(cells, size, m, k) for m in METHODS)
        if bm25 is not None and dense is not None and hybrid is not None:
            by_size[size] = outcome_buckets(bm25, dense, hybrid)
    size_sensitive = {}
    for method in METHODS:
        per_size = {size: h for size in sizes if (h := hits(cells, size, method, k)) is not None}
        if len(per_size) > 1:
            ids = list(next(iter(per_size.values())))
            size_sensitive[method] = {
                i: {size: h[i] for size, h in per_size.items()}
                for i in ids
                if len({h[i] for h in per_size.values()}) > 1
            }
    return {
        "k": k,
        "by_size": by_size,
        "size_sensitive": size_sensitive,
        "rank_changes": {m: rank_changes(cells, sizes, m) for m in METHODS},
        "multi_target_partial": multi_target_partial(cells, k),
    }


def quote_warnings(items: Sequence[EvalItem], lines_path: Path) -> list[str]:
    """Targets whose expected_quote is no longer inside their row span (parser drift)."""
    episodes = by_episode(list(read_jsonl(lines_path)))
    warnings = []
    for item in items:
        for span in item.targets:
            status = quote_status(episodes.get((span.season, span.episode), []), span)
            if status in ("moved", "missing"):
                warnings.append(f"{item.id}: expected_quote {status} ({span.label})")
    return warnings


def chunk_set_record(chunk_set: ChunkSet, index_identity: Mapping[str, Any] | None, index_dir: Path) -> dict[str, Any]:
    """What identifies the chunk set (and dense index) used for one size."""
    record = {
        key: chunk_set.manifest[key]
        for key in (
            "config_id",
            "size",
            "overlap",
            "tokenizer",
            "tokenizer_revision",
            "n_chunks",
            "lines_sha256",
            "chunks_sha256",
        )
    }
    if index_identity is not None:
        key, directory = index_location(chunk_set, dict(index_identity), index_dir)
        record["dense_index"] = {"key": key, "path": directory.name}
    return record


def format_summary(cells: Mapping[str, Any], k_values: Sequence[int], all_targets_k: Sequence[int]) -> list[str]:
    """One printable row per (chunk size, retriever) cell."""
    names = metric_names(k_values, all_targets_k)
    header = f"{'cell':14s}" + "".join(f"{n.replace('all_targets_hit', 'all_tgt'):>11s}" for n in names)
    lines = [header + f"{'n':>5s}{'s/query':>9s}"]
    for name, cell in cells.items():
        agg = cell["aggregate"]["all"]
        values = "".join(f"{'-' if agg[n] is None else format(agg[n], '.3f'):>11s}" for n in names)
        lines.append(f"{name:14s}{values}{agg['n']:5d}{cell['seconds_per_query']:9.3f}")
    return lines


def chunk_configs_for(request: EvalRequest, settings: Settings, services: Services) -> list[ChunkConfig]:
    """The configured chunk settings to evaluate (all of them unless sizes were given)."""
    tokenizer, revision = services.tokenizer_id(settings)
    sizes = request.sizes or tuple(size for size, _ in settings.chunking.configs)
    return [chunk_config_for_size(size, settings.chunking, tokenizer, revision) for size in sizes]


def run_evaluation(settings: Settings, request: EvalRequest, services: Services) -> RunOutcome:
    """Run the benchmark matrix and save a new run directory."""
    started, clock = utc_now(), time.perf_counter()
    evaluation, paths = settings.evaluation, settings.paths
    eval_file, label = resolve_eval_file(request, settings)
    items = load_items(eval_file)
    held_out = guard_held_out(eval_file, items, paths, request.allow_heldout)
    if eval_file.parent.resolve() == paths.frozen_dir.resolve():
        require_intact(paths.frozen_dir)
    unknown = sorted(set(request.methods) - set(METHODS))
    if unknown:
        raise ConfigurationError(f"unknown retrieval method(s): {', '.join(unknown)}", expected=", ".join(METHODS))
    chunk_configs = chunk_configs_for(request, settings, services)
    log.info(
        "evaluating %d item(s) from %s on %s x %s",
        len(items),
        eval_file.name,
        ", ".join(c.config_id for c in chunk_configs),
        ", ".join(request.methods),
    )
    for warning in quote_warnings(items, paths.lines_jsonl):
        log.warning("%s", warning)
    embedder: Embedder | None = None
    timing: dict[str, Any] = {"query_embedding_seconds_per_query": None}
    if {"dense", "hybrid"} & set(request.methods):
        embedder = services.embedder(settings)
        t0 = time.perf_counter()
        embedder.embed_queries([item.question for item in items])
        timing["query_embedding_seconds_per_query"] = (time.perf_counter() - t0) / len(items)
    k_values, all_k = evaluation.k_values, evaluation.all_targets_k
    cells: dict[str, dict[str, Any]] = {}
    params: dict[str, Any] = {}
    chunk_sets = []
    for chunk_config in chunk_configs:
        chunk_set = load_chunk_set(chunk_config, paths.chunks_dir, paths.lines_jsonl)
        identity = services.embedder_identity(settings) if embedder is not None else None
        chunk_sets.append(chunk_set_record(chunk_set, identity, paths.index_dir))
        retrievers = build_retrievers(chunk_set, request.methods, embedder, paths.index_dir, settings.retrieval)
        for method, retriever in retrievers.items():
            params[method] = getattr(retriever, "params", {})
            rows, seconds = run_cell(retriever, items, k_values, evaluation.depth, all_k)
            cells[f"{chunk_config.size}/{method}"] = {
                "chunk_config": chunk_config.config_id,
                "retriever": method,
                "seconds_per_query": seconds / len(items),
                "aggregate": summarize(rows, k_values, all_k),
                "queries": rows,
            }
            log.info("ran %s/%s: %d queries in %.1fs", chunk_config.size, method, len(rows), seconds)
    sizes = [c.size for c in chunk_configs]
    git = git_state(paths.root)
    run_dir = create_run_dir(paths.runs_dir, run_id(started, label, git))
    finished = utc_now()
    write_run(
        run_dir,
        run={
            "run_id": run_dir.name,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "wall_seconds": round(time.perf_counter() - clock, 2),
            "eval": {
                "file": str(eval_file),
                "sha256": file_sha256(eval_file),
                "split": request.split if request.eval_file is None else None,
                "held_out": held_out,
                "n_items": len(items),
            },
            "git": git,
        },
        config={
            "settings": settings.as_dict(),
            "retrievers": list(request.methods),
            "retriever_params": params,
            "embedder": services.embedder_identity(settings) if embedder is not None else None,
            "chunk_sets": chunk_sets,
            "hashes": {
                "eval_file": file_sha256(eval_file),
                "raw_csv": file_sha256(paths.raw_csv) if paths.raw_csv.is_file() else None,
                "lines_jsonl": file_sha256(paths.lines_jsonl),
                "frozen_manifest": file_sha256(paths.manifest_file) if paths.manifest_file.is_file() else None,
            },
        },
        environment=environment(),
        metrics={
            "cells": {name: {k: v for k, v in cell.items() if k != "queries"} for name, cell in cells.items()},
            "comparisons": compare_outcomes(cells, sizes, evaluation.compare_k),
            "timing": timing,
        },
        query_rows=({"cell": name, **row} for name, cell in cells.items() for row in cell["queries"]),
    )
    summaries = {name: {k: v for k, v in cell.items() if k != "queries"} for name, cell in cells.items()}
    return RunOutcome(run_dir, summaries, held_out)
