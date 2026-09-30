"""Retrieval benchmark: every chunk setting x every retriever over one eval file.

Run: python -m eval.run_eval [--eval-file eval/eval.json] [--sizes 256 512 1024] [--retrievers bm25 dense hybrid]
Writes eval/results/<UTC timestamp>.json with the run configuration, aggregates and per-query
results (top-k provenance with relevance flags). Retrievers only ever receive the question
text; targets, quotes and answers stay in this script.
"""
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.artifacts import chunk_config_for_size, file_sha256
from src.chunk import load_chunk_set
from src.embedders import SentenceTransformerEmbedder
from src.index import library_versions
from src.jsonl import read_jsonl
from src.metrics import average, metric_names, score_ranking
from src.provenance import is_relevant, item_targets, validate_target
from src.retrieve import build_retrievers


def load_items(path: Path) -> list[dict]:
    """Read eval items and check each has an id, a question and well-formed targets."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [item.get("id") for item in items]
    if None in ids or len(set(ids)) != len(ids):
        raise ValueError("every eval item needs a unique 'id'")
    for item in items:
        if not item.get("question"):
            raise ValueError(f"{item['id']}: missing 'question'")
        for target in item_targets(item):
            if not isinstance(target.get("season"), int) or not isinstance(target.get("episode"), int):
                raise ValueError(f"{item['id']}: each target needs integer 'season' and 'episode'")
    return items


def target_warnings(items: list[dict]) -> list[str]:
    """Targets whose expected_quote is no longer inside their row span (parser drift)."""
    lines = list(read_jsonl(config.LINES_JSONL))
    warnings = []
    for item in items:
        for target in item_targets(item):
            status = validate_target(lines, target)["status"]
            if status in ("moved", "missing"):
                warnings.append(f"{item['id']}: expected_quote {status} "
                                f"(S{target['season']}E{target['episode']} rows "
                                f"{target.get('source_row_start')}-{target.get('source_row_end')})")
    return warnings


def git_state() -> dict:
    """Current commit and whether the working tree has uncommitted changes (None outside git)."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", "HEAD")
    return {"commit": commit if len(commit) == 40 else None, "dirty": bool(git("status", "--porcelain"))}


def score_query(item: dict, results: list, k_values: tuple[int, ...]) -> dict:
    """Metrics plus the retrieved list (provenance and relevance) for one query."""
    ranked = [result.provenance() for result in results]
    retrieved = [{key: chunk[key] for key in ("rank", "chunk_id", "season", "episode", "source_row_start",
                                              "source_row_end", "score", "components")}
                 | {"relevant": is_relevant(chunk, item)} for chunk in ranked]
    return {"id": item["id"], "type": item.get("type"), "question": item["question"],
            **score_ranking(ranked, item, k_values), "retrieved": retrieved}


def embed_queries(embedder, items: list[dict]) -> float:
    """Embed every question once up front (the embedder caches them); return seconds per query.

    Without this, whichever chunk size runs first would pay for query embedding and the
    others would not, making per-cell search times incomparable.
    """
    started = time.perf_counter()
    for item in items:
        embedder.embed_query(item["question"])
    return (time.perf_counter() - started) / len(items)


def run_cell(retriever, items: list[dict], k_values: tuple[int, ...]) -> tuple[list[dict], float]:
    """Run every question through one retriever; return per-query rows and seconds spent searching."""
    rows, started = [], time.perf_counter()
    for item in items:
        rows.append(score_query(item, retriever.search(item["question"], max(k_values)), k_values))
    return rows, time.perf_counter() - started


def summarize(rows: list[dict], k_values: tuple[int, ...]) -> dict:
    """Averages over all queries and per question type."""
    names = metric_names(k_values)
    types = sorted({row["type"] for row in rows if row["type"]})
    return {"all": average(rows, names),
            "by_type": {t: average([r for r in rows if r["type"] == t], names) for t in types}}


def hits(cells: dict, size: int, method: str, k: int) -> dict[str, bool] | None:
    """{query id: hit@k} for one cell, or None when the cell was not run."""
    cell = cells.get(f"{size}/{method}")
    return {row["id"]: row[f"hit@{k}"] == 1.0 for row in cell["queries"]} if cell else None


def compare_outcomes(cells: dict, sizes: list[int], k: int) -> dict:
    """Queries where retrievers or chunk sizes disagree at Hit@k (material for failure analysis)."""
    by_size = {}
    for size in sizes:
        bm25, dense, hybrid = (hits(cells, size, m, k) for m in ("bm25", "dense", "hybrid"))
        if not (bm25 and dense and hybrid):
            continue
        ids = list(bm25)
        by_size[size] = {
            "bm25_only": [i for i in ids if bm25[i] and not dense[i]],
            "dense_only": [i for i in ids if dense[i] and not bm25[i]],
            "hybrid_recovered": [i for i in ids if hybrid[i] and not bm25[i] and not dense[i]],
            "hybrid_lost": [i for i in ids if not hybrid[i] and (bm25[i] or dense[i])],
            "all_missed": [i for i in ids if not (bm25[i] or dense[i] or hybrid[i])],
        }
    size_sensitive = {}
    for method in ("bm25", "dense", "hybrid"):
        per_size = {size: hits(cells, size, method, k) for size in sizes}
        per_size = {size: h for size, h in per_size.items() if h}
        if len(per_size) > 1:
            ids = list(next(iter(per_size.values())))
            size_sensitive[method] = {i: {size: h[i] for size, h in per_size.items()}
                                      for i in ids if len({h[i] for h in per_size.values()}) > 1}
    return {"k": k, "by_size": by_size, "size_sensitive": size_sensitive}


def chunk_set_record(size: int) -> dict:
    """What identifies the chunk set used for one size."""
    _, manifest = load_chunk_set(chunk_config_for_size(size))
    return {key: manifest[key] for key in ("config_id", "size", "overlap", "tokenizer", "tokenizer_revision",
                                           "n_chunks", "lines_sha256", "chunks_sha256")}


def run_configuration(args, items_path: Path, retriever_params: dict, embedder) -> dict:
    """Everything needed to reproduce or compare this run."""
    return {
        "eval_file": str(items_path.relative_to(config.ROOT) if items_path.is_relative_to(config.ROOT) else items_path),
        "eval_sha256": file_sha256(items_path),
        "raw_csv_sha256": file_sha256(config.RAW_CSV),
        "lines_sha256": file_sha256(config.LINES_JSONL),
        "chunk_sets": [chunk_set_record(size) for size in args.sizes],
        "retrievers": args.retrievers,
        "retriever_params": retriever_params,
        "embedder": embedder.identity() if embedder else None,
        "k_values": list(args.k_values),
        "top_k": max(args.k_values),
        "compare_k": args.compare_k,
        "git": git_state(),
        "versions": library_versions(),
    }


def print_summary(cells: dict, k_values: tuple[int, ...]) -> None:
    """Print one row per (chunk size, retriever) cell."""
    names = metric_names(k_values)
    print(f"\n{'cell':14s}" + "".join(f"{n:>10s}" for n in names) + f"{'n':>5s}{'sec/query':>11s}")
    for name, cell in cells.items():
        agg = cell["aggregate"]["all"]
        print(f"{name:14s}" + "".join(f"{agg[n]:10.3f}" for n in names)
              + f"{agg['n']:5d}{cell['seconds_per_query']:11.3f}")


def main(argv: list[str] | None = None) -> None:
    """Run the benchmark matrix and save the results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-file", type=Path, default=config.EVAL_FILE)
    parser.add_argument("--sizes", type=int, nargs="+", default=[size for size, _ in config.CHUNK_CONFIGS])
    parser.add_argument("--retrievers", nargs="+", default=["bm25", "dense", "hybrid"],
                        choices=["bm25", "dense", "hybrid"])
    parser.add_argument("--k-values", type=int, nargs="+", default=list(config.EVAL_K_VALUES))
    parser.add_argument("--compare-k", type=int, default=config.EVAL_COMPARE_K)
    parser.add_argument("--out-dir", type=Path, default=config.EVAL_RESULTS_DIR)
    args = parser.parse_args(argv)
    k_values = tuple(sorted(args.k_values))
    items_path = args.eval_file.resolve()
    items = load_items(items_path)
    for warning in target_warnings(items):
        print("WARNING:", warning)
    embedder = SentenceTransformerEmbedder() if {"dense", "hybrid"} & set(args.retrievers) else None
    timing = {"query_embedding_seconds_per_query": embed_queries(embedder, items) if embedder else None}
    cells, retriever_params = {}, {}
    for size in args.sizes:
        retrievers = build_retrievers(chunk_config_for_size(size), tuple(args.retrievers), embedder)
        for method, retriever in retrievers.items():
            retriever_params[method] = getattr(retriever, "params", {})
            rows, seconds = run_cell(retriever, items, k_values)
            cells[f"{size}/{method}"] = {"chunk_config": chunk_config_for_size(size).config_id,
                                         "retriever": method, "seconds_per_query": seconds / len(items),
                                         "aggregate": summarize(rows, k_values), "queries": rows}
            print(f"ran {size}/{method}: {len(rows)} queries in {seconds:.1f}s", flush=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = {"run_id": run_id, "config": run_configuration(args, items_path, retriever_params, embedder),
              "timing": timing, "comparisons": compare_outcomes(cells, args.sizes, args.compare_k), "cells": cells}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(cells, k_values)
    if timing["query_embedding_seconds_per_query"] is not None:
        print(f"(query embedding, done once up front: {timing['query_embedding_seconds_per_query']:.3f} s/query)")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
