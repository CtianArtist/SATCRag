"""Retrieval metrics on top of source-row relevance (src/provenance.py).

For an item with one target, Hit@k and Recall@k coincide. For an item with several
targets they differ: Hit@k asks whether ANY target was reached in the top k, Recall@k
asks what FRACTION of the targets was reached. MRR uses the first relevant rank within
the retrieved list (so it is MRR@max_k when the list is cut at max_k).
"""
from statistics import mean

from src.provenance import chunk_hits_target, first_relevant_rank, item_targets


def targets_reached(ranked: list[dict], item: dict, k: int) -> list[bool]:
    """For each of the item's targets: does any of the top-k chunks hit it?"""
    top = ranked[:k]
    return [any(chunk_hits_target(chunk, target) for chunk in top) for target in item_targets(item)]


def hit_at_k(ranked: list[dict], item: dict, k: int) -> float:
    """1.0 when at least one target is reached in the top k, else 0.0."""
    return float(any(targets_reached(ranked, item, k)))


def recall_at_k(ranked: list[dict], item: dict, k: int) -> float:
    """Fraction of the item's targets reached in the top k."""
    reached = targets_reached(ranked, item, k)
    return sum(reached) / len(reached)


def reciprocal_rank(ranked: list[dict], item: dict) -> float:
    """1 / rank of the first relevant chunk, or 0.0 when none is relevant."""
    rank = first_relevant_rank(ranked, item)
    return 1.0 / rank if rank else 0.0


def score_ranking(ranked: list[dict], item: dict, k_values: tuple[int, ...]) -> dict:
    """All metrics for one ranked list of chunk provenance dicts."""
    scores = {"first_relevant_rank": first_relevant_rank(ranked, item), "rr": reciprocal_rank(ranked, item)}
    for k in k_values:
        scores[f"hit@{k}"] = hit_at_k(ranked, item, k)
        scores[f"recall@{k}"] = recall_at_k(ranked, item, k)
    return scores


def metric_names(k_values: tuple[int, ...]) -> list[str]:
    """Names of the averaged metrics, in display order."""
    return [f"hit@{k}" for k in k_values] + [f"recall@{k}" for k in k_values] + ["rr"]


def average(rows: list[dict], names: list[str]) -> dict:
    """Mean of each named metric over per-query rows (MRR is the mean of 'rr')."""
    return {name: round(mean(row[name] for row in rows), 4) for name in names} | {"n": len(rows)}
