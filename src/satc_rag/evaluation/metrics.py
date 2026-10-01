"""Retrieval metrics over row-level relevance.

An item's scored evidence is a set of independent requirements (see schema.EvalItem): each
target of an "all" group is one requirement; an "any" group is one requirement satisfied by any
of its alternatives.

* Hit@k: any scored target is reached in the top k.
* Recall@k: the fraction of requirements reached in the top k (equals Hit@k for one requirement).
* AllTargetsHit@k: every requirement is reached in the top k. It only applies to items needing
  several pieces of evidence; for other items it is None and averages skip it.
* MRR: 1 / rank of the first relevant chunk within the retrieved list (MRR@max_k when the list
  is cut at the largest k).
"""

from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from satc_rag.config import DEFAULT_EVALUATION
from satc_rag.evaluation.schema import EvalItem

ALL_TARGETS = "all_targets_hit"
Chunk = Mapping[str, Any]


def first_relevant_rank(ranked: Sequence[Chunk], item: EvalItem) -> int | None:
    """1-based rank of the first chunk that overlaps any scored target, or None."""
    return next((rank for rank, chunk in enumerate(ranked, 1) if item.is_relevant(chunk)), None)


def targets_reached(ranked: Sequence[Chunk], item: EvalItem, k: int) -> list[bool]:
    """For each scored target: does any of the top-k chunks hit it?"""
    top = ranked[:k]
    return [any(target.hits(chunk) for chunk in top) for target in item.targets]


def requirements_reached(ranked: Sequence[Chunk], item: EvalItem, k: int) -> list[bool]:
    """For each independent requirement: is any of its spans hit in the top k?"""
    top = ranked[:k]
    return [any(span.hits(chunk) for span in requirement for chunk in top) for requirement in item.requirements]


def hit_at_k(ranked: Sequence[Chunk], item: EvalItem, k: int) -> float:
    """1.0 when at least one scored target is reached in the top k, else 0.0."""
    return float(any(targets_reached(ranked, item, k)))


def recall_at_k(ranked: Sequence[Chunk], item: EvalItem, k: int) -> float:
    """Fraction of the item's evidence requirements reached in the top k."""
    reached = requirements_reached(ranked, item, k)
    return sum(reached) / len(reached)


def all_targets_hit_at_k(ranked: Sequence[Chunk], item: EvalItem, k: int) -> float | None:
    """1.0 when every requirement is reached in the top k; None unless the item needs several."""
    if not item.needs_several:
        return None
    return float(all(requirements_reached(ranked, item, k)))


def target_first_ranks(ranked: Sequence[Chunk], item: EvalItem) -> list[int | None]:
    """For each target: the 1-based rank of the first chunk that hits it, or None (for recall analysis)."""
    return [next((rank for rank, chunk in enumerate(ranked, 1) if target.hits(chunk)), None) for target in item.targets]


def reciprocal_rank(ranked: Sequence[Chunk], item: EvalItem) -> float:
    """1 / rank of the first relevant chunk, or 0.0 when none is relevant."""
    rank = first_relevant_rank(ranked, item)
    return 1.0 / rank if rank else 0.0


def all_targets_cutoffs(k_values: Sequence[int], all_targets_k: Sequence[int]) -> list[int]:
    """The cut-offs at which AllTargetsHit is reported (configured and requested)."""
    return [k for k in k_values if k in all_targets_k]


def score_ranking(
    ranked: Sequence[Chunk],
    item: EvalItem,
    k_values: Sequence[int],
    all_targets_k: Sequence[int] = DEFAULT_EVALUATION.all_targets_k,
) -> dict[str, Any]:
    """All metrics for one ranked list of chunk provenance dicts."""
    scores: dict[str, Any] = {
        "first_relevant_rank": first_relevant_rank(ranked, item),
        "rr": reciprocal_rank(ranked, item),
    }
    for k in k_values:
        scores[f"hit@{k}"] = hit_at_k(ranked, item, k)
        scores[f"recall@{k}"] = recall_at_k(ranked, item, k)
    for k in all_targets_cutoffs(k_values, all_targets_k):
        scores[f"{ALL_TARGETS}@{k}"] = all_targets_hit_at_k(ranked, item, k)
    return scores


def metric_names(k_values: Sequence[int], all_targets_k: Sequence[int] = DEFAULT_EVALUATION.all_targets_k) -> list[str]:
    """Names of the averaged metrics, in display order."""
    return (
        [f"hit@{k}" for k in k_values]
        + [f"recall@{k}" for k in k_values]
        + [f"{ALL_TARGETS}@{k}" for k in all_targets_cutoffs(k_values, all_targets_k)]
        + ["rr"]
    )


def average(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> dict[str, Any]:
    """Mean of each named metric over the rows it applies to (MRR is the mean of 'rr').

    AllTargetsHit is averaged only over items needing several pieces of evidence; their number is
    reported as n_all_targets.
    """
    summary: dict[str, Any] = {}
    for name in names:
        values = [row[name] for row in rows if row.get(name) is not None]
        summary[name] = round(mean(values), 4) if values else None
    summary["n"] = len(rows)
    all_target_names = [name for name in names if name.startswith(ALL_TARGETS)]
    if all_target_names:
        summary["n_all_targets"] = sum(1 for row in rows if row.get(all_target_names[0]) is not None)
    return summary
