"""Hit@k, Recall@k, AllTargetsHit@k and MRR over row-level relevance (evaluation.metrics)."""

import pytest

from satc_rag.evaluation.metrics import (
    all_targets_hit_at_k,
    average,
    hit_at_k,
    metric_names,
    recall_at_k,
    reciprocal_rank,
    score_ranking,
    target_first_ranks,
)
from satc_rag.evaluation.schema import parse_item
from tests.support.builders import chunk_at


def span(season, episode, start, end):
    return {"season": season, "episode": episode, "source_row_start": start, "source_row_end": end}


SINGLE = parse_item({"id": "q1", "question": "?", **span(4, 13, 17, 22)})
MULTI = parse_item(
    {
        "id": "q2",
        "question": "?",
        "requires_all_targets": True,
        "targets": [span(3, 12, 100, 110), span(4, 12, 300, 305), span(6, 20, 10, 12)],
    }
)
ARC = MULTI  # three required passages
RANKED = [
    chunk_at(1, 1, 0, 40),
    chunk_at(4, 12, 290, 330),
    chunk_at(4, 13, 10, 18),
    chunk_at(3, 12, 90, 120),
    chunk_at(6, 20, 0, 30),
]


def test_single_target_hit_and_recall_coincide():
    for k in (1, 2, 3, 5):
        assert hit_at_k(RANKED, SINGLE, k) == recall_at_k(RANKED, SINGLE, k) == float(k >= 3)
    assert reciprocal_rank(RANKED, SINGLE) == pytest.approx(1 / 3)


def test_multi_target_recall_counts_each_required_target():
    assert hit_at_k(RANKED, MULTI, 1) == 0.0
    assert recall_at_k(RANKED, MULTI, 1) == 0.0
    assert hit_at_k(RANKED, MULTI, 2) == 1.0
    assert recall_at_k(RANKED, MULTI, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, MULTI, 4) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, MULTI, 5) == 1.0
    assert reciprocal_rank(RANKED, MULTI) == 0.5
    assert target_first_ranks(RANKED, MULTI) == [4, 2, 5]


def test_the_all_targets_example_from_the_spec():
    # Three required passages; the top 10 retrieves two of them.
    ranked = [chunk_at(3, 12, 95, 105), chunk_at(2, 2, 0, 10), chunk_at(4, 12, 300, 320)] + [chunk_at(5, 5, 0, 10)] * 7
    assert hit_at_k(ranked, ARC, 10) == 1.0
    assert recall_at_k(ranked, ARC, 10) == pytest.approx(2 / 3)
    assert all_targets_hit_at_k(ranked, ARC, 10) == 0.0
    complete = [*ranked[:9], chunk_at(6, 20, 0, 30)]  # the third passage arrives at rank 10
    assert all_targets_hit_at_k(complete, ARC, 10) == 1.0
    assert all_targets_hit_at_k(complete, ARC, 5) == 0.0


def test_any_groups_are_satisfied_by_any_alternative():
    alternatives = parse_item(
        {
            "id": "q3",
            "question": "?",
            "requires_all_targets": False,
            "targets": [span(3, 5, 253, 255), span(3, 5, 343, 349)],
        }
    )
    ranked = [chunk_at(3, 5, 340, 350)]  # only the second alternative
    assert hit_at_k(ranked, alternatives, 1) == recall_at_k(ranked, alternatives, 1) == 1.0
    assert all_targets_hit_at_k(ranked, alternatives, 1) is None  # one requirement: not a multi-evidence item


def test_mixed_groups_need_every_group():
    mixed = parse_item(
        {
            "id": "q4",
            "question": "?",
            "required_groups": [
                {"mode": "all", "targets": [span(1, 1, 5, 6)]},
                {"mode": "any", "targets": [span(2, 2, 10, 12), span(2, 2, 40, 42)]},
            ],
        }
    )
    only_alternative = [chunk_at(2, 2, 40, 45)]
    both = [chunk_at(2, 2, 40, 45), chunk_at(1, 1, 0, 9)]
    assert recall_at_k(only_alternative, mixed, 5) == 0.5
    assert all_targets_hit_at_k(only_alternative, mixed, 5) == 0.0
    assert recall_at_k(both, mixed, 5) == 1.0
    assert all_targets_hit_at_k(both, mixed, 5) == 1.0


def test_nothing_relevant_scores_zero():
    item = parse_item({"id": "q5", "question": "?", **span(2, 2, 5, 6)})
    scores = score_ranking(RANKED, item, (1, 5, 10))
    assert scores["first_relevant_rank"] is None
    assert scores["rr"] == 0.0
    assert all(scores[f"hit@{k}"] == 0.0 == scores[f"recall@{k}"] for k in (1, 5, 10))


def test_k_larger_than_the_list_only_counts_what_was_retrieved():
    assert hit_at_k(RANKED, SINGLE, 10) == 1.0
    assert recall_at_k(RANKED[:2], SINGLE, 10) == 0.0


def test_averages_skip_items_a_metric_does_not_apply_to():
    rows = [score_ranking(RANKED, SINGLE, (1, 5, 10)), score_ranking(RANKED, MULTI, (1, 5, 10))]
    assert rows[0]["all_targets_hit@5"] is None
    assert rows[1]["all_targets_hit@5"] == 1.0
    means = average(rows, metric_names((1, 5, 10)))
    assert means["rr"] == pytest.approx((1 / 3 + 1 / 2) / 2, abs=1e-4)
    assert means["hit@5"] == 1.0
    assert means["recall@1"] == 0.0
    assert means["n"] == 2
    assert means["all_targets_hit@5"] == 1.0
    assert means["n_all_targets"] == 1
    assert "all_targets_hit@1" not in means  # reported at k = 5 and 10 only
