"""Hit@k, Recall@k and MRR over source-row relevance (src/metrics.py)."""
import pytest

from src.metrics import average, hit_at_k, metric_names, recall_at_k, reciprocal_rank, score_ranking


def chunk(season: int, episode: int, start: int, end: int) -> dict:
    return {"season": season, "episode": episode, "source_row_start": start, "source_row_end": end}


SINGLE = {"id": "q1", "season": 4, "episode": 13, "source_row_start": 17, "source_row_end": 22}
MULTI = {"id": "q2", "targets": [
    {"season": 3, "episode": 12, "source_row_start": 100, "source_row_end": 110},
    {"season": 4, "episode": 12, "source_row_start": 300, "source_row_end": 305},
    {"season": 6, "episode": 20},                                   # episode-level target
]}
RANKED = [chunk(1, 1, 0, 40), chunk(4, 12, 290, 330), chunk(4, 13, 10, 18), chunk(3, 12, 90, 120), chunk(6, 20, 0, 30)]


def test_single_target_hit_and_recall_coincide():
    for k in (1, 2, 3, 5):
        assert hit_at_k(RANKED, SINGLE, k) == recall_at_k(RANKED, SINGLE, k) == float(k >= 3)
    assert reciprocal_rank(RANKED, SINGLE) == pytest.approx(1 / 3)


def test_multi_target_recall_counts_each_target_separately():
    assert hit_at_k(RANKED, MULTI, 1) == 0.0 and recall_at_k(RANKED, MULTI, 1) == 0.0
    assert hit_at_k(RANKED, MULTI, 2) == 1.0 and recall_at_k(RANKED, MULTI, 2) == pytest.approx(1 / 3)
    assert recall_at_k(RANKED, MULTI, 4) == pytest.approx(2 / 3)
    assert recall_at_k(RANKED, MULTI, 5) == 1.0
    assert reciprocal_rank(RANKED, MULTI) == 0.5


def test_nothing_relevant_scores_zero():
    item = {"id": "q3", "season": 2, "episode": 2, "source_row_start": 5, "source_row_end": 6}
    scores = score_ranking(RANKED, item, (1, 5, 10))
    assert scores["first_relevant_rank"] is None and scores["rr"] == 0.0
    assert all(scores[f"hit@{k}"] == 0.0 == scores[f"recall@{k}"] for k in (1, 5, 10))


def test_k_larger_than_the_list_only_counts_what_was_retrieved():
    assert hit_at_k(RANKED, SINGLE, 10) == 1.0
    assert recall_at_k(RANKED[:2], SINGLE, 10) == 0.0


def test_averages_are_plain_means_and_mrr_is_the_mean_rr():
    rows = [score_ranking(RANKED, SINGLE, (1, 5)), score_ranking(RANKED, MULTI, (1, 5))]
    means = average(rows, metric_names((1, 5)))
    assert means["rr"] == pytest.approx((1 / 3 + 1 / 2) / 2, abs=1e-4)
    assert means["hit@5"] == 1.0 and means["recall@1"] == 0.0 and means["n"] == 2
