"""The benchmark runner's own logic (evaluation.runner): scoring rows, summaries, comparisons, eval resolution."""

import pytest

from sexandrag.config import Settings
from sexandrag.errors import ConfigurationError, HeldOutSetError
from sexandrag.evaluation.runner import (
    EvalRequest,
    compare_outcomes,
    format_summary,
    resolve_eval_file,
    score_query,
    summarize,
)
from sexandrag.evaluation.schema import parse_item
from sexandrag.retrieve import SearchResult

K = (1, 5, 10)
ALL_K = (5, 10)


def result(rank, season, episode, start, end):
    return SearchResult(f"c{rank}", season, episode, None, start, end, (), 0.0, rank, "bm25", {}, "chunk text")


def cell(hits_at_5, ranks=None, recall_at_5=None):
    """A minimal result cell: one row per query id with hit@5, recall@5 and the deep first-relevant rank."""
    ranks, recall_at_5 = ranks or {}, recall_at_5 or {}
    return {
        "queries": [
            {
                "id": i,
                "hit@5": 1.0 if hit else 0.0,
                "recall@5": recall_at_5.get(i, 1.0 if hit else 0.0),
                "first_relevant_rank_in_depth": ranks.get(i),
            }
            for i, hit in hits_at_5.items()
        ]
    }


def test_comparisons_sort_queries_into_win_and_loss_buckets():
    cells = {
        "512/bm25": cell({"q1": True, "q2": False, "q3": False, "q4": True, "q5": False}),
        "512/dense": cell({"q1": False, "q2": True, "q3": False, "q4": True, "q5": False}),
        "512/hybrid": cell({"q1": True, "q2": True, "q3": True, "q4": False, "q5": False}),
        "256/bm25": cell({"q1": False, "q2": False, "q3": False, "q4": True, "q5": False}),
    }
    result_ = compare_outcomes(cells, [256, 512], k=5)
    assert result_["by_size"][512] == {
        "both_hit": ["q4"],
        "bm25_only": ["q1"],
        "dense_only": ["q2"],
        "both_missed": ["q3", "q5"],
        "hybrid_recovery": ["q3"],
        "hybrid_regression": ["q4"],
    }
    assert 256 not in result_["by_size"]
    assert result_["size_sensitive"]["bm25"] == {"q1": {256: False, 512: True}}


def test_rank_changes_and_partial_multi_target_recall_are_recorded():
    cells = {
        "256/dense": cell({"q1": False, "q2": True}, ranks={"q1": 23, "q2": 1}),
        "1024/dense": cell({"q1": True, "q2": True}, ranks={"q1": 2, "q2": 4}, recall_at_5={"q2": 0.5}),
    }
    result_ = compare_outcomes(cells, [256, 1024], k=5)
    assert result_["rank_changes"]["dense"] == {"q1": {256: 23, 1024: 2}, "q2": {256: 1, 1024: 4}}
    assert result_["rank_changes"]["bm25"] == {}
    assert result_["multi_target_partial"] == {"256/dense": [], "1024/dense": ["q2"]}


def test_metrics_use_the_top_k_even_when_a_deeper_ranking_is_stored():
    item = parse_item(
        {"id": "q", "question": "?", "season": 1, "episode": 1, "source_row_start": 200, "source_row_end": 200}
    )
    results = [result(rank, 1, 1, 10 * rank, 10 * rank + 5) for rank in range(1, 31)]  # rank 20 covers row 200
    row = score_query(item, results, K, ALL_K)
    assert row["hit@10"] == 0.0
    assert row["rr"] == 0.0
    assert row["first_relevant_rank"] is None
    assert row["first_relevant_rank_in_depth"] == 20
    assert row["target_first_ranks"] == [20]
    assert len(row["retrieved"]) == 30
    assert "text" not in row["retrieved"][0]


def test_all_targets_hit_is_scored_per_query_and_summarised_over_the_items_it_applies_to():
    arc = parse_item(
        {
            "id": "arc",
            "question": "?",
            "question_type": "arc",
            "requires_all_targets": True,
            "targets": [
                {"season": 3, "episode": 1, "source_row_start": 150, "source_row_end": 153},
                {"season": 3, "episode": 14, "source_row_start": 309, "source_row_end": 312},
                {"season": 4, "episode": 1, "source_row_start": 55, "source_row_end": 62},
            ],
        }
    )
    single = parse_item(
        {
            "id": "one",
            "question": "?",
            "question_type": "who_said",
            "season": 3,
            "episode": 1,
            "source_row_start": 150,
            "source_row_end": 153,
        }
    )
    results = [result(1, 3, 1, 140, 160), result(2, 3, 14, 300, 320)] + [result(r, 6, 20, 0, 9) for r in range(3, 11)]
    arc_row, single_row = score_query(arc, results, K, ALL_K), score_query(single, results, K, ALL_K)
    assert arc_row["hit@10"] == 1.0
    assert arc_row["recall@10"] == pytest.approx(2 / 3)
    assert arc_row["all_targets_hit@10"] == 0.0
    assert single_row["all_targets_hit@10"] is None
    summary = summarize([arc_row, single_row], K, ALL_K)
    assert summary["all"]["all_targets_hit@10"] == 0.0
    assert summary["all"]["n_all_targets"] == 1
    assert summary["by_type"]["who_said"]["all_targets_hit@10"] is None


def test_summaries_average_overall_and_by_question_type():
    rows = [
        {"question_type": "who_said", "hit@1": 1.0, "recall@1": 1.0, "rr": 1.0},
        {"question_type": "who_said", "hit@1": 0.0, "recall@1": 0.0, "rr": 0.5},
        {"question_type": "episode", "hit@1": 1.0, "recall@1": 1.0, "rr": 1.0},
    ]
    summary = summarize(rows, (1,), ALL_K)
    assert summary["all"] == {"hit@1": 0.6667, "recall@1": 0.6667, "rr": 0.8333, "n": 3}
    assert summary["by_type"]["who_said"] == {"hit@1": 0.5, "recall@1": 0.5, "rr": 0.75, "n": 2}


def test_the_default_request_is_the_development_set(tmp_path):
    settings = Settings.defaults(tmp_path)
    assert EvalRequest().split == "dev"
    assert resolve_eval_file(EvalRequest(), settings) == (settings.paths.dev_file, "dev")


def test_the_test_split_needs_the_explicit_opt_in(tmp_path):
    settings = Settings.defaults(tmp_path)
    with pytest.raises(HeldOutSetError, match="--allow-heldout"):
        resolve_eval_file(EvalRequest(split="test"), settings)
    assert resolve_eval_file(EvalRequest(split="test", allow_heldout=True), settings)[1] == "test"
    with pytest.raises(ConfigurationError, match="unknown split"):
        resolve_eval_file(EvalRequest(split="train"), settings)


def test_the_summary_table_shows_every_metric():
    cells = {
        "512/bm25": {
            "aggregate": {
                "all": {
                    "hit@1": 0.5,
                    "hit@5": 1.0,
                    "hit@10": 1.0,
                    "recall@1": 0.5,
                    "recall@5": 1.0,
                    "recall@10": 1.0,
                    "all_targets_hit@5": None,
                    "all_targets_hit@10": 1.0,
                    "rr": 0.75,
                    "n": 2,
                }
            },
            "seconds_per_query": 0.01,
        }
    }
    header, row = format_summary(cells, K, ALL_K)
    assert "all_tgt@10" in header
    assert row.startswith("512/bm25")
    assert " - " in row
