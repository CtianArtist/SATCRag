"""The benchmark runner's own logic (eval/run_eval.py): item validation, comparisons, summaries."""
import json

import pytest

from eval.run_eval import compare_outcomes, load_items, summarize


def write_items(tmp_path, items) -> str:
    path = tmp_path / "items.json"
    path.write_text(json.dumps(items))
    return path


@pytest.mark.parametrize("items, problem", [
    ([{"id": "a", "question": "q", "season": 1, "episode": 1}, {"id": "a", "question": "q", "season": 1, "episode": 1}], "unique"),
    ([{"id": "a", "season": 1, "episode": 1}], "question"),
    ([{"id": "a", "question": "q", "season": "1", "episode": 1}], "integer"),
    ([{"id": "a", "question": "q", "targets": [{"season": 1}]}], "integer"),
])
def test_malformed_eval_items_are_rejected(tmp_path, items, problem):
    with pytest.raises(ValueError, match=problem):
        load_items(write_items(tmp_path, items))


def test_valid_items_load_including_multi_target(tmp_path):
    items = [{"id": "a", "question": "q", "season": 1, "episode": 1},
             {"id": "b", "question": "q", "targets": [{"season": 2, "episode": 3}, {"season": 4, "episode": 5}]}]
    assert load_items(write_items(tmp_path, items)) == items


def cell(hits_at_5: dict[str, bool]) -> dict:
    """A minimal result cell: one query row per id with only hit@5 set."""
    return {"queries": [{"id": i, "hit@5": 1.0 if hit else 0.0} for i, hit in hits_at_5.items()]}


def test_comparisons_sort_queries_into_win_and_loss_buckets():
    cells = {
        "512/bm25": cell({"q1": True, "q2": False, "q3": False, "q4": True, "q5": False}),
        "512/dense": cell({"q1": False, "q2": True, "q3": False, "q4": True, "q5": False}),
        "512/hybrid": cell({"q1": True, "q2": True, "q3": True, "q4": False, "q5": False}),
        "256/bm25": cell({"q1": False, "q2": False, "q3": False, "q4": True, "q5": False}),
    }
    result = compare_outcomes(cells, [256, 512], k=5)
    buckets = result["by_size"][512]
    assert buckets == {"bm25_only": ["q1"], "dense_only": ["q2"], "hybrid_recovered": ["q3"],
                       "hybrid_lost": ["q4"], "all_missed": ["q5"]}
    assert 256 not in result["by_size"]                       # dense and hybrid were not run at 256
    assert result["size_sensitive"]["bm25"] == {"q1": {256: False, 512: True}}


def test_summaries_average_overall_and_by_question_type():
    rows = [{"type": "who_said", "hit@1": 1.0, "recall@1": 1.0, "rr": 1.0},
            {"type": "who_said", "hit@1": 0.0, "recall@1": 0.0, "rr": 0.5},
            {"type": "episode", "hit@1": 1.0, "recall@1": 1.0, "rr": 1.0}]
    summary = summarize(rows, (1,))
    assert summary["all"] == {"hit@1": 0.6667, "recall@1": 0.6667, "rr": 0.8333, "n": 3}
    assert summary["by_type"]["who_said"] == {"hit@1": 0.5, "recall@1": 0.5, "rr": 0.75, "n": 2}
