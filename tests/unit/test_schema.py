"""The eval-item schema: ALL/ANY evidence groups, premise evidence, and load-time rejection (evaluation.schema)."""

import json

import pytest

from satc_rag.errors import EvaluationSchemaError
from satc_rag.evaluation.schema import load_items, parse_item, parse_items
from satc_rag.provenance import Span
from tests.support.builders import chunk_at

A = {"season": 3, "episode": 5, "source_row_start": 253, "source_row_end": 255, "expected_quote": "a"}
B = {"season": 3, "episode": 5, "source_row_start": 343, "source_row_end": 349, "expected_quote": "b"}
C = {"season": 3, "episode": 6, "source_row_start": 10, "source_row_end": 12, "expected_quote": "c"}
PREMISE = {"season": 3, "episode": 5, "source_row_start": 107, "source_row_end": 107, "role": "premise"}


def item(**fields):
    return {"id": "q", "question": "What happened?", **fields}


def test_a_single_target_is_one_required_group():
    parsed = parse_item(item(**A))
    assert [g.mode for g in parsed.groups] == ["all"]
    assert parsed.targets == (Span(3, 5, 253, 255, "a"),)
    assert len(parsed.requirements) == 1
    assert not parsed.needs_several


def test_targets_with_requires_all_is_one_all_group():
    parsed = parse_item(item(targets=[A, B], requires_all_targets=True))
    assert [g.mode for g in parsed.groups] == ["all"]
    assert len(parsed.requirements) == 2
    assert parsed.needs_several


def test_targets_without_requires_all_are_alternatives():
    parsed = parse_item(item(targets=[A, B], requires_all_targets=False))
    assert [g.mode for g in parsed.groups] == ["any"]
    assert len(parsed.requirements) == 1
    assert not parsed.needs_several


def test_required_groups_mix_all_and_any():
    parsed = parse_item(item(required_groups=[{"mode": "all", "targets": [C]}, {"mode": "any", "targets": [A, B]}]))
    assert [g.mode for g in parsed.groups] == ["all", "any"]
    assert [len(r) for r in parsed.requirements] == [1, 2]  # C alone; A or B
    assert parsed.needs_several
    assert len(parsed.targets) == 3


def test_relevance_uses_scored_targets_never_premise_evidence():
    parsed = parse_item(item(**A, evidence=[PREMISE]))
    assert parsed.premise == (Span(3, 5, 107, 107, role="premise"),)
    assert not parsed.is_relevant(chunk_at(3, 5, 100, 110))  # overlaps only the premise span
    assert not parsed.is_relevant(chunk_at(3, 5, 400, 440))  # the right episode, an unrelated scene
    assert parsed.is_relevant(chunk_at(3, 5, 250, 253))


@pytest.mark.parametrize(
    ("fields", "problem"),
    [
        ({"season": 3, "episode": 5}, "no source rows"),  # episode-wide relevance
        ({"targets": [{"season": 3, "episode": 5}], "requires_all_targets": True}, "no source rows"),
        ({"targets": [A, B]}, "explicit requires_all_targets"),
        ({"targets": [A], "requires_all_targets": True}, "only for several targets"),
        ({**A, "targets": [B]}, "exactly one way"),
        ({"targets": [A, B], "requires_all_targets": True, "required_groups": []}, "exactly one way"),
        ({"required_groups": [{"mode": "some", "targets": [A]}]}, "'all' or 'any'"),
        ({"required_groups": [{"mode": "any", "targets": [A]}]}, "single target"),
        ({"required_groups": []}, "non-empty list"),
        ({**A, "evidence": [{**B, "role": "answer"}]}, "role 'premise'"),
        ({**A, "source_row_start": 9, "source_row_end": 3}, "0 <= start <= end"),
        ({**A, "season": "3"}, "integer season"),
        ({**A, "question_type": "trivia"}, "unknown question_type"),
    ],
)
def test_invalid_items_are_rejected_with_the_reason(fields, problem):
    with pytest.raises(EvaluationSchemaError, match=problem):
        parse_item(item(**fields))


def test_missing_id_or_question_is_rejected():
    with pytest.raises(EvaluationSchemaError, match="'id'"):
        parse_item({"question": "q", **A})
    with pytest.raises(EvaluationSchemaError, match="question"):
        parse_item({"id": "x", **A})


def test_every_problem_in_a_file_is_reported_at_once():
    with pytest.raises(EvaluationSchemaError) as error:
        parse_items([item(**A), item(season=1, episode=1), {"id": "q", "question": "again", **B}], "items.json")
    message = str(error.value)
    assert "2 schema problem(s)" in message
    assert "no source rows" in message
    assert "duplicate ids: q" in message


@pytest.mark.parametrize(
    ("content", "problem"), [("{broken", "not valid JSON"), ("{}", "non-empty JSON list"), ("[]", "non-empty")]
)
def test_unreadable_eval_files_are_rejected_at_load(tmp_path, content, problem):
    path = tmp_path / "items.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(EvaluationSchemaError, match=problem):
        load_items(path)


def test_a_missing_eval_file_is_reported(tmp_path):
    with pytest.raises(EvaluationSchemaError, match="not found"):
        load_items(tmp_path / "absent.json")


def test_the_frozen_benchmark_keeps_its_evidence_structure(repo_root):
    items = load_items(repo_root / "eval" / "frozen" / "benchmark.json")
    assert len(items) == 60
    assert sum(len(i.targets) for i in items) == 100
    assert sum(len(i.premise) for i in items) == 24
    assert sum(1 for i in items if len(i.targets) > 1) == 29
    assert sum(1 for i in items if i.needs_several) == 28
    alternatives = [i.id for i in items if len(i.targets) > 1 and not i.needs_several]
    assert len(alternatives) == 1  # the one item whose targets are independent alternatives
    assert json.loads((repo_root / "eval" / "frozen" / "benchmark.json").read_text())[0]["status"] == "approved"
