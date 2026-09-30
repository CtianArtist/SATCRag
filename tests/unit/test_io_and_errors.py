"""Small boundaries: JSON Lines I/O, run directories, episode metadata, and error messages."""

import json
from datetime import UTC, datetime

import pytest

from sexandrag.episodes import load_episode_titles
from sexandrag.errors import ArtifactMismatchError, ArtifactMissingError, MetadataError, SexAndRagError
from sexandrag.evaluation.results import create_run_dir, run_id
from sexandrag.jsonl import read_json, read_jsonl, write_jsonl


def test_jsonl_round_trips_and_skips_blank_lines(tmp_path):
    path = tmp_path / "x.jsonl"
    assert write_jsonl(path, [{"a": 1}, {"b": "é"}]) == 2
    path.write_text(path.read_text() + "\n\n", encoding="utf-8")
    assert list(read_jsonl(path)) == [{"a": 1}, {"b": "é"}]
    assert not path.with_name("x.jsonl.tmp").exists()


def test_a_corrupt_jsonl_line_names_the_file_and_line(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"a": 1}\n{"b": \n', encoding="utf-8")
    with pytest.raises(ArtifactMismatchError, match="line 2"):
        list(read_jsonl(path))


def test_missing_json_files_are_reported(tmp_path):
    with pytest.raises(ArtifactMissingError):
        list(read_jsonl(tmp_path / "absent.jsonl"))
    with pytest.raises(ArtifactMissingError, match="manifest"):
        read_json(tmp_path / "absent.json", "manifest")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(ArtifactMismatchError, match="not valid JSON"):
        read_json(tmp_path / "bad.json", "manifest")


def test_run_directories_are_never_reused(tmp_path):
    started = datetime(2026, 9, 30, 12, 0, 0, tzinfo=UTC)
    base = run_id(started, "dev", {"commit": "bf7535250c7622525d75949de2c50c117dbebbec", "dirty": True})
    assert base == "20260930T120000Z-dev-bf75352-dirty"
    first, second = create_run_dir(tmp_path, base), create_run_dir(tmp_path, base)
    assert first.name == base
    assert second.name == base + "-2"
    assert first.is_dir()
    assert second.is_dir()


def test_run_ids_work_outside_git():
    assert (
        run_id(datetime(2026, 1, 1, tzinfo=UTC), "my file!", {"commit": None, "dirty": None})
        == "20260101T000000Z-my-file-nogit"
    )


def test_episode_metadata_must_be_complete_and_unique(tmp_path):
    path = tmp_path / "episodes.csv"
    path.write_text("season,episode,episode_title\n1,1,Pilot\n1,2,Second\n", encoding="utf-8")
    assert load_episode_titles(path) == {(1, 1): "Pilot", (1, 2): "Second"}
    path.write_text("season,episode,episode_title\n1,1,Pilot\n1,1,Again\n", encoding="utf-8")
    with pytest.raises(MetadataError, match="duplicate"):
        load_episode_titles(path)
    path.write_text("season,title\n1,Pilot\n", encoding="utf-8")
    with pytest.raises(MetadataError, match="unexpected columns"):
        load_episode_titles(path)
    with pytest.raises(MetadataError, match="not found"):
        load_episode_titles(tmp_path / "absent.csv")


def test_errors_explain_what_expected_actual_and_how_to_fix():
    error = SexAndRagError("the cache is stale", expected="abc", actual="def", recovery="rebuild it")
    assert str(error) == "the cache is stale\n  expected: abc\n  actual:   def\n  fix:      rebuild it"
    assert str(SexAndRagError("plain")) == "plain"


def test_write_json_output_is_valid_json(tmp_path):
    from sexandrag.jsonl import write_json

    write_json(tmp_path / "a" / "b.json", {"x": [1, 2]})
    assert json.loads((tmp_path / "a" / "b.json").read_text()) == {"x": [1, 2]}
