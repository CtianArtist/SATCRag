"""The held-out guard (evaluation.heldout): the test set only runs with an explicit opt-in."""

import json
import logging
import shutil

import pytest

from sexandrag.config import PathsConfig
from sexandrag.errors import HeldOutSetError
from sexandrag.evaluation.heldout import guard_held_out
from sexandrag.evaluation.schema import load_items


@pytest.fixture
def paths(tmp_path, repo_root):
    """A project whose eval/frozen is a copy of the real frozen benchmark."""
    shutil.copytree(repo_root / "eval" / "frozen", tmp_path / "eval" / "frozen")
    return PathsConfig.under(tmp_path)


def write_items(path, items):
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


def test_the_development_set_passes_the_guard(paths):
    assert guard_held_out(paths.dev_file, load_items(paths.dev_file), paths, allow_heldout=False) is False


def test_the_test_file_is_refused_without_the_opt_in(paths):
    with pytest.raises(HeldOutSetError, match="frozen test file"):
        guard_held_out(paths.test_file, load_items(paths.test_file), paths, allow_heldout=False)


def test_a_renamed_copy_of_the_test_file_is_refused(paths, tmp_path):
    copy = shutil.copyfile(paths.test_file, tmp_path / "my_questions.json")
    with pytest.raises(HeldOutSetError, match="byte-identical copy"):
        guard_held_out(copy, load_items(copy), paths, allow_heldout=False)


def test_a_file_with_some_test_items_is_refused(paths, tmp_path):
    test_items = json.loads(paths.test_file.read_text())
    dev_items = json.loads(paths.dev_file.read_text())
    mixed = write_items(tmp_path / "mixed.json", dev_items[:2] + test_items[:1])
    with pytest.raises(HeldOutSetError, match="held-out item id"):
        guard_held_out(mixed, load_items(mixed), paths, allow_heldout=False)


def test_test_questions_under_new_ids_are_refused(paths, tmp_path):
    item = json.loads(paths.test_file.read_text())[0]
    renamed = write_items(tmp_path / "renamed.json", [dict(item, id="looks-new")])
    with pytest.raises(HeldOutSetError, match="held-out question"):
        guard_held_out(renamed, load_items(renamed), paths, allow_heldout=False)


def test_the_explicit_opt_in_allows_the_run_and_logs_it(paths, caplog):
    with caplog.at_level(logging.WARNING, logger="sexandrag"):
        assert guard_held_out(paths.test_file, load_items(paths.test_file), paths, allow_heldout=True) is True
    assert "HELD-OUT TEST SET RUN" in caplog.text


def test_without_a_frozen_benchmark_there_is_nothing_to_guard(tmp_path, repo_root):
    paths = PathsConfig.under(tmp_path)
    smoke = repo_root / "eval" / "smoke.json"
    assert guard_held_out(smoke, load_items(smoke), paths, allow_heldout=False) is False


def test_the_smoke_file_touches_no_held_out_data(repo_root):
    paths = PathsConfig.under(repo_root)
    smoke = repo_root / "eval" / "smoke.json"
    assert guard_held_out(smoke, load_items(smoke), paths, allow_heldout=False) is False
