"""The committed frozen benchmark: hashes, split, and the guarantees it must keep (no corpus needed)."""

import json

from satc_rag.config import Settings
from satc_rag.evaluation.schema import load_items
from satc_rag.evaluation.split import read_manifest, verify_frozen
from tests.conftest import REPO_ROOT

FROZEN = REPO_ROOT / "eval" / "frozen"


def test_the_frozen_files_match_their_hashes_and_the_split_reproduces():
    assert verify_frozen(FROZEN) == []


def test_dev_and_test_partition_the_sixty_items():
    manifest = read_manifest(FROZEN)
    ids = {
        name: [i["id"] for i in json.loads((FROZEN / name).read_text())]
        for name in ("benchmark.json", "dev.json", "test.json")
    }
    assert (len(ids["benchmark.json"]), len(ids["dev.json"]), len(ids["test.json"])) == (60, 20, 40)
    assert not set(ids["dev.json"]) & set(ids["test.json"])
    assert sorted(ids["dev.json"] + ids["test.json"]) == sorted(ids["benchmark.json"])
    assert sorted(ids["dev.json"]) == manifest["split"]["dev_ids"]


def test_every_scored_target_has_explicit_rows_and_premise_is_separate():
    items = load_items(FROZEN / "benchmark.json")  # a target without rows would raise here
    assert sum(len(i.targets) for i in items) == 100
    assert sum(len(i.premise) for i in items) == 24
    assert all(t.start <= t.end for i in items for t in i.targets)


def test_the_default_evaluation_is_the_development_set():
    settings = Settings.defaults(REPO_ROOT)
    assert settings.paths.dev_file == FROZEN / "dev.json"
    assert settings.evaluation.split_seed == read_manifest(FROZEN)["split"]["seed"]
