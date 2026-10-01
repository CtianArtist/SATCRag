"""Token-based chunk boundaries and chunk-set manifests (satc_rag.chunk)."""

import json
import random
from itertools import pairwise

import pytest

from satc_rag.artifacts import ChunkConfig
from satc_rag.chunk import build_chunk_set, chunk_all, format_line, load_chunk_set, pack_windows
from satc_rag.config import DEFAULT_SCENES
from satc_rag.errors import ArtifactMismatchError, ArtifactMissingError, StaleArtifactError
from satc_rag.tokens import regex_counter
from tests.support.builders import episode_lines


def check_windows(counts: list[int], size: int, overlap: int) -> None:
    """Assert every packing invariant for one input."""
    windows = pack_windows(counts, size, overlap)
    assert windows[0][0] == 0
    assert windows[-1][1] == len(counts)
    for (s1, e1), (s2, e2) in pairwise(windows):
        assert s1 < s2 <= e1 < e2  # no gaps, always progresses
        assert sum(counts[s2:e1]) <= overlap  # repeated lines within budget
    for start, end in windows:
        assert end - start == 1 or sum(counts[start:end]) <= size  # size limit (or one long line)


@pytest.mark.parametrize("seed", range(200))
def test_packing_invariants_hold_for_random_inputs(seed):
    rng = random.Random(seed)
    counts = [rng.randint(1, 60) for _ in range(rng.randint(1, 120))]
    size = rng.choice([16, 64, 256, 512])
    check_windows(counts, size, rng.randint(0, size - 1))


def test_zero_overlap_gives_disjoint_consecutive_windows():
    assert pack_windows([5, 5, 5, 5, 5], size=10, overlap=0) == [(0, 2), (2, 4), (4, 5)]


def test_overlap_repeats_trailing_lines():
    assert pack_windows([4, 4, 4, 4, 4], size=12, overlap=4) == [(0, 3), (2, 5)]


def test_overlap_never_crowds_out_the_next_new_line():
    assert pack_windows([10, 10, 10, 100], size=105, overlap=20) == [(0, 3), (3, 4)]


def test_a_line_longer_than_the_limit_becomes_its_own_chunk():
    assert pack_windows([3, 50, 3], size=10, overlap=2) == [(0, 1), (1, 2), (2, 3)]


@pytest.mark.parametrize(("size", "overlap"), [(10, 10), (10, 11), (0, 0), (10, -1)])
def test_invalid_settings_are_rejected(size, overlap):
    with pytest.raises(ValueError, match="overlap < size"):
        pack_windows([1, 2, 3], size, overlap)


@pytest.fixture
def two_episodes():
    """Two short episodes, including a line with no known speaker."""
    ep1 = episode_lines(4, 13, [("Carrie", f"line {i} about the apartment and Aidan") for i in range(30)])
    ep1[5]["speaker"], ep1[5]["speakers"] = None, []
    ep2 = episode_lines(4, 14, [("Miranda" if i % 2 else "Steve", f"line {i} about Brady") for i in range(30)])
    return ep1 + ep2


def test_chunks_respect_episodes_rows_and_order(two_episodes):
    chunks = chunk_all(two_episodes, regex_counter, size=40, overlap=8, titles={(4, 13): "The Good Fight"}, scenes=None)
    by_id = {line["line_id"]: line for line in two_episodes}
    for chunk in chunks:
        lines = [by_id[i] for i in chunk["line_ids"]]
        assert {(line["season"], line["episode"]) for line in lines} == {(chunk["season"], chunk["episode"])}
        assert (chunk["source_row_start"], chunk["source_row_end"]) == (lines[0]["source_row"], lines[-1]["source_row"])
        assert chunk["text"] == "\n".join(format_line(line) for line in lines)
        assert chunk["speakers"] == list(dict.fromkeys(n for line in lines for n in line["speakers"]))
    assert {i for chunk in chunks for i in chunk["line_ids"]} == set(by_id)


def test_chunk_ids_links_and_metadata(two_episodes):
    chunks = chunk_all(two_episodes, regex_counter, size=40, overlap=8, titles={(4, 13): "The Good Fight"}, scenes=None)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    first_ep = [c for c in chunks if c["episode"] == 13]
    assert first_ep[0]["prev_chunk_id"] is None
    assert first_ep[-1]["next_chunk_id"] is None
    assert first_ep[1]["prev_chunk_id"] == first_ep[0]["chunk_id"]
    assert all(c["episode_title"] == "The Good Fight" and "Good Fight" not in c["text"] for c in first_ep)
    assert any("(unknown): line 5 " in c["text"] for c in first_ep)


def test_scene_metadata_is_optional_and_changes_nothing_else(two_episodes):
    plain = chunk_all(two_episodes, regex_counter, 40, 8, {}, scenes=None)
    with_scenes = chunk_all(two_episodes, regex_counter, 40, 8, {}, scenes=DEFAULT_SCENES)
    assert all("inferred_scenes" not in c for c in plain)
    assert [{k: v for k, v in c.items() if k != "inferred_scenes"} for c in with_scenes] == plain


@pytest.fixture
def built(tmp_path, two_episodes):
    """A chunk set written to tmp_path with its manifest, plus the lines file it came from."""
    lines_path = tmp_path / "lines.jsonl"
    lines_path.write_text("".join(json.dumps(line) + "\n" for line in two_episodes), encoding="utf-8")
    config = ChunkConfig(40, 8, "regex", None)
    chunk_set = build_chunk_set(config, two_episodes, regex_counter, {}, None, tmp_path / "chunks", lines_path)
    return config, chunk_set, tmp_path / "chunks", lines_path


def test_a_saved_chunk_set_loads_back_identically(built):
    config, chunk_set, chunks_dir, lines_path = built
    loaded = load_chunk_set(config, chunks_dir, lines_path)
    assert loaded.chunks == chunk_set.chunks
    assert loaded.manifest["n_chunks"] == len(chunk_set.chunks)


def test_chunk_sets_refuse_altered_or_stale_files(built):
    config, _, chunks_dir, lines_path = built
    with pytest.raises(StaleArtifactError, match="different tokenizer"):
        load_chunk_set(ChunkConfig(40, 8, "regex", "another-revision"), chunks_dir, lines_path)
    path = config.jsonl_path(chunks_dir)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(StaleArtifactError, match="chunk file changed") as error:
        load_chunk_set(config, chunks_dir, lines_path)
    assert "satc-rag chunk --size 40 --overlap 8" in str(error.value)


def test_a_changed_lines_file_makes_the_chunks_stale(built):
    config, _, chunks_dir, lines_path = built
    lines_path.write_text(lines_path.read_text() + "{}\n")
    with pytest.raises(StaleArtifactError, match=r"lines\.jsonl changed"):
        load_chunk_set(config, chunks_dir, lines_path)


def test_missing_and_corrupt_manifests_are_reported(built):
    config, _, chunks_dir, lines_path = built
    with pytest.raises(ArtifactMissingError, match="has not been built"):
        load_chunk_set(ChunkConfig(80, 16, "regex", None), chunks_dir, lines_path)
    config.manifest_path(chunks_dir).write_text("{truncated", encoding="utf-8")
    with pytest.raises(ArtifactMismatchError, match="not valid JSON"):
        load_chunk_set(config, chunks_dir, lines_path)
    config.manifest_path(chunks_dir).write_text(json.dumps({"config_id": "x"}), encoding="utf-8")
    with pytest.raises(ArtifactMismatchError, match="lacks"):
        load_chunk_set(config, chunks_dir, lines_path)
