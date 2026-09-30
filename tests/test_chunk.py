"""Token-based chunk boundaries in src/chunk.py."""
import random

import pytest

from src.chunk import chunk_all, format_line, pack_windows
from src.tokens import regex_counter
from tests.helpers import episode_lines


def check_windows(counts: list[int], size: int, overlap: int) -> None:
    """Assert every packing invariant for one input."""
    windows = pack_windows(counts, size, overlap)
    assert windows[0][0] == 0 and windows[-1][1] == len(counts)            # covers the start and end
    for (s1, e1), (s2, e2) in zip(windows, windows[1:]):
        assert s1 < s2 <= e1 < e2                                           # no gaps, always progresses
        assert sum(counts[s2:e1]) <= overlap                                # repeated lines within budget
    for start, end in windows:
        assert end - start == 1 or sum(counts[start:end]) <= size           # size limit (or one long line)


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


@pytest.mark.parametrize("size, overlap", [(10, 10), (10, 11), (0, 0), (10, -1)])
def test_invalid_settings_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        pack_windows([1, 2, 3], size, overlap)


@pytest.fixture
def two_episodes():
    """Two short episodes, including a line with no known speaker."""
    ep1 = episode_lines(4, 13, [("Carrie", f"line {i} about the apartment and Aidan") for i in range(30)])
    ep1[5]["speaker"], ep1[5]["speakers"] = None, []
    ep2 = episode_lines(4, 14, [("Miranda" if i % 2 else "Steve", f"line {i} about Brady") for i in range(30)])
    return ep1 + ep2


def test_chunks_respect_episodes_rows_and_order(two_episodes):
    chunks = chunk_all(two_episodes, regex_counter, size=40, overlap=8,
                       titles={(4, 13): "The Good Fight"}, with_scenes=False)
    by_id = {line["line_id"]: line for line in two_episodes}
    for chunk in chunks:
        lines = [by_id[i] for i in chunk["line_ids"]]
        assert {(l["season"], l["episode"]) for l in lines} == {(chunk["season"], chunk["episode"])}
        assert (chunk["source_row_start"], chunk["source_row_end"]) == (lines[0]["source_row"], lines[-1]["source_row"])
        assert chunk["text"] == "\n".join(format_line(l) for l in lines)
        assert chunk["speakers"] == list(dict.fromkeys(n for l in lines for n in l["speakers"]))
    covered = {i for chunk in chunks for i in chunk["line_ids"]}
    assert covered == set(by_id)


def test_chunk_ids_links_and_metadata(two_episodes):
    chunks = chunk_all(two_episodes, regex_counter, size=40, overlap=8,
                       titles={(4, 13): "The Good Fight"}, with_scenes=False)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    first_ep = [c for c in chunks if c["episode"] == 13]
    assert first_ep[0]["prev_chunk_id"] is None and first_ep[-1]["next_chunk_id"] is None
    assert first_ep[1]["prev_chunk_id"] == first_ep[0]["chunk_id"]
    assert all(c["episode_title"] == "The Good Fight" and "Good Fight" not in c["text"] for c in first_ep)
    assert any("(unknown): line 5 " in c["text"] for c in first_ep)


def test_scene_metadata_is_optional_and_changes_nothing_else(two_episodes):
    kwargs = dict(count_tokens=regex_counter, size=40, overlap=8, titles={})
    plain = chunk_all(two_episodes, with_scenes=False, **kwargs)
    with_scenes = chunk_all(two_episodes, with_scenes=True, **kwargs)
    assert all("inferred_scenes" not in c for c in plain)
    assert [{k: v for k, v in c.items() if k != "inferred_scenes"} for c in with_scenes] == plain
