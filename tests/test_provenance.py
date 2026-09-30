"""Evaluation ground truth by source-row overlap in src/provenance.py."""
import pytest

from src.chunk import chunk_all
from src.provenance import (chunk_hits_target, find_quote, first_relevant_rank, is_relevant,
                            spans_overlap, validate_target)
from src.tokens import regex_counter
from tests.helpers import episode_lines, make_line


def chunk(season: int, episode: int, start: int, end: int) -> dict:
    """A minimal chunk covering source rows start..end."""
    return {"season": season, "episode": episode, "source_row_start": start, "source_row_end": end}


@pytest.mark.parametrize("a, b, expected", [
    ((10, 20), (20, 25), True),      # share the edge row
    ((10, 19), (20, 25), False),     # adjacent but disjoint
    ((10, 30), (15, 16), True),      # containment
    ((15, 15), (15, 15), True),      # single row
])
def test_spans_overlap_is_inclusive(a, b, expected):
    assert spans_overlap(*a, *b) is expected
    assert spans_overlap(*b, *a) is expected


def test_same_rows_in_another_episode_do_not_count():
    target = {"season": 4, "episode": 13, "source_row_start": 17, "source_row_end": 22}
    assert chunk_hits_target(chunk(4, 13, 0, 17), target)
    assert not chunk_hits_target(chunk(4, 12, 0, 40), target)
    assert not chunk_hits_target(chunk(4, 13, 23, 60), target)


def test_episode_level_target_matches_any_chunk_of_the_episode():
    target = {"season": 3, "episode": 5}
    assert chunk_hits_target(chunk(3, 5, 200, 240), target)
    assert not chunk_hits_target(chunk(3, 6, 200, 240), target)


def test_items_with_several_targets_and_ranks():
    item = {"targets": [{"season": 1, "episode": 1, "source_row_start": 5, "source_row_end": 6},
                        {"season": 6, "episode": 20, "source_row_start": 400, "source_row_end": 410}]}
    ranked = [chunk(2, 1, 0, 50), chunk(6, 20, 380, 401), chunk(1, 1, 0, 30)]
    assert is_relevant(ranked[1], item) and not is_relevant(ranked[0], item)
    assert first_relevant_rank(ranked, item) == 2
    assert first_relevant_rank(ranked[:1], item) is None


@pytest.fixture
def s1e1():
    """The end of the S1E1 cold open, where one sentence spans two rows."""
    return [make_line(1, 1, 36, "Carrie", "No one has breakfast at Tiffany's, and no one has affairs to remember."),
            make_line(1, 1, 37, "Carrie", "Instead, we have breakfast at 7:00 a.m."),
            make_line(1, 1, 38, "Carrie", "and affairs we try to forget as quickly as possible.")]


def test_quote_is_found_across_rows_ignoring_case_and_punctuation(s1e1):
    hits = find_quote(s1e1, "WE HAVE BREAKFAST AT 7:00 AM and affairs we try to forget")
    assert [(h["source_row_start"], h["source_row_end"]) for h in hits] == [(37, 38)]
    assert find_quote(s1e1, "no one has affairs to remember")[0]["source_row_start"] == 36
    assert find_quote(s1e1, "tiffanys")                     # apostrophes are ignored
    assert find_quote(s1e1, "brunch at noon") == []


@pytest.mark.parametrize("target, status", [
    ({"source_row_start": 37, "source_row_end": 38, "expected_quote": "breakfast at 7:00"}, "ok"),
    ({"source_row_start": 30, "source_row_end": 36, "expected_quote": "breakfast at 7:00"}, "moved"),
    ({"source_row_start": 37, "source_row_end": 38, "expected_quote": "something never said"}, "missing"),
    ({"source_row_start": 37, "source_row_end": 38}, "no_quote"),
])
def test_validate_target(s1e1, target, status):
    assert validate_target(s1e1, {"season": 1, "episode": 1, **target})["status"] == status


@pytest.mark.parametrize("size, overlap", [(16, 0), (32, 4), (64, 8), (128, 16)])
def test_relevance_is_defined_by_rows_whatever_the_chunking(size, overlap):
    lines = episode_lines(2, 5, [("Carrie" if i % 3 else "Miranda", f"words {i} " * (1 + i % 4)) for i in range(80)])
    target = {"season": 2, "episode": 5, "source_row_start": 41, "source_row_end": 43}
    chunks = chunk_all(lines, regex_counter, size, overlap, titles={}, with_scenes=False)
    relevant = [c for c in chunks if chunk_hits_target(c, target)]
    assert relevant, "some chunk must always cover the target rows"
    for c in chunks:
        rows = {int(i.split("-r")[1].split("-")[0]) for i in c["line_ids"]}
        assert chunk_hits_target(c, target) == bool(rows & {41, 42, 43})
