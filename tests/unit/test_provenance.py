"""Source-row spans, relevance by overlap, and quote lookup (satc_rag.provenance)."""

import pytest

from satc_rag.chunk import chunk_all
from satc_rag.provenance import Span, find_quote, quote_status, spans_overlap
from satc_rag.tokens import regex_counter
from tests.support.builders import chunk_at, episode_lines, make_line


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((10, 20), (20, 25), True),  # share the edge row
        ((10, 19), (20, 25), False),  # adjacent but disjoint
        ((10, 30), (15, 16), True),  # containment
        ((15, 15), (15, 15), True),  # single row
    ],
)
def test_spans_overlap_is_inclusive(a, b, expected):
    assert spans_overlap(*a, *b) is expected
    assert spans_overlap(*b, *a) is expected


def test_same_rows_in_another_episode_do_not_count():
    span = Span(4, 13, 17, 22)
    assert span.hits(chunk_at(4, 13, 0, 17))
    assert not span.hits(chunk_at(4, 12, 0, 40))
    assert not span.hits(chunk_at(4, 13, 23, 60))


def test_labels_and_lengths():
    assert Span(3, 5, 253, 255).label == "S3E5 r253-255"
    assert Span(3, 5, 253, 255).n_rows == 3


@pytest.fixture
def s1e1():
    """The end of the S1E1 cold open, where one sentence spans two rows."""
    return [
        make_line(1, 1, 36, "Carrie", "No one has breakfast at Tiffany's, and no one has affairs to remember."),
        make_line(1, 1, 37, "Carrie", "Instead, we have breakfast at 7:00 a.m."),
        make_line(1, 1, 38, "Carrie", "and affairs we try to forget as quickly as possible."),
    ]


def test_quote_is_found_across_rows_ignoring_case_and_punctuation(s1e1):
    hits = find_quote(s1e1, "WE HAVE BREAKFAST AT 7:00 AM and affairs we try to forget")
    assert [(h["source_row_start"], h["source_row_end"]) for h in hits] == [(37, 38)]
    assert find_quote(s1e1, "no one has affairs to remember")[0]["source_row_start"] == 36
    assert find_quote(s1e1, "tiffanys")
    assert find_quote(s1e1, "brunch at noon") == []


@pytest.mark.parametrize(
    ("span", "status"),
    [
        (Span(1, 1, 37, 38, "breakfast at 7:00"), "ok"),
        (Span(1, 1, 30, 36, "breakfast at 7:00"), "moved"),
        (Span(1, 1, 37, 38, "something never said"), "missing"),
        (Span(1, 1, 37, 38), "no_quote"),
    ],
)
def test_quote_status(s1e1, span, status):
    assert quote_status(s1e1, span) == status


@pytest.mark.parametrize(("size", "overlap"), [(16, 0), (32, 4), (64, 8), (128, 16)])
def test_relevance_is_defined_by_rows_whatever_the_chunking(size, overlap):
    lines = episode_lines(2, 5, [("Carrie" if i % 3 else "Miranda", f"words {i} " * (1 + i % 4)) for i in range(80)])
    span = Span(2, 5, 41, 43)
    chunks = chunk_all(lines, regex_counter, size, overlap, titles={}, scenes=None)
    assert [c for c in chunks if span.hits(c)], "some chunk must always cover the target rows"
    for c in chunks:
        rows = {int(i.split("-r")[1].split("-")[0]) for i in c["line_ids"]}
        assert span.hits(c) == bool(rows & {41, 42, 43})
