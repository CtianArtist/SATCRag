"""Source-row provenance: spans of dialogue rows, relevance by overlap, and quote lookup.

A Span is an inclusive range of an episode's source rows (the CSV's per-episode row index).
A retrieved chunk is relevant to a span when it comes from the same episode and its row range
overlaps the span. This is independent of chunk size and overlap, so the same ground truth
scores every chunk setting. An episode alone never makes a chunk relevant.
"""

import re
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from typing import Any

Line = Mapping[str, Any]


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """True when inclusive ranges [a_start, a_end] and [b_start, b_end] share at least one row."""
    return a_start <= b_end and b_start <= a_end


@dataclass(frozen=True)
class Span:
    """An inclusive range of source rows in one episode, with an optional short quote and role."""

    season: int
    episode: int
    start: int
    end: int
    quote: str | None = None
    role: str | None = None  # "anchor"/"answer" for sequence targets, "premise" for premise evidence

    def hits(self, chunk: Mapping[str, Any]) -> bool:
        """True when the chunk is from this span's episode and overlaps its rows."""
        return (chunk["season"], chunk["episode"]) == (self.season, self.episode) and spans_overlap(
            chunk["source_row_start"], chunk["source_row_end"], self.start, self.end
        )

    @property
    def label(self) -> str:
        """Short human-readable location, e.g. 'S3E5 r253-255'."""
        return f"S{self.season}E{self.episode} r{self.start}-{self.end}"

    @property
    def n_rows(self) -> int:
        """Number of source rows the span covers."""
        return self.end - self.start + 1


def normalize_for_match(text: str) -> str:
    """Lowercase, drop apostrophes and periods ('a.m.' == 'am'), turn other punctuation into spaces."""
    text = re.sub("['\u2019.]", "", text.lower())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def find_quote(lines: Sequence[Line], quote: str) -> list[dict[str, Any]]:
    """Find every occurrence of `quote` in the lines, which may span consecutive lines of an episode.

    Returns [{season, episode, source_row_start, source_row_end, line_ids}] in file order.
    """
    needle = normalize_for_match(quote)
    if not needle:
        return []
    hits = []
    for _, group in groupby(lines, key=lambda line: (line["season"], line["episode"])):
        hits += find_in_episode(list(group), needle)
    return hits


def find_in_episode(lines: list[Line], needle: str) -> list[dict[str, Any]]:
    """Search one episode's concatenated normalized text and map matches back to lines."""
    texts = [normalize_for_match(line["clean_text"]) for line in lines]
    starts, offset = [], 0
    for text in texts:
        starts.append(offset)
        offset += len(text) + 1  # +1 for the joining space
    haystack = " ".join(texts)
    hits, pos = [], haystack.find(needle)
    while pos != -1:
        first = bisect_right(starts, pos) - 1
        last = bisect_right(starts, pos + len(needle) - 1) - 1
        span = lines[first : last + 1]
        hits.append(
            {
                "season": span[0]["season"],
                "episode": span[0]["episode"],
                "source_row_start": span[0]["source_row"],
                "source_row_end": span[-1]["source_row"],
                "line_ids": [line["line_id"] for line in span],
            }
        )
        pos = haystack.find(needle, pos + 1)
    return hits


def quote_status(episode_lines: Sequence[Line], span: Span) -> str:
    """Whether the span's quote still lies inside it: 'ok', 'moved', 'missing' or 'no_quote'."""
    if not span.quote:
        return "no_quote"
    found = find_quote(episode_lines, span.quote)
    inside = [h for h in found if span.start <= h["source_row_start"] and h["source_row_end"] <= span.end]
    return "ok" if inside else ("moved" if found else "missing")
