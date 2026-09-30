"""Evaluation ground truth by source-row provenance, independent of chunk size or overlap.

An eval item names a location: season, episode and optionally an inclusive span
source_row_start..source_row_end (the CSV's per-episode row index), plus an optional
expected_quote. Multi-location items may instead give a list under "targets".
A retrieved chunk is relevant when it comes from the same episode and its row span
overlaps a target span; a target with no rows matches any chunk of its episode.
"""
import re
from bisect import bisect_right
from itertools import groupby


def spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """True when inclusive ranges [a_start, a_end] and [b_start, b_end] share at least one row."""
    return a_start <= b_end and b_start <= a_end


def item_targets(item: dict) -> list[dict]:
    """The item's target locations: item['targets'] if given, else the item itself."""
    return item.get("targets") or [item]


def target_span(target: dict) -> tuple[int, int] | None:
    """Inclusive (start, end) rows of a target, or None for an episode-level target."""
    start = target.get("source_row_start")
    if start is None:
        return None
    end = target.get("source_row_end")
    return start, start if end is None else end


def chunk_hits_target(chunk: dict, target: dict) -> bool:
    """True when the chunk is from the target's episode and overlaps its row span (if any)."""
    if (chunk["season"], chunk["episode"]) != (target["season"], target["episode"]):
        return False
    span = target_span(target)
    return span is None or spans_overlap(chunk["source_row_start"], chunk["source_row_end"], *span)


def is_relevant(chunk: dict, item: dict) -> bool:
    """True when the chunk hits any of the item's targets."""
    return any(chunk_hits_target(chunk, target) for target in item_targets(item))


def first_relevant_rank(ranked_chunks: list[dict], item: dict) -> int | None:
    """1-based rank of the first relevant chunk in a ranked list, or None if none is relevant."""
    return next((rank for rank, chunk in enumerate(ranked_chunks, 1) if is_relevant(chunk, item)), None)


def normalize_for_match(text: str) -> str:
    """Lowercase, drop apostrophes and periods ('a.m.' == 'am'), turn other punctuation into spaces."""
    text = re.sub(r"['’.]", "", text.lower())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def find_quote(lines: list[dict], quote: str) -> list[dict]:
    """Find every occurrence of `quote` in the lines, which may span consecutive lines of an episode.

    Returns [{season, episode, source_row_start, source_row_end, line_ids}] in file order.
    """
    needle = normalize_for_match(quote)
    if not needle:
        return []
    hits = []
    for _, group in groupby(lines, key=lambda l: (l["season"], l["episode"])):
        hits += find_in_episode(list(group), needle)
    return hits


def find_in_episode(lines: list[dict], needle: str) -> list[dict]:
    """Search one episode's concatenated normalized text and map matches back to lines."""
    texts = [normalize_for_match(line["clean_text"]) for line in lines]
    starts, offset = [], 0
    for text in texts:
        starts.append(offset)
        offset += len(text) + 1                       # +1 for the joining space
    haystack = " ".join(texts)
    hits, pos = [], haystack.find(needle)
    while pos != -1:
        first = bisect_right(starts, pos) - 1
        last = bisect_right(starts, pos + len(needle) - 1) - 1
        span = lines[first:last + 1]
        hits.append({"season": span[0]["season"], "episode": span[0]["episode"],
                     "source_row_start": span[0]["source_row"], "source_row_end": span[-1]["source_row"],
                     "line_ids": [line["line_id"] for line in span]})
        pos = haystack.find(needle, pos + 1)
    return hits


def validate_target(lines: list[dict], target: dict) -> dict:
    """Check that the target's expected_quote still lies inside its episode and row span.

    status: 'ok' (quote inside the span), 'moved' (quote elsewhere in the episode),
    'missing' (quote not in the episode), or 'no_quote' (nothing to check).
    """
    quote = target.get("expected_quote")
    if not quote:
        return {"status": "no_quote", "found": []}
    episode_lines = [l for l in lines if (l["season"], l["episode"]) == (target["season"], target["episode"])]
    found = find_quote(episode_lines, quote)
    span = target_span(target)
    inside = [h for h in found
              if span is None or span[0] <= h["source_row_start"] and h["source_row_end"] <= span[1]]
    status = "ok" if inside else ("moved" if found else "missing")
    return {"status": status, "found": found}
