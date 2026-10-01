"""Validate eval items against the parsed corpus and describe the set for human review.

On top of the schema (schema.py), benchmark items must carry a question_type and style, and
every scored target and premise span must name an existing episode and rows with an
expected_quote that lies inside the span. The report shows the spread of question types,
styles, seasons and characters, span and target counts, and lexical overlap: the share of a
question's content words that also appear in its scored text (high overlap favours BM25).

Evidence sources: dialogue text is corpus evidence and speaker labels are corpus metadata
evidence (both are part of the indexed chunk text); inferred scenes are never evidence.

The review sheet quotes the corpus next to each question, so it belongs under
data/processed/ (git-ignored). Nothing here touches retrieval.
"""

import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from itertools import groupby
from pathlib import Path
from typing import Any

from satc_rag.errors import EvaluationSchemaError
from satc_rag.evaluation.schema import QUESTION_TYPES, EvalItem, parse_item
from satc_rag.provenance import Span, normalize_for_match, quote_status

Line = Mapping[str, Any]
Episodes = dict[tuple[int, int], list[Line]]
STOPWORD_TEXT = """
a about after again all also an and any are as at be because been before being both but by can could
did do does doing during each for from had has have having he her here hers him his how i if in into
is it its just me more most my no nor not of off on once only or other our out over own same she
should so some such than that the their them then there these they this those through to too under
until up very was we were what when where which while who whom why will with would you your s t d ll
re ve m don doesn didn isn wasn won
"""
STOPWORDS = frozenset(STOPWORD_TEXT.split())
RARE_EPISODES = 3
MAX_QUOTE_WORDS = 8
NAME_VARIANTS = {"Jack Berger": ("Berger",), "Aleksandr": ("Aleksandr", "Russian", "Petrovsky")}


def by_episode(lines: Sequence[Line]) -> Episodes:
    """Lines grouped by (season, episode), in file order."""
    return {key: list(group) for key, group in groupby(lines, key=lambda line: (line["season"], line["episode"]))}


def span_lines(episodes: Episodes, span: Span) -> list[Line]:
    """The lines inside a span."""
    return [
        line for line in episodes.get((span.season, span.episode), []) if span.start <= line["source_row"] <= span.end
    ]


def check_span(episodes: Episodes, span: Span) -> list[str]:
    """Problems with one span: missing episode or rows, missing quote, or quote outside the span."""
    key = (span.season, span.episode)
    if key not in episodes:
        return [f"no episode S{key[0]}E{key[1]}"]
    last_row = episodes[key][-1]["source_row"]
    if span.end > last_row:
        return [f"{span.label} is outside the episode's rows 0-{last_row}"]
    if not span.quote:
        return [f"{span.label} has no expected_quote"]
    status = quote_status(episodes[key], span)
    return [] if status == "ok" else [f"{span.label}: quote {status}"]


def check_item(raw: Mapping[str, Any], episodes: Episodes) -> list[str]:
    """All problems with one benchmark item: schema, labels, and every span against the corpus."""
    try:
        item = parse_item(raw)
    except EvaluationSchemaError as exc:
        return [exc.message]
    problems = []
    if item.question_type is None:
        problems.append(f"missing question_type (one of {', '.join(QUESTION_TYPES)})")
    if item.style is None:
        problems.append("missing style")
    for span in (*item.targets, *item.premise):
        problems += check_span(episodes, span)
    return problems


def content_words(text: str) -> set[str]:
    """Normalized words minus stopwords and one-letter tokens."""
    return {w for w in normalize_for_match(text).split() if w not in STOPWORDS and len(w) > 1}


def scored_text(item: EvalItem, episodes: Episodes) -> str:
    """The text of the item's scored targets (premise evidence excluded)."""
    return " ".join(line["clean_text"] for span in item.targets for line in span_lines(episodes, span))


def lexical_overlap(item: EvalItem, episodes: Episodes) -> float | None:
    """Share of the question's content words found in its scored text (premise evidence does not count)."""
    words = content_words(item.question)
    text = scored_text(item, episodes)
    if not words or not text:
        return None
    return len(words & content_words(text)) / len(words)


def rare_words(episodes: Episodes) -> set[str]:
    """Content words that occur in at most RARE_EPISODES episodes (the kind of term BM25 latches onto)."""
    seen_in: Counter[str] = Counter()
    for lines in episodes.values():
        seen_in.update({w for line in lines for w in content_words(line["clean_text"])})
    return {w for w, n in seen_in.items() if n <= RARE_EPISODES}


def shared_rare_words(item: EvalItem, episodes: Episodes, rare: set[str]) -> set[str]:
    """Rare words that appear in both the question and its scored text."""
    return content_words(item.question) & content_words(scored_text(item, episodes)) & rare


def overlap_band(value: float | None) -> str:
    """Bucket an overlap fraction."""
    if value is None:
        return "n/a"
    return "low (<1/3)" if value < 1 / 3 else ("medium" if value < 2 / 3 else "high (>2/3)")


def main_characters(lines: Sequence[Line], min_lines: int = 60) -> set[str]:
    """Speakers with at least `min_lines` lines, used to spot character names in questions."""
    counts = Counter(name for line in lines for name in line["speakers"])
    return {name for name, n in counts.items() if n >= min_lines}


def mentions(text: str, name: str) -> bool:
    """True when the text names the character (or one of its listed variants)."""
    return any(re.search(rf"\b{re.escape(v)}\b", text) for v in NAME_VARIANTS.get(name, (name,)))


def percentile(values: Sequence[int], q: float) -> int:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * len(ordered)) - 1))]


def overlapping_targets(items: Sequence[EvalItem]) -> list[tuple[str, str]]:
    """Pairs of items whose scored targets overlap (the same passage tested twice)."""
    placed = [(i.id, t.season, t.episode, t.start, t.end) for i in items for t in i.targets]
    return sorted(
        {
            (a[0], b[0])
            for a in placed
            for b in placed
            if a[0] < b[0] and a[1:3] == b[1:3] and a[3] <= b[4] and b[3] <= a[4]
        }
    )


def distribution(items: Sequence[EvalItem], episodes: Episodes, characters: set[str]) -> list[str]:
    """Readable distribution of a valid set: types, styles, seasons, targets, spans, quotes, overlap."""
    out = [f"{len(items)} items"]
    out.append(f"by question_type: {dict(Counter(i.question_type for i in items))}")
    styles = Counter(i.style for i in items)
    out.append(f"by style: { {s: f'{n} ({n / len(items):.0%})' for s, n in styles.most_common()} }")
    seasons = Counter(s for i in items for s in {t.season for t in i.targets})
    out.append(f"items touching each season: {dict(sorted(seasons.items()))}")
    touched = {(t.season, t.episode) for i in items for t in i.targets}
    out.append(f"distinct episodes represented: {len(touched)} of {len(episodes)}")
    out.append(f"targets per item: {dict(sorted(Counter(len(i.targets) for i in items).items()))}")
    multi = [i for i in items if len(i.targets) > 1]
    need_all = sum(1 for i in multi if i.needs_several)
    out.append(
        f"multi-target items: {len(multi)} ({need_all} need every required group/target, "
        f"{len(multi) - need_all} have only alternatives)"
    )
    lengths = [t.n_rows for i in items for t in i.targets]
    premise = sum(len(i.premise) for i in items)
    out.append(f"scored target spans: {len(lengths)}; premise spans (never scored): {premise}")
    out.append(
        f"target span length (rows): p10 {percentile(lengths, 0.1)}, median {statistics.median(lengths)}, "
        f"p90 {percentile(lengths, 0.9)}, max {max(lengths)}"
    )
    spans = [s for i in items for s in (*i.targets, *i.premise) if s.quote]
    words = [len((s.quote or "").split()) for s in spans]
    long_quotes = [
        (i.id, s.quote)
        for i in items
        for s in (*i.targets, *i.premise)
        if len((s.quote or "").split()) > MAX_QUOTE_WORDS
    ]
    out.append(
        f"expected_quote length (words): median {statistics.median(words)}, max {max(words)}; "
        f"over {MAX_QUOTE_WORDS} words: {long_quotes or 'none'}"
    )
    named = Counter(name for i in items for name in characters if mentions(f"{i.question} {i.expected_answer}", name))
    out.append(f"characters named in question or answer: {dict(named.most_common())}")
    out.append(f"items whose targets overlap another item's: {overlapping_targets(items) or 'none'}")
    out += overlap_report(items, episodes)
    return out


def overlap_report(items: Sequence[EvalItem], episodes: Episodes) -> list[str]:
    """Lexical overlap bands by question type, and rare words shared with the target by style."""
    bands: defaultdict[str | None, Counter[str]] = defaultdict(Counter)
    for i in items:
        bands[i.question_type][overlap_band(lexical_overlap(i, episodes))] += 1
    out = ["lexical overlap (question content words found in target text):"]
    out += [f"  {qtype:10s} {dict(bands[qtype])}" for qtype in QUESTION_TYPES if qtype in bands]
    out.append(f"  {'all':10s} {dict(sum(bands.values(), Counter()))}")
    rare = rare_words(episodes)
    hooks: defaultdict[str | None, list[bool]] = defaultdict(list)
    for i in items:
        hooks[i.style].append(bool(shared_rare_words(i, episodes, rare)))
    out.append(f"questions sharing a rare word (used in <= {RARE_EPISODES} episodes) with their target, by style:")
    out += [f"  {style!s:14s} {sum(flags)}/{len(flags)}" for style, flags in sorted(hooks.items(), key=str)]
    return out


def review_sheet(items: Sequence[EvalItem], episodes: Episodes, unknown_speaker: str) -> str:
    """A Markdown sheet with each question next to the lines its spans point at (quotes the corpus)."""
    out = ["# Eval review sheet\n", "Generated by `satc-rag validate`. Quotes the corpus: do not commit.\n"]
    for item in items:
        overlap = lexical_overlap(item, episodes)
        shown = "n/a" if overlap is None else f"{overlap:.2f}"
        out.append(f"\n## {item.id} ({item.question_type}, {item.style}, overlap {shown})\n")
        out.append(f"**Q:** {item.question}  \n**Expected answer:** {item.expected_answer}  ")
        if item.raw.get("notes"):
            out.append(f"**Notes:** {item.raw['notes']}  ")
        if len(item.targets) > 1:
            groups = "; ".join(f"{g.mode} of {len(g.targets)}" for g in item.groups)
            out.append(f"**Scoring:** {len(item.targets)} targets in groups: {groups}  ")
        spans = [("target", s) for s in item.targets] + [("premise (not scored)", s) for s in item.premise]
        for kind, span in spans:
            label = kind + (f" [{span.role}]" if kind == "target" and span.role else "")
            out.append(f'\n{label} {span.label} ({span.n_rows} rows), quote: "{span.quote}"\n')
            out += [
                f"    r{line['source_row']:03d} {line['speaker'] or unknown_speaker}: {line['clean_text']}"
                for line in span_lines(episodes, span)
            ]
    return "\n".join(out) + "\n"


def validate_raw_items(raw_items: Sequence[Mapping[str, Any]], episodes: Episodes) -> dict[str, list[str]]:
    """{item id: problems} for every item with at least one problem (duplicate ids included)."""
    problems = {str(raw.get("id")): found for raw in raw_items if (found := check_item(raw, episodes))}
    duplicates = [i for i, n in Counter(str(raw.get("id")) for raw in raw_items).items() if n > 1]
    for item_id in duplicates:
        problems.setdefault(item_id, []).append("duplicate id")
    return problems


def write_review(path: Path, text: str) -> None:
    """Write the review sheet (callers keep it under the git-ignored data/processed/)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
