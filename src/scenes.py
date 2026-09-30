"""Optional heuristic scene grouping: deterministic, labeled inferred, never ground truth.

Two signals mark a candidate boundary before line i of an episode:
  * transition_phrase: line i opens with a narration-style transition ("Later that night").
  * speaker_turnover:  the speakers of the SCENE_WINDOW lines before i barely overlap the
    speakers of the SCENE_WINDOW lines after i (rarity-weighted Jaccard <= SCENE_MAX_SIMILARITY,
    at a local minimum).
Candidates are accepted strongest-first while every scene stays >= SCENE_MIN_LINES long.
Scenes are contiguous ranges of an episode's lines; line order is never changed, and no
location or voice-over is inferred.

Run: python -m src.scenes [--show S4E13] [--limit 80]
"""
import argparse
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from itertools import groupby

from src import config
from src.jsonl import read_jsonl, write_jsonl


@dataclass
class Candidate:
    """A possible scene boundary before line `index` of an episode."""
    index: int
    strength: float = 0.0
    signals: list[str] = field(default_factory=list)
    similarity: float | None = None


def window_speakers(lines: list[dict], start: int, end: int) -> set[str]:
    """Named speakers in lines[start:end]; unknown speakers are ignored."""
    return {name for line in lines[max(start, 0):end] for name in line["speakers"]}


def speaker_weights(lines: list[dict]) -> dict[str, float]:
    """Weight each speaker by rarity in the episode, log(total lines / speaker's lines).

    Someone present almost everywhere (usually the narrator) says little about where one
    scene ends and the next begins, so they count for less.
    """
    counts = Counter(name for line in lines for name in line["speakers"])
    total = sum(counts.values())
    return {name: math.log(total / n) for name, n in counts.items()}


def weighted_jaccard(a: set, b: set, weights: dict[str, float]) -> float:
    """Weighted set overlap: weight of a & b over weight of a | b (1.0 when the union weighs nothing)."""
    union = sum(weights[s] for s in a | b)
    return sum(weights[s] for s in a & b) / union if union else 1.0


def similarity_profile(lines: list[dict], window: int) -> list[float | None]:
    """Speaker-set similarity for the gap before each line (None where a side has no speakers)."""
    weights = speaker_weights(lines)
    sims = [None]
    for i in range(1, len(lines)):
        before, after = window_speakers(lines, i - window, i), window_speakers(lines, i, i + window)
        sims.append(weighted_jaccard(before, after, weights) if before and after else None)
    return sims


def turnover_candidates(sims: list[float | None], max_similarity: float) -> list[Candidate]:
    """Gaps whose similarity is at most `max_similarity` and no higher than either neighbour."""
    def value(j: int) -> float:
        return sims[j] if 0 <= j < len(sims) and sims[j] is not None else float("inf")
    return [Candidate(i, 1.0 + (max_similarity - s), ["speaker_turnover"], s)
            for i, s in enumerate(sims)
            if s is not None and s <= max_similarity and s <= value(i - 1) and s <= value(i + 1)]


def transition_candidates(lines: list[dict], pattern: str) -> list[Candidate]:
    """Lines (after the first) that open with a narration-style transition phrase."""
    regex = re.compile(pattern, re.IGNORECASE)
    return [Candidate(i, 2.0, ["transition_phrase"])
            for i, line in enumerate(lines) if i > 0 and regex.search(line["clean_text"])]


def merge_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Combine candidates at the same gap by adding their strengths and signals."""
    merged: dict[int, Candidate] = {}
    for c in candidates:
        m = merged.setdefault(c.index, Candidate(c.index))
        m.strength += c.strength
        m.signals += c.signals
        m.similarity = c.similarity if c.similarity is not None else m.similarity
    return list(merged.values())


def select_boundaries(candidates: list[Candidate], n_lines: int, min_lines: int) -> list[Candidate]:
    """Accept candidates strongest-first (ties: earliest) while every scene keeps >= min_lines lines."""
    chosen: list[Candidate] = []
    for c in sorted(candidates, key=lambda c: (-c.strength, c.index)):
        fits_episode = min_lines <= c.index <= n_lines - min_lines
        if fits_episode and all(abs(c.index - k.index) >= min_lines for k in chosen):
            chosen.append(c)
    return sorted(chosen, key=lambda c: c.index)


def make_scene(lines: list[dict], number: int, opener: Candidate, end: int) -> dict:
    """Describe lines[opener.index:end] as one inferred scene."""
    part = lines[opener.index:end]
    return {
        "season": part[0]["season"], "episode": part[0]["episode"], "scene": number, "inferred": True,
        "line_start": opener.index, "line_end": end,              # indices into the episode's lines
        "source_row_start": part[0]["source_row"], "source_row_end": part[-1]["source_row"],
        "first_line_id": part[0]["line_id"], "last_line_id": part[-1]["line_id"],
        "n_lines": len(part),
        "speakers": sorted(window_speakers(part, 0, len(part))),
        "boundary_signals": opener.signals, "boundary_similarity": opener.similarity,
    }


def detect_scenes(lines: list[dict], window: int = config.SCENE_WINDOW,
                  max_similarity: float = config.SCENE_MAX_SIMILARITY,
                  min_lines: int = config.SCENE_MIN_LINES,
                  pattern: str = config.SCENE_TRANSITION_PATTERN) -> list[dict]:
    """Group one episode's ordered lines into inferred scenes covering every line exactly once."""
    if not lines:
        return []
    sims = similarity_profile(lines, window)
    candidates = merge_candidates(turnover_candidates(sims, max_similarity)
                                  + transition_candidates(lines, pattern))
    openers = [Candidate(0, signals=["episode_start"])] + select_boundaries(candidates, len(lines), min_lines)
    ends = [c.index for c in openers[1:]] + [len(lines)]
    return [make_scene(lines, n, opener, end) for n, (opener, end) in enumerate(zip(openers, ends), 1)]


def scene_numbers(scenes: list[dict], n_lines: int) -> list[int]:
    """Inferred scene number for each line index of one episode."""
    numbers = [0] * n_lines
    for scene in scenes:
        numbers[scene["line_start"]:scene["line_end"]] = [scene["scene"]] * scene["n_lines"]
    return numbers


def episodes(lines: list[dict]) -> list[list[dict]]:
    """Group lines into episodes in file order."""
    return [list(g) for _, g in groupby(lines, key=lambda l: (l["season"], l["episode"]))]


def print_stats(scenes: list[dict], n_episodes: int) -> None:
    """Print how many scenes were inferred, how long they are, and which signals opened them."""
    per_episode = Counter((s["season"], s["episode"]) for s in scenes).values()
    lengths = [s["n_lines"] for s in scenes]
    signals = Counter("+".join(s["boundary_signals"]) for s in scenes)
    print(f"Inferred scenes: {len(scenes)} over {n_episodes} episodes")
    print(f"  scenes/episode min/median/max: {min(per_episode)}/{statistics.median(per_episode)}/{max(per_episode)}")
    print(f"  lines/scene    min/median/max: {min(lengths)}/{statistics.median(lengths)}/{max(lengths)}")
    print(f"  boundary signals: {dict(signals.most_common())}")


def show_episode(lines: list[dict], scenes: list[dict], limit: int) -> None:
    """Print an episode's first `limit` lines with inferred scene breaks marked."""
    starts = {s["line_start"]: s for s in scenes}
    for i, line in enumerate(lines[:limit]):
        if i in starts:
            scene = starts[i]
            detail = ", ".join(scene["boundary_signals"])
            if scene["boundary_similarity"] is not None:
                detail += f", similarity {scene['boundary_similarity']:.2f}"
            print(f"  ---- inferred scene {scene['scene']} ({detail}) ----")
        print(f"  r{line['source_row']:03d}  {line['speaker'] or config.UNKNOWN_SPEAKER}: {line['clean_text'][:90]}")


def main(argv: list[str] | None = None) -> None:
    """Infer scenes for all episodes, write scenes_inferred.jsonl, and print stats."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--show", help="print one episode with scene breaks, e.g. S4E13")
    parser.add_argument("--limit", type=int, default=80, help="lines to print with --show")
    args = parser.parse_args(argv)
    groups = episodes(list(read_jsonl(config.LINES_JSONL)))
    scenes = [scene for group in groups for scene in detect_scenes(group)]
    write_jsonl(config.SCENES_JSONL, scenes)
    print_stats(scenes, len(groups))
    print(f"Wrote {config.SCENES_JSONL.relative_to(config.ROOT)}")
    if args.show:
        season, episode = map(int, re.findall(r"\d+", args.show))
        group = next(g for g in groups if (g[0]["season"], g[0]["episode"]) == (season, episode))
        print(f"\n{args.show} with inferred scene breaks:")
        show_episode(group, [s for s in scenes if (s["season"], s["episode"]) == (season, episode)], args.limit)


if __name__ == "__main__":
    main()
