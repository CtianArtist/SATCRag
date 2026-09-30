"""Small record builders shared by the tests."""

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HEADER = ["", "Season", "Episode", "Speaker", "Line", "date_job"]


def write_csv(path: Path, rows: list[list[str]]) -> Path:
    """Write rows under the dataset's real header and return the path."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path


def make_line(season: int, episode: int, row: int, speaker: str | None, text: str, part: int = 0) -> dict[str, Any]:
    """A minimal line record like the ones in lines.jsonl."""
    return {
        "line_id": f"s{season:02d}e{episode:02d}-r{row:03d}-{part}",
        "season": season,
        "episode": episode,
        "source_row": row,
        "part": part,
        "n_parts": 1,
        "speaker": speaker,
        "speakers": [speaker] if speaker else [],
        "clean_text": text,
    }


def episode_lines(
    season: int, episode: int, speakers_and_texts: Sequence[tuple[str | None, str]]
) -> list[dict[str, Any]]:
    """Consecutive line records for one episode, rows numbered from 0."""
    return [make_line(season, episode, i, s, t) for i, (s, t) in enumerate(speakers_and_texts)]


def make_chunk(
    season: int,
    episode: int,
    index: int,
    start: int,
    end: int,
    text: str,
    speakers: list[str] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """A minimal chunk record like the ones in data/processed/chunks/*.jsonl."""
    return {
        "chunk_id": f"s{season:02d}e{episode:02d}-test-{index:03d}",
        "season": season,
        "episode": episode,
        "episode_title": title,
        "source_row_start": start,
        "source_row_end": end,
        "speakers": speakers or [],
        "text": text,
    }


def chunk_at(season: int, episode: int, start: int, end: int) -> dict[str, Any]:
    """A minimal chunk covering source rows start..end (for relevance and metric tests)."""
    return {"season": season, "episode": episode, "source_row_start": start, "source_row_end": end}
