"""Episode titles from data/meta/episodes.csv (metadata only; never used for relevance)."""

import csv
from pathlib import Path

from sexandrag.errors import MetadataError

REQUIRED_COLUMNS = ("season", "episode", "episode_title")


def load_episode_titles(path: Path) -> dict[tuple[int, int], str]:
    """Return {(season, episode): title}, rejecting a missing file, bad columns or duplicates."""
    if not path.is_file():
        raise MetadataError(f"episode metadata not found: {path}", recovery="restore data/meta/ from the repository")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not set(REQUIRED_COLUMNS) <= set(reader.fieldnames):
            raise MetadataError(f"unexpected columns in {path}", expected=REQUIRED_COLUMNS, actual=reader.fieldnames)
        titles: dict[tuple[int, int], str] = {}
        for number, row in enumerate(reader, 2):
            try:
                key = (int(row["season"]), int(row["episode"]))
            except (TypeError, ValueError) as exc:
                raise MetadataError(f"{path} line {number}: season and episode must be integers") from exc
            if key in titles:
                raise MetadataError(f"{path} line {number}: duplicate entry for S{key[0]}E{key[1]}")
            titles[key] = row["episode_title"]
    return titles
