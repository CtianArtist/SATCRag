"""Small builders shared by the tests."""
import csv
from pathlib import Path

HEADER = ["", "Season", "Episode", "Speaker", "Line", "date_job"]


def write_csv(path: Path, rows: list[list[str]]) -> Path:
    """Write rows under the dataset's real header and return the path."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path


def make_line(season: int, episode: int, row: int, speaker: str | None, text: str, part: int = 0) -> dict:
    """A minimal line record like the ones in lines.jsonl."""
    return {
        "line_id": f"s{season:02d}e{episode:02d}-r{row:03d}-{part}",
        "season": season, "episode": episode, "source_row": row, "part": part, "n_parts": 1,
        "speaker": speaker, "speakers": [speaker] if speaker else [], "clean_text": text,
    }


def episode_lines(season: int, episode: int, speakers_and_texts: list[tuple[str | None, str]]) -> list[dict]:
    """Consecutive line records for one episode, rows numbered from 0."""
    return [make_line(season, episode, i, s, t) for i, (s, t) in enumerate(speakers_and_texts)]
