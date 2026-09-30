"""Episode titles from data/meta/episodes.csv (metadata only; never used for relevance)."""
import csv
from pathlib import Path


def load_episode_titles(path: Path) -> dict[tuple[int, int], str]:
    """Return {(season, episode): title}; an absent file gives an empty dict."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {(int(r["season"]), int(r["episode"])): r["episode_title"] for r in csv.DictReader(f)}
