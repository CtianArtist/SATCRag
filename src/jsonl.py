"""Tiny helpers for reading and writing JSON Lines files."""
import json
from pathlib import Path
from typing import Iterable, Iterator


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    """Write one JSON object per line, creating parent folders; return the record count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yield the JSON objects stored one per line in `path`."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
