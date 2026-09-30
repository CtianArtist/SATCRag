"""Reading and writing JSON Lines files, with errors that name the file and line."""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from sexandrag.errors import ArtifactMismatchError, ArtifactMissingError


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write one JSON object per line, atomically (temporary file, then rename); return the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects stored one per line in `path`."""
    if not path.is_file():
        raise ArtifactMissingError(f"{path} does not exist", recovery="build it first (see `sexandrag --help`)")
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArtifactMismatchError(
                    f"{path} line {number} is not valid JSON ({exc.msg})",
                    recovery="the file is corrupt or truncated; rebuild it",
                ) from exc
            if not isinstance(record, dict):
                raise ArtifactMismatchError(f"{path} line {number} is not a JSON object", recovery="rebuild the file")
            yield record


def write_json(path: Path, data: Any) -> None:
    """Write indented JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, what: str) -> Any:
    """Read a JSON file, naming `what` it is in any error."""
    if not path.is_file():
        raise ArtifactMissingError(f"{what} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactMismatchError(f"{what} is not valid JSON: {path} ({exc.msg}, line {exc.lineno})") from exc
