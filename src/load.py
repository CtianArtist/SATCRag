"""Read the raw CSV into ordered rows and repair row-level defects (no text changes here).

Repairs, each recorded in RawRow.fixes and in the load report:
  * S6E3 stores (speaker, line) as a Python tuple literal in the index column -> unpacked.
  * Truly empty padding rows (no speaker, no text) -> dropped.
  * Blank season/episode -> filled only when the nearest complete rows on both sides agree.
  * source_row is the CSV's own per-episode index column. Where that column holds a
    tuple instead of a number (S6E3) it is rebuilt from the row's position in the episode.
"""
import ast
import csv
from dataclasses import dataclass, field
from pathlib import Path

EXPECTED_HEADER = ["", "Season", "Episode", "Speaker", "Line", "date_job"]


@dataclass
class RawRow:
    """One CSV record after row-level repairs, before any text or speaker cleaning."""
    csv_record: int            # 0-based record position in the CSV, header excluded
    index_field: str           # the CSV's unnamed first column, verbatim
    season: int | None
    episode: int | None
    speaker_raw: str
    text_raw: str
    source_row: int | None = None
    fixes: list[str] = field(default_factory=list)


def read_records(path: Path) -> list[list[str]]:
    """Read all data records with the csv module, which handles quoted commas and line breaks."""
    with open(path, newline="", encoding="utf-8") as f:
        records = list(csv.reader(f))
    if not records or records[0] != EXPECTED_HEADER:
        raise ValueError(f"unexpected CSV header: {records[0] if records else None!r}")
    return records[1:]


def parse_number(value: str) -> int | None:
    """Turn '2.0' or '2' into 2 and a blank field into None."""
    value = value.strip()
    return int(float(value)) if value else None


def unpack_tuple_field(value: str) -> tuple[str, str] | None:
    """Return (speaker, text) when `value` is a literal like "('Jack', 'Come on in.')", else None."""
    if not value.lstrip().startswith("("):
        return None
    parsed = ast.literal_eval(value)
    if not (isinstance(parsed, tuple) and len(parsed) == 2 and all(isinstance(x, str) for x in parsed)):
        raise ValueError(f"index column is not a (speaker, line) tuple: {value[:60]!r}")
    return parsed


def row_from_record(csv_record: int, fields: list[str]) -> RawRow:
    """Build a RawRow from one CSV record, unpacking a tuple-encoded row if needed."""
    index_field, season, episode, speaker, text = fields[:5]
    row = RawRow(csv_record, index_field, parse_number(season), parse_number(episode), speaker, text)
    unpacked = unpack_tuple_field(index_field)
    if unpacked and not speaker.strip() and not text.strip():
        row.speaker_raw, row.text_raw = unpacked
        row.fixes.append("unpacked_tuple_row")
    return row


def rows_from_records(records: list[list[str]]) -> tuple[list[RawRow], list[dict]]:
    """Convert records to RawRows, collecting unparseable records instead of crashing."""
    rows, failures = [], []
    for i, fields in enumerate(records):
        try:
            rows.append(row_from_record(i, fields))
        except (ValueError, SyntaxError, IndexError) as exc:
            failures.append({"csv_record": i, "error": str(exc), "fields": fields[:5]})
    return rows, failures


def is_empty(row: RawRow) -> bool:
    """True when a row has neither a speaker nor any text (end-of-episode padding)."""
    return not row.speaker_raw.strip() and not row.text_raw.strip()


def nearest_episode(rows: list[RawRow], k: int, step: int) -> tuple[int, int] | None:
    """Return (season, episode) of the nearest row from index k (moving by step) that has both."""
    j = k + step
    while 0 <= j < len(rows):
        if rows[j].season is not None and rows[j].episode is not None:
            return rows[j].season, rows[j].episode
        j += step
    return None


def agreed_episode(rows: list[RawRow], k: int) -> tuple[int, int] | None:
    """Return the episode that row k's neighbours on both sides agree on, or None."""
    before, after = nearest_episode(rows, k, -1), nearest_episode(rows, k, +1)
    return before if before is not None and before == after else None


def fill_missing_episode_ids(rows: list[RawRow]) -> tuple[list[RawRow], list[RawRow]]:
    """Fill blank season/episode where both neighbours agree; return (resolved, unresolved)."""
    resolved, unresolved = [], []
    for k, row in enumerate(rows):
        if row.season is not None and row.episode is not None:
            resolved.append(row)
            continue
        guess = agreed_episode(rows, k)
        if guess and row.season in (None, guess[0]) and row.episode in (None, guess[1]):
            row.season, row.episode = guess
            row.fixes.append("filled_season_episode")
            resolved.append(row)
        else:
            unresolved.append(row)
    return resolved, unresolved


def assign_source_rows(rows: list[RawRow]) -> None:
    """Set source_row from the index column, rebuilding it from row position where it isn't a number."""
    first_record = {}
    for row in rows:
        first_record.setdefault((row.season, row.episode), row.csv_record)
    for row in rows:
        if row.index_field.strip().isdigit():
            row.source_row = int(row.index_field)
        else:
            row.source_row = row.csv_record - first_record[(row.season, row.episode)]
            row.fixes.append("reconstructed_source_row")


def source_row_anomalies(rows: list[RawRow]) -> list[dict]:
    """List rows whose source_row does not increase within its episode, or episodes out of order."""
    anomalies, last_row, last_episode = [], {}, None
    for row in rows:
        key = (row.season, row.episode)
        if key in last_row and row.source_row <= last_row[key]:
            anomalies.append({"csv_record": row.csv_record, "issue": "source_row not increasing"})
        if last_episode is not None and key != last_episode and key in last_row:
            anomalies.append({"csv_record": row.csv_record, "issue": "episode block split"})
        last_row[key], last_episode = row.source_row, key
    return anomalies


def to_ranges(numbers: list[int]) -> list[str]:
    """Compress sorted integers into readable ranges: [1, 2, 3, 7] -> ['1-3', '7']."""
    ranges, start = [], None
    for i, n in enumerate(numbers):
        if start is None:
            start = n
        if i + 1 == len(numbers) or numbers[i + 1] != n + 1:
            ranges.append(str(start) if start == n else f"{start}-{n}")
            start = None
    return ranges


def describe(row: RawRow) -> dict:
    """Small dict describing a row for the report."""
    return {"csv_record": row.csv_record, "season": row.season, "episode": row.episode,
            "speaker_raw": row.speaker_raw, "text_raw": row.text_raw}


def load_rows(path: Path) -> tuple[list[RawRow], dict]:
    """Read the CSV and apply all row-level repairs; return (rows in file order, load report)."""
    records = read_records(path)
    rows, failures = rows_from_records(records)
    empty = [r for r in rows if is_empty(r)]
    rows = [r for r in rows if not is_empty(r)]
    rows, unresolved = fill_missing_episode_ids(rows)
    assign_source_rows(rows)
    report = {
        "csv_records": len(records),
        "parse_failures": failures,
        "unpacked_tuple_rows": sum("unpacked_tuple_row" in r.fixes for r in rows),
        "reconstructed_source_rows": sum("reconstructed_source_row" in r.fixes for r in rows),
        "dropped_empty_rows": len(empty),
        "dropped_empty_csv_records": to_ranges([r.csv_record for r in empty]),
        "filled_season_episode": [describe(r) for r in rows if "filled_season_episode" in r.fixes],
        "unresolved_episode_rows": [describe(r) for r in unresolved],
        "source_row_anomalies": source_row_anomalies(rows),
    }
    return rows, report
