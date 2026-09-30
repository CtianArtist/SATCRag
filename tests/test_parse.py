"""End-to-end parse of the real CSV: every line must trace back to its source row."""
import ast
import csv

import pytest

from src import config
from src.load import RawRow, load_rows
from src.parse import build_lines, make_line_records
from src.speakers import build_case_map, label_counts, load_aliases

pytestmark = pytest.mark.skipif(not config.RAW_CSV.exists(), reason="raw CSV not present")


@pytest.fixture(scope="module")
def parsed():
    """Parse the real CSV in memory (independent of any lines.jsonl on disk)."""
    rows, report = load_rows(config.RAW_CSV)
    aliases = load_aliases(config.SPEAKER_ALIASES_JSON)
    case_map = build_case_map(label_counts(r.speaker_raw for r in rows))
    lines, no_content = build_lines(rows, aliases, case_map, keep_label_on_first=False)
    return rows, report, lines, no_content


@pytest.fixture(scope="module")
def raw_records():
    """The CSV records exactly as the file stores them."""
    with open(config.RAW_CSV, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))[1:]


def test_every_line_keeps_its_raw_text_and_label(parsed, raw_records):
    _, _, lines, _ = parsed
    for line in lines:
        fields = raw_records[line["csv_record"]]
        speaker, text = fields[3], fields[4]
        if not speaker and not text and fields[0].startswith("("):
            speaker, text = ast.literal_eval(fields[0])
        assert (line["speaker_raw"], line["raw_text"]) == (speaker, text), line["line_id"]


def test_line_order_follows_the_file(parsed):
    _, _, lines, _ = parsed
    keys = [(line["csv_record"], line["part"]) for line in lines]
    assert keys == sorted(keys)


def test_source_rows_increase_within_each_episode(parsed):
    _, _, lines, _ = parsed
    last = {}
    for line in lines:
        key = (line["season"], line["episode"])
        position = (line["source_row"], line["part"])
        assert key not in last or position > last[key], line["line_id"]
        last[key] = position
    assert len(last) == 94


def test_every_csv_record_is_accounted_for(parsed, raw_records):
    rows, report, lines, no_content = parsed
    kept = {line["csv_record"] for line in lines}
    accounted = (len(kept) + report["dropped_empty_rows"] + len(report["unresolved_episode_rows"])
                 + len(no_content) + len(report["parse_failures"]))
    assert accounted == len(raw_records)


def test_line_ids_are_unique(parsed):
    _, _, lines, _ = parsed
    assert len({line["line_id"] for line in lines}) == len(lines)


def test_split_turns_get_no_speaker_but_keep_the_label(parsed):
    _, _, lines, _ = parsed
    split = [line for line in lines if line["n_parts"] > 1]
    assert split, "expected some rows with two turns"
    assert all(line["speaker"] is None and line["speaker_raw"] for line in split)


def test_label_can_be_kept_on_the_first_turn_when_configured():
    row = RawRow(0, "0", 1, 1, "Laney", "- Susan... - Hello.", source_row=0)
    turns = [("Susan...", ["split_turns"]), ("Hello.", ["split_turns"])]
    first, second = make_line_records(row, turns, "Laney", ["Laney"], [], keep_label_on_first=True)
    assert first["speaker"] == "Laney" and second["speaker"] is None
