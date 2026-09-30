"""Malformed-row parsing and source-row provenance in src/load.py."""
import pytest

from src.load import load_rows, read_records
from tests.helpers import write_csv


def test_tuple_encoded_row_is_unpacked_and_its_source_row_rebuilt(tmp_path):
    path = write_csv(tmp_path / "t.csv", [
        ["0", "6.0", "2.0", "Carrie", "In fact, it was kind of hot.", ""],
        ["('Carrie', \"In a single gal's life.\")", "6.0", "3.0", "", "", ""],
        ["('Jack', 'Come on in.')", "6.0", "3.0", "", "", ""],
    ])
    rows, report = load_rows(path)
    jack = rows[2]
    assert (jack.speaker_raw, jack.text_raw) == ("Jack", "Come on in.")
    assert (jack.season, jack.episode, jack.source_row) == (6, 3, 1)
    assert jack.fixes == ["unpacked_tuple_row", "reconstructed_source_row"]
    assert report["unpacked_tuple_rows"] == 2


def test_unparseable_tuple_is_reported_not_raised(tmp_path):
    path = write_csv(tmp_path / "t.csv", [
        ["0", "1.0", "1.0", "Carrie", "Hello.", ""],
        ["('Carrie', broken", "1.0", "1.0", "", "", ""],
    ])
    rows, report = load_rows(path)
    assert len(rows) == 1
    assert report["parse_failures"][0]["csv_record"] == 1


def test_truly_empty_padding_rows_are_dropped(tmp_path):
    path = write_csv(tmp_path / "t.csv", [
        ["0", "1.0", "4.0", "Carrie", "Damn.", ""],
        ["1", "", "", "", "", ""],
        ["2", "", "", " ", "", ""],
        ["0", "1.0", "5.0", "Carrie", "The most powerful woman.", ""],
    ])
    rows, report = load_rows(path)
    assert [r.text_raw for r in rows] == ["Damn.", "The most powerful woman."]
    assert report["dropped_empty_rows"] == 2
    assert report["dropped_empty_csv_records"] == ["1-2"]


def test_missing_episode_is_filled_when_both_neighbours_agree(tmp_path):
    path = write_csv(tmp_path / "t.csv", [
        ["38", "1.0", "2.0", "Yvette", "Charlie Sheen?", ""],
        ["39", "1.0", "", "Carrie", "They'd come to dinner.", ""],
        ["40", "1.0", "2.0", "Nick", "Veronica Lake.", ""],
    ])
    rows, report = load_rows(path)
    assert (rows[1].season, rows[1].episode) == (1, 2)
    assert rows[1].fixes == ["filled_season_episode"]
    assert len(report["filled_season_episode"]) == 1


def test_missing_episode_is_not_guessed_at_an_episode_boundary(tmp_path):
    path = write_csv(tmp_path / "t.csv", [
        ["376", "2.0", "2.0", "Carrie", "Big was.", ""],
        ["377", "", "", "", "Stray text", ""],
        ["0", "2.0", "3.0", "Carrie", "Manhattan.", ""],
    ])
    rows, report = load_rows(path)
    assert [r.text_raw for r in rows] == ["Big was.", "Manhattan."]
    assert report["unresolved_episode_rows"][0]["csv_record"] == 1


def test_present_value_must_agree_with_neighbours(tmp_path):
    path = write_csv(tmp_path / "t.csv", [
        ["0", "1.0", "2.0", "A", "one", ""],
        ["1", "3.0", "", "B", "two", ""],          # season 3 contradicts the neighbours' season 1
        ["2", "1.0", "2.0", "C", "three", ""],
    ])
    rows, report = load_rows(path)
    assert len(rows) == 2 and len(report["unresolved_episode_rows"]) == 1


def test_quoted_field_with_comma_and_line_break_stays_one_record(tmp_path):
    path = write_csv(tmp_path / "t.csv", [
        ["0", "1.0", "1.0", "Carrie", "Well, I --\nnever mind.", ""],
        ["1", "1.0", "1.0", "Big", "Okay.", ""],
    ])
    rows, _ = load_rows(path)
    assert len(rows) == 2 and rows[0].text_raw == "Well, I --\nnever mind."


def test_source_row_is_the_csv_index_column(tmp_path):
    path = write_csv(tmp_path / "t.csv", [
        ["17", "4.0", "13.0", "Carrie", "Hello, Petey.", ""],
        ["18", "4.0", "13.0", "Carrie", "A plant!", ""],
    ])
    rows, report = load_rows(path)
    assert [(r.source_row, r.csv_record) for r in rows] == [(17, 0), (18, 1)]
    assert report["source_row_anomalies"] == []


def test_unexpected_header_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_records(path)
