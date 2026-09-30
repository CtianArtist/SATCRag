"""Line-record construction and corpus verification (sexandrag.parse), without the real corpus."""

from dataclasses import replace

import pytest

from sexandrag.config import CorpusConfig, PathsConfig
from sexandrag.errors import CorpusChecksumError, CorpusError
from sexandrag.load import RawRow
from sexandrag.parse import line_id, make_line_records, verify_corpus
from tests.support.builders import write_csv


def test_line_ids_are_stable_and_zero_padded():
    assert line_id(4, 13, 17, 0) == "s04e13-r017-0"


def test_label_can_be_kept_on_the_first_turn_when_configured():
    row = RawRow(0, "0", 1, 1, "Laney", "- Susan... - Hello.", source_row=0)
    turns = [("Susan...", ["split_turns"]), ("Hello.", ["split_turns"])]
    first, second = make_line_records(row, turns, "Laney", ["Laney"], [], keep_label_on_first=True)
    assert first["speaker"] == "Laney"
    assert second["speaker"] is None


def test_by_default_neither_split_turn_inherits_the_label():
    row = RawRow(0, "0", 1, 1, "Laney", "- Susan... - Hello.", source_row=0)
    turns = [("Susan...", ["split_turns"]), ("Hello.", ["split_turns"])]
    records = make_line_records(row, turns, "Laney", ["Laney"], [], keep_label_on_first=False)
    assert [r["speaker"] for r in records] == [None, None]
    assert all(r["speaker_rules"] == ["split_turn_speaker_unknown"] for r in records)


@pytest.fixture
def corpus_paths(tmp_path):
    raw = write_csv(tmp_path / "SATC_all_lines.csv", [["0", "1.0", "1.0", "Carrie", "Hello.", ""]])
    return replace(PathsConfig.under(tmp_path), raw_csv=raw)


def test_a_different_corpus_is_refused_unless_explicitly_allowed(corpus_paths):
    corpus = CorpusConfig(raw_csv_sha256="0" * 64)
    with pytest.raises(CorpusChecksumError) as error:
        verify_corpus(corpus_paths, corpus)
    assert "--allow-unverified-corpus" in str(error.value)
    assert verify_corpus(corpus_paths, corpus, allow_mismatch=True)["matches_expected"] is False


def test_the_pinned_corpus_is_accepted(corpus_paths):
    from sexandrag.artifacts import file_sha256

    corpus = CorpusConfig(raw_csv_sha256=file_sha256(corpus_paths.raw_csv))
    assert verify_corpus(corpus_paths, corpus) == {
        "path": "SATC_all_lines.csv",
        "sha256": corpus.raw_csv_sha256,
        "matches_expected": True,
    }


def test_a_missing_corpus_points_to_the_download_command(tmp_path):
    with pytest.raises(CorpusError, match="sexandrag download"):
        verify_corpus(PathsConfig.under(tmp_path), CorpusConfig())
