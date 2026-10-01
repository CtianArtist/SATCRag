"""Invariants over the real SATC corpus and its local artifacts (skipped when the corpus is not present)."""

import ast
import csv
import json

import pytest

from satc_rag.artifacts import chunk_configs, file_sha256
from satc_rag.chunk import load_chunk_set
from satc_rag.config import Settings
from satc_rag.evaluation.validate import by_episode, validate_raw_items
from satc_rag.jsonl import read_jsonl
from satc_rag.load import load_rows
from satc_rag.parse import build_lines
from satc_rag.speakers import build_case_map, label_counts, load_aliases
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.corpus
SETTINGS = Settings.defaults(REPO_ROOT)


@pytest.fixture(scope="module")
def parsed():
    """Parse the real CSV in memory (independent of any lines.jsonl on disk)."""
    rows, report = load_rows(SETTINGS.paths.raw_csv)
    aliases = load_aliases(SETTINGS.paths.speaker_aliases)
    case_map = build_case_map(label_counts(r.speaker_raw for r in rows))
    lines, no_content = build_lines(rows, aliases, case_map, keep_label_on_first=False)
    return rows, report, lines, no_content


@pytest.fixture(scope="module")
def raw_records():
    with SETTINGS.paths.raw_csv.open(newline="", encoding="utf-8") as f:
        return list(csv.reader(f))[1:]


def test_the_corpus_is_the_pinned_version():
    assert file_sha256(SETTINGS.paths.raw_csv) == SETTINGS.corpus.raw_csv_sha256


def test_every_line_keeps_its_raw_text_and_label(parsed, raw_records):
    for line in parsed[2]:
        fields = raw_records[line["csv_record"]]
        speaker, text = fields[3], fields[4]
        if not speaker and not text and fields[0].startswith("("):
            speaker, text = ast.literal_eval(fields[0])
        assert (line["speaker_raw"], line["raw_text"]) == (speaker, text), line["line_id"]


def test_line_order_and_source_rows_follow_the_file(parsed):
    lines = parsed[2]
    assert [(line["csv_record"], line["part"]) for line in lines] == sorted(
        (line["csv_record"], line["part"]) for line in lines
    )
    last: dict[tuple[int, int], tuple[int, int]] = {}
    for line in lines:
        key, position = (line["season"], line["episode"]), (line["source_row"], line["part"])
        assert key not in last or position > last[key], line["line_id"]
        last[key] = position
    assert len(last) == 94


def test_every_csv_record_is_accounted_for(parsed, raw_records):
    _, report, lines, no_content = parsed
    kept = {line["csv_record"] for line in lines}
    accounted = (
        len(kept)
        + report["dropped_empty_rows"]
        + len(report["unresolved_episode_rows"])
        + len(no_content)
        + len(report["parse_failures"])
    )
    assert accounted == len(raw_records)
    assert len({line["line_id"] for line in lines}) == len(lines)


def test_the_parsed_lines_on_disk_match_a_fresh_parse(parsed):
    on_disk = list(read_jsonl(SETTINGS.paths.lines_jsonl))
    assert on_disk == parsed[2]


def test_the_committed_chunk_manifests_match_the_local_chunk_sets():
    spec = SETTINGS.model.spec
    for config in chunk_configs(SETTINGS.chunking, spec.repo_id, spec.revision):
        chunk_set = load_chunk_set(config, SETTINGS.paths.chunks_dir, SETTINGS.paths.lines_jsonl)
        assert chunk_set.manifest["lines_sha256"] == file_sha256(SETTINGS.paths.lines_jsonl)


@pytest.mark.parametrize("name", ["eval/frozen/benchmark.json", "eval/smoke.json"])
def test_every_quote_still_lies_inside_its_span(name):
    episodes = by_episode(list(read_jsonl(SETTINGS.paths.lines_jsonl)))
    raw_items = json.loads((REPO_ROOT / name).read_text(encoding="utf-8"))
    assert validate_raw_items(raw_items, episodes) == {}
