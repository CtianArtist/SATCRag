"""Raw CSV -> data/processed/lines.jsonl (one record per dialogue turn) plus a parse report.

Every record keeps its provenance (csv_record, source_row, raw text, raw speaker label) and the
names of the cleaning/normalization rules that touched it. The raw file must be the exact
version the eval targets were written against (its SHA-256 is pinned in the configuration).
"""

import logging
import statistics
from collections import Counter, defaultdict
from typing import Any

from sexandrag.artifacts import file_sha256
from sexandrag.clean import clean_row_text
from sexandrag.config import CorpusConfig, PathsConfig
from sexandrag.errors import CorpusChecksumError, CorpusError, CorpusFormatError
from sexandrag.jsonl import write_json, write_jsonl
from sexandrag.load import RawRow, describe, load_rows
from sexandrag.speakers import AliasMap, build_case_map, label_counts, load_aliases, normalize_speaker

log = logging.getLogger(__name__)
SPLIT_UNKNOWN_RULE = "split_turn_speaker_unknown"


def line_id(season: int, episode: int, source_row: int, part: int) -> str:
    """Stable id for one line record, e.g. 's04e13-r017-0'."""
    return f"s{season:02d}e{episode:02d}-r{source_row:03d}-{part}"


def located(row: RawRow) -> tuple[int, int, int]:
    """(season, episode, source_row) of a repaired row; rows without them never reach this point."""
    if row.season is None or row.episode is None or row.source_row is None:
        raise CorpusFormatError(f"CSV record {row.csv_record} has no season, episode or row index after repair")
    return row.season, row.episode, row.source_row


def make_line_records(
    row: RawRow,
    turns: list[tuple[str, list[str]]],
    speaker: str | None,
    speakers: list[str],
    speaker_rules: list[str],
    keep_label_on_first: bool,
) -> list[dict[str, Any]]:
    """Build one output record per cleaned turn of a CSV row."""
    season, episode, source_row = located(row)
    records = []
    for part, (clean_text, text_rules) in enumerate(turns):
        labeled = len(turns) == 1 or (keep_label_on_first and part == 0)
        records.append(
            {
                "line_id": line_id(season, episode, source_row, part),
                "season": season,
                "episode": episode,
                "source_row": source_row,
                "part": part,
                "n_parts": len(turns),
                "csv_record": row.csv_record,
                "speaker_raw": row.speaker_raw,
                "speaker": speaker if labeled else None,
                "speakers": speakers if labeled else [],
                "raw_text": row.text_raw,
                "clean_text": clean_text,
                "text_rules": text_rules,
                "speaker_rules": speaker_rules if labeled else [SPLIT_UNKNOWN_RULE],
                "row_fixes": row.fixes,
            }
        )
    return records


def build_lines(
    rows: list[RawRow], aliases: AliasMap, case_map: dict[str, str], keep_label_on_first: bool
) -> tuple[list[dict[str, Any]], list[RawRow]]:
    """Clean and normalize every row; return (line records in file order, rows with no content)."""
    lines: list[dict[str, Any]] = []
    no_content = []
    for row in rows:
        turns = clean_row_text(row.text_raw)
        if not turns:
            no_content.append(row)
            continue
        season, episode, _ = located(row)
        speaker, speakers, rules = normalize_speaker(row.speaker_raw, season, episode, aliases, case_map)
        lines += make_line_records(row, turns, speaker, speakers, rules, keep_label_on_first)
    return lines, no_content


def rule_summary(lines: list[dict[str, Any]], examples: int = 3) -> dict[str, Any]:
    """Per text rule: how many CSV rows it changed, with a few before/after examples."""
    seen: set[tuple[int, str]] = set()
    summary: defaultdict[str, dict[str, Any]] = defaultdict(lambda: {"rows": 0, "examples": []})
    for line in lines:
        for rule in line["text_rules"]:
            if (line["csv_record"], rule) in seen:
                continue
            seen.add((line["csv_record"], rule))
            entry = summary[rule]
            entry["rows"] += 1
            if len(entry["examples"]) < examples:
                entry["examples"].append({"raw": line["raw_text"], "clean": line["clean_text"]})
    return dict(sorted(summary.items(), key=lambda kv: -kv[1]["rows"]))


def speaker_mappings(lines: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """List every label mapping except pure whitespace fixes; also return the whitespace-only count."""
    mappings: Counter[tuple[str, str, str]] = Counter()
    whitespace_only = 0
    for line in lines:
        rules = line["speaker_rules"]
        if line["part"] > 0 or not rules or SPLIT_UNKNOWN_RULE in rules:
            continue
        if rules == ["whitespace"]:
            whitespace_only += 1
        else:
            mappings[(line["speaker_raw"].strip(), line["speaker"], ", ".join(rules))] += 1
    table = [{"from": f, "to": t, "rules": r, "rows": n} for (f, t, r), n in mappings.most_common()]
    return table, whitespace_only


def split_summary(lines: list[dict[str, Any]], examples: int = 5) -> dict[str, Any]:
    """Count rows split into several turns and show a few of them."""
    split = [line for line in lines if line["n_parts"] > 1]
    rows = {line["csv_record"] for line in split}
    shown: list[dict[str, Any]] = []
    for line in split:
        if line["part"] == 0 and len(shown) < examples:
            parts = [other["clean_text"] for other in split if other["csv_record"] == line["csv_record"]]
            shown.append({"label": line["speaker_raw"], "raw": line["raw_text"], "turns": parts})
    return {"rows_split": len(rows), "line_records": len(split), "examples": shown}


def top_speakers(lines: list[dict[str, Any]], n: int = 20) -> list[tuple[str, int]]:
    """Most frequent speakers by line count (each name in a multi-speaker line counts)."""
    return Counter(name for line in lines for name in line["speakers"]).most_common(n)


def episode_line_counts(lines: list[dict[str, Any]]) -> dict[str, int]:
    """Line records per episode, keyed like 'S4E13', in file order."""
    return dict(Counter(f"S{line['season']}E{line['episode']}" for line in lines))


def build_report(load_report: dict[str, Any], lines: list[dict[str, Any]], no_content: list[RawRow]) -> dict[str, Any]:
    """Assemble the full parse report."""
    per_episode = episode_line_counts(lines)
    mappings, whitespace_only = speaker_mappings(lines)
    counts = list(per_episode.values())
    return {
        "load": load_report,
        "dropped_no_content_rows": [describe(r) for r in no_content],
        "line_records": len(lines),
        "episodes": len(per_episode),
        "lines_per_episode": {
            "min": min(counts),
            "median": statistics.median(counts),
            "max": max(counts),
            "by_episode": per_episode,
        },
        "unattributed_lines": {
            "no_label_in_source": sum(1 for line in lines if line["speaker"] is None and line["n_parts"] == 1),
            "split_turns": sum(1 for line in lines if SPLIT_UNKNOWN_RULE in line["speaker_rules"]),
        },
        "split_turns": split_summary(lines),
        "text_rules": rule_summary(lines),
        "speaker_mappings": mappings,
        "speaker_whitespace_only_rows": whitespace_only,
        "top_speakers": top_speakers(lines),
    }


def format_report(report: dict[str, Any]) -> list[str]:
    """The parse report as readable lines (printed by `sexandrag parse --report`)."""
    load = report["load"]
    lpe, split = report["lines_per_episode"], report["split_turns"]
    out = [
        f"CSV records read: {load['csv_records']}   parse failures: {len(load['parse_failures'])}",
        f"Tuple-encoded rows unpacked: {load['unpacked_tuple_rows']}   "
        f"source_row rebuilt from position: {load['reconstructed_source_rows']}",
        f"Empty padding rows dropped: {load['dropped_empty_rows']}",
        f"Season/episode filled from neighbours: {len(load['filled_season_episode'])}   "
        f"unresolved (dropped): {len(load['unresolved_episode_rows'])}",
        f"Rows with no letters/digits after cleaning (dropped): {len(report['dropped_no_content_rows'])}",
        f"source_row anomalies: {len(load['source_row_anomalies'])}",
        f"Line records: {report['line_records']}   episodes: {report['episodes']}   "
        f"lines/episode min/median/max: {lpe['min']}/{lpe['median']}/{lpe['max']}",
        f"Rows split into turns: {split['rows_split']} -> {split['line_records']} records",
        f"Lines with speaker=None: {report['unattributed_lines']}",
        "Text rules applied (CSV rows changed):",
    ]
    out += [f"  {rule:15s} {entry['rows']:6d}" for rule, entry in report["text_rules"].items()]
    out.append(f"Speaker label mappings (plus {report['speaker_whitespace_only_rows']} whitespace-only fixes):")
    out += [
        f"  {m['from']!r:34s} -> {m['to']!r:24s} {m['rows']:4d}  [{m['rules']}]" for m in report["speaker_mappings"]
    ]
    out.append("Top speakers by line count:")
    out += [f"  {i:2d}. {name:16s} {n}" for i, (name, n) in enumerate(report["top_speakers"], 1)]
    return out


def verify_corpus(paths: PathsConfig, corpus: CorpusConfig, allow_mismatch: bool = False) -> dict[str, Any]:
    """Check the raw CSV's SHA-256 against the pinned version; a mismatch is an error unless allowed."""
    if not paths.raw_csv.is_file():
        raise CorpusError(
            f"raw corpus not found: {paths.raw_csv}",
            recovery="run `sexandrag download` (needs Kaggle access), or set [paths] raw_csv in your config",
        )
    digest = file_sha256(paths.raw_csv)
    matches = digest == corpus.raw_csv_sha256
    if matches:
        log.info("corpus verified: %s (sha256 %s...)", paths.raw_csv.name, digest[:12])
    elif allow_mismatch:
        log.warning(
            "corpus checksum mismatch ALLOWED: %s has sha256 %s, expected %s; source rows and eval targets "
            "will not line up with the frozen benchmark",
            paths.raw_csv.name,
            digest,
            corpus.raw_csv_sha256,
        )
    else:
        raise CorpusChecksumError(
            f"{paths.raw_csv.name} is not the corpus version this project was built from",
            expected=corpus.raw_csv_sha256,
            actual=digest,
            recovery="re-download it with `sexandrag download`; to parse a different file on purpose, pass "
            "--allow-unverified-corpus (eval targets will then not line up)",
        )
    try:
        shown = str(paths.raw_csv.relative_to(paths.root))
    except ValueError:
        shown = str(paths.raw_csv)
    return {"path": shown, "sha256": digest, "matches_expected": matches}


def parse_corpus(paths: PathsConfig, corpus: CorpusConfig, allow_unverified: bool = False) -> dict[str, Any]:
    """Verify, parse and normalize the raw CSV; write lines.jsonl and parse_report.json; return the report."""
    source = verify_corpus(paths, corpus, allow_unverified)
    rows, load_report = load_rows(paths.raw_csv)
    aliases = load_aliases(paths.speaker_aliases)
    case_map = build_case_map(label_counts(row.speaker_raw for row in rows))
    lines, no_content = build_lines(rows, aliases, case_map, corpus.split_turns_keep_label_on_first)
    if not lines:
        raise CorpusFormatError(f"{paths.raw_csv} produced no dialogue lines", recovery="check the input file")
    write_jsonl(paths.lines_jsonl, lines)
    report = {"source": source} | build_report(load_report, lines, no_content)
    write_json(paths.parse_report, report)
    log.info(
        "parsed %d CSV records into %d lines over %d episodes (%d rows split into turns, %d empty rows dropped)",
        load_report["csv_records"],
        len(lines),
        report["episodes"],
        report["split_turns"]["rows_split"],
        load_report["dropped_empty_rows"],
    )
    log.info("wrote %s and %s", paths.lines_jsonl, paths.parse_report)
    return report
