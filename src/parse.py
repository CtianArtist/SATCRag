"""Raw CSV -> data/processed/lines.jsonl (one record per dialogue turn) plus a parse report.

Run: python -m src.parse
Every record keeps its provenance (csv_record, source_row, raw text, raw speaker label)
and the names of the cleaning/normalization rules that touched it.
"""
import json
import statistics
from collections import Counter, defaultdict

from src import config
from src.artifacts import file_sha256
from src.clean import clean_row_text
from src.jsonl import write_jsonl
from src.load import RawRow, describe, load_rows
from src.speakers import build_case_map, label_counts, load_aliases, normalize_speaker

SPLIT_UNKNOWN_RULE = "split_turn_speaker_unknown"


def line_id(season: int, episode: int, source_row: int, part: int) -> str:
    """Stable id for one line record, e.g. 's04e13-r017-0'."""
    return f"s{season:02d}e{episode:02d}-r{source_row:03d}-{part}"


def make_line_records(row: RawRow, turns: list, speaker: str | None, speakers: list[str],
                      speaker_rules: list[str], keep_label_on_first: bool) -> list[dict]:
    """Build one output record per cleaned turn of a CSV row."""
    records = []
    for part, (clean_text, text_rules) in enumerate(turns):
        labeled = len(turns) == 1 or (keep_label_on_first and part == 0)
        records.append({
            "line_id": line_id(row.season, row.episode, row.source_row, part),
            "season": row.season,
            "episode": row.episode,
            "source_row": row.source_row,
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
        })
    return records


def build_lines(rows: list[RawRow], aliases, case_map: dict,
                keep_label_on_first: bool) -> tuple[list[dict], list[RawRow]]:
    """Clean and normalize every row; return (line records in file order, rows with no content)."""
    lines, no_content = [], []
    for row in rows:
        turns = clean_row_text(row.text_raw)
        if not turns:
            no_content.append(row)
            continue
        speaker, speakers, rules = normalize_speaker(row.speaker_raw, row.season, row.episode,
                                                     aliases, case_map)
        lines += make_line_records(row, turns, speaker, speakers, rules, keep_label_on_first)
    return lines, no_content


def rule_summary(lines: list[dict], examples: int = 3) -> dict:
    """Per text rule: how many CSV rows it changed, with a few before/after examples."""
    seen, summary = set(), defaultdict(lambda: {"rows": 0, "examples": []})
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


def speaker_mappings(lines: list[dict]) -> tuple[list[dict], int]:
    """List every label mapping except pure whitespace fixes; also return the whitespace-only count."""
    mappings, whitespace_only = Counter(), 0
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


def split_summary(lines: list[dict], examples: int = 5) -> dict:
    """Count rows split into several turns and show a few of them."""
    split = [line for line in lines if line["n_parts"] > 1]
    rows = {line["csv_record"] for line in split}
    shown = []
    for line in split:
        if line["part"] == 0 and len(shown) < examples:
            parts = [l["clean_text"] for l in split if l["csv_record"] == line["csv_record"]]
            shown.append({"label": line["speaker_raw"], "raw": line["raw_text"], "turns": parts})
    return {"rows_split": len(rows), "line_records": len(split), "examples": shown}


def top_speakers(lines: list[dict], n: int = 20) -> list[tuple[str, int]]:
    """Most frequent speakers by line count (each name in a multi-speaker line counts)."""
    return Counter(name for line in lines for name in line["speakers"]).most_common(n)


def episode_line_counts(lines: list[dict]) -> dict[str, int]:
    """Line records per episode, keyed like 'S4E13', in file order."""
    return dict(Counter(f"S{l['season']}E{l['episode']}" for l in lines))


def build_report(load_report: dict, lines: list[dict], no_content: list[RawRow]) -> dict:
    """Assemble the full parse report."""
    per_episode = episode_line_counts(lines)
    mappings, whitespace_only = speaker_mappings(lines)
    counts = list(per_episode.values())
    return {
        "load": load_report,
        "dropped_no_content_rows": [describe(r) for r in no_content],
        "line_records": len(lines),
        "episodes": len(per_episode),
        "lines_per_episode": {"min": min(counts), "median": statistics.median(counts),
                              "max": max(counts), "by_episode": per_episode},
        "unattributed_lines": {
            "no_label_in_source": sum(1 for l in lines if l["speaker"] is None and l["n_parts"] == 1),
            "split_turns": sum(1 for l in lines if SPLIT_UNKNOWN_RULE in l["speaker_rules"]),
        },
        "split_turns": split_summary(lines),
        "text_rules": rule_summary(lines),
        "speaker_mappings": mappings,
        "speaker_whitespace_only_rows": whitespace_only,
        "top_speakers": top_speakers(lines),
    }


def print_report(report: dict) -> None:
    """Print a readable summary of the parse report."""
    load = report["load"]
    print("=== PARSE REPORT ===")
    print(f"CSV records read: {load['csv_records']}   parse failures: {len(load['parse_failures'])}")
    print(f"Tuple-encoded rows unpacked (S6E3): {load['unpacked_tuple_rows']}"
          f"   source_row rebuilt from position: {load['reconstructed_source_rows']}")
    print(f"Empty padding rows dropped: {load['dropped_empty_rows']}")
    print(f"Season/episode filled from neighbours: {len(load['filled_season_episode'])}"
          f"   unresolved (dropped): {len(load['unresolved_episode_rows'])}")
    print(f"Rows with no letters/digits after cleaning (dropped): {len(report['dropped_no_content_rows'])}")
    print(f"source_row anomalies: {len(load['source_row_anomalies'])}")
    lpe = report["lines_per_episode"]
    print(f"\nLine records: {report['line_records']}   episodes: {report['episodes']}"
          f"   lines/episode min/median/max: {lpe['min']}/{lpe['median']}/{lpe['max']}")
    split = report["split_turns"]
    print(f"Rows split into turns: {split['rows_split']} -> {split['line_records']} records "
          f"(speaker=None unless SPLIT_TURNS_KEEP_LABEL_ON_FIRST)")
    print(f"Lines with speaker=None: {report['unattributed_lines']}")
    print("\nText rules applied (CSV rows changed):")
    for rule, entry in report["text_rules"].items():
        example = entry["examples"][0]
        print(f"  {rule:15s} {entry['rows']:6d}   e.g. {example['raw'][:48]!r} -> {example['clean'][:48]!r}")
    print(f"\nSpeaker label mappings (whitespace-only fixes on {report['speaker_whitespace_only_rows']} "
          f"more rows not listed):")
    for m in report["speaker_mappings"]:
        print(f"  {m['from']!r:34s} -> {m['to']!r:24s} {m['rows']:4d}  [{m['rules']}]")
    print("\nTop 20 speakers by line count:")
    for i, (name, n) in enumerate(report["top_speakers"], 1):
        print(f"  {i:2d}. {name:16s} {n}")


def check_source_file() -> dict:
    """Compare the raw CSV's SHA-256 with the version this project was built from; warn on mismatch."""
    digest = file_sha256(config.RAW_CSV)
    matches = digest == config.RAW_CSV_SHA256
    if not matches:
        print(f"WARNING: {config.RAW_CSV.name} SHA-256 {digest} differs from the expected "
              f"{config.RAW_CSV_SHA256}; source rows and eval targets may not line up.\n")
    return {"path": str(config.RAW_CSV.relative_to(config.ROOT)), "sha256": digest, "matches_expected": matches}


def main() -> None:
    """Parse the raw CSV, write lines.jsonl and parse_report.json, and print the report."""
    source = check_source_file()
    rows, load_report = load_rows(config.RAW_CSV)
    aliases = load_aliases(config.SPEAKER_ALIASES_JSON)
    case_map = build_case_map(label_counts(row.speaker_raw for row in rows))
    lines, no_content = build_lines(rows, aliases, case_map, config.SPLIT_TURNS_KEEP_LABEL_ON_FIRST)
    write_jsonl(config.LINES_JSONL, lines)
    report = {"source": source} | build_report(load_report, lines, no_content)
    config.PARSE_REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_report(report)
    print(f"\nWrote {len(lines)} lines to {config.LINES_JSONL.relative_to(config.ROOT)}"
          f" and the full report to {config.PARSE_REPORT_JSON.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
