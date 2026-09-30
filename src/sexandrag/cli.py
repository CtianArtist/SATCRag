"""The `sexandrag` command line.

    sexandrag verify                         check corpus, metadata, lines, chunks, indexes, model, benchmark
    sexandrag download                       fetch the corpus from Kaggle and verify its checksum
    sexandrag model download                 fetch the pinned embedding model (the only networked model step)
    sexandrag parse                          raw CSV -> data/processed/lines.jsonl
    sexandrag chunk --all                    build every configured chunk set
    sexandrag index --all                    build (or confirm) the dense indexes
    sexandrag search "question" --method hybrid
    sexandrag eval                           the benchmark on the DEVELOPMENT set (the default)
    sexandrag inspect s04e13-tok512o64-003   show one chunk with its provenance

Global options come before the command: --root, --config, -v/--verbose, -q/--quiet, --debug.
Expected problems print a short explanation with a fix and exit 1; --debug adds the traceback.
Only `download` and `model download` may use the network.
"""

import argparse
import json
import logging
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import sexandrag
from sexandrag.artifacts import ChunkConfig, chunk_config_for_size, chunk_configs, custom_chunk_config, file_sha256
from sexandrag.chunk import build_chunk_set, load_chunk_set
from sexandrag.config import Settings, load_settings
from sexandrag.download import download_corpus
from sexandrag.episodes import load_episode_titles
from sexandrag.errors import ArtifactMissingError, ConfigurationError, SexAndRagError
from sexandrag.evaluation.runner import EvalRequest, format_summary, run_evaluation
from sexandrag.evaluation.schema import parse_items
from sexandrag.evaluation.split import freeze
from sexandrag.evaluation.validate import by_episode, distribution, main_characters, review_sheet, validate_raw_items
from sexandrag.evaluation.validate import write_review as save_review
from sexandrag.index import build_dense_index, cached_indexes, index_location, load_index, rebuild_hint
from sexandrag.jsonl import read_jsonl, write_jsonl
from sexandrag.logs import configure_logging
from sexandrag.model import download_snapshot, snapshot_path, tokenizer_only_files, verify_snapshot
from sexandrag.parse import format_report, parse_corpus
from sexandrag.provenance import find_quote
from sexandrag.retrieve import METHODS, Filters, SearchResult, build_retrievers
from sexandrag.scenes import detect_with, episodes, log_stats, show_episode
from sexandrag.services import PinnedModelServices, Services
from sexandrag.verify import COMPONENTS, run_checks

log = logging.getLogger("sexandrag.cli")
EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_INTERNAL, EXIT_INTERRUPTED = 0, 1, 2, 70, 130
CHUNK_ID_RE = re.compile(r"^s(\d{2})e(\d{2})-tok(\d+)o(\d+)-(\d{3})$")
NETWORK_COMMANDS = {("download", None), ("model", "download")}


def out(text: str = "") -> None:
    """Print a command result to stdout (diagnostics go to stderr through logging)."""
    print(text)


# ----------------------------------------------------------------------------- commands


def cmd_config(args: argparse.Namespace, settings: Settings, services: Services) -> int:
    """Print the effective, validated configuration."""
    out(json.dumps(settings.as_dict(), indent=2))
    return EXIT_OK


def cmd_download(args: argparse.Namespace, settings: Settings, services: Services) -> int:
    """Fetch the corpus from Kaggle and verify its checksum."""
    path = download_corpus(settings.paths, settings.corpus, force=args.force)
    out(f"corpus ready: {path}")
    return EXIT_OK


def cmd_model(args: argparse.Namespace, settings: Settings, services: Services) -> int:
    """Download, verify or describe the pinned embedding model."""
    spec, cache = settings.model.spec, settings.paths.model_cache_dir
    if args.model_command == "download":
        files = tokenizer_only_files(spec) if args.tokenizer_only else None
        path = download_snapshot(spec, cache, tokenizer_only=args.tokenizer_only)
        verify_snapshot(path, spec, deep=not args.tokenizer_only, files=files)
        out(f"model ready: {path}")
    elif args.model_command == "verify":
        verify_snapshot(snapshot_path(spec, cache), spec, deep=args.deep)
        out(f"{spec.repo_id}@{spec.revision}: verified" + (" (deep)" if args.deep else ""))
    else:
        out(
            json.dumps(
                {"identity": spec.identity(), "modules": list(spec.modules), "files": dict(spec.files)}, indent=2
            )
        )
    return EXIT_OK


def cmd_parse(args: argparse.Namespace, settings: Settings, services: Services) -> int:
    """Parse the raw CSV into lines.jsonl and the parse report."""
    report = parse_corpus(settings.paths, settings.corpus, allow_unverified=args.allow_unverified_corpus)
    if args.report:
        out("\n".join(format_report(report)))
    out(f"wrote {report['line_records']} lines to {settings.paths.lines_jsonl}")
    return EXIT_OK


def cmd_scenes(args: argparse.Namespace, settings: Settings, services: Services) -> int:
    """Infer optional scene metadata for every episode."""
    groups = episodes(list(read_jsonl(settings.paths.lines_jsonl)))
    scenes = [scene for group in groups for scene in detect_with(group, settings.scenes)]
    write_jsonl(settings.paths.scenes_jsonl, scenes)
    log_stats(scenes, len(groups))
    if args.show:
        season, episode = parse_episode(args.show)
        group = next((g for g in groups if (g[0]["season"], g[0]["episode"]) == (season, episode)), None)
        if group is None:
            raise ArtifactMissingError(f"no episode {args.show} in lines.jsonl")
        chosen = [s for s in scenes if (s["season"], s["episode"]) == (season, episode)]
        out("\n".join(show_episode(group, chosen, args.limit, settings.corpus.unknown_speaker)))
    out(f"wrote {len(scenes)} inferred scenes to {settings.paths.scenes_jsonl}")
    return EXIT_OK


def requested_chunk_configs(args: argparse.Namespace, settings: Settings, services: Services) -> list[ChunkConfig]:
    """The chunk settings a command asked for: --all, a configured --size, or a custom --size/--overlap."""
    tokenizer, revision = services.tokenizer_id(settings)
    if getattr(args, "all", False):
        return chunk_configs(settings.chunking, tokenizer, revision)
    size = args.size or settings.chunking.size
    overlap = getattr(args, "overlap", None)
    if overlap is not None:
        return [custom_chunk_config(size, overlap, settings.model.spec.max_tokens, tokenizer, revision)]
    return [chunk_config_for_size(size, settings.chunking, tokenizer, revision)]


def cmd_chunk(args: argparse.Namespace, settings: Settings, services: Services) -> int:
    """Build chunk sets (and their manifests) from lines.jsonl."""
    paths = settings.paths
    configs = requested_chunk_configs(args, settings, services)
    lines = list(read_jsonl(paths.lines_jsonl))
    titles = load_episode_titles(paths.episodes_csv)
    counter = services.token_counter(settings)
    scenes = settings.scenes if settings.scenes.enabled and not args.no_scenes else None
    for config in configs:
        chunk_set = build_chunk_set(
            config, lines, counter, titles, scenes, paths.chunks_dir, paths.lines_jsonl, settings.corpus.unknown_speaker
        )
        out(f"{config.config_id}: {len(chunk_set.chunks)} chunks -> {config.jsonl_path(paths.chunks_dir)}")
    return EXIT_OK


def cmd_index(args: argparse.Namespace, settings: Settings, services: Services) -> int:
    """Build missing dense indexes (or all of them with --rebuild); load the model only if needed."""
    paths = settings.paths
    if args.list:
        for summary in cached_indexes(paths.index_dir):
            out(json.dumps(summary))
        return EXIT_OK
    identity = services.embedder_identity(settings)
    for config in requested_chunk_configs(args, settings, services):
        chunk_set = load_chunk_set(config, paths.chunks_dir, paths.lines_jsonl)
        key, directory = index_location(chunk_set, identity, paths.index_dir)
        if directory.is_dir() and not args.rebuild:
            ids = [chunk["chunk_id"] for chunk in chunk_set.chunks]
            load_index(directory, key, ids, identity["dim"], rebuild_hint(config))
            out(f"{config.config_id}: cached and valid ({directory})")
            continue
        index = build_dense_index(chunk_set, services.embedder(settings), paths.index_dir, rebuild=args.rebuild)
        out(f"{config.config_id}: built {index.vectors.shape[0]} vectors ({index.path})")
    return EXIT_OK


def print_result(result: SearchResult, full: bool, preview_lines: int = 6) -> None:
    """Print one search result: rank, score, provenance, then its lines."""
    found_by = ", ".join(f"{method} #{rank}" for method, rank in result.components.items())
    out(
        f"\n#{result.rank}  {result.method} score {result.score:.4f}  {result.chunk_id}  "
        f"S{result.season}E{result.episode} {result.episode_title or ''!r}  "
        f"rows {result.source_row_start}-{result.source_row_end}" + (f"  [{found_by}]" if found_by else "")
    )
    out(f"    speakers: {', '.join(result.speakers)}")
    lines = result.text.splitlines()
    for line in lines if full else lines[:preview_lines]:
        out(f"    {line}")
    if not full and len(lines) > preview_lines:
        out(f"    ... ({len(lines) - preview_lines} more lines; --full shows all)")


def cmd_search(args: argparse.Namespace, settings: Settings, services: Services) -> int:
    """Run one retriever on one chunk set and print the ranked chunks."""
    paths = settings.paths
    config = requested_chunk_configs(args, settings, services)[0]
    chunk_set = load_chunk_set(config, paths.chunks_dir, paths.lines_jsonl)
    embedder = services.embedder(settings) if args.method in ("dense", "hybrid") else None
    retriever = build_retrievers(chunk_set, (args.method,), embedder, paths.index_dir, settings.retrieval)[args.method]
    filters = (
        Filters(args.season, args.episode, tuple(args.speaker))
        if (args.season or args.episode or args.speaker)
        else None
    )
    k = args.k or settings.retrieval.default_k
    log.info("searching %s with %s (k=%d%s)", config.config_id, args.method, k, ", filtered" if filters else "")
    results = retriever.search(args.query, k, filters)
    out(f"{args.method} on {config.config_id}: {len(results)} result(s)")
    for result in results:
        print_result(result, args.full)
    return EXIT_OK


def cmd_find(args: argparse.Namespace, settings: Settings, services: Services) -> int:
    """Print every place a quote occurs, with its lines and a paste-ready target."""
    lines = [
        line
        for line in read_jsonl(settings.paths.lines_jsonl)
        if (args.season is None or line["season"] == args.season)
        and (args.episode is None or line["episode"] == args.episode)
    ]
    by_id = {line["line_id"]: line for line in lines}
    titles = load_episode_titles(settings.paths.episodes_csv)
    hits = find_quote(lines, args.quote)
    out(f"{len(hits)} match(es)")
    for hit in hits:
        title = titles.get((hit["season"], hit["episode"]), "")
        out(f"\nS{hit['season']}E{hit['episode']} {title!r} rows {hit['source_row_start']}-{hit['source_row_end']}")
        for line_id in hit["line_ids"]:
            line = by_id[line_id]
            speaker = line["speaker"] or settings.corpus.unknown_speaker
            out(f"  r{line['source_row']:03d}  {speaker}: {line['clean_text']}")
        stub = {key: hit[key] for key in ("season", "episode", "source_row_start", "source_row_end")}
        out("  target: " + json.dumps(stub | {"expected_quote": args.quote}, ensure_ascii=False))
    return EXIT_OK


def cmd_inspect(args: argparse.Namespace, settings: Settings, services: Services) -> int:
    """Show one chunk with its full provenance and text."""
    match = CHUNK_ID_RE.match(args.chunk_id)
    if not match:
        raise ConfigurationError(f"{args.chunk_id!r} is not a chunk id", expected="like s04e13-tok512o64-003")
    size, overlap = int(match.group(3)), int(match.group(4))
    tokenizer, revision = services.tokenizer_id(settings)
    config = ChunkConfig(size, overlap, tokenizer, revision)
    chunk_set = load_chunk_set(config, settings.paths.chunks_dir, settings.paths.lines_jsonl)
    chunk = next((c for c in chunk_set.chunks if c["chunk_id"] == args.chunk_id), None)
    if chunk is None:
        raise ArtifactMissingError(f"no chunk {args.chunk_id} in {config.config_id}")
    out(json.dumps({k: v for k, v in chunk.items() if k != "text"}, indent=2, ensure_ascii=False))
    out("\n" + chunk["text"])
    return EXIT_OK


def cmd_validate(args: argparse.Namespace, settings: Settings, services: Services) -> int:
    """Validate an eval file against the corpus and print its distribution (and a review sheet)."""
    path = args.eval_file
    try:
        raw_items = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read eval file {path}: {exc}") from exc
    lines = list(read_jsonl(settings.paths.lines_jsonl))
    episodes_ = by_episode(lines)
    problems = validate_raw_items(raw_items, episodes_)
    for item_id, found in problems.items():
        out(f"INVALID {item_id}: {'; '.join(found)}")
    out(f"provenance: {len(raw_items) - len(problems)}/{len(raw_items)} items valid\n")
    valid = (
        parse_items([raw for raw in raw_items if str(raw.get("id")) not in problems], str(path))
        if len(problems) < len(raw_items)
        else []
    )
    if valid:
        out("\n".join(distribution(valid, episodes_, main_characters(lines))))
    if args.review and valid:
        save_review(args.review, review_sheet(valid, episodes_, settings.corpus.unknown_speaker))
        out(f"\nreview sheet: {args.review}")
    return EXIT_ERROR if problems else EXIT_OK


def cmd_eval(args: argparse.Namespace, settings: Settings, services: Services) -> int:
    """Run the benchmark (development set unless told otherwise) and save a new run directory."""
    if args.allow_heldout and args.split != "test" and args.eval_file is None:
        log.warning("--allow-heldout has no effect without --split test")
    request = EvalRequest(
        split=None if args.eval_file else args.split,
        eval_file=args.eval_file,
        sizes=tuple(args.sizes) if args.sizes else None,
        methods=tuple(args.methods),
        allow_heldout=args.allow_heldout,
    )
    outcome = run_evaluation(settings, request, services)
    out("\n".join(format_summary(outcome.cells, settings.evaluation.k_values, settings.evaluation.all_targets_k)))
    out(f"\nsaved {outcome.run_dir}" + ("  (HELD-OUT TEST RUN)" if outcome.held_out else ""))
    return EXIT_OK


def cmd_freeze(args: argparse.Namespace, settings: Settings, services: Services) -> int:
    """Freeze an approved eval file and split it into dev and test (once)."""
    paths = settings.paths
    manifest = freeze(
        args.source,
        paths,
        settings.evaluation,
        by_episode(list(read_jsonl(paths.lines_jsonl))),
        file_sha256(paths.lines_jsonl),
        settings.corpus.raw_csv_sha256,
    )
    out(json.dumps({"files": manifest["files"], "dev_ids": manifest["split"]["dev_ids"]}, indent=2))
    return EXIT_OK


def cmd_verify(args: argparse.Namespace, settings: Settings, services: Services) -> int:
    """Check local artifacts and print one line per check."""
    components = parse_components(args.only)
    checks = run_checks(settings, services, components, deep=args.deep)
    for check in checks:
        first_line = check.detail.splitlines()[0] if check.detail else ""
        out(f"{check.status.upper():8s} {check.component:9s} {check.name}: {first_line}")
    failed = [c for c in checks if c.status == "failed"]
    missing = [c for c in checks if c.status == "missing"]
    out(f"\n{len(checks)} checks: {len(failed)} failed, {len(missing)} missing")
    return EXIT_ERROR if failed or (args.strict and missing) else EXIT_OK


# ----------------------------------------------------------------------------- argument parsing


def parse_components(value: str | None) -> tuple[str, ...]:
    """The --only list of verify components (all by default)."""
    if not value:
        return COMPONENTS
    chosen = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(chosen) - set(COMPONENTS))
    if unknown:
        raise ConfigurationError(f"unknown verify component(s): {', '.join(unknown)}", expected=", ".join(COMPONENTS))
    return chosen


def parse_episode(code: str) -> tuple[int, int]:
    """'S4E13' -> (4, 13)."""
    match = re.fullmatch(r"[Ss](\d+)[Ee](\d+)", code.strip())
    if not match:
        raise ConfigurationError(f"{code!r} is not an episode code", expected="like S4E13")
    return int(match.group(1)), int(match.group(2))


def positive_int(value: str) -> int:
    """argparse type for integers >= 1."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def add_chunk_selection(parser: argparse.ArgumentParser, overlap: bool) -> None:
    """--all / --size (and optionally --overlap) for commands that work on chunk sets."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="every configured chunk setting")
    group.add_argument("--size", type=positive_int, help="one chunk size (default: the configured default)")
    if overlap:
        parser.add_argument("--overlap", type=int, help="custom overlap (with --size) for a non-configured setting")


def build_parser() -> argparse.ArgumentParser:
    """The full argument parser."""
    parser = argparse.ArgumentParser(
        prog="sexandrag",
        description="Retrieval benchmark over noisy TV-dialogue subtitles: parse, chunk, index, search, evaluate.",
        epilog="Run `sexandrag <command> --help` for a command's options. Docs: README.md and docs/.",
    )
    parser.add_argument("--version", action="version", version=f"sexandrag {sexandrag.__version__}")
    parser.add_argument("--root", type=Path, help="project root (default: $SEXANDRAG_ROOT or the current directory)")
    parser.add_argument("--config", type=Path, help="TOML settings file (default: <root>/sexandrag.toml if present)")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="log DEBUG messages")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="log warnings and errors only")
    parser.add_argument("--debug", action="store_true", help="DEBUG logging plus full tracebacks on errors")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    config = sub.add_parser("config", help="show the effective configuration")
    config.add_argument("action", choices=["show"], help="what to do")
    config.set_defaults(func=cmd_config)

    download = sub.add_parser("download", help="fetch the corpus from Kaggle (needs the download extra)")
    download.add_argument("--force", action="store_true", help="replace an existing file that does not verify")
    download.set_defaults(func=cmd_download)

    model = sub.add_parser("model", help="download, verify or describe the pinned embedding model")
    model_sub = model.add_subparsers(dest="model_command", required=True, metavar="action")
    model_download = model_sub.add_parser("download", help="fetch the pinned files (the only networked model step)")
    model_download.add_argument("--tokenizer-only", action="store_true", help="only the files needed for chunking")
    model_verify = model_sub.add_parser("verify", help="check the local snapshot against the pinned digests")
    model_verify.add_argument("--deep", action="store_true", help="also hash the 2.3 GB weights file")
    model_sub.add_parser("info", help="print the pinned model identity and file digests")
    model.set_defaults(func=cmd_model)

    parse = sub.add_parser("parse", help="parse the raw CSV into lines.jsonl")
    parse.add_argument(
        "--allow-unverified-corpus",
        action="store_true",
        help="parse a CSV whose checksum differs from the pinned version (targets will not line up)",
    )
    parse.add_argument("--report", action="store_true", help="print the full parse report")
    parse.set_defaults(func=cmd_parse)

    scenes = sub.add_parser("scenes", help="infer optional scene metadata (never evidence)")
    scenes.add_argument("--show", help="print one episode with scene breaks, e.g. S4E13")
    scenes.add_argument("--limit", type=positive_int, default=80, help="lines to print with --show")
    scenes.set_defaults(func=cmd_scenes)

    chunk = sub.add_parser("chunk", help="build chunk sets from lines.jsonl")
    add_chunk_selection(chunk, overlap=True)
    chunk.add_argument("--no-scenes", action="store_true", help="skip inferred-scene metadata")
    chunk.set_defaults(func=cmd_chunk)

    index = sub.add_parser("index", help="build or confirm dense indexes (slow on CPU; explicit only)")
    add_chunk_selection(index, overlap=False)
    index.add_argument("--rebuild", action="store_true", help="rebuild even if a valid cached index exists")
    index.add_argument("--list", action="store_true", help="only list cached indexes")
    index.set_defaults(func=cmd_index)

    search = sub.add_parser("search", help="retrieve chunks for a query")
    search.add_argument("query")
    search.add_argument("--method", choices=METHODS, default="hybrid")
    search.add_argument("--size", type=positive_int, help="chunk size (default: the configured default)")
    search.add_argument("-k", type=positive_int, help="number of results (default: retrieval.default_k)")
    search.add_argument("--season", type=positive_int)
    search.add_argument("--episode", type=positive_int)
    search.add_argument("--speaker", action="append", default=[], help="require this speaker (repeatable)")
    search.add_argument("--full", action="store_true", help="print every line of each chunk")
    search.set_defaults(func=cmd_search, all=False)

    find = sub.add_parser("find", help="locate a quote in the parsed lines (for writing eval targets)")
    find.add_argument("quote")
    find.add_argument("--season", type=positive_int)
    find.add_argument("--episode", type=positive_int)
    find.set_defaults(func=cmd_find)

    inspect = sub.add_parser("inspect", help="show one chunk with its provenance")
    inspect.add_argument("chunk_id", help="e.g. s04e13-tok512o64-003")
    inspect.set_defaults(func=cmd_inspect)

    validate = sub.add_parser("validate", help="validate an eval file against the corpus")
    validate.add_argument("eval_file", type=Path)
    validate.add_argument("--review", type=Path, help="write a review sheet (quotes the corpus: keep it git-ignored)")
    validate.set_defaults(func=cmd_validate)

    evaluate = sub.add_parser("eval", help="run the retrieval benchmark (development set by default)")
    source = evaluate.add_mutually_exclusive_group()
    source.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="frozen split to evaluate (test also needs --allow-heldout)",
    )
    source.add_argument("--eval-file", type=Path, help="evaluate another eval file (held-out items are refused)")
    evaluate.add_argument(
        "--allow-heldout",
        action="store_true",
        help="deliberately evaluate held-out test data (only once the approach is final)",
    )
    evaluate.add_argument("--sizes", type=positive_int, nargs="+", help="chunk sizes (default: all configured)")
    evaluate.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    evaluate.set_defaults(func=cmd_eval)

    freeze_ = sub.add_parser("freeze", help="freeze an approved eval file and split it (once; never overwrites)")
    freeze_.add_argument("source", type=Path)
    freeze_.set_defaults(func=cmd_freeze)

    verify = sub.add_parser("verify", help="check local artifacts against their manifests and hashes")
    verify.add_argument("--only", help=f"comma-separated components: {', '.join(COMPONENTS)}")
    verify.add_argument("--deep", action="store_true", help="also hash the model weights")
    verify.add_argument("--strict", action="store_true", help="treat missing artifacts as failures")
    verify.set_defaults(func=cmd_verify)
    return parser


def set_network_policy(args: argparse.Namespace) -> None:
    """Keep the Hugging Face hub offline except for the explicit download commands."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if (args.command, getattr(args, "model_command", None)) not in NETWORK_COMMANDS:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def main(argv: Sequence[str] | None = None, services: Services | None = None) -> int:
    """Parse arguments, run one command, and turn failures into exit codes and readable messages."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging("WARNING" if args.quiet else ("DEBUG" if args.verbose else "INFO"), debug=args.debug)
    set_network_policy(args)
    try:
        level = "WARNING" if args.quiet else ("DEBUG" if args.verbose or args.debug else None)
        settings = load_settings(args.root, args.config, level)
        if not (args.quiet or args.verbose or args.debug):
            configure_logging(settings.log_level)
        code: int = args.func(args, settings, services or PinnedModelServices())
        return code
    except SexAndRagError as exc:
        if args.debug:
            log.exception("%s failed", args.command)
        else:
            log.error("%s", exc)
        return EXIT_ERROR
    except KeyboardInterrupt:
        log.error("interrupted")
        return EXIT_INTERRUPTED
    except Exception as exc:
        if args.debug:
            raise
        log.error("unexpected internal error: %s: %s (rerun with --debug for the traceback)", type(exc).__name__, exc)
        return EXIT_INTERNAL


def run(argv: Sequence[str] | None = None) -> Any:
    """Console-script entry point."""
    sys.exit(main(argv))
