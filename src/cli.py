"""Command-line entry point.

  python -m src.cli find "we have breakfast at 7:00 a.m."             # locate a quote -> eval stub
  python -m src.cli search "Aidan moves his stuff in" [--retriever hybrid] [--size 512] [-k 8]
                    [--season 4] [--episode 13] [--speaker Carrie] [--full]
"""
import argparse
import json

from src import config
from src.artifacts import chunk_config_for_size
from src.embedders import SentenceTransformerEmbedder
from src.episodes import load_episode_titles
from src.jsonl import read_jsonl
from src.provenance import find_quote
from src.retrieve import Filters, SearchResult, build_retrievers


def eval_stub(hit: dict, quote: str) -> dict:
    """An eval.json target for a found quote, ready to paste."""
    return {"season": hit["season"], "episode": hit["episode"],
            "source_row_start": hit["source_row_start"], "source_row_end": hit["source_row_end"],
            "expected_quote": quote}


def cmd_find(args: argparse.Namespace) -> None:
    """Print every place a quote occurs, with its lines and an eval stub."""
    lines = [l for l in read_jsonl(config.LINES_JSONL)
             if (args.season is None or l["season"] == args.season)
             and (args.episode is None or l["episode"] == args.episode)]
    by_id = {line["line_id"]: line for line in lines}
    titles = load_episode_titles(config.EPISODES_CSV)
    hits = find_quote(lines, args.quote)
    print(f"{len(hits)} match(es) for {args.quote!r}")
    for hit in hits:
        title = titles.get((hit["season"], hit["episode"]), "")
        print(f"\nS{hit['season']}E{hit['episode']} {title!r} rows {hit['source_row_start']}-{hit['source_row_end']}")
        for line_id in hit["line_ids"]:
            line = by_id[line_id]
            print(f"  r{line['source_row']:03d}  {line['speaker'] or config.UNKNOWN_SPEAKER}: {line['clean_text']}")
        print("  eval stub:", json.dumps(eval_stub(hit, args.quote), ensure_ascii=False))


def print_result(result: SearchResult, full: bool, preview_lines: int = 6) -> None:
    """Print one search result: rank, score, provenance, then its lines."""
    found_by = ", ".join(f"{method} #{rank}" for method, rank in result.components.items())
    print(f"\n#{result.rank}  {result.method} score {result.score:.4f}  {result.chunk_id}  "
          f"S{result.season}E{result.episode} {result.episode_title or ''!r}  "
          f"rows {result.source_row_start}-{result.source_row_end}" + (f"  [{found_by}]" if found_by else ""))
    print(f"    speakers: {', '.join(result.speakers)}")
    lines = result.text.splitlines()
    for line in lines if full else lines[:preview_lines]:
        print(f"    {line}")
    if not full and len(lines) > preview_lines:
        print(f"    ... ({len(lines) - preview_lines} more lines; --full shows all)")


def cmd_search(args: argparse.Namespace) -> None:
    """Run one retriever on one chunk set and print the ranked chunks."""
    embedder = SentenceTransformerEmbedder() if args.retriever in ("dense", "hybrid") else None
    chunk_config = chunk_config_for_size(args.size)
    retriever = build_retrievers(chunk_config, (args.retriever,), embedder)[args.retriever]
    filters = Filters(args.season, args.episode, tuple(args.speaker)) if (
        args.season or args.episode or args.speaker) else None
    results = retriever.search(args.query, args.k, filters)
    print(f"{args.retriever} on {chunk_config.config_id}: {len(results)} result(s) for {args.query!r}")
    for result in results:
        print_result(result, args.full)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the chosen subcommand."""
    parser = argparse.ArgumentParser(description="SATC RAG command line")
    sub = parser.add_subparsers(dest="command", required=True)
    find = sub.add_parser("find", help="locate a quote in the parsed lines")
    find.add_argument("quote")
    find.add_argument("--season", type=int)
    find.add_argument("--episode", type=int)
    find.set_defaults(func=cmd_find)
    search = sub.add_parser("search", help="retrieve chunks for a query")
    search.add_argument("query")
    search.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="hybrid")
    search.add_argument("--size", type=int, default=config.CHUNK_SIZE, help="chunk size (256, 512 or 1024)")
    search.add_argument("-k", type=int, default=config.DEFAULT_K)
    search.add_argument("--season", type=int)
    search.add_argument("--episode", type=int)
    search.add_argument("--speaker", action="append", default=[], help="require this speaker (repeatable)")
    search.add_argument("--full", action="store_true", help="print every line of each chunk")
    search.set_defaults(func=cmd_search)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
