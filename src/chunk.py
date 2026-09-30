"""Token-based chunking of each episode's ordered lines: the primary retrieval unit.

Lines are never split or reordered. A chunk holds consecutive whole lines, at most
`size` tokens (one line longer than that becomes a chunk by itself). The next chunk
starts by repeating the previous chunk's trailing lines worth at most `overlap` tokens.
Chunks never cross episode boundaries. Chunk text holds only 'Speaker: line' rows, so
episode titles and ids stay metadata and cannot influence relevance.

Run: python -m src.chunk [--all | --size 512 --overlap 64] [--no-scenes] [--show CHUNK_ID]
"""
import argparse
import json
import statistics
from itertools import groupby

from src import config
from src.artifacts import ChunkConfig, StaleArtifactError, file_sha256, primary_chunk_configs
from src.episodes import load_episode_titles
from src.jsonl import read_jsonl, write_jsonl
from src.scenes import detect_scenes, scene_numbers
from src.tokens import TokenCounter, get_token_counter


def format_line(line: dict) -> str:
    """Render one line for chunk text: 'Speaker: text' ('(unknown)' when speaker is None)."""
    return f"{line['speaker'] or config.UNKNOWN_SPEAKER}: {line['clean_text']}"


def next_window_start(counts: list[int], start: int, end: int, overlap: int, size: int) -> int:
    """Where the window after [start, end) begins.

    Backs up from `end` over trailing lines worth at most `overlap` tokens, but only while
    the repeated lines plus the next new line still fit in `size`, and never back to `start`.
    """
    room = min(overlap, size - counts[end])
    next_start, used = end, 0
    while next_start - 1 > start and used + counts[next_start - 1] <= room:
        next_start -= 1
        used += counts[next_start]
    return next_start


def pack_windows(counts: list[int], size: int, overlap: int) -> list[tuple[int, int]]:
    """Split line token counts into windows [start, end) of consecutive whole lines."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError(f"need size > 0 and 0 <= overlap < size (got {size}, {overlap})")
    windows, start = [], 0
    while start < len(counts):
        end, total = start + 1, counts[start]
        while end < len(counts) and total + counts[end] <= size:
            total += counts[end]
            end += 1
        windows.append((start, end))
        if end == len(counts):
            break
        start = next_window_start(counts, start, end, overlap, size)
    return windows


def unique_speakers(lines: list[dict]) -> list[str]:
    """Named speakers in order of first appearance."""
    names: list[str] = []
    for line in lines:
        names += [n for n in line["speakers"] if n not in names]
    return names


def make_chunk(lines: list[dict], counts: list[int], index: int, tag: str, title: str | None) -> dict:
    """Build one chunk record from consecutive lines of a single episode."""
    first, last = lines[0], lines[-1]
    return {
        "chunk_id": f"s{first['season']:02d}e{first['episode']:02d}-{tag}-{index:03d}",
        "chunking": tag,
        "season": first["season"],
        "episode": first["episode"],
        "episode_title": title,
        "source_row_start": first["source_row"],
        "source_row_end": last["source_row"],
        "line_ids": [line["line_id"] for line in lines],
        "speakers": unique_speakers(lines),
        "n_lines": len(lines),
        "n_tokens": sum(counts),
        "text": "\n".join(format_line(line) for line in lines),
    }


def link_neighbors(chunks: list[dict]) -> None:
    """Set prev_chunk_id / next_chunk_id between consecutive chunks of one episode."""
    for i, chunk in enumerate(chunks):
        chunk["prev_chunk_id"] = chunks[i - 1]["chunk_id"] if i > 0 else None
        chunk["next_chunk_id"] = chunks[i + 1]["chunk_id"] if i + 1 < len(chunks) else None


def attach_scenes(chunks: list[dict], windows: list[tuple[int, int]], line_scenes: list[int]) -> None:
    """Record which inferred scenes each chunk overlaps (secondary metadata only)."""
    for chunk, (start, end) in zip(chunks, windows):
        chunk["inferred_scenes"] = sorted(set(line_scenes[start:end]))


def chunk_episode(lines: list[dict], counts: list[int], size: int, overlap: int,
                  title: str | None, with_scenes: bool) -> list[dict]:
    """Chunk one episode's ordered lines, given each line's token count."""
    tag = f"tok{size}o{overlap}"
    windows = pack_windows(counts, size, overlap)
    chunks = [make_chunk(lines[a:b], counts[a:b], i, tag, title) for i, (a, b) in enumerate(windows)]
    link_neighbors(chunks)
    if with_scenes:
        attach_scenes(chunks, windows, scene_numbers(detect_scenes(lines), len(lines)))
    return chunks


def group_episodes(lines: list[dict]) -> list[list[dict]]:
    """Group lines by episode in file order; each episode must be one contiguous block."""
    groups = [list(g) for _, g in groupby(lines, key=lambda line: (line["season"], line["episode"]))]
    keys = [(g[0]["season"], g[0]["episode"]) for g in groups]
    if len(keys) != len(set(keys)):
        raise ValueError("an episode's lines are not contiguous in the input")
    return groups


def chunk_all(lines: list[dict], count_tokens: TokenCounter, size: int, overlap: int,
              titles: dict, with_scenes: bool) -> list[dict]:
    """Chunk every episode; each formatted line is token-counted once."""
    chunks = []
    for episode_lines in group_episodes(lines):
        counts = [count_tokens(format_line(line)) for line in episode_lines]
        key = (episode_lines[0]["season"], episode_lines[0]["episode"])
        chunks += chunk_episode(episode_lines, counts, size, overlap, titles.get(key), with_scenes)
    return chunks


def spread(values: list[int]) -> str:
    """Summarize a list of numbers as 'min / p10 / median / p90 / max'."""
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return f"{ordered[0]} / {pick(0.1)} / {statistics.median(ordered):g} / {pick(0.9)} / {ordered[-1]}"


def print_stats(chunks: list[dict], size: int) -> None:
    """Print chunk counts and size distributions."""
    per_episode = [len(list(g)) for _, g in groupby(chunks, key=lambda c: (c["season"], c["episode"]))]
    over = [c for c in chunks if c["n_tokens"] > size]
    print(f"Chunks: {len(chunks)}   (min / p10 / median / p90 / max below)")
    print(f"  tokens per chunk:   {spread([c['n_tokens'] for c in chunks])}")
    print(f"  lines per chunk:    {spread([c['n_lines'] for c in chunks])}")
    print(f"  chunks per episode: {spread(per_episode)}")
    print(f"  single-line chunks over the size limit: {len(over)}")


def show_chunk(chunk: dict) -> None:
    """Print one chunk with its metadata header."""
    meta = {k: v for k, v in chunk.items() if k not in ("text", "line_ids")}
    print(f"\n--- {chunk['chunk_id']} ---\n{meta}\n{chunk['text']}")


def save_chunk_set(chunk_config: ChunkConfig, chunks: list[dict], with_scenes: bool) -> dict:
    """Write a chunk set and a manifest recording its settings and input/output hashes."""
    write_jsonl(chunk_config.path, chunks)
    manifest = {
        "config_id": chunk_config.config_id,
        "size": chunk_config.size,
        "overlap": chunk_config.overlap,
        "tokenizer": chunk_config.tokenizer,
        "tokenizer_revision": chunk_config.tokenizer_revision,
        "inferred_scenes": with_scenes,
        "n_chunks": len(chunks),
        "lines_sha256": file_sha256(config.LINES_JSONL),
        "chunks_sha256": file_sha256(chunk_config.path),
    }
    chunk_config.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_chunk_set(chunk_config: ChunkConfig) -> tuple[list[dict], dict]:
    """Read a chunk set and its manifest, refusing files that are missing, altered or stale."""
    rebuild = f"python -m src.chunk --size {chunk_config.size} --overlap {chunk_config.overlap}"
    if not (chunk_config.path.exists() and chunk_config.manifest_path.exists()):
        raise FileNotFoundError(f"chunk set {chunk_config.config_id} not built; run: {rebuild}")
    manifest = json.loads(chunk_config.manifest_path.read_text(encoding="utf-8"))
    problems = []
    if manifest["chunks_sha256"] != file_sha256(chunk_config.path):
        problems.append("the chunk file changed after its manifest was written")
    if manifest["lines_sha256"] != file_sha256(config.LINES_JSONL):
        problems.append("lines.jsonl changed since these chunks were built")
    if (manifest["tokenizer"], manifest["tokenizer_revision"]) != (chunk_config.tokenizer,
                                                                   chunk_config.tokenizer_revision):
        problems.append("the chunks were counted with a different tokenizer")
    if problems:
        raise StaleArtifactError(f"{chunk_config.config_id}: {'; '.join(problems)}. Rebuild: {rebuild}")
    return list(read_jsonl(chunk_config.path)), manifest


def main(argv: list[str] | None = None) -> None:
    """Chunk lines.jsonl for one setting (or all primary settings), write each set, print stats."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="build every setting in config.CHUNK_CONFIGS")
    parser.add_argument("--size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--no-scenes", action="store_true", help="skip inferred-scene metadata")
    parser.add_argument("--show", action="append", default=[], help="print this chunk id (repeatable)")
    args = parser.parse_args(argv)
    chunk_configs = primary_chunk_configs() if args.all else [ChunkConfig(args.size, args.overlap)]
    lines = list(read_jsonl(config.LINES_JSONL))
    titles = load_episode_titles(config.EPISODES_CSV)
    with_scenes = config.SCENE_DETECTION and not args.no_scenes
    count_tokens = get_token_counter(config.TOKENIZER, config.TOKENIZER_REVISION)
    for chunk_config in chunk_configs:
        chunks = chunk_all(lines, count_tokens, chunk_config.size, chunk_config.overlap, titles, with_scenes)
        save_chunk_set(chunk_config, chunks, with_scenes)
        print(f"\n{chunk_config.config_id}: size={chunk_config.size} overlap={chunk_config.overlap} "
              f"tokenizer={chunk_config.tokenizer} scenes={with_scenes}")
        print_stats(chunks, chunk_config.size)
        print(f"Wrote {chunk_config.path.relative_to(config.ROOT)} (+ manifest)")
        for chunk in chunks:
            if chunk["chunk_id"] in args.show:
                show_chunk(chunk)


if __name__ == "__main__":
    main()
