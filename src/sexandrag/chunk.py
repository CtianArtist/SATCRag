"""Token-based chunking of each episode's ordered lines: the primary retrieval unit.

Lines are never split or reordered. A chunk holds consecutive whole lines, at most `size`
tokens (one line longer than that becomes a chunk by itself). The next chunk starts by
repeating the previous chunk's trailing lines worth at most `overlap` tokens. Chunks never
cross episode boundaries. Chunk text holds only 'Speaker: line' rows, so episode titles and
ids stay metadata and cannot influence relevance.
"""

import json
import logging
import statistics
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any

from sexandrag.artifacts import ChunkConfig, file_sha256
from sexandrag.config import DEFAULT_CORPUS, DEFAULT_SCENES, SceneConfig
from sexandrag.errors import ArtifactMismatchError, ArtifactMissingError, CorpusFormatError, StaleArtifactError
from sexandrag.jsonl import read_jsonl, write_jsonl
from sexandrag.scenes import detect_with, scene_numbers
from sexandrag.tokens import TokenCounter

log = logging.getLogger(__name__)
Line = dict[str, Any]
Chunk = dict[str, Any]
MANIFEST_KEYS = (
    "config_id",
    "size",
    "overlap",
    "tokenizer",
    "tokenizer_revision",
    "n_chunks",
    "lines_sha256",
    "chunks_sha256",
)


@dataclass(frozen=True)
class ChunkSet:
    """One loaded, verified chunk set."""

    config: ChunkConfig
    chunks: list[Chunk]
    manifest: dict[str, Any]


def format_line(line: Line, unknown_speaker: str = DEFAULT_CORPUS.unknown_speaker) -> str:
    """Render one line for chunk text: 'Speaker: text' ('(unknown)' when speaker is None)."""
    return f"{line['speaker'] or unknown_speaker}: {line['clean_text']}"


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


def unique_speakers(lines: list[Line]) -> list[str]:
    """Named speakers in order of first appearance."""
    names: list[str] = []
    for line in lines:
        names += [n for n in line["speakers"] if n not in names]
    return names


def make_chunk(
    lines: list[Line], counts: list[int], index: int, tag: str, title: str | None, unknown_speaker: str
) -> Chunk:
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
        "text": "\n".join(format_line(line, unknown_speaker) for line in lines),
    }


def link_neighbors(chunks: list[Chunk]) -> None:
    """Set prev_chunk_id / next_chunk_id between consecutive chunks of one episode."""
    for i, chunk in enumerate(chunks):
        chunk["prev_chunk_id"] = chunks[i - 1]["chunk_id"] if i > 0 else None
        chunk["next_chunk_id"] = chunks[i + 1]["chunk_id"] if i + 1 < len(chunks) else None


def attach_scenes(chunks: list[Chunk], windows: list[tuple[int, int]], line_scenes: list[int]) -> None:
    """Record which inferred scenes each chunk overlaps (secondary metadata only)."""
    for chunk, (start, end) in zip(chunks, windows, strict=True):
        chunk["inferred_scenes"] = sorted(set(line_scenes[start:end]))


def chunk_episode(
    lines: list[Line],
    counts: list[int],
    size: int,
    overlap: int,
    title: str | None,
    scenes: SceneConfig | None,
    unknown_speaker: str = DEFAULT_CORPUS.unknown_speaker,
) -> list[Chunk]:
    """Chunk one episode's ordered lines, given each line's token count (scenes=None skips them)."""
    tag = f"tok{size}o{overlap}"
    windows = pack_windows(counts, size, overlap)
    chunks = [make_chunk(lines[a:b], counts[a:b], i, tag, title, unknown_speaker) for i, (a, b) in enumerate(windows)]
    link_neighbors(chunks)
    if scenes is not None:
        attach_scenes(chunks, windows, scene_numbers(detect_with(lines, scenes), len(lines)))
    return chunks


def group_episodes(lines: list[Line]) -> list[list[Line]]:
    """Group lines by episode in file order; each episode must be one contiguous block."""
    groups = [list(g) for _, g in groupby(lines, key=lambda line: (line["season"], line["episode"]))]
    keys = [(g[0]["season"], g[0]["episode"]) for g in groups]
    if len(keys) != len(set(keys)):
        raise CorpusFormatError(
            "an episode's lines are not contiguous in lines.jsonl", recovery="re-run `sexandrag parse`"
        )
    return groups


def chunk_all(
    lines: list[Line],
    count_tokens: TokenCounter,
    size: int,
    overlap: int,
    titles: dict[tuple[int, int], str],
    scenes: SceneConfig | None = DEFAULT_SCENES,
    unknown_speaker: str = DEFAULT_CORPUS.unknown_speaker,
) -> list[Chunk]:
    """Chunk every episode; each formatted line is token-counted once."""
    chunks: list[Chunk] = []
    for episode_lines in group_episodes(lines):
        counts = [count_tokens(format_line(line, unknown_speaker)) for line in episode_lines]
        key = (episode_lines[0]["season"], episode_lines[0]["episode"])
        chunks += chunk_episode(episode_lines, counts, size, overlap, titles.get(key), scenes, unknown_speaker)
    return chunks


def spread(values: list[int]) -> str:
    """Summarize a list of numbers as 'min/p10/median/p90/max'."""
    ordered = sorted(values)

    def pick(q: float) -> int:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return f"{ordered[0]}/{pick(0.1)}/{statistics.median(ordered):g}/{pick(0.9)}/{ordered[-1]}"


def log_stats(chunks: list[Chunk], size: int) -> None:
    """Log chunk counts and size distributions."""
    per_episode = [len(list(g)) for _, g in groupby(chunks, key=lambda c: (c["season"], c["episode"]))]
    over = sum(1 for c in chunks if c["n_tokens"] > size)
    log.info(
        "%d chunks; tokens/chunk %s, lines/chunk %s, chunks/episode %s (min/p10/median/p90/max); "
        "%d single-line chunks over the size limit",
        len(chunks),
        spread([c["n_tokens"] for c in chunks]),
        spread([c["n_lines"] for c in chunks]),
        spread(per_episode),
        over,
    )


def save_chunk_set(
    chunk_config: ChunkConfig, chunks: list[Chunk], with_scenes: bool, chunks_dir: Path, lines_path: Path
) -> dict[str, Any]:
    """Write a chunk set and a manifest recording its settings and input/output hashes."""
    path = chunk_config.jsonl_path(chunks_dir)
    write_jsonl(path, chunks)
    manifest = {
        "config_id": chunk_config.config_id,
        "size": chunk_config.size,
        "overlap": chunk_config.overlap,
        "tokenizer": chunk_config.tokenizer,
        "tokenizer_revision": chunk_config.tokenizer_revision,
        "inferred_scenes": with_scenes,
        "n_chunks": len(chunks),
        "lines_sha256": file_sha256(lines_path),
        "chunks_sha256": file_sha256(path),
    }
    manifest_path = chunk_config.manifest_path(chunks_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("wrote chunk set %s (%d chunks) and its manifest", chunk_config.config_id, len(chunks))
    return manifest


def read_manifest(chunk_config: ChunkConfig, chunks_dir: Path, rebuild: str) -> dict[str, Any]:
    """Read and structurally check one chunk-set manifest."""
    path = chunk_config.manifest_path(chunks_dir)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactMismatchError(f"chunk manifest {path} is not valid JSON", recovery=rebuild) from exc
    missing = (
        [key for key in MANIFEST_KEYS if key not in manifest] if isinstance(manifest, dict) else list(MANIFEST_KEYS)
    )
    if missing:
        raise ArtifactMismatchError(f"chunk manifest {path} lacks {', '.join(missing)}", recovery=rebuild)
    return dict(manifest)


def check_manifest(
    manifest: dict[str, Any], chunk_config: ChunkConfig, chunks_dir: Path, lines_path: Path, rebuild: str
) -> None:
    """Raise StaleArtifactError when a chunk set no longer matches its file, its input or its tokenizer."""
    checks = [
        (
            "the chunk file changed after its manifest was written",
            manifest["chunks_sha256"],
            file_sha256(chunk_config.jsonl_path(chunks_dir)),
        ),
        (
            "lines.jsonl changed since these chunks were built",
            manifest["lines_sha256"],
            file_sha256(lines_path) if lines_path.is_file() else "missing",
        ),
        (
            "the chunks were counted with a different tokenizer",
            f"{chunk_config.tokenizer}@{chunk_config.tokenizer_revision}",
            f"{manifest['tokenizer']}@{manifest['tokenizer_revision']}",
        ),
        (
            "the manifest describes a different chunk setting",
            [chunk_config.size, chunk_config.overlap],
            [manifest["size"], manifest["overlap"]],
        ),
    ]
    for problem, expected, actual in checks:
        if expected != actual:
            raise StaleArtifactError(
                f"chunk set {chunk_config.config_id} is stale: {problem}",
                expected=expected,
                actual=actual,
                recovery=rebuild,
            )


def load_chunk_set(chunk_config: ChunkConfig, chunks_dir: Path, lines_path: Path) -> ChunkSet:
    """Read a chunk set and its manifest, refusing files that are missing, corrupt or stale."""
    rebuild = f"run `sexandrag chunk --size {chunk_config.size} --overlap {chunk_config.overlap}`"
    path = chunk_config.jsonl_path(chunks_dir)
    if not (path.is_file() and chunk_config.manifest_path(chunks_dir).is_file()):
        raise ArtifactMissingError(f"chunk set {chunk_config.config_id} has not been built", recovery=rebuild)
    manifest = read_manifest(chunk_config, chunks_dir, rebuild)
    check_manifest(manifest, chunk_config, chunks_dir, lines_path, rebuild)
    chunks = list(read_jsonl(path))
    if len(chunks) != manifest["n_chunks"]:
        raise ArtifactMismatchError(
            f"chunk set {chunk_config.config_id} holds a different number of chunks than its manifest",
            expected=manifest["n_chunks"],
            actual=len(chunks),
            recovery=rebuild,
        )
    log.debug("loaded chunk set %s (%d chunks)", chunk_config.config_id, len(chunks))
    return ChunkSet(chunk_config, chunks, manifest)


def build_chunk_set(
    chunk_config: ChunkConfig,
    lines: list[Line],
    count_tokens: TokenCounter,
    titles: dict[tuple[int, int], str],
    scenes: SceneConfig | None,
    chunks_dir: Path,
    lines_path: Path,
    unknown_speaker: str = DEFAULT_CORPUS.unknown_speaker,
) -> ChunkSet:
    """Chunk all lines for one setting, write the set and its manifest, and return it."""
    log.info(
        "chunking %d lines: size %d, overlap %d, tokenizer %s, inferred scenes %s",
        len(lines),
        chunk_config.size,
        chunk_config.overlap,
        chunk_config.tokenizer,
        "on" if scenes is not None else "off",
    )
    chunks = chunk_all(lines, count_tokens, chunk_config.size, chunk_config.overlap, titles, scenes, unknown_speaker)
    log_stats(chunks, chunk_config.size)
    manifest = save_chunk_set(chunk_config, chunks, scenes is not None, chunks_dir, lines_path)
    return ChunkSet(chunk_config, chunks, manifest)
