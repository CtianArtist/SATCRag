"""Identities and fingerprints of on-disk artifacts, so caches can never be mixed up.

A chunk set is named by (size, overlap, tokenizer) and fingerprinted by content hashes; a dense
index is keyed by a hash of exactly what was embedded plus the embedder's identity. Changing
chunk size, overlap, tokenizer, parser output or model always gives a new key.
"""

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sexandrag.config import DEFAULT_MODEL, ChunkingConfig, chunk_setting_problems
from sexandrag.errors import ConfigurationError


def slug(name: str) -> str:
    """Short filesystem-safe form of a model name: 'BAAI/bge-m3' -> 'bge-m3'."""
    return re.sub(r"[^a-z0-9]+", "-", name.split("/")[-1].lower()).strip("-")


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes, read in 1 MB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """Git's blob hash of a file (how the Hugging Face cache names small, non-LFS files)."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def stable_hash(obj: Any) -> str:
    """SHA-256 of an object's canonical JSON form (key order does not matter)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def embedded_content_hash(chunks: Iterable[dict[str, Any]]) -> str:
    """Hash of exactly what a dense index embeds: every (chunk_id, text) pair, in order."""
    return stable_hash([[chunk["chunk_id"], chunk["text"]] for chunk in chunks])


@dataclass(frozen=True)
class ChunkConfig:
    """One chunking setting: token budget, overlap, and the tokenizer that counts them."""

    size: int
    overlap: int
    tokenizer: str = DEFAULT_MODEL.repo_id
    tokenizer_revision: str | None = DEFAULT_MODEL.revision

    @property
    def tag(self) -> str:
        """Short tag used inside chunk ids, e.g. 'tok512o64'."""
        return f"tok{self.size}o{self.overlap}"

    @property
    def config_id(self) -> str:
        """Name of the chunk set on disk, e.g. 'tok512o64-bge-m3'."""
        return f"{self.tag}-{slug(self.tokenizer)}"

    def jsonl_path(self, chunks_dir: Path) -> Path:
        """The chunk set's JSONL file."""
        return chunks_dir / f"{self.config_id}.jsonl"

    def manifest_path(self, chunks_dir: Path) -> Path:
        """The chunk set's manifest (settings plus input and output hashes)."""
        return chunks_dir / f"{self.config_id}.manifest.json"


def chunk_configs(chunking: ChunkingConfig, tokenizer: str, revision: str | None) -> list[ChunkConfig]:
    """Every configured chunk setting, counted with the given tokenizer."""
    return [ChunkConfig(size, overlap, tokenizer, revision) for size, overlap in chunking.configs]


def chunk_config_for_size(size: int, chunking: ChunkingConfig, tokenizer: str, revision: str | None) -> ChunkConfig:
    """The configured chunk setting with this size (its overlap comes from the configuration)."""
    for chunk_config in chunk_configs(chunking, tokenizer, revision):
        if chunk_config.size == size:
            return chunk_config
    raise ConfigurationError(
        f"no chunk set of size {size} is configured",
        expected=", ".join(str(s) for s, _ in chunking.configs),
        actual=size,
        recovery="pick one of the configured sizes, or add the (size, overlap) pair to [chunking] configs",
    )


def custom_chunk_config(size: int, overlap: int, max_tokens: int, tokenizer: str, revision: str | None) -> ChunkConfig:
    """A validated chunk setting given on the command line."""
    problems = chunk_setting_problems(size, overlap, max_tokens)
    if problems:
        raise ConfigurationError("invalid chunk setting: " + "; ".join(problems))
    return ChunkConfig(size, overlap, tokenizer, revision)
