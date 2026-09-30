"""Identities and fingerprints of on-disk artifacts, so caches can never be mixed up.

A chunk set is named by (size, overlap, tokenizer) and fingerprinted by content hashes;
a dense index is keyed by a hash of exactly what was embedded plus the embedder's identity.
Changing chunk size, overlap, tokenizer, parser output or model always gives a new key.
"""
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from src import config


class StaleArtifactError(RuntimeError):
    """An artifact on disk no longer matches its inputs and must be rebuilt."""


def slug(name: str) -> str:
    """Short filesystem-safe form of a model name: 'BAAI/bge-m3' -> 'bge-m3'."""
    return re.sub(r"[^a-z0-9]+", "-", name.split("/")[-1].lower()).strip("-")


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(obj) -> str:
    """SHA-256 of an object's canonical JSON form (key order does not matter)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def embedded_content_hash(chunks: list[dict]) -> str:
    """Hash of exactly what a dense index embeds: every (chunk_id, text) pair, in order."""
    return stable_hash([[chunk["chunk_id"], chunk["text"]] for chunk in chunks])


@dataclass(frozen=True)
class ChunkConfig:
    """One chunking setting: token budget, overlap, and the tokenizer that counts them."""
    size: int
    overlap: int
    tokenizer: str = config.TOKENIZER
    tokenizer_revision: str | None = config.TOKENIZER_REVISION

    @property
    def tag(self) -> str:
        """Short tag used inside chunk ids, e.g. 'tok512o64'."""
        return f"tok{self.size}o{self.overlap}"

    @property
    def config_id(self) -> str:
        """Name of the chunk set on disk, e.g. 'tok512o64-bge-m3'."""
        return f"{self.tag}-{slug(self.tokenizer)}"

    @property
    def path(self) -> Path:
        """The chunk set's JSONL file."""
        return config.CHUNKS_DIR / f"{self.config_id}.jsonl"

    @property
    def manifest_path(self) -> Path:
        """The chunk set's manifest (settings plus input and output hashes)."""
        return config.CHUNKS_DIR / f"{self.config_id}.manifest.json"


def primary_chunk_configs() -> list[ChunkConfig]:
    """The chunk settings of the primary experiment, from config.CHUNK_CONFIGS."""
    return [ChunkConfig(size, overlap) for size, overlap in config.CHUNK_CONFIGS]


def chunk_config_for_size(size: int) -> ChunkConfig:
    """The primary chunk setting with this size (its overlap comes from config)."""
    for chunk_config in primary_chunk_configs():
        if chunk_config.size == size:
            return chunk_config
    raise ValueError(f"no chunk size {size} in config.CHUNK_CONFIGS {config.CHUNK_CONFIGS}")
