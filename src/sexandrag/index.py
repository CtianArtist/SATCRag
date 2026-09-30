"""Build and cache dense indexes: one per (chunk set, embedding model).

Layout: <index dir>/dense/<chunk config id>/<model slug>-<key12>/{vectors.npy, chunk_ids.json, manifest.json}

The key hashes exactly what is embedded (every chunk id and text, in order) with the embedder's
identity (model, revision, context, pooling, dimension), so a different chunk size, overlap,
tokenizer, parser output or model can never reuse a cache. Builds are explicit (retrieval never
builds), written to a temporary folder and renamed into place, so an interrupted build can never
leave a half-written index behind. BM25 needs no cache: it is rebuilt from the chunk file in
well under a second. Indexing knows nothing about evaluation.
"""

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from sexandrag.artifacts import ChunkConfig, embedded_content_hash, slug, stable_hash
from sexandrag.chunk import ChunkSet
from sexandrag.embedders import Embedder, Vectors, ensure_fits
from sexandrag.environment import library_versions
from sexandrag.errors import ArtifactMismatchError, ArtifactMissingError

log = logging.getLogger(__name__)
EMBED_BATCH = 128  # chunks per embedding call while building (only affects progress logging)
INDEX_FORMAT = 1


@dataclass
class DenseIndex:
    """L2-normalized chunk vectors in chunk-file order, plus the manifest describing them."""

    chunk_ids: list[str]
    vectors: npt.NDArray[np.float32]
    manifest: dict[str, Any]
    path: Path


def dense_cache_key(content_hash: str, embedder_identity: dict[str, Any]) -> str:
    """Key for one dense index: what was embedded plus how."""
    return stable_hash({"content": content_hash, "embedder": embedder_identity, "format": INDEX_FORMAT})


def dense_cache_dir(chunk_config: ChunkConfig, embedder_identity: dict[str, Any], key: str, index_dir: Path) -> Path:
    """Folder holding one cached dense index."""
    return index_dir / "dense" / chunk_config.config_id / f"{slug(embedder_identity['model'])}-{key[:12]}"


def rebuild_hint(chunk_config: ChunkConfig) -> str:
    """The command that rebuilds one dense index."""
    return f"run `sexandrag index --size {chunk_config.size} --rebuild`"


def check_vectors(vectors: npt.NDArray[Any], n_rows: int, dim: int) -> str | None:
    """Why vectors are unusable (wrong shape, non-finite, not unit length), or None when they are fine."""
    if vectors.shape != (n_rows, dim):
        return f"shape {vectors.shape} instead of {(n_rows, dim)}"
    if not np.all(np.isfinite(vectors)):
        return "NaN or infinite values"
    if n_rows and not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
        return "vectors that are not L2-normalized"
    return None


def embed_chunks(chunks: list[dict[str, Any]], embedder: Embedder) -> Vectors:
    """Embed chunk texts in file order; any chunk longer than the context raises before any work."""
    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]
    ensure_fits(texts, embedder.count_tokens, embedder.max_tokens, ids)
    parts, started = [], time.perf_counter()
    for start in range(0, len(texts), EMBED_BATCH):
        end = start + EMBED_BATCH
        parts.append(np.asarray(embedder.embed_documents(texts[start:end], ids[start:end]), dtype=np.float32))
        log.info("embedded %d/%d chunks (%.0fs)", min(end, len(texts)), len(texts), time.perf_counter() - started)
    vectors = np.vstack(parts) if parts else np.zeros((0, embedder.dim), dtype=np.float32)
    problem = check_vectors(vectors, len(chunks), embedder.dim)
    if problem:
        raise ArtifactMismatchError(f"the embedder returned {problem}", recovery="check the model installation")
    return vectors


def save_index(directory: Path, chunk_ids: list[str], vectors: Vectors, manifest: dict[str, Any]) -> None:
    """Write an index atomically: fill a temporary folder, then rename it into place."""
    tmp = directory.with_name(directory.name + ".tmp")
    if tmp.exists():
        log.warning("removing an incomplete index build left at %s", tmp)
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    np.save(tmp / "vectors.npy", vectors)
    (tmp / "chunk_ids.json").write_text(json.dumps(chunk_ids), encoding="utf-8")
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.rename(directory)


def load_index(directory: Path, key: str, chunk_ids: list[str], dim: int, hint: str) -> DenseIndex:
    """Load a cached index and verify it is exactly the one requested."""
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        stored_ids = json.loads((directory / "chunk_ids.json").read_text(encoding="utf-8"))
        vectors = np.load(directory / "vectors.npy", allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:  # missing, truncated or corrupt files
        raise ArtifactMismatchError(f"the cached index at {directory} is unreadable ({exc})", recovery=hint) from exc
    if manifest.get("key") != key or stored_ids != chunk_ids:
        raise ArtifactMismatchError(
            f"the cached index at {directory} does not match its key or chunk ids",
            expected=key,
            actual=manifest.get("key"),
            recovery=hint,
        )
    problem = check_vectors(vectors, len(chunk_ids), dim)
    if problem:
        raise ArtifactMismatchError(f"the cached index at {directory} holds {problem}", recovery=hint)
    return DenseIndex(stored_ids, vectors.astype(np.float32, copy=False), manifest, directory)


def index_location(chunk_set: ChunkSet, embedder_identity: dict[str, Any], index_dir: Path) -> tuple[str, Path]:
    """(cache key, folder) of the dense index for this chunk set and embedder identity."""
    key = dense_cache_key(embedded_content_hash(chunk_set.chunks), embedder_identity)
    return key, dense_cache_dir(chunk_set.config, embedder_identity, key, index_dir)


def get_dense_index(chunk_set: ChunkSet, embedder: Embedder, index_dir: Path) -> DenseIndex:
    """Load the cached index for these chunks and this embedder (never builds one)."""
    identity = embedder.identity()
    key, directory = index_location(chunk_set, identity, index_dir)
    hint = rebuild_hint(chunk_set.config)
    if not directory.is_dir():
        raise ArtifactMissingError(
            f"no dense index for {chunk_set.config.config_id} with {identity['model']}",
            recovery=f"run `sexandrag index --size {chunk_set.config.size}` (about 20 minutes per size on CPU)",
        )
    index = load_index(directory, key, [c["chunk_id"] for c in chunk_set.chunks], embedder.dim, hint)
    log.info("dense index cache hit: %s (%d vectors)", directory.name, len(index.chunk_ids))
    return index


def build_dense_index(chunk_set: ChunkSet, embedder: Embedder, index_dir: Path, rebuild: bool = False) -> DenseIndex:
    """Build the index for this chunk set unless a valid cached copy exists (rebuild=True replaces it)."""
    identity = embedder.identity()
    key, directory = index_location(chunk_set, identity, index_dir)
    chunk_ids = [chunk["chunk_id"] for chunk in chunk_set.chunks]
    if directory.is_dir() and not rebuild:
        index = load_index(directory, key, chunk_ids, embedder.dim, rebuild_hint(chunk_set.config))
        log.info("dense index for %s is cached and valid: %s", chunk_set.config.config_id, directory)
        return index
    if directory.is_dir():
        log.warning("rebuilding the dense index at %s (--rebuild)", directory)
        shutil.rmtree(directory)
    log.info(
        "building the dense index for %s (%d chunks) with %s",
        chunk_set.config.config_id,
        len(chunk_ids),
        identity["model"],
    )
    started = time.perf_counter()
    vectors = embed_chunks(chunk_set.chunks, embedder)
    runtime = getattr(embedder, "runtime", None)
    manifest = {
        "key": key,
        "chunk_config": asdict(chunk_set.config),
        "content_hash": embedded_content_hash(chunk_set.chunks),
        "n_chunks": len(chunk_ids),
        "embedder": identity,
        "versions": library_versions(),
        "runtime": runtime() if callable(runtime) else {},
        "build_seconds": round(time.perf_counter() - started, 1),
    }
    save_index(directory, chunk_ids, vectors, manifest)
    log.info("saved %s (%d x %d, %.1fs)", directory, vectors.shape[0], vectors.shape[1], manifest["build_seconds"])
    return DenseIndex(chunk_ids, vectors, manifest, directory)


def cached_indexes(index_dir: Path) -> list[dict[str, Any]]:
    """One summary per cached dense index (and incomplete build folders), for `sexandrag index --list`."""
    summaries: list[dict[str, Any]] = []
    for folder in sorted(index_dir.glob("dense/*/*")):
        if not folder.is_dir():
            continue
        if folder.name.endswith(".tmp"):
            summaries.append({"path": str(folder), "status": "incomplete build (safe to delete)"})
            continue
        try:
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summaries.append({"path": str(folder), "status": "unreadable manifest"})
            continue
        size_mb = sum(f.stat().st_size for f in folder.iterdir()) / 1e6
        summaries.append(
            {
                "path": str(folder),
                "status": "ok",
                "chunks": manifest.get("n_chunks"),
                "model": f"{manifest['embedder']['model']}@{str(manifest['embedder']['revision'])[:8]}",
                "dim": manifest["embedder"].get("dim"),
                "size_mb": round(size_mb, 1),
                "build_seconds": manifest.get("build_seconds"),
            }
        )
    return summaries
