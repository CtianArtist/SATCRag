"""Build and cache dense indexes: one per (chunk set, embedding model).

Run: python -m src.index [--all | --size 512]
Layout: index/dense/<chunk config id>/<model slug>-<key12>/{vectors.npy, chunk_ids.json, manifest.json}
The key hashes exactly what is embedded (every chunk id and text, in order) with the embedder's
identity (model, revision, context, pooling, dimension), so a different chunk size, overlap,
tokenizer, parser output or model can never reuse a cache. BM25 needs no cache: it is rebuilt
from the chunk file in well under a second. Indexing knows nothing about evaluation.
"""
import argparse
import importlib.metadata
import json
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src import config
from src.artifacts import (ChunkConfig, chunk_config_for_size, embedded_content_hash, primary_chunk_configs,
                           slug, stable_hash)
from src.chunk import load_chunk_set
from src.embedders import Embedder, SentenceTransformerEmbedder, ensure_fits

PROGRESS_EVERY = 128   # chunks per embedding call while building (only affects progress reporting)


@dataclass
class DenseIndex:
    """L2-normalized chunk vectors in chunk-file order, plus the manifest describing them."""
    chunk_ids: list[str]
    vectors: np.ndarray
    manifest: dict
    path: Path


def dense_cache_key(content_hash: str, embedder_identity: dict) -> str:
    """Key for one dense index: what was embedded plus how."""
    return stable_hash({"content": content_hash, "embedder": embedder_identity, "format": 1})


def dense_cache_dir(chunk_config: ChunkConfig, embedder_identity: dict, key: str,
                    index_dir: Path | None = None) -> Path:
    """Folder holding one cached dense index."""
    base = (index_dir or config.INDEX_DIR) / "dense" / chunk_config.config_id
    return base / f"{slug(embedder_identity['model'])}-{key[:12]}"


def check_vectors(vectors: np.ndarray, n_rows: int, dim: int) -> None:
    """Raise unless vectors are (n_rows, dim), finite, and unit length."""
    if vectors.shape != (n_rows, dim):
        raise ValueError(f"expected vectors of shape {(n_rows, dim)}, got {vectors.shape}")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("vectors contain NaN or infinity")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3):
        raise ValueError("vectors are not L2-normalized")


def embed_chunks(chunks: list[dict], embedder: Embedder, progress: bool = False) -> np.ndarray:
    """Embed chunk texts in file order; any chunk longer than the context raises before any work."""
    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]
    ensure_fits(texts, embedder.count_tokens, embedder.max_tokens, ids)
    parts, started = [], time.perf_counter()
    for start in range(0, len(texts), PROGRESS_EVERY):
        end = start + PROGRESS_EVERY
        parts.append(np.asarray(embedder.embed_documents(texts[start:end], ids[start:end]), dtype=np.float32))
        if progress:
            print(f"  embedded {min(end, len(texts))}/{len(texts)} chunks "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)
    vectors = np.vstack(parts) if parts else np.zeros((0, embedder.dim), dtype=np.float32)
    check_vectors(vectors, len(chunks), embedder.dim)
    return vectors


def library_versions() -> dict:
    """Versions of the libraries that can change embeddings or rankings."""
    versions = {"python": platform.python_version()}
    for package in ("torch", "sentence-transformers", "transformers", "numpy", "rank-bm25"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def save_index(directory: Path, chunk_ids: list[str], vectors: np.ndarray, manifest: dict) -> None:
    """Write an index atomically: fill a temporary folder, then rename it into place."""
    tmp = directory.with_name(directory.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    np.save(tmp / "vectors.npy", vectors)
    (tmp / "chunk_ids.json").write_text(json.dumps(chunk_ids), encoding="utf-8")
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.rename(directory)


def load_index(directory: Path, key: str, chunk_ids: list[str], dim: int) -> DenseIndex:
    """Load a cached index and verify it is exactly the one requested."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    stored_ids = json.loads((directory / "chunk_ids.json").read_text(encoding="utf-8"))
    if manifest.get("key") != key or stored_ids != chunk_ids:
        raise ValueError(f"cached index at {directory} does not match its key or chunk ids; delete it")
    vectors = np.load(directory / "vectors.npy")
    check_vectors(vectors, len(chunk_ids), dim)
    return DenseIndex(stored_ids, vectors, manifest, directory)


def get_dense_index(chunk_config: ChunkConfig, chunks: list[dict], embedder: Embedder,
                    build: bool = False, index_dir: Path | None = None, progress: bool = False) -> DenseIndex:
    """Load the cached index for these chunks and this embedder; build it only if `build`."""
    identity = embedder.identity()
    key = dense_cache_key(embedded_content_hash(chunks), identity)
    directory = dense_cache_dir(chunk_config, identity, key, index_dir)
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if directory.exists():
        return load_index(directory, key, chunk_ids, embedder.dim)
    if not build:
        raise FileNotFoundError(f"no dense index for {chunk_config.config_id} with {identity['model']}; "
                                f"run: python -m src.index --size {chunk_config.size}")
    started = time.perf_counter()
    vectors = embed_chunks(chunks, embedder, progress)
    manifest = {"key": key, "chunk_config": asdict(chunk_config), "content_hash": embedded_content_hash(chunks),
                "n_chunks": len(chunks), "embedder": identity, "versions": library_versions(),
                "runtime": getattr(embedder, "runtime", dict)(),
                "build_seconds": round(time.perf_counter() - started, 1)}
    save_index(directory, chunk_ids, vectors, manifest)
    return DenseIndex(chunk_ids, vectors, manifest, directory)


def list_cached(index_dir: Path | None = None) -> None:
    """Print every cached dense index with the facts from its manifest."""
    for manifest_path in sorted((index_dir or config.INDEX_DIR).glob("dense/*/*/manifest.json")):
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        size_mb = sum(f.stat().st_size for f in manifest_path.parent.iterdir()) / 1e6
        print(f"{manifest_path.parent.relative_to(config.ROOT)}: {m['n_chunks']} x {m['embedder']['dim']} "
              f"({m['embedder']['model']}@{str(m['embedder']['revision'])[:8]}, {size_mb:.1f} MB, "
              f"built in {m['build_seconds']}s, runtime {m.get('runtime')})")


def main(argv: list[str] | None = None) -> None:
    """Build (or confirm cached) dense indexes for one chunk size or all primary sizes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="every setting in config.CHUNK_CONFIGS")
    parser.add_argument("--size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--list", action="store_true", help="only list cached indexes")
    args = parser.parse_args(argv)
    if args.list:
        list_cached()
        return
    chunk_configs = primary_chunk_configs() if args.all else [chunk_config_for_size(args.size)]
    embedder = SentenceTransformerEmbedder()
    for chunk_config in chunk_configs:
        chunks, _ = load_chunk_set(chunk_config)
        print(f"{chunk_config.config_id}: {len(chunks)} chunks", flush=True)
        started = time.perf_counter()
        index = get_dense_index(chunk_config, chunks, embedder, build=True, progress=True)
        print(f"{chunk_config.config_id}: {index.vectors.shape[0]} vectors x {index.vectors.shape[1]} dims "
              f"({index.manifest['build_seconds']}s to build, {time.perf_counter() - started:.1f}s this run) "
              f"-> {index.path.relative_to(config.ROOT)}", flush=True)


if __name__ == "__main__":
    main()
