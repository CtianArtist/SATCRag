"""Dense index construction, caching, corruption handling and truncation refusal (satc_rag.index)."""

import json

import numpy as np
import pytest

from satc_rag.artifacts import ChunkConfig, embedded_content_hash
from satc_rag.chunk import ChunkSet
from satc_rag.errors import ArtifactMismatchError, ArtifactMissingError, TextTooLongError
from satc_rag.index import build_dense_index, cached_indexes, dense_cache_key, get_dense_index, index_location
from satc_rag.retrieve import build_retrievers
from tests.support.builders import make_chunk
from tests.support.fakes import FakeEmbedder

CC = ChunkConfig(256, 32, tokenizer="regex", tokenizer_revision=None)


@pytest.fixture
def chunk_set():
    chunks = [
        make_chunk(1, 1, i, 10 * i, 10 * i + 12, f"Carrie: line {i} about shoes and brunch number {i}")
        for i in range(7)
    ]
    return ChunkSet(CC, chunks, {})


def test_index_holds_one_unit_vector_per_chunk_in_file_order(tmp_path, chunk_set):
    index = build_dense_index(chunk_set, FakeEmbedder(dim=48), tmp_path)
    assert index.vectors.shape == (7, 48)
    assert index.vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0)
    assert index.chunk_ids == [c["chunk_id"] for c in chunk_set.chunks]
    assert index.manifest["embedder"]["dim"] == 48
    assert index.manifest["n_chunks"] == 7


def test_a_valid_cache_is_reused_without_re_embedding(tmp_path, chunk_set):
    embedder = FakeEmbedder()
    first = build_dense_index(chunk_set, embedder, tmp_path)
    embedded_so_far = embedder.documents_embedded
    second = build_dense_index(chunk_set, embedder, tmp_path)
    assert embedder.documents_embedded == embedded_so_far
    assert second.path == first.path
    assert np.array_equal(second.vectors, first.vectors)


def test_rebuild_replaces_a_cached_index_explicitly(tmp_path, chunk_set):
    embedder = FakeEmbedder()
    build_dense_index(chunk_set, embedder, tmp_path)
    before = embedder.documents_embedded
    build_dense_index(chunk_set, embedder, tmp_path, rebuild=True)
    assert embedder.documents_embedded == 2 * before


def test_retrieval_never_builds_an_index(tmp_path, chunk_set):
    with pytest.raises(ArtifactMissingError, match="satc-rag index --size 256"):
        get_dense_index(chunk_set, FakeEmbedder(), tmp_path)


@pytest.mark.parametrize("change", ["text", "chunk_id", "model", "revision", "context", "dim"])
def test_any_change_to_embedded_content_or_model_gives_a_separate_cache(tmp_path, chunk_set, change):
    first = build_dense_index(chunk_set, FakeEmbedder(), tmp_path)
    changed = [dict(c) for c in chunk_set.chunks]
    embedder = FakeEmbedder()
    if change == "text":
        changed[3]["text"] += " extra"
    elif change == "chunk_id":
        changed[3]["chunk_id"] = "renamed"
    elif change == "model":
        embedder = FakeEmbedder(name="fake/other-model")
    elif change == "revision":
        embedder = FakeEmbedder(revision="v2")
    elif change == "context":
        embedder = FakeEmbedder(max_tokens=1024)
    else:
        embedder = FakeEmbedder(dim=32)
    second = build_dense_index(ChunkSet(CC, changed, {}), embedder, tmp_path)
    assert second.manifest["key"] != first.manifest["key"]
    assert second.path != first.path


def test_chunk_settings_are_cached_in_separate_folders(tmp_path, chunk_set):
    small = build_dense_index(chunk_set, FakeEmbedder(), tmp_path)
    large = build_dense_index(
        ChunkSet(ChunkConfig(1024, 128, "regex", None), chunk_set.chunks, {}), FakeEmbedder(), tmp_path
    )
    assert small.path.parent.name == "tok256o32-regex"
    assert large.path.parent.name == "tok1024o128-regex"


def test_cache_key_is_stable_and_ignores_metadata(chunk_set):
    key = dense_cache_key(embedded_content_hash(chunk_set.chunks), FakeEmbedder().identity())
    relabeled = [dict(c, speakers=["Someone"], inferred_scenes=[1]) for c in chunk_set.chunks]
    assert dense_cache_key(embedded_content_hash(relabeled), FakeEmbedder().identity()) == key


def test_a_tampered_cache_is_rejected_with_the_rebuild_command(tmp_path, chunk_set):
    index = build_dense_index(chunk_set, FakeEmbedder(), tmp_path)
    (index.path / "chunk_ids.json").write_text(json.dumps(list(reversed(index.chunk_ids))))
    with pytest.raises(ArtifactMismatchError, match="--rebuild"):
        get_dense_index(chunk_set, FakeEmbedder(), tmp_path)


def test_a_truncated_vector_file_is_rejected(tmp_path, chunk_set):
    index = build_dense_index(chunk_set, FakeEmbedder(), tmp_path)
    vectors = index.path / "vectors.npy"
    vectors.write_bytes(vectors.read_bytes()[:100])
    with pytest.raises(ArtifactMismatchError, match="unreadable"):
        get_dense_index(chunk_set, FakeEmbedder(), tmp_path)


def test_non_unit_vectors_are_rejected(tmp_path, chunk_set):
    index = build_dense_index(chunk_set, FakeEmbedder(), tmp_path)
    np.save(index.path / "vectors.npy", index.vectors * 3)
    with pytest.raises(ArtifactMismatchError, match="L2-normalized"):
        get_dense_index(chunk_set, FakeEmbedder(), tmp_path)


def test_an_interrupted_build_leaves_nothing_usable_and_is_cleaned_up(tmp_path, chunk_set):
    embedder = FakeEmbedder()
    _, directory = index_location(chunk_set, embedder.identity(), tmp_path)
    partial = directory.with_name(directory.name + ".tmp")
    partial.mkdir(parents=True)
    (partial / "vectors.npy").write_bytes(b"half written")
    with pytest.raises(ArtifactMissingError):
        get_dense_index(chunk_set, embedder, tmp_path)  # the half-written folder is never mistaken for an index
    assert any("incomplete" in s["status"] for s in cached_indexes(tmp_path))
    index = build_dense_index(chunk_set, embedder, tmp_path)
    assert not partial.exists()
    assert index.path == directory


def test_over_long_chunks_are_refused_before_any_embedding(tmp_path, chunk_set):
    embedder = FakeEmbedder(max_tokens=8)  # every chunk is longer than 8 tokens
    with pytest.raises(TextTooLongError) as error:
        build_dense_index(chunk_set, embedder, tmp_path)
    assert chunk_set.chunks[0]["chunk_id"] in str(error.value)
    assert embedder.documents_embedded == 0
    assert not list(tmp_path.rglob("vectors.npy"))


def test_retrievers_load_but_never_build_indexes(tmp_path, chunk_set):
    embedder = FakeEmbedder()
    with pytest.raises(ArtifactMissingError):
        build_retrievers(chunk_set, ("dense",), embedder, tmp_path)
    build_dense_index(chunk_set, embedder, tmp_path)
    retrievers = build_retrievers(chunk_set, ("bm25", "dense", "hybrid"), embedder, tmp_path)
    for name, retriever in retrievers.items():
        assert retriever.name == name
        assert len(retriever.search("number 1 or 2 or 3", 3)) == 3
