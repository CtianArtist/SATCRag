"""Dense index construction, caching and truncation refusal (src/index.py, src/chunk.py)."""
import json

import numpy as np
import pytest

from src import config
from src.artifacts import ChunkConfig, StaleArtifactError, embedded_content_hash
from src.chunk import load_chunk_set, save_chunk_set
from src.embedders import TextTooLongError, ensure_fits
from src.index import dense_cache_key, get_dense_index
from src.retrieve import build_retrievers
from tests.fakes import FakeEmbedder, make_chunk

CC = ChunkConfig(256, 32, tokenizer="regex", tokenizer_revision=None)


@pytest.fixture
def chunks():
    return [make_chunk(1, 1, i, 10 * i, 10 * i + 12, f"Carrie: line {i} about shoes and brunch number {i}")
            for i in range(7)]


def test_index_holds_one_unit_vector_per_chunk_in_file_order(tmp_path, chunks):
    index = get_dense_index(CC, chunks, FakeEmbedder(dim=48), build=True, index_dir=tmp_path)
    assert index.vectors.shape == (7, 48) and index.vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0)
    assert index.chunk_ids == [c["chunk_id"] for c in chunks]
    assert index.manifest["embedder"]["dim"] == 48 and index.manifest["n_chunks"] == 7


def test_cached_index_is_reused_without_re_embedding(tmp_path, chunks):
    embedder = FakeEmbedder()
    first = get_dense_index(CC, chunks, embedder, build=True, index_dir=tmp_path)
    embedded_so_far = embedder.documents_embedded
    second = get_dense_index(CC, chunks, embedder, build=True, index_dir=tmp_path)
    assert embedder.documents_embedded == embedded_so_far
    assert second.path == first.path and np.array_equal(second.vectors, first.vectors)


def test_a_missing_index_is_never_built_implicitly(tmp_path, chunks):
    with pytest.raises(FileNotFoundError):
        get_dense_index(CC, chunks, FakeEmbedder(), build=False, index_dir=tmp_path)


@pytest.mark.parametrize("change", ["text", "chunk_id", "model", "revision", "context", "dim"])
def test_any_change_to_embedded_content_or_model_gives_a_separate_cache(tmp_path, chunks, change):
    first = get_dense_index(CC, chunks, FakeEmbedder(), build=True, index_dir=tmp_path)
    changed = [dict(c) for c in chunks]
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
    second = get_dense_index(CC, changed, embedder, build=True, index_dir=tmp_path)
    assert second.manifest["key"] != first.manifest["key"] and second.path != first.path


def test_chunk_settings_are_cached_in_separate_folders(tmp_path, chunks):
    small = get_dense_index(ChunkConfig(256, 32, "regex", None), chunks, FakeEmbedder(), build=True, index_dir=tmp_path)
    large = get_dense_index(ChunkConfig(1024, 128, "regex", None), chunks, FakeEmbedder(), build=True, index_dir=tmp_path)
    assert small.path.parent.name == "tok256o32-regex" and large.path.parent.name == "tok1024o128-regex"


def test_cache_key_is_stable_and_ignores_metadata(tmp_path, chunks):
    key = dense_cache_key(embedded_content_hash(chunks), FakeEmbedder().identity())
    relabeled = [dict(c, speakers=["Someone"], inferred_scenes=[1]) for c in chunks]
    assert dense_cache_key(embedded_content_hash(relabeled), FakeEmbedder().identity()) == key


def test_a_tampered_cache_is_rejected(tmp_path, chunks):
    index = get_dense_index(CC, chunks, FakeEmbedder(), build=True, index_dir=tmp_path)
    (index.path / "chunk_ids.json").write_text(json.dumps(list(reversed(index.chunk_ids))))
    with pytest.raises(ValueError):
        get_dense_index(CC, chunks, FakeEmbedder(), build=False, index_dir=tmp_path)


def test_over_long_chunks_are_refused_before_any_embedding(tmp_path, chunks):
    embedder = FakeEmbedder(max_tokens=8)          # every chunk is longer than 8 tokens
    with pytest.raises(TextTooLongError) as error:
        get_dense_index(CC, chunks, embedder, build=True, index_dir=tmp_path)
    assert chunks[0]["chunk_id"] in str(error.value)
    assert embedder.documents_embedded == 0 and not list(tmp_path.rglob("vectors.npy"))


def test_ensure_fits_counts_special_tokens_and_names_offenders():
    count = lambda text: len(text.split()) + 2
    ensure_fits(["a b c"], count, max_tokens=5)    # 3 words + 2 special tokens = 5: fits exactly
    with pytest.raises(TextTooLongError, match="q7"):
        ensure_fits(["a b c", "a b c d"], count, max_tokens=5, ids=["q6", "q7"])


def test_over_long_queries_are_refused():
    with pytest.raises(TextTooLongError):
        FakeEmbedder(max_tokens=4).embed_query("far too many words for this context")


@pytest.fixture
def isolated_paths(tmp_path, monkeypatch):
    """Point chunk, index and lines paths at a temporary folder."""
    monkeypatch.setattr(config, "CHUNKS_DIR", tmp_path / "chunks")
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    lines = tmp_path / "lines.jsonl"
    lines.write_text('{"line": 1}\n')
    monkeypatch.setattr(config, "LINES_JSONL", lines)
    return lines


def test_chunk_sets_refuse_altered_or_stale_files(isolated_paths, chunks):
    save_chunk_set(CC, chunks, with_scenes=False)
    assert load_chunk_set(CC)[0] == chunks
    assert ChunkConfig(256, 32, "other/tokenizer", None).path != CC.path
    with pytest.raises(StaleArtifactError, match="different tokenizer"):
        load_chunk_set(ChunkConfig(256, 32, "regex", "another-revision"))
    CC.path.write_text(CC.path.read_text() + "\n")
    with pytest.raises(StaleArtifactError, match="chunk file changed"):
        load_chunk_set(CC)
    save_chunk_set(CC, chunks, with_scenes=False)
    isolated_paths.write_text('{"line": 2}\n')
    with pytest.raises(StaleArtifactError, match="lines.jsonl changed"):
        load_chunk_set(CC)


def test_retrievers_load_but_never_build_indexes(isolated_paths, chunks):
    save_chunk_set(CC, chunks, with_scenes=False)
    embedder = FakeEmbedder()
    with pytest.raises(FileNotFoundError):
        build_retrievers(CC, ("dense",), embedder)
    get_dense_index(CC, chunks, embedder, build=True)
    retrievers = build_retrievers(CC, ("bm25", "dense", "hybrid"), embedder)
    for name, retriever in retrievers.items():
        assert retriever.name == name and len(retriever.search("number 1 or 2 or 3", 3)) == 3
