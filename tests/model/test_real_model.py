"""The real BGE-M3 model (2.3 GB, slow): opt in with `pytest --run-model` or RUN_MODEL_TESTS=1."""

import json

import numpy as np
import pytest

from sexandrag.artifacts import chunk_config_for_size
from sexandrag.chunk import load_chunk_set
from sexandrag.config import Settings
from sexandrag.embedders import SentenceTransformerEmbedder
from sexandrag.errors import ModelConfigurationError, TextTooLongError
from sexandrag.index import index_location
from sexandrag.services import PinnedModelServices
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.model
SETTINGS = Settings.defaults(REPO_ROOT)


@pytest.fixture(scope="module")
def small_context():
    """The pinned model with a deliberately small 64-token context."""
    return SentenceTransformerEmbedder(SETTINGS.model, SETTINGS.paths.model_cache_dir, max_tokens=64)


def test_identity_and_vector_dimensions(small_context):
    identity = small_context.identity()
    assert (identity["model"], identity["revision"], identity["dim"]) == (
        SETTINGS.model.repo_id,
        SETTINGS.model.revision,
        1024,
    )
    assert identity["pooling"]["pooling_mode"] == "cls"
    assert identity["max_tokens"] == 64
    vectors = small_context.embed_documents(["Carrie: Hello.", "Aidan: Pop tart."])
    assert vectors.shape == (2, 1024)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)


def test_the_real_embedder_refuses_to_truncate(small_context):
    long_text = "Carrie: " + "shoes " * 100
    assert small_context.count_tokens(long_text) > 64
    with pytest.raises(TextTooLongError):
        small_context.embed_documents([long_text], ["too-long"])
    with pytest.raises(TextTooLongError):
        small_context.embed_query(long_text)


def test_a_context_beyond_the_model_limit_is_rejected():
    with pytest.raises(ModelConfigurationError):
        SentenceTransformerEmbedder(SETTINGS.model, SETTINGS.paths.model_cache_dir, max_tokens=100_000)


def test_the_chunk_tokenizer_agrees_with_the_model(small_context):
    counter = PinnedModelServices().token_counter(SETTINGS)
    text = "Carrie: I couldn't help but wonder about love in Manhattan."
    assert counter(text) + 2 == small_context.count_tokens(text)  # the embedder adds [CLS] and [SEP]


@pytest.mark.corpus
def test_re_embedding_a_chunk_reproduces_its_cached_vector():
    embedder = PinnedModelServices().embedder(SETTINGS)
    spec = SETTINGS.model.spec
    chunk_set = load_chunk_set(
        chunk_config_for_size(512, SETTINGS.chunking, spec.repo_id, spec.revision),
        SETTINGS.paths.chunks_dir,
        SETTINGS.paths.lines_jsonl,
    )
    _, directory = index_location(chunk_set, embedder.identity(), SETTINGS.paths.index_dir)
    cached = np.load(directory / "vectors.npy")
    assert json.loads((directory / "chunk_ids.json").read_text())[0] == chunk_set.chunks[0]["chunk_id"]
    fresh = embedder.embed_documents([chunk_set.chunks[0]["text"]])[0]
    assert float(fresh @ cached[0]) > 0.9999
