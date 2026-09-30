"""The real embedding model. Slow and memory-hungry, so opt in: RUN_MODEL_TESTS=1 python -m pytest."""
import os

import numpy as np
import pytest

from src import config
from src.embedders import SentenceTransformerEmbedder, TextTooLongError

pytestmark = pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1",
                                reason="set RUN_MODEL_TESTS=1 to load the real embedding model")


@pytest.fixture(scope="module")
def embedder():
    """The configured model with a deliberately small 64-token context."""
    return SentenceTransformerEmbedder(max_tokens=64)


def test_identity_and_vector_dimensions(embedder):
    identity = embedder.identity()
    assert (identity["model"], identity["revision"]) == (config.EMBEDDING_MODEL, config.EMBEDDING_REVISION)
    assert identity["dim"] == 1024 and identity["pooling"]["pooling_mode"] == "cls"
    vectors = embedder.embed_documents(["Carrie: Hello.", "Aidan: Pop tart."])
    assert vectors.shape == (2, 1024) and np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)


def test_the_real_embedder_refuses_to_truncate(embedder):
    long_text = "Carrie: " + "shoes " * 100
    assert embedder.count_tokens(long_text) > 64
    with pytest.raises(TextTooLongError):
        embedder.embed_documents([long_text], ["too-long"])
    with pytest.raises(TextTooLongError):
        embedder.embed_query(long_text)


def test_a_context_beyond_the_model_limit_is_rejected():
    with pytest.raises(ValueError):
        SentenceTransformerEmbedder(max_tokens=100_000)
