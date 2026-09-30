"""Truncation refusal and device handling (sexandrag.embedders), without the real model."""

import pytest

from sexandrag.embedders import ensure_fits, resolve_device
from sexandrag.errors import ModelConfigurationError, TextTooLongError
from tests.support.fakes import FakeEmbedder


def count(text: str) -> int:
    return len(text.split()) + 2


def test_ensure_fits_counts_special_tokens_and_names_offenders():
    ensure_fits(["a b c"], count, max_tokens=5)  # 3 words + 2 special tokens = 5: fits exactly
    with pytest.raises(TextTooLongError, match="q7"):
        ensure_fits(["a b c", "a b c d"], count, max_tokens=5, ids=["q6", "q7"])


def test_over_long_queries_are_refused():
    with pytest.raises(TextTooLongError):
        FakeEmbedder(max_tokens=4).embed_query("far too many words for this context")


def test_queries_are_embedded_once_and_reused():
    embedder = FakeEmbedder()
    embedder.embed_queries(["a question", "another question", "a question"])
    embedder.embed_query("a question")
    assert embedder.queries_embedded == 2


def test_cpu_is_used_as_configured():
    assert resolve_device("cpu") == "cpu"


def test_a_missing_gpu_is_an_error_not_a_silent_cpu_fallback():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("this machine has a CUDA device")
    with pytest.raises(ModelConfigurationError, match="no CUDA device"):
        resolve_device("cuda")
