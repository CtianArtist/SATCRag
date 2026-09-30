"""Text embedders with an enforced context limit: over-long input raises, it is never truncated.

sentence-transformers silently cuts text at `max_seq_length`. That would make a 1024-token chunk
look like a 1024-token chunk to BM25 but not to the dense model, so every text is token-counted
with the model's own tokenizer (special tokens included) before encoding.

The model is loaded from the verified local snapshot of the pinned revision (sexandrag.model),
with trust_remote_code=False and local_files_only=True, and checked against its specification
after loading. The device is used exactly as configured; a missing GPU is an error, not a
silent switch to CPU.
"""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from sexandrag.config import ModelConfig
from sexandrag.errors import ModelConfigurationError, TextTooLongError
from sexandrag.model import check_loaded_model, snapshot_path, verify_snapshot

log = logging.getLogger(__name__)
Vectors = npt.NDArray[np.float32]


class Embedder(Protocol):
    """What the index and the dense retriever need from an embedding model."""

    name: str
    max_tokens: int
    dim: int

    def count_tokens(self, text: str) -> int:
        """Tokens the model will see for `text`, special tokens included."""

    def embed_documents(self, texts: list[str], ids: list[str] | None = None) -> Vectors:
        """Unit-length vectors for texts that all fit the context."""

    def embed_query(self, text: str) -> Vectors:
        """The unit-length vector of one query."""

    def embed_queries(self, texts: Sequence[str]) -> None:
        """Embed many queries at once so later embed_query calls are cache hits."""

    def identity(self) -> dict[str, Any]:
        """Everything about the model that changes its vectors (part of cache keys)."""


def ensure_fits(
    texts: Sequence[str], count_tokens: Callable[[str], int], max_tokens: int, ids: Sequence[str] | None = None
) -> None:
    """Raise TextTooLongError naming every text longer than `max_tokens` tokens."""
    too_long = []
    for i, text in enumerate(texts):
        n_tokens = count_tokens(text)
        if n_tokens > max_tokens:
            too_long.append((ids[i] if ids else i, n_tokens))
    if too_long:
        raise TextTooLongError(
            f"{len(too_long)} text(s) exceed the configured {max_tokens}-token context and would be truncated; "
            "refusing",
            actual=f"first few (id, tokens): {too_long[:5]}",
            recovery="use smaller chunks or a model with a longer context; truncation is never applied",
        )


def resolve_device(device: str) -> str:
    """The configured device, verified to exist (no silent fallback to CPU)."""
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise ModelConfigurationError(
                "model.device is 'cuda' but no CUDA device is available",
                recovery='set [model] device = "cpu" (the baseline), or run on a machine with a GPU',
            )
    return device


class SentenceTransformerEmbedder:
    """The pinned sentence-transformers model with an enforced context limit and cached query vectors."""

    def __init__(
        self,
        model_config: ModelConfig,
        cache_dir: Path | None = None,
        verify_deep: bool = False,
        max_tokens: int | None = None,
    ):
        spec = model_config.spec
        max_tokens = spec.max_tokens if max_tokens is None else max_tokens
        if not 0 < max_tokens <= spec.max_tokens:
            raise ModelConfigurationError(
                "the requested context is outside the model's range",
                expected=f"1..{spec.max_tokens} tokens",
                actual=max_tokens,
            )
        path = snapshot_path(spec, cache_dir)
        verify_snapshot(path, spec, deep=verify_deep)
        from sentence_transformers import SentenceTransformer  # heavy import, only when embedding

        device = resolve_device(model_config.device)
        log.info("loading %s@%s on %s", spec.repo_id, spec.revision[:12], device)
        self.model = SentenceTransformer(str(path), device=device, trust_remote_code=False, local_files_only=True)
        check_loaded_model(self.model, spec, max_tokens)
        self.model.max_seq_length = max_tokens
        self.spec = spec
        self.name, self.revision, self.max_tokens, self.dim = spec.repo_id, spec.revision, max_tokens, spec.dim
        self.batch_size = model_config.batch_size
        self._query_vectors: dict[str, Vectors] = {}

    def count_tokens(self, text: str) -> int:
        """Tokens the model will see for `text`, special tokens included."""
        return len(self.model.tokenizer(text, add_special_tokens=True)["input_ids"])

    def _encode(self, texts: list[str], show_progress: bool) -> Vectors:
        """Encode already-checked texts into L2-normalized float32 vectors."""
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: list[str], ids: list[str] | None = None, show_progress: bool = False) -> Vectors:
        """Embed documents after checking that none would be truncated."""
        ensure_fits(texts, self.count_tokens, self.max_tokens, ids)
        return self._encode(texts, show_progress)

    def embed_queries(self, texts: Sequence[str]) -> None:
        """Embed every new query in batches; later embed_query calls reuse the vectors."""
        new = list(dict.fromkeys(text for text in texts if text not in self._query_vectors))
        if new:
            ensure_fits(new, self.count_tokens, self.max_tokens)
            self._query_vectors.update(zip(new, self._encode(new, show_progress=False), strict=True))

    def embed_query(self, text: str) -> Vectors:
        """Embed a query (checked like documents); repeated queries reuse the first vector."""
        self.embed_queries([text])
        return self._query_vectors[text]

    def identity(self) -> dict[str, Any]:
        """Everything about the model that changes its vectors; part of every dense cache key."""
        return self.spec.identity() | {"max_tokens": self.max_tokens}

    def runtime(self) -> dict[str, Any]:
        """How the model ran (recorded in manifests, not part of cache keys)."""
        import torch

        return {
            "device": str(self.model.device),
            "torch_threads": torch.get_num_threads(),
            "batch_size": self.batch_size,
        }
