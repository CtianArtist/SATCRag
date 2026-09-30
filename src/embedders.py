"""Text embedders with an enforced context limit: over-long input raises, it is never truncated.

sentence-transformers silently cuts text at `max_seq_length`. That would make a 1024-token
chunk look like a 1024-token chunk to BM25 but not to the dense model, so every text is
token-counted with the model's own tokenizer (special tokens included) before encoding.

The model is loaded from a local copy of exactly the pinned revision's files. Loading by
repo name instead lets transformers fetch weights from other revisions (for .bin-only repos
it downloads a safetensors conversion from an open pull request in a background thread).
"""
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from src import config

MODEL_FILES = ["*.json", "*.model", "vocab.txt", "*.safetensors", "pytorch_model.bin"]   # configs, tokenizer, weights
SKIPPED_FILES = ["onnx/*"]


class TextTooLongError(ValueError):
    """Raised instead of letting an embedding model silently truncate its input."""


class Embedder(Protocol):
    """What the index and the dense retriever need from an embedding model."""
    name: str
    max_tokens: int
    dim: int

    def count_tokens(self, text: str) -> int: ...
    def embed_documents(self, texts: list[str], ids: list[str] | None = None) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
    def identity(self) -> dict: ...


def ensure_fits(texts: list[str], count_tokens, max_tokens: int, ids: list[str] | None = None) -> None:
    """Raise TextTooLongError naming every text longer than `max_tokens` tokens."""
    too_long = []
    for i, text in enumerate(texts):
        n_tokens = count_tokens(text)
        if n_tokens > max_tokens:
            too_long.append((ids[i] if ids else i, n_tokens))
    if too_long:
        raise TextTooLongError(f"{len(too_long)} text(s) exceed the configured {max_tokens}-token context "
                               f"and would be truncated; refusing. First few (id, tokens): {too_long[:5]}")


def local_model_path(model_name: str, revision: str | None) -> Path:
    """Download (once) the files a sentence-transformers model needs from one revision; return the folder."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_name, revision=revision, allow_patterns=MODEL_FILES,
                                  ignore_patterns=SKIPPED_FILES))


def load_sentence_transformer(path: Path, device: str):
    """Load a local sentence-transformers model, failing if any module in modules.json is missing."""
    from sentence_transformers import SentenceTransformer
    listed = json.loads((path / "modules.json").read_text(encoding="utf-8"))
    model = SentenceTransformer(str(path), device=device)
    if len(model) != len(listed):
        raise RuntimeError(f"{path} loaded {len(model)} modules but modules.json lists {len(listed)}")
    return model


class SentenceTransformerEmbedder:
    """A sentence-transformers model on a pinned revision with an enforced context limit."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL,
                 revision: str | None = config.EMBEDDING_REVISION,
                 max_tokens: int = config.EMBED_MAX_TOKENS, device: str = config.EMBED_DEVICE,
                 batch_size: int = config.EMBED_BATCH_SIZE):
        self.model = load_sentence_transformer(local_model_path(model_name, revision), device)
        capacity = self.model.max_seq_length
        if max_tokens > capacity:
            raise ValueError(f"configured context {max_tokens} exceeds {model_name}'s limit of {capacity}")
        self.model.max_seq_length = max_tokens
        self.name, self.revision, self.max_tokens = model_name, revision, max_tokens
        self.batch_size = batch_size
        self.dim = self.model.get_embedding_dimension()
        self._query_vectors: dict[str, np.ndarray] = {}

    def count_tokens(self, text: str) -> int:
        """Tokens the model will see for `text`, special tokens included."""
        return len(self.model.tokenizer(text, add_special_tokens=True)["input_ids"])

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        """Encode already-checked texts into L2-normalized float32 vectors."""
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=show_progress)
        return np.asarray(vectors, dtype=np.float32)

    def embed_documents(self, texts: list[str], ids: list[str] | None = None,
                        show_progress: bool = False) -> np.ndarray:
        """Embed documents after checking that none would be truncated."""
        ensure_fits(texts, self.count_tokens, self.max_tokens, ids)
        return self._encode(texts, show_progress)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query (checked like documents); repeated queries reuse the first vector."""
        if text not in self._query_vectors:
            ensure_fits([text], self.count_tokens, self.max_tokens)
            self._query_vectors[text] = self._encode([text], show_progress=False)[0]
        return self._query_vectors[text]

    def identity(self) -> dict:
        """Everything about the model that changes its vectors; part of every dense cache key."""
        pooling = self.model[1].get_config_dict() if len(self.model) > 1 else None
        return {"model": self.name, "revision": self.revision, "max_tokens": self.max_tokens,
                "dim": self.dim, "normalized": True, "pooling": pooling}

    def runtime(self) -> dict:
        """How the model ran (recorded in manifests, not part of cache keys)."""
        import torch
        return {"device": str(self.model.device), "torch_threads": torch.get_num_threads(),
                "batch_size": self.batch_size}
