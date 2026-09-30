"""Deterministic stand-ins for the embedding model and the model services."""

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sexandrag.config import Settings
from sexandrag.embedders import Vectors, ensure_fits
from sexandrag.tokens import TokenCounter, regex_counter

WORD_RE = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Bag-of-words vectors hashed into `dim` buckets; texts count one token per word plus 2 specials."""

    def __init__(self, dim: int = 64, max_tokens: int = 512, name: str = "fake/bag-of-words", revision: str = "test"):
        self.dim, self.max_tokens, self.name, self.revision = dim, max_tokens, name, revision
        self.documents_embedded = 0
        self.queries_embedded = 0
        self._queries: dict[str, Vectors] = {}

    def count_tokens(self, text: str) -> int:
        """One token per word, plus two special tokens."""
        return len(text.split()) + 2

    def vector(self, text: str) -> Vectors:
        """Unit-length hashed bag-of-words vector (never all zeros)."""
        v = np.zeros(self.dim)
        for word in WORD_RE.findall(text.lower()):
            v[int(hashlib.md5(word.encode(), usedforsecurity=False).hexdigest(), 16) % self.dim] += 1.0
        v[0] += 1e-3
        return (v / np.linalg.norm(v)).astype(np.float32)

    def embed_documents(self, texts: list[str], ids: list[str] | None = None) -> Vectors:
        """Check lengths like the real embedder, then embed."""
        ensure_fits(texts, self.count_tokens, self.max_tokens, ids)
        self.documents_embedded += len(texts)
        return np.stack([self.vector(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_queries(self, texts: Sequence[str]) -> None:
        """Embed new queries once (like the real embedder's cache)."""
        for text in texts:
            if text not in self._queries:
                ensure_fits([text], self.count_tokens, self.max_tokens)
                self._queries[text] = self.vector(text)
                self.queries_embedded += 1

    def embed_query(self, text: str) -> Vectors:
        """Check length like the real embedder, then embed (cached)."""
        self.embed_queries([text])
        return self._queries[text]

    def identity(self) -> dict[str, Any]:
        """Cache-key identity, mirroring SentenceTransformerEmbedder.identity()."""
        return {
            "model": self.name,
            "revision": self.revision,
            "max_tokens": self.max_tokens,
            "dim": self.dim,
            "normalized": True,
            "pooling": "bag-of-words",
        }


@dataclass
class FakeServices:
    """Services backed by FakeEmbedder and the regex token counter (no model, no network)."""

    embedder_instance: FakeEmbedder = field(default_factory=FakeEmbedder)
    embedder_loads: int = 0

    def embedder(self, settings: Settings) -> FakeEmbedder:
        """The fake embedder (counts how often a command asked for it)."""
        self.embedder_loads += 1
        return self.embedder_instance

    def embedder_identity(self, settings: Settings) -> dict[str, Any]:
        """The fake embedder's identity."""
        return self.embedder_instance.identity()

    def tokenizer_id(self, settings: Settings) -> tuple[str, str | None]:
        """Chunks are counted with the regex tokenizer."""
        return "regex", None

    def token_counter(self, settings: Settings) -> TokenCounter:
        """The dependency-free regex counter."""
        return regex_counter
